#!/usr/bin/env python3
"""Small molecular meta-GGA tau-path benchmark for CP2K/GauXC."""

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ANG_TO_BOHR = 1.8897261254578281

ROOT = Path(os.environ.get("TAU_MGGA_CP2K_ROOT", "/home/kuehne88/cp2k-master-20260520"))
BUILD = Path(os.environ.get("TAU_MGGA_CP2K_BUILD", ROOT / "build-gauxc-skala"))
CP2K = Path(os.environ.get("TAU_MGGA_CP2K", BUILD / "bin/cp2k.psmp"))
WORK = Path(os.environ.get("TAU_MGGA_WORK", "/home/kuehne88/tau-mgga-gauxc-bench"))
RESULTS = WORK / "results.json"
SUMMARY = WORK / "summary.tsv"

BASE_ENV = os.environ.copy()
BASE_ENV.update(
    {
        "CP2K_DATA_DIR": str(ROOT / "data"),
        "LD_LIBRARY_PATH": ":".join(
            [
                str(BUILD / "src"),
                "/home/kuehne88/cp2k-gauxc-skala-toolchain/libtorch-2.7.1/lib",
                "/home/kuehne88/cp2k-gauxc-skala-toolchain/torch-venv-aarch64/lib/python3.12/site-packages/torch.libs",
                "/home/kuehne88/cp2k-gauxc-skala-toolchain/gauxc-1.1-skala-cp2k-fixes/lib",
                "/home/kuehne88/cp2k-jpoto/tools/toolchain/install/libxsmm-e0c4a2389afba36c453233ad7de07bd92c715bec/lib",
                os.environ.get("LD_LIBRARY_PATH", ""),
            ]
        ),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "OMP_STACKSIZE": "512M",
    }
)

SYSTEMS = {
    "H2O": {
        "uks": False,
        "multiplicity": 1,
        "fd_component": (1, 1),
        "coords": [
            ("O", 6.000000, 6.000000, 6.000000),
            ("H", 6.000000, 6.756950, 6.585880),
            ("H", 6.000000, 5.243050, 6.585880),
        ],
    },
    "OH": {
        "uks": True,
        "multiplicity": 2,
        "fd_component": (1, 2),
        "coords": [
            ("O", 6.000000, 6.000000, 6.000000),
            ("H", 6.000000, 6.000000, 6.970000),
        ],
    },
}

METHODS = {
    "GPW_SCAN_GTH": {
        "method": "GPW",
        "basis": {
            "H": "TZV2P-MOLOPT-SCAN-GTH-q1",
            "O": "TZV2P-MOLOPT-SCAN-GTH-q6",
        },
        "potential": {"H": "GTH-SCAN-q1", "O": "GTH-SCAN-q6"},
    },
    "GAPW_AE": {
        "method": "GAPW",
        "basis": {
            "H": "TZVPP-MOLOPT-PBE-ae",
            "O": "TZVPP-MOLOPT-PBE-ae",
        },
        "potential": {"H": "ALL", "O": "ALL"},
    },
}

FUNCTIONALS = {
    "TPSS": {
        "gauxc": "TPSS",
        "native_rks": "shortcut",
        "native_uks": "libxc",
        "libxc_sections": ("MGGA_X_TPSS", "MGGA_C_TPSS"),
    },
    "R2SCAN": {
        "gauxc": "R2SCAN",
        "native_rks": "libxc",
        "native_uks": "libxc",
        "libxc_sections": ("MGGA_X_R2SCAN", "MGGA_C_R2SCAN"),
    },
}


def unique_elements(coords):
    out = []
    for el, *_ in coords:
        if el not in out:
            out.append(el)
    return out


def center(coords):
    n = len(coords)
    return tuple(sum(c[i] for _, *c in coords) / n for i in range(3))


def displaced(coords, atom_index, axis, delta):
    out = []
    for i, (el, x, y, z) in enumerate(coords):
        vals = [x, y, z]
        if i == atom_index:
            vals[axis] += delta
        out.append((el, *vals))
    return out


def scaled(coords, lam):
    c = center(coords)
    out = []
    for el, x, y, z in coords:
        vals = [x, y, z]
        out.append((el, *(c[i] + (1.0 + lam) * (vals[i] - c[i]) for i in range(3))))
    return out


def native_xc_block(functional, uks):
    info = FUNCTIONALS[functional]
    flavor = info["native_uks" if uks else "native_rks"]
    if flavor == "shortcut":
        return f"""      &XC_FUNCTIONAL {functional}
      &END XC_FUNCTIONAL"""
    sections = "\n".join(f"""        &{name}
        &END {name}""" for name in info["libxc_sections"])
    return f"""      &XC_FUNCTIONAL
{sections}
      &END XC_FUNCTIONAL"""


def gauxc_xc_block(functional, debug_virial):
    extra = ""
    if debug_virial:
        extra = """
          MOLECULAR_VIRIAL T
          MOLECULAR_VIRIAL_DEBUG T
          MOLECULAR_VIRIAL_DEBUG_DX 1.0E-4"""
    return f"""      &XC_FUNCTIONAL
        &GAUXC
          FUNCTIONAL {FUNCTIONALS[functional]["gauxc"]}
          MODEL NONE
          GRID SUPERFINE
          PRUNING_SCHEME UNPRUNED{extra}
        &END GAUXC
      &END XC_FUNCTIONAL"""


def xc_block(functional, path, uks, debug_virial):
    if path == "native":
        return native_xc_block(functional, uks)
    if path == "gauxc":
        return gauxc_xc_block(functional, debug_virial)
    raise ValueError(path)


def render_input(system, method, functional, path, coords, run_type="ENERGY_FORCE", debug_virial=False):
    sysinfo = SYSTEMS[system]
    spec = METHODS[method]
    elems = unique_elements(coords)
    qs_lines = [
        "      EPS_DEFAULT 1.0E-10",
        "      EXTRAPOLATION USE_PREV_P",
        f"      METHOD {spec['method']}",
    ]
    if method == "GAPW_AE":
        qs_lines += [
            "      GAPW_ACCURATE_XCINT T",
            "      QUADRATURE GC_LOG",
            "      LMAXN0 6",
            "      LMAXN1 6",
            "      EPSISO 1.0E-12",
            "      EPSRHO0 1.0E-8",
            "      EPSFIT 1.0E-5",
            "      ALPHA0_H 10.0",
        ]
    coord_lines = "\n".join(f"      {el:2s} {x:14.8f} {y:14.8f} {z:14.8f}" for el, x, y, z in coords)
    kind_lines = []
    for el in elems:
        kind_lines.append(
            f"""    &KIND {el}
      BASIS_SET {spec["basis"][el]}
      POTENTIAL {spec["potential"][el]}
    &END KIND"""
        )
    scf_extra = ""
    diagonalization_block = """      &DIAGONALIZATION
      &END DIAGONALIZATION
"""
    if sysinfo["uks"]:
        diagonalization_block = ""
        scf_extra = """      &OT
        PRECONDITIONER FULL_ALL
        MINIMIZER DIIS
      &END OT
"""
    return f"""&GLOBAL
  PRINT_LEVEL LOW
  PROJECT {system}_{method}_{functional}_{path}
  RUN_TYPE {run_type}
&END GLOBAL

&FORCE_EVAL
  METHOD Quickstep
  &DFT
    BASIS_SET_FILE_NAME BASIS_MOLOPT_UZH
    POTENTIAL_FILE_NAME POTENTIAL_UZH
    UKS {str(sysinfo["uks"]).upper()}
    MULTIPLICITY {sysinfo["multiplicity"]}
    &MGRID
      CUTOFF 600
      REL_CUTOFF 60
    &END MGRID
    &POISSON
      PERIODIC NONE
      POISSON_SOLVER MT
    &END POISSON
    &QS
{chr(10).join(qs_lines)}
    &END QS
    &SCF
      EPS_SCF 1.0E-8
      MAX_SCF 100
      SCF_GUESS ATOMIC
{diagonalization_block.rstrip()}
{scf_extra.rstrip()}
      &PRINT
        &RESTART OFF
        &END RESTART
      &END PRINT
    &END SCF
    &XC
      DENSITY_CUTOFF 1.0E-11
      GRADIENT_CUTOFF 1.0E-11
      TAU_CUTOFF 1.0E-11
{xc_block(functional, path, sysinfo["uks"], debug_virial)}
      &XC_GRID
        XC_DERIV NN50_SMOOTH
        XC_SMOOTH_RHO NONE
        USE_FINER_GRID
      &END XC_GRID
    &END XC
  &END DFT
  &SUBSYS
    &CELL
      ABC 12.0 12.0 12.0
      PERIODIC NONE
    &END CELL
    &COORD
{coord_lines}
    &END COORD
{chr(10).join(kind_lines)}
  &END SUBSYS
  &PRINT
    &FORCES ON
    &END FORCES
  &END PRINT
&END FORCE_EVAL
"""


def parse_output(text):
    energy = None
    forces = []
    for line in text.splitlines():
        if "ENERGY| Total FORCE_EVAL" in line:
            energy = float(line.split()[-1])
    in_forces = False
    in_forces_pipe = False
    for line in text.splitlines():
        if "ATOMIC FORCES in [a.u.]" in line:
            in_forces = True
            forces = []
            continue
        if "FORCES| Atomic forces [hartree/bohr]" in line:
            in_forces_pipe = True
            forces = []
            continue
        if in_forces and "SUM OF ATOMIC FORCES" in line:
            in_forces = False
            continue
        if in_forces_pipe and "FORCES| Sum" in line:
            in_forces_pipe = False
            continue
        if in_forces:
            m = re.match(r"\s*\d+\s+\d+\s+([A-Za-z]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)", line)
            if m:
                forces.append((m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))))
        if in_forces_pipe:
            m = re.match(r"\s*FORCES\|\s+(\d+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+[-+0-9.Ee]+", line)
            if m:
                idx = int(m.group(1))
                forces.append((f"A{idx}", float(m.group(2)), float(m.group(3)), float(m.group(4))))
    gauxc_virial_fd = None
    for line in text.splitlines():
        if "GAUXC| Molecular XC virial FD 1/3 Trace" in line:
            vals = [float(v) for v in line.split()[-3:]]
            gauxc_virial_fd = {"analytic": vals[0], "fd": vals[1], "diff": vals[2]}
    return {"energy": energy, "forces": forces, "gauxc_virial_fd": gauxc_virial_fd}


def run_input(label, inp):
    run_dir = WORK / "runs" / label
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "input.inp").write_text(inp)
    out = run_dir / "cp2k.out"
    if out.exists() and "PROGRAM ENDED" in out.read_text(errors="replace")[-5000:]:
        parsed = parse_output(out.read_text(errors="replace"))
        parsed.update({"returncode": 0, "label": label})
        return parsed
    if out.exists():
        out.unlink()
    stdout = run_dir / "stdout.log"
    if stdout.exists():
        stdout.unlink()
    proc = subprocess.run(
        [str(CP2K), "-i", "input.inp", "-o", "cp2k.out"],
        cwd=run_dir,
        env=BASE_ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=1800,
    )
    stdout.write_text(proc.stdout)
    text = out.read_text(errors="replace") if out.exists() else ""
    parsed = parse_output(text)
    parsed.update({"returncode": proc.returncode, "label": label})
    if proc.returncode != 0:
        parsed["error_tail"] = "\n".join((proc.stdout + "\n" + text).splitlines()[-80:])
    return parsed


def molecular_virial(coords, forces):
    if not forces:
        return None
    c = center(coords)
    value = 0.0
    for (el, x, y, z), force in zip(coords, forces):
        disp_bohr = [(x - c[0]) * ANG_TO_BOHR, (y - c[1]) * ANG_TO_BOHR, (z - c[2]) * ANG_TO_BOHR]
        f = [force[1], force[2], force[3]]
        value += sum(disp_bohr[i] * f[i] for i in range(3))
    return -value / 3.0


def run_case(system, method, functional, path, fd_dx, virial_dx):
    coords = SYSTEMS[system]["coords"]
    atom_i, axis = SYSTEMS[system]["fd_component"]
    key = f"{system}/{method}/{functional}/{path}"
    base = run_input(
        f"{system}_{method}_{functional}_{path}_base",
        render_input(system, method, functional, path, coords, debug_virial=(path == "gauxc")),
    )
    rec = {"base": base}
    if base["returncode"] != 0 or base["energy"] is None:
        return key, rec
    plus = run_input(
        f"{system}_{method}_{functional}_{path}_force_plus",
        render_input(system, method, functional, path, displaced(coords, atom_i, axis, fd_dx), run_type="ENERGY"),
    )
    minus = run_input(
        f"{system}_{method}_{functional}_{path}_force_minus",
        render_input(system, method, functional, path, displaced(coords, atom_i, axis, -fd_dx), run_type="ENERGY"),
    )
    rec["force_fd_component"] = {
        "atom_index_1based": atom_i + 1,
        "axis": "xyz"[axis],
        "dx_angstrom": fd_dx,
        "fd": None,
        "analytic": None,
        "diff": None,
    }
    if plus["energy"] is not None and minus["energy"] is not None and base["forces"]:
        fd = -(plus["energy"] - minus["energy"]) / (2.0 * fd_dx * ANG_TO_BOHR)
        analytic = base["forces"][atom_i][axis + 1]
        rec["force_fd_component"].update({"fd": fd, "analytic": analytic, "diff": analytic - fd})
    plus_v = run_input(
        f"{system}_{method}_{functional}_{path}_virial_plus",
        render_input(system, method, functional, path, scaled(coords, virial_dx), run_type="ENERGY"),
    )
    minus_v = run_input(
        f"{system}_{method}_{functional}_{path}_virial_minus",
        render_input(system, method, functional, path, scaled(coords, -virial_dx), run_type="ENERGY"),
    )
    analytic_v = molecular_virial(coords, base["forces"])
    rec["virial_fd"] = {"dx": virial_dx, "fd": None, "analytic": analytic_v, "diff": None}
    if plus_v["energy"] is not None and minus_v["energy"] is not None:
        fdv = (plus_v["energy"] - minus_v["energy"]) / (2.0 * virial_dx) / 3.0
        rec["virial_fd"].update({"fd": fdv, "diff": None if analytic_v is None else analytic_v - fdv})
    return key, rec


def write_summary(results):
    rows = [
        [
            "system",
            "method",
            "functional",
            "path",
            "returncode",
            "energy_ha",
            "force_fd_diff_ha_bohr",
            "virial_fd_diff_ha",
            "gauxc_xc_virial_diff_ha",
        ]
    ]
    for key in sorted(results):
        system, method, functional, path = key.split("/")
        rec = results[key]
        base = rec.get("base", {})
        ffd = rec.get("force_fd_component", {})
        vfd = rec.get("virial_fd", {})
        gvir = base.get("gauxc_virial_fd") or {}
        rows.append(
            [
                system,
                method,
                functional,
                path,
                base.get("returncode"),
                base.get("energy"),
                ffd.get("diff"),
                vfd.get("diff"),
                gvir.get("diff"),
            ]
        )
    SUMMARY.write_text("\n".join("\t".join("" if x is None else str(x) for x in row) for row in rows) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", default="H2O,OH")
    parser.add_argument("--methods", default="GPW_SCAN_GTH,GAPW_AE")
    parser.add_argument("--functionals", default="TPSS,R2SCAN")
    parser.add_argument("--paths", default="native,gauxc")
    parser.add_argument("--fd-dx", type=float, default=1.0e-3)
    parser.add_argument("--virial-dx", type=float, default=1.0e-4)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "runs").mkdir(parents=True, exist_ok=True)

    systems = [x for x in args.systems.split(",") if x]
    methods = [x for x in args.methods.split(",") if x]
    functionals = [x for x in args.functionals.split(",") if x]
    paths = [x for x in args.paths.split(",") if x]
    tasks = [(s, m, f, p) for s in systems for m in methods for f in functionals for p in paths]

    results = {}
    if RESULTS.exists() and not args.clean:
        results = json.loads(RESULTS.read_text())

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        future_map = {
            pool.submit(run_case, s, m, f, p, args.fd_dx, args.virial_dx): (s, m, f, p)
            for s, m, f, p in tasks
            if f"{s}/{m}/{f}/{p}" not in results
        }
        for future in concurrent.futures.as_completed(future_map):
            key, rec = future.result()
            results[key] = rec
            RESULTS.write_text(json.dumps(results, indent=2, sort_keys=True))
            write_summary(results)
            print("DONE", key, rec.get("base", {}).get("returncode"), flush=True)
    write_summary(results)
    print(RESULTS)
    print(SUMMARY)


if __name__ == "__main__":
    main()
