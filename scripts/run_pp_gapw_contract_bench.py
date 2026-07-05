#!/usr/bin/env python3
"""Pseudopotential GAPW contract diagnostics for molecular GauXC/Skala."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ANG_TO_BOHR = 1.8897261254578281

ROOT = Path(os.environ.get("PP_GAPW_CP2K_ROOT", "/home/kuehne88/cp2k-latest-skala-gapw-ecp-20260704"))
BUILD = Path(os.environ.get("PP_GAPW_CP2K_BUILD", ROOT / "build-host"))
CP2K = Path(os.environ.get("PP_GAPW_CP2K", BUILD / "bin/cp2k.psmp"))
WORK = Path(os.environ.get("PP_GAPW_WORK", "/home/kuehne88/pp-gapw-contracts-20260705"))
RESULTS = WORK / "results.json"
SUMMARY = WORK / "summary.tsv"


BASE_ENV = os.environ.copy()
BASE_ENV.update(
    {
        "CP2K_DATA_DIR": str(ROOT / "data"),
        "LD_LIBRARY_PATH": ":".join(
            [
                str(BUILD / "lib"),
                str(BUILD / "src"),
                "/home/kuehne88/cp2k-gauxc-skala-toolchain/libtorch-2.7.1/lib",
                "/home/kuehne88/cp2k-gauxc-skala-toolchain/libtorch-2.7.1/lib/torch.libs",
                "/home/kuehne88/cp2k-gauxc-skala-toolchain/gauxc-1.1-skala-cp2k-fixes/lib",
                "/home/kuehne88/cp2k-jpoto/tools/toolchain/install/libxsmm-e0c4a2389afba36c453233ad7de07bd92c715bec/lib",
                os.environ.get("LD_LIBRARY_PATH", ""),
            ]
        ),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "2"),
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "OMP_STACKSIZE": "512M",
        "GAUXC_SKALA_MODEL": os.environ.get("GAUXC_SKALA_MODEL", "/home/kuehne88/skala-cpu.fun"),
    }
)


SYSTEMS = {
    "H2_GTH": {
        "basis_file": "BASIS_MOLOPT_UZH",
        "potential_file": "POTENTIAL_UZH",
        "cell": 12.0,
        "coords": [
            ("H", 6.000000, 6.000000, 5.630000),
            ("H", 6.000000, 6.000000, 6.370000),
        ],
        "fd_component": (1, 2),
        "basis": {"H": "TZV2P-MOLOPT-PBE-GTH-q1"},
        "potential": {"H": "GTH-PBE-q1"},
    },
    "NH3_GTH": {
        "basis_file": "BASIS_MOLOPT_UZH",
        "potential_file": "POTENTIAL_UZH",
        "cell": 12.0,
        "coords": [
            ("N", 6.000000, 6.000000, 6.000000),
            ("H", 6.000000, 6.937700, 5.618000),
            ("H", 5.187900, 5.531150, 5.618000),
            ("H", 6.812100, 5.531150, 5.618000),
        ],
        "fd_component": (1, 2),
        "basis": {"N": "TZV2P-MOLOPT-PBE-GTH-q5", "H": "TZV2P-MOLOPT-PBE-GTH-q1"},
        "potential": {"N": "GTH-PBE-q5", "H": "GTH-PBE-q1"},
    },
    "H2O_GTH": {
        "basis_file": "BASIS_MOLOPT_UZH",
        "potential_file": "POTENTIAL_UZH",
        "cell": 12.0,
        "coords": [
            ("O", 6.000000, 6.000000, 6.000000),
            ("H", 6.000000, 6.756950, 6.585880),
            ("H", 6.000000, 5.243050, 6.585880),
        ],
        "fd_component": (1, 1),
        "basis": {"O": "TZV2P-MOLOPT-PBE-GTH-q6", "H": "TZV2P-MOLOPT-PBE-GTH-q1"},
        "potential": {"O": "GTH-PBE-q6", "H": "GTH-PBE-q1"},
    },
    "HCl_ECP": {
        "basis_file": None,
        "potential_file": str(ROOT / "tests/QS/regtest-ecp/ECP_BASIS_POT"),
        "cell": 6.0,
        "coords": [
            ("Cl", 0.000000, 0.000000, 0.000000),
            ("H", 0.000000, 0.000000, 1.300000),
        ],
        "fd_component": (1, 2),
        "basis": {"Cl": "DZVP-GTH-PADE", "H": "DZV-GTH-PADE"},
        "potential": {"Cl": "ECP ccECP", "H": "ECP ccECP"},
    },
}


METHODS = {
    "GPWTYPE": {"gpw_type": True},
    "PAW": {"gapw_one_center": True},
}


KIND_DEFAULTS = {
    "H": {"hard_exp_radius": "1.00", "radial_grid": 50, "lebedev_grid": 50},
    "N": {"hard_exp_radius": "1.40", "radial_grid": 50, "lebedev_grid": 50},
    "O": {"hard_exp_radius": "1.40", "radial_grid": 50, "lebedev_grid": 50},
    "Cl": {"hard_exp_radius": "1.60", "radial_grid": 50, "lebedev_grid": 50},
}


PATHS = ("native_pbe", "gauxc_pbe", "skala")


def unique_elements(coords):
    elems = []
    for el, *_ in coords:
        if el not in elems:
            elems.append(el)
    return elems


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


def xc_block(path, debug_virial):
    if path == "native_pbe":
        return """      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL"""
    if path == "gauxc_pbe":
        model = "PBE"
        functional = "          FUNCTIONAL PBE\n"
    elif path == "skala":
        model = BASE_ENV["GAUXC_SKALA_MODEL"]
        functional = ""
    else:
        raise ValueError(path)
    extra = ""
    if debug_virial:
        extra = """
          MOLECULAR_VIRIAL T
          MOLECULAR_VIRIAL_DEBUG T
          MOLECULAR_VIRIAL_DEBUG_DX 1.0E-4"""
    if path == "skala":
        extra += """
          ONEDFT_GRADIENT_RUNTIME SELF"""
    return f"""      &XC_FUNCTIONAL
        &GAUXC
{functional}          MODEL {model}
          GRID FINE
          PRUNING_SCHEME ROBUST{extra}
        &END GAUXC
      &END XC_FUNCTIONAL"""


def qs_block(method):
    spec = METHODS[method]
    lines = [
        "      EPS_DEFAULT 1.0E-14",
        "      EXTRAPOLATION USE_PREV_P",
        "      METHOD GAPW",
        "      GAPW_ACCURATE_XCINT T",
        "      EPSFIT 1.0E-4",
        "      EPSISO 1.0E-12",
        "      EPSRHO0 1.0E-8" if spec.get("gapw_one_center") else "      EPSRHO0 1.0E-6",
        "      EPS_GVG 1.0E-6",
        "      EPS_PGF_ORB 1.0E-6",
        "      EPSSVD 0.0",
        "      LMAXN0 4",
        "      GAPW_1C_BASIS EXT_SMALL",
    ]
    if spec.get("gapw_one_center"):
        lines += [
            "      QUADRATURE GC_LOG",
            "      LMAXN1 6",
            "      ALPHA0_HARD 10.0",
        ]
    return "\n".join(lines)


def kind_block(system, method, elems):
    sysinfo = SYSTEMS[system]
    spec = METHODS[method]
    blocks = []
    for el in elems:
        gapw_lines = []
        if spec.get("gpw_type"):
            gapw_lines.append("      GPW_TYPE")
        if spec.get("gapw_one_center"):
            defaults = KIND_DEFAULTS[el]
            gapw_lines += [
                f"      HARD_EXP_RADIUS {defaults['hard_exp_radius']}",
                f"      LEBEDEV_GRID {defaults['lebedev_grid']}",
                f"      RADIAL_GRID {defaults['radial_grid']}",
            ]
        gapw_extra = "\n" + "\n".join(gapw_lines) if gapw_lines else ""
        blocks.append(
            f"""    &KIND {el}
      BASIS_SET {sysinfo["basis"][el]}
      POTENTIAL {sysinfo["potential"][el]}{gapw_extra}
    &END KIND"""
        )
    return "\n".join(blocks)


def render_input(system, method, path, coords, run_type="ENERGY_FORCE", debug_virial=False):
    sysinfo = SYSTEMS[system]
    elems = unique_elements(coords)
    basis_line = f"    BASIS_SET_FILE_NAME {sysinfo['basis_file']}\n" if sysinfo["basis_file"] else ""
    potential_line = f"    POTENTIAL_FILE_NAME {sysinfo['potential_file']}\n"
    coord_lines = "\n".join(f"      {el:2s} {x:14.8f} {y:14.8f} {z:14.8f}" for el, x, y, z in coords)
    cell = sysinfo["cell"]
    max_scf = 200 if METHODS[method].get("gapw_one_center") else 100
    return f"""&GLOBAL
  PRINT_LEVEL LOW
  PROJECT {system}_{method}_{path}
  RUN_TYPE {run_type}
&END GLOBAL

&FORCE_EVAL
  METHOD Quickstep
  &DFT
{basis_line}{potential_line}    UKS FALSE
    MULTIPLICITY 1
    &MGRID
      CUTOFF 400
      NGRIDS 5
      REL_CUTOFF 60
    &END MGRID
    &POISSON
      PERIODIC NONE
      POISSON_SOLVER MT
    &END POISSON
    &QS
{qs_block(method)}
    &END QS
    &SCF
      EPS_SCF 1.0E-8
      MAX_SCF {max_scf}
      SCF_GUESS ATOMIC
      &DIAGONALIZATION
      &END DIAGONALIZATION
      &PRINT
        &RESTART OFF
        &END RESTART
      &END PRINT
    &END SCF
    &XC
{xc_block(path, debug_virial)}
    &END XC
  &END DFT
  &SUBSYS
    &CELL
      ABC {cell:.8f} {cell:.8f} {cell:.8f}
      PERIODIC NONE
    &END CELL
    &COORD
{coord_lines}
    &END COORD
{kind_block(system, method, elems)}
    &TOPOLOGY
      &CENTER_COORDINATES
      &END CENTER_COORDINATES
    &END TOPOLOGY
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


def run_input(label, inp, timeout):
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
    proc = subprocess.run(
        [str(CP2K), "-i", "input.inp", "-o", "cp2k.out"],
        cwd=run_dir,
        env=BASE_ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    (run_dir / "stdout.log").write_text(proc.stdout)
    text = out.read_text(errors="replace") if out.exists() else ""
    parsed = parse_output(text)
    parsed.update({"returncode": proc.returncode, "label": label})
    if proc.returncode != 0:
        parsed["error_tail"] = "\n".join((proc.stdout + "\n" + text).splitlines()[-100:])
    return parsed


def molecular_virial(coords, forces):
    if not forces:
        return None
    c = center(coords)
    value = 0.0
    for (_, x, y, z), force in zip(coords, forces):
        disp_bohr = [(x - c[0]) * ANG_TO_BOHR, (y - c[1]) * ANG_TO_BOHR, (z - c[2]) * ANG_TO_BOHR]
        f = [force[1], force[2], force[3]]
        value += sum(disp_bohr[i] * f[i] for i in range(3))
    return -value / 3.0


def run_case(system, method, path, fd_dx, virial_dx, timeout):
    coords = SYSTEMS[system]["coords"]
    atom_i, axis = SYSTEMS[system]["fd_component"]
    key = f"{system}/{method}/{path}"
    base = run_input(
        f"{system}_{method}_{path}_base",
        render_input(system, method, path, coords, debug_virial=False),
        timeout,
    )
    rec = {"base": base}
    if os.environ.get("PP_GAPW_BASE_ONLY") == "1":
        return key, rec
    if base["returncode"] != 0 or base["energy"] is None:
        return key, rec

    plus = run_input(
        f"{system}_{method}_{path}_force_plus",
        render_input(system, method, path, displaced(coords, atom_i, axis, fd_dx), run_type="ENERGY"),
        timeout,
    )
    minus = run_input(
        f"{system}_{method}_{path}_force_minus",
        render_input(system, method, path, displaced(coords, atom_i, axis, -fd_dx), run_type="ENERGY"),
        timeout,
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
        f"{system}_{method}_{path}_virial_plus",
        render_input(system, method, path, scaled(coords, virial_dx), run_type="ENERGY"),
        timeout,
    )
    minus_v = run_input(
        f"{system}_{method}_{path}_virial_minus",
        render_input(system, method, path, scaled(coords, -virial_dx), run_type="ENERGY"),
        timeout,
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
            "path",
            "returncode",
            "energy_ha",
            "force_fd_diff_ha_bohr",
            "virial_fd_diff_ha",
            "gauxc_xc_virial_diff_ha",
        ]
    ]
    for key in sorted(results):
        system, method, path = key.split("/")
        rec = results[key]
        base = rec.get("base", {})
        ffd = rec.get("force_fd_component", {})
        vfd = rec.get("virial_fd", {})
        gvir = base.get("gauxc_virial_fd") or {}
        rows.append(
            [
                system,
                method,
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
    parser.add_argument("--systems", default="H2_GTH,NH3_GTH,H2O_GTH,HCl_ECP")
    parser.add_argument("--methods", default="GPWTYPE,PAW")
    parser.add_argument("--paths", default="native_pbe,gauxc_pbe,skala")
    parser.add_argument("--fd-dx", type=float, default=1.0e-3)
    parser.add_argument("--virial-dx", type=float, default=1.0e-4)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "runs").mkdir(parents=True, exist_ok=True)

    systems = [x for x in args.systems.split(",") if x]
    methods = [x for x in args.methods.split(",") if x]
    paths = [x for x in args.paths.split(",") if x]
    for x in systems:
        if x not in SYSTEMS:
            raise ValueError(f"Unknown system: {x}")
    for x in methods:
        if x not in METHODS:
            raise ValueError(f"Unknown method: {x}")
    for x in paths:
        if x not in PATHS:
            raise ValueError(f"Unknown path: {x}")

    tasks = [
        (s, m, p)
        for s in systems
        for m in methods
        for p in paths
    ]
    results = {}
    if RESULTS.exists() and not args.clean:
        results = json.loads(RESULTS.read_text())

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        future_map = {
            pool.submit(run_case, s, m, p, args.fd_dx, args.virial_dx, args.timeout): (s, m, p)
            for s, m, p in tasks
            if f"{s}/{m}/{p}" not in results
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
