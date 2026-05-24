#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ANG_TO_BOHR = 1.8897261254578281

ROOT = Path("/home/kuehne88/cp2k-master-20260520")
BUILD = ROOT / "build-gauxc-skala"
CP2K = BUILD / "bin" / "cp2k.psmp"
WORK = Path("/home/kuehne88/skala-molopt-bench/runs")
RESULTS = Path("/home/kuehne88/skala-molopt-bench/results.json")

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
                os.environ.get("LD_LIBRARY_PATH", ""),
            ]
        ),
        "OMP_NUM_THREADS": "1",
        "OMP_STACKSIZE": "512M",
    }
)

SYSTEMS = {
    "H2": {
        "coords": [
            ("H", 6.000000, 6.000000, 5.630000),
            ("H", 6.000000, 6.000000, 6.370000),
        ],
        "fd_component": (1, 2),
    },
    "NH3": {
        "coords": [
            ("N", 6.000000, 6.000000, 6.000000),
            ("H", 6.000000, 6.937700, 5.618000),
            ("H", 5.187900, 5.531150, 5.618000),
            ("H", 6.812100, 5.531150, 5.618000),
        ],
        "fd_component": (1, 2),
    },
    "H2O": {
        "coords": [
            ("O", 6.000000, 6.000000, 6.000000),
            ("H", 6.000000, 6.756950, 6.585880),
            ("H", 6.000000, 5.243050, 6.585880),
        ],
        "fd_component": (1, 1),
    },
}

KINDS = {
    "GPW": {
        "method": "GPW",
        "basis": {
            "H": "TZV2P-MOLOPT-PBE-GTH-q1",
            "N": "TZV2P-MOLOPT-PBE-GTH-q5",
            "O": "TZV2P-MOLOPT-PBE-GTH-q6",
        },
        "potential": {"H": "GTH-PBE-q1", "N": "GTH-PBE-q5", "O": "GTH-PBE-q6"},
    },
    "GAPW_AE": {
        "method": "GAPW",
        "basis": {
            "H": "QZVPP-MOLOPT-PBE-ae",
            "N": "QZVPP-MOLOPT-PBE-ae",
            "O": "QZVPP-MOLOPT-PBE-ae",
        },
        "potential": {"H": "ALL", "N": "ALL", "O": "ALL"},
    },
}

PATHS = ["native_pbe", "gauxc_pbe", "skala"]


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
        new = [c[i] + (1.0 + lam) * (vals[i] - c[i]) for i in range(3)]
        out.append((el, *new))
    return out


def xc_block(path, debug_virial):
    if path == "native_pbe":
        return """      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL"""
    model = "PBE" if path == "gauxc_pbe" else "SKALA"
    extra = ""
    if debug_virial:
        extra = """
          MOLECULAR_VIRIAL T
          MOLECULAR_VIRIAL_DEBUG T
          MOLECULAR_VIRIAL_DEBUG_DX 1.0E-4"""
    return f"""      &XC_FUNCTIONAL
        &GAUXC
          FUNCTIONAL PBE
          MODEL {model}
          GRID FINE
          PRUNING_SCHEME ROBUST{extra}
        &END GAUXC
      &END XC_FUNCTIONAL"""


def render_input(system, method, path, coords, run_type="ENERGY_FORCE", debug_virial=False):
    spec = KINDS[method]
    elems = []
    for el, *_ in coords:
        if el not in elems:
            elems.append(el)

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
      BASIS_SET {spec['basis'][el]}
      POTENTIAL {spec['potential'][el]}
    &END KIND"""
        )

    return f"""&GLOBAL
  PRINT_LEVEL LOW
  PROJECT {system}_{method}_{path}
  RUN_TYPE {run_type}
&END GLOBAL

&FORCE_EVAL
  METHOD Quickstep
  &DFT
    BASIS_SET_FILE_NAME BASIS_MOLOPT_UZH
    POTENTIAL_FILE_NAME POTENTIAL_UZH
    UKS FALSE
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
            try:
                energy = float(line.split()[-1])
            except ValueError:
                pass
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
        if in_forces:
            if "SUM OF ATOMIC FORCES" in line:
                in_forces = False
                continue
            m = re.match(r"\s*\d+\s+\d+\s+([A-Za-z]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)", line)
            if m:
                forces.append((m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))))
        if in_forces_pipe:
            if "FORCES| Sum" in line:
                in_forces_pipe = False
                continue
            m = re.match(r"\s*FORCES\|\s+(\d+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+[-+0-9.Ee]+", line)
            if m:
                idx = int(m.group(1)) - 1
                el = f"A{idx+1}"
                forces.append((el, float(m.group(2)), float(m.group(3)), float(m.group(4))))
    gauxc_virial = None
    gauxc_virial_fd = None
    for line in text.splitlines():
        if "GAUXC| Molecular XC gradient virial 1/3 Trace" in line:
            gauxc_virial = float(line.split()[-1])
        if "GAUXC| Molecular XC virial FD 1/3 Trace" in line:
            vals = [float(v) for v in line.split()[-3:]]
            gauxc_virial_fd = {"analytic": vals[0], "fd": vals[1], "diff": vals[2]}
    return {"energy": energy, "forces": forces, "gauxc_virial": gauxc_virial, "gauxc_virial_fd": gauxc_virial_fd}


def run_case(label, inp):
    d = WORK / label
    d.mkdir(parents=True, exist_ok=True)
    (d / "input.inp").write_text(inp)
    out = d / "cp2k.out"
    if out.exists() and "PROGRAM ENDED" in out.read_text(errors="replace")[-4000:]:
        parsed = parse_output(out.read_text(errors="replace"))
        parsed["returncode"] = 0
        parsed["label"] = label
        return parsed
    cmd = [str(CP2K), "-i", "input.inp", "-o", "cp2k.out"]
    proc = subprocess.run(cmd, cwd=d, env=BASE_ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (d / "stdout.log").write_text(proc.stdout)
    text = out.read_text(errors="replace") if out.exists() else ""
    parsed = parse_output(text)
    parsed["returncode"] = proc.returncode
    parsed["label"] = label
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="H2,NH3,H2O")
    ap.add_argument("--methods", default="GPW,GAPW_AE")
    ap.add_argument("--paths", default="native_pbe,gauxc_pbe,skala")
    ap.add_argument("--fd-dx", type=float, default=1.0e-3)
    ap.add_argument("--virial-dx", type=float, default=1.0e-4)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    if args.clean and WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True, exist_ok=True)

    systems = [s for s in args.systems.split(",") if s]
    methods = [m for m in args.methods.split(",") if m]
    paths = [p for p in args.paths.split(",") if p]
    results = {}

    for system in systems:
        coords = SYSTEMS[system]["coords"]
        atom_i, axis = SYSTEMS[system]["fd_component"]
        for method in methods:
            for path in paths:
                key = f"{system}/{method}/{path}"
                print(f"RUN {key}", flush=True)
                base = run_case(
                    f"{system}_{method}_{path}_base",
                    render_input(system, method, path, coords, debug_virial=(path != "native_pbe")),
                )
                rec = {"base": base}
                if base["returncode"] == 0 and base["energy"] is not None:
                    rec["analytic_virial"] = molecular_virial(coords, base["forces"])
                    plus_coords = displaced(coords, atom_i, axis, args.fd_dx)
                    minus_coords = displaced(coords, atom_i, axis, -args.fd_dx)
                    plus = run_case(
                        f"{system}_{method}_{path}_force_plus",
                        render_input(system, method, path, plus_coords, run_type="ENERGY"),
                    )
                    minus = run_case(
                        f"{system}_{method}_{path}_force_minus",
                        render_input(system, method, path, minus_coords, run_type="ENERGY"),
                    )
                    rec["force_fd_component"] = {
                        "atom_index_1based": atom_i + 1,
                        "axis": "xyz"[axis],
                        "dx_angstrom": args.fd_dx,
                        "fd": None,
                        "analytic": None,
                        "diff": None,
                    }
                    if plus["energy"] is not None and minus["energy"] is not None and base["forces"]:
                        fd = -(plus["energy"] - minus["energy"]) / (2.0 * args.fd_dx * ANG_TO_BOHR)
                        analytic = base["forces"][atom_i][axis + 1]
                        rec["force_fd_component"].update({"fd": fd, "analytic": analytic, "diff": analytic - fd})
                    plus_v = run_case(
                        f"{system}_{method}_{path}_virial_plus",
                        render_input(system, method, path, scaled(coords, args.virial_dx), run_type="ENERGY"),
                    )
                    minus_v = run_case(
                        f"{system}_{method}_{path}_virial_minus",
                        render_input(system, method, path, scaled(coords, -args.virial_dx), run_type="ENERGY"),
                    )
                    rec["virial_fd"] = {"dx": args.virial_dx, "fd": None, "analytic": rec.get("analytic_virial"), "diff": None}
                    if plus_v["energy"] is not None and minus_v["energy"] is not None:
                        fdv = (plus_v["energy"] - minus_v["energy"]) / (2.0 * args.virial_dx) / 3.0
                        an = rec.get("analytic_virial")
                        rec["virial_fd"].update({"fd": fdv, "diff": None if an is None else an - fdv})
                results[key] = rec
                RESULTS.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(RESULTS)


if __name__ == "__main__":
    main()
