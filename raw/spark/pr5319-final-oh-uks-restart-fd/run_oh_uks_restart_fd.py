#!/usr/bin/env python3
"""OH/UKS force and virial finite-difference check with restart continuity."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import run_bench as rb


def with_restart_output(text):
    return text.replace(
        "        &RESTART OFF\n        &END RESTART",
        "        &RESTART ON\n        &END RESTART",
    )


def with_restart_guess(text, wfn_path):
    text = text.replace("      SCF_GUESS ATOMIC\n", "      SCF_GUESS RESTART\n", 1)
    text = text.replace(
        "    &SCF\n",
        f"    WFN_RESTART_FILE_NAME {wfn_path}\n    &SCF\n",
        1,
    )
    return text


def make_uks(text):
    eps_scf = os.environ.get("FINAL_OH_EPS_SCF", "1.0E-7")
    text = text.replace("    UKS FALSE\n", "    MULTIPLICITY 2\n    UKS TRUE\n", 1)
    return text.replace("      EPS_SCF 1.0E-8\n", f"      EPS_SCF {eps_scf}\n", 1)


def render_oh(method, path, coords, run_type="ENERGY_FORCE", debug_virial=False, restart_wfn=None):
    text = rb.render_input("OH", method, path, coords, run_type=run_type, debug_virial=debug_virial)
    text = make_uks(text)
    grid = os.environ.get("FINAL_GAUXC_GRID")
    pruning = os.environ.get("FINAL_GAUXC_PRUNING")
    if grid:
        text = text.replace("          GRID FINE\n", f"          GRID {grid}\n")
    if pruning:
        text = text.replace("          PRUNING_SCHEME ROBUST", f"          PRUNING_SCHEME {pruning}")
    if restart_wfn is None:
        return with_restart_output(text)
    return with_restart_guess(text, restart_wfn)


def run_case(label, inp):
    d = rb.WORK / label
    d.mkdir(parents=True, exist_ok=True)
    (d / "input.inp").write_text(inp)
    out = d / "cp2k.out"
    proc = subprocess.run(
        [str(rb.CP2K), "-i", "input.inp", "-o", "cp2k.out"],
        cwd=d,
        env=rb.BASE_ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    (d / "stdout.log").write_text(proc.stdout)
    text = out.read_text(errors="replace") if out.exists() else ""
    parsed = rb.parse_output(text)
    parsed["returncode"] = proc.returncode
    parsed["label"] = label
    if proc.returncode != 0:
        parsed["error_tail"] = "\n".join((proc.stdout + "\n" + text).splitlines()[-80:])
    return parsed


def first_restart_wfn(label):
    d = rb.WORK / label
    files = sorted(d.glob("*-RESTART.wfn"))
    if not files:
        raise FileNotFoundError(f"No restart WFN generated in {d}")
    return files[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="GPW,GAPW_AE")
    ap.add_argument("--paths", default="native_pbe,gauxc_pbe,skala")
    ap.add_argument("--fd-dx", type=float, default=1.0e-3)
    ap.add_argument("--virial-dx", type=float, default=1.0e-4)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    cp2k_root = Path(os.environ["FINAL_CP2K_ROOT"])
    build_name = os.environ.get("FINAL_CP2K_BUILD", "build-gauxc-final")
    work_root = Path(os.environ["FINAL_WORK_ROOT"])
    torch_lib = os.environ.get("FINAL_TORCH_LIB", "")
    skala_model = os.environ.get("FINAL_SKALA_MODEL", "")

    rb.ROOT = cp2k_root
    rb.BUILD = cp2k_root / build_name
    rb.CP2K = rb.BUILD / "bin" / "cp2k.psmp"
    rb.WORK = work_root / "oh-uks-restart-fd" / "runs"
    rb.RESULTS = work_root / "oh-uks-restart-fd" / "results.json"

    if args.clean and rb.WORK.exists():
        shutil.rmtree(rb.WORK)
    rb.WORK.mkdir(parents=True, exist_ok=True)

    rb.SYSTEMS["OH"] = {
        "coords": [
            ("O", 6.000000, 6.000000, 6.000000),
            ("H", 6.000000, 6.000000, 6.970000),
        ],
        "fd_component": (1, 2),
    }

    env = os.environ.copy()
    env["CP2K_DATA_DIR"] = str(cp2k_root / "data")
    env["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["OMP_STACKSIZE"] = os.environ.get("OMP_STACKSIZE", "512M")
    if torch_lib:
        env["LD_LIBRARY_PATH"] = torch_lib + ":" + env.get("LD_LIBRARY_PATH", "")
    if skala_model:
        env["GAUXC_SKALA_MODEL"] = skala_model
    rb.BASE_ENV = env

    coords = rb.SYSTEMS["OH"]["coords"]
    atom_i, axis = rb.SYSTEMS["OH"]["fd_component"]
    results = {}
    for method in [m for m in args.methods.split(",") if m]:
        for path in [p for p in args.paths.split(",") if p]:
            key = f"OH/{method}/{path}"
            print(f"RUN {key}", flush=True)
            base_label = f"OH_{method}_{path}_base"
            base = run_case(base_label, render_oh(method, path, coords, debug_virial=(path != "native_pbe")))
            rec = {"base": base}
            if base["returncode"] == 0 and base["energy"] is not None:
                wfn = str(first_restart_wfn(base_label))
                rec["analytic_virial"] = rb.molecular_virial(coords, base["forces"])
                plus_coords = rb.displaced(coords, atom_i, axis, args.fd_dx)
                minus_coords = rb.displaced(coords, atom_i, axis, -args.fd_dx)
                plus = run_case(
                    f"OH_{method}_{path}_force_plus",
                    render_oh(method, path, plus_coords, run_type="ENERGY", restart_wfn=wfn),
                )
                minus = run_case(
                    f"OH_{method}_{path}_force_minus",
                    render_oh(method, path, minus_coords, run_type="ENERGY", restart_wfn=wfn),
                )
                rec["force_fd_component"] = {
                    "atom_index_1based": atom_i + 1,
                    "axis": "xyz"[axis],
                    "dx_angstrom": args.fd_dx,
                    "fd": None,
                    "analytic": None,
                    "diff": None,
                    "plus_returncode": plus["returncode"],
                    "minus_returncode": minus["returncode"],
                }
                if plus["energy"] is not None and minus["energy"] is not None and base["forces"]:
                    fd = -(plus["energy"] - minus["energy"]) / (2.0 * args.fd_dx * rb.ANG_TO_BOHR)
                    analytic = base["forces"][atom_i][axis + 1]
                    rec["force_fd_component"].update({"fd": fd, "analytic": analytic, "diff": analytic - fd})
                plus_v = run_case(
                    f"OH_{method}_{path}_virial_plus",
                    render_oh(method, path, rb.scaled(coords, args.virial_dx), run_type="ENERGY", restart_wfn=wfn),
                )
                minus_v = run_case(
                    f"OH_{method}_{path}_virial_minus",
                    render_oh(method, path, rb.scaled(coords, -args.virial_dx), run_type="ENERGY", restart_wfn=wfn),
                )
                rec["virial_fd"] = {
                    "dx": args.virial_dx,
                    "fd": None,
                    "analytic": rec.get("analytic_virial"),
                    "diff": None,
                    "plus_returncode": plus_v["returncode"],
                    "minus_returncode": minus_v["returncode"],
                }
                if plus_v["energy"] is not None and minus_v["energy"] is not None:
                    fdv = (plus_v["energy"] - minus_v["energy"]) / (2.0 * args.virial_dx) / 3.0
                    an = rec.get("analytic_virial")
                    rec["virial_fd"].update({"fd": fdv, "diff": None if an is None else an - fdv})
            results[key] = rec
            rb.RESULTS.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(rb.RESULTS)


if __name__ == "__main__":
    main()
