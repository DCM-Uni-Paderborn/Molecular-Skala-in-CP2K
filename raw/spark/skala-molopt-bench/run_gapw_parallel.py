#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import time
from pathlib import Path

import run_bench as rb


def run_case(system, method, path):
    coords = rb.SYSTEMS[system]["coords"]
    atom_i, axis = rb.SYSTEMS[system]["fd_component"]
    key = f"{system}/{method}/{path}"
    debug_virial = path != "native_pbe"

    base = rb.run_case(
        f"{system}_{method}_{path}_base",
        rb.render_input(system, method, path, coords, debug_virial=debug_virial),
    )
    rec = {"base": base}
    if base["returncode"] == 0 and base["energy"] is not None:
        rec["analytic_virial"] = rb.molecular_virial(coords, base["forces"])

        plus_coords = rb.displaced(coords, atom_i, axis, rb_fd_dx)
        minus_coords = rb.displaced(coords, atom_i, axis, -rb_fd_dx)
        plus = rb.run_case(
            f"{system}_{method}_{path}_force_plus",
            rb.render_input(system, method, path, plus_coords, run_type="ENERGY"),
        )
        minus = rb.run_case(
            f"{system}_{method}_{path}_force_minus",
            rb.render_input(system, method, path, minus_coords, run_type="ENERGY"),
        )
        rec["force_fd_component"] = {
            "atom_index_1based": atom_i + 1,
            "axis": "xyz"[axis],
            "dx_angstrom": rb_fd_dx,
            "fd": None,
            "analytic": None,
            "diff": None,
        }
        if plus["energy"] is not None and minus["energy"] is not None and base["forces"]:
            fd = -(plus["energy"] - minus["energy"]) / (2.0 * rb_fd_dx * rb.ANG_TO_BOHR)
            analytic = base["forces"][atom_i][axis + 1]
            rec["force_fd_component"].update({"fd": fd, "analytic": analytic, "diff": analytic - fd})

        plus_v = rb.run_case(
            f"{system}_{method}_{path}_virial_plus",
            rb.render_input(system, method, path, rb.scaled(coords, rb_virial_dx), run_type="ENERGY"),
        )
        minus_v = rb.run_case(
            f"{system}_{method}_{path}_virial_minus",
            rb.render_input(system, method, path, rb.scaled(coords, -rb_virial_dx), run_type="ENERGY"),
        )
        rec["virial_fd"] = {"dx": rb_virial_dx, "fd": None, "analytic": rec.get("analytic_virial"), "diff": None}
        if plus_v["energy"] is not None and minus_v["energy"] is not None:
            fdv = (plus_v["energy"] - minus_v["energy"]) / (2.0 * rb_virial_dx) / 3.0
            an = rec.get("analytic_virial")
            rec["virial_fd"].update({"fd": fdv, "diff": None if an is None else an - fdv})
    return key, rec


def load_results():
    if rb.RESULTS.exists():
        return json.loads(rb.RESULTS.read_text())
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", default="H2,NH3,H2O")
    parser.add_argument("--methods", default="GAPW_AE")
    parser.add_argument("--paths", default="native_pbe,gauxc_pbe,skala")
    parser.add_argument("--fd-dx", type=float, default=1.0e-3)
    parser.add_argument("--virial-dx", type=float, default=1.0e-4)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    global rb_fd_dx, rb_virial_dx
    rb_fd_dx = args.fd_dx
    rb_virial_dx = args.virial_dx

    rb.BASE_ENV.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "OMP_STACKSIZE": "512M",
        }
    )

    rb.WORK.mkdir(parents=True, exist_ok=True)
    systems = [item for item in args.systems.split(",") if item]
    methods = [item for item in args.methods.split(",") if item]
    paths = [item for item in args.paths.split(",") if item]
    tasks = [(system, method, path) for system in systems for method in methods for path in paths]

    results = load_results()
    print(f"parallel tasks={len(tasks)} workers={args.max_workers}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        future_map = {pool.submit(run_case, *task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_map):
            task = future_map[future]
            key = "/".join(task)
            try:
                key, rec = future.result()
                results[key] = rec
                rb.RESULTS.write_text(json.dumps(results, indent=2, sort_keys=True))
                base = rec.get("base", {})
                print(f"DONE {key} rc={base.get('returncode')} energy={base.get('energy')}", flush=True)
            except Exception as exc:
                results[key] = {"error": repr(exc), "time": time.time()}
                rb.RESULTS.write_text(json.dumps(results, indent=2, sort_keys=True))
                print(f"FAILED {key}: {exc!r}", flush=True)
    print(rb.RESULTS, flush=True)


if __name__ == "__main__":
    main()
