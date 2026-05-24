#!/usr/bin/env python3
import argparse
import concurrent.futures
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

HOME = Path.home()
ROOT = Path(os.environ.get("SKALA_HEXAMER_ROOT", HOME / "begdb-water-hexamer-skala-bench"))
STRUCTURES = Path(os.environ.get("SKALA_HEXAMER_STRUCTURES", ROOT / "structures"))
RESULTS = ROOT / "results.json"
SUMMARY = ROOT / "begdb_hexamer_binding_energies.tsv"
RELATIVE = ROOT / "begdb_hexamer_relative_energies.tsv"
ABSOLUTE = ROOT / "begdb_hexamer_absolute_energies.tsv"
AE_BASIS = os.environ.get("SKALA_HEXAMER_AE_BASIS", "TZVPP-MOLOPT-PBE-ae")

sys.path.insert(0, str(HOME / "skala-molopt-bench"))
import run_bench as rb  # noqa: E402

KCAL_PER_HARTREE = 627.5094740631

ISOMERS = [
    ("water-6-PR", -46.14),
    ("water-6-CA", -45.93),
    ("water-6-BK-1", -45.51),
    ("water-6-BK-2", -45.14),
    ("water-6-CC", -44.60),
    ("water-6-BAG", -44.59),
    ("water-6-CB-1", -43.57),
    ("water-6-CB-2", -43.51),
]


def configure():
    rb.WORK = ROOT / "runs"
    rb.RESULTS = RESULTS
    rb.KINDS["GPW_PBE_GTH"] = {
        "method": "GPW",
        "basis": {
            "H": "TZV2P-MOLOPT-PBE-GTH-q1",
            "O": "TZV2P-MOLOPT-PBE-GTH-q6",
        },
        "potential": {"H": "GTH-PBE-q1", "O": "GTH-PBE-q6"},
    }
    rb.KINDS["GPW_SCAN_GTH"] = {
        "method": "GPW",
        "basis": {
            "H": "TZV2P-MOLOPT-SCAN-GTH-q1",
            "O": "TZV2P-MOLOPT-SCAN-GTH-q6",
        },
        "potential": {"H": "GTH-SCAN-q1", "O": "GTH-SCAN-q6"},
    }
    rb.KINDS["GAPW_AE"]["basis"]["H"] = AE_BASIS
    rb.KINDS["GAPW_AE"]["basis"]["O"] = AE_BASIS
    extra_libs = [
        HOME / "cp2k-gauxc-skala-toolchain/torch-venv-aarch64/lib/python3.12/site-packages/torch/lib",
        HOME / "cp2k-gauxc-skala-toolchain/torch-venv-aarch64/lib/python3.12/site-packages/torch.libs",
        HOME / "cp2k-gauxc-skala-toolchain/gauxc-1.1-skala-cp2k-fixes/lib",
        HOME / "cp2k-jpoto/tools/toolchain/install/libxsmm-e0c4a2389afba36c453233ad7de07bd92c715bec/lib",
    ]
    rb.BASE_ENV["LD_LIBRARY_PATH"] = ":".join(str(path) for path in extra_libs) + ":" + rb.BASE_ENV.get("LD_LIBRARY_PATH", "")
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


def read_xyz(path):
    lines = path.read_text().splitlines()
    natom = int(lines[0].strip())
    coords = []
    for line in lines[2 : 2 + natom]:
        fields = line.split()
        coords.append((fields[0], float(fields[1]), float(fields[2]), float(fields[3])))
    return coords


def shifted_to_cell_center(coords, target=6.0):
    center = [sum(atom[i] for atom in (c[1:] for c in coords)) / len(coords) for i in range(3)]
    out = []
    for el, x, y, z in coords:
        out.append((el, x - center[0] + target, y - center[1] + target, z - center[2] + target))
    return out


def load_coords(system):
    return shifted_to_cell_center(read_xyz(STRUCTURES / f"{system}.xyz"))


def add_d3bj(input_text, reference_functional):
    d3_block = """      &VDW_POTENTIAL
        DISPERSION_FUNCTIONAL PAIR_POTENTIAL
        &PAIR_POTENTIAL
          PARAMETER_FILE_NAME dftd3.dat
          REFERENCE_FUNCTIONAL {reference_functional}
          R_CUTOFF 20.0
          TYPE DFTD3(BJ)
        &END PAIR_POTENTIAL
      &END VDW_POTENTIAL
""".format(reference_functional=reference_functional)
    return input_text.replace("      &XC_FUNCTIONAL PBE", d3_block + "      &XC_FUNCTIONAL PBE", 1)


def load_results():
    if RESULTS.exists():
        return json.loads(RESULTS.read_text())
    return {}


def run_one(system, method, path):
    coords = load_coords(system)
    project = system.replace("-", "_")
    label = f"{project}_{method}_{path}_sp"
    render_path = "native_pbe" if path in ("native_pbe_d3bj", "native_pbe_d3bj_b3lyp5") else path
    inp = rb.render_input(project, method, render_path, coords, run_type="ENERGY")
    if path == "native_pbe_d3bj":
        inp = add_d3bj(inp, "PBE")
    elif path == "native_pbe_d3bj_b3lyp5":
        inp = add_d3bj(inp, "B3LYP")
    elif path in ("gauxc_pbe", "skala"):
        native_project = f"{project}_{method}_native_pbe"
        src = rb.WORK / f"{native_project}_sp" / f"{native_project}-RESTART.wfn"
        dst_dir = rb.WORK / label
        dst = dst_dir / f"{project}_{method}_{path}-RESTART.wfn"
        if src.exists():
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            inp = inp.replace("SCF_GUESS ATOMIC", "SCF_GUESS RESTART")
    rec = rb.run_case(label, inp)
    return f"{system}/{method}/{path}", rec


def column_specs(methods):
    direct_columns = []
    report_columns = []
    for method in methods:
        direct = [
            (f"{method}/native_pbe", f"{method}_PBE"),
            (f"{method}/native_pbe_d3bj", f"{method}_PBE_D3BJ"),
            (f"{method}/gauxc_pbe", f"{method}_GauXC_PBE"),
            (f"{method}/skala", f"{method}_SKALA_XC"),
            (f"{method}/native_pbe_d3bj_b3lyp5", f"{method}_D3BJ_B3LYP5_AUX"),
        ]
        direct_columns.extend(direct)
        report_columns.extend(direct[:-1] + [(f"{method}/skala_d3bj", f"{method}_SKALA_D3BJ")])
    return direct_columns, report_columns


def write_tables(results, methods):
    direct_columns, report_columns = column_specs(methods)

    monomers = {method_path: results.get(f"H2O/{method_path}", {}).get("energy") for method_path, _ in direct_columns}
    energies = {}
    for system, _ in ISOMERS:
        for method_path, _ in direct_columns:
            energies[(system, method_path)] = results.get(f"{system}/{method_path}", {}).get("energy")

    binding = {}
    for system, _ in ISOMERS:
        for method_path, _ in direct_columns:
            cluster = energies[(system, method_path)]
            monomer = monomers.get(method_path)
            if cluster is None or monomer is None:
                binding[(system, method_path)] = None
            else:
                binding[(system, method_path)] = (cluster - 6.0 * monomer) * KCAL_PER_HARTREE
        for method in methods:
            skala = binding[(system, f"{method}/skala")]
            pbe = binding[(system, f"{method}/native_pbe")]
            b3lyp_d3_aux = binding[(system, f"{method}/native_pbe_d3bj_b3lyp5")]
            if skala is None or pbe is None or b3lyp_d3_aux is None:
                binding[(system, f"{method}/skala_d3bj")] = None
            else:
                binding[(system, f"{method}/skala_d3bj")] = skala + (b3lyp_d3_aux - pbe)

    with SUMMARY.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["system", "BEGDB_CCSDT_CBS_noCP_binding_kcal_mol"] + [name + "_binding_kcal_mol" for _, name in report_columns])
        for system, reference in ISOMERS:
            row = [system, f"{reference:.2f}"]
            for method_path, _ in report_columns:
                value = binding[(system, method_path)]
                row.append("" if value is None else f"{value:.3f}")
            writer.writerow(row)

    min_ref = min(ref for _, ref in ISOMERS)
    minima = {}
    for method_path, _ in report_columns:
        values = [binding[(system, method_path)] for system, _ in ISOMERS]
        values = [value for value in values if value is not None]
        minima[method_path] = min(values) if values else None
    with RELATIVE.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["system", "BEGDB_relative_kcal_mol"] + [name + "_relative_kcal_mol" for _, name in report_columns])
        for system, reference in ISOMERS:
            row = [system, f"{reference - min_ref:.2f}"]
            for method_path, _ in report_columns:
                value = binding[(system, method_path)]
                minimum = minima[method_path]
                row.append("" if value is None or minimum is None else f"{value - minimum:.3f}")
            writer.writerow(row)

    with ABSOLUTE.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["system"] + [name + "_total_hartree" for _, name in direct_columns])
        writer.writerow(["H2O"] + ["" if monomers[method_path] is None else f"{monomers[method_path]:.12f}" for method_path, _ in direct_columns])
        for system, _ in ISOMERS:
            writer.writerow([system] + ["" if energies[(system, method_path)] is None else f"{energies[(system, method_path)]:.12f}" for method_path, _ in direct_columns])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="GAPW_AE")
    parser.add_argument("--paths", default="native_pbe,native_pbe_d3bj,native_pbe_d3bj_b3lyp5,gauxc_pbe,skala")
    parser.add_argument("--systems", default="")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--refresh-tables-only", action="store_true")
    args = parser.parse_args()

    configure()
    print(f"root={ROOT}", flush=True)
    print(f"structures={STRUCTURES}", flush=True)
    print(f"ae_basis={AE_BASIS}", flush=True)
    methods = [item for item in args.methods.split(",") if item]
    paths = [item for item in args.paths.split(",") if item]
    systems = [item for item in args.systems.split(",") if item] or ["H2O"] + [system for system, _ in ISOMERS]
    tasks = [(system, method, path) for system in systems for method in methods for path in paths]
    results = load_results()

    if not args.refresh_tables_only:
        print(f"tasks={len(tasks)} workers={args.max_workers}", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {pool.submit(run_one, *task): task for task in tasks}
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                key = f"{task[0]}/{task[1]}/{task[2]}"
                try:
                    key, rec = future.result()
                    results[key] = rec
                    print(f"DONE {key} rc={rec.get('returncode')} energy={rec.get('energy')}", flush=True)
                except Exception as exc:
                    results[key] = {"error": repr(exc), "time": time.time()}
                    print(f"FAILED {key}: {exc!r}", flush=True)
                RESULTS.write_text(json.dumps(results, indent=2, sort_keys=True))
                write_tables(results, methods)

    write_tables(results, methods)
    print(SUMMARY, flush=True)
    print(RELATIVE, flush=True)
    print(ABSOLUTE, flush=True)


if __name__ == "__main__":
    main()
