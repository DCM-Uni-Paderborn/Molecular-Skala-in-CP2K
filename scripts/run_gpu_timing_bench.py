#!/usr/bin/env python3
"""CPU/GPU timing benchmark for molecular SKALA through GauXC in CP2K."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path


DEFAULT_CPU_CP2K = Path(
    "/home/kuehne88/cp2k-current-gauxc-gpu-20260524/build-gauxc-gpu/bin/cp2k.psmp"
)
DEFAULT_GPU_CP2K = Path(
    "/home/kuehne88/cp2k-current-gauxc-gpu-20260524/build-gauxc-device-torch-cu130/bin/cp2k.psmp"
)
DEFAULT_DATA_DIR = Path("/home/kuehne88/cp2k-current-gauxc-gpu-20260524/data")
DEFAULT_WORK = Path("/home/kuehne88/skala-gauxc-gpu-timing")
CPU_MODEL_SOURCE = Path("/home/kuehne88/cp2k-gauxc-skala-toolchain/gauxc-1.1-skala-cp2k-fixes/share/gauxc/onedft_models/skala-1.1.fun")
GPU_MODEL_SOURCE = Path("/home/kuehne88/cp2k-gauxc-skala-toolchain/gauxc-skala-gpu-torch-cu130-20260524/share/gauxc/onedft_models/skala-1.1-cuda.fun")
DEFAULT_CPU_MODEL = "/home/kuehne88/skala-cpu.fun"
DEFAULT_GPU_MODEL = "/home/kuehne88/skala-gpu.fun"


WATER = [
    ("O", 0.000000, 0.000000, 0.000000),
    ("H", 0.000000, 0.756950, 0.585880),
    ("H", 0.000000, -0.756950, 0.585880),
]


def make_water_cluster(nwater: int, spacing: float = 3.2) -> tuple[list[tuple[str, float, float, float]], float]:
    side = math.ceil(nwater ** (1.0 / 3.0))
    coords: list[tuple[str, float, float, float]] = []
    count = 0
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                if count >= nwater:
                    break
                ox = ix * spacing
                oy = iy * spacing
                oz = iz * spacing
                # Alternate the molecular orientation a little to avoid an exactly repeated motif.
                sign = -1.0 if (ix + iy + iz) % 2 else 1.0
                for el, x, y, z in WATER:
                    coords.append((el, ox + x, oy + sign * y, oz + z))
                count += 1
            if count >= nwater:
                break
        if count >= nwater:
            break

    minv = [min(atom[i] for atom in (c[1:] for c in coords)) for i in range(3)]
    maxv = [max(atom[i] for atom in (c[1:] for c in coords)) for i in range(3)]
    padding = 6.0
    shifted = []
    for el, x, y, z in coords:
        shifted.append((el, x - minv[0] + 0.5 * padding, y - minv[1] + 0.5 * padding, z - minv[2] + 0.5 * padding))
    cell = max(maxv[i] - minv[i] for i in range(3)) + padding
    return shifted, cell


def render_input(
    nwater: int,
    coords: list[tuple[str, float, float, float]],
    cell: float,
    mode: str,
    model_path: str,
    cutoff: int,
    rel_cutoff: int,
    max_scf: int,
    eps_scf: str,
    grid: str,
    pruning: str,
    batch_size: int,
) -> str:
    execution_space = "DEVICE" if mode == "gpu_device" else "HOST"
    coord_lines = "\n".join(f"      {el:2s} {x:14.8f} {y:14.8f} {z:14.8f}" for el, x, y, z in coords)
    return f"""&GLOBAL
  PRINT_LEVEL LOW
  PROJECT skala_h2o_{nwater}_{mode}
  RUN_TYPE ENERGY
&END GLOBAL

&FORCE_EVAL
  METHOD Quickstep
  &DFT
    BASIS_SET_FILE_NAME BASIS_MOLOPT_UZH
    POTENTIAL_FILE_NAME POTENTIAL_UZH
    UKS FALSE
    &MGRID
      CUTOFF {cutoff}
      REL_CUTOFF {rel_cutoff}
    &END MGRID
    &POISSON
      PERIODIC NONE
      POISSON_SOLVER MT
    &END POISSON
    &QS
      EPS_DEFAULT 1.0E-8
      EXTRAPOLATION USE_PREV_P
      METHOD GPW
    &END QS
    &SCF
      EPS_SCF {eps_scf}
      MAX_SCF {max_scf}
      IGNORE_CONVERGENCE_FAILURE T
      SCF_GUESS ATOMIC
      &DIAGONALIZATION
      &END DIAGONALIZATION
      &PRINT
        &RESTART OFF
        &END RESTART
      &END PRINT
    &END SCF
    &XC
      &XC_FUNCTIONAL
        &GAUXC
          FUNCTIONAL PBE
          MODEL {model_path}
          GRID {grid}
          PRUNING_SCHEME {pruning}
          BATCH_SIZE {batch_size}
          LB_EXECUTION_SPACE {execution_space}
          INT_EXECUTION_SPACE {execution_space}
        &END GAUXC
      &END XC_FUNCTIONAL
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
    &KIND H
      BASIS_SET TZV2P-MOLOPT-PBE-GTH-q1
      POTENTIAL GTH-PBE-q1
    &END KIND
    &KIND O
      BASIS_SET TZV2P-MOLOPT-PBE-GTH-q6
      POTENTIAL GTH-PBE-q6
    &END KIND
  &END SUBSYS
&END FORCE_EVAL
"""


def base_env(data_dir: Path, omp_threads: int, mode: str) -> dict[str, str]:
    env = os.environ.copy()
    if mode == "gpu_device":
        lib_paths = [
            "/lib/aarch64-linux-gnu",
            "/usr/lib/aarch64-linux-gnu",
            "/usr/local/cuda-13.0/lib64",
            "/home/kuehne88/cp2k-current-gauxc-gpu-20260524/build-gauxc-device-torch-cu130/src",
            "/home/kuehne88/cp2k-gauxc-skala-toolchain/gauxc-skala-gpu-torch-cu130-20260524/lib",
            "/home/kuehne88/cp2k-gauxc-skala-toolchain/pytorch-cu130-venv/lib/python3.12/site-packages/torch/lib",
            env.get("LD_LIBRARY_PATH", ""),
        ]
        ld_preload = "/lib/aarch64-linux-gnu/libgfortran.so.5"
    else:
        lib_paths = [
            "/home/kuehne88/cp2k-master-20260520/build-gauxc-skala/src",
            "/home/kuehne88/cp2k-jpoto/tools/toolchain/install/libxsmm-e0c4a2389afba36c453233ad7de07bd92c715bec/lib",
            "/home/kuehne88/cp2k-gauxc-skala-toolchain/gauxc-1.1-skala-cp2k-fixes/lib",
            "/home/kuehne88/cp2k-gauxc-skala-toolchain/torch-venv-aarch64/lib/python3.12/site-packages/torch/lib",
            "/home/kuehne88/cp2k-current-gauxc-gpu-20260524/build-gauxc-gpu/src",
            "/home/kuehne88/cp2k-gauxc-skala-toolchain/libtorch-2.7.1/lib",
            "/home/kuehne88/cp2k-gauxc-skala-toolchain/gauxc-skala-gpu-20260524/lib",
            "/home/kuehne88/cp2k-gauxc-skala-toolchain/torch-venv-aarch64/lib/python3.12/site-packages/torch.libs",
            env.get("LD_LIBRARY_PATH", ""),
        ]
        ld_preload = env.get("LD_PRELOAD", "")
    env.update(
        {
            "CP2K_DATA_DIR": str(data_dir),
            "LD_LIBRARY_PATH": ":".join(path for path in lib_paths if path),
            "OMP_NUM_THREADS": str(omp_threads),
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "OMP_STACKSIZE": "128M",
        }
    )
    if ld_preload:
        env["LD_PRELOAD"] = ld_preload
    elif "LD_PRELOAD" in env:
        del env["LD_PRELOAD"]
    return env


def parse_timing(text: str) -> dict[str, float | int | None]:
    rec: dict[str, float | int | None] = {
        "energy_ha": None,
        "cp2k_total_s": None,
        "ks_matrix_total_s": None,
        "ks_matrix_calls": None,
        "program_ended": 1 if "PROGRAM ENDED" in text else 0,
    }
    for line in text.splitlines():
        if "ENERGY| Total FORCE_EVAL" in line:
            try:
                rec["energy_ha"] = float(line.split()[-1])
            except ValueError:
                pass
        fields = line.split()
        if len(fields) >= 7 and fields[0] == "CP2K" and fields[1].isdigit():
            rec["cp2k_total_s"] = float(fields[-1])
        if len(fields) >= 7 and fields[0] == "qs_ks_build_kohn_sham_matrix":
            rec["ks_matrix_calls"] = int(fields[1])
            rec["ks_matrix_total_s"] = float(fields[-1])
    return rec


def run_case(
    cpu_cp2k: Path,
    gpu_cp2k: Path,
    data_dir: Path,
    work: Path,
    nwater: int,
    mode: str,
    repeat: int,
    omp_threads: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    coords, cell = make_water_cluster(nwater)
    model_path = args.gpu_model if mode == "gpu_device" else args.cpu_model
    label = f"h2o{nwater:03d}_{mode}_r{repeat}"
    run_dir = work / "runs" / label
    run_dir.mkdir(parents=True, exist_ok=True)
    inp = render_input(
        nwater,
        coords,
        cell,
        mode,
        model_path,
        cutoff=args.cutoff,
        rel_cutoff=args.rel_cutoff,
        max_scf=args.max_scf,
        eps_scf=args.eps_scf,
        grid=args.grid,
        pruning=args.pruning,
        batch_size=args.batch_size,
    )
    (run_dir / "input.inp").write_text(inp)
    cp2k = gpu_cp2k if mode == "gpu_device" else cpu_cp2k
    env = base_env(data_dir, omp_threads, mode)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    start = time.perf_counter()
    proc = subprocess.run(
        ["/usr/bin/time", "-p", str(cp2k), "-i", "input.inp", "-o", "cp2k.out"],
        cwd=run_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout,
    )
    elapsed = time.perf_counter() - start
    stdout = proc.stdout
    (run_dir / "stdout.log").write_text(stdout)
    out_text = (run_dir / "cp2k.out").read_text(errors="replace") if (run_dir / "cp2k.out").exists() else ""
    rec = parse_timing(out_text)
    time_real = None
    for line in stdout.splitlines():
        m = re.match(r"real\s+([-+0-9.Ee]+)\s*$", line)
        if m:
            time_real = float(m.group(1))
    rec.update(
        {
            "label": label,
            "nwater": nwater,
            "natoms": 3 * nwater,
            "mode": mode,
            "repeat": repeat,
            "omp_threads": omp_threads,
            "returncode": proc.returncode,
            "wall_s": time_real if time_real is not None else elapsed,
            "elapsed_s": elapsed,
            "run_dir": str(run_dir),
        }
    )
    if proc.returncode != 0:
        rec["error_tail"] = "\n".join((stdout + "\n" + out_text).splitlines()[-80:])
    return rec


def median(values: list[float]) -> float | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def ensure_model_links(cpu_model: str, gpu_model: str) -> None:
    links = [(Path(cpu_model), CPU_MODEL_SOURCE), (Path(gpu_model), GPU_MODEL_SOURCE)]
    for link, source in links:
        if link.exists() or link.is_symlink():
            continue
        link.symlink_to(source)


def write_summary(records: list[dict[str, object]], work: Path) -> None:
    raw_path = work / "results.json"
    raw_path.write_text(json.dumps(records, indent=2, sort_keys=True))
    fields = [
        "nwater",
        "natoms",
        "mode",
        "repeat",
        "omp_threads",
        "returncode",
        "program_ended",
        "wall_s",
        "cp2k_total_s",
        "ks_matrix_total_s",
        "ks_matrix_calls",
        "energy_ha",
        "run_dir",
    ]
    with (work / "summary.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow({field: rec.get(field, "") for field in fields})

    grouped: dict[tuple[int, str], list[dict[str, object]]] = {}
    for rec in records:
        if rec.get("repeat") == 0 or rec.get("returncode") != 0 or rec.get("program_ended") != 1:
            continue
        grouped.setdefault((int(rec["nwater"]), str(rec["mode"])), []).append(rec)

    rows = []
    for nwater in sorted({int(rec["nwater"]) for rec in records}):
        cpu = grouped.get((nwater, "cpu_host"), [])
        gpu = grouped.get((nwater, "gpu_device"), [])
        cpu_wall = median([float(rec["wall_s"]) for rec in cpu])
        gpu_wall = median([float(rec["wall_s"]) for rec in gpu])
        cpu_ks = median([float(rec["ks_matrix_total_s"]) for rec in cpu if rec.get("ks_matrix_total_s") is not None])
        gpu_ks = median([float(rec["ks_matrix_total_s"]) for rec in gpu if rec.get("ks_matrix_total_s") is not None])
        rows.append(
            {
                "nwater": nwater,
                "natoms": 3 * nwater,
                "cpu_wall_median_s": cpu_wall,
                "gpu_wall_median_s": gpu_wall,
                "wall_speedup": None if cpu_wall is None or gpu_wall in (None, 0.0) else cpu_wall / gpu_wall,
                "cpu_ks_matrix_median_s": cpu_ks,
                "gpu_ks_matrix_median_s": gpu_ks,
                "ks_matrix_speedup": None if cpu_ks is None or gpu_ks in (None, 0.0) else cpu_ks / gpu_ks,
                "n_repeats_cpu": len(cpu),
                "n_repeats_gpu": len(gpu),
            }
        )
    with (work / "median_summary.tsv").open("w", newline="") as handle:
        fields = [
            "nwater",
            "natoms",
            "cpu_wall_median_s",
            "gpu_wall_median_s",
            "wall_speedup",
            "cpu_ks_matrix_median_s",
            "gpu_ks_matrix_median_s",
            "ks_matrix_speedup",
            "n_repeats_cpu",
            "n_repeats_gpu",
        ]
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-cp2k", type=Path, default=DEFAULT_CPU_CP2K)
    parser.add_argument("--gpu-cp2k", type=Path, default=DEFAULT_GPU_CP2K)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--cpu-model", default=DEFAULT_CPU_MODEL)
    parser.add_argument("--gpu-model", default=DEFAULT_GPU_MODEL)
    parser.add_argument("--nwaters", default="8,16,32")
    parser.add_argument("--modes", default="cpu_host,gpu_device")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--omp-threads", type=int, default=10)
    parser.add_argument("--cutoff", type=int, default=150)
    parser.add_argument("--rel-cutoff", type=int, default=30)
    parser.add_argument("--max-scf", type=int, default=1)
    parser.add_argument("--eps-scf", default="1.0E-4")
    parser.add_argument("--grid", default="SUPERFINE")
    parser.add_argument("--pruning", default="UNPRUNED")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and args.work.exists():
        shutil.rmtree(args.work)
    args.work.mkdir(parents=True, exist_ok=True)
    ensure_model_links(args.cpu_model, args.gpu_model)

    records: list[dict[str, object]] = []
    nwaters = [int(item) for item in args.nwaters.split(",") if item]
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    for nwater in nwaters:
        for mode in modes:
            if mode not in {"cpu_host", "gpu_device"}:
                raise ValueError(f"Unsupported mode: {mode}")
            for repeat in range(args.warmups + args.repeats):
                print(f"RUN nwater={nwater} mode={mode} repeat={repeat}", flush=True)
                rec = run_case(args.cpu_cp2k, args.gpu_cp2k, args.data_dir, args.work, nwater, mode, repeat, args.omp_threads, args)
                records.append(rec)
                write_summary(records, args.work)
                print(
                    f"DONE {rec['label']} rc={rec['returncode']} wall={rec.get('wall_s')} "
                    f"ks={rec.get('ks_matrix_total_s')}",
                    flush=True,
                )
    write_summary(records, args.work)
    print(args.work / "median_summary.tsv")


if __name__ == "__main__":
    main()
