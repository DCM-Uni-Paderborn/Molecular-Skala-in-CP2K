#!/usr/bin/env python3
"""Current-stack CPU/GPU benchmark for molecular SKALA through GauXC."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import signal
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path


WATER = [
    ("O", 0.000000, 0.000000, 0.000000),
    ("H", 0.000000, 0.756950, 0.585880),
    ("H", 0.000000, -0.756950, 0.585880),
]


@dataclass(frozen=True)
class Profile:
    label: str
    execution_space: str
    mpi_ranks: int
    omp_threads: int
    devices: tuple[int, ...]


def parse_profiles(value: str) -> list[Profile]:
    profiles = []
    for item in value.split(";"):
        if not item.strip():
            continue
        fields = item.split(":")
        if len(fields) != 5:
            raise ValueError(
                "Profiles use label:HOST|DEVICE:mpi_ranks:omp_threads:devices; "
                "use '-' for no device."
            )
        label, execution_space, ranks, threads, devices = fields
        execution_space = execution_space.upper()
        if execution_space not in {"HOST", "DEVICE"}:
            raise ValueError(f"Unsupported execution space: {execution_space}")
        device_ids = () if devices == "-" else tuple(int(x) for x in devices.split("+"))
        profiles.append(Profile(label, execution_space, int(ranks), int(threads), device_ids))
    return profiles


def make_water_cluster(nwater: int, spacing: float = 3.2):
    side = math.ceil(nwater ** (1.0 / 3.0))
    coords = []
    count = 0
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                if count >= nwater:
                    break
                shift = (ix * spacing, iy * spacing, iz * spacing)
                sign = -1.0 if (ix + iy + iz) % 2 else 1.0
                for element, x, y, z in WATER:
                    coords.append((element, shift[0] + x, shift[1] + sign * y, shift[2] + z))
                count += 1
            if count >= nwater:
                break
        if count >= nwater:
            break

    minima = [min(coord[i] for _, *coord in coords) for i in range(3)]
    maxima = [max(coord[i] for _, *coord in coords) for i in range(3)]
    padding = 6.0
    shifted = [
        (element, x - minima[0] + padding / 2, y - minima[1] + padding / 2, z - minima[2] + padding / 2)
        for element, x, y, z in coords
    ]
    cell = max(maxima[i] - minima[i] for i in range(3)) + padding
    return shifted, cell


def render_input(nwater: int, profile: Profile, model: Path, args: argparse.Namespace) -> str:
    coords, cell = make_water_cluster(nwater)
    coord_lines = "\n".join(
        f"      {element:2s} {x:14.8f} {y:14.8f} {z:14.8f}" for element, x, y, z in coords
    )
    return f"""&GLOBAL
  PRINT_LEVEL LOW
  PROJECT skala_h2o_{nwater}_{profile.label}
  RUN_TYPE ENERGY
&END GLOBAL

&FORCE_EVAL
  METHOD Quickstep
  &DFT
    BASIS_SET_FILE_NAME BASIS_MOLOPT_UZH
    POTENTIAL_FILE_NAME POTENTIAL_UZH
    UKS FALSE
    &MGRID
      CUTOFF {args.cutoff}
      NGRIDS {args.ngrids}
      REL_CUTOFF {args.rel_cutoff}
    &END MGRID
    &POISSON
      PERIODIC NONE
      POISSON_SOLVER MT
    &END POISSON
    &QS
      EPS_DEFAULT {args.eps_default}
      EXTRAPOLATION USE_PREV_P
      METHOD GPW
    &END QS
    &SCF
      EPS_SCF {args.eps_scf}
      MAX_SCF {args.max_scf}
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
          MODEL {model}
          GRID {args.grid}
          PRUNING_SCHEME {args.pruning}
          BATCH_SIZE {args.batch_size}
          LB_EXECUTION_SPACE {profile.execution_space}
          INT_EXECUTION_SPACE {profile.execution_space}
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture(command: list[str], env: dict[str, str] | None = None) -> str:
    try:
        proc = subprocess.run(
            command,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable: {error}"
    return proc.stdout.strip()


def process_tree_rss_kib(root_pid: int) -> int | None:
    try:
        output = subprocess.check_output(
            ["ps", "-e", "-o", "pid=", "-o", "ppid=", "-o", "rss="], text=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    rows = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 3:
            rows.append(tuple(int(x) for x in fields))
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, ppid, _ in rows:
            if ppid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    values = [rss for pid, _, rss in rows if pid in descendants]
    return sum(values) if values else None


def gpu_memory_mib(devices: tuple[int, ...]) -> int | None:
    if not devices:
        return None
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    selected = set(devices)
    values = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            index = int(fields[0])
            used = int(fields[1])
        except ValueError:
            continue
        if index in selected:
            values.append(used)
    return sum(values) if values else None


def monitor_resources(
    process: subprocess.Popen[str], devices: tuple[int, ...], stop: threading.Event, samples: dict[str, int | None]
) -> None:
    baseline_gpu = gpu_memory_mib(devices)
    peak_rss = None
    peak_gpu = baseline_gpu
    while not stop.wait(0.5):
        rss = process_tree_rss_kib(process.pid)
        gpu = gpu_memory_mib(devices)
        if rss is not None:
            peak_rss = rss if peak_rss is None else max(peak_rss, rss)
        if gpu is not None:
            peak_gpu = gpu if peak_gpu is None else max(peak_gpu, gpu)
        if process.poll() is not None:
            break
    samples["peak_host_rss_kib"] = peak_rss
    samples["gpu_baseline_mib"] = baseline_gpu
    samples["peak_gpu_memory_mib"] = peak_gpu
    samples["peak_gpu_delta_mib"] = (
        None if baseline_gpu is None or peak_gpu is None else max(0, peak_gpu - baseline_gpu)
    )


def parse_output(text: str) -> dict[str, float | int | None]:
    record: dict[str, float | int | None] = {
        "energy_ha": None,
        "cp2k_total_s": None,
        "ks_matrix_total_s": None,
        "ks_matrix_calls": None,
        "program_ended": int("PROGRAM ENDED" in text),
    }
    for line in text.splitlines():
        if "ENERGY| Total FORCE_EVAL" in line:
            try:
                record["energy_ha"] = float(line.split()[-1])
            except ValueError:
                pass
        fields = line.split()
        if len(fields) >= 7 and fields[0] == "CP2K" and fields[1].isdigit():
            record["cp2k_total_s"] = float(fields[-1])
        if len(fields) >= 7 and fields[0] == "qs_ks_build_kohn_sham_matrix":
            record["ks_matrix_calls"] = int(fields[1])
            record["ks_matrix_total_s"] = float(fields[-1])
    return record


def run_case(
    profile: Profile,
    nwater: int,
    repeat: int,
    cp2k: Path,
    data_dir: Path,
    work: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    model = args.gpu_model if profile.execution_space == "DEVICE" else args.cpu_model
    label = f"h2o{nwater:03d}_{profile.label}_r{repeat}"
    run_dir = work / "runs" / label
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "input.inp").write_text(render_input(nwater, profile, model, args))

    env = os.environ.copy()
    env.update(
        {
            "CP2K_DATA_DIR": str(data_dir),
            "OMP_NUM_THREADS": str(profile.omp_threads),
            "OMP_STACKSIZE": args.omp_stacksize,
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    if args.library_path:
        env["LD_LIBRARY_PATH"] = args.library_path + ":" + env.get("LD_LIBRARY_PATH", "")
    if args.preload:
        env["LD_PRELOAD"] = args.preload
    if profile.devices:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in profile.devices)
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)

    command = [str(cp2k), "-i", "input.inp", "-o", "cp2k.out"]
    if profile.mpi_ranks > 1:
        command = [str(args.mpiexec), "-np", str(profile.mpi_ranks)] + command

    start = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=run_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stop = threading.Event()
    samples: dict[str, int | None] = {}
    monitor = threading.Thread(target=monitor_resources, args=(process, profile.devices, stop, samples))
    monitor.start()
    try:
        stdout, _ = process.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, _ = process.communicate()
    finally:
        stop.set()
        monitor.join()
    elapsed = time.perf_counter() - start
    (run_dir / "stdout.log").write_text(stdout)
    output = (run_dir / "cp2k.out").read_text(errors="replace") if (run_dir / "cp2k.out").exists() else ""
    record = parse_output(output)
    record.update(samples)
    record.update(
        {
            "label": label,
            "nwater": nwater,
            "natoms": 3 * nwater,
            "profile": profile.label,
            "execution_space": profile.execution_space,
            "mpi_ranks": profile.mpi_ranks,
            "omp_threads": profile.omp_threads,
            "devices": "+".join(str(x) for x in profile.devices),
            "repeat": repeat,
            "returncode": process.returncode,
            "wall_s": elapsed,
            "run_dir": str(run_dir),
        }
    )
    if process.returncode != 0 or record["program_ended"] != 1:
        record["error_tail"] = "\n".join((stdout + "\n" + output).splitlines()[-100:])
    return record


def median(values: list[float]) -> float | None:
    clean = sorted(values)
    if not clean:
        return None
    middle = len(clean) // 2
    if len(clean) % 2:
        return clean[middle]
    return 0.5 * (clean[middle - 1] + clean[middle])


def write_results(records: list[dict[str, object]], work: Path) -> None:
    (work / "results.json").write_text(json.dumps(records, indent=2, sort_keys=True))
    fields = [
        "nwater",
        "natoms",
        "profile",
        "execution_space",
        "mpi_ranks",
        "omp_threads",
        "devices",
        "repeat",
        "returncode",
        "program_ended",
        "wall_s",
        "cp2k_total_s",
        "ks_matrix_total_s",
        "ks_matrix_calls",
        "energy_ha",
        "peak_host_rss_kib",
        "gpu_baseline_mib",
        "peak_gpu_memory_mib",
        "peak_gpu_delta_mib",
        "run_dir",
    ]
    with (work / "summary.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})

    grouped: dict[tuple[int, str], list[dict[str, object]]] = {}
    for record in records:
        if record["repeat"] == 0 or record["returncode"] != 0 or record["program_ended"] != 1:
            continue
        grouped.setdefault((int(record["nwater"]), str(record["profile"])), []).append(record)
    rows = []
    for (nwater, profile), group in sorted(grouped.items()):
        row: dict[str, object] = {"nwater": nwater, "natoms": 3 * nwater, "profile": profile, "samples": len(group)}
        for field in (
            "wall_s",
            "cp2k_total_s",
            "ks_matrix_total_s",
            "energy_ha",
            "peak_host_rss_kib",
            "peak_gpu_delta_mib",
        ):
            values = [float(record[field]) for record in group if record.get(field) is not None]
            row[field] = median(values)
        row["ks_matrix_calls"] = group[0].get("ks_matrix_calls")
        rows.append(row)
    median_fields = [
        "nwater",
        "natoms",
        "profile",
        "samples",
        "wall_s",
        "cp2k_total_s",
        "ks_matrix_total_s",
        "ks_matrix_calls",
        "energy_ha",
        "peak_gpu_delta_mib",
        "peak_host_rss_kib",
    ]
    with (work / "median_summary.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, delimiter="\t", fieldnames=median_fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp2k", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cpu-model", type=Path, required=True)
    parser.add_argument("--gpu-model", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--profiles", default="cpu:HOST:1:10:-;gpu1:DEVICE:1:10:0")
    parser.add_argument("--nwaters", default="1,2,4,8")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--cutoff", type=int, default=600)
    parser.add_argument("--ngrids", type=int, default=5)
    parser.add_argument("--rel-cutoff", type=int, default=60)
    parser.add_argument("--max-scf", type=int, default=6)
    parser.add_argument("--eps-scf", default="1.0E-20")
    parser.add_argument("--eps-default", default="1.0E-12")
    parser.add_argument("--omp-stacksize", default="256M")
    parser.add_argument("--grid", default="SUPERFINE")
    parser.add_argument("--pruning", default="UNPRUNED")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--mpiexec", type=Path, default=Path("mpiexec"))
    parser.add_argument("--library-path", default="")
    parser.add_argument("--preload", default="")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    profiles = parse_profiles(args.profiles)
    nwaters = [int(value) for value in args.nwaters.split(",") if value]
    if args.clean and args.work.exists():
        shutil.rmtree(args.work)
    args.work.mkdir(parents=True, exist_ok=True)

    version_env = os.environ.copy()
    if args.library_path:
        version_env["LD_LIBRARY_PATH"] = args.library_path + ":" + version_env.get("LD_LIBRARY_PATH", "")
    if args.preload:
        version_env["LD_PRELOAD"] = args.preload

    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cp2k": str(args.cp2k),
        "cp2k_version": capture([str(args.cp2k), "--version"], env=version_env),
        "data_dir": str(args.data_dir),
        "cpu_model": str(args.cpu_model),
        "cpu_model_sha256": sha256(args.cpu_model),
        "gpu_model": str(args.gpu_model),
        "gpu_model_sha256": sha256(args.gpu_model),
        "nvidia_smi": capture(["nvidia-smi", "-L"]),
        "profiles": [profile.__dict__ | {"devices": list(profile.devices)} for profile in profiles],
        "settings": {
            "cutoff": args.cutoff,
            "ngrids": args.ngrids,
            "rel_cutoff": args.rel_cutoff,
            "max_scf": args.max_scf,
            "eps_scf": args.eps_scf,
            "eps_default": args.eps_default,
            "omp_stacksize": args.omp_stacksize,
            "grid": args.grid,
            "pruning": args.pruning,
            "batch_size": args.batch_size,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "timeout": args.timeout,
        },
    }
    (args.work / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    records = []
    for nwater in nwaters:
        for profile in profiles:
            for repeat in range(args.warmups + args.repeats):
                print(f"RUN nwater={nwater} profile={profile.label} repeat={repeat}", flush=True)
                record = run_case(profile, nwater, repeat, args.cp2k, args.data_dir, args.work, args)
                records.append(record)
                write_results(records, args.work)
                print(
                    f"DONE {record['label']} rc={record['returncode']} wall={record['wall_s']:.3f} "
                    f"ks={record.get('ks_matrix_total_s')} rss={record.get('peak_host_rss_kib')} "
                    f"gpu={record.get('peak_gpu_delta_mib')}",
                    flush=True,
                )
    write_results(records, args.work)


if __name__ == "__main__":
    main()
