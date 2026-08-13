#!/usr/bin/env python3
"""Compare the historical and current molecular GauXC/SKALA timings."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def current_rows(path: Path) -> dict[int, dict[str, dict[str, float]]]:
    grouped: dict[int, dict[str, dict[str, float]]] = {}
    for row in read_tsv(path):
        values = {
            key: float(row[key])
            for key in ("wall_s", "ks_matrix_total_s", "energy_ha", "peak_host_rss_kib")
            if row.get(key)
        }
        grouped.setdefault(int(row["nwater"]), {})[row["profile"]] = values
    return grouped


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator


def percent_change(current: float, previous: float) -> float:
    return 100.0 * (current / previous - 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("processed/gpu_timing_current_comparison_20260813.tsv"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    historical = {
        int(row["nwater"]): row
        for row in read_tsv(root / "processed/gpu_timing_summary_20260613.tsv")
    }
    rev1 = current_rows(
        root
        / "raw/spark/skala-gauxc-current-fine-robust-150-30-rev1-master21ef8686/median_summary.tsv"
    )
    same_build_rev1 = current_rows(
        root
        / "raw/spark/skala-gauxc-current-fine-robust-150-30-rev1/median_summary.tsv"
    )
    original_models = current_rows(
        root
        / "raw/spark/skala-gauxc-current-fine-robust-150-30-original-models/median_summary.tsv"
    )

    fields = [
        "nwater",
        "natoms",
        "historical_cpu_wall_s",
        "historical_gpu_wall_s",
        "historical_wall_speedup",
        "current_rev1_cpu_wall_s",
        "current_rev1_gpu_wall_s",
        "current_rev1_wall_speedup",
        "current_rev1_cpu_wall_change_percent",
        "current_rev1_gpu_wall_change_percent",
        "current_rev1_speedup_change_percent",
        "same_build_original_cpu_wall_s",
        "same_build_original_gpu_wall_s",
        "same_build_original_wall_speedup",
        "same_build_rev1_cpu_wall_s",
        "same_build_rev1_gpu_wall_s",
        "same_build_rev1_wall_speedup",
        "same_build_rev1_vs_original_cpu_change_percent",
        "same_build_rev1_vs_original_gpu_change_percent",
        "historical_ks_speedup",
        "current_rev1_ks_speedup",
        "same_build_original_ks_speedup",
        "current_rev1_delta_energy_ha",
        "same_build_original_delta_energy_ha",
        "current_rev1_cpu_peak_host_rss_mib",
        "current_rev1_gpu_peak_host_rss_mib",
    ]

    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for nwater in sorted(historical):
            old = historical[nwater]
            new_cpu = rev1[nwater]["cpu"]
            new_gpu = rev1[nwater]["gpu1"]
            same_build_cpu = same_build_rev1[nwater]["cpu"]
            same_build_gpu = same_build_rev1[nwater]["gpu1"]
            original_cpu = original_models[nwater]["cpu"]
            original_gpu = original_models[nwater]["gpu1"]

            old_cpu_wall = float(old["cpu_wall_s"])
            old_gpu_wall = float(old["gpu_wall_s"])
            old_speedup = float(old["wall_speedup"])
            new_speedup = ratio(new_cpu["wall_s"], new_gpu["wall_s"])
            original_speedup = ratio(original_cpu["wall_s"], original_gpu["wall_s"])

            writer.writerow(
                {
                    "nwater": nwater,
                    "natoms": int(old["natoms"]),
                    "historical_cpu_wall_s": f"{old_cpu_wall:.6f}",
                    "historical_gpu_wall_s": f"{old_gpu_wall:.6f}",
                    "historical_wall_speedup": f"{old_speedup:.6f}",
                    "current_rev1_cpu_wall_s": f"{new_cpu['wall_s']:.6f}",
                    "current_rev1_gpu_wall_s": f"{new_gpu['wall_s']:.6f}",
                    "current_rev1_wall_speedup": f"{new_speedup:.6f}",
                    "current_rev1_cpu_wall_change_percent": (
                        f"{percent_change(new_cpu['wall_s'], old_cpu_wall):.3f}"
                    ),
                    "current_rev1_gpu_wall_change_percent": (
                        f"{percent_change(new_gpu['wall_s'], old_gpu_wall):.3f}"
                    ),
                    "current_rev1_speedup_change_percent": (
                        f"{percent_change(new_speedup, old_speedup):.3f}"
                    ),
                    "same_build_original_cpu_wall_s": f"{original_cpu['wall_s']:.6f}",
                    "same_build_original_gpu_wall_s": f"{original_gpu['wall_s']:.6f}",
                    "same_build_original_wall_speedup": f"{original_speedup:.6f}",
                    "same_build_rev1_cpu_wall_s": f"{same_build_cpu['wall_s']:.6f}",
                    "same_build_rev1_gpu_wall_s": f"{same_build_gpu['wall_s']:.6f}",
                    "same_build_rev1_wall_speedup": (
                        f"{ratio(same_build_cpu['wall_s'], same_build_gpu['wall_s']):.6f}"
                    ),
                    "same_build_rev1_vs_original_cpu_change_percent": (
                        f"{percent_change(same_build_cpu['wall_s'], original_cpu['wall_s']):.3f}"
                    ),
                    "same_build_rev1_vs_original_gpu_change_percent": (
                        f"{percent_change(same_build_gpu['wall_s'], original_gpu['wall_s']):.3f}"
                    ),
                    "historical_ks_speedup": f"{float(old['ks_matrix_speedup']):.6f}",
                    "current_rev1_ks_speedup": (
                        f"{ratio(new_cpu['ks_matrix_total_s'], new_gpu['ks_matrix_total_s']):.6f}"
                    ),
                    "same_build_original_ks_speedup": (
                        f"{ratio(original_cpu['ks_matrix_total_s'], original_gpu['ks_matrix_total_s']):.6f}"
                    ),
                    "current_rev1_delta_energy_ha": (
                        f"{new_gpu['energy_ha'] - new_cpu['energy_ha']:.12e}"
                    ),
                    "same_build_original_delta_energy_ha": (
                        f"{original_gpu['energy_ha'] - original_cpu['energy_ha']:.12e}"
                    ),
                    "current_rev1_cpu_peak_host_rss_mib": (
                        f"{new_cpu['peak_host_rss_kib'] / 1024.0:.3f}"
                    ),
                    "current_rev1_gpu_peak_host_rss_mib": (
                        f"{new_gpu['peak_host_rss_kib'] / 1024.0:.3f}"
                    ),
                }
            )

    print(output)


if __name__ == "__main__":
    main()
