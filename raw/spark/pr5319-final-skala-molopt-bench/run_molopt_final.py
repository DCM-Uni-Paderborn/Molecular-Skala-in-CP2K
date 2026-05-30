#!/usr/bin/env python3
"""Run the small molecular SKALA validation matrix against a selected CP2K tree."""

import os
import sys
from pathlib import Path

import run_bench as rb


def main() -> None:
    cp2k_root = Path(os.environ["FINAL_CP2K_ROOT"])
    build_name = os.environ.get("FINAL_CP2K_BUILD", "build-gauxc-periodic")
    work_root = Path(os.environ["FINAL_WORK_ROOT"])
    torch_lib = os.environ.get("FINAL_TORCH_LIB", "")
    skala_model = os.environ.get("FINAL_SKALA_MODEL", "")

    rb.ROOT = cp2k_root
    rb.BUILD = cp2k_root / build_name
    rb.CP2K = rb.BUILD / "bin" / "cp2k.psmp"
    rb.WORK = work_root / "skala-molopt-bench" / "runs"
    rb.RESULTS = work_root / "skala-molopt-bench" / "results.json"

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

    rb.main()


if __name__ == "__main__":
    sys.exit(main())
