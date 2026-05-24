# Molecular SKALA in CP2K companion repository

This repository contains the curated companion data for the manuscript

`Molecular SKALA in CP2K: Machine-Learned Exchange-Correlation through a GauXC Atomic-Orbital Interface`.

It contains CP2K input files, selected raw CP2K outputs, extraction scripts, processed benchmark tables, regression-test inputs, and Supplementary Information files used for the molecular GPW/GAPW + GauXC/SKALA validation reported in the manuscript.

The working copy is versioned in the private companion repository
`DCM-Uni-Paderborn/Molecular-Skala-in-CP2K`.

## Folder layout

- `cp2k-regtests/regtest-gauxc/`
  - CP2K regression-test inputs for GauXC, OneDFT/SKALA, CDFT, CDFT-CI, and RKS/UKS smoke checks.
- `raw/spark/skala-molopt-bench/`
  - Raw CP2K inputs and outputs for the small MOLOPT-UZH H2 validation set.
- `raw/spark/tau-mgga-gauxc-bench-1200-80/`
  - Raw `results.json` and `summary.tsv` from the closed-shell H2O TPSS/r2SCAN kinetic-energy-density diagnostic rerun with \(E_{\mathrm{cut}}=1200\) Ry and \(E_{\mathrm{rel}}=80\) Ry.
- `raw/spark/skala-gauxc-gpu-timing-20260524/`
  - Raw CP2K inputs, outputs, and timing summaries for the Spark CPU/GPU SKALA-through-GauXC timing diagnostic.
- `processed/`
  - Machine-readable summaries extracted from raw CP2K outputs, including the BEGDB water-hexamer binding and relative-energy tables.  The top-level water-hexamer tables are the QZVPP all-electron GAPW values reported in the main manuscript; the previous TZVPP set is retained in a `tzvpp/` subfolder, and the complementary GPW/GTH protocol check is retained in a `gth/` subfolder.
- `scripts/`
  - Lightweight extraction scripts used to regenerate the processed summaries.
- `raw/begdb-water-hexamers/`
  - BEGDB source material, selected water-hexamer structures, and raw Spark result JSON for the water-hexamer application benchmark.  The top-level `results.json` is the QZVPP all-electron GAPW production run; `tzvpp/` and `qzvpp/` subfolders retain both raw all-electron result sets, and the `gth/` subfolder retains the complementary GPW/GTH result set.
- `supplementary/`
  - Current Supplementary Information source and compiled PDF, synchronized with the Overleaf manuscript copy.
- `metadata/`
  - File manifest and checksums.

## Reproducing the summary table

From this folder, run

```bash
./scripts/extract_cp2k_summary.sh . processed/cp2k-output-summary.tsv
```

The resulting TSV contains one row per retained CP2K output file with the detected project name, total FORCE_EVAL energy, GauXC molecular-virial finite-difference diagnostic, CP2K DEBUG force-difference line, and whether the run reached `PROGRAM ENDED`.

## Water-hexamer application benchmark

The paper's water-hexamer table uses the updated BEGDB water-cluster data set. The selected structures are stored in

```text
raw/begdb-water-hexamers/structures/
```

The BEGDB reference table and downloaded source files are stored in

```text
raw/begdb-water-hexamers/source/
```

The production Spark result JSON for the all-electron GAPW table is stored in

```text
raw/begdb-water-hexamers/results.json
```

The processed all-electron GAPW tables used by the manuscript are

```text
processed/begdb-water-hexamers/begdb_hexamer_binding_energies.tsv
processed/begdb-water-hexamers/begdb_hexamer_relative_energies.tsv
processed/begdb-water-hexamers/begdb_hexamer_absolute_energies.tsv
```

The complementary GPW/GTH PBE-pseudopotential protocol-check tables used in the Supplementary Information are stored in

```text
raw/begdb-water-hexamers/gth/results.json
processed/begdb-water-hexamers/gth/begdb_hexamer_binding_energies.tsv
processed/begdb-water-hexamers/gth/begdb_hexamer_relative_energies.tsv
processed/begdb-water-hexamers/gth/begdb_hexamer_absolute_energies.tsv
```

They were generated with

```bash
python3 scripts/run_begdb_hexamer_bench.py --refresh-tables-only
```

on the Spark workstation after the all-electron GAPW calculations had completed. The main manuscript table uses the QZVPP-quality MOLOPT_UZH all-electron basis set, `POTENTIAL ALL`, and `GAPW_ACCURATE_XCINT`; the earlier TZVPP run is retained in the `tzvpp/` subfolders for traceability. The complementary GPW/GTH table used in the Supplementary Information uses the same structures with the PBE-optimized MOLOPT_UZH GTH basis/pseudopotential protocol; this protocol label describes the basis and pseudopotential choice, not the XC model in the PBE-through-GauXC, native PBE, or SKALA columns. The PBE-D3(BJ) column uses native CP2K pair-potential D3(BJ) with `REFERENCE_FUNCTIONAL PBE`. The SKALA-D3(BJ) column is an additive correction: the SKALA XC binding energy plus the D3(BJ) binding contribution obtained with CP2K's `REFERENCE_FUNCTIONAL B3LYP` parameter set. This is the CP2K-side representation of the public SKALA 1.1 B3LYP5-D3(BJ) convention; current follow-up code also accepts `REFERENCE_FUNCTIONAL SKALA` as an alias for that same D3(BJ) parameter set. Additive pair-potential dispersion is independent of the GauXC XC energy/matrix evaluation, whereas non-local density-dependent `VDW_POTENTIAL` corrections remain outside the supported GauXC contract.

## Meta-GGA kinetic-energy-density diagnostic

The manuscript Table II uses the completed Spark rerun stored in

```text
raw/spark/tau-mgga-gauxc-bench-1200-80/results.json
raw/spark/tau-mgga-gauxc-bench-1200-80/summary.tsv
```

This rerun uses closed-shell H2O, TPSS and r2SCAN, both the native CP2K and GauXC paths, and \(E_{\mathrm{cut}}=1200\) Ry with \(E_{\mathrm{rel}}=80\) Ry.  The manuscript reports the GauXC-minus-native energy differences in hartree.  The tighter grid reduces the kinetic-energy-density sensitivity seen in the earlier \(600/60\) Ry diagnostic while leaving the force and molecular-virial finite-difference checks at the \(10^{-7}\) hartree/bohr and hartree level.

## CPU/GPU SKALA timing diagnostic

The Spark timing diagnostic is stored in

```text
raw/spark/skala-gauxc-gpu-timing-20260524/
processed/gpu_timing_summary_20260524.tsv
```

It uses isolated GPW water clusters with one SCF Kohn-Sham matrix build, the SKALA 1.1 OneDFT model through GauXC, a deliberately lightweight GauXC `FINE`/`ROBUST` molecular quadrature, \(E_{\mathrm{cut}}=150\) Ry, and \(E_{\mathrm{rel}}=30\) Ry.  These runs are intended only as a performance diagnostic on the Spark workstation, not as production-quality energy benchmarks.  The one-water `SUPERFINE`/`UNPRUNED` CUDA test was at the memory limit of the NVIDIA GB10, so the reported timing protocol uses the smaller grid to obtain reproducible CPU/GPU timings.

## Checksums

SHA256 checksums are listed in

```text
metadata/sha256sums.txt
```

Regenerate them with

```bash
find . -type f ! -name .DS_Store ! -path './metadata/sha256sums.txt' \
  ! -path './.git/*' ! -path '*/__pycache__/*' ! -name '*.pyc' \
  | sed 's#^./##' | sort > metadata/file-list.txt
while IFS= read -r file; do shasum -a 256 "./$file"; done \
  < metadata/file-list.txt > metadata/sha256sums.txt
```

## Repository

GitHub repository: `DCM-Uni-Paderborn/Molecular-Skala-in-CP2K`.
