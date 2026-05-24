#!/usr/bin/env bash
set -euo pipefail

root="${1:-.}"
out="${2:-processed/cp2k-output-summary.tsv}"

mkdir -p "$(dirname "$out")"
printf "path\tproject\tforce_eval_energy_ha\tgauxc_molecular_virial_fd\tdebug_sum_of_differences\tprogram_ended\n" > "$out"

find "$root/raw" -type f -name "*.out" | sort | while IFS= read -r file; do
  rel="${file#$root/}"
  project="$(
    awk '/GLOBAL\\| Project name/ {print $NF; exit}' "$file" || true
  )"
  energy="$(
    awk '/ENERGY\\| Total FORCE_EVAL/ {print $NF; exit}' "$file" || true
  )"
  virial="$(
    awk '/GAUXC\\| Molecular XC virial FD/ {
      for (i=1; i<=NF; i++) {
        if ($i ~ /^[+-]?[0-9.]+[Ee][+-]?[0-9]+$/) vals=vals (vals=="" ? "" : " ") $i
      }
      print vals
      exit
    }' "$file" || true
  )"
  diff="$(
    awk '/DEBUG\\| Sum of differences/ {print $(NF-1) " " $NF; exit}' "$file" || true
  )"
  ended="$(
    awk '/PROGRAM ENDED/ {print "yes"; found=1; exit} END {if (!found) print "no"}' "$file" || true
  )"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$rel" "$project" "$energy" "$virial" "$diff" "$ended" >> "$out"
done
