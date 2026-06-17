#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 <baseline_prefix> <candidate_prefix> <levels_file> <num_partitions> <graph_file> [strict_coarse_consistency]"
  echo "       strict_coarse_consistency: 0 (default, relaxed) or 1 (strict)"
  exit 1
fi

baseline_prefix="$1"
candidate_prefix="$2"
levels_file="$3"
num_partitions="$4"
graph_file="$5"
strict_mode="${6:-0}"

if [[ "${strict_mode}" != "0" && "${strict_mode}" != "1" ]]; then
  echo "strict_coarse_consistency must be 0 or 1"
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${script_dir}/compare_with_baseline.py" \
  --baseline-prefix "${baseline_prefix}" \
  --candidate-prefix "${candidate_prefix}"

python3 "${script_dir}/check_levels_invariants.py" \
  --levels "${levels_file}" \
  --num-partitions "${num_partitions}"

multilevel_args=(
  "${script_dir}/check_multilevel_consistency.py"
  --levels "${levels_file}"
)

strict_levels_file="${levels_file}"
if [[ "${strict_mode}" == "1" && -f "${candidate_prefix}.pre_refine.levels" ]]; then
  strict_levels_file="${candidate_prefix}.pre_refine.levels"
fi

if [[ "${strict_mode}" == "1" ]]; then
  multilevel_args+=(--require-coarse-partition-consistency)
fi

multilevel_args[2]="${strict_levels_file}"

python3 "${multilevel_args[@]}"

python3 "${script_dir}/check_coarsened_weight_conservation.py" \
  --graph "${graph_file}" \
  --levels "${levels_file}"
