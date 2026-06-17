#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 <baseline_prefix> <candidate_prefix> <levels_file> <num_partitions> <graph_file>"
  exit 1
fi

baseline_prefix="$1"
candidate_prefix="$2"
levels_file="$3"
num_partitions="$4"
graph_file="$5"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${script_dir}/compare_with_baseline.py" \
  --baseline-prefix "${baseline_prefix}" \
  --candidate-prefix "${candidate_prefix}"

python3 "${script_dir}/check_levels_invariants.py" \
  --levels "${levels_file}" \
  --num-partitions "${num_partitions}"

python3 "${script_dir}/check_multilevel_consistency.py" \
  --levels "${levels_file}"

python3 "${script_dir}/check_coarsened_weight_conservation.py" \
  --graph "${graph_file}" \
  --levels "${levels_file}"
