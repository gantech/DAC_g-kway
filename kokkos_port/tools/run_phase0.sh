#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <build_dir> [n]"
  exit 1
fi

build_dir="$1"
n="${2:-4194304}"

"${build_dir}/exec/gkway-kokkos-phase0" "$n"
