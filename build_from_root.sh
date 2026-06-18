#!/usr/bin/env bash
set -euo pipefail

# Build helper that is safe to invoke from anywhere.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
JOBS="${JOBS:-4}"

# Keep the same flags and option currently used for the Kokkos-enabled build.
CFLAGS="-march=znver3"
CXXFLAGS="-march=znver3"

cmake \
  -S "${ROOT_DIR}" \
  -B "${BUILD_DIR}" \
  -DCMAKE_C_FLAGS="${CFLAGS}" \
  -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
  -DGKWAY_ENABLE_KOKKOS_PORT=ON

cmake --build "${BUILD_DIR}" -- -j"${JOBS}"
