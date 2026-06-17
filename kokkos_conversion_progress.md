# Kokkos Conversion Progress

Last updated: 2026-06-17

## Status Summary

- Phase 0 (Feasibility and Baseline): Completed
- Phase 1 (Build and Baseline Harness): Completed
- Phase 2 (Memory and Orchestration Port): In progress
- Phase 3 (Coarsening Port): Not started
- Phase 4 (Uncoarsening and Refinement Port): Not started
- Phase 5 (Hardening and Cleanup): Not started

## Phase 0: Feasibility and Baseline

### Completed

- Added primitive coverage matrix:
  - `kokkos_port/phase0/primitive_coverage.csv`
- Added primitive benchmark executable:
  - `kokkos_port/src/phase0_primitives.cpp`
- Added phase 0 runner script:
  - `kokkos_port/tools/run_phase0.sh`

### Validation

Executed:

```bash
./build-kokkos/exec/gkway-kokkos-phase0 500000
```

Observed output:

- `reduce,sum=1499994,ms=0`
- `scan,last=500000,ms=1`
- `compaction_like,count=166667`

Result: PASS (phase 0 feasibility checks operational)

## Phase 1: Build and Baseline Harness

### Completed

- Added optional Kokkos port entry in root CMake:
  - `CMakeLists.txt` (`GKWAY_ENABLE_KOKKOS_PORT`)
- Added Kokkos port build and entrypoint:
  - `kokkos_port/CMakeLists.txt`
  - `kokkos_port/src/main.cpp`
- Added initial data model and pipeline interfaces:
  - `kokkos_port/include/gkway_kokkos/data_model.hpp`
  - `kokkos_port/include/gkway_kokkos/pipeline.hpp`
- Added baseline/invariant tooling:
  - `kokkos_port/tools/compare_with_baseline.py`
  - `kokkos_port/tools/check_levels_invariants.py`
  - `kokkos_port/tools/run_phase1_compare.sh`

### Validation

Executed:

```bash
./build-kokkos/exec/gkway-kokkos mesh_graph.metis 64 out_kokkos_stub
python3 kokkos_port/tools/check_levels_invariants.py --levels out_kokkos_stub.levels --num-partitions 64
```

Observed output:

- `[kokkos_port] phase-1 stub completed for 2897387 vertices`
- `PASS: rows=2897387 columns=2`

Result: PASS (phase 1 harness and invariant checks operational)

## Phase 2: Memory and Orchestration Port

### Completed So Far

- Added host graph parser/loader for METIS-like input:
  - `kokkos_port/include/gkway_kokkos/graph_loader.hpp`
  - `kokkos_port/src/graph_loader.cpp`
- Updated pipeline stub to consume parsed graph data and deep-copy into Kokkos views:
  - `kokkos_port/src/pipeline_stub.cpp`
- Wired loader source into Kokkos build:
  - `kokkos_port/CMakeLists.txt`

### Validation

Executed full configure/build/run sequence in Spack environment with required flags:

```bash
cmake -S . -B build-kokkos -DGKWAY_ENABLE_KOKKOS_PORT=ON -DCMAKE_C_FLAGS="-march=znver3" -DCMAKE_CXX_FLAGS="-march=znver3"
cmake --build build-kokkos --target gkway-kokkos gkway-kokkos-phase0 -j2
./build-kokkos/exec/gkway-kokkos-phase0 500000
./build-kokkos/exec/gkway-kokkos mesh_graph.metis 64 out_kokkos_stub
python3 kokkos_port/tools/check_levels_invariants.py --levels out_kokkos_stub.levels --num-partitions 64
```

Result: PASS (loader-integrated pipeline builds and runs)

### Remaining for Phase 2

- Replace remaining placeholder logic in orchestration with real multilevel state flow.
- Remove reliance on synthetic partition assignment in stub path.
- Add additional sanity checks for adjacency/index integrity after load and transfer.

## Planned Next Milestone

Phase 2 milestone A:

- construct level objects from loaded graph with explicit invariants
- preserve existing output contracts (`.out`, `.levels`) while reducing stub logic
- keep CI-like local validation command sequence passing