# Kokkos Conversion Progress

Last updated: 2026-06-17

## Status Summary

- Phase 0 (Feasibility and Baseline): Completed
- Phase 1 (Build and Baseline Harness): Completed
- Phase 2 (Memory and Orchestration Port): In progress
- Phase 3 (Coarsening Port): In progress
- Phase 4 (Uncoarsening and Refinement Port): In progress
- Phase 5 (Hardening and Cleanup): In progress

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
- Added graph-derived partition accounting in Kokkos:
  - partition weights via atomic accumulation from `vwgt`
  - boundary flags via adjacency partition checks
  - cutsize via edge-cut reduction (with undirected divide-by-two)
  - implemented in `kokkos_port/src/pipeline_stub.cpp`
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

Latest observed pipeline output:

- `[kokkos_port] phase-1 stub completed for 2897387 vertices, cutsize=20070839, max_partition_wgt=45272`

## Phase 3: Coarsening Port

### Completed So Far

- Implemented deterministic one-level pairwise coarsening map (`fine -> coarse`) in Kokkos:
  - `L1` IDs now generated from `L0` pair groups
  - `cmap` semantics remain 1-based for consistency with existing code conventions
  - implemented in `kokkos_port/src/pipeline_stub.cpp`
- Added coarse-level vertex-weight aggregation to support coarse partitioning decisions.

## Phase 4: Uncoarsening and Refinement Port

### Completed So Far

- Implemented host-side coarse partition initialization using coarse vertex weights with deterministic lightest-bucket assignment.
- Implemented uncoarsening propagation from coarse partitions back to fine partitions via `cmap`.
- Extended lineage output from `PartitionID,L0` to `PartitionID,L1,L0`.
- Replaced host coarse initializer with Kokkos-native weighted-prefix partition assignment.
  - coarse partition IDs now computed in Kokkos from prefix weights and total coarse weight
  - implemented in `kokkos_port/src/pipeline_stub.cpp`
- Added first refinement-state data path:
  - introduced `RefinementState` (gain, target partition, move flag)
  - computes per-vertex gain and move candidates from current adjacency partition context
  - implemented in `kokkos_port/include/gkway_kokkos/data_model.hpp` and `kokkos_port/src/pipeline_stub.cpp`
- Added first refinement move-application pass:
  - applies positive-gain move candidates once
  - recomputes partition weights post-move
  - reports `proposed_moves` diagnostic in pipeline output
- Extended refinement to configurable multi-pass iteration:
  - CLI now accepts optional `refinement_passes` argument
  - pipeline runs repeated candidate/move rounds with early stop when no moves are proposed
  - reports `refinement_passes` in pipeline diagnostics
- Added deterministic move-conflict mitigation and balance guardrails:
  - proposal acceptance is parity-gated per pass for deterministic concurrent move filtering
  - move proposals are rejected when target partition exceeds a computed/ configured weight cap
  - partition weights are recomputed each pass before candidate evaluation
- Added per-pass and balance diagnostics to pipeline output:
  - `executed_refinement_passes`, `last_pass_moves`
  - `min_partition_wgt`, `avg_partition_wgt`, `partition_wgt_cap`

### Validation

Executed:

```bash
cmake --build build-kokkos --target gkway-kokkos -j2
./build-kokkos/exec/gkway-kokkos mesh_graph.metis 64 out_kokkos_stub
python3 kokkos_port/tools/check_levels_invariants.py --levels out_kokkos_stub.levels --num-partitions 64
```

Observed output:

- `[kokkos_port] phase-1 stub completed for 2897387 vertices, cutsize=1197861, max_partition_wgt=45599, proposed_moves=56758, refinement_passes=3, executed_refinement_passes=3, last_pass_moves=13783, min_partition_wgt=44955, avg_partition_wgt=45271, partition_wgt_cap=47536`
- `PASS: rows=2897387 columns=3`

Result: PASS (one-level coarsen/uncoarsen path is functional)

## Phase 5: Hardening and Cleanup

### Completed So Far

- Updated migration direction to keep both CUDA and Kokkos pathways in the repo.
  - reflected in `convert_kokkos.md` Phase 5 tasks
- Added explicit multilevel consistency checker:
  - `kokkos_port/tools/check_multilevel_consistency.py`
  - supports optional strict `L1 -> PartitionID` consistency
  - validates pairwise coarse grouping invariants (`max_group_size <= 2`, consecutive pairs)
- Wired consistency checker into phase compare workflow:
  - `kokkos_port/tools/run_phase1_compare.sh`
- Added coarsened-weight conservation checker:
  - `kokkos_port/tools/check_coarsened_weight_conservation.py`
  - validates `L0 -> L1` pairwise mapping, full L0 coverage, and total fine/coarse weight conservation
- Wired coarsened-weight conservation checker into phase compare workflow:
  - `kokkos_port/tools/run_phase1_compare.sh` now requires `<graph_file>` input
- Extended compare harness with optional strict coarse-consistency mode:
  - relaxed mode remains default for post-refinement snapshots
  - strict mode can be enabled with a trailing `1` argument for pre-refinement snapshots

### Validation

Executed:

```bash
./build-kokkos/exec/gkway-kokkos mesh_graph.metis 64 out_kokkos_stub
python3 kokkos_port/tools/check_levels_invariants.py --levels out_kokkos_stub.levels --num-partitions 64
python3 kokkos_port/tools/check_multilevel_consistency.py --levels out_kokkos_stub.levels
```

Observed output:

- `PASS: rows=2897387 columns=3`
- `PASS: coarse_vertices=1448694 max_group_size=2 mixed_partition_coarse_vertices=36620`
- `PASS: fine_vertices=2897387 coarse_vertices=1448694 fine_total_weight=4108790870859 coarse_total_weight=4108790870859`
- strict compare harness smoke test on current post-refinement snapshot: `FAIL: coarse partition consistency violated for 34179 coarse vertices`

Result: PASS (hardening checks active and passing)

Latest observed hardening output:

- `PASS: coarse_vertices=1448694 max_group_size=2 mixed_partition_coarse_vertices=36620`

### Remaining for Phase 2

- Replace remaining placeholder logic in orchestration with real multilevel state flow.
- Remove reliance on synthetic partition assignment in stub path.
- Add additional sanity checks for adjacency/index integrity after load and transfer.

## Planned Next Milestone

Phase 4/5 milestone H:

- add optional snapshot export for pre- and post-refinement partition states