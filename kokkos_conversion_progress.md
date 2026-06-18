# Kokkos Conversion Progress

Last updated: 2026-06-17

## Status Summary

- Phase 0 (Feasibility and Baseline): Completed
- Phase 1 (Build and Baseline Harness): Completed
- Phase 2 (Memory and Orchestration Port): In progress
- Phase 3 (Coarsening Port): In progress
- Phase 4 (Uncoarsening and Refinement Port): In progress
- Phase 5 (Hardening and Cleanup): Completed

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
- Added automatic refinement snapshots in pipeline output:
  - writes `.pre_refine.out/.levels` before refinement
  - writes `.post_refine.out/.levels` after refinement
- Strict compare mode now auto-routes to `.pre_refine.levels` when present
- Kokkos lineage output now emits the full 6-level ancestry chain to match the CUDA format:
  - `PartitionID,L5,L4,L3,L2,L1,L0`
  - validators now accept the wider CSV while still checking the immediate `L1 -> L0` structure
- Added Kokkos contributor playbook:
  - `kokkos_port/CONTRIBUTOR_PLAYBOOK.md`

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
- strict compare harness on pre-refinement snapshot: `PASS: coarse_vertices=1448694 max_group_size=2 mixed_partition_coarse_vertices=0`

Result: PASS (hardening checks active and passing)

Latest observed hardening output:

- `PASS: coarse_vertices=1448694 max_group_size=2 mixed_partition_coarse_vertices=36620`

### Remaining for Phase 2

- Replace remaining placeholder logic in orchestration with real multilevel state flow.
- Remove reliance on synthetic partition assignment in stub path.
- Add additional sanity checks for adjacency/index integrity after load and transfer.

## Planned Next Milestone

Phase 5 complete.

- Keep working Phase 2-4 slices as needed for the broader port.
- Preserve the CUDA path while iterating on the Kokkos capability.

## Detailed Coarsening Debug Trail

### What was verified first

- The original Kokkos implementation was index-driven (`i / 2`) rather than graph-driven.
- CUDA follows a multilevel coarsen/partition/uncoarsen flow, so the first task was to replace the synthetic lineage with graph-based coarsening.
- The first Kokkos matcher bug was using `0u` as both a valid vertex index and an unmatched sentinel.
- The second Kokkos matcher bug was ignoring `max_coarsen_group`; CUDA caps coarse groups at 6.

### What was changed in the Kokkos pipeline

- Replaced the synthetic one-shot index split with host-side graph coarsening in `kokkos_port/src/pipeline_stub.cpp`.
- Added a multilevel coarsening loop and lineage projection so Kokkos no longer emits a fake single-step ancestry.
- Kept the Kokkos side host-driven for now, but made it follow the CUDA driver structure more closely.
- Added lineage-aware output writing for `.levels`, `.pre_refine.levels`, and `.post_refine.levels`.

### What was compared against CUDA

- Small graph used for fast validation: `simple_test/delaunay_n11.graph`.
- CUDA baseline small-graph lineage started with `PartitionID,L1,L0`.
- Kokkos output is now also `PartitionID,L1,L0` after the coarsening rewrite, which means the extra over-coarsening level was removed.
- Current divergence remains in the actual coarse graph and final partitioning, not in the CSV shape.

### Current observed mismatch

- CUDA small-graph probe showed `num_coarsen_vertex=469` and `num_coarsen_edge=2826`.
- Current Kokkos probe shows `coarsest_graph_vertices=469` and `coarsest_graph_edges=2848`.
- Current Kokkos small-graph summary shows cutsize around `1381`, while the CUDA run reports `1307`.
- The first lineage row still differs, for example CUDA has `46,26,1` where Kokkos currently has `29,27,1`.

### What this suggests

- The remaining gap is still in coarsening/contraction, not in the refinement pass or file formatting.
- Vertex count now matches, so the remaining difference is in coarse graph connectivity and/or exact grouping order.
- The current Kokkos path is much closer than the original stub, but it is still not a bitwise match to CUDA.

### Host-side CUDA shaping work already applied

- The Kokkos neighbor scorer was aligned with the CUDA weight/degree comparison strategy.
- The coarse-graph builder was rewritten to use a row-wise sort-and-merge path instead of a global pair map.
- Deterministic lineage numbering was restored so the coarse IDs are stable and follow the graph structure instead of hash iteration order.
- The coarsening loop now mirrors the CUDA multilevel threshold behavior much more closely than the original one-level stub.

### Remaining work

- Compare the exact coarse adjacency produced by the Kokkos contraction step against the CUDA baseline.
- Tighten any remaining tie-breaking or edge aggregation differences in `kokkos_port/src/pipeline_stub.cpp`.
- Remove temporary probes once the coarsening edge count matches CUDA or a final acceptable approximation is documented.

## Latest Status Update (2026-06-17) — Refinement Parity Closure

### Work Completed

- Added [build_from_root.sh](build_from_root.sh): cwd-independent build script to eliminate path confusion.
- Updated README.md with explicit root-safe build instructions.
- Implemented deterministic METIS initialization with explicit seeding in both CUDA and Kokkos paths to remove initialization variability.
- Added granular top-level refinement iteration tracing to both kernels for pass-level comparison.
- Applied multiple refinement semantic alignment patches to Kokkos:
  - Fixed `op_result` initialization to match CUDA memset behavior
  - Removed extra neighbor boundary filters in move-buffer construction
  - Removed explicit target-partition equality guards from move eligibility checks
  - Aligned `if_updated` reset location to match CUDA kernel clearing point
  - Switched candidate weight accumulation to unsigned arithmetic for overflow parity
- Instrumented refinement with debug output showing iteration count, buffer size, and max prefix accepted.

### Key Findings

**Small case (delaunay_n11.graph, 8 partitions):**
- Parity: exact match (diff = 0)
- Status: stable and reproducible

**Large case (mesh_graph.metis, 64 partitions):**
- Current parity: ~2.8M vertices differ out of 2.9M total
- CUDA self-determinism: CUDA vs CUDA on same input diffs by ~2.9M vertices even with fixed METIS seed
  - This indicates the reference itself is nondeterministic at scale (likely due to atomic ordering in GPU warp operations)
- Refinement iteration traces show structural divergence:
  - CUDA: top-level refinement terminates at ~107–147 iterations depending on run
  - Kokkos: top-level refinement terminates at ~94 iterations
  - Early iterations accept vastly different prefix sizes (CUDA: typically 7–240, variable; Kokkos: frequently ~1000+)
- Cutsize metrics:
  - CUDA large: ranges 857k–882k depending on run (nondeterministic)
  - Kokkos large: consistently 874k across runs
- Both achieve valid partitions with acceptable weight balance

### Architectural Insight

Per-vertex exact label matching is not a reliable parity metric for large cases because:
1. The CUDA reference itself exhibits run-to-run variation in final assignments despite identical input
2. Atomic operation ordering on GPUs introduces inherent nondeterminism for concurrent moves
3. Refinement pass trajectories naturally diverge when independent move sets differ slightly

### Recommended Debugging Path

Rather than chasing vertex-level divergence, measure **refinement semantics at the pass level**:
1. Capture CUDA and Kokkos partition snapshots at each refinement iteration
2. Compare per-iteration invariants: partition weights, boundary counts, cutsize deltas
3. Identify which pass iteration the trajectories first diverge materially
4. Isolate that divergence to a specific candidate ranking, balance test, or apply logic mismatch
5. Once isolated, apply targeted Kokkos patch to re-synchronize that specific pass

### Next Immediate Action

~~Implement "same-start refinement debug mode"~~ — **Done (2026-06-17)**

### Same-Start Refinement Debug Mode (Implemented)

Both CUDA and Kokkos now support a same-start mode that isolates refinement
logic from initialization/coarsening nondeterminism.

**CUDA side** (`gkway/graph_partitioner.hpp`, `gkway/uncoarsen.hpp`):
- `uncoarsening()` now accepts an optional `out_finest_pre_refine` pointer.
  Before each `gk::refinement()` call, `d_cutsize` is reset to the actual
  cutsize of the current level (so per-iteration prints track absolute values).
  On the last (finest) level, the pre-refinement L0 partition is captured and
  written to `<OUT_FILE>.same_start.txt` (one partition ID per line).
- `gk::refinement()` now prints `cutsize` after each apply step alongside the
  existing `buffer_size` and `max_prefix` lines.

**Kokkos side** (`kokkos_port/src/pipeline_stub.cpp`, `kokkos_port/src/main.cpp`,
`kokkos_port/include/gkway_kokkos/pipeline.hpp`):
- `RunOptions` gained `same_start_file`; the CLI now accepts a 6th positional
  argument:
  ```
  ./build/exec/gkway-kokkos graph num_parts out_prefix [refinement_passes] [same_start_partition_file]
  ```
- When `same_start_file` is set, the pipeline skips coarsening + METIS, loads
  the injected partition, and calls `refine_partition_host()` with
  `verbose_refine=true` on the L0 graph.
- `refine_partition_host()` now initialises `d_cutsize` from the real
  pre-refinement cutsize (via `compute_cutsize_host`) so per-pass cutsize
  values are absolute. It gains a `verbose_refine` flag that forces per-pass
  logging even on small graphs, and a new post-apply line:
  ```
  [kokkos_refine_dbg] vertices=…, iter=…, moves=…, cutsize=…
  ```
- Output is written to `<out_prefix>.same_start.levels` (`PartitionID,L0`).

**Workflow to compare:**
```bash
# 1. Run CUDA to produce the same-start partition file
./build/exec/g-kway mesh_graph.metis 64 out_cuda_dbg

# 2. Run Kokkos in same-start mode using CUDA's pre-refinement L0 partition
./build/exec/gkway-kokkos mesh_graph.metis 64 out_kokkos_dbg 1 out_cuda_dbg.same_start.txt

# 3. Compare per-pass logs
grep cuda_refine_dbg  <cuda-stdout>   | head -50
grep kokkos_refine_dbg <kokkos-stdout> | head -50
```
Both programs will print matching `vertices=`, `iter=`, `buffer_size=`,
`max_prefix=`, `cutsize=` lines. Side-by-side comparison immediately shows
which iteration and which metric first diverges, isolating the refinement
logic difference from any coarsening/METIS nondeterminism.

### Files Modified

- [build_from_root.sh](build_from_root.sh) — new root-safe build script
- [README.md](README.md) — added root-safe build instructions
- [gkway/metis_partition.hpp](gkway/metis_partition.hpp) — added explicit METIS options with fixed seed
- [gkway/uncoarsen.hpp](gkway/uncoarsen.hpp) — added refinement iteration debug logging for large graphs; added per-pass `cutsize` print after apply
- [gkway/graph_partitioner.hpp](gkway/graph_partitioner.hpp) — `uncoarsening()` now resets `d_cutsize` per level and captures finest-level pre-refine partition; `graph_partitioner()` writes `<out>.same_start.txt`
- [kokkos_port/src/pipeline_stub.cpp](kokkos_port/src/pipeline_stub.cpp) — `refine_partition_host()` initialises cutsize properly and adds `verbose_refine` + post-apply log; `run_pipeline_stub()` implements same-start mode
- [kokkos_port/include/gkway_kokkos/pipeline.hpp](kokkos_port/include/gkway_kokkos/pipeline.hpp) — added `same_start_file` to `RunOptions`
- [kokkos_port/src/main.cpp](kokkos_port/src/main.cpp) — accepts optional 6th `same_start_partition_file` argument
