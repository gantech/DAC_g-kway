# Kokkos Conversion Plan (Revised)

## Goal

Build a Kokkos-based multilevel partitioner that preserves the current
coarsen -> partition -> uncoarsen -> refine behavior while removing direct CUDA dependencies from algorithm code.

This is a capability-development effort, not a production migration.

## Scope

- In scope: CUDA and Moderngpu code paths in the multilevel pipeline.
- Out of scope: METIS-integrated code paths.
- Baseline model: separate CUDA repo snapshot used for cross-repo comparison.

## Design Targets

1. Keep output semantics aligned with current artifacts (`out.out`, `out.levels`, cut size, partition balance).
2. Use Kokkos as the primary memory and execution abstraction.
3. Make backend support explicit (do not assume all backends from day one).
4. Freeze major architecture decisions early (backend scope and data layout).
5. Keep migration incremental with measurable gates at each phase.

## Hard Decisions to Freeze Early

## Backend Policy

Choose supported backends before coding deep kernels.

Recommended initial target:

- Primary: CUDA backend
- Secondary: OpenMP backend
- Deferred: HIP backend

Define pass criteria per backend:

- correctness parity bands
- performance floor relative to baseline

## Data Layout Policy

Choose one model after a small spike:

1. Packed multilevel arrays with offsets (closer to current code).
2. Per-level objects (`GraphLevel`) with isolated views.

Recommendation:

- Prefer per-level objects unless packed layout is proven necessary for memory/performance.

## Phase 0: Feasibility and Baseline (Mandatory)

Before broad refactoring, prove the hardest pieces are feasible.

Tasks:

1. Lock benchmark datasets and expected artifact set.
2. Build primitive coverage matrix for current Moderngpu usage:
	- scan
	- sort/mergesort
	- segmented operations
	- reductions/compaction patterns
3. Implement microbenchmarks for required primitives in Kokkos/KokkosKernels.
4. Freeze backend policy and data-layout policy.

Exit criteria:

- Primitive replacement path selected for every required operation.
- Clear go/no-go decision documented.

## Phase 1: Build and Baseline Harness

Tasks:

1. Integrate Kokkos into CMake with pinned version/toolchain matrix.
2. Decide whether `main/main.cu` remains `.cu` during transition or moves to `.cpp`.
3. Add cross-repo comparison scripts that run identical inputs and compare:
	- `out.out`
	- `out.levels`
	- cut size
	- partition weight balance
	- level counts
4. Add invariant checks for multilevel state.

Exit criteria:

- Kokkos build path compiles and runs smoke tests.
- Comparison script produces pass/fail report against CUDA baseline repo.

## Phase 2: Memory and Orchestration Port

Tasks:

1. Replace core allocations/copies with Kokkos views/deep_copy.
2. Port top-level orchestration in `gkway/graph_partitioner.hpp` to Kokkos views and fences.
3. Keep temporary adapter code minimal and short-lived.

Exit criteria:

- Smoke-test functional parity on selected graphs.
- No raw `cudaMalloc/cudaMemcpy` in orchestrator path.

## Phase 3: Coarsening Port

Tasks:

1. Port matching/group logic in `gkway/coarsen.hpp`.
2. Port contraction/compaction path with chosen primitive replacements.
3. Preserve and validate `cmap` semantics level by level.

Required invariants:

- CSR validity at each coarse level.
- `cmap` index validity.
- expected level-size monotonic behavior.

Exit criteria:

- Coarsening artifacts within accepted parity bands.
- Performance within phase budget.

## Phase 4: Uncoarsening and Refinement Port

Tasks:

1. Port `expand_partition`, boundary updates, gain computation, and move application in `gkway/uncoarsen.hpp`.
2. Define refinement state as explicit struct-of-views.
3. Define conflict-resolution/tie-break behavior for concurrent updates.
4. Add deterministic debug mode (fixed tie-break rules).

Exit criteria:

- End-to-end artifact parity within acceptance bands.
- Deterministic mode available for debugging regressions.

## Phase 5: Hardening and Cleanup

Tasks:

1. Keep CUDA and Kokkos pathways side-by-side; do not remove CUDA path.
2. Remove temporary migration scaffolding.
3. Add contributor playbook for Kokkos kernel conventions, testing, and profiling.
	- done: `kokkos_port/CONTRIBUTOR_PLAYBOOK.md`

Exit criteria:

- Stable end-to-end results on benchmark suite.
- Performance and memory usage meet agreed thresholds.

## File-by-File Port Notes

### `declarations.h`

- Replace CUDA constant memory usage with runtime config objects passed into kernels/functors.

### `main/main.cu`

- Introduce Kokkos initialize/finalize lifecycle.
- Keep CLI behavior unchanged.

### `gkway/cuda_check.hpp`

- Replace CUDA error wrappers with Kokkos-compatible checks/asserts.
- Move most debug inspection to host-side utilities.

### `gkway/coarsen.hpp`

- Highest risk for primitive and data-layout interaction.
- Port in small validated slices (group formation -> compaction -> coarse graph build).

### `gkway/uncoarsen.hpp`

- Highest risk for mutable state and races.
- Port with explicit state layout and deterministic debugging path.

### `gkway/graph_partitioner.hpp`

- Refactor into coordinator + helper modules.
- Keep level bookkeeping and output semantics stable while internals change.

## Primitive Replacement Map

| Current Pattern | Kokkos Direction |
| --- | --- |
| `cudaMalloc/cudaFree` | `Kokkos::View` ownership |
| `cudaMemcpy/cudaMemset` | `Kokkos::deep_copy` / fill kernels |
| `__global__` kernels | `Kokkos::parallel_for/reduce/scan` |
| stream sync | execution-space fence |
| Moderngpu scan/sort/segmented ops | KokkosKernels or custom kernels from Phase 0 decision |

## Validation and Acceptance

Validate at every phase with two layers:

1. Artifact comparison against separate CUDA baseline repo.
2. Internal invariants at each level.

Track these metrics:

- cut size
- partition weight balance
- level counts and sizes
- end-to-end runtime
- memory footprint

Define acceptance bands explicitly. This code is not production-critical, so exact bitwise parity is not mandatory unless required for debugging.

## Risks and Mitigations

1. Primitive gaps for Moderngpu replacements.
	- Mitigation: mandatory Phase 0 feasibility matrix.
2. Non-deterministic refinement behavior.
	- Mitigation: deterministic debug mode and explicit tie-break policy.
3. Data-layout churn causing rework.
	- Mitigation: freeze layout decision early.
4. Toolchain friction in mixed CUDA/Kokkos builds.
	- Mitigation: pinned compiler/Kokkos matrix and early build hardening.

## Bottom Line

Treat this as a greenfield capability built with continuous comparison to a frozen CUDA baseline repo. The plan succeeds only if feasibility is proven first, architecture choices are frozen early, and every phase is gated by explicit correctness and performance checks.