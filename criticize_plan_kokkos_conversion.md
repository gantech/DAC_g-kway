# Critique of the CUDA -> Kokkos Conversion Plan

## Executive Summary

The existing plan is directionally good but under-specifies the hardest engineering decisions. It reads like a roadmap, not an execution plan. The biggest issue is that it assumes Kokkos replacements exist for key Moderngpu operations without explicitly proving they do for this algorithm.

In short:

- Strong on intent and structure.
- Weak on feasibility proofs for core primitives.
- Missing performance and correctness gates that can stop regressions early.
- Missing a concrete backend and memory-layout policy.
- Needs an explicit no-METIS scope statement so migration work does not drift.

Scope note for this critique:

- METIS-integrated code paths are intentionally out of scope.
- Recommendations focus on the multilevel CUDA/Moderngpu path only.

## What Is Good

1. Correctly identifies coarsening/uncoarsening as the highest-risk components.
2. Avoids a one-shot rewrite and proposes phased migration.
3. Recognizes that the current code is not just CUDA kernels but also tightly coupled data layout + primitives.

These are important and should be preserved.

## Key Weaknesses

## 1) No Feasibility Spike for Moderngpu Replacements

The plan says to replace scans/sorts/segmented operations later, but this is the main uncertainty in the entire effort.

Why this is a problem:

- If the exact operation set is not available or has poor performance in Kokkos/KokkosKernels, the rest of the migration may stall after substantial effort.

Correction:

- Add a Phase 0 feasibility spike before broad refactoring.
- Implement microbenchmarks for each required primitive on representative sizes.
- Decide early whether to use KokkosKernels only, custom kernels, or a hybrid backend-specific layer.

## 2) Backend Strategy Is Too Vague

The plan targets CUDA/HIP/CPU portability, but does not define what "supported" means for each backend.

Why this is a problem:

- You can accidentally optimize around one backend and produce unacceptable behavior on others.

Correction:

- Define official backends up front (for example: CUDA and OpenMP first; HIP later).
- Define a "passing" matrix: correctness + performance thresholds by backend.

## 3) Data Layout Decision Is Deferred Too Long

The plan postpones packed-vs-per-level layout choice. That decision drives most kernel signatures and memory movement.

Why this is a problem:

- Delaying it causes rework in every subsequent phase.

Correction:

- Make this decision in an explicit architecture checkpoint early.
- Require a short comparison prototype for both layouts, then freeze one.

## 4) Missing Determinism and Numerical Stability Policy

Graph contraction/refinement can be order-sensitive.

Why this is a problem:

- Kokkos execution ordering may differ from CUDA, producing output drift that is hard to diagnose.

Correction:

- Define acceptable nondeterminism boundaries now.
- Add deterministic test mode for debugging (fixed seeds, stabilized tie-breaks).

## 5) Validation Criteria Are Too Coarse

The current validation checks final outputs and cutsize, but not intermediate invariants.

Why this is a problem:

- Bugs in one level may be masked by later refinement.

Correction:

- Add per-level invariants:
  - valid CSR boundaries at each level
  - cmap index validity and surjectivity expectations
  - partition label bounds and boundary flags consistency
  - conservation checks on vertex weights

## 6) No Performance Budget or Regression Gates

The plan mentions correctness but lacks explicit performance targets.

Why this is a problem:

- A correct port can still be unusable.

Correction:

- Set per-phase performance guardrails (for example: no worse than X% vs CUDA baseline on chosen datasets).
- Track both end-to-end time and kernel-level hot spots.

## 7) Missing Build-System Migration Details

The plan says "add Kokkos to CMake" but omits practical build constraints.

Why this is a problem:

- Real projects fail in toolchain integration before algorithmic porting starts.

Correction:

- Define compiler/toolchain matrix, required C++ standard, and Kokkos version pin.
- Define whether `main/main.cu` remains `.cu` or becomes `.cpp` and how mixed translation units are handled during transition.

## 8) Underestimates Refinement State Complexity

The plan notes refinement is stateful, but not how that state will be represented and updated safely.

Why this is a problem:

- Refinement bugs are typically race-condition bugs and hard to reproduce.

Correction:

- Design refinement state as an explicit struct-of-views with versioned update phases.
- Define conflict-resolution strategy for concurrent moves before coding.

## 9) Baseline Comparison Strategy Is Underdefined

If you are intentionally building a new capability in a separate repository copy, in-tree side-by-side fallback is unnecessary. But you still need a disciplined comparison workflow.

Why this is a problem:

- Without a repeatable baseline comparison process, it is hard to distinguish intended behavior changes from accidental regressions.

Correction:

- Use the CUDA repo snapshot as an external baseline, not an in-tree fallback.
- Add scripts that run identical inputs in both repos and compare key artifacts (cut size, partition balance, level counts, output files).
- Define acceptance bands for differences that are expected in a redesigned implementation.

## 10) Documentation Scope Is Incomplete

The plan provides architecture text but not developer workflow requirements.

Why this is a problem:

- Contributors cannot reliably continue migration without coding standards and checklists.

Correction:

- Add a migration playbook:
  - coding conventions for Kokkos kernels
  - allowed abstractions
  - profiling workflow
  - mandatory tests per PR

## Suggested Revised Plan Structure

## Phase 0: Feasibility and Baseline (mandatory)

1. Lock benchmark datasets and baseline metrics (runtime, memory, cutsize, partition quality).
2. Prototype required primitives in Kokkos/KokkosKernels.
3. Choose backend support scope and freeze it.
4. Decide data layout model and freeze it.

Exit criteria:

- Primitive coverage confirmed.
- Clear "go/no-go" decision documented.

## Phase 1: Infrastructure and Baseline Harness

1. Integrate Kokkos into build.
2. Add artifact comparators and invariant checks.
3. Add scripts to run and compare against the separate CUDA baseline repository.

Exit criteria:

- Kokkos path compiles and runs on smoke tests.
- Baseline comparison script produces a clear pass/fail report.

## Phase 2: Memory and Orchestration Port

1. Port top-level storage and orchestration to Kokkos views.
2. Keep temporary adapter code minimal; prefer direct Kokkos kernels unless a short-lived bridge is needed.

Exit criteria:

- Functionally equivalent outputs on smoke tests.

## Phase 3: Coarsening Port

1. Port matching/contraction.
2. Validate per-level invariants and performance.

Exit criteria:

- Coarsening parity and acceptable performance against baseline.

## Phase 4: Uncoarsening and Refinement Port

1. Port expansion and move/refinement path.
2. Validate determinism mode and conflict handling.

Exit criteria:

- End-to-end parity on full benchmarks.

## Phase 5: Hardening and Cleanup

1. Remove dead CUDA-only code where safe.
2. Remove temporary migration scaffolding once Kokkos behavior is validated against baseline artifacts.

Exit criteria:

- CI green across supported backends.
- Performance within agreed budget.

## Recommended Additional Deliverables

1. Primitive coverage matrix (Moderngpu usage -> Kokkos implementation choice).
2. Benchmark report template with required metrics.
3. Invariant checker module for each multilevel stage.
4. Cross-repo comparison harness for CUDA-baseline vs Kokkos artifacts.
5. Risk register with owner and mitigation for each major uncertainty.

## Bottom Line

The current plan is a solid conceptual roadmap but not yet an executable migration program. To make it actionable, add a feasibility-first Phase 0, freeze backend/layout decisions early, define explicit correctness and performance gates, and establish a repeatable cross-repo baseline comparison workflow.