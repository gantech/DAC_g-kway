# Kokkos Port Contributor Playbook

This port is an incremental Kokkos capability that stays side-by-side with the existing CUDA path. Keep changes small, validated, and easy to compare.

## Code Conventions

- Prefer per-level data structures and explicit views over packed one-off state.
- Keep host-only parsing and file I/O separate from execution-space code.
- Use `Kokkos::View`, `Kokkos::deep_copy`, `Kokkos::parallel_for`, `Kokkos::parallel_reduce`, and `Kokkos::parallel_scan` for device-side work.
- Preserve the current CSV artifact formats unless a change is explicitly part of the migration.
- Keep the CUDA path intact; do not remove or rewrite it as part of Kokkos work.

## Build Setup

The validated build uses the configured Spack/Kokkos environment and the `znver3` host flags.

```bash
cmake -S . -B build-kokkos -DGKWAY_ENABLE_KOKKOS_PORT=ON -DCMAKE_C_FLAGS="-march=znver3" -DCMAKE_CXX_FLAGS="-march=znver3"
cmake --build build-kokkos --target gkway-kokkos gkway-kokkos-phase0 -j2
```

## Validation Matrix

Run the narrowest checks that cover the change:

```bash
./build-kokkos/exec/gkway-kokkos-phase0 500000
./build-kokkos/exec/gkway-kokkos mesh_graph.metis 64 out_kokkos_stub 3
python3 kokkos_port/tools/check_levels_invariants.py --levels out_kokkos_stub.levels --num-partitions 64
python3 kokkos_port/tools/check_multilevel_consistency.py --levels out_kokkos_stub.pre_refine.levels --require-coarse-partition-consistency
python3 kokkos_port/tools/check_coarsened_weight_conservation.py --graph mesh_graph.metis --levels out_kokkos_stub.pre_refine.levels
python3 kokkos_port/tools/run_phase1_compare.sh out_kokkos_stub out_kokkos_stub out_kokkos_stub.levels 64 mesh_graph.metis 1
```

Use relaxed multilevel consistency for post-refinement snapshots and strict mode for pre-refinement snapshots.

## Profiling Notes

- Start with artifact and invariant checks before measuring runtime.
- Use refinement diagnostics from pipeline output to see whether move generation is still active.
- Watch `cutsize`, `max_partition_wgt`, `min_partition_wgt`, `avg_partition_wgt`, and `proposed_moves` together; a single metric is not enough.
- If a change touches coarsening or partition assignment, validate the `.pre_refine` snapshot first.

## Change Discipline

- Prefer one behavior slice per commit.
- Update `kokkos_conversion_progress.md` whenever a milestone is validated.
- If a strict checker fails on a post-refinement snapshot, confirm whether the pre-refinement snapshot is the intended target before changing the checker.
