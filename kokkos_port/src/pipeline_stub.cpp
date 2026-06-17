#include "gkway_kokkos/pipeline.hpp"
#include "gkway_kokkos/graph_loader.hpp"

#include <Kokkos_Core.hpp>

#include <iostream>
#include <fstream>

namespace gkway_kokkos {

template <typename ExecSpace>
void run_pipeline_stub(const RunOptions& options, const PartitionConfig& config) {
  using view_u = Kokkos::View<unsigned*, typename ExecSpace::memory_space>;
  using view_ull = Kokkos::View<unsigned long long*, typename ExecSpace::memory_space>;

  HostGraph host_graph;
  try {
    host_graph = load_graph_from_metis_like(options.graph_file);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return;
  }

  const std::size_t num_vertices = host_graph.num_vertices;
  if (num_vertices == 0) {
    std::cerr << "invalid graph: zero vertices\n";
    return;
  }

  GraphLevel<ExecSpace> level0;
  level0.num_vertices = num_vertices;
  level0.num_edges = host_graph.num_edges;
  level0.adjp = view_u("adjp", host_graph.adjp.size());
  level0.adjncy = view_u("adjncy", host_graph.adjncy.size());
  level0.adjwgt = view_u("adjwgt", host_graph.adjwgt.size());
  level0.vwgt = view_u("vwgt", num_vertices);
  level0.cmap = view_u("cmap", num_vertices);

  {
    Kokkos::View<unsigned*, Kokkos::HostSpace> h_adjp(host_graph.adjp.data(), host_graph.adjp.size());
    Kokkos::View<unsigned*, Kokkos::HostSpace> h_adjncy(host_graph.adjncy.data(), host_graph.adjncy.size());
    Kokkos::View<unsigned*, Kokkos::HostSpace> h_adjwgt(host_graph.adjwgt.data(), host_graph.adjwgt.size());
    Kokkos::View<unsigned*, Kokkos::HostSpace> h_vwgt(host_graph.vwgt.data(), host_graph.vwgt.size());
    Kokkos::deep_copy(level0.adjp, h_adjp);
    Kokkos::deep_copy(level0.adjncy, h_adjncy);
    Kokkos::deep_copy(level0.adjwgt, h_adjwgt);
    Kokkos::deep_copy(level0.vwgt, h_vwgt);
  }

  PartitionState<ExecSpace> state;
  state.partition = view_u("partition", num_vertices);
  state.partition_wgt = view_u("partition_wgt", config.num_partitions > 0 ? config.num_partitions : 1);
  state.if_boundary = view_u("if_boundary", num_vertices);
  state.cutsize = view_u("cutsize", 1);

  RefinementState<ExecSpace> refine;
  refine.gain = Kokkos::View<int*, typename ExecSpace::memory_space>("gain", num_vertices);
  refine.target_partition = view_u("target_partition", num_vertices);
  refine.move_flag = view_u("move_flag", num_vertices);

  Kokkos::deep_copy(state.partition_wgt, 0u);
  Kokkos::deep_copy(state.if_boundary, 0u);
  Kokkos::deep_copy(state.cutsize, 0u);

  const std::size_t num_coarse_vertices = (num_vertices + 1) / 2;
  view_u coarse_vwgt("coarse_vwgt", num_coarse_vertices);
  view_ull coarse_prefix("coarse_prefix", num_coarse_vertices);
  view_u coarse_partition("coarse_partition", num_coarse_vertices);
  Kokkos::deep_copy(coarse_vwgt, 0u);

  Kokkos::parallel_for(
      "build_pairwise_cmap", Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(num_vertices)),
      KOKKOS_LAMBDA(const int i) {
        const unsigned coarse_idx = static_cast<unsigned>(i / 2);
        level0.cmap(i) = coarse_idx + 1;
        Kokkos::atomic_add(&coarse_vwgt(coarse_idx), level0.vwgt(i));
      });

  unsigned long long total_coarse_wgt = 0;
  Kokkos::parallel_reduce(
      "reduce_total_coarse_vwgt", Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(num_coarse_vertices)),
      KOKKOS_LAMBDA(const int i, unsigned long long& local_sum) {
        local_sum += static_cast<unsigned long long>(coarse_vwgt(i));
      },
      total_coarse_wgt);

  if (total_coarse_wgt == 0) {
    total_coarse_wgt = 1;
  }

  Kokkos::parallel_scan(
      "scan_coarse_prefix", Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(num_coarse_vertices)),
      KOKKOS_LAMBDA(const int i, unsigned long long& update, const bool final_pass) {
        const unsigned long long weight = static_cast<unsigned long long>(coarse_vwgt(i));
        if (final_pass) {
          coarse_prefix(i) = update;
        }
        update += weight;
      });

  const int partition_count = config.num_partitions > 0 ? config.num_partitions : 1;
  Kokkos::parallel_for(
      "init_coarse_partition", Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(num_coarse_vertices)),
      KOKKOS_LAMBDA(const int i) {
        const unsigned long long scaled_prefix =
            coarse_prefix(i) * static_cast<unsigned long long>(partition_count);
        unsigned partition_id = static_cast<unsigned>(scaled_prefix / total_coarse_wgt);
        if (partition_id >= static_cast<unsigned>(partition_count)) {
          partition_id = static_cast<unsigned>(partition_count - 1);
        }
        coarse_partition(i) = partition_id;
      });

  Kokkos::parallel_for(
      "uncoarsen_partition", Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(num_vertices)),
      KOKKOS_LAMBDA(const int i) {
        const unsigned coarse_idx = level0.cmap(i) - 1;
        state.partition(i) = coarse_partition(coarse_idx);
      });

  Kokkos::parallel_for(
      "accumulate_partition_weights", Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(num_vertices)),
      KOKKOS_LAMBDA(const int i) {
        const unsigned partition_id = state.partition(i);
        Kokkos::atomic_add(&state.partition_wgt(partition_id), level0.vwgt(i));
      });

  Kokkos::parallel_for(
      "compute_refinement_candidates", Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(num_vertices)),
      KOKKOS_LAMBDA(const int i) {
        const unsigned partition_id = state.partition(i);
        const unsigned edge_begin = level0.adjp(i);
        const unsigned edge_end = level0.adjp(i + 1);

        int internal_weight = 0;
        int external_weight = 0;
        unsigned selected_target = partition_id;
        int best_target_weight = -1;

        for (unsigned e = edge_begin; e < edge_end; ++e) {
          const unsigned neighbor_raw = level0.adjncy(e);
          if (neighbor_raw == 0 || neighbor_raw > num_vertices) {
            continue;
          }

          const unsigned neighbor = neighbor_raw - 1;
          const unsigned edge_weight = level0.adjwgt(e);
          const unsigned neighbor_partition = state.partition(neighbor);
          if (neighbor_partition == partition_id) {
            internal_weight += static_cast<int>(edge_weight);
            continue;
          }

          external_weight += static_cast<int>(edge_weight);
          if (static_cast<int>(edge_weight) > best_target_weight) {
            best_target_weight = static_cast<int>(edge_weight);
            selected_target = neighbor_partition;
          }
        }

        refine.gain(i) = external_weight - internal_weight;
        refine.target_partition(i) = selected_target;
        refine.move_flag(i) = (selected_target != partition_id && refine.gain(i) > 0) ? 1u : 0u;
      });

  unsigned long long total_cut = 0;
  Kokkos::parallel_reduce(
      "mark_boundary_and_cut", Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(num_vertices)),
      KOKKOS_LAMBDA(const int i, unsigned long long& local_cut) {
        const unsigned partition_id = state.partition(i);
        const unsigned edge_begin = level0.adjp(i);
        const unsigned edge_end = level0.adjp(i + 1);
        unsigned is_boundary = 0;

        for (unsigned e = edge_begin; e < edge_end; ++e) {
          const unsigned neighbor_raw = level0.adjncy(e);
          if (neighbor_raw == 0 || neighbor_raw > num_vertices) {
            continue;
          }

          const unsigned neighbor = neighbor_raw - 1;
          if (state.partition(neighbor) != partition_id) {
            is_boundary = 1;
            local_cut += level0.adjwgt(e);
          }
        }

        state.if_boundary(i) = is_boundary;
      },
      total_cut);

  {
    auto h_cut = Kokkos::create_mirror_view(state.cutsize);
    h_cut(0) = static_cast<unsigned>(total_cut / 2);
    Kokkos::deep_copy(state.cutsize, h_cut);
  }

  Kokkos::fence();

  auto h_partition = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.partition);
  auto h_cmap = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), level0.cmap);
  auto h_partition_wgt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.partition_wgt);
  auto h_cutsize = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.cutsize);

  {
    std::ofstream out_part(options.out_prefix + ".out");
    for (std::size_t i = 0; i < num_vertices; ++i) {
      out_part << h_partition(i) << '\n';
    }
  }

  {
    std::ofstream out_levels(options.out_prefix + ".levels");
    out_levels << "PartitionID,L1,L0\n";
    for (std::size_t i = 0; i < num_vertices; ++i) {
      out_levels << h_partition(i) << ',' << h_cmap(i) << ',' << (i + 1) << '\n';
    }
  }

  unsigned max_partition_wgt = 0;
  for (std::size_t p = 0; p < static_cast<std::size_t>(h_partition_wgt.extent(0)); ++p) {
    if (h_partition_wgt(p) > max_partition_wgt) {
      max_partition_wgt = h_partition_wgt(p);
    }
  }

  std::cout << "[kokkos_port] phase-1 stub completed for " << num_vertices
            << " vertices, cutsize=" << h_cutsize(0)
            << ", max_partition_wgt=" << max_partition_wgt << "\n";
}

template void run_pipeline_stub<Kokkos::DefaultExecutionSpace>(const RunOptions& options,
                                                               const PartitionConfig& config);

}  // namespace gkway_kokkos
