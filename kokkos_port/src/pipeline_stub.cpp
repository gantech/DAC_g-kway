#include "gkway_kokkos/pipeline.hpp"
#include "gkway_kokkos/graph_loader.hpp"

#include "metis/metis.h"

#include <Kokkos_Core.hpp>

#include <array>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gkway_kokkos {

namespace {

struct HostCoarseGraph {
  std::vector<unsigned> adjp;
  std::vector<unsigned> adjncy;
  std::vector<unsigned> adjwgt;
  std::vector<unsigned> vwgt;
};

template <typename GraphLike>
unsigned choose_heaviest_neighbor(const GraphLike& graph, std::size_t vertex) {
  const unsigned edge_begin = graph.adjp[vertex];
  const unsigned edge_end = graph.adjp[vertex + 1];
  const unsigned num_vertices = static_cast<unsigned>(graph.vwgt.size());
  unsigned best_neighbor = num_vertices > 0
                               ? std::min(static_cast<unsigned>(vertex + 1u), num_vertices - 1u)
                               : 0u;
  unsigned best_weight = 0;
  unsigned best_degree = 0;

  for (unsigned edge = edge_begin; edge < edge_end; ++edge) {
    const unsigned neighbor_raw = graph.adjncy[edge];
    if (neighbor_raw == 0 || neighbor_raw > num_vertices) {
      continue;
    }

    const unsigned neighbor = neighbor_raw - 1u;
    if (neighbor == vertex) {
      continue;
    }

    const unsigned weight = graph.adjwgt[edge];
    const unsigned degree = graph.adjp[neighbor + 1] - graph.adjp[neighbor];
    if (weight > best_weight || (weight == best_weight && degree < best_degree)) {
      best_weight = weight;
      best_degree = degree;
      best_neighbor = neighbor;
    }
  }

  return best_neighbor;
}

template <typename GraphLike>
std::vector<unsigned> build_matching_cmap(const GraphLike& graph, unsigned max_group_size) {
  const std::size_t num_vertices = graph.vwgt.size();
  const unsigned group_cap = std::max(2u, max_group_size);
  std::vector<unsigned> candidate(num_vertices, 0u);
  std::vector<unsigned long long> group_id(num_vertices, 0ull);
  std::vector<unsigned> vertex_id(num_vertices, 0u);

  for (std::size_t vertex = 0; vertex < num_vertices; ++vertex) {
    candidate[vertex] = choose_heaviest_neighbor(graph, vertex) + 1u;
    group_id[vertex] = (static_cast<unsigned long long>(vertex + 1u) << 32);
    vertex_id[vertex] = static_cast<unsigned>(vertex + 1u);
  }

  auto get_first = [](unsigned long long value) -> unsigned {
    return static_cast<unsigned>(value >> 32);
  };

  auto make_combo = [](unsigned first, unsigned second) -> unsigned long long {
    return (static_cast<unsigned long long>(first) << 32) | static_cast<unsigned long long>(second);
  };

  bool group_changed = true;
  unsigned iteration = 1u;
  while (group_changed) {
    group_changed = false;
    for (std::size_t vertex = 0; vertex < num_vertices; ++vertex) {
      const unsigned partner = candidate[vertex];
      if (partner == 0u || partner > num_vertices) {
        continue;
      }

      const unsigned current_group = get_first(group_id[vertex]);
      const unsigned neighbor_group = get_first(group_id[partner - 1u]);

      if (current_group > neighbor_group) {
        group_id[partner - 1u] = make_combo(current_group, iteration);
        group_changed = true;
      } else if (current_group < neighbor_group) {
        group_id[vertex] = make_combo(neighbor_group, iteration);
        group_changed = true;
      }
    }

    ++iteration;
  }

  std::vector<std::pair<unsigned, unsigned>> sorted_members;
  sorted_members.reserve(num_vertices);
  for (std::size_t vertex = 0; vertex < num_vertices; ++vertex) {
    sorted_members.emplace_back(get_first(group_id[vertex]), static_cast<unsigned>(vertex + 1u));
  }
  std::sort(sorted_members.begin(), sorted_members.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.first != rhs.first) {
      return lhs.first < rhs.first;
    }
    return lhs.second < rhs.second;
  });

  for (std::size_t begin = 0; begin < sorted_members.size();) {
    const unsigned root = sorted_members[begin].first;
    std::size_t end = begin;
    while (end < sorted_members.size() && sorted_members[end].first == root) {
      ++end;
    }

    const std::size_t group_size = end - begin;
    if (group_size > group_cap) {
      for (std::size_t offset = begin; offset < end; offset += group_cap) {
        const unsigned chunk_head = sorted_members[offset].second;
        const std::size_t chunk_end = std::min(offset + static_cast<std::size_t>(group_cap), end);
        for (std::size_t index = offset; index < chunk_end; ++index) {
          group_id[sorted_members[index].second - 1u] = make_combo(chunk_head, 0u);
        }
      }
    }

    begin = end;
  }

  std::vector<std::pair<unsigned, unsigned>> heads;
  heads.reserve(num_vertices);
  for (std::size_t vertex = 0; vertex < num_vertices; ++vertex) {
    if (get_first(group_id[vertex]) == vertex + 1u) {
      heads.emplace_back(static_cast<unsigned>(vertex + 1u), static_cast<unsigned>(vertex + 1u));
    }
  }

  std::sort(heads.begin(), heads.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.first < rhs.first;
  });

  std::unordered_map<unsigned, unsigned> head_to_cmap;
  head_to_cmap.reserve(heads.size());
  unsigned coarse_id = 1u;
  for (const auto& head : heads) {
    head_to_cmap[head.first] = coarse_id++;
  }

  std::vector<unsigned> cmap(num_vertices, 0u);
  for (std::size_t vertex = 0; vertex < num_vertices; ++vertex) {
    cmap[vertex] = head_to_cmap[get_first(group_id[vertex])];
  }

  return cmap;
}

template <typename GraphLike>
HostCoarseGraph build_coarse_graph(const GraphLike& graph, const std::vector<unsigned>& cmap) {
  HostCoarseGraph coarse;
  const unsigned num_coarse_vertices = *std::max_element(cmap.begin(), cmap.end());
  coarse.vwgt.assign(num_coarse_vertices, 0u);

  for (std::size_t vertex = 0; vertex < cmap.size(); ++vertex) {
    coarse.vwgt[cmap[vertex] - 1u] += graph.vwgt[vertex];
  }

  std::vector<std::vector<std::pair<unsigned, unsigned>>> rows(num_coarse_vertices);
  for (std::size_t source = 0; source < cmap.size(); ++source) {
    const unsigned source_coarse = cmap[source] - 1u;
    const unsigned edge_begin = graph.adjp[source];
    const unsigned edge_end = graph.adjp[source + 1];

    for (unsigned edge = edge_begin; edge < edge_end; ++edge) {
      const unsigned neighbor_raw = graph.adjncy[edge];
      if (neighbor_raw == 0 || neighbor_raw > cmap.size()) {
        continue;
      }

      const unsigned neighbor = neighbor_raw - 1u;
      const unsigned target_coarse = cmap[neighbor] - 1u;
      if (source_coarse == target_coarse) {
        continue;
      }

      rows[source_coarse].emplace_back(target_coarse + 1u, graph.adjwgt[edge]);
    }
  }

  coarse.adjp.assign(num_coarse_vertices + 1u, 0u);
  for (std::size_t row = 0; row < rows.size(); ++row) {
    auto& entries = rows[row];
    std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
      if (lhs.first != rhs.first) {
        return lhs.first < rhs.first;
      }
      return lhs.second < rhs.second;
    });

    std::size_t compacted = 0;
    for (std::size_t index = 0; index < entries.size();) {
      const unsigned target = entries[index].first;
      unsigned weight = entries[index].second;
      std::size_t next = index + 1;
      while (next < entries.size() && entries[next].first == target) {
        weight += entries[next].second;
        ++next;
      }
      if (target != row + 1u) {
        entries[compacted++] = {target, weight};
      }
      index = next;
    }
    entries.resize(compacted);
    coarse.adjp[row + 1u] = static_cast<unsigned>(entries.size());
  }
  for (std::size_t i = 1; i < coarse.adjp.size(); ++i) {
    coarse.adjp[i] += coarse.adjp[i - 1u];
  }

  const std::size_t total_edges = coarse.adjp.back();
  coarse.adjncy.assign(total_edges, 0u);
  coarse.adjwgt.assign(total_edges, 0u);
  std::vector<unsigned> cursor(num_coarse_vertices, 0u);
  for (std::size_t row = 0; row < rows.size(); ++row) {
    for (const auto& entry : rows[row]) {
      const unsigned slot = coarse.adjp[row] + cursor[row]++;
      coarse.adjncy[slot] = entry.first;
      coarse.adjwgt[slot] = entry.second;
    }
  }

  return coarse;
}

}  // namespace

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

  std::vector<std::vector<unsigned>> level_cmaps;
  HostCoarseGraph coarse_graph;
  coarse_graph.adjp = host_graph.adjp;
  coarse_graph.adjncy = host_graph.adjncy;
  coarse_graph.adjwgt = host_graph.adjwgt;
  coarse_graph.vwgt = host_graph.vwgt;

  const unsigned partition_count_u = config.num_partitions > 0 ? static_cast<unsigned>(config.num_partitions) : 1u;
  const unsigned coarsen_threshold = std::max(2u, 20u * partition_count_u);

  while (coarse_graph.vwgt.size() > coarsen_threshold) {
    const std::vector<unsigned> cmap = build_matching_cmap(coarse_graph, config.max_coarsen_group);
    HostCoarseGraph next_graph = build_coarse_graph(coarse_graph, cmap);
    if (next_graph.vwgt.size() == coarse_graph.vwgt.size() ||
        next_graph.vwgt.size() < static_cast<std::size_t>(partition_count_u + 1u)) {
      break;
    }

    level_cmaps.push_back(std::move(cmap));
    coarse_graph = std::move(next_graph);
  }

  auto write_outputs = [&](const std::string& prefix,
                           const Kokkos::View<unsigned*, Kokkos::HostSpace>& partition_host) {
    std::ofstream out_part(prefix + ".out");
    for (std::size_t i = 0; i < num_vertices; ++i) {
      out_part << partition_host(i) << '\n';
    }

    std::ofstream out_levels(prefix + ".levels");
    out_levels << "PartitionID";
    for (std::size_t level = level_cmaps.size(); level > 0; --level) {
      out_levels << ",L" << level;
    }
    out_levels << ",L0\n";

    for (std::size_t i = 0; i < num_vertices; ++i) {
      std::vector<unsigned> lineage;
      lineage.reserve(level_cmaps.size());
      unsigned current_vertex = static_cast<unsigned>(i + 1u);
      for (const std::vector<unsigned>& cmap : level_cmaps) {
        current_vertex = cmap[current_vertex - 1u];
        lineage.push_back(current_vertex);
      }

      out_levels << partition_host(i);
      for (auto it = lineage.rbegin(); it != lineage.rend(); ++it) {
        out_levels << ',' << *it;
      }
      out_levels << ',' << (i + 1) << '\n';
    }
  };

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

  {
    if (!level_cmaps.empty()) {
      Kokkos::View<unsigned*, Kokkos::HostSpace> h_cmap(level_cmaps.front().data(), level_cmaps.front().size());
      Kokkos::deep_copy(level0.cmap, h_cmap);
    } else {
      Kokkos::parallel_for(
          "init_identity_cmap", Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(num_vertices)),
          KOKKOS_LAMBDA(const int i) { level0.cmap(i) = static_cast<unsigned>(i + 1u); });
    }
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

  const int partition_count = config.num_partitions > 0 ? config.num_partitions : 1;

  const std::size_t num_coarse_vertices = coarse_graph.vwgt.size();
  view_u coarse_vwgt("coarse_vwgt", num_coarse_vertices);
  {
    Kokkos::View<unsigned*, Kokkos::HostSpace> h_coarse_vwgt(coarse_graph.vwgt.data(), coarse_graph.vwgt.size());
    Kokkos::deep_copy(coarse_vwgt, h_coarse_vwgt);
  }

  unsigned long long total_coarse_wgt = 0;
  for (unsigned weight : coarse_graph.vwgt) {
    total_coarse_wgt += static_cast<unsigned long long>(weight);
  }

  if (total_coarse_wgt == 0) {
    total_coarse_wgt = 1;
  }

  const unsigned long long ideal_partition_wgt =
      (total_coarse_wgt + static_cast<unsigned long long>(partition_count) - 1ull) /
      static_cast<unsigned long long>(partition_count);
  const unsigned long long partition_wgt_cap =
      (config.max_partition_wgt > 0.0f)
          ? static_cast<unsigned long long>(config.max_partition_wgt)
          : (ideal_partition_wgt + (ideal_partition_wgt / 20ull) + 1ull);

  view_u coarse_adjp("coarse_adjp", coarse_graph.adjp.size());
  view_u coarse_adjncy("coarse_adjncy", coarse_graph.adjncy.size());
  view_u coarse_adjwgt("coarse_adjwgt", coarse_graph.adjwgt.size());
  {
    Kokkos::View<unsigned*, Kokkos::HostSpace> h_coarse_adjp(coarse_graph.adjp.data(), coarse_graph.adjp.size());
    Kokkos::View<unsigned*, Kokkos::HostSpace> h_coarse_adjncy(coarse_graph.adjncy.data(), coarse_graph.adjncy.size());
    Kokkos::View<unsigned*, Kokkos::HostSpace> h_coarse_adjwgt(coarse_graph.adjwgt.data(), coarse_graph.adjwgt.size());
    Kokkos::deep_copy(coarse_adjp, h_coarse_adjp);
    Kokkos::deep_copy(coarse_adjncy, h_coarse_adjncy);
    Kokkos::deep_copy(coarse_adjwgt, h_coarse_adjwgt);
  }

  auto h_coarse_adjp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coarse_adjp);
  auto h_coarse_adjncy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coarse_adjncy);
  auto h_coarse_adjwgt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coarse_adjwgt);
  auto h_coarse_vwgt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coarse_vwgt);

  std::vector<idx_t> mt_vwgt(num_coarse_vertices);
  std::vector<idx_t> mt_adjp(num_coarse_vertices + 1);
  std::vector<idx_t> mt_adjncy(h_coarse_adjncy.extent(0));
  std::vector<idx_t> mt_adjwgt(h_coarse_adjwgt.extent(0));
  std::vector<idx_t> mt_partition(num_coarse_vertices);
  idx_t mt_num_vertex = static_cast<idx_t>(num_coarse_vertices);
  idx_t ncon = 1;
  idx_t mt_num_part = static_cast<idx_t>(partition_count);
  idx_t mt_cutsize = 0;

  for (std::size_t i = 0; i < num_coarse_vertices; ++i) {
    mt_vwgt[i] = static_cast<idx_t>(h_coarse_vwgt(i));
    mt_adjp[i] = static_cast<idx_t>(h_coarse_adjp(i));
  }
  mt_adjp[num_coarse_vertices] = static_cast<idx_t>(h_coarse_adjp(num_coarse_vertices));

  for (std::size_t i = 0; i < h_coarse_adjncy.extent(0); ++i) {
    mt_adjncy[i] = static_cast<idx_t>(h_coarse_adjncy(i) - 1u);
    mt_adjwgt[i] = static_cast<idx_t>(h_coarse_adjwgt(i));
  }

  const int metis_result = METIS_PartGraphKway(
      &mt_num_vertex,
      &ncon,
      mt_adjp.data(),
      mt_adjncy.data(),
      mt_vwgt.data(),
      nullptr,
      mt_adjwgt.data(),
      &mt_num_part,
      nullptr,
      nullptr,
      nullptr,
      &mt_cutsize,
      mt_partition.data());

  if (metis_result != METIS_OK) {
    std::cerr << "METIS_PartGraphKway failed with code " << metis_result << '\n';
    return;
  }

  Kokkos::View<unsigned*, Kokkos::HostSpace> h_metis_partition("h_partition", num_coarse_vertices);
  for (std::size_t i = 0; i < num_coarse_vertices; ++i) {
    h_metis_partition(i) = static_cast<unsigned>(mt_partition[i]);
  }

  view_u coarse_partition("coarse_partition", num_coarse_vertices);
  Kokkos::deep_copy(coarse_partition, h_metis_partition);

  std::vector<unsigned> projected_partition(num_coarse_vertices);
  for (std::size_t i = 0; i < num_coarse_vertices; ++i) {
    projected_partition[i] = h_metis_partition(i);
  }
  for (std::size_t level = level_cmaps.size(); level > 0; --level) {
    const std::vector<unsigned>& cmap = level_cmaps[level - 1];
    std::vector<unsigned> finer_partition(cmap.size());
    for (std::size_t vertex = 0; vertex < cmap.size(); ++vertex) {
      finer_partition[vertex] = projected_partition[cmap[vertex] - 1u];
    }
    projected_partition = std::move(finer_partition);
  }

  Kokkos::View<unsigned*, Kokkos::HostSpace> h_projected_partition(projected_partition.data(), projected_partition.size());
  Kokkos::deep_copy(state.partition, h_projected_partition);
  Kokkos::deep_copy(state.partition_wgt, 0u);

  auto pre_refine_partition = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.partition);
  write_outputs(options.out_prefix + ".pre_refine", pre_refine_partition);

  unsigned long long proposed_moves = 0;
  unsigned long long last_pass_moves = 0;
  int executed_refinement_passes = 0;
  const int refinement_passes = options.refinement_passes > 0 ? options.refinement_passes : 1;
  for (int pass = 0; pass < refinement_passes; ++pass) {
    executed_refinement_passes = pass + 1;
    Kokkos::deep_copy(state.partition_wgt, 0u);
    Kokkos::parallel_for(
        "recompute_partition_weights_pass", Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(num_vertices)),
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
          if ((selected_target != partition_id) && (refine.gain(i) > 0)) {
            const unsigned vertex_wgt = level0.vwgt(i);
            const unsigned target_wgt = state.partition_wgt(selected_target);
            const bool parity_selected = (((static_cast<unsigned>(i) + static_cast<unsigned>(pass)) & 1u) == 0u);
            const bool within_cap =
                static_cast<unsigned long long>(target_wgt) + static_cast<unsigned long long>(vertex_wgt) <=
                partition_wgt_cap;
            refine.move_flag(i) = (parity_selected && within_cap) ? 1u : 0u;
          } else {
            refine.move_flag(i) = 0u;
          }
        });

    unsigned long long pass_moves = 0;
    Kokkos::parallel_reduce(
        "count_proposed_moves", Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(num_vertices)),
        KOKKOS_LAMBDA(const int i, unsigned long long& local_count) {
          local_count += static_cast<unsigned long long>(refine.move_flag(i));
        },
        pass_moves);

    last_pass_moves = pass_moves;
    proposed_moves += pass_moves;
    if (pass_moves == 0) {
      break;
    }

    Kokkos::parallel_for(
        "apply_refinement_moves", Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(num_vertices)),
        KOKKOS_LAMBDA(const int i) {
          if (refine.move_flag(i) != 0u) {
            state.partition(i) = refine.target_partition(i);
          }
        });
  }

  Kokkos::deep_copy(state.partition_wgt, 0u);
  Kokkos::parallel_for(
      "recompute_partition_weights", Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(num_vertices)),
      KOKKOS_LAMBDA(const int i) {
        const unsigned partition_id = state.partition(i);
        Kokkos::atomic_add(&state.partition_wgt(partition_id), level0.vwgt(i));
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

  auto h_final_partition = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.partition);
  auto h_final_partition_wgt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.partition_wgt);
  auto h_cutsize = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state.cutsize);

  write_outputs(options.out_prefix, h_final_partition);
  write_outputs(options.out_prefix + ".post_refine", h_final_partition);

  unsigned max_partition_wgt = 0;
  unsigned min_partition_wgt = h_final_partition_wgt.extent(0) > 0 ? h_final_partition_wgt(0) : 0;
  unsigned long long total_partition_wgt = 0;
  for (std::size_t p = 0; p < static_cast<std::size_t>(h_final_partition_wgt.extent(0)); ++p) {
    total_partition_wgt += static_cast<unsigned long long>(h_final_partition_wgt(p));
    if (h_final_partition_wgt(p) < min_partition_wgt) {
      min_partition_wgt = h_final_partition_wgt(p);
    }
    if (h_final_partition_wgt(p) > max_partition_wgt) {
      max_partition_wgt = h_final_partition_wgt(p);
    }
  }

  const unsigned long long avg_partition_wgt =
      partition_count > 0 ? (total_partition_wgt / static_cast<unsigned long long>(partition_count)) : 0;

  std::cout << "[kokkos_port] phase-1 stub completed for " << num_vertices
            << " vertices, cutsize=" << h_cutsize(0)
            << ", max_partition_wgt=" << max_partition_wgt
            << ", proposed_moves=" << proposed_moves
            << ", refinement_passes=" << refinement_passes
            << ", executed_refinement_passes=" << executed_refinement_passes
            << ", last_pass_moves=" << last_pass_moves
            << ", min_partition_wgt=" << min_partition_wgt
            << ", avg_partition_wgt=" << avg_partition_wgt
            << ", partition_wgt_cap=" << partition_wgt_cap << "\n";
}

template void run_pipeline_stub<Kokkos::DefaultExecutionSpace>(const RunOptions& options,
                                                               const PartitionConfig& config);

}  // namespace gkway_kokkos
