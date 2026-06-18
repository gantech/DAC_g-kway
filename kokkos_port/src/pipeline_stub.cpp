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
#include <cstdint>

namespace gkway_kokkos {

namespace {

struct HostCoarseGraph {
  std::vector<unsigned> adjp;
  std::vector<unsigned> adjncy;
  std::vector<unsigned> adjwgt;
  std::vector<unsigned> vwgt;
};

struct MoveRequestHost {
  unsigned vertex_id = 0;       // 1-based, matches CUDA mvRequest
  unsigned source_partition = 0;
  unsigned des_partition = 0;
  int gain = 0;
};

struct CandidateHost {
  unsigned target_partition = 0;
  int gain = 0;
  bool is_boundary = false;
  bool movable = false;
};

std::vector<unsigned> compute_partition_weights_host(const std::vector<unsigned>& partition,
                                                     const std::vector<unsigned>& vwgt,
                                                     int partition_count) {
  std::vector<unsigned> partition_wgt(static_cast<std::size_t>(partition_count), 0u);
  for (std::size_t v = 0; v < partition.size(); ++v) {
    partition_wgt[partition[v]] += vwgt[v];
  }
  return partition_wgt;
}

unsigned compute_cutsize_host(const HostCoarseGraph& graph,
                              const std::vector<unsigned>& partition,
                              std::vector<unsigned>* if_boundary) {
  if (if_boundary != nullptr) {
    if_boundary->assign(graph.vwgt.size(), 0u);
  }

  unsigned long long cut_twice = 0;
  for (std::size_t v = 0; v < graph.vwgt.size(); ++v) {
    const unsigned part = partition[v];
    bool boundary = false;
    const unsigned edge_begin = graph.adjp[v];
    const unsigned edge_end = graph.adjp[v + 1];
    for (unsigned e = edge_begin; e < edge_end; ++e) {
      const unsigned neighbor_raw = graph.adjncy[e];
      if (neighbor_raw == 0 || neighbor_raw > graph.vwgt.size()) {
        continue;
      }

      const unsigned neighbor = neighbor_raw - 1u;
      if (partition[neighbor] != part) {
        boundary = true;
        cut_twice += static_cast<unsigned long long>(graph.adjwgt[e]);
      }
    }
    if (if_boundary != nullptr && boundary) {
      (*if_boundary)[v] = 1u;
    }
  }

  return static_cast<unsigned>(cut_twice / 2ull);
}

std::vector<CandidateHost> build_refinement_candidates_host(const HostCoarseGraph& graph,
                                                            const std::vector<unsigned>& partition,
                                                            const std::vector<unsigned>& partition_wgt,
                                                            unsigned long long partition_wgt_cap,
                                                            int partition_count) {
  std::vector<CandidateHost> candidates(graph.vwgt.size());
  for (std::size_t v = 0; v < graph.vwgt.size(); ++v) {
    const unsigned vertex_partition = partition[v];
    const unsigned edge_begin = graph.adjp[v];
    const unsigned edge_end = graph.adjp[v + 1];

    int internal_weight = 0;
    std::unordered_map<unsigned, int> external_by_partition;
    bool is_boundary = false;

    for (unsigned e = edge_begin; e < edge_end; ++e) {
      const unsigned neighbor_raw = graph.adjncy[e];
      if (neighbor_raw == 0 || neighbor_raw > graph.vwgt.size()) {
        continue;
      }

      const unsigned neighbor = neighbor_raw - 1u;
      const unsigned neighbor_partition = partition[neighbor];
      const int edge_wgt = static_cast<int>(graph.adjwgt[e]);
      if (neighbor_partition == vertex_partition) {
        internal_weight += edge_wgt;
      } else {
        is_boundary = true;
        external_by_partition[neighbor_partition] += edge_wgt;
      }
    }

    int best_gain = 0;
    unsigned best_target = vertex_partition;
    for (int p = 0; p < partition_count; ++p) {
      const unsigned candidate_partition = static_cast<unsigned>(p);
      if (candidate_partition == vertex_partition) {
        continue;
      }

      auto it = external_by_partition.find(candidate_partition);
      if (it == external_by_partition.end()) {
        continue;
      }

      const int gain = it->second - internal_weight;
      if (gain > best_gain) {
        best_gain = gain;
        best_target = candidate_partition;
      }
    }

    const unsigned vertex_wgt = graph.vwgt[v];
    const bool within_cap =
        static_cast<unsigned long long>(partition_wgt[best_target]) + static_cast<unsigned long long>(vertex_wgt) <=
        partition_wgt_cap;

    candidates[v].target_partition = best_target;
    candidates[v].gain = best_gain;
    candidates[v].is_boundary = is_boundary;
    candidates[v].movable =
        is_boundary && (best_target != vertex_partition) && (best_gain > 0) && within_cap;
  }

  return candidates;
}

std::vector<MoveRequestHost> build_independent_move_buffer_host(const HostCoarseGraph& graph,
                                                                const std::vector<unsigned>& partition,
                                                                const std::vector<CandidateHost>& candidates,
                                                                const std::vector<unsigned>& partition_wgt,
                                                                unsigned long long partition_wgt_cap) {
  std::vector<MoveRequestHost> buffer;
  buffer.reserve(graph.vwgt.size());

  for (std::size_t v = 0; v < graph.vwgt.size(); ++v) {
    if (!candidates[v].movable) {
      continue;
    }

    bool independent = true;
    const unsigned edge_begin = graph.adjp[v];
    const unsigned edge_end = graph.adjp[v + 1];
    for (unsigned e = edge_begin; e < edge_end; ++e) {
      const unsigned neighbor_raw = graph.adjncy[e];
      if (neighbor_raw == 0 || neighbor_raw > graph.vwgt.size()) {
        continue;
      }

      const unsigned neighbor = neighbor_raw - 1u;
      if (!candidates[neighbor].movable) {
        continue;
      }

      const unsigned n_target = candidates[neighbor].target_partition;
      const unsigned n_wgt = graph.vwgt[neighbor];
      const bool n_within_cap = static_cast<unsigned long long>(partition_wgt[n_target]) +
                                    static_cast<unsigned long long>(n_wgt) <=
                                partition_wgt_cap;
      if (n_within_cap && (v + 1u) > (neighbor + 1u)) {
        independent = false;
        break;
      }
    }

    if (!independent) {
      continue;
    }

    MoveRequestHost mv;
    mv.vertex_id = static_cast<unsigned>(v + 1u);
    mv.source_partition = partition[v];
    mv.des_partition = candidates[v].target_partition;
    mv.gain = candidates[v].gain;
    buffer.push_back(mv);
  }

  std::sort(buffer.begin(), buffer.end(), [](const MoveRequestHost& lhs, const MoveRequestHost& rhs) {
    if (lhs.gain == rhs.gain) {
      return lhs.vertex_id < rhs.vertex_id;
    }
    return lhs.gain > rhs.gain;
  });

  return buffer;
}

int find_max_balance_prefix_host(const std::vector<MoveRequestHost>& buffer,
                                 const std::vector<unsigned>& vwgt,
                                 const std::vector<unsigned>& partition_wgt,
                                 unsigned long long partition_wgt_cap,
                                 int partition_count) {
  if (buffer.empty()) {
    return -1;
  }

  std::vector<std::int64_t> delta(static_cast<std::size_t>(partition_count), 0);
  int best = -1;
  for (std::size_t i = 0; i < buffer.size(); ++i) {
    const MoveRequestHost& mv = buffer[i];
    const unsigned vertex_wgt = vwgt[mv.vertex_id - 1u];
    delta[mv.source_partition] -= static_cast<std::int64_t>(vertex_wgt);
    delta[mv.des_partition] += static_cast<std::int64_t>(vertex_wgt);

    bool balance_ok = true;
    for (int p = 0; p < partition_count; ++p) {
      const std::uint64_t new_wgt = static_cast<std::uint64_t>(
          static_cast<std::int64_t>(partition_wgt[static_cast<std::size_t>(p)]) + delta[static_cast<std::size_t>(p)]);
      if (new_wgt > partition_wgt_cap) {
        balance_ok = false;
        break;
      }
    }

    if (balance_ok) {
      best = static_cast<int>(i);
    }
  }

  return best;
}

unsigned refine_partition_host(const HostCoarseGraph& graph,
                               std::vector<unsigned>& partition,
                               int partition_count,
                               unsigned long long partition_wgt_cap,
                               int max_iterations) {
  unsigned total_moves = 0;
  int iteration = 0;

  while (true) {
    if (max_iterations > 0 && iteration >= max_iterations) {
      break;
    }
    ++iteration;

    const std::vector<unsigned> partition_wgt =
        compute_partition_weights_host(partition, graph.vwgt, partition_count);
    const std::vector<CandidateHost> candidates =
        build_refinement_candidates_host(graph, partition, partition_wgt, partition_wgt_cap, partition_count);
    std::vector<MoveRequestHost> buffer =
        build_independent_move_buffer_host(graph, partition, candidates, partition_wgt, partition_wgt_cap);

    if (buffer.empty()) {
      break;
    }

    const int max_prefix =
        find_max_balance_prefix_host(buffer, graph.vwgt, partition_wgt, partition_wgt_cap, partition_count);
    if (max_prefix < 0) {
      break;
    }

    for (int i = 0; i <= max_prefix; ++i) {
      const MoveRequestHost& mv = buffer[static_cast<std::size_t>(i)];
      partition[mv.vertex_id - 1u] = mv.des_partition;
      ++total_moves;
    }
  }

  return total_moves;
}

template <typename GraphLike>
unsigned choose_heaviest_neighbor(const GraphLike& graph, std::size_t vertex) {
  const unsigned edge_begin = graph.adjp[vertex];
  const unsigned edge_end = graph.adjp[vertex + 1];
  const unsigned num_vertices = static_cast<unsigned>(graph.vwgt.size());
  unsigned best_neighbor = static_cast<unsigned>(vertex);
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
  std::vector<unsigned long long> group_key(num_vertices, 0ull);
  std::vector<unsigned> vertex_id(num_vertices, 0u);

  for (std::size_t vertex = 0; vertex < num_vertices; ++vertex) {
    candidate[vertex] = choose_heaviest_neighbor(graph, vertex) + 1u;
    group_key[vertex] = (static_cast<unsigned long long>(vertex + 1u) << 32);
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
    std::vector<unsigned long long> next_group_key = group_key;
    for (std::size_t vertex = 0; vertex < num_vertices; ++vertex) {
      const unsigned partner = candidate[vertex];
      if (partner == 0u || partner > num_vertices) {
        continue;
      }

      const unsigned current_group = get_first(group_key[vertex]);
      const unsigned neighbor_group = get_first(group_key[partner - 1u]);

      if (current_group > neighbor_group) {
        const unsigned prior = get_first(next_group_key[partner - 1u]);
        if (current_group > prior) {
          next_group_key[partner - 1u] = make_combo(current_group, iteration);
        }
        group_changed = true;
      } else if (current_group < neighbor_group) {
        const unsigned prior = get_first(next_group_key[vertex]);
        if (neighbor_group > prior) {
          next_group_key[vertex] = make_combo(neighbor_group, iteration);
        }
        group_changed = true;
      }
    }

    group_key.swap(next_group_key);

    ++iteration;
  }

  // Mirror CUDA constraint_group_size + construct_cmap flow exactly.
  std::vector<unsigned> group_head(num_vertices, 0u);
  unsigned exclusive = 0u;
  for (std::size_t i = 0; i < num_vertices; ++i) {
    const unsigned is_group_head = (get_first(group_key[i]) == vertex_id[i]) ? 1u : 0u;
    group_head[i] = exclusive;
    exclusive += is_group_head;
  }
  const unsigned num_coarse_vertices = exclusive;

  std::vector<std::pair<unsigned long long, unsigned>> sorted;
  sorted.reserve(num_vertices);
  for (std::size_t i = 0; i < num_vertices; ++i) {
    sorted.emplace_back(group_key[i], vertex_id[i]);
  }
  std::stable_sort(sorted.begin(), sorted.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.first < rhs.first;
  });

  for (std::size_t i = 0; i < num_vertices; ++i) {
    group_key[i] = sorted[i].first;
    vertex_id[i] = sorted[i].second;
  }

  std::vector<unsigned> group_ptr(num_coarse_vertices + 1u, static_cast<unsigned>(num_vertices));
  if (num_coarse_vertices <= num_vertices) {
    group_ptr[num_coarse_vertices] = static_cast<unsigned>(num_vertices);
  }

  for (std::size_t gid = 0; gid < num_vertices; ++gid) {
    const unsigned root = get_first(group_key[gid]);
    if (root == vertex_id[gid]) {
      const unsigned new_idx = group_head[root - 1u];
      group_ptr[new_idx] = static_cast<unsigned>(gid);
    }
  }

  for (std::size_t gid = 0; gid < num_vertices; ++gid) {
    const unsigned root = get_first(group_key[gid]);
    const unsigned group_idx = group_head[root - 1u];
    const unsigned left_in_group = static_cast<unsigned>(gid) - group_ptr[group_idx];
    const unsigned subgroup_id = left_in_group / group_cap;
    const unsigned subgroup_head_pos = group_ptr[group_idx] + subgroup_id * group_cap;
    const unsigned subgroup_head_vertex = vertex_id[subgroup_head_pos];
    group_key[gid] = make_combo(subgroup_head_vertex, 0u);
  }

  std::vector<unsigned> cmap(num_vertices, 0u);
  unsigned inclusive = 0u;
  for (std::size_t gid = 0; gid < num_vertices; ++gid) {
    if (get_first(group_key[gid]) == vertex_id[gid]) {
      ++inclusive;
    }
    cmap[vertex_id[gid] - 1u] = inclusive;
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
  std::vector<HostCoarseGraph> level_graphs;
  level_graphs.push_back(coarse_graph);

  const unsigned partition_count_u = config.num_partitions > 0 ? static_cast<unsigned>(config.num_partitions) : 1u;
  const unsigned coarsen_threshold = std::max(2u, 20u * partition_count_u);

  while (coarse_graph.vwgt.size() > coarsen_threshold) {
    const std::size_t fine_vertices = coarse_graph.vwgt.size();
    const std::size_t fine_edges = coarse_graph.adjncy.size();
    const std::vector<unsigned> cmap = build_matching_cmap(coarse_graph, config.max_coarsen_group);
    HostCoarseGraph next_graph = build_coarse_graph(coarse_graph, cmap);
    const auto max_cmap_it = std::max_element(cmap.begin(), cmap.end());
    const unsigned reported_coarse_vertices =
        (max_cmap_it != cmap.end()) ? *max_cmap_it : 0u;
    std::cout << "[kokkos_port][coarsen] fine_vertices=" << fine_vertices
              << ", fine_edges=" << fine_edges
              << ", coarse_vertices=" << next_graph.vwgt.size()
              << ", coarse_edges=" << next_graph.adjncy.size()
              << ", cmap_max=" << reported_coarse_vertices << "\n";
    if (next_graph.vwgt.size() == coarse_graph.vwgt.size() ||
        next_graph.vwgt.size() < static_cast<std::size_t>(partition_count_u + 1u)) {
      break;
    }

    level_cmaps.push_back(std::move(cmap));
    level_graphs.push_back(next_graph);
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
        : static_cast<unsigned long long>(
          (static_cast<double>(total_coarse_wgt) / static_cast<double>(partition_count)) * 1.03 + 1.0);

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

  std::vector<unsigned> pre_refine_partition;
  std::vector<unsigned> last_pass_moves_per_level;
  unsigned long long proposed_moves = 0;
  int executed_refinement_passes = 0;
  const int refinement_pass_limit = options.refinement_passes;

  for (std::size_t level = level_cmaps.size(); level > 0; --level) {
    const std::vector<unsigned>& cmap = level_cmaps[level - 1u];
    std::vector<unsigned> finer_partition(cmap.size());
    for (std::size_t vertex = 0; vertex < cmap.size(); ++vertex) {
      finer_partition[vertex] = projected_partition[cmap[vertex] - 1u];
    }

    if (level == 1u) {
      pre_refine_partition = finer_partition;
    }

    const HostCoarseGraph& finer_graph = level_graphs[level - 1u];
    const unsigned moves_this_level =
        refine_partition_host(finer_graph, finer_partition, partition_count, partition_wgt_cap, refinement_pass_limit);
    proposed_moves += static_cast<unsigned long long>(moves_this_level);
    last_pass_moves_per_level.push_back(moves_this_level);
    if (moves_this_level > 0) {
      ++executed_refinement_passes;
    }

    projected_partition = std::move(finer_partition);
  }

  if (pre_refine_partition.empty()) {
    pre_refine_partition = projected_partition;
  }

  Kokkos::View<unsigned*, Kokkos::HostSpace> h_pre_refine_partition(pre_refine_partition.data(), pre_refine_partition.size());
  write_outputs(options.out_prefix + ".pre_refine", h_pre_refine_partition);

  std::vector<unsigned> final_partition_wgt =
      compute_partition_weights_host(projected_partition, host_graph.vwgt, partition_count);
  std::vector<unsigned> if_boundary_host;
  const unsigned final_cutsize = compute_cutsize_host(level_graphs.front(), projected_partition, &if_boundary_host);

  auto h_final_partition = Kokkos::create_mirror_view(state.partition);
  auto h_final_partition_wgt = Kokkos::create_mirror_view(state.partition_wgt);
  auto h_cutsize = Kokkos::create_mirror_view(state.cutsize);
  for (std::size_t i = 0; i < projected_partition.size(); ++i) {
    h_final_partition(i) = projected_partition[i];
  }
  for (std::size_t p = 0; p < final_partition_wgt.size(); ++p) {
    h_final_partition_wgt(p) = final_partition_wgt[p];
  }
  h_cutsize(0) = final_cutsize;

  Kokkos::deep_copy(state.partition, h_final_partition);
  Kokkos::deep_copy(state.partition_wgt, h_final_partition_wgt);
  Kokkos::deep_copy(state.cutsize, h_cutsize);

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
            << ", refinement_passes=" << refinement_pass_limit
            << ", executed_refinement_passes=" << executed_refinement_passes
            << ", last_pass_moves="
            << (last_pass_moves_per_level.empty() ? 0ull : static_cast<unsigned long long>(last_pass_moves_per_level.back()))
            << ", min_partition_wgt=" << min_partition_wgt
            << ", avg_partition_wgt=" << avg_partition_wgt
            << ", partition_wgt_cap=" << partition_wgt_cap << "\n";
}

template void run_pipeline_stub<Kokkos::DefaultExecutionSpace>(const RunOptions& options,
                                                               const PartitionConfig& config);

}  // namespace gkway_kokkos
