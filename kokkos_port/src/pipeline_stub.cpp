#include "gkway_kokkos/pipeline.hpp"
#include "gkway_kokkos/graph_loader.hpp"

#include "metis/metis.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

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

struct MoveRequestDevice {
  unsigned vertex_id = 0;
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

    std::uint32_t best_gain_u = 0u;
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

      const std::uint32_t gain_u =
          static_cast<std::uint32_t>(it->second) - static_cast<std::uint32_t>(internal_weight);
      const int gain = static_cast<int>(gain_u);
      if (gain_u > best_gain_u) {
        best_gain_u = gain_u;
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

template <typename ExecSpace>
void build_refinement_candidates_device(
    const Kokkos::View<unsigned*, typename ExecSpace::memory_space>& adjp,
    const Kokkos::View<unsigned*, typename ExecSpace::memory_space>& adjncy,
    const Kokkos::View<unsigned*, typename ExecSpace::memory_space>& adjwgt,
    const Kokkos::View<unsigned*, typename ExecSpace::memory_space>& vwgt,
    const Kokkos::View<unsigned*, typename ExecSpace::memory_space>& partition,
    const Kokkos::View<unsigned*, typename ExecSpace::memory_space>& partition_wgt,
    const Kokkos::View<unsigned*, typename ExecSpace::memory_space>& if_updated,
    unsigned long long partition_wgt_cap,
    int partition_count,
    Kokkos::View<unsigned*, typename ExecSpace::memory_space>& target_partition,
    Kokkos::View<int*, typename ExecSpace::memory_space>& gain,
    Kokkos::View<unsigned*, typename ExecSpace::memory_space>& is_boundary,
    Kokkos::View<unsigned*, typename ExecSpace::memory_space>& movable,
    Kokkos::View<unsigned*, typename ExecSpace::memory_space>& external_weights) {
  const std::size_t num_vertices = vwgt.extent(0);
  Kokkos::parallel_for(
      "build_refinement_candidates_device",
      Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(num_vertices)),
      KOKKOS_LAMBDA(const int v) {
        if (if_updated(v) == 0u) {
          return;
        }
        if_updated(v) = 0u;

        const std::size_t row_base = static_cast<std::size_t>(v) * static_cast<std::size_t>(partition_count);
        for (int p = 0; p < partition_count; ++p) {
          external_weights(row_base + static_cast<std::size_t>(p)) = 0;
        }

        const unsigned vertex_partition = partition(v);
        const unsigned edge_begin = adjp(v);
        const unsigned edge_end = adjp(v + 1);

        unsigned internal_weight = 0u;
        bool boundary = false;
        for (unsigned e = edge_begin; e < edge_end; ++e) {
          const unsigned neighbor_raw = adjncy(e);
          if (neighbor_raw == 0 || neighbor_raw > num_vertices) {
            continue;
          }

          const unsigned neighbor = neighbor_raw - 1u;
          const unsigned neighbor_partition = partition(neighbor);
          const unsigned edge_wgt = adjwgt(e);
          if (neighbor_partition == vertex_partition) {
            internal_weight += edge_wgt;
          } else {
            boundary = true;
            external_weights(row_base + static_cast<std::size_t>(neighbor_partition)) += edge_wgt;
          }
        }

        std::uint32_t best_gain_u = 0u;
        int best_gain = 0;
        unsigned best_target = vertex_partition;
        for (int p = 0; p < partition_count; ++p) {
          const unsigned candidate_partition = static_cast<unsigned>(p);
          if (candidate_partition == vertex_partition) {
            continue;
          }

          const unsigned external_weight = external_weights(row_base + static_cast<std::size_t>(p));
          if (external_weight == 0u) {
            continue;
          }

          const std::uint32_t gain_u = static_cast<std::uint32_t>(external_weight - internal_weight);
          if (gain_u > best_gain_u) {
            best_gain_u = gain_u;
            best_gain = static_cast<int>(gain_u);
            best_target = candidate_partition;
          }
        }

        const unsigned vertex_wgt = vwgt(v);
        const bool within_cap = static_cast<unsigned long long>(partition_wgt(best_target)) +
                                    static_cast<unsigned long long>(vertex_wgt) <=
                                partition_wgt_cap;

        target_partition(v) = best_target;
        gain(v) = best_gain;
        is_boundary(v) = boundary ? 1u : 0u;
        movable(v) = (boundary && best_target != vertex_partition && best_gain > 0 && within_cap) ? 1u : 0u;
      });
  ExecSpace().fence();
}

template <typename ExecSpace>
int find_balance_prefix_device(
    const Kokkos::View<MoveRequestDevice*, typename ExecSpace::memory_space>& buffer,
    std::size_t buffer_size,
    const Kokkos::View<unsigned*, typename ExecSpace::memory_space>& vwgt,
    const Kokkos::View<unsigned*, typename ExecSpace::memory_space>& partition_wgt,
    unsigned long long partition_wgt_cap,
    int partition_count,
    Kokkos::View<int*, typename ExecSpace::memory_space>& delta,
    Kokkos::View<unsigned*, typename ExecSpace::memory_space>& balance_sequence,
    Kokkos::View<int*, typename ExecSpace::memory_space>& op_result) {
  Kokkos::deep_copy(delta, 0);
  Kokkos::deep_copy(balance_sequence, 0u);
  // Mirror CUDA refinement behavior where d_op_result is memset to 0.
  Kokkos::deep_copy(op_result, 0);

  const std::size_t prefix_limit = buffer_size;
  Kokkos::parallel_for(
      "find_mv_wgt_sequence_device",
      Kokkos::RangePolicy<ExecSpace>(0, partition_count),
      KOKKOS_LAMBDA(const int p) {
        int running = 0;
        for (std::size_t i = 0; i < prefix_limit; ++i) {
          const MoveRequestDevice mv = buffer(i);
          const unsigned vertex_wgt = vwgt(mv.vertex_id - 1u);
          if (static_cast<int>(mv.source_partition) == p) {
            running -= static_cast<int>(vertex_wgt);
          }
          if (static_cast<int>(mv.des_partition) == p) {
            running += static_cast<int>(vertex_wgt);
          }
          delta(static_cast<std::size_t>(p) * prefix_limit + i) = running;
        }
      });

  Kokkos::parallel_for(
      "find_balance_sequence_device",
      Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(prefix_limit)),
      KOKKOS_LAMBDA(const int gid) {
        int if_balance = 1;
        for (int p = 0; p < partition_count; ++p) {
          const int delta_partition_wgt = delta(static_cast<std::size_t>(p) * prefix_limit + static_cast<std::size_t>(gid));
          if (delta_partition_wgt > 0) {
            const unsigned new_partition_wgt =
                partition_wgt(static_cast<std::size_t>(p)) + static_cast<unsigned>(delta_partition_wgt);
            if (new_partition_wgt > partition_wgt_cap) {
              if_balance = 0;
            }
          }
        }
        balance_sequence(gid) = static_cast<unsigned>(if_balance);
        if (if_balance == 1) {
          Kokkos::atomic_fetch_max(&op_result(0), gid);
        }
      });

  ExecSpace().fence();

  const auto h_op_result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), op_result);
  return h_op_result(0);
}

template <typename ExecSpace>
void apply_move_prefix_device(
    const Kokkos::View<MoveRequestDevice*, typename ExecSpace::memory_space>& buffer,
    std::size_t buffer_size,
    int pos,
    Kokkos::View<int*, typename ExecSpace::memory_space>& delta,
    Kokkos::View<unsigned*, typename ExecSpace::memory_space>& partition,
    Kokkos::View<unsigned*, typename ExecSpace::memory_space>& partition_wgt,
    Kokkos::View<unsigned*, typename ExecSpace::memory_space>& if_updated,
    const Kokkos::View<unsigned*, typename ExecSpace::memory_space>& adjp,
    const Kokkos::View<unsigned*, typename ExecSpace::memory_space>& adjncy,
    Kokkos::View<unsigned*, typename ExecSpace::memory_space>& cutsize) {
  Kokkos::parallel_for(
      "apply_partition_weight_prefix",
      Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(partition_wgt.extent(0))),
      KOKKOS_LAMBDA(const int p) {
        const int delta_partition_wgt = delta(static_cast<std::size_t>(p) * buffer_size + static_cast<std::size_t>(pos));
        partition_wgt(p) += delta_partition_wgt;
      });

  Kokkos::parallel_for(
      "apply_sequence_move_device",
      Kokkos::RangePolicy<ExecSpace>(0, pos + 1),
      KOKKOS_LAMBDA(const int gid) {
        const MoveRequestDevice mv = buffer(static_cast<std::size_t>(gid));
        const unsigned vertex = mv.vertex_id - 1u;
        partition(vertex) = mv.des_partition;
        if_updated(vertex) = 1u;
        Kokkos::atomic_fetch_sub(&cutsize(0), static_cast<unsigned>(mv.gain));

        const unsigned start = adjp(vertex);
        const unsigned end = adjp(vertex + 1u);
        for (unsigned e = start; e < end; ++e) {
          const unsigned neighbor_raw = adjncy(e);
          if (neighbor_raw == 0u) {
            continue;
          }
          if_updated(neighbor_raw - 1u) = 1u;
        }
      });
}

template <typename ExecSpace>
std::size_t build_independent_move_buffer_device(
    const Kokkos::View<unsigned*, typename ExecSpace::memory_space>& adjp,
    const Kokkos::View<unsigned*, typename ExecSpace::memory_space>& adjncy,
    const Kokkos::View<unsigned*, typename ExecSpace::memory_space>& vwgt,
    const Kokkos::View<unsigned*, typename ExecSpace::memory_space>& partition,
    const Kokkos::View<unsigned*, typename ExecSpace::memory_space>& partition_wgt,
    unsigned long long partition_wgt_cap,
    const Kokkos::View<unsigned*, typename ExecSpace::memory_space>& target_partition,
    const Kokkos::View<int*, typename ExecSpace::memory_space>& gain,
    const Kokkos::View<unsigned*, typename ExecSpace::memory_space>& is_boundary,
    Kokkos::View<unsigned*, typename ExecSpace::memory_space>& if_updated,
    Kokkos::View<MoveRequestDevice*, typename ExecSpace::memory_space>& buffer,
    Kokkos::View<unsigned long long*, typename ExecSpace::memory_space>& sort_keys,
    Kokkos::View<unsigned*, typename ExecSpace::memory_space>& sort_indices,
    Kokkos::View<MoveRequestDevice*, typename ExecSpace::memory_space>& sorted_buffer) {
  using memory_space = typename ExecSpace::memory_space;
  Kokkos::View<unsigned*, memory_space> buffer_size("refine_buffer_size", 1);
  Kokkos::deep_copy(buffer_size, 0u);
  Kokkos::deep_copy(sort_keys, std::numeric_limits<unsigned long long>::max());

  const std::size_t num_vertices = vwgt.extent(0);
  Kokkos::parallel_for(
      "build_independent_move_buffer_device",
      Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(num_vertices)),
      KOKKOS_LAMBDA(const int v) {
        // Match CUDA create_independent_move_buffer: clear update flag for all vertices.
        if_updated(v) = 0u;
        if (is_boundary(v) == 0u) {
          return;
        }
        const unsigned vertex_partition = partition(v);
        const unsigned vertex_wgt = vwgt(v);
        const unsigned current_target = target_partition(v);
        const int current_gain = gain(v);
        const bool within_cap = static_cast<unsigned long long>(partition_wgt(current_target)) +
                                    static_cast<unsigned long long>(vertex_wgt) <=
                                partition_wgt_cap;
        if (current_gain <= 0 || !within_cap) {
          return;
        }

        bool independent = true;
        const unsigned edge_begin = adjp(v);
        const unsigned edge_end = adjp(v + 1);
        for (unsigned e = edge_begin; e < edge_end; ++e) {
          const unsigned neighbor_raw = adjncy(e);
          if (neighbor_raw == 0 || neighbor_raw > num_vertices) {
            continue;
          }

          const unsigned neighbor = neighbor_raw - 1u;
          const unsigned neighbor_partition = partition(neighbor);
          const unsigned neighbor_target = target_partition(neighbor);
          const int neighbor_gain = gain(neighbor);
          const unsigned neighbor_wgt = vwgt(neighbor);
          const bool neighbor_within_cap = static_cast<unsigned long long>(partition_wgt(neighbor_target)) +
                                               static_cast<unsigned long long>(neighbor_wgt) <=
                                           partition_wgt_cap;
          if (neighbor_gain <= 0 || !neighbor_within_cap) {
            continue;
          }
          if (neighbor_within_cap && (v + 1) > static_cast<int>(neighbor + 1u)) {
            independent = false;
            break;
          }
        }

        if (!independent) {
          return;
        }

        const unsigned pos = Kokkos::atomic_fetch_add(&buffer_size(0), 1u);
        const unsigned vertex_id = static_cast<unsigned>(v + 1u);
        buffer(pos) = MoveRequestDevice{vertex_id, partition(v), target_partition(v), gain(v)};
        const std::uint32_t gain_u = static_cast<std::uint32_t>(gain(v));
        sort_keys(pos) = (static_cast<unsigned long long>(std::numeric_limits<std::uint32_t>::max() - gain_u) << 32) |
                         static_cast<unsigned long long>(vertex_id);
        sort_indices(pos) = pos;
      });
  ExecSpace().fence();

  const auto h_buffer_size = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), buffer_size);
  const std::size_t current_size = static_cast<std::size_t>(h_buffer_size(0));
  if (current_size > 1u) {
    Kokkos::Experimental::sort_by_key(ExecSpace{}, sort_keys, sort_indices);
    Kokkos::parallel_for(
        "permute_sorted_move_buffer",
        Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(current_size)),
        KOKKOS_LAMBDA(const int i) { sorted_buffer(i) = buffer(sort_indices(i)); });
    ExecSpace().fence();
  } else if (current_size == 1u) {
    Kokkos::deep_copy(sorted_buffer, buffer);
  }

  return current_size;
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
  // Mirror CUDA behavior where op_result is memset to 0 before searching.
  int best = 0;
  for (std::size_t i = 0; i < buffer.size(); ++i) {
    const MoveRequestHost& mv = buffer[i];
    const unsigned vertex_wgt = vwgt[mv.vertex_id - 1u];
    delta[mv.source_partition] -= static_cast<std::int64_t>(vertex_wgt);
    delta[mv.des_partition] += static_cast<std::int64_t>(vertex_wgt);

    bool balance_ok = true;
    for (int p = 0; p < partition_count; ++p) {
      const std::int64_t d = delta[static_cast<std::size_t>(p)];
      if (d <= 0) {
        continue;
      }

      const std::int64_t new_wgt =
          static_cast<std::int64_t>(partition_wgt[static_cast<std::size_t>(p)]) + d;
      if (new_wgt > static_cast<std::int64_t>(partition_wgt_cap)) {
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

  using memory_space = typename Kokkos::DefaultExecutionSpace::memory_space;
  using view_u = Kokkos::View<unsigned*, memory_space>;
  using view_i = Kokkos::View<int*, memory_space>;
  using view_key = Kokkos::View<unsigned long long*, memory_space>;
  using view_move = Kokkos::View<MoveRequestDevice*, memory_space>;

  const std::size_t num_vertices = graph.vwgt.size();
  const std::size_t partition_slots = static_cast<std::size_t>(std::max(1, partition_count));

  view_u d_adjp("refine_adjp", graph.adjp.size());
  view_u d_adjncy("refine_adjncy", graph.adjncy.size());
  view_u d_adjwgt("refine_adjwgt", graph.adjwgt.size());
  view_u d_vwgt("refine_vwgt", graph.vwgt.size());

  {
    Kokkos::View<unsigned*, Kokkos::HostSpace> h_adjp("h_adjp", graph.adjp.size());
    Kokkos::View<unsigned*, Kokkos::HostSpace> h_adjncy("h_adjncy", graph.adjncy.size());
    Kokkos::View<unsigned*, Kokkos::HostSpace> h_adjwgt("h_adjwgt", graph.adjwgt.size());
    Kokkos::View<unsigned*, Kokkos::HostSpace> h_vwgt("h_vwgt", graph.vwgt.size());
    for (std::size_t i = 0; i < graph.adjp.size(); ++i) {
      h_adjp(i) = graph.adjp[i];
    }
    for (std::size_t i = 0; i < graph.adjncy.size(); ++i) {
      h_adjncy(i) = graph.adjncy[i];
    }
    for (std::size_t i = 0; i < graph.adjwgt.size(); ++i) {
      h_adjwgt(i) = graph.adjwgt[i];
    }
    for (std::size_t i = 0; i < graph.vwgt.size(); ++i) {
      h_vwgt(i) = graph.vwgt[i];
    }
    Kokkos::deep_copy(d_adjp, h_adjp);
    Kokkos::deep_copy(d_adjncy, h_adjncy);
    Kokkos::deep_copy(d_adjwgt, h_adjwgt);
    Kokkos::deep_copy(d_vwgt, h_vwgt);
  }

  view_u d_partition("refine_partition", num_vertices);
  view_u d_partition_wgt("refine_partition_wgt", partition_slots);
  view_u d_target_partition("refine_target_partition", num_vertices);
  view_i d_gain("refine_gain", num_vertices);
  view_u d_is_boundary("refine_is_boundary", num_vertices);
  view_u d_movable("refine_movable", num_vertices);
  view_u d_if_updated("refine_if_updated", num_vertices);
  view_u d_external_weights("refine_external_weights", num_vertices * partition_slots);
  view_move d_buffer("refine_buffer", num_vertices);
  view_key d_sort_keys("refine_sort_keys", num_vertices);
  view_u d_sort_indices("refine_sort_indices", num_vertices);
  view_move d_sorted_buffer("refine_sorted_buffer", num_vertices);
  view_i d_delta("refine_delta", partition_slots * 1024u);
  view_u d_balance_sequence("refine_balance_sequence", num_vertices);
  view_i d_op_result("refine_op_result", 1);
  view_u d_cutsize("refine_cutsize", 1);

  Kokkos::View<unsigned*, Kokkos::HostSpace> h_partition("h_partition", num_vertices);
  Kokkos::View<unsigned*, Kokkos::HostSpace> h_partition_wgt("h_partition_wgt", partition_slots);

  Kokkos::deep_copy(d_cutsize, 0u);
  Kokkos::deep_copy(d_if_updated, 1u);

  for (std::size_t i = 0; i < num_vertices; ++i) {
    h_partition(i) = partition[i];
  }
  Kokkos::deep_copy(d_partition, h_partition);

  const std::vector<unsigned> initial_partition_wgt =
      compute_partition_weights_host(partition, graph.vwgt, partition_count);
  for (std::size_t p = 0; p < partition_slots; ++p) {
    h_partition_wgt(p) = initial_partition_wgt[p];
  }
  Kokkos::deep_copy(d_partition_wgt, h_partition_wgt);

  while (true) {
    if (max_iterations > 0 && iteration >= max_iterations) {
      break;
    }
    ++iteration;

    build_refinement_candidates_device<Kokkos::DefaultExecutionSpace>(
        d_adjp,
        d_adjncy,
        d_adjwgt,
        d_vwgt,
        d_partition,
        d_partition_wgt,
        d_if_updated,
        partition_wgt_cap,
        partition_count,
        d_target_partition,
        d_gain,
        d_is_boundary,
        d_movable,
        d_external_weights);

    const std::size_t buffer_size = build_independent_move_buffer_device<Kokkos::DefaultExecutionSpace>(
        d_adjp,
        d_adjncy,
        d_vwgt,
        d_partition,
        d_partition_wgt,
        partition_wgt_cap,
        d_target_partition,
        d_gain,
        d_is_boundary,
        d_if_updated,
        d_buffer,
        d_sort_keys,
        d_sort_indices,
        d_sorted_buffer);

    if (num_vertices > 1000000u) {
      std::cout << "[kokkos_refine_dbg] vertices=" << num_vertices
                << ", iter=" << iteration
                << ", buffer_size=" << buffer_size << "\n";
    }

    if (buffer_size == 0u) {
      break;
    }

    const std::size_t buffer_limit = std::min<std::size_t>(buffer_size, 1024u);
    const int max_prefix = find_balance_prefix_device<Kokkos::DefaultExecutionSpace>(
        d_sorted_buffer,
        buffer_limit,
        d_vwgt,
        d_partition_wgt,
        partition_wgt_cap,
        partition_count,
        d_delta,
        d_balance_sequence,
        d_op_result);
    if (num_vertices > 1000000u) {
      std::cout << "[kokkos_refine_dbg] vertices=" << num_vertices
                << ", iter=" << iteration
                << ", max_prefix=" << max_prefix << "\n";
    }
    if (max_prefix < 0 || max_prefix >= static_cast<int>(buffer_limit)) {
      break;
    }

    apply_move_prefix_device<Kokkos::DefaultExecutionSpace>(
        d_sorted_buffer,
        buffer_limit,
        max_prefix,
        d_delta,
        d_partition,
        d_partition_wgt,
        d_if_updated,
        d_adjp,
        d_adjncy,
        d_cutsize);

    total_moves += static_cast<unsigned>(max_prefix + 1);
  }

  Kokkos::deep_copy(h_partition, d_partition);
  for (std::size_t i = 0; i < num_vertices; ++i) {
    partition[i] = h_partition(i);
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
  idx_t mt_options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(mt_options);
  mt_options[METIS_OPTION_NUMBERING] = 0;
  mt_options[METIS_OPTION_SEED] = 0;

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
      mt_options,
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
      refine_partition_host(finer_graph, finer_partition, partition_count, partition_wgt_cap, 0);
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
