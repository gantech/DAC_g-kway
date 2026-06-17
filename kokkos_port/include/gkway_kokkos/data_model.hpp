#pragma once

#include <Kokkos_Core.hpp>
#include <cstddef>

namespace gkway_kokkos {

struct PartitionConfig {
  int num_partitions = 0;
  unsigned max_coarsen_group = 6;
  float max_partition_wgt = 0.0f;
};

template <typename ExecSpace>
struct GraphLevel {
  using memory_space = typename ExecSpace::memory_space;
  using view_u = Kokkos::View<unsigned*, memory_space>;

  view_u adjp;
  view_u adjncy;
  view_u adjwgt;
  view_u vwgt;
  view_u cmap;
  std::size_t num_vertices = 0;
  std::size_t num_edges = 0;
};

template <typename ExecSpace>
struct PartitionState {
  using memory_space = typename ExecSpace::memory_space;
  using view_u = Kokkos::View<unsigned*, memory_space>;

  view_u partition;
  view_u partition_wgt;
  view_u if_boundary;
  view_u cutsize;
};

}  // namespace gkway_kokkos
