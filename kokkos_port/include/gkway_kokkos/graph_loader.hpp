#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace gkway_kokkos {

struct HostGraph {
  std::size_t num_vertices = 0;
  std::size_t num_edges = 0;
  std::vector<unsigned> adjp;
  std::vector<unsigned> adjncy;
  std::vector<unsigned> adjwgt;
  std::vector<unsigned> vwgt;
};

HostGraph load_graph_from_metis_like(const std::string& input_path);

}  // namespace gkway_kokkos
