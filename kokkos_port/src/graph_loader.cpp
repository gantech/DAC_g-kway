#include "gkway_kokkos/graph_loader.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace gkway_kokkos {

HostGraph load_graph_from_metis_like(const std::string& input_path) {
  std::ifstream file(input_path);
  if (!file.is_open()) {
    throw std::runtime_error("unable to open graph input: " + input_path);
  }

  HostGraph graph;
  std::string line;
  if (!std::getline(file, line)) {
    throw std::runtime_error("empty graph input: " + input_path);
  }

  {
    std::istringstream header_stream(line);
    int format = 0;
    header_stream >> graph.num_vertices;
    std::size_t half_edges = 0;
    header_stream >> half_edges;
    graph.num_edges = half_edges * 2;
    if (!(header_stream >> format)) {
      format = 0;
    }

    graph.adjp.resize(graph.num_vertices + 1, 0);
    graph.adjncy.reserve(graph.num_edges);
    graph.adjwgt.reserve(graph.num_edges);
    graph.vwgt.assign(graph.num_vertices, 1);

    // Match CUDA parser semantics exactly: only format 11 means edge weights.
    const bool weighted_edges = (format == 11);

    std::size_t vertex_idx = 0;
    std::size_t edge_count = 0;
    while (vertex_idx < graph.num_vertices && std::getline(file, line)) {
      std::istringstream row_stream(line);
      if (!weighted_edges) {
        unsigned neighbor = 0;
        while (row_stream >> neighbor) {
          graph.adjncy.push_back(neighbor);
          graph.adjwgt.push_back(1);
          ++edge_count;
        }
      } else {
        unsigned neighbor = 0;
        unsigned weight = 0;
        while (row_stream >> neighbor >> weight) {
          graph.adjncy.push_back(neighbor);
          graph.adjwgt.push_back(weight);
          ++edge_count;
        }
      }

      graph.adjp[vertex_idx + 1] = static_cast<unsigned>(edge_count);
      ++vertex_idx;
    }

    graph.num_edges = edge_count;
  }

  return graph;
}

}  // namespace gkway_kokkos
