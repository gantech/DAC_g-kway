#pragma once

#include "gkway_kokkos/data_model.hpp"

#include <string>

namespace gkway_kokkos {

struct RunOptions {
  std::string graph_file;
  int num_partitions = 0;
  std::string out_prefix;
  int refinement_passes = 1;
  // When non-empty: skip coarsening + METIS, load this pre-computed partition
  // for the L0 graph, and run same-start refinement debug mode.
  std::string same_start_file;
};

template <typename ExecSpace>
void run_pipeline_stub(const RunOptions& options, const PartitionConfig& config);

}  // namespace gkway_kokkos
