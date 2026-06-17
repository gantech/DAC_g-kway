#pragma once

#include "gkway_kokkos/data_model.hpp"

#include <string>

namespace gkway_kokkos {

struct RunOptions {
  std::string graph_file;
  int num_partitions = 0;
  std::string out_prefix;
};

template <typename ExecSpace>
void run_pipeline_stub(const RunOptions& options, const PartitionConfig& config);

}  // namespace gkway_kokkos
