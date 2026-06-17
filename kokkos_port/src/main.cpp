#include "gkway_kokkos/pipeline.hpp"

#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    if (argc != 4) {
      std::cerr << "usage: ./gkway-kokkos graph_file num_partition out_prefix\n";
      Kokkos::finalize();
      return 1;
    }

    gkway_kokkos::RunOptions options;
    options.graph_file = argv[1];
    options.num_partitions = std::atoi(argv[2]);
    options.out_prefix = argv[3];

    gkway_kokkos::PartitionConfig config;
    config.num_partitions = options.num_partitions;

    gkway_kokkos::run_pipeline_stub<Kokkos::DefaultExecutionSpace>(options, config);
  }
  Kokkos::finalize();
  return 0;
}
