#include "gkway_kokkos/pipeline.hpp"

#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    if (argc != 4 && argc != 5) {
      std::cerr << "usage: ./gkway-kokkos graph_file num_partition out_prefix [refinement_passes]\n";
      Kokkos::finalize();
      return 1;
    }

    gkway_kokkos::RunOptions options;
    options.graph_file = argv[1];
    options.num_partitions = std::atoi(argv[2]);
    options.out_prefix = argv[3];
    if (argc == 5) {
      options.refinement_passes = std::atoi(argv[4]);
    }

    if (options.refinement_passes < 1) {
      std::cerr << "refinement_passes must be >= 1\n";
      Kokkos::finalize();
      return 1;
    }

    gkway_kokkos::PartitionConfig config;
    config.num_partitions = options.num_partitions;

    gkway_kokkos::run_pipeline_stub<Kokkos::DefaultExecutionSpace>(options, config);
  }
  Kokkos::finalize();
  return 0;
}
