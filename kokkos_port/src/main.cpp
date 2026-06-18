#include "gkway_kokkos/pipeline.hpp"

#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    if (argc != 4) {
      std::cerr << "usage: ./gkway-kokkos graph_file num_partition out_prefix\n";
      std::cerr << "  env GKWAY_REFINEMENT_PASSES=<n>  override refinement pass limit (default 1)\n";
      std::cerr << "  env GKWAY_SAME_START_FILE=<path>  load pre-computed L0 partition for debug\n";
      Kokkos::finalize();
      return 1;
    }

    gkway_kokkos::RunOptions options;
    options.graph_file = argv[1];
    options.num_partitions = std::atoi(argv[2]);
    options.out_prefix = argv[3];

    // Optional tuning/debug via environment variables so the CLI stays
    // identical to the CUDA binary (graph_file num_partition out_prefix).
    if (const char* env_passes = std::getenv("GKWAY_REFINEMENT_PASSES")) {
      options.refinement_passes = std::atoi(env_passes);
    }
    if (const char* env_ss = std::getenv("GKWAY_SAME_START_FILE")) {
      options.same_start_file = env_ss;
    }

    if (options.refinement_passes < 1) {
      std::cerr << "GKWAY_REFINEMENT_PASSES must be >= 1\n";
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
