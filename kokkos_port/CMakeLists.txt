cmake_minimum_required(VERSION 3.16)

project(gkway_kokkos_port LANGUAGES CXX)

find_package(Kokkos REQUIRED)

add_executable(gkway-kokkos
  src/main.cpp
  src/graph_loader.cpp
  src/pipeline_stub.cpp
)
target_include_directories(gkway-kokkos PRIVATE include)
target_link_libraries(gkway-kokkos PRIVATE Kokkos::kokkos)
set_target_properties(gkway-kokkos PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/exec")

add_executable(gkway-kokkos-phase0
  src/phase0_primitives.cpp
)
target_link_libraries(gkway-kokkos-phase0 PRIVATE Kokkos::kokkos)
set_target_properties(gkway-kokkos-phase0 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/exec")
