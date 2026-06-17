#include <Kokkos_Core.hpp>

#include <chrono>
#include <iostream>

namespace {

void benchmark_reduce(const int n) {
  Kokkos::View<int*> values("values", n);
  Kokkos::parallel_for(
      "fill_values", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) { values(i) = i % 7; });

  Kokkos::fence();
  auto start = std::chrono::high_resolution_clock::now();
  long long sum = 0;
  Kokkos::parallel_reduce(
      "reduce_values", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i, long long& local) {
        local += values(i);
      },
      sum);
  Kokkos::fence();
  auto end = std::chrono::high_resolution_clock::now();

  std::cout << "reduce,sum=" << sum << ",ms="
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << '\n';
}

void benchmark_scan(const int n) {
  Kokkos::View<int*> input("scan_input", n);
  Kokkos::View<int*> output("scan_output", n);

  Kokkos::parallel_for(
      "fill_scan_input", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) { input(i) = 1; });

  Kokkos::fence();
  auto start = std::chrono::high_resolution_clock::now();
  Kokkos::parallel_scan(
      "prefix_scan", Kokkos::RangePolicy<>(0, n),
      KOKKOS_LAMBDA(const int i, int& update, const bool final) {
        update += input(i);
        if (final) {
          output(i) = update;
        }
      });
  Kokkos::fence();
  auto end = std::chrono::high_resolution_clock::now();

  auto h_output = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output);
  std::cout << "scan,last=" << h_output(n - 1) << ",ms="
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << '\n';
}

void benchmark_compaction_like(const int n) {
  Kokkos::View<int*> flags("flags", n);
  Kokkos::View<int*> positions("positions", n);

  Kokkos::parallel_for(
      "fill_flags", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) { flags(i) = (i % 3 == 0) ? 1 : 0; });

  Kokkos::parallel_scan(
      "flags_scan", Kokkos::RangePolicy<>(0, n),
      KOKKOS_LAMBDA(const int i, int& update, const bool final) {
        update += flags(i);
        if (final) {
          positions(i) = update;
        }
      });
  Kokkos::fence();

  auto h_pos = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), positions);
  std::cout << "compaction_like,count=" << h_pos(n - 1) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const int n = (argc >= 2) ? std::atoi(argv[1]) : 1 << 22;
    benchmark_reduce(n);
    benchmark_scan(n);
    benchmark_compaction_like(n);
  }
  Kokkos::finalize();
  return 0;
}
