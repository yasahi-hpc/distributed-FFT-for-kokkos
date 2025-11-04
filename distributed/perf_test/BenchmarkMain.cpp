#include <benchmark/benchmark.h>
#include "Benchmark_Context.hpp"
#include <Kokkos_Core.hpp>

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    benchmark::Initialize(&argc, argv);
    benchmark::SetDefaultTimeUnit(benchmark::kSecond);
    KokkosFFT::Distributed::Benchmark::add_benchmark_context(true);
    ::benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
  }
  Kokkos::finalize();
  return 0;
}
