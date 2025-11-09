#include <benchmark/benchmark.h>
#include "Benchmark_Context.hpp"
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Benchmark {

using execution_space = Kokkos::DefaultExecutionSpace;

template <typename T, typename LayoutType>
static void benchmark_fft3D(benchmark::State& state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const std::size_t n = state.range(0), py = state.range(1);

  if (size % py != 0) {
    if (rank == 0) {
      state.SkipWithError(
          "The total processes of benchmark_fft3D: " + std::to_string(size) +
          " must be divisible by " + std::to_string(py));
    }
    return;
  }
  const std::size_t px = size / py;

  if (n % px != 0 || n % py != 0) {
    if (rank == 0) {
      state.SkipWithError("The total size of benchmark_fft3D: " +
                          std::to_string(n) + " must be divisible by " +
                          std::to_string(px) + " and " + std::to_string(py));
    }
    return;
  }

  // Z-pencil or X-slab (if py == 1)
  std::array<std::size_t, 3> in_topology{px, py, 1};

  // X-pencil or Y-slab (if py == 1)
  std::array<std::size_t, 3> out_topology{1, px, py};

  using View3DType =
      Kokkos::View<Kokkos::complex<T>***, LayoutType, execution_space>;
  View3DType in("in", n / px, n / py, n), out("out", n, n / px, n / py);

  execution_space exec_space;
  KokkosFFT::Distributed::Plan plan(exec_space, in, out,
                                    KokkosFFT::axis_type<3>{0, 1, 2},
                                    in_topology, out_topology, MPI_COMM_WORLD);

  for (auto _ : state) {
    Kokkos::Timer timer;
    KokkosFFT::Distributed::execute(plan, in, out,
                                    KokkosFFT::Direction::forward);
    exec_space.fence();
    auto elapsed = timer.seconds();
    double max_elapsed_second;
    MPI_Allreduce(&elapsed, &max_elapsed_second, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    report_results(state, in, out, max_elapsed_second);
  }
}

#define BENCHMARK_FFT3D(type, layout, n_begin, n_end, py_begin, py_end) \
  BENCHMARK(benchmark_fft3D<type, Kokkos::layout>)                      \
      ->UseManualTime()                                                 \
      ->Unit(benchmark::kMillisecond)                                   \
      ->ArgNames({"N", "py"})                                           \
      ->RangeMultiplier(2)                                              \
      ->Ranges({{n_begin, n_end}, {py_begin, py_end}})

BENCHMARK_FFT3D(float, LayoutLeft, 256, 1024, 1, 16);
BENCHMARK_FFT3D(float, LayoutRight, 256, 1024, 1, 16);
BENCHMARK_FFT3D(double, LayoutLeft, 256, 1024, 1, 16);
BENCHMARK_FFT3D(double, LayoutRight, 256, 1024, 1, 16);

#undef BENCHMARK_FFT3D

}  // namespace Benchmark
}  // namespace Distributed
}  // namespace KokkosFFT
