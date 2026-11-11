#include <exception>
#include <benchmark/benchmark.h>
#include "Benchmark_Context.hpp"
#include "Benchmark_Timing.hpp"
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Benchmark {

using execution_space = Kokkos::DefaultExecutionSpace;

template <typename T, typename LayoutType>
void benchmark_fft3D_impl(benchmark::State& state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const std::size_t n = state.range(0), py = state.range(1);

  if (size % py != 0) {
    state.SkipWithError(
        "The total processes of benchmark_fft3D: " + std::to_string(size) +
        " must be divisible by " + std::to_string(py));
    return;
  }
  const std::size_t px = size / py;

  if (n % px != 0 || n % py != 0) {
    state.SkipWithError("The total size of benchmark_fft3D: " +
                        std::to_string(n) + " must be divisible by " +
                        std::to_string(px) + " and " + std::to_string(py));
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

  // The plan that ran (native or vendor)
  state.SetLabel(plan.name());

  auto forward = [&]() {
    KokkosFFT::Distributed::execute(plan, in, out,
                                    KokkosFFT::Direction::forward);
  };
  warmup(exec_space, MPI_COMM_WORLD, forward);
  for (auto _ : state) {
    const double elapsed =
        time_max_over_ranks(exec_space, MPI_COMM_WORLD, forward);
    report_results(state, in, out, elapsed);
  }
}

/// \brief benchmark_fft3D_impl, with an error (the same on every rank:
/// plan creation, buffers) reported as the error of this benchmark, so that
/// the next benchmarks still run
template <typename T, typename LayoutType>
static void benchmark_fft3D(benchmark::State& state) {
  try {
    benchmark_fft3D_impl<T, LayoutType>(state);
  } catch (const std::exception& e) {
    state.SkipWithError(e.what());
  }
}

#define BENCHMARK_FFT3D(type, layout, n_begin, n_end, py_begin, py_end) \
  BENCHMARK(benchmark_fft3D<type, Kokkos::layout>)                      \
      ->UseManualTime()                                                 \
      ->Unit(benchmark::kMillisecond)                                   \
      ->ArgNames({"N", "py"})                                           \
      ->RangeMultiplier(2)                                              \
      ->Ranges({{n_begin, n_end}, {py_begin, py_end}})                  \
      ->ComputeStatistics("min", min_statistic)

BENCHMARK_FFT3D(float, LayoutLeft, 256, 1024, 1, 16);
BENCHMARK_FFT3D(float, LayoutRight, 256, 1024, 1, 16);
BENCHMARK_FFT3D(double, LayoutLeft, 256, 1024, 1, 16);
BENCHMARK_FFT3D(double, LayoutRight, 256, 1024, 1, 16);

#undef BENCHMARK_FFT3D

}  // namespace Benchmark
}  // namespace Distributed
}  // namespace KokkosFFT
