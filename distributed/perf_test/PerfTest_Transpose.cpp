#include <benchmark/benchmark.h>
#include "Benchmark_Context.hpp"
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_Extents.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Benchmark {

using exec_space = Kokkos::DefaultExecutionSpace;

template <typename T, typename LayoutType, std::size_t DIM, std::size_t order,
          std::size_t axis>
static void benchmark_transpose(benchmark::State& state) {
  using data_type  = KokkosFFT::Impl::add_pointer_n_t<T, DIM>;
  using ViewType   = Kokkos::View<data_type, LayoutType, exec_space>;
  using map_type   = std::array<int, DIM>;
  using shape_type = std::array<std::size_t, DIM>;

  std::size_t size = state.range(0), nprocs = state.range(1);

  if (size / nprocs == 0) {
    state.SkipWithError(
        "The size of benchmark_transpose: " + std::to_string(size) +
        " must be divisible by " + std::to_string(nprocs));
    return;
  }

  map_type map = {};
  if constexpr (DIM == 2) {
    map = (order == 0) ? map_type{0, 1} : map_type{1, 0};
  } else {
    map = (order == 0)   ? map_type({0, 1, 2})
          : (order == 1) ? map_type({0, 2, 1})
          : (order == 2) ? map_type({1, 0, 2})
          : (order == 3) ? map_type({1, 2, 0})
          : (order == 4) ? map_type({2, 0, 1})
                         : map_type({2, 1, 0});
  }

  shape_type src_extents = {};
  for (std::size_t i = 0; i < src_extents.size(); i++) {
    src_extents.at(i) = i == axis ? size / nprocs : size;
  }

  auto dst_extents =
      KokkosFFT::Distributed::Impl::get_mapped_extents(src_extents, map);

  ViewType src("src", KokkosFFT::Impl::create_layout<LayoutType>(src_extents));
  ViewType dst("dst", KokkosFFT::Impl::create_layout<LayoutType>(dst_extents));

  exec_space exec;
  for (auto _ : state) {
    Kokkos::Timer timer;

    KokkosFFT::Impl::transpose(exec, src, dst, map, true);
    exec.fence();
    report_results(state, src, dst, timer.seconds());
  }
}

#define BENCHMARK_Transpose(type, layout, dim, order, axis, b0, e0, b1, e1) \
  BENCHMARK(benchmark_transpose<type, Kokkos::layout, dim, order, axis>)    \
      ->UseManualTime()                                                     \
      ->Unit(benchmark::kMillisecond)                                       \
      ->ArgNames({"size", "nprocs"})                                        \
      ->RangeMultiplier(2)                                                  \
      ->Ranges({{b0, e0}, {b1, e1}})

BENCHMARK_Transpose(Kokkos::complex<double>, LayoutLeft, 3, 0, 0, 64, 1024, 16,
                    64);
BENCHMARK_Transpose(Kokkos::complex<double>, LayoutLeft, 3, 0, 1, 64, 1024, 16,
                    64);
BENCHMARK_Transpose(Kokkos::complex<double>, LayoutLeft, 3, 0, 2, 64, 1024, 16,
                    64);

BENCHMARK_Transpose(Kokkos::complex<double>, LayoutRight, 3, 0, 0, 64, 1024, 16,
                    64);
BENCHMARK_Transpose(Kokkos::complex<double>, LayoutRight, 3, 0, 1, 64, 1024, 16,
                    64);
BENCHMARK_Transpose(Kokkos::complex<double>, LayoutRight, 3, 0, 2, 64, 1024, 16,
                    64);

BENCHMARK_Transpose(Kokkos::complex<double>, LayoutLeft, 3, 4, 0, 64, 1024, 16,
                    64);
BENCHMARK_Transpose(Kokkos::complex<double>, LayoutLeft, 3, 4, 1, 64, 1024, 16,
                    64);
BENCHMARK_Transpose(Kokkos::complex<double>, LayoutLeft, 3, 4, 2, 64, 1024, 16,
                    64);

BENCHMARK_Transpose(Kokkos::complex<double>, LayoutRight, 3, 4, 0, 64, 1024, 16,
                    64);
BENCHMARK_Transpose(Kokkos::complex<double>, LayoutRight, 3, 4, 1, 64, 1024, 16,
                    64);
BENCHMARK_Transpose(Kokkos::complex<double>, LayoutRight, 3, 4, 2, 64, 1024, 16,
                    64);

#undef BENCHMARK_Transpose

}  // namespace Benchmark
}  // namespace Distributed
}  // namespace KokkosFFT
