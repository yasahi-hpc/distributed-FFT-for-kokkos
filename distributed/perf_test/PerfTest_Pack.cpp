#include <benchmark/benchmark.h>
#include "Benchmark_Context.hpp"
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_PackUnpack.hpp"
#include "KokkosFFT_Distributed_Extents.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Benchmark {
using exec_space = Kokkos::DefaultExecutionSpace;

template <typename T, typename LayoutType, std::size_t DIM, std::size_t order,
          std::size_t axis>
static void benchmark_pack(benchmark::State& state) {
  using src_data_type  = KokkosFFT::Impl::add_pointer_n_t<T, DIM>;
  using dst_data_type  = KokkosFFT::Impl::add_pointer_n_t<T, DIM + 1>;
  using SrcViewType    = Kokkos::View<src_data_type, LayoutType, exec_space>;
  using DstViewType    = Kokkos::View<dst_data_type, LayoutType, exec_space>;
  using map_type       = std::array<std::size_t, DIM>;
  using src_shape_type = std::array<std::size_t, DIM>;

  std::size_t size = state.range(0), nprocs = state.range(1);

  src_shape_type global_extents = {}, local_extents = {};
  map_type src_topology = {}, dst_topology = {};
  for (std::size_t i = 0; i < global_extents.size(); i++) {
    global_extents.at(i) = size;
  }

  // Only consider X->Y, Y->Z and Z->X packing
  for (std::size_t i = 0; i < src_topology.size(); i++) {
    src_topology.at(i) = i == ((axis + DIM - 1) % DIM) ? nprocs : 1;
    dst_topology.at(i) = i == axis ? nprocs : 1;
  }

  // Local extents
  for (std::size_t i = 0; i < local_extents.size(); i++) {
    local_extents.at(i) = size / src_topology.at(i);
  }

  map_type src_map = {};
  if constexpr (DIM == 2) {
    src_map = (order == 0) ? map_type{0, 1} : map_type{1, 0};
  } else {
    src_map = (order == 0)   ? map_type({0, 1, 2})
              : (order == 1) ? map_type({0, 2, 1})
              : (order == 2) ? map_type({1, 0, 2})
              : (order == 3) ? map_type({1, 2, 0})
              : (order == 4) ? map_type({2, 0, 1})
                             : map_type({2, 1, 0});
  }

  auto src_extents =
      KokkosFFT::Distributed::Impl::get_mapped_extents(local_extents, src_map);
  auto dst_extents =
      KokkosFFT::Distributed::Impl::compute_buffer_extents<LayoutType>(
          global_extents, src_topology, dst_topology);

  SrcViewType src("src",
                  KokkosFFT::Impl::create_layout<LayoutType>(src_extents));
  DstViewType dst("dst",
                  KokkosFFT::Impl::create_layout<LayoutType>(dst_extents));

  exec_space exec;
  for (auto _ : state) {
    Kokkos::Timer timer;

    KokkosFFT::Distributed::Impl::pack(exec, src, dst, src_map, axis);
    exec.fence();
    report_results(state, src, dst, timer.seconds());
  }
}

#define BENCHMARK_Pack(type, layout, dim, order, axis, b0, e0, b1, e1) \
  BENCHMARK(benchmark_pack<type, Kokkos::layout, dim, order, axis>)    \
      ->UseManualTime()                                                \
      ->Unit(benchmark::kMillisecond)                                  \
      ->ArgNames({"size", "nprocs"})                                   \
      ->RangeMultiplier(2)                                             \
      ->Ranges({{b0, e0}, {b1, e1}})

// 2D cases
// BENCHMARK_Pack(double, LayoutLeft, 2, 0, 0, 64, 1024, 16, 64);
// BENCHMARK_Pack(double, LayoutLeft, 2, 0, 1, 64, 1024, 16, 64);
// BENCHMARK_Pack(double, LayoutRight, 2, 0, 0, 64, 1024, 16, 64);
// BENCHMARK_Pack(double, LayoutRight, 2, 0, 1, 64, 1024, 16, 64);

// BENCHMARK_Pack(Kokkos::complex<double>, LayoutLeft, 2, 0, 0, 64, 1024, 16,
// 64);
// BENCHMARK_Pack(Kokkos::complex<double>, LayoutLeft, 2, 0, 1, 64, 1024, 16,
// 64);
// BENCHMARK_Pack(Kokkos::complex<double>, LayoutRight, 2, 0, 0, 64, 1024, 16,
// 64);
// BENCHMARK_Pack(Kokkos::complex<double>, LayoutRight, 2, 0, 1, 64, 1024, 16,
// 64);

// 3D cases
// BENCHMARK_Pack(double, LayoutLeft, 3, 0, 0, 64, 1024, 16, 64);
// BENCHMARK_Pack(double, LayoutLeft, 3, 0, 1, 64, 1024, 16, 64);
// BENCHMARK_Pack(double, LayoutLeft, 3, 0, 2, 64, 1024, 16, 64);

// BENCHMARK_Pack(double, LayoutRight, 3, 0, 0, 64, 1024, 16, 64);
// BENCHMARK_Pack(double, LayoutRight, 3, 0, 1, 64, 1024, 16, 64);
// BENCHMARK_Pack(double, LayoutRight, 3, 0, 2, 64, 1024, 16, 64);

BENCHMARK_Pack(Kokkos::complex<double>, LayoutLeft, 3, 0, 0, 64, 1024, 16, 64);
BENCHMARK_Pack(Kokkos::complex<double>, LayoutLeft, 3, 0, 1, 64, 1024, 16, 64);
BENCHMARK_Pack(Kokkos::complex<double>, LayoutLeft, 3, 0, 2, 64, 1024, 16, 64);

BENCHMARK_Pack(Kokkos::complex<double>, LayoutRight, 3, 0, 0, 64, 1024, 16, 64);
BENCHMARK_Pack(Kokkos::complex<double>, LayoutRight, 3, 0, 1, 64, 1024, 16, 64);
BENCHMARK_Pack(Kokkos::complex<double>, LayoutRight, 3, 0, 2, 64, 1024, 16, 64);

#undef BENCHMARK_Pack

}  // namespace Benchmark
}  // namespace Distributed
}  // namespace KokkosFFT
