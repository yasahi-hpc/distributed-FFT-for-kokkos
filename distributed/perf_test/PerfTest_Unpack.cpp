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
auto prepare_benchmark(const std::size_t size, const std::size_t nprocs) {
  using src_data_type  = KokkosFFT::Impl::add_pointer_n_t<T, DIM + 1>;
  using dst_data_type  = KokkosFFT::Impl::add_pointer_n_t<T, DIM>;
  using SrcViewType    = Kokkos::View<src_data_type, LayoutType, exec_space>;
  using DstViewType    = Kokkos::View<dst_data_type, LayoutType, exec_space>;
  using map_type       = std::array<std::size_t, DIM>;
  using src_shape_type = std::array<std::size_t, DIM>;

  src_shape_type global_extents = {};
  src_shape_type local_extents  = {};
  map_type src_topology = {}, dst_topology = {};

  for (std::size_t i = 0; i < global_extents.size(); i++) {
    global_extents.at(i) = size;
  }

  // Only consider X->Y, Y->Z and Z->X packing
  for (std::size_t i = 0; i < global_extents.size(); i++) {
    src_topology.at(i) = i == ((axis + DIM - 1) % DIM) ? nprocs : 1;
    dst_topology.at(i) = i == axis ? nprocs : 1;
  }

  // Local extents
  for (std::size_t i = 0; i < local_extents.size(); i++) {
    local_extents.at(i) = size / src_topology.at(i);
  }

  map_type dst_map = {};
  if constexpr (DIM == 2) {
    dst_map = (order == 0) ? map_type{0, 1} : map_type{1, 0};
  } else {
    dst_map = (order == 0)   ? map_type({0, 1, 2})
              : (order == 1) ? map_type({0, 2, 1})
              : (order == 2) ? map_type({1, 0, 2})
              : (order == 3) ? map_type({1, 2, 0})
              : (order == 4) ? map_type({2, 0, 1})
                             : map_type({2, 1, 0});
  }

  auto src_extents =
      KokkosFFT::Distributed::Impl::get_buffer_extents<LayoutType>(
          global_extents, src_topology, dst_topology);
  auto dst_extents =
      KokkosFFT::Distributed::Impl::get_mapped_extents(local_extents, dst_map);

  SrcViewType src("src",
                  KokkosFFT::Impl::create_layout<LayoutType>(src_extents));
  DstViewType dst("dst",
                  KokkosFFT::Impl::create_layout<LayoutType>(dst_extents));

  return std::make_tuple(src, dst, dst_map);
}

template <typename T, typename LayoutType, std::size_t DIM, std::size_t order,
          std::size_t axis>
static void benchmark_unpack(benchmark::State& state) {
  using src_data_type  = KokkosFFT::Impl::add_pointer_n_t<T, DIM + 1>;
  using dst_data_type  = KokkosFFT::Impl::add_pointer_n_t<T, DIM>;
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
  for (std::size_t i = 0; i < global_extents.size(); i++) {
    src_topology.at(i) = i == ((axis + DIM - 1) % DIM) ? nprocs : 1;
    dst_topology.at(i) = i == axis ? nprocs : 1;
  }

  // Local extents
  for (std::size_t i = 0; i < local_extents.size(); i++) {
    local_extents.at(i) = size / src_topology.at(i);
  }

  map_type dst_map = {};
  if constexpr (DIM == 2) {
    dst_map = (order == 0) ? map_type{0, 1} : map_type{1, 0};
  } else {
    dst_map = (order == 0)   ? map_type({0, 1, 2})
              : (order == 1) ? map_type({0, 2, 1})
              : (order == 2) ? map_type({1, 0, 2})
              : (order == 3) ? map_type({1, 2, 0})
              : (order == 4) ? map_type({2, 0, 1})
                             : map_type({2, 1, 0});
  }

  auto src_extents =
      KokkosFFT::Distributed::Impl::get_buffer_extents<LayoutType>(
          global_extents, src_topology, dst_topology);
  auto dst_extents =
      KokkosFFT::Distributed::Impl::get_mapped_extents(local_extents, dst_map);

  SrcViewType src("src",
                  KokkosFFT::Impl::create_layout<LayoutType>(src_extents));
  DstViewType dst("dst",
                  KokkosFFT::Impl::create_layout<LayoutType>(dst_extents));

  exec_space exec;
  for (auto _ : state) {
    Kokkos::Timer timer;

    KokkosFFT::Distributed::Impl::unpack(exec, src, dst, dst_map, axis);
    exec.fence();
    report_results(state, src, dst, timer.seconds());
  }
}

#define BENCHMARK_Unpack(type, layout, dim, order, axis, b0, e0, b1, e1) \
  BENCHMARK(benchmark_unpack<type, Kokkos::layout, dim, order, axis>)    \
      ->UseManualTime()                                                  \
      ->Unit(benchmark::kMillisecond)                                    \
      ->ArgNames({"size", "nprocs"})                                     \
      ->RangeMultiplier(2)                                               \
      ->Ranges({{b0, e0}, {b1, e1}})

// 2D cases
// BENCHMARK_Unpack(double, LayoutLeft, 2, 0, 0, 64, 1024, 8, 64);
// BENCHMARK_Unpack(double, LayoutLeft, 2, 0, 1, 64, 1024, 8, 64);
// BENCHMARK_Unpack(double, LayoutRight, 2, 0, 0, 64, 1024, 8, 64);
// BENCHMARK_Unpack(double, LayoutRight, 2, 0, 1, 64, 1024, 8, 64);

// BENCHMARK_Unpack(Kokkos::complex<double>, LayoutLeft, 2, 0, 0, 64, 1024, 8,
// 64);
// BENCHMARK_Unpack(Kokkos::complex<double>, LayoutLeft, 2, 0, 1, 64, 1024, 8,
// 64);
// BENCHMARK_Unpack(Kokkos::complex<double>, LayoutRight, 2, 0, 0, 64, 1024, 8,
// 64);
// BENCHMARK_Unpack(Kokkos::complex<double>, LayoutRight, 2, 0, 1, 64, 1024, 8,
// 64);

// 3D cases
// BENCHMARK_Unpack(double, LayoutLeft, 3, 0, 0, 64, 1024, 16, 64);
// BENCHMARK_Unpack(double, LayoutLeft, 3, 0, 1, 64, 1024, 16, 64);
// BENCHMARK_Unpack(double, LayoutLeft, 3, 0, 2, 64, 1024, 16, 64);

// BENCHMARK_Unpack(double, LayoutRight, 3, 0, 0, 64, 1024, 16, 64);
// BENCHMARK_Unpack(double, LayoutRight, 3, 0, 1, 64, 1024, 16, 64);
// BENCHMARK_Unpack(double, LayoutRight, 3, 0, 2, 64, 1024, 16, 64);

BENCHMARK_Unpack(Kokkos::complex<double>, LayoutLeft, 3, 0, 0, 64, 1024, 16,
                 64);
BENCHMARK_Unpack(Kokkos::complex<double>, LayoutLeft, 3, 0, 1, 64, 1024, 16,
                 64);
BENCHMARK_Unpack(Kokkos::complex<double>, LayoutLeft, 3, 0, 2, 64, 1024, 16,
                 64);

BENCHMARK_Unpack(Kokkos::complex<double>, LayoutRight, 3, 0, 0, 64, 1024, 16,
                 64);
BENCHMARK_Unpack(Kokkos::complex<double>, LayoutRight, 3, 0, 1, 64, 1024, 16,
                 64);
BENCHMARK_Unpack(Kokkos::complex<double>, LayoutRight, 3, 0, 2, 64, 1024, 16,
                 64);

#undef BENCHMARK_Unpack

}  // namespace Benchmark
}  // namespace Distributed
}  // namespace KokkosFFT
