#include <benchmark/benchmark.h>
#include "Benchmark_Context.hpp"
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_All2All.hpp"
#include "PerfTest_Utils.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Benchmark {

template <typename ExecutionSpace, typename CommType, typename ViewType>
void BM_all2all(benchmark::State&, const CommType& comm,
                const ExecutionSpace& exec_space, const ViewType& send,
                const ViewType& recv) {
  KokkosFFT::Distributed::Impl::all2all(exec_space, send, recv, comm);
}

template <typename T, typename LayoutType>
void benchmark_all2all(benchmark::State& state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    state.SkipWithError("benchmark_all2all needs at least 2 ranks");
  }

  const int n = state.range(0);
  if (n % size != 0) {
    state.SkipWithError("Input size must be divisible by number of ranks");
  }

  using View3DType =
      Kokkos::View<T***, LayoutType, Kokkos::DefaultExecutionSpace>;

  int n0_buffer = 0, n1_buffer = 0, n2_buffer = 0;
  if constexpr (std::is_same<LayoutType, Kokkos::LayoutLeft>::value) {
    n0_buffer = n / size;
    n1_buffer = n;
    n2_buffer = size;
  } else {
    n0_buffer = size;
    n1_buffer = n;
    n2_buffer = n / size;
  }
  View3DType send("send", n0_buffer, n1_buffer, n2_buffer),
      recv("recv", n0_buffer, n1_buffer, n2_buffer);

  Kokkos::DefaultExecutionSpace exec_space;
  using CommType = KokkosFFT::Distributed::Impl::TplComm;
  CommType comm(MPI_COMM_WORLD);
  while (state.KeepRunning()) {
    do_iteration(
        state, comm,
        BM_all2all<Kokkos::DefaultExecutionSpace, CommType, View3DType>,
        exec_space, send, recv);
  }
  state.counters["bytes"] = send.size() * 2;
}

#define BENCHMARK_All2All(type, layout, start, stop) \
  BENCHMARK(benchmark_all2all<type, Kokkos::layout>) \
      ->UseManualTime()                              \
      ->Unit(benchmark::kMillisecond)                \
      ->ArgName("N")                                 \
      ->RangeMultiplier(2)                           \
      ->Range(start, stop)

BENCHMARK_All2All(float, LayoutLeft, 256, 4096);
BENCHMARK_All2All(float, LayoutRight, 256, 4096);
BENCHMARK_All2All(double, LayoutLeft, 256, 4096);
BENCHMARK_All2All(double, LayoutRight, 256, 4096);

#undef BENCHMARK_All2All

}  // namespace Benchmark
}  // namespace Distributed
}  // namespace KokkosFFT
