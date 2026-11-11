#include <exception>
#include <benchmark/benchmark.h>
#include "Benchmark_Context.hpp"
#include "Benchmark_Timing.hpp"
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_All2All.hpp"
#include "KokkosFFT_Distributed_CommBuffer.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Benchmark {
using execution_space = Kokkos::DefaultExecutionSpace;

/// \brief The all-to-all of the communicator wrapper of the plans
/// (TplComm), on send and receive buffers allocated as the plans allocate
/// them (CommBuffer: a View, or the symmetric heap for NVSHMEM and
/// rocSHMEM)
template <typename T, typename LayoutType>
void benchmark_all2all_impl(benchmark::State& state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    state.SkipWithError("benchmark_all2all needs at least 2 ranks");
    return;
  }

  const int n = state.range(0);

  using View3DType = Kokkos::View<T***, LayoutType, execution_space,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  int n0_buffer = 0, n1_buffer = 0, n2_buffer = 0;
  if constexpr (std::is_same<LayoutType, Kokkos::LayoutLeft>::value) {
    n0_buffer = (n - 1) / size + 1;
    n1_buffer = n;
    n2_buffer = size;
  } else {
    n0_buffer = size;
    n1_buffer = n;
    n2_buffer = (n - 1) / size + 1;
  }
  using CommType = KokkosFFT::Distributed::Impl::TplComm<execution_space>;
  using BufferType =
      KokkosFFT::Distributed::Impl::CommBuffer<CommType, execution_space, T>;

  // The communicator first: it sets up the backend (NVSHMEM) that allocates
  // the buffers, which are freed before it
  execution_space exec_space;
  CommType comm(MPI_COMM_WORLD, exec_space);
  const std::size_t buffer_size = static_cast<std::size_t>(n0_buffer) *
                                  static_cast<std::size_t>(n1_buffer) *
                                  static_cast<std::size_t>(n2_buffer);
  BufferType send_buffer("send", buffer_size, /*is_exchanged=*/true),
      recv_buffer("recv", buffer_size, /*is_exchanged=*/true);
  View3DType send(send_buffer.data(), n0_buffer, n1_buffer, n2_buffer),
      recv(recv_buffer.data(), n0_buffer, n1_buffer, n2_buffer);
  auto exchange = [&]() {
    KokkosFFT::Distributed::Impl::all2all(send, recv, comm);
  };
  warmup(exec_space, MPI_COMM_WORLD, exchange);
  for (auto _ : state) {
    const double elapsed =
        time_max_over_ranks(exec_space, MPI_COMM_WORLD, exchange);
    report_results(state, send, recv, elapsed);
  }
}

/// \brief benchmark_all2all_impl, with an error (the same on every rank:
/// buffers, sizes) reported as the error of this benchmark, so that the
/// next benchmarks still run
template <typename T, typename LayoutType>
void benchmark_all2all(benchmark::State& state) {
  try {
    benchmark_all2all_impl<T, LayoutType>(state);
  } catch (const std::exception& e) {
    state.SkipWithError(e.what());
  }
}

#define BENCHMARK_All2All(type, layout, start, stop) \
  BENCHMARK(benchmark_all2all<type, Kokkos::layout>) \
      ->UseManualTime()                              \
      ->Unit(benchmark::kMillisecond)                \
      ->ArgName("N")                                 \
      ->RangeMultiplier(2)                           \
      ->Range(start, stop)                           \
      ->ComputeStatistics("min", min_statistic)

BENCHMARK_All2All(float, LayoutLeft, 256, 4096);
BENCHMARK_All2All(float, LayoutRight, 256, 4096);
BENCHMARK_All2All(double, LayoutLeft, 256, 4096);
BENCHMARK_All2All(double, LayoutRight, 256, 4096);

#undef BENCHMARK_All2All

}  // namespace Benchmark
}  // namespace Distributed
}  // namespace KokkosFFT
