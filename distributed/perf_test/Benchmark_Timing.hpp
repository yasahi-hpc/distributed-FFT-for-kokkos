#ifndef KOKKOSFFT_BENCHMARK_TIMING_HPP
#define KOKKOSFFT_BENCHMARK_TIMING_HPP

#include <algorithm>
#include <vector>
#include <mpi.h>
#include <benchmark/benchmark.h>
#include <Kokkos_Core.hpp>

namespace KokkosFFT {
namespace Distributed {
namespace Benchmark {

/// \brief Number of untimed executions before the timed ones: the first
/// executions include lazy initializations (NCCL communicators, NVSHMEM,
/// the first execution of a vendor plan, the caches of the kernels)
inline constexpr int default_warmup_iterations = 3;

/// \brief Run f, untimed, n times, then wait for the execution space and
/// for every rank of comm
/// \param[in] exec_space Execution space on which f enqueues its work
/// \param[in] comm MPI communicator of the benchmark
/// \param[in] f Work of one iteration
/// \param[in] n Number of warm-up executions
template <typename ExecutionSpace, typename FunctorType>
void warmup(const ExecutionSpace& exec_space, MPI_Comm comm,
            const FunctorType& f, int n = default_warmup_iterations) {
  for (int i = 0; i < n; ++i) {
    f();
  }
  exec_space.fence();
  ::MPI_Barrier(comm);
}

/// \brief Time one execution of f on every rank of comm: the ranks wait for
/// each other before the timer starts, f is timed up to the completion of
/// its work on exec_space, and the time is the maximum over the ranks (the
/// same on every rank, so that every rank runs the same number of
/// iterations)
/// \param[in] exec_space Execution space on which f enqueues its work
/// \param[in] comm MPI communicator of the benchmark
/// \param[in] f Work of one iteration
/// \return The time of the slowest rank, in seconds
template <typename ExecutionSpace, typename FunctorType>
double time_max_over_ranks(const ExecutionSpace& exec_space, MPI_Comm comm,
                           const FunctorType& f) {
  exec_space.fence();
  ::MPI_Barrier(comm);
  Kokkos::Timer timer;
  f();
  exec_space.fence();
  double elapsed = timer.seconds(), max_elapsed = 0.0;
  ::MPI_Allreduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, comm);
  return max_elapsed;
}

/// \brief The minimum over the repetitions (--benchmark_repetitions), next to
/// the mean, median and standard deviation that Google Benchmark computes
inline double min_statistic(const std::vector<double>& v) {
  return v.empty() ? 0.0 : *std::min_element(v.begin(), v.end());
}

}  // namespace Benchmark
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
