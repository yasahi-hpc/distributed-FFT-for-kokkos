#ifndef PERFTEST_UTILS_HPP
#define PERFTEST_UTILS_HPP

#include <mpi.h>
#include <benchmark/benchmark.h>
#include <Kokkos_Core.hpp>

// F is a function that takes (state, MPI_Comm, args...)
template <typename CommType, typename F, typename... Args>
void do_iteration(benchmark::State &state, const CommType &comm, F &&func,
                  Args... args) {
  Kokkos::fence();
  Kokkos::Timer timer;
  func(state, comm, args...);
  auto elapsed = timer.seconds();

  double max_elapsed_second;
  MPI_Allreduce(&elapsed, &max_elapsed_second, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
  state.SetIterationTime(max_elapsed_second);
}

#endif  // PERFTEST_UTILS_HPP
