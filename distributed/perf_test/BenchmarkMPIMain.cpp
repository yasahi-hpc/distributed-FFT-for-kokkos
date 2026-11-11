#include <iostream>
#include <string>
#include <vector>
#include <mpi.h>
#include <benchmark/benchmark.h>
#include "Benchmark_Context.hpp"
#include "Benchmark_MPI_Context.hpp"
#include <Kokkos_Core.hpp>

// This reporter does nothing.
// We can use it to disable output from all but the root process
class NullReporter : public ::benchmark::BenchmarkReporter {
 public:
  NullReporter() {}
  virtual bool ReportContext(const Context &) { return true; }
  virtual void ReportRuns(const std::vector<Run> &) {}
  virtual void Finalize() {}
};

/// \brief The arguments of a rank other than 0: without the output file
/// arguments, and with an empty --benchmark_out last, which overrides the
/// BENCHMARK_OUT environment variable. Google Benchmark opens the output
/// file in RunSpecifiedBenchmarks on every process that has one, even with a
/// null file reporter, so that the processes would write the same file.
std::vector<char *> args_without_output_file(int argc, char **argv) {
  static char empty_out[] = "--benchmark_out=";
  std::vector<char *> args;
  for (int i = 0; i < argc; ++i) {
    const std::string arg = argv[i];
    if (i > 0 && (arg.rfind("--benchmark_out=", 0) == 0 ||
                  arg.rfind("--benchmark_out_format=", 0) == 0)) {
      continue;
    }
    args.push_back(argv[i]);
  }
  args.push_back(empty_out);
  args.push_back(nullptr);
  return args;
}

int main(int argc, char **argv) {
  // The library calls MPI from the thread that calls it only; ask for more,
  // but run with what the MPI library provides
  int provided = MPI_THREAD_SINGLE;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0 && provided < MPI_THREAD_FUNNELED) {
    std::cerr << "Warning: MPI provides "
              << KokkosFFT::Distributed::Benchmark::thread_level_name(provided)
              << " only" << std::endl;
  }

  Kokkos::initialize(argc, argv);
  {
    std::vector<char *> args;
    if (rank != 0) {
      args = args_without_output_file(argc, argv);
      argc = static_cast<int>(args.size()) - 1;
      argv = args.data();
    }
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::SetDefaultTimeUnit(benchmark::kSecond);

    // The MPI context is collective (every rank); the rest is reported by
    // rank 0 only
    KokkosFFT::Distributed::Benchmark::add_mpi_benchmark_context(MPI_COMM_WORLD,
                                                                 provided);
    if (rank == 0) {
      KokkosFFT::Distributed::Benchmark::add_benchmark_context(true);
      // root process will use a reporter from the usual set provided by
      // ::benchmark
      ::benchmark::RunSpecifiedBenchmarks();
    } else {
      // reporting from other processes is disabled by passing a custom
      // reporter; they have no output file (args_without_output_file)
      NullReporter null;
      ::benchmark::RunSpecifiedBenchmarks(/* display_reporter = */ &null);
    }

    ::benchmark::Shutdown();
  }
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
