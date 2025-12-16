#include <string>
#include <cstdlib>
#include <mpi.h>
#include <benchmark/benchmark.h>
#include "Benchmark_Context.hpp"
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

bool has_output_flag(int argc, char **argv) {
  for (int i = 0; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.find("--benchmark_out=") == 0) {
      return true;
    }
  }
  return false;
}

bool has_output_envvar() { return std::getenv("BENCHMARK_OUT") != nullptr; }

int main(int argc, char **argv) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if (provided != MPI_THREAD_MULTIPLE) {
    throw std::runtime_error("MPI_THREAD_MULTIPLE is needed");
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Kokkos::initialize(argc, argv);
  {
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::SetDefaultTimeUnit(benchmark::kSecond);

    if (rank == 0) {
      // root process will use a reporter from the usual set provided by
      // ::benchmark
      KokkosFFT::Distributed::Benchmark::add_benchmark_context(true);
      ::benchmark::RunSpecifiedBenchmarks();
    } else {
      // reporting from other processes is disabled by passing a custom reporter
      NullReporter null;
      bool has_file_output = has_output_flag(argc, argv) || has_output_envvar();
      if (has_file_output) {
        ::benchmark::RunSpecifiedBenchmarks(/* display_reporter = */ &null,
                                            /* file_reporter = */ &null);
      } else {
        ::benchmark::RunSpecifiedBenchmarks(/* display_reporter = */ &null);
      }
    }

    ::benchmark::Shutdown();
  }
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
