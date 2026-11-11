#ifndef KOKKOSFFT_BENCHMARK_CONTEXT_HPP
#define KOKKOSFFT_BENCHMARK_CONTEXT_HPP

#include <algorithm>
#include <cstdlib>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_PrintConfiguration.hpp"
#include <KokkosFFT_Distributed_Version_Info.hpp>

// POSIX: the environment of the process
extern char** environ;

namespace KokkosFFT {
namespace Distributed {
namespace Benchmark {

/// \brief Remove unwanted spaces and colon signs from input string. In case of
/// invalid input it will return an empty string.
inline std::string remove_unwanted_characters(std::string str) {
  auto from = str.find_first_not_of(" :");
  auto to   = str.find_last_not_of(" :");

  if (from == std::string::npos || to == std::string::npos) {
    return "";
  }

  // return extracted part of string without unwanted spaces and colon signs
  return str.substr(from, to + 1);
}

/// \brief Extract all key:value pairs from kokkos configuration and add it to
/// the benchmark context
inline void add_kokkos_configuration(bool verbose) {
  std::ostringstream msg;
  Kokkos::print_configuration(msg, verbose);
  KokkosFFT::Distributed::print_configuration(msg);

  // Iterate over lines returned from kokkos and extract key:value pairs
  std::set<std::string> keys;
  std::stringstream ss{msg.str()};
  for (std::string line; std::getline(ss, line, '\n');) {
    auto found = line.find_first_of(':');
    if (found != std::string::npos) {
      auto val = remove_unwanted_characters(line.substr(found + 1));
      auto key = remove_unwanted_characters(line.substr(0, found));
      // Ignore line without value, for example a category name, and the
      // keys that the configuration prints more than once (the first value
      // is kept)
      if (!val.empty() && !key.empty() && keys.insert(key).second) {
        benchmark::AddCustomContext(key, val);
      }
    }
  }
}

/// \brief Add Kokkos Kernels git info and google benchmark release to
/// benchmark context.
inline void add_version_info() {
  using namespace KokkosFFT::Distributed::Impl;

  if (!GIT_BRANCH.empty()) {
    benchmark::AddCustomContext("GIT_BRANCH", std::string(GIT_BRANCH));
    benchmark::AddCustomContext("GIT_COMMIT_HASH",
                                std::string(GIT_COMMIT_HASH));
    benchmark::AddCustomContext("GIT_CLEAN_STATUS",
                                std::string(GIT_CLEAN_STATUS));
    benchmark::AddCustomContext("GIT_COMMIT_DESCRIPTION",
                                std::string(GIT_COMMIT_DESCRIPTION));
    benchmark::AddCustomContext("GIT_COMMIT_DATE",
                                std::string(GIT_COMMIT_DATE));
  }
}

/// \brief Whether an environment variable is recorded in the context: the
/// variables that change how the benchmarks run (OpenMP, Kokkos and
/// Kokkos-FFT, the communication libraries, GPU-aware MPI, the visible GPUs)
inline bool is_recorded_env(const std::string& name) {
  static const char* const prefixes[] = {"OMP_",
                                         "KOKKOS_",
                                         "KOKKOSFFT_",
                                         "NVSHMEM_",
                                         "ROCSHMEM_",
                                         "NCCL_",
                                         "RCCL_",
                                         "UCX_",
                                         "MPICH_GPU_",
                                         "MPICH_OFI_",
                                         "OMPI_MCA_",
                                         "CUDA_VISIBLE_DEVICES",
                                         "HIP_VISIBLE_DEVICES",
                                         "ROCR_VISIBLE_DEVICES",
                                         "CUFFT_",
                                         "ROCFFT_"};
  // Set by the Open MPI launcher for every process (job ids, URIs,
  // directories, process counts): not settings of the run
  static const char* const launcher_prefixes[] = {
      "OMPI_MCA_orte_",        "OMPI_MCA_ess",
      "OMPI_MCA_pmix",         "OMPI_MCA_plm_",
      "OMPI_MCA_initial_wdir", "OMPI_MCA_shmem_RUNTIME_QUERY_hint"};
  auto starts_with = [&name](const char* prefix) {
    return name.rfind(prefix, 0) == 0;
  };
  for (const char* prefix : launcher_prefixes) {
    if (starts_with(prefix)) return false;
  }
  for (const char* prefix : prefixes) {
    if (starts_with(prefix)) return true;
  }
  return false;
}

/// \brief Add the recorded environment variables (is_recorded_env) to the
/// benchmark context, sorted by name
inline void add_env_info() {
  std::vector<std::pair<std::string, std::string>> vars;
  for (char** env = environ; env != nullptr && *env != nullptr; ++env) {
    const std::string entry(*env);
    const auto eq = entry.find('=');
    if (eq == std::string::npos) continue;
    std::string name = entry.substr(0, eq);
    if (is_recorded_env(name)) vars.emplace_back(name, entry.substr(eq + 1));
  }
  std::sort(vars.begin(), vars.end());
  for (const auto& [name, value] : vars) {
    benchmark::AddCustomContext(name, value);
  }
}

/// \brief Gather all context information and add it to benchmark context
inline void add_benchmark_context(bool verbose = false) {
  add_kokkos_configuration(verbose);
  add_version_info();
  add_env_info();
}

/**
 * \brief Report throughput and amount of data processed for simple View
 * operations
 */
template <typename InViewType, typename OutViewType>
void report_results(benchmark::State& state, const InViewType& in,
                    const OutViewType& out, double time) {
  // data processed in megabytes
  const double in_data_processed =
      static_cast<double>(in.size() * sizeof(typename InViewType::value_type)) /
      1.0e6;
  const double out_data_processed =
      static_cast<double>(out.size() *
                          sizeof(typename OutViewType::value_type)) /
      1.0e6;

  state.SetIterationTime(time);
  state.counters["MB (In)"]  = benchmark::Counter(in_data_processed);
  state.counters["MB (Out)"] = benchmark::Counter(out_data_processed);
  state.counters["GB/s"] =
      benchmark::Counter((in_data_processed + out_data_processed) / 1.0e3,
                         benchmark::Counter::kIsIterationInvariantRate);
}

}  // namespace Benchmark
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
