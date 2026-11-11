#ifndef KOKKOSFFT_BENCHMARK_MPI_CONTEXT_HPP
#define KOKKOSFFT_BENCHMARK_MPI_CONTEXT_HPP

#include <string>
#include <mpi.h>
#include <benchmark/benchmark.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_config.hpp"

#if defined(ENABLE_TPL_NVSHMEM) || defined(ENABLE_TPL_CUFFT_MP)
#include <nvshmem.h>
#endif

#if defined(ENABLE_TPL_CUFFT_MP)
#include <cufftMp.h>
#endif

#if defined(ENABLE_TPL_ROCSHMEM)
#include <rocshmem/rocshmem.hpp>
#endif

#if defined(ENABLE_TPL_NVSHMEM) || defined(ENABLE_TPL_ROCSHMEM)
#include "KokkosFFT_Distributed_SHMEM_Mode.hpp"
#endif

namespace KokkosFFT {
namespace Distributed {
namespace Benchmark {

/// \brief The communication backend of the native plans, fixed at compile
/// time
inline std::string comm_backend_name() {
#if defined(ENABLE_TPL_NVSHMEM)
  return "NVSHMEM";
#elif defined(ENABLE_TPL_ROCSHMEM)
  return "rocSHMEM";
#elif defined(KOKKOSFFT_DISTRIBUTED_ENABLE_TPL_RCCL)
  return "RCCL";
#elif defined(ENABLE_TPL_NCCL)
  return "NCCL";
#elif defined(ENABLE_TPL_ONECCL)
  return "oneCCL";
#else
  return "MPI";
#endif
}

/// \brief The vendor distributed FFT backend, if one is built
inline std::string vendor_plan_backend_name() {
#if defined(ENABLE_TPL_CUFFT_MP)
  return "cuFFTMp";
#elif defined(ENABLE_TPL_ROCFFT_MPI)
  return "rocFFT-MPI";
#else
  return "none";
#endif
}

inline std::string thread_level_name(int level) {
  switch (level) {
    case MPI_THREAD_SINGLE: return "MPI_THREAD_SINGLE";
    case MPI_THREAD_FUNNELED: return "MPI_THREAD_FUNNELED";
    case MPI_THREAD_SERIALIZED: return "MPI_THREAD_SERIALIZED";
    case MPI_THREAD_MULTIPLE: return "MPI_THREAD_MULTIPLE";
    default: return std::to_string(level);
  }
}

/// \brief The name of the GPU of the default execution space on this rank,
/// or the name of the execution space without a GPU
inline std::string device_name() {
  Kokkos::DefaultExecutionSpace exec;
#if defined(KOKKOS_ENABLE_CUDA)
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, exec.cuda_device()) == cudaSuccess) {
    return std::string(prop.name) + " (sm_" + std::to_string(prop.major) +
           std::to_string(prop.minor) + ")";
  }
#elif defined(KOKKOS_ENABLE_HIP)
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, exec.hip_device()) == hipSuccess) {
    return std::string(prop.name) + " (" + prop.gcnArchName + ")";
  }
#endif
  return exec.name();
}

/// \brief Add the MPI and backend information to the benchmark context:
/// the number of processes and nodes, the MPI library and thread level, the
/// communication backend of the native plans and its version, the vendor
/// plan backend, the SHMEM communication mode, and the GPU of rank 0.
/// Collective over comm: every rank must call it (only rank 0 reports).
/// \param[in] comm MPI communicator of the benchmarks
/// \param[in] thread_level The thread level that MPI_Init_thread provided
inline void add_mpi_benchmark_context(MPI_Comm comm, int thread_level) {
  int rank = 0, nprocs = 0;
  ::MPI_Comm_rank(comm, &rank);
  ::MPI_Comm_size(comm, &nprocs);

  // The number of nodes: the processes of local rank 0
  MPI_Comm node_comm;
  ::MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                        &node_comm);
  int node_rank = 0, node_size = 0, max_node_size = 0, nnodes = 0;
  ::MPI_Comm_rank(node_comm, &node_rank);
  ::MPI_Comm_size(node_comm, &node_size);
  int is_node_root = node_rank == 0 ? 1 : 0;
  ::MPI_Allreduce(&is_node_root, &nnodes, 1, MPI_INT, MPI_SUM, comm);
  ::MPI_Allreduce(&node_size, &max_node_size, 1, MPI_INT, MPI_MAX, comm);
  ::MPI_Comm_free(&node_comm);

  char library[MPI_MAX_LIBRARY_VERSION_STRING];
  int len = 0;
  ::MPI_Get_library_version(library, &len);
  std::string library_version(library, len);
  // The first line only (Open MPI and MPICH print several)
  library_version = library_version.substr(0, library_version.find('\n'));

  benchmark::AddCustomContext("MPI processes", std::to_string(nprocs));
  benchmark::AddCustomContext("MPI nodes", std::to_string(nnodes));
  benchmark::AddCustomContext("MPI processes per node (max)",
                              std::to_string(max_node_size));
  benchmark::AddCustomContext("MPI library", library_version);
  benchmark::AddCustomContext("MPI thread level",
                              thread_level_name(thread_level));
  benchmark::AddCustomContext("Device (rank 0)", device_name());

  benchmark::AddCustomContext("Communication backend", comm_backend_name());
#if defined(ENABLE_TPL_NVSHMEM) || defined(ENABLE_TPL_CUFFT_MP)
  benchmark::AddCustomContext(
      "NVSHMEM version", std::to_string(NVSHMEM_VENDOR_MAJOR_VERSION) + "." +
                             std::to_string(NVSHMEM_VENDOR_MINOR_VERSION) +
                             "." +
                             std::to_string(NVSHMEM_VENDOR_PATCH_VERSION));
#endif
#if defined(ENABLE_TPL_ROCSHMEM)
  std::string rocshmem_version = std::to_string(ROCSHMEM_VENDOR_MAJOR_VERSION) +
                                 "." +
                                 std::to_string(ROCSHMEM_VENDOR_MINOR_VERSION);
#if defined(ROCSHMEM_VENDOR_PATCH_VERSION)
  rocshmem_version += "." + std::to_string(ROCSHMEM_VENDOR_PATCH_VERSION);
#endif
  benchmark::AddCustomContext("rocSHMEM version", rocshmem_version);
#endif
#if defined(ENABLE_TPL_NVSHMEM) || defined(ENABLE_TPL_ROCSHMEM)
  // The mode the SHMEM communicators will use (the library reads the same
  // environment variables)
  using KokkosFFT::Distributed::Impl::SHMEMCommMode;
  const bool is_device =
      KokkosFFT::Distributed::Impl::default_shmem_comm_mode() ==
      SHMEMCommMode::device;
  benchmark::AddCustomContext("SHMEM communication mode",
                              is_device ? "device" : "host");
  if (is_device) {
    benchmark::AddCustomContext(
        "SHMEM fused pack",
        KokkosFFT::Distributed::Impl::default_shmem_fused_pack() ? "yes"
                                                                 : "no");
  }
#endif
#if defined(ENABLE_TPL_SHMEM_DEVICE_COMM)
  benchmark::AddCustomContext("SHMEM device communication compiled", "yes");
#endif

  benchmark::AddCustomContext("Vendor plan backend",
                              vendor_plan_backend_name());
#if defined(ENABLE_TPL_CUFFT_MP)
  benchmark::AddCustomContext("cuFFTMp version",
                              std::to_string(CUFFT_VER_MAJOR) + "." +
                                  std::to_string(CUFFT_VER_MINOR) + "." +
                                  std::to_string(CUFFT_VER_PATCH));
#endif
#if defined(PRIORITIZE_TPL_PLAN_IF_AVAILABLE)
  benchmark::AddCustomContext("Plan picks the vendor plan if available", "yes");
#else
  benchmark::AddCustomContext("Plan picks the vendor plan if available", "no");
#endif
}

}  // namespace Benchmark
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
