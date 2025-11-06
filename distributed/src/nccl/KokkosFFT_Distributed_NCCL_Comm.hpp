#ifndef KOKKOSFFT_DISTRIBUTED_NCCL_COMM_HPP
#define KOKKOSFFT_DISTRIBUTED_NCCL_COMM_HPP

#include <cstdint>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_MPI_Comm.hpp"
#include "KokkosFFT_Distributed_NCCL_Types.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

template <typename ExecutionSpace>
struct ScopedNCCLComm {
 private:
  using execution_space = ExecutionSpace;
  execution_space m_exec_space;
  ncclComm_t m_comm;
  int m_rank = 0;
  int m_size = 1;

 public:
  explicit ScopedNCCLComm(const MPI_Comm &comm,
                          const execution_space &exec_space)
      : m_exec_space(exec_space) {
    ::MPI_Comm_rank(comm, &m_rank);
    ::MPI_Comm_size(comm, &m_size);

    ncclUniqueId id;
    if (m_rank == 0) ncclGetUniqueId(&id);
    ::MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, comm);
    ::MPI_Barrier(comm);
    ncclCommInitRank(&m_comm, m_size, id, m_rank);
  }
  explicit ScopedNCCLComm(const MPI_Comm &comm)
      : ScopedNCCLComm(comm, execution_space{}) {}

  ScopedNCCLComm() = delete;
  ~ScopedNCCLComm() { ncclCommDestroy(m_comm); }

  ncclComm_t comm() const { return m_comm; }
  execution_space exec_space() const { return m_exec_space; }
  int rank() const { return m_rank; }
  int size() const { return m_size; }
};

#if defined(KOKKOS_ENABLE_CUDA)
template <typename ExecutionSpace>
using TplComm = std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::Cuda>,
                                   ScopedNCCLComm<ExecutionSpace>,
                                   ScopedMPIComm<ExecutionSpace>>;
#elif defined(KOKKOS_ENABLE_HIP)
template <typename ExecutionSpace>
using TplComm = std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                                   ScopedNCCLComm<ExecutionSpace>,
                                   ScopedMPIComm<ExecutionSpace>>;
#else
static_assert(false,
              "You need to enable CUDA (HIP) backend to use NCCL (RCCL).");
#endif

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
