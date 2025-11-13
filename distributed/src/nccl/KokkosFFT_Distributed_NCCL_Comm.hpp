#ifndef KOKKOSFFT_DISTRIBUTED_NCCL_COMM_HPP
#define KOKKOSFFT_DISTRIBUTED_NCCL_COMM_HPP

#include <cstdint>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_MPI_Comm.hpp"
#include "KokkosFFT_Distributed_NCCL_Types.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

struct ScopedNCCLComm {
 private:
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  ExecutionSpace m_exec_space;
  ncclComm_t m_comm;
  int m_rank = 0;
  int m_size = 1;

 public:
  explicit ScopedNCCLComm(MPI_Comm comm, const ExecutionSpace &exec_space)
      : m_exec_space(exec_space) {
    ::MPI_Comm_rank(comm, &m_rank);
    ::MPI_Comm_size(comm, &m_size);

    ncclUniqueId id;
    if (m_rank == 0) ncclGetUniqueId(&id);
    ::MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, comm);
    ::MPI_Barrier(comm);
    ncclResult_t res = ncclCommInitRank(&m_comm, m_size, id, m_rank);
    KOKKOSFFT_THROW_IF(res != ncclSuccess, "ncclCommInitRank failed");
  }
  explicit ScopedNCCLComm(MPI_Comm comm)
      : ScopedNCCLComm(comm, ExecutionSpace{}) {}

  ScopedNCCLComm() = delete;

  // Delete copy semantics
  ScopedNCCLComm(const ScopedNCCLComm &)            = delete;
  ScopedNCCLComm &operator=(const ScopedNCCLComm &) = delete;

  // Allow move semantics
  ScopedNCCLComm(ScopedNCCLComm &&other) noexcept
      : m_exec_space(other.m_exec_space),
        m_comm(other.m_comm),
        m_rank(other.m_rank),
        m_size(other.m_size) {
    other.m_comm = nullptr;
  }

  // Move assignment operator
  ScopedNCCLComm &operator=(ScopedNCCLComm &&other) noexcept {
    if (this != &other) {
      if (m_comm) {
        ncclResult_t res = ncclCommDestroy(m_comm);
        KOKKOSFFT_THROW_IF(res != ncclSuccess, "ncclCommDestroy failed");
        m_comm = nullptr;
      }
      m_exec_space = other.m_exec_space;
      m_comm       = other.m_comm;
      m_rank       = other.m_rank;
      m_size       = other.m_size;
      other.m_comm = nullptr;
    }
    return *this;
  }

  ~ScopedNCCLComm() {
    ncclResult_t res = ncclCommDestroy(m_comm);
    if (res != ncclSuccess) Kokkos::abort("ncclCommDestroy failed");
    m_comm = nullptr;
  }

  ncclComm_t comm() const { return m_comm; }
  ExecutionSpace exec_space() const { return m_exec_space; }
  int rank() const { return m_rank; }
  int size() const { return m_size; }
};

#if defined(KOKKOS_ENABLE_CUDA)
template <typename ExecutionSpace>
using TplComm =
    std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::Cuda>,
                       ScopedNCCLComm, ScopedMPIComm<ExecutionSpace>>;
#elif defined(KOKKOS_ENABLE_HIP)
template <typename ExecutionSpace>
using TplComm =
    std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                       ScopedNCCLComm, ScopedMPIComm<ExecutionSpace>>;
#else
static_assert(false,
              "You need to enable CUDA (HIP) backend to use NCCL (RCCL).");
#endif

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
