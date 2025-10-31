#ifndef KOKKOSFFT_DISTRIBUTED_NCCL_COMM_HPP
#define KOKKOSFFT_DISTRIBUTED_NCCL_COMM_HPP

#include <cstdint>
#include <mpi.h>
#include "KokkosFFT_Distributed_NCCL_Types.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

struct ScopedNCCLComm {
  ncclComm_t m_comm;
  int m_rank = 0;
  int m_size = 1;

  ScopedNCCLComm(const MPI_Comm &comm) {
    ::MPI_Comm_rank(comm, &m_rank);
    ::MPI_Comm_size(comm, &m_size);

    ncclUniqueId id;
    if (m_rank == 0) ncclGetUniqueId(&id);
    ::MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, comm);
    ::MPI_Barrier(comm);
    ncclCommInitRank(&m_comm, m_size, id, m_rank);
  }

  ScopedNCCLComm() = delete;
  ~ScopedNCCLComm() { ncclCommDestroy(m_comm); }

  ncclComm_t comm() const { return m_comm; }
  int rank() const { return m_rank; }
  int size() const { return m_size; }
};

using TplComm = ScopedNCCLComm;

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
