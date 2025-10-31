#ifndef KOKKOSFFT_DISTRIBUTED_MPI_COMM_HPP
#define KOKKOSFFT_DISTRIBUTED_MPI_COMM_HPP

#include <mpi.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_MPI_Types.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

struct ScopedMPIComm {
 private:
  MPI_Comm m_comm;
  int m_rank = 0;
  int m_size = 1;

 public:
  explicit ScopedMPIComm(MPI_Comm comm) : m_comm(comm) {
    ::MPI_Comm_rank(comm, &m_rank);
    ::MPI_Comm_size(comm, &m_size);
  }

  ScopedMPIComm() : ScopedMPIComm(MPI_COMM_WORLD) {}
  ~ScopedMPIComm() = default;

  MPI_Comm comm() const { return m_comm; }
  int rank() const { return m_rank; }
  int size() const { return m_size; }
};

using TplComm = ScopedMPIComm;

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
