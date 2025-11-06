#ifndef KOKKOSFFT_DISTRIBUTED_MPI_COMM_HPP
#define KOKKOSFFT_DISTRIBUTED_MPI_COMM_HPP

#include <mpi.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_MPI_Types.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

template <typename ExecutionSpace>
struct ScopedMPIComm {
 private:
  using execution_space = ExecutionSpace;
  execution_space m_exec_space;
  MPI_Comm m_comm;
  int m_rank = 0;
  int m_size = 1;

 public:
  explicit ScopedMPIComm(MPI_Comm comm, const ExecutionSpace& exec_space)
      : m_comm(comm), m_exec_space(exec_space) {
    ::MPI_Comm_rank(comm, &m_rank);
    ::MPI_Comm_size(comm, &m_size);
  }
  explicit ScopedMPIComm(MPI_Comm comm)
      : ScopedMPIComm(comm, ExecutionSpace{}) {}

  ScopedMPIComm() : ScopedMPIComm(MPI_COMM_WORLD) {}
  ~ScopedMPIComm() = default;

  MPI_Comm comm() const { return m_comm; }
  execution_space exec_space() const { return m_exec_space; }
  int rank() const { return m_rank; }
  int size() const { return m_size; }
};

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
