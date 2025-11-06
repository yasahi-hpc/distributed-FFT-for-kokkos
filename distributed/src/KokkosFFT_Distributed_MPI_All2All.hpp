#ifndef KOKKOSFFT_DISTRIBUTED_MPI_ALL2ALL_HPP
#define KOKKOSFFT_DISTRIBUTED_MPI_ALL2ALL_HPP

#include <type_traits>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_MPI_Types.hpp"
#include "KokkosFFT_Distributed_MPI_Comm.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

template <typename ExecutionSpace, typename ViewType, typename CommType>
struct All2All;

/// \brief MPI all-to-all communication for distributed data redistribution
/// This class implements MPI_Alltoall communication pattern on Kokkos Views,
/// used in distributed FFT operations for data redistribution between different
/// domain decompositions. The class handles the computation of send counts
/// based on the layout type and performs the actual MPI communication.
///
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam ViewType Kokkos View type containing the data to be communicated,
/// must have rank >= 2
///
/// The outermost dimension corresponds to the number of processes involved in
/// the communication.
/// For LayoutLeft, we expect the input and output views to have the shape
/// (n0, n1, ..., nprocs).
/// For LayoutRight, we expect the input and output views to have the shape
/// (nprocs, n0, ..., n_N).
/// The send_count and recv_count are the product of the other dimensions of the
/// input and output views. It will raise an error if the nprocs obtained from
/// the input and output views is not the same as the MPI size.
template <typename ExecutionSpace, typename ViewType>
struct All2All<ExecutionSpace, ViewType, ScopedMPIComm<ExecutionSpace>> {
  static_assert(ViewType::rank() >= 2,
                "All2All: View rank must be larger than or equal to 2");
  using value_type = typename ViewType::non_const_value_type;

  ExecutionSpace m_exec_space;
  MPI_Comm m_comm;
  MPI_Datatype m_mpi_data_type;

  /// \brief Constructor for All2All communication
  /// \param[in] send Input view to be sent
  /// \param[in] recv Output view to be received
  /// \param[in] comm MPI communicator (default to MPI_COMM_WORLD)
  /// \param[in] exec_space Execution space (default to ExecutionSpace()
  /// instance)
  /// \throws std::runtime_error if the extent of the dimension to be transposed
  /// does not match MPI size
  All2All(const ViewType& send, const ViewType& recv,
          const ScopedMPIComm<ExecutionSpace>& comm,
          const ExecutionSpace exec_space = ExecutionSpace())
      : m_exec_space(exec_space),
        m_comm(comm.comm()),
        m_mpi_data_type(MPIDataType<value_type>::type()) {
    using LayoutType = typename ViewType::array_layout;
    std::string msg  = KokkosFFT::Impl::is_real_v<value_type>
                           ? "KokkosFFT::Distributed::MPI_all2all (real)"
                           : "KokkosFFT::Distributed::MPI_all2all (complex)";
    Kokkos::Profiling::ScopedRegion region(msg);
    int size_send = std::is_same_v<LayoutType, Kokkos::LayoutLeft>
                        ? send.extent_int(ViewType::rank() - 1)
                        : send.extent_int(0);
    int size_recv = std::is_same_v<LayoutType, Kokkos::LayoutLeft>
                        ? recv.extent_int(ViewType::rank() - 1)
                        : recv.extent_int(0);

    int size = comm.size();
    KOKKOSFFT_THROW_IF(
        (size_send != size) || (size_recv != size),
        "Extent of dimension to be transposed: " + std::to_string(size_send) +
            " does not match MPI size: " + std::to_string(size));

    // Compute the outermost dimension size
    int send_count = static_cast<int>(send.size()) / size_send;
    ::MPI_Alltoall(send.data(), send_count, m_mpi_data_type, recv.data(),
                   send_count, m_mpi_data_type, m_comm);
  }
};

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
