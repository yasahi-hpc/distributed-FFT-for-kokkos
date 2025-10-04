#ifndef KOKKOSFFT_DISTRIBUTED_ALL2ALL_HPP
#define KOKKOSFFT_DISTRIBUTED_ALL2ALL_HPP

#include <type_traits>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_MPI_Types.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

template <typename ExecutionSpace, typename ViewType>
struct All2All {
  static_assert(ViewType::rank() >= 2);
  using value_type   = typename ViewType::non_const_value_type;
  using LayoutType   = typename ViewType::array_layout;
  using extents_type = KokkosFFT::shape_type<ViewType::rank()>;

  ExecutionSpace m_exec_space;
  int m_send_count = 0;
  MPI_Comm m_comm;
  MPI_Datatype m_mpi_data_type;
  extents_type m_send_extents, m_recv_extents;

  All2All(const ViewType& send, const ViewType& recv,
          MPI_Comm comm                   = MPI_COMM_WORLD,
          const ExecutionSpace exec_space = ExecutionSpace())
      : m_exec_space(exec_space),
        m_comm(comm),
        m_mpi_data_type(MPIDataType<value_type>::type()) {
    // Compute the outermost dimension size
    if (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
      m_send_count = send.size() / send.extent(ViewType::rank() - 1);
    } else {
      m_send_count = send.size() / send.extent(0);
    }

    m_send_extents = KokkosFFT::Impl::extract_extents(send);
    m_recv_extents = KokkosFFT::Impl::extract_extents(recv);
  }

  void operator()(const ViewType& send, const ViewType& recv) const {
    Kokkos::Profiling::ScopedRegion region("All2All");
    auto send_extents = KokkosFFT::Impl::extract_extents(send);
    auto recv_extents = KokkosFFT::Impl::extract_extents(recv);

    KOKKOSFFT_THROW_IF(send_extents != m_send_extents,
                       "extents of input View for plan and "
                       "execution are not identical.");

    KOKKOSFFT_THROW_IF(recv_extents != m_recv_extents,
                       "extents of output View for plan and "
                       "execution are not identical.");

    ::MPI_Alltoall(send.data(), m_send_count, m_mpi_data_type, recv.data(),
                   m_send_count, m_mpi_data_type, m_comm);
  }
};

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
