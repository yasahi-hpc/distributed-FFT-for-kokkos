#ifndef KOKKOSFFT_DISTRIBUTED_ROCSHMEM_ALL2ALL_HPP
#define KOKKOSFFT_DISTRIBUTED_ROCSHMEM_ALL2ALL_HPP

#include <string>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_rocSHMEM_Types.hpp"
#include "KokkosFFT_Distributed_rocSHMEM_Comm.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief rocSHMEM all-to-all communication for distributed data
/// redistribution. It is enqueued on the stream of the execution space of
/// the communicator, after a barrier, so that every PE has finished reading
/// its receive buffer before a peer writes into it.
///
/// The barrier is on the team with rocSHMEM 3.5 or later. Before, it is on
/// the world team (rocshmem_barrier_all_on_stream): every rank of the SHMEM
/// communicator must then call the all-to-all at the same time, as the
/// transposes of a plan do.
///
/// The buffers must be on the symmetric heap (CommBuffer). This is not
/// checked: rocshmem_ptr is not documented for the calling PE.
///
/// \tparam ViewType Kokkos View type containing the data to be communicated,
/// must have rank >= 2
///
/// \param[in] send Input view to be sent, on the symmetric heap
/// \param[out] recv Output view to be received, on the symmetric heap
/// \param[in] scoped_comm rocSHMEM communicator wrapper
/// \throws std::runtime_error if the transposed extent does not match the
/// size of the communicator
template <typename ViewType>
void all2all(const ViewType& send, const ViewType& recv,
             const ScopedrocSHMEMComm& scoped_comm) {
  static_assert(ViewType::rank() >= 2,
                "all2all: View rank must be larger than or equal to 2");
  using LayoutType = typename ViewType::array_layout;
  using value_type = typename ViewType::non_const_value_type;
  std::string msg =
      KokkosFFT::Impl::is_real_v<value_type>
          ? "KokkosFFT::Distributed::all2all[TPL_ROCSHMEM,real]"
          : "KokkosFFT::Distributed::all2all[TPL_ROCSHMEM,complex]";
  Kokkos::Profiling::ScopedRegion region(msg);
  int size_send = std::is_same_v<LayoutType, Kokkos::LayoutLeft>
                      ? send.extent_int(ViewType::rank() - 1)
                      : send.extent_int(0);
  int size_recv = std::is_same_v<LayoutType, Kokkos::LayoutLeft>
                      ? recv.extent_int(ViewType::rank() - 1)
                      : recv.extent_int(0);

  int size               = scoped_comm.size();
  auto size_mismatch_msg = [size, size_send, size_recv]() -> std::string {
    std::string message;
    message = "Extent of dimension to be transposed of send (" +
              std::to_string(size_send) + ") or recv (" +
              std::to_string(size_recv) +
              ") buffer does not match MPI size: " + std::to_string(size);
    return message;
  };

  KOKKOSFFT_THROW_IF((size_send != size) || (size_recv != size),
                     size_mismatch_msg());

  // Bytes sent to each PE
  const std::size_t nbytes = send.size() / size_send * sizeof(value_type);
  auto stream              = scoped_comm.exec_space().hip_stream();

  // The on-stream calls return void: errors show up as HIP errors on the
  // stream
#if defined(KOKKOSFFT_ROCSHMEM_HAS_TEAM_BARRIER_ON_STREAM)
  rocshmem::rocshmem_barrier_on_stream(scoped_comm.team(), stream);
#else
  rocshmem::rocshmem_barrier_all_on_stream(stream);
#endif
  rocshmem::rocshmem_alltoallmem_on_stream(scoped_comm.team(), recv.data(),
                                           send.data(), nbytes, stream);
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
