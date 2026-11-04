#ifndef KOKKOSFFT_DISTRIBUTED_ROCSHMEM_EXCHANGE_HPP
#define KOKKOSFFT_DISTRIBUTED_ROCSHMEM_EXCHANGE_HPP

#include <array>
#include <cstdint>
#include <string>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_Exchange.hpp"
#include "KokkosFFT_Distributed_rocSHMEM_Types.hpp"
#include "KokkosFFT_Distributed_rocSHMEM_Comm.hpp"
#include "KokkosFFT_Distributed_rocSHMEM_All2All.hpp"

#if !defined(ENABLE_TPL_SHMEM_DEVICE_COMM)
static_assert(false,
              "The rocSHMEM device mode needs ENABLE_TPL_SHMEM_DEVICE_COMM "
              "(ENABLE_SHMEM_DEVICE_COMM) and relocatable device code.");
#endif

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief The put kernel of rocshmem_put_all2all: league rank p puts chunk p
/// of the send buffer into the slot of this PE in the receive buffer of the
/// team PE p, and adds 1 to the signal word of p once the data has arrived.
/// The put is the blocking work-group one: it returns once the send buffer
/// can be reused, so the kernel ends with the local completion of its puts,
/// without a quiet on the stream (rocshmem_quiet_on_stream, from rocSHMEM
/// 3.4, would allow the non-blocking put instead). It does not wait for any
/// peer.
/// A functor, not a lambda: the members are only used on the device
/// (KOKKOS_IF_ON_DEVICE), which the extended lambdas do not accept.
/// \tparam MemberType Kokkos team member type
template <typename MemberType>
struct rocSHMEMPutChunks {
  const char* m_send;
  char* m_recv;
  std::size_t m_nbytes;
  std::uint64_t* m_signal;
  int m_me, m_start, m_stride;

  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& team) const {
    // A Kokkos team on HIP is one work-group, as the work-group put requires
    const int p = team.league_rank();
    KOKKOS_IF_ON_DEVICE((
        rocshmem::rocshmem_putmem_signal_wg(
            m_recv + m_me * m_nbytes, m_send + p * m_nbytes, m_nbytes, m_signal,
            1, rocshmem::ROCSHMEM_SIGNAL_ADD, m_start + p * m_stride);))
    KOKKOS_IF_ON_HOST(((void)p;))
  }
};

/// \brief rocSHMEM all-to-all with device-initiated puts (device mode),
/// enqueued on the stream of exec_space:
/// 1. a barrier (rocshmem_team_barrier_on_stream): every PE has finished
///    reading the receive buffer that its peers are about to write into;
/// 2. a kernel with one team (HIP work-group) per peer p, which puts chunk p
///    of the send buffer into the slot of this PE in the receive buffer of
///    p, with a signal (rocSHMEMPutChunks);
/// 3. a wait until the signal word has the value of this exchange (epoch):
///    every peer's chunk has arrived.
/// No kernel waits for another PE: the wait is enqueued on the stream.
///
/// \tparam ExecutionSpace Kokkos execution space type (Kokkos::HIP)
/// \tparam ViewType Kokkos View type containing the data to be communicated,
/// must have rank >= 2
/// \param[in] exec_space Kokkos execution space
/// \param[in] send Input view to be sent, on the symmetric heap
/// \param[out] recv Output view to be received, on the symmetric heap
/// \param[in] scoped_comm rocSHMEM communicator wrapper in device mode
/// \throws std::runtime_error if the wrapper is not in device mode, or if
/// the transposed extent does not match the size of the communicator
template <typename ExecutionSpace, typename ViewType>
void put_all2all(const ExecutionSpace& exec_space, const ViewType& send,
                 const ViewType& recv, const ScopedrocSHMEMComm& scoped_comm) {
  using value_type  = typename ViewType::non_const_value_type;
  using policy_type = Kokkos::TeamPolicy<ExecutionSpace>;
  using member_type = typename policy_type::member_type;
  std::string msg =
      KokkosFFT::Impl::is_real_v<value_type>
          ? "KokkosFFT::Distributed::put_all2all[TPL_ROCSHMEM,real]"
          : "KokkosFFT::Distributed::put_all2all[TPL_ROCSHMEM,complex]";
  Kokkos::Profiling::ScopedRegion region(msg);

  KOKKOSFFT_THROW_IF(scoped_comm.mode() != SHMEMCommMode::device ||
                         scoped_comm.signal() == nullptr,
                     "put_all2all: the rocSHMEM communicator wrapper must be "
                     "in device mode");
  const std::size_t nbytes =
      check_rocshmem_all2all_buffers(send, recv, scoped_comm);

  auto stream                = exec_space.hip_stream();
  const std::uint64_t target = scoped_comm.next_epoch();
  std::uint64_t* signal      = scoped_comm.signal();
  const rocSHMEMPutChunks<member_type> put_chunks{
      reinterpret_cast<const char*>(send.data()),
      reinterpret_cast<char*>(recv.data()),
      nbytes,
      signal,
      scoped_comm.rank(),
      scoped_comm.triplet().start,
      scoped_comm.triplet().stride};

  rocshmem_team_barrier_on_stream(scoped_comm, stream);
  Kokkos::parallel_for(
      "KokkosFFT::Distributed::put_chunks[TPL_ROCSHMEM]",
      policy_type(exec_space, scoped_comm.size(), Kokkos::AUTO), put_chunks);
  rocshmem::rocshmem_signal_wait_until_on_stream(
      signal, rocshmem::ROCSHMEM_CMP_GE, target, stream);
}

/// \brief Data exchange of a transpose with a rocSHMEM communicator wrapper.
/// In host mode, the default sequence (default_exchange). In device mode,
/// pack, the device-initiated all-to-all (put_all2all) and unpack, all on
/// the stream of exec_space, without a fence. The fused pack of the NVSHMEM
/// device mode (D2) is not implemented for rocSHMEM.
template <>
struct Exchange<ScopedrocSHMEMComm> {
  /// \brief Arguments as default_exchange
  template <typename ExecutionSpace, typename InViewType,
            typename BufferViewType, typename OutViewType, std::size_t DIM>
  static void apply(const ExecutionSpace& exec_space,
                    const ScopedrocSHMEMComm& comm, const InViewType& in,
                    const BufferViewType& send, const BufferViewType& recv,
                    const OutViewType& out,
                    const std::array<std::size_t, DIM>& src_map,
                    std::size_t src_axis,
                    const std::array<std::size_t, DIM>& dst_map,
                    std::size_t dst_axis) {
    if (comm.mode() == SHMEMCommMode::host) {
      default_exchange(exec_space, comm, in, send, recv, out, src_map, src_axis,
                       dst_map, dst_axis);
      return;
    }
    pack(exec_space, in, send, src_map, src_axis);
    put_all2all(exec_space, send, recv, comm);
    unpack(exec_space, recv, out, dst_map, dst_axis);
  }
};

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
