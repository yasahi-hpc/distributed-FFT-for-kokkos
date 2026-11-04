#ifndef KOKKOSFFT_DISTRIBUTED_NVSHMEM_EXCHANGE_HPP
#define KOKKOSFFT_DISTRIBUTED_NVSHMEM_EXCHANGE_HPP

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_Exchange.hpp"
#include "KokkosFFT_Distributed_NVSHMEM_Types.hpp"
#include "KokkosFFT_Distributed_NVSHMEM_Comm.hpp"
#include "KokkosFFT_Distributed_NVSHMEM_All2All.hpp"

#if !defined(ENABLE_TPL_SHMEM_DEVICE_COMM)
static_assert(false,
              "The NVSHMEM device mode needs ENABLE_TPL_SHMEM_DEVICE_COMM "
              "(ENABLE_SHMEM_DEVICE_COMM) and relocatable device code.");
#endif

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief The put kernel of put_all2all: league rank p puts chunk p of the
/// send buffer into the slot of this PE in the receive buffer of the team PE
/// p, and adds 1 to the signal word of p once the data has arrived.
/// A functor, not a lambda: the members are only used on the device
/// (KOKKOS_IF_ON_DEVICE), which nvcc does not accept for the captures of an
/// extended lambda.
/// \tparam MemberType Kokkos team member type
template <typename MemberType>
struct PutChunks {
  const char* m_send;
  char* m_recv;
  std::size_t m_nbytes;
  std::uint64_t* m_signal;
  int m_me, m_start, m_stride;

  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& team) const {
    // A Kokkos team on CUDA is one block, as the block-scope put requires
    const int p = team.league_rank();
    KOKKOS_IF_ON_DEVICE(
        (nvshmemx_putmem_signal_nbi_block(
             m_recv + m_me * m_nbytes, m_send + p * m_nbytes, m_nbytes,
             m_signal, 1, NVSHMEM_SIGNAL_ADD, m_start + p * m_stride);))
    KOKKOS_IF_ON_HOST(((void)p;))
  }
};

/// \brief NVSHMEM all-to-all with device-initiated puts (device mode),
/// enqueued on the stream of exec_space:
/// 1. a barrier on the team: every PE has finished reading the receive
///    buffer that its peers are about to write into;
/// 2. a kernel with one team (CUDA block) per peer p, which puts chunk p of
///    the send buffer into the slot of this PE in the receive buffer of p,
///    and adds 1 to the signal word of p once the data has arrived;
/// 3. a quiet: the puts of this PE have completed, so that the send buffer
///    can be overwritten afterwards;
/// 4. a wait until the signal word has the value of this exchange (epoch):
///    every peer's chunk has arrived.
/// No kernel waits: the waits are enqueued on the stream by the host.
///
/// \tparam ExecutionSpace Kokkos execution space type (Kokkos::Cuda)
/// \tparam ViewType Kokkos View type containing the data to be communicated,
/// must have rank >= 2
/// \param[in] exec_space Kokkos execution space
/// \param[in] send Input view to be sent, on the symmetric heap
/// \param[out] recv Output view to be received, on the symmetric heap
/// \param[in] scoped_comm NVSHMEM communicator wrapper in device mode
/// \throws std::runtime_error if the wrapper is not in device mode, if the
/// transposed extent does not match the size of the communicator, or if a
/// buffer is not on the symmetric heap
template <typename ExecutionSpace, typename ViewType>
void put_all2all(const ExecutionSpace& exec_space, const ViewType& send,
                 const ViewType& recv, const ScopedNVSHMEMComm& scoped_comm) {
  using value_type  = typename ViewType::non_const_value_type;
  using policy_type = Kokkos::TeamPolicy<ExecutionSpace>;
  using member_type = typename policy_type::member_type;
  std::string msg =
      KokkosFFT::Impl::is_real_v<value_type>
          ? "KokkosFFT::Distributed::put_all2all[TPL_NVSHMEM,real]"
          : "KokkosFFT::Distributed::put_all2all[TPL_NVSHMEM,complex]";
  Kokkos::Profiling::ScopedRegion region(msg);

  KOKKOSFFT_THROW_IF(scoped_comm.mode() != SHMEMCommMode::device ||
                         scoped_comm.signal() == nullptr,
                     "put_all2all: the NVSHMEM communicator wrapper must be "
                     "in device mode");
  const std::size_t nbytes =
      check_nvshmem_all2all_buffers(send, recv, scoped_comm);

  auto stream                = exec_space.cuda_stream();
  const std::uint64_t target = scoped_comm.next_epoch();
  std::uint64_t* signal      = scoped_comm.signal();
  const PutChunks<member_type> put_chunks{
      reinterpret_cast<const char*>(send.data()),
      reinterpret_cast<char*>(recv.data()),
      nbytes,
      signal,
      scoped_comm.rank(),
      scoped_comm.triplet().start,
      scoped_comm.triplet().stride};

  KOKKOSFFT_CHECK_NVSHMEM_CALL(
      nvshmemx_barrier_on_stream(scoped_comm.team(), stream));
  Kokkos::parallel_for(
      "KokkosFFT::Distributed::put_chunks[TPL_NVSHMEM]",
      policy_type(exec_space, scoped_comm.size(), Kokkos::AUTO), put_chunks);
  nvshmemx_quiet_on_stream(stream);
  nvshmemx_signal_wait_until_on_stream(signal, NVSHMEM_CMP_GE, target, stream);
}

/// \brief The fused pack kernel of fused_put_all2all: Pack, except that the
/// chunk of a peer reachable by load/store is stored directly into the slot
/// of this PE in the peer's receive buffer. The other chunks are stored into
/// the local send buffer, as Pack does, for the puts of SignalChunks.
///
/// It iterates over the send buffer in memory order. The chunk dimension is
/// the outermost one (§4.2 of support-shmem.md), so the element at linear
/// index i is in chunk p = i / chunk, at offset i - p * chunk, and goes to
/// offset me * chunk + (i - p * chunk) of the receive buffer of p.
///
/// \tparam InViewType Kokkos View type of the input (source) data
/// \tparam BufferViewType Kokkos View type of the send/receive buffers
/// \tparam iType Integer type for indices
template <typename InViewType, typename BufferViewType, typename iType>
struct FusedPackChunks {
  using value_type   = typename BufferViewType::non_const_value_type;
  using LayoutType   = typename BufferViewType::array_layout;
  using PeersType    = Kokkos::View<std::uintptr_t*, Kokkos::CudaSpace>;
  using IndexMapType = PackIndexMap<LayoutType, iType, InViewType::rank()>;
  static constexpr std::size_t m_rank = BufferViewType::rank();

  InViewType m_in;
  value_type* m_send;
  PeersType m_peers;
  IndexMapType m_index_map;
  Kokkos::Array<iType, m_rank> m_extents;
  iType m_chunk, m_me;

  FusedPackChunks(const InViewType& in, const BufferViewType& send,
                  const PeersType& peers,
                  const typename IndexMapType::ShapeType& map, std::size_t axis,
                  std::size_t merged_size, int me, int npes)
      : m_in(in),
        m_send(send.data()),
        m_peers(peers),
        m_index_map(in, send, map, axis, merged_size),
        m_chunk(iType(send.size() / npes)),
        m_me(iType(me)) {
    for (std::size_t r = 0; r < m_rank; ++r) {
      m_extents[r] = iType(send.extent(r));
    }
  }

  KOKKOS_INLINE_FUNCTION void operator()(const iType i) const {
    // The indices of the element at linear index i, in memory order
    iType dst_idx[m_rank];
    iType rest = i;
    if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutRight>) {
      for (std::size_t r = m_rank; r-- > 0;) {
        dst_idx[r] = rest % m_extents[r];
        rest /= m_extents[r];
      }
    } else {
      for (std::size_t r = 0; r < m_rank; ++r) {
        dst_idx[r] = rest % m_extents[r];
        rest /= m_extents[r];
      }
    }

    const value_type value =
        get_src_value(dst_idx, std::make_index_sequence<m_rank - 1>{});
    const iType p = i / m_chunk;
    auto* peer    = reinterpret_cast<value_type*>(m_peers(p));
    if (peer != nullptr) {
      peer[m_me * m_chunk + (i - p * m_chunk)] = value;
    } else {
      m_send[i] = value;
    }
  }

  /// \brief The source value of a send buffer element, 0 for padding
  template <std::size_t... Is>
  KOKKOS_INLINE_FUNCTION value_type
  get_src_value(const iType dst_idx[], std::index_sequence<Is...>) const {
    iType src_idx[m_rank - 1];
    if (!m_index_map.src_indices(dst_idx, src_idx)) return value_type(0);
    return m_in(src_idx[Is]...);
  }
};

/// \brief The signal kernel of fused_put_all2all: league rank p signals the
/// team PE p. If p is reachable by load/store, its chunk was stored by
/// FusedPackChunks: one thread makes the stores visible (nvshmem_quiet) and
/// adds 1 to the signal word of p. Otherwise, the chunk is put from the send
/// buffer with the signal, as PutChunks does.
/// \tparam MemberType Kokkos team member type
template <typename MemberType>
struct SignalChunks {
  PutChunks<MemberType> m_put;
  Kokkos::View<std::uintptr_t*, Kokkos::CudaSpace> m_peers;

  KOKKOS_INLINE_FUNCTION void operator()(const MemberType& team) const {
    const int p = team.league_rank();
    if (m_peers(p) == 0) {
      m_put(team);
      return;
    }
    KOKKOS_IF_ON_DEVICE((Kokkos::single(Kokkos::PerTeam(team), [&]() {
                           nvshmem_quiet();
                           nvshmemx_signal_op(
                               m_put.m_signal, 1, NVSHMEM_SIGNAL_ADD,
                               m_put.m_start + p * m_put.m_stride);
                         });))
    KOKKOS_IF_ON_HOST(((void)p;))
  }
};

/// \brief NVSHMEM all-to-all fused with the pack (device mode, D2),
/// enqueued on the stream of exec_space:
/// 1. a barrier on the team: the receive buffers of the peers are free;
/// 2. the fused pack (FusedPackChunks): the chunks of the peers reachable by
///    load/store are stored into their receive buffers, the other chunks
///    into the send buffer;
/// 3. the signal kernel (SignalChunks): a signal for each stored chunk, a
///    put with signal for each other chunk;
/// 4. a quiet and a wait on the signal word, as in put_all2all.
///
/// \tparam ExecutionSpace Kokkos execution space type (Kokkos::Cuda)
/// \tparam InViewType Kokkos View type of the input data
/// \tparam ViewType Kokkos View type of the send/receive buffers
/// \tparam DIM Rank of the input view
/// \param[in] exec_space Kokkos execution space
/// \param[in] in Input view to be packed
/// \param[out] send Send buffer, on the symmetric heap
/// \param[out] recv Receive buffer, on the symmetric heap
/// \param[in] scoped_comm NVSHMEM communicator wrapper in device mode
/// \param[in] peers The peer table of recv (ScopedNVSHMEMComm::peer_table)
/// \param[in] src_map The map of the input view
/// \param[in] src_axis The axis to be split
/// \throws std::runtime_error as put_all2all
template <typename ExecutionSpace, typename InViewType, typename ViewType,
          std::size_t DIM>
void fused_put_all2all(const ExecutionSpace& exec_space, const InViewType& in,
                       const ViewType& send, const ViewType& recv,
                       const ScopedNVSHMEMComm& scoped_comm,
                       const NVSHMEMPeerTable& peers,
                       const std::array<std::size_t, DIM>& src_map,
                       std::size_t src_axis) {
  using value_type  = typename ViewType::non_const_value_type;
  using policy_type = Kokkos::TeamPolicy<ExecutionSpace>;
  using member_type = typename policy_type::member_type;
  static_assert(InViewType::rank() == DIM && ViewType::rank() == DIM + 1,
                "fused_put_all2all: the buffers must have the rank of the "
                "input view + 1");
  std::string msg =
      KokkosFFT::Impl::is_real_v<value_type>
          ? "KokkosFFT::Distributed::fused_put_all2all[TPL_NVSHMEM,real]"
          : "KokkosFFT::Distributed::fused_put_all2all[TPL_NVSHMEM,complex]";
  Kokkos::Profiling::ScopedRegion region(msg);

  KOKKOSFFT_THROW_IF(scoped_comm.mode() != SHMEMCommMode::device ||
                         scoped_comm.signal() == nullptr,
                     "fused_put_all2all: the NVSHMEM communicator wrapper "
                     "must be in device mode");
  KOKKOSFFT_THROW_IF(peers.m_buffer != recv.data(),
                     "fused_put_all2all: the peer table must be the one of "
                     "the receive buffer");
  const std::size_t nbytes =
      check_nvshmem_all2all_buffers(send, recv, scoped_comm);

  auto stream                = exec_space.cuda_stream();
  const std::uint64_t target = scoped_comm.next_epoch();
  std::uint64_t* signal      = scoped_comm.signal();
  const int me = scoped_comm.rank(), npes = scoped_comm.size();

  const Kokkos::Array<std::size_t, DIM> map =
      KokkosFFT::Impl::to_array(src_map);
  const std::size_t merged_size =
      in.extent(KokkosFFT::Impl::get_index(src_map, src_axis));

  KOKKOSFFT_CHECK_NVSHMEM_CALL(
      nvshmemx_barrier_on_stream(scoped_comm.team(), stream));

  // The same choice of the index type as pack
  auto fused_pack = [&](auto index) {
    using iType = decltype(index);
    Kokkos::parallel_for(
        "KokkosFFT::Distributed::fused_pack[TPL_NVSHMEM]",
        Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<iType>>(
            exec_space, 0, iType(send.size())),
        FusedPackChunks<InViewType, ViewType, iType>(
            in, send, peers.m_peers, map, src_axis, merged_size, me, npes));
  };
  if (in.span() >= std::size_t(std::numeric_limits<int>::max()) ||
      send.span() >= std::size_t(std::numeric_limits<int>::max())) {
    fused_pack(int64_t(0));
  } else {
    fused_pack(int(0));
  }

  const SignalChunks<member_type> signal_chunks{
      PutChunks<member_type>{reinterpret_cast<const char*>(send.data()),
                             reinterpret_cast<char*>(recv.data()), nbytes,
                             signal, me, scoped_comm.triplet().start,
                             scoped_comm.triplet().stride},
      peers.m_peers};
  Kokkos::parallel_for("KokkosFFT::Distributed::signal_chunks[TPL_NVSHMEM]",
                       policy_type(exec_space, npes, Kokkos::AUTO),
                       signal_chunks);
  nvshmemx_quiet_on_stream(stream);
  nvshmemx_signal_wait_until_on_stream(signal, NVSHMEM_CMP_GE, target, stream);
}

/// \brief Data exchange of a transpose with an NVSHMEM communicator wrapper.
/// In host mode, the default sequence (default_exchange). In device mode,
/// all on the stream of exec_space, without a fence:
/// - if the pack is fused (ScopedNVSHMEMComm::fused_pack) and a peer is
///   reachable by load/store: fused_put_all2all (D2), then unpack;
/// - otherwise: pack, put_all2all (D1), then unpack.
template <>
struct Exchange<ScopedNVSHMEMComm> {
  /// \brief Arguments as default_exchange
  template <typename ExecutionSpace, typename InViewType,
            typename BufferViewType, typename OutViewType, std::size_t DIM>
  static void apply(const ExecutionSpace& exec_space,
                    const ScopedNVSHMEMComm& comm, const InViewType& in,
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
    if (comm.fused_pack()) {
      const auto peers = comm.peer_table(recv.data());
      if (peers.m_has_peer) {
        fused_put_all2all(exec_space, in, send, recv, comm, peers, src_map,
                          src_axis);
        unpack(exec_space, recv, out, dst_map, dst_axis);
        return;
      }
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
