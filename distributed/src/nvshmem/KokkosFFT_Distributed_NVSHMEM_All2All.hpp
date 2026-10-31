#ifndef KOKKOSFFT_DISTRIBUTED_NVSHMEM_ALL2ALL_HPP
#define KOKKOSFFT_DISTRIBUTED_NVSHMEM_ALL2ALL_HPP

#include <string>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_NVSHMEM_Types.hpp"
#include "KokkosFFT_Distributed_NVSHMEM_Comm.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief Whether ptr is on the NVSHMEM symmetric heap
inline bool is_nvshmem_symmetric(const void* ptr) {
  return ptr != nullptr && nvshmem_ptr(ptr, nvshmem_my_pe()) != nullptr;
}

/// \brief NVSHMEM all-to-all communication for distributed data
/// redistribution. It is enqueued on the stream of the execution space of
/// the communicator, after a barrier on the team, so that every PE has
/// finished reading its receive buffer before a peer writes into it.
/// \tparam ViewType Kokkos View type containing the data to be communicated,
/// must have rank >= 2
///
/// \param[in] send Input view to be sent, on the symmetric heap
/// \param[out] recv Output view to be received, on the symmetric heap
/// \param[in] scoped_comm NVSHMEM communicator wrapper
/// \throws std::runtime_error if the transposed extent does not match the
/// size of the communicator, or if a buffer is not on the symmetric heap
template <typename ViewType>
void all2all(const ViewType& send, const ViewType& recv,
             const ScopedNVSHMEMComm& scoped_comm) {
  static_assert(ViewType::rank() >= 2,
                "all2all: View rank must be larger than or equal to 2");
  using LayoutType = typename ViewType::array_layout;
  using value_type = typename ViewType::non_const_value_type;
  std::string msg =
      KokkosFFT::Impl::is_real_v<value_type>
          ? "KokkosFFT::Distributed::all2all[TPL_NVSHMEM,real]"
          : "KokkosFFT::Distributed::all2all[TPL_NVSHMEM,complex]";
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
  KOKKOSFFT_THROW_IF(
      !is_nvshmem_symmetric(send.data()) || !is_nvshmem_symmetric(recv.data()),
      "all2all: NVSHMEM buffers must be allocated on the symmetric heap "
      "(CommBuffer)");

  // Bytes sent to each PE
  const std::size_t nbytes = send.size() / size_send * sizeof(value_type);
  auto stream              = scoped_comm.exec_space().cuda_stream();

  KOKKOSFFT_CHECK_NVSHMEM_CALL(
      nvshmemx_barrier_on_stream(scoped_comm.team(), stream));
  KOKKOSFFT_CHECK_NVSHMEM_CALL(nvshmemx_alltoallmem_on_stream(
      scoped_comm.team(), recv.data(), send.data(), nbytes, stream));
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
