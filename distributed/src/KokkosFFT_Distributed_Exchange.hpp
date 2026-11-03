#ifndef KOKKOSFFT_DISTRIBUTED_EXCHANGE_HPP
#define KOKKOSFFT_DISTRIBUTED_EXCHANGE_HPP

#include <array>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_PackUnpack.hpp"
#include "KokkosFFT_Distributed_All2All.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief The default data exchange of a transpose, with host calls: pack,
/// fence, all2all and unpack. Used by every communication backend, and by
/// the host mode of the SHMEM backends.
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam CommType Type of the communicator wrapper
/// \tparam InViewType Type of the input view
/// \tparam BufferViewType Type of the send/receive buffers
/// \tparam OutViewType Type of the output view
/// \tparam DIM Rank of the input/output views
/// \param[in] exec_space Kokkos execution space
/// \param[in] comm Communicator wrapper
/// \param[in] in Input view
/// \param[out] send Send buffer
/// \param[out] recv Receive buffer
/// \param[out] out Output view
/// \param[in] src_map The map of the input view
/// \param[in] src_axis The axis to be split
/// \param[in] dst_map The map of the output view
/// \param[in] dst_axis The axis to be merged
template <typename ExecutionSpace, typename CommType, typename InViewType,
          typename BufferViewType, typename OutViewType, std::size_t DIM>
void default_exchange(const ExecutionSpace& exec_space, const CommType& comm,
                      const InViewType& in, const BufferViewType& send,
                      const BufferViewType& recv, const OutViewType& out,
                      const std::array<std::size_t, DIM>& src_map,
                      std::size_t src_axis,
                      const std::array<std::size_t, DIM>& dst_map,
                      std::size_t dst_axis) {
  pack(exec_space, in, send, src_map, src_axis);
  exec_space.fence();
  all2all(send, recv, comm);
  unpack(exec_space, recv, out, dst_map, dst_axis);
}

/// \brief Data exchange of a transpose: packs the input view into the send
/// buffer, exchanges the buffers between the processes of the communicator
/// and unpacks the receive buffer into the output view.
/// The default is default_exchange. A backend can specialize it for its
/// communicator wrapper, to replace the pack and the exchange together
/// (e.g. with kernels that write the chunks into the peers' buffers).
///
/// \tparam CommType Type of the communicator wrapper
template <typename CommType>
struct Exchange {
  /// \brief Arguments as default_exchange
  template <typename ExecutionSpace, typename InViewType,
            typename BufferViewType, typename OutViewType, std::size_t DIM>
  static void apply(const ExecutionSpace& exec_space, const CommType& comm,
                    const InViewType& in, const BufferViewType& send,
                    const BufferViewType& recv, const OutViewType& out,
                    const std::array<std::size_t, DIM>& src_map,
                    std::size_t src_axis,
                    const std::array<std::size_t, DIM>& dst_map,
                    std::size_t dst_axis) {
    default_exchange(exec_space, comm, in, send, recv, out, src_map, src_axis,
                     dst_map, dst_axis);
  }
};

/// \brief Run Exchange for a communicator wrapper
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam CommType Type of the communicator wrapper
/// \tparam InViewType Type of the input view
/// \tparam BufferViewType Type of the send/receive buffers
/// \tparam OutViewType Type of the output view
/// \tparam DIM Rank of the input/output views
/// \param[in] exec_space Kokkos execution space
/// \param[in] comm Communicator wrapper
/// \param[in] in Input view
/// \param[out] send Send buffer
/// \param[out] recv Receive buffer
/// \param[out] out Output view
/// \param[in] src_map The map of the input view
/// \param[in] src_axis The axis to be split
/// \param[in] dst_map The map of the output view
/// \param[in] dst_axis The axis to be merged
template <typename ExecutionSpace, typename CommType, typename InViewType,
          typename BufferViewType, typename OutViewType, std::size_t DIM>
void exchange(const ExecutionSpace& exec_space, const CommType& comm,
              const InViewType& in, const BufferViewType& send,
              const BufferViewType& recv, const OutViewType& out,
              const std::array<std::size_t, DIM>& src_map, std::size_t src_axis,
              const std::array<std::size_t, DIM>& dst_map,
              std::size_t dst_axis) {
  Exchange<CommType>::apply(exec_space, comm, in, send, recv, out, src_map,
                            src_axis, dst_map, dst_axis);
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

// The specializations of the device mode, after the primary template and
// before TransBlock uses them
#if defined(ENABLE_TPL_SHMEM_DEVICE_COMM) && defined(ENABLE_TPL_NVSHMEM)
#include "nvshmem/KokkosFFT_Distributed_NVSHMEM_Exchange.hpp"
#elif defined(ENABLE_TPL_SHMEM_DEVICE_COMM) && defined(ENABLE_TPL_ROCSHMEM)
#include "rocSHMEM/KokkosFFT_Distributed_rocSHMEM_Exchange.hpp"
#endif

#endif
