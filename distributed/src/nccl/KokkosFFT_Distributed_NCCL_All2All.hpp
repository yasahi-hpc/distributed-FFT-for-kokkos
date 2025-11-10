#ifndef KOKKOSFFT_DISTRIBUTED_NCCL_All2All_HPP
#define KOKKOSFFT_DISTRIBUTED_NCCL_All2All_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_NCCL_Types.hpp"
#include "KokkosFFT_Distributed_NCCL_Comm.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief NCCL all-to-all communication for distributed data redistribution
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam ViewType Kokkos View type containing the data to be communicated,
/// must have rank >= 2
///
/// \param[in] send Input view to be sent
/// \param[out] recv Output view to be received
/// \param[in] scoped_comm NCCL communicator wrapper
template <typename ExecutionSpace, typename ViewType>
void all2all(const ViewType& send, const ViewType& recv,
             const ScopedNCCLComm<ExecutionSpace>& scoped_comm) {
  static_assert(ViewType::rank() >= 2,
                "all2all: View rank must be larger than or equal to 2");
  using LayoutType = typename ViewType::array_layout;
  using value_type = typename ViewType::non_const_value_type;
  using floating_point_type =
      KokkosFFT::Impl::base_floating_point_type<value_type>;
  std::string msg = KokkosFFT::Impl::is_real_v<value_type>
                        ? "KokkosFFT::Distributed::all2all[TPL_NCCL,real]"
                        : "KokkosFFT::Distributed::all2all[TPL_NCCL,complex]";
  Kokkos::Profiling::ScopedRegion region(msg);
  int size_send = std::is_same_v<LayoutType, Kokkos::LayoutLeft>
                      ? send.extent_int(ViewType::rank() - 1)
                      : send.extent_int(0);
  int size_recv = std::is_same_v<LayoutType, Kokkos::LayoutLeft>
                      ? recv.extent_int(ViewType::rank() - 1)
                      : recv.extent_int(0);

  auto comm = scoped_comm.comm();
  int size  = scoped_comm.size();
  KOKKOSFFT_THROW_IF(
      (size_send != size) || (size_recv != size),
      "Extent of dimension to be transposed: " + std::to_string(size_send) +
          " does not match Comm size: " + std::to_string(size));

  // Compute the outermost dimension size
  // As NCCL does not directly support complex data type,
  // we cast them in float data type
  const int size_factor = KokkosFFT::Impl::is_real_v<value_type> ? 1 : 2;
  int count       = size_factor * static_cast<int>(send.size()) / size_send;
  auto type       = NCCLDataType<floating_point_type>::type();
  auto* send_data = reinterpret_cast<floating_point_type*>(send.data());
  auto* recv_data = reinterpret_cast<floating_point_type*>(recv.data());

#if defined(KOKKOS_ENABLE_CUDA)
  auto stream = scoped_comm.exec_space().cuda_stream();
#elif defined(KOKKOS_ENABLE_HIP)
  auto stream = scoped_comm.exec_space().hip_stream();
#else
  static_assert(false,
                "You need to enable CUDA (HIP) backend to use NCCL (RCCL).");
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
  ncclResult_t res =
      ncclAlltoAll(send_data, recv_data, count, type, comm, stream);
  KOKKOSFFT_THROW_IF(res != ncclSuccess, "ncclAlltoAll failed");
#else
  ncclResult_t res = ncclGroupStart();
  KOKKOSFFT_THROW_IF(res != ncclSuccess, "ncclGroupStart failed");
  for (int r = 0; r < size; ++r) {
    res = ncclSend(send_data + r * count, count, type, r, comm, stream);
    KOKKOSFFT_THROW_IF(res != ncclSuccess, "ncclSend failed");
    res = ncclRecv(recv_data + r * count, count, type, r, comm, stream);
    KOKKOSFFT_THROW_IF(res != ncclSuccess, "ncclRecv failed");
  }
  res = ncclGroupEnd();
  KOKKOSFFT_THROW_IF(res != ncclSuccess, "ncclGroupEnd failed");
#endif
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
