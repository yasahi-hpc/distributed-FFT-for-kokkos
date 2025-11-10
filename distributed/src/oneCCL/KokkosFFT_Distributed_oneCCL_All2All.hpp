#ifndef KOKKOSFFT_DISTRIBUTED_ONECCL_All2All_HPP
#define KOKKOSFFT_DISTRIBUTED_ONECCL_All2All_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_oneCCL_Types.hpp"
#include "KokkosFFT_Distributed_oneCCL_Comm.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief MPI all-to-all communication for distributed data redistribution
/// \tparam ViewType Kokkos View type containing the data to be communicated,
/// must have rank >= 2
///
/// \param[in] send Input view to be sent
/// \param[out] recv Output view to be received
/// \param[in] scoped_comm oneCCL communicator wrapper
template <typename ViewType>
void all2all(const ViewType& send, const ViewType& recv,
             const ScopedoneCCLComm& scoped_comm) {
  static_assert(ViewType::rank() >= 2,
                "all2all: View rank must be larger than or equal to 2");
  using value_type = typename ViewType::non_const_value_type;
  using floating_point_type =
      KokkosFFT::Impl::base_floating_point_type<value_type>;

  using LayoutType = typename ViewType::array_layout;
  int size_send    = std::is_same_v<LayoutType, Kokkos::LayoutLeft>
                         ? send.extent_int(ViewType::rank() - 1)
                         : send.extent_int(0);
  int size_recv    = std::is_same_v<LayoutType, Kokkos::LayoutLeft>
                         ? recv.extent_int(ViewType::rank() - 1)
                         : recv.extent_int(0);

  const auto& comms = scoped_comm.comms();
  auto stream       = scoped_comm.stream();
  int size          = scoped_comm.size();
  KOKKOSFFT_THROW_IF(
      (size_send != size) || (size_recv != size),
      "Extent of dimension to be transposed: " + std::to_string(size_send) +
          " does not match Comm size: " + std::to_string(size));

  // Compute the outermost dimension size
  // As oneCCL does not directly support complex data type,
  // we cast them in float data type
  const int size_factor = KokkosFFT::Impl::is_real_v<value_type> ? 1 : 2;
  int count       = size_factor * static_cast<int>(send.size()) / size_send;
  auto type       = oneCCLDataType<floating_point_type>::type();
  auto* send_data = reinterpret_cast<floating_point_type*>(send.data());
  auto* recv_data = reinterpret_cast<floating_point_type*>(recv.data());
  ccl::alltoall(send_data, recv_data, count, type, comms[0], stream).wait();
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
