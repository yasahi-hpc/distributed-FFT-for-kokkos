#ifndef KOKKOSFFT_DISTRIBUTED_ROCFFT_MPI_TRANSFORM_HPP
#define KOKKOSFFT_DISTRIBUTED_ROCFFT_MPI_TRANSFORM_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <KokkosFFT.hpp>

#include "KokkosFFT_Distributed_Helper.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType>
inline void exec_plan(
    [[maybe_unused]] const ExecutionSpace& exec_space,
    const PlanType& scoped_plan, const InViewType& in, const OutViewType& out,
    [[maybe_unused]] const std::array<std::size_t, InViewType::rank()>&
        in_extents,
    [[maybe_unused]] const std::array<std::size_t, OutViewType::rank()>&
        out_extents,
    const KokkosFFT::axis_type<InViewType::rank()>& in_map,
    const KokkosFFT::axis_type<OutViewType::rank()>& out_map,
    KokkosFFT::Direction direction) {
  bool is_transpose_needed = KokkosFFT::Impl::is_transpose_needed(in_map) ||
	                     KokkosFFT::Impl::is_transpose_needed(out_map);
  if (is_transpose_needed) {
    using in_value_type  = typename InViewType::non_const_value_type;
    using out_value_type = typename OutViewType::non_const_value_type;
    using in_data_type =
        KokkosFFT::Impl::add_pointer_n_t<in_value_type, InViewType::rank()>;
    using out_data_type =
        KokkosFFT::Impl::add_pointer_n_t<out_value_type, OutViewType::rank()>;
    using execution_space_type = typename InViewType::execution_space;
    using InRightViewType =
        Kokkos::View<in_data_type, Kokkos::LayoutRight, execution_space_type>;
    using OutRightViewType =
        Kokkos::View<out_data_type, Kokkos::LayoutRight, execution_space_type>;

    auto in_buffer_extents = KokkosFFT::Impl::extract_extents(in);
    auto in_buffer =
	scoped_plan.template buffer_data<InRightViewType>(in_buffer_extents);
    auto in_T =
	InRightViewType(in.data(),
	KokkosFFT::Impl::create_layout<Kokkos::LayoutRight>(in_extents));

    Kokkos::deep_copy(exec_space, in_buffer, in);
    KokkosFFT::Impl::transpose(exec_space, in_buffer, in_T, in_map, true);

    // FIXME remove fencee. need to bind stream to the plan which is not done yet
    exec_space.fence();
    auto out_buffer =
	scoped_plan.template buffer_data<OutRightViewType, OutViewType::rank()>(out_extents);
    {
      Kokkos::Profiling::ScopedRegion region("exec_plan[TPL_rocFFTMPIExec]");
      auto* in_data  = in.data();
      auto* out_data = out_buffer.data();
      KOKKOSFFT_CHECK_ROCFFT_CALL(rocfft_execute(scoped_plan.plan(direction),
			          (void**)&in_data,
                                  (void**)&out_data, nullptr));
    }
    KokkosFFT::Impl::transpose(exec_space, out_buffer, out, out_map, true);
  } else {
    Kokkos::Profiling::ScopedRegion region("exec_plan[TPL_rocFFTMPIExec]");
    auto* in_data  = in.data();
    auto* out_data = out.data();
    KOKKOSFFT_CHECK_ROCFFT_CALL(rocfft_execute(scoped_plan.plan(direction),
			        (void**)&in_data,
                                (void**)&out_data, nullptr));
  }
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
