#ifndef CUFFT_MP_TRANSFORM_HPP
#define CUFFT_MP_TRANSFORM_HPP

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "Utils.hpp"

template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType>
inline void exec_plan(
    const ExecutionSpace& exec_space, const PlanType& scoped_plan,
    const InViewType& in, const OutViewType& out,
    const std::array<std::size_t, InViewType::rank()>& in_extents,
    const std::array<std::size_t, OutViewType::rank()>& out_extents,
    const KokkosFFT::axis_type<InViewType::rank()>& in_map,
    const KokkosFFT::axis_type<OutViewType::rank()>& out_map,
    KokkosFFT::Direction direction) {
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
  Kokkos::Profiling::ScopedRegion region("exec_plan[TPL_cufftMpExec]");
  cudaLibXtDesc* desc = scoped_plan.desc();
  InRightViewType in_desc(
      reinterpret_cast<in_value_type*>(desc->descriptor->data[0]),
      KokkosFFT::Impl::create_layout<Kokkos::LayoutRight>(in_extents));
  OutRightViewType out_desc(
      reinterpret_cast<out_value_type*>(desc->descriptor->data[0]),
      KokkosFFT::Impl::create_layout<Kokkos::LayoutRight>(out_extents));

  safe_transpose(exec_space, in, in_desc, in_map);
  auto const cufft_direction = direction == KokkosFFT::Direction::forward
                                   ? CUFFT_FORWARD
                                   : CUFFT_INVERSE;
  cufftResult cufft_rt = cufftXtExecDescriptor(scoped_plan.plan(direction),
                                               desc, desc, cufft_direction);
  KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftXtExecDescriptor failed");
  safe_transpose(exec_space, out_desc, out, out_map);
}

#endif
