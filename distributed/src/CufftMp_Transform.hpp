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
    KokkosFFT::Direction direction) {
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;
  using LayoutType     = typename InViewType::array_layout;
  Kokkos::Profiling::ScopedRegion region("exec_plan[TPL_cufftMpExec]");
  cudaLibXtDesc* desc = scoped_plan.desc();
  InViewType in_desc(
      reinterpret_cast<in_value_type*>(desc->descriptor->data[0]),
      KokkosFFT::Impl::create_layout<LayoutType>(in_extents));
  OutViewType out_desc(
      reinterpret_cast<out_value_type*>(desc->descriptor->data[0]),
      KokkosFFT::Impl::create_layout<LayoutType>(out_extents));

  Kokkos::deep_copy(exec_space, in_desc, in);
  auto const cufft_direction = direction == KokkosFFT::Direction::forward
                                   ? CUFFT_FORWARD
                                   : CUFFT_INVERSE;
  cufftResult cufft_rt = cufftXtExecDescriptor(scoped_plan.plan(direction),
                                               desc, desc, cufft_direction);
  KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftXtExecDescriptor failed");
  Kokkos::deep_copy(exec_space, out, out_desc);
}

#endif
