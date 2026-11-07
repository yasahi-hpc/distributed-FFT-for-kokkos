#ifndef KOKKOSFFT_DISTRIBUTED_ROCFFT_MPI_TRANSFORM_HPP
#define KOKKOSFFT_DISTRIBUTED_ROCFFT_MPI_TRANSFORM_HPP

#include <array>
#include <cstddef>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_RocfftMPI_Plan.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief Execute the rocFFT MPI plan for distributed FFT, on the stream of
/// the plan's execution info and with its work buffer
/// (ScopedRocfftMPIBidirectionalPlan::commit). Directly on the input and
/// output Views when they are LayoutRight in the order of the plan and do not
/// share their data. In-place, the input is first copied into the staging
/// buffer, and the plan runs from it into the output. Otherwise (a transpose
/// or a change of layout) through the plan's buffers: in is transposed into
/// the input buffer, the plan runs from it into the output buffer, which is
/// transposed into out. Out-of-place, the input View is never written. The
/// output is normalized by rocFFT (the scale factors of the plans).
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam PlanType ScopedRocfftMPIBidirectionalPlan type
/// \tparam InViewType Kokkos View type for input data
/// \tparam OutViewType Kokkos View type for output data
/// \tparam ScaleType Floating point type of the coefficient
///
/// \param[in] exec_space Kokkos execution space
/// \param[in] scoped_plan rocFFT MPI plans
/// \param[in] in Input data view
/// \param[out] out Output data view
/// \param[in] in_extents Extents of the input data, in the mapped order
/// \param[in] out_extents Extents of the output data, in the mapped order
/// \param[in] in_map Axis mapping for input data
/// \param[in] out_map Axis mapping for output data
/// \param[in] direction Direction of the FFT (forward/backward)
/// \param[in] coef Normalization coefficient (TplPlan, scales_in_transform):
/// not used, the plans of each direction apply theirs (create_plan)
/// \throws std::runtime_error if rocfft_execute fails
template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType, typename ScaleType>
inline void exec_plan(
    const ExecutionSpace& exec_space, const PlanType& scoped_plan,
    const InViewType& in, const OutViewType& out,
    const std::array<std::size_t, InViewType::rank()>& in_extents,
    const std::array<std::size_t, OutViewType::rank()>& out_extents,
    const KokkosFFT::axis_type<InViewType::rank()>& in_map,
    const KokkosFFT::axis_type<OutViewType::rank()>& out_map,
    KokkosFFT::Direction direction, ScaleType /*coef*/) {
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

  if (!is_rocfft_mpi_buffer_needed<InViewType, OutViewType>(in_map)) {
    if (check_inplace(in, out)) {
      // The plans are out-of-place: run from a copy of the input
      auto staging =
          scoped_plan.template staging_data<InRightViewType>(in_extents);
      Kokkos::deep_copy(exec_space, staging, in);
      Kokkos::Profiling::ScopedRegion region("exec_plan[TPL_rocFFTMPIExec]");
      auto* in_data  = staging.data();
      auto* out_data = out.data();
      KOKKOSFFT_CHECK_ROCFFT_CALL(
          rocfft_execute(scoped_plan.plan(direction), (void**)&in_data,
                         (void**)&out_data, scoped_plan.execution_info()));
    } else {
      Kokkos::Profiling::ScopedRegion region("exec_plan[TPL_rocFFTMPIExec]");
      auto* in_data  = in.data();
      auto* out_data = out.data();
      KOKKOSFFT_CHECK_ROCFFT_CALL(
          rocfft_execute(scoped_plan.plan(direction), (void**)&in_data,
                         (void**)&out_data, scoped_plan.execution_info()));
    }
    return;
  }

  auto in_buffer = scoped_plan.template buffer_data<InRightViewType>(
      direction, true, in_extents);
  auto out_buffer = scoped_plan.template buffer_data<OutRightViewType>(
      direction, false, out_extents);

  // A deep_copy (with the layout change) when the map is the identity. In
  // place, the input is fully read here before the output is written.
  KokkosFFT::Impl::transpose(exec_space, in, in_buffer, in_map, true);
  {
    Kokkos::Profiling::ScopedRegion region("exec_plan[TPL_rocFFTMPIExec]");
    auto* in_data  = in_buffer.data();
    auto* out_data = out_buffer.data();
    KOKKOSFFT_CHECK_ROCFFT_CALL(
        rocfft_execute(scoped_plan.plan(direction), (void**)&in_data,
                       (void**)&out_data, scoped_plan.execution_info()));
  }
  KokkosFFT::Impl::transpose(exec_space, out_buffer, out, out_map, true);
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
