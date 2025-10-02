#ifndef KOKKOSFFT_DISTRIBUTED_ROCFFT_MPI_TRANSFORM_HPP
#define KOKKOSFFT_DISTRIBUTED_ROCFFT_MPI_TRANSFORM_HPP

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_Utils.hpp"

template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType>
inline void exec_plan(
    const ExecutionSpace& /*exec_space*/, const PlanType& scoped_plan,
    const InViewType& in, const OutViewType& out,
    const std::array<std::size_t, InViewType::rank()>& /*in_extents*/,
    const std::array<std::size_t, OutViewType::rank()>& /*out_extents*/,
    const KokkosFFT::axis_type<InViewType::rank()>& /*in_map*/,
    const KokkosFFT::axis_type<OutViewType::rank()>& /*out_map*/,
    KokkosFFT::Direction direction) {
  Kokkos::Profiling::ScopedRegion region("exec_plan[TPL_RocfftMPIExec]");
  rocfft_status status =
      rocfft_execute(scoped_plan.plan(direction), (void**)in.data(),
                     (void**)out.data(), nullptr);
  KOKKOSFFT_THROW_IF(status != rocfft_status_success, "rocfft_execute failed");
}

#endif
