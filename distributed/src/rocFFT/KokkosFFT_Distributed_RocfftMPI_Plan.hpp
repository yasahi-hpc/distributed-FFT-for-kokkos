#ifndef KOKKOSFFT_DISTRIBUTED_ROCFFT_MPI_PLAN_HPP
#define KOKKOSFFT_DISTRIBUTED_ROCFFT_MPI_PLAN_HPP

#include <vector>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_MPI_Extents.hpp"
#include "KokkosFFT_Distributed_Topologies.hpp"
#include "KokkosFFT_Distributed_Extents.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM, typename InLayoutType = Kokkos::LayoutRight,
          typename OutLayoutType = Kokkos::LayoutRight>
bool is_tpl_available(
    const ExecutionSpace& /*exec_space*/, const InViewType& in,
    const OutViewType& /*out*/, const KokkosFFT::axis_type<DIM>& axes,
    const Topology<std::size_t, InViewType::rank(), InLayoutType>& in_topology,
    const Topology<std::size_t, OutViewType::rank(), OutLayoutType>&
        out_topology) {
  using InLayout  = typename InViewType::array_layout;
  using OutLayout = typename OutViewType::array_layout;

  if constexpr (!std::is_same_v<InLayout, Kokkos::LayoutRight> ||
                !std::is_same_v<OutLayout, Kokkos::LayoutRight>) {
    return false;
  }

  [[maybe_unused]] auto [map, map_inv] =
      KokkosFFT::Impl::get_map_axes(in, axes);
  bool is_transpose_needed = KokkosFFT::Impl::is_transpose_needed(map);
  if (is_transpose_needed) return false;

  bool is_slab =
      are_specified_topologies(TopologyType::Slab, in_topology, out_topology);
  bool is_pencil =
      are_specified_topologies(TopologyType::Pencil, in_topology, out_topology);

  if constexpr (InViewType::rank() == 2 && DIM == 2) {
    if (is_slab) return true;
  } else if constexpr (InViewType::rank() == 3 && DIM == 3) {
    if (is_slab || is_pencil) return true;
  }

  return false;
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM>
bool is_tpl_available(
    const ExecutionSpace& exec_space, const InViewType& in,
    const OutViewType& out, const KokkosFFT::axis_type<DIM>& axes,
    const std::array<std::size_t, InViewType::rank()>& in_topology,
    const std::array<std::size_t, OutViewType::rank()>& out_topology) {
  return is_tpl_available(
      exec_space, in, out, axes,
      Topology<std::size_t, InViewType::rank()>(in_topology),
      Topology<std::size_t, OutViewType::rank()>(out_topology));
}

// General interface
template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType, std::size_t DIM>
void create_plan(const ExecutionSpace& exec_space,
                 std::unique_ptr<PlanType>& plan, const InViewType& in,
                 const OutViewType& out, const KokkosFFT::axis_type<DIM>& axes,
                 const KokkosFFT::axis_type<DIM>& map,
                 const KokkosFFT::shape_type<InViewType::rank()>& in_topology,
                 const KokkosFFT::shape_type<InViewType::rank()>& out_topology,
                 const MPI_Comm& comm) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "create_plan: InViewType and OutViewType must have the same base "
      "floating point type (float/double), the same layout "
      "(LayoutLeft/LayoutRight), "
      "and the same rank. ExecutionSpace must be accessible to the data in "
      "InViewType and OutViewType.");
  if constexpr (InViewType::rank() == DIM) {
    auto non_negative_axes =
        KokkosFFT::Impl::convert_base_int_type<std::size_t>(
            KokkosFFT::Impl::convert_negative_axes(axes, DIM));
    auto gin_extents  = compute_global_extents(in, in_topology, comm);
    auto gout_extents = compute_global_extents(out, out_topology, comm);
    auto fft_extents =
        compute_fft_extents(gin_extents, gout_extents, non_negative_axes);

    auto in_raw_extents = KokkosFFT::Impl::extract_extents(in);
    auto out_raw_extents = KokkosFFT::Impl::extract_extents(out);

    auto in_raw_starts = compute_local_starts(
      in_raw_extents, in_topology, comm);
    auto out_raw_starts = compute_local_starts(
      out_raw_extents, out_topology, comm);

    auto in_extents = compute_mapped_extents(in_raw_extents, map);
    auto out_extents = compute_mapped_extents(out_raw_extents, map);
    auto in_starts = compute_mapped_extents(in_raw_starts, map);
    auto out_starts = compute_mapped_extents(out_raw_starts, map);

    std::array<std::size_t, DIM> in_ends{}, out_ends{};
    std::transform(in_starts.begin(), in_starts.end(), in_extents.begin(),
                   in_ends.begin(), std::plus<std::size_t>());
    std::transform(out_starts.begin(), out_starts.end(), out_extents.begin(),
                   out_ends.begin(), std::plus<std::size_t>());

    auto in_lower  = KokkosFFT::Impl::reversed(KokkosFFT::Impl::to_vector(in_starts));
    auto in_upper  = KokkosFFT::Impl::reversed(KokkosFFT::Impl::to_vector(in_ends));
    auto out_lower = KokkosFFT::Impl::reversed(KokkosFFT::Impl::to_vector(out_starts));
    auto out_upper = KokkosFFT::Impl::reversed(KokkosFFT::Impl::to_vector(out_ends));
    auto in_strides = KokkosFFT::Impl::to_vector(
        KokkosFFT::Impl::compute_strides(in_extents));
    auto out_strides = KokkosFFT::Impl::to_vector(
        KokkosFFT::Impl::compute_strides(out_extents));
    auto reversed_fft_extents =
        KokkosFFT::Impl::convert_int_type_and_reverse<std::size_t, std::size_t>(
            KokkosFFT::Impl::to_vector(fft_extents));

    std::size_t buffer_size = 0;
    if (KokkosFFT::Impl::is_transpose_needed(map)) {
      using in_value_type  = typename InViewType::non_const_value_type;
      constexpr bool is_R2C = KokkosFFT::Impl::is_real_v<in_value_type>;
      // We allocate a buffer based on the output size
      // Input can be either real or complex, but Output is always complex
      auto in_size = KokkosFFT::Impl::extent_after_transform(KokkosFFT::Impl::total_size(in_raw_extents), is_R2C);
      auto out_size = KokkosFFT::Impl::total_size(out_raw_extents);

      // allocation is made in complex type
      buffer_size = std::max(in_size, out_size);
    }
    plan = std::make_unique<PlanType>(reversed_fft_extents, in_lower, in_upper,
                                      out_lower, out_upper, in_strides,
                                      out_strides, buffer_size, comm);
    plan->commit(exec_space);
  } else {
    KOKKOSFFT_THROW_IF(true, "Plan can be made for 2D or 3D Views only");
  }
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
