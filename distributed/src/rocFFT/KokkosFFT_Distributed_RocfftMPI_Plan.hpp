#ifndef KOKKOSFFT_DISTRIBUTED_ROCFFT_MPI_PLAN_HPP
#define KOKKOSFFT_DISTRIBUTED_ROCFFT_MPI_PLAN_HPP

#include <vector>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_MPI_Helper.hpp"
#include "KokkosFFT_Distributed_Topologies.hpp"
#include "KokkosFFT_Distributed_Utils.hpp"

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

  auto in_topo = in_topology.array(), out_topo = out_topology.array();

  if (in_topo == out_topo) return false;

  bool is_slab   = is_slab_topology(in_topo) && is_slab_topology(out_topo);
  bool is_pencil = is_pencil_topology(in_topo) && is_pencil_topology(out_topo);

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

// Helper to convert the integer type of vectors
template <typename InType, typename OutType>
auto convert_int_type_and_reverse(const std::vector<InType>& in)
    -> std::vector<OutType> {
  std::vector<OutType> out(in.size());
  std::transform(
      in.cbegin(), in.cend(), out.begin(),
      [](const InType v) -> OutType { return static_cast<OutType>(v); });

  std::reverse(out.begin(), out.end());
  return out;
}

// General interface
template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType, std::size_t DIM>
std::size_t create_plan(
    const ExecutionSpace& exec_space, std::unique_ptr<PlanType>& plan,
    const InViewType& in, const OutViewType& out,
    const KokkosFFT::axis_type<DIM>& axes,
    const KokkosFFT::axis_type<DIM>& /*map*/,
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
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;
  using LayoutType     = typename InViewType::array_layout;

  if constexpr (InViewType::rank() == DIM) {
    auto gin_extents  = get_global_shape(in, in_topology, comm);
    auto gout_extents = get_global_shape(out, out_topology, comm);

    auto non_negative_axes =
        KokkosFFT::Impl::convert_base_int_type<std::size_t>(
            KokkosFFT::Impl::convert_negative_axes(axes, DIM));

    auto fft_extents =
        get_fft_extents(gin_extents, gout_extents, non_negative_axes);
    auto fft_size = KokkosFFT::Impl::total_size(fft_extents);

    auto [in_extents, in_starts] =
        get_local_extents(gin_extents, in_topology, comm);
    auto [out_extents, out_starts] =
        get_local_extents(gout_extents, out_topology, comm);
    std::array<std::size_t, DIM> in_ends = {}, out_ends = {};
    std::transform(in_starts.begin(), in_starts.end(), in_extents.begin(),
                   in_ends.begin(), std::plus<std::size_t>());
    std::transform(out_starts.begin(), out_starts.end(), out_extents.begin(),
                   out_ends.begin(), std::plus<std::size_t>());

    auto in_lower  = to_vector(in_starts);
    auto in_upper  = to_vector(in_ends);
    auto out_lower = to_vector(out_starts);
    auto out_upper = to_vector(out_ends);

    auto in_strides = to_vector(KokkosFFT::Impl::compute_strides(gin_extents));
    auto out_strides =
        to_vector(KokkosFFT::Impl::compute_strides(gout_extents));
    auto reversed_fft_extents =
        convert_int_type_and_reverse<std::size_t, std::size_t>(
            to_vector(fft_extents));
    plan = std::make_unique<PlanType>(reversed_fft_extents, in_lower, in_upper,
                                      out_lower, out_upper, in_strides,
                                      out_strides, comm);
    plan->commit(exec_space);
    return fft_size;
  } else {
    KOKKOSFFT_THROW_IF(true, "Plan can be made for 2D or 3D Views only");
    return 1;
  }
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
