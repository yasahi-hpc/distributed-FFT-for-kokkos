#ifndef KOKKOSFFT_DISTRIBUTED_ROCFFT_MPI_PLAN_HPP
#define KOKKOSFFT_DISTRIBUTED_ROCFFT_MPI_PLAN_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>
#include <vector>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_MPI_Extents.hpp"
#include "KokkosFFT_Distributed_Topologies.hpp"
#include "KokkosFFT_Distributed_Extents.hpp"
#include "KokkosFFT_Distributed_TplAxes.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM, typename InLayoutType = Kokkos::LayoutRight,
          typename OutLayoutType = Kokkos::LayoutRight>
bool is_tpl_available(
    const ExecutionSpace& /*exec_space*/, const InViewType& /*in*/,
    const OutViewType& /*out*/, const KokkosFFT::axis_type<DIM>& /*axes*/,
    const Topology<std::size_t, InViewType::rank(), InLayoutType>& in_topology,
    const Topology<std::size_t, OutViewType::rank(), OutLayoutType>&
        out_topology) {
  // Any order of the axes and either layout (T4a): LayoutLeft Views run on
  // the reversed LayoutRight Views when that needs no transpose (no copy);
  // otherwise exec_plan transposes (and changes the layout) through the
  // buffers A and B. The Views must have the same layout.
  if constexpr (!std::is_same_v<typename InViewType::array_layout,
                                typename OutViewType::array_layout>) {
    return false;
  }

  // In-place transforms (in and out share their data) run through the
  // staging buffer (exec_plan), with the out-of-place plans: no rejection

  if constexpr ((InViewType::rank() == 2 && DIM == 2) ||
                (InViewType::rank() == 3 && DIM == 3)) {
    // The bricks describe any rectangular decomposition: slabs, pencils,
    // bricks, and pairs of different kinds (R4). Not on a single process
    // (Shared), where the internal plans make no communication, nor with an
    // empty or invalid topology.
    auto is_distributed = [](const auto& topology) {
      const auto topology_type = to_topology_type(topology);
      return topology_type == TopologyType::Slab ||
             topology_type == TopologyType::Pencil ||
             topology_type == TopologyType::Brick;
    };
    return is_distributed(in_topology) && is_distributed(out_topology);
  } else {
    return false;
  }
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

/// \brief Whether a rocFFT MPI plan must run on buffers in the mapped order
/// (ScopedRocfftMPIBidirectionalPlan's A and B) rather than on the Views: if
/// the axes need a transpose, or a View is not LayoutRight (the bricks are
/// described in the LayoutRight order of the mapped extents)
/// \tparam InViewType Kokkos View type for input data
/// \tparam OutViewType Kokkos View type for output data
/// \param[in] map Axis mapping
template <typename InViewType, typename OutViewType, std::size_t Rank>
bool is_rocfft_mpi_buffer_needed(const KokkosFFT::axis_type<Rank>& map) {
  constexpr bool is_layout_right =
      std::is_same_v<typename InViewType::array_layout, Kokkos::LayoutRight> &&
      std::is_same_v<typename OutViewType::array_layout, Kokkos::LayoutRight>;
  return !is_layout_right || KokkosFFT::Impl::is_transpose_needed(map);
}

template <typename ExecutionSpace, typename T,
          std::enable_if_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                           std::nullptr_t> = nullptr>
void setup(MPI_Comm /*comm*/) {
  KokkosFFT::Impl::setup<ExecutionSpace, T>();
}

// General interface. The topologies keep their layout, which decides the
// bricks each rank owns (compute_local_starts). The scale factors are the
// normalization of each direction, applied by rocFFT (R8).
template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType, std::size_t DIM, typename InLayoutType,
          typename OutLayoutType>
void create_plan(
    const ExecutionSpace& exec_space, std::unique_ptr<PlanType>& plan,
    const InViewType& in, const OutViewType& out,
    const KokkosFFT::axis_type<DIM>& axes, const KokkosFFT::axis_type<DIM>& map,
    const Topology<std::size_t, InViewType::rank(), InLayoutType>& in_topology,
    const Topology<std::size_t, OutViewType::rank(), OutLayoutType>&
        out_topology,
    const MPI_Comm& comm, double forward_scale = 1.0,
    double backward_scale = 1.0) {
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

    auto in_raw_extents  = KokkosFFT::Impl::extract_extents(in);
    auto out_raw_extents = KokkosFFT::Impl::extract_extents(out);

    auto in_raw_starts =
        compute_local_starts(in_raw_extents, in_topology, comm);
    auto out_raw_starts =
        compute_local_starts(out_raw_extents, out_topology, comm);

    auto in_extents =
        KokkosFFT::Impl::compute_mapped_extents(in_raw_extents, map);
    auto out_extents =
        KokkosFFT::Impl::compute_mapped_extents(out_raw_extents, map);
    auto in_starts =
        KokkosFFT::Impl::compute_mapped_extents(in_raw_starts, map);
    auto out_starts =
        KokkosFFT::Impl::compute_mapped_extents(out_raw_starts, map);

    std::array<std::size_t, DIM> in_ends{}, out_ends{};
    std::transform(in_starts.begin(), in_starts.end(), in_extents.begin(),
                   in_ends.begin(), std::plus<std::size_t>());
    std::transform(out_starts.begin(), out_starts.end(), out_extents.begin(),
                   out_ends.begin(), std::plus<std::size_t>());

    auto in_lower =
        KokkosFFT::Impl::reversed(KokkosFFT::Impl::to_vector(in_starts));
    auto in_upper =
        KokkosFFT::Impl::reversed(KokkosFFT::Impl::to_vector(in_ends));
    auto out_lower =
        KokkosFFT::Impl::reversed(KokkosFFT::Impl::to_vector(out_starts));
    auto out_upper =
        KokkosFFT::Impl::reversed(KokkosFFT::Impl::to_vector(out_ends));
    auto in_strides = KokkosFFT::Impl::to_vector(
        KokkosFFT::Impl::compute_strides(in_extents));
    auto out_strides = KokkosFFT::Impl::to_vector(
        KokkosFFT::Impl::compute_strides(out_extents));
    auto reversed_fft_extents =
        KokkosFFT::Impl::convert_int_type_and_reverse<std::size_t, std::size_t>(
            KokkosFFT::Impl::to_vector(fft_extents));

    // The plan runs directly on the input and output Views if they are in
    // its order (LayoutRight, no transpose) and do not share their data.
    // Otherwise, through buffers (exec_plan): with a transpose, two buffers
    // in the mapped order, A for the forward input, B for its output;
    // in-place, or for the backward (C2R) transform of an R2C plan, which may
    // overwrite its input, without a transpose: one staging buffer A for the
    // input of either direction.
    using in_value_type  = typename InViewType::non_const_value_type;
    using out_value_type = typename OutViewType::non_const_value_type;
    const std::size_t in_bytes =
        KokkosFFT::Impl::total_size(in_raw_extents) * sizeof(in_value_type);
    const std::size_t out_bytes =
        KokkosFFT::Impl::total_size(out_raw_extents) * sizeof(out_value_type);
    std::size_t in_buffer_size = 0, out_buffer_size = 0;
    if (is_rocfft_mpi_buffer_needed<InViewType, OutViewType>(map)) {
      in_buffer_size  = in_bytes;
      out_buffer_size = out_bytes;
    } else if (KokkosFFT::Impl::is_real_v<in_value_type> ||
               check_inplace(in, out)) {
      in_buffer_size = std::max(in_bytes, out_bytes);
    }
    plan = std::make_unique<PlanType>(
        reversed_fft_extents, in_lower, in_upper, out_lower, out_upper,
        in_strides, out_strides, in_buffer_size, out_buffer_size, forward_scale,
        backward_scale, exec_space.hip_device(), comm);
    plan->commit(exec_space);
  } else {
    KOKKOSFFT_THROW_IF(true, "Plan can be made for 2D or 3D Views only");
  }
}

/// \brief Create the rocFFT MPI plans with LayoutRight topologies
template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType, std::size_t DIM>
void create_plan(
    const ExecutionSpace& exec_space, std::unique_ptr<PlanType>& plan,
    const InViewType& in, const OutViewType& out,
    const KokkosFFT::axis_type<DIM>& axes, const KokkosFFT::axis_type<DIM>& map,
    const std::array<std::size_t, InViewType::rank()>& in_topology,
    const std::array<std::size_t, OutViewType::rank()>& out_topology,
    const MPI_Comm& comm, double forward_scale = 1.0,
    double backward_scale = 1.0) {
  create_plan(exec_space, plan, in, out, axes, map,
              Topology<std::size_t, InViewType::rank()>(in_topology),
              Topology<std::size_t, OutViewType::rank()>(out_topology), comm,
              forward_scale, backward_scale);
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
