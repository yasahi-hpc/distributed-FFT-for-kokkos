#ifndef KOKKOSFFT_DISTRIBUTED_ROCFFT_MPI_PLAN_HPP
#define KOKKOSFFT_DISTRIBUTED_ROCFFT_MPI_PLAN_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
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

  // 2D and 3D FFTs, of Views of the same rank or with one more dimension,
  // the batch dimension of the bricks (R7, T4b)
  constexpr std::size_t rank = InViewType::rank();
  if constexpr ((DIM == 2 || DIM == 3) && (rank == DIM || rank == DIM + 1)) {
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

/// \brief Whether the rocFFT plans may describe permuted or non-LayoutRight
/// Views with their own strides (strided bricks, T4b) instead of
/// transposing them through buffers: KOKKOSFFT_ROCFFT_STRIDED_BRICKS=1. Off
/// by default until rocFFT is known to accept such bricks for fields.
inline bool is_rocfft_strided_bricks_enabled() {
  const char* value = std::getenv("KOKKOSFFT_ROCFFT_STRIDED_BRICKS");
  return value != nullptr && std::string(value) == "1";
}

/// \brief The strides of the brick of a View, in elements, fastest first: the
/// dimension i of the brick is the mapped dimension Rank - 1 - i, that is the
/// dimension map[Rank - 1 - i] of the View
/// \tparam ViewType Kokkos View type
/// \param[in] view The View
/// \param[in] map Axis mapping (the order of the plan)
template <typename ViewType, std::size_t Rank>
std::vector<std::size_t> brick_strides(const ViewType& view,
                                       const KokkosFFT::axis_type<Rank>& map) {
  std::vector<std::size_t> strides(Rank);
  for (std::size_t i = 0; i < Rank; ++i) {
    strides.at(i) = static_cast<std::size_t>(view.stride(map.at(Rank - 1 - i)));
  }
  return strides;
}

template <typename ExecutionSpace, typename T,
          std::enable_if_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                           std::nullptr_t> = nullptr>
void setup(MPI_Comm /*comm*/) {
  KokkosFFT::Impl::setup<ExecutionSpace, T>();
}

// General interface. The topologies keep their layout, which decides the
// bricks each rank owns (compute_local_starts). The scale factors are the
// normalization of each direction, applied by rocFFT (R8). The Views have
// the rank of the FFT, or one more dimension: the batch (R7), which the map
// puts first (the slowest), and which is the last dimension of the bricks.
template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType, std::size_t DIM, typename InLayoutType,
          typename OutLayoutType>
void create_plan(
    const ExecutionSpace& exec_space, std::unique_ptr<PlanType>& plan,
    const InViewType& in, const OutViewType& out,
    const KokkosFFT::axis_type<DIM>& axes,
    const KokkosFFT::axis_type<InViewType::rank()>& map,
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
  constexpr std::size_t Rank = InViewType::rank();
  if constexpr (Rank == DIM || Rank == DIM + 1) {
    auto non_negative_axes =
        KokkosFFT::Impl::convert_base_int_type<std::size_t>(
            KokkosFFT::Impl::convert_negative_axes(axes, Rank));
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

    std::array<std::size_t, Rank> in_ends{}, out_ends{};
    std::transform(in_starts.begin(), in_starts.end(), in_extents.begin(),
                   in_ends.begin(), std::plus<std::size_t>());
    std::transform(out_starts.begin(), out_starts.end(), out_extents.begin(),
                   out_ends.begin(), std::plus<std::size_t>());

    // The bricks, fastest first: the FFT dimensions, then the batch (the
    // first mapped dimension, or one transform without a batch dimension)
    auto in_lower =
        KokkosFFT::Impl::reversed(KokkosFFT::Impl::to_vector(in_starts));
    auto in_upper =
        KokkosFFT::Impl::reversed(KokkosFFT::Impl::to_vector(in_ends));
    auto out_lower =
        KokkosFFT::Impl::reversed(KokkosFFT::Impl::to_vector(out_starts));
    auto out_upper =
        KokkosFFT::Impl::reversed(KokkosFFT::Impl::to_vector(out_ends));
    auto reversed_fft_extents =
        KokkosFFT::Impl::convert_int_type_and_reverse<std::size_t, std::size_t>(
            KokkosFFT::Impl::to_vector(fft_extents));

    // The plan runs directly on the input and output Views if they are in
    // its order (LayoutRight, no transpose) and do not share their data. A
    // permutation or another layout goes through buffers (two, A for the
    // forward input, B for its output, in the mapped order), unless the
    // bricks describe the Views with their own strides (strided bricks,
    // opt-in, out-of-place and contiguous Views only). In-place, or for the
    // backward (C2R) transform of an R2C plan, which may overwrite its
    // input, without the two buffers: one staging buffer A for the input of
    // either direction.
    using in_value_type  = typename InViewType::non_const_value_type;
    using out_value_type = typename OutViewType::non_const_value_type;
    const bool is_buffer_needed =
        is_rocfft_mpi_buffer_needed<InViewType, OutViewType>(map);
    const bool is_strided = is_buffer_needed &&
                            is_rocfft_strided_bricks_enabled() &&
                            !check_inplace(in, out) &&
                            in.span_is_contiguous() && out.span_is_contiguous();

    auto in_strides  = is_strided
                           ? brick_strides(in, map)
                           : KokkosFFT::Impl::to_vector(
                                KokkosFFT::Impl::compute_strides(in_extents));
    auto out_strides = is_strided
                           ? brick_strides(out, map)
                           : KokkosFFT::Impl::to_vector(
                                 KokkosFFT::Impl::compute_strides(out_extents));

    std::size_t number_of_transforms = 1;
    if constexpr (Rank == DIM) {
      // One transform: a batch dimension of extent 1
      for (auto* brick : {&in_lower, &out_lower}) brick->push_back(0);
      for (auto* brick : {&in_upper, &out_upper}) brick->push_back(1);
      for (auto* brick : {&in_strides, &out_strides}) brick->push_back(0);
    } else {
      // The global batch: the extent of the batch dimension of the View
      number_of_transforms =
          gin_extents.at(static_cast<std::size_t>(map.at(0)));
    }

    const std::size_t in_bytes =
        KokkosFFT::Impl::total_size(in_raw_extents) * sizeof(in_value_type);
    const std::size_t out_bytes =
        KokkosFFT::Impl::total_size(out_raw_extents) * sizeof(out_value_type);
    std::size_t in_buffer_size = 0, out_buffer_size = 0;
    if (is_buffer_needed && !is_strided) {
      in_buffer_size  = in_bytes;
      out_buffer_size = out_bytes;
    } else if (KokkosFFT::Impl::is_real_v<in_value_type> ||
               check_inplace(in, out)) {
      in_buffer_size = std::max(in_bytes, out_bytes);
    }
    plan = std::make_unique<PlanType>(
        reversed_fft_extents, in_lower, in_upper, out_lower, out_upper,
        in_strides, out_strides, in_buffer_size, out_buffer_size,
        number_of_transforms, is_strided, forward_scale, backward_scale,
        exec_space.hip_device(), comm);
    plan->commit(exec_space);
  } else {
    KOKKOSFFT_THROW_IF(true,
                       "Plan can be made for Views of the rank of the FFT or "
                       "one more (the batch) only");
  }
}

/// \brief Create the rocFFT MPI plans with LayoutRight topologies
template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType, std::size_t DIM>
void create_plan(
    const ExecutionSpace& exec_space, std::unique_ptr<PlanType>& plan,
    const InViewType& in, const OutViewType& out,
    const KokkosFFT::axis_type<DIM>& axes,
    const KokkosFFT::axis_type<InViewType::rank()>& map,
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
