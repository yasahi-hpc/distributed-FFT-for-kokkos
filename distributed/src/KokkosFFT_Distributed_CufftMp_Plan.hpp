#ifndef KOKKOSFFT_DISTRIBUTED_CUFFT_MP_PLAN_HPP
#define KOKKOSFFT_DISTRIBUTED_CUFFT_MP_PLAN_HPP

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
    const ExecutionSpace& exec_space, const InViewType& in,
    const OutViewType& out, const KokkosFFT::axis_type<DIM>& axes,
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

// Helper to compute strides from extents
// (n0, n1, n2) -> (1, n0, n0*n1)
// (n0, n1) -> (1, n0)
// (n0) -> (1)
template <std::size_t DIM = 1>
auto compute_strides(const std::array<std::size_t, DIM>& extents) {
  std::array<std::size_t, DIM> strides;

  auto reversed_extents = extents;
  std::reverse(reversed_extents.begin(), reversed_extents.end());

  strides.at(0) = 1;
  for (std::size_t i = 1; i < reversed_extents.size(); i++) {
    strides.at(i) = strides.at(i - 1) * reversed_extents.at(i - 1);
  }

  // Need to reverse back to original order
  std::reverse(strides.begin(), strides.end());

  return strides;
}

// General interface
template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType, std::size_t DIM>
std::size_t create_plan(
    const ExecutionSpace& exec_space, std::unique_ptr<PlanType>& plan,
    const InViewType& in, const OutViewType& out,
    const KokkosFFT::axis_type<DIM>& axes, const KokkosFFT::axis_type<DIM>& map,
    const KokkosFFT::shape_type<InViewType::rank()>& in_topology,
    const KokkosFFT::shape_type<InViewType::rank()>& out_topology,
    const MPI_Comm& comm) {
  KOKKOSFFT_THROW_IF(true, "Plan can be made for 2D or 3D Views only");
  return 1;
}

// 2D transform
template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType>
std::size_t create_plan(const ExecutionSpace& exec_space,
                        std::unique_ptr<PlanType>& plan, const InViewType& in,
                        const OutViewType& out,
                        const KokkosFFT::axis_type<2>& axes,
                        const KokkosFFT::axis_type<2>& map,
                        const KokkosFFT::shape_type<2>& in_topology,
                        const KokkosFFT::shape_type<2>& out_topology,
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

  KOKKOSFFT_THROW_IF(in_topology == out_topology,
                     "Input and output topologies must not be identical");

  Kokkos::Profiling::ScopedRegion region("KokkosFFT::create_plan[TPL_CufftMp]");

  auto gin_extents       = get_global_shape(in, in_topology, comm);
  auto gout_extents      = get_global_shape(out, out_topology, comm);
  auto non_negative_axes = KokkosFFT::Impl::convert_base_int_type<std::size_t>(
      KokkosFFT::Impl::convert_negative_axes(axes, 2));
  auto fft_extents =
      get_fft_extents(gin_extents, gout_extents, non_negative_axes);
  const auto [nx, ny]  = fft_extents;
  std::size_t fft_size = std::accumulate(fft_extents.begin(), fft_extents.end(),
                                         1, std::multiplies<std::size_t>());

  auto last_axis          = axes.back();
  auto in_first_dim       = in_topology.at(last_axis),
       out_first_dim      = out_topology.at(last_axis);
  bool is_first_dim_ready = in_first_dim == 1 && out_first_dim == 1;
  if (is_first_dim_ready) {
    auto in_mapped_topology = get_mapped_extents(in_topology, map);
    bool is_xslab           = in_mapped_topology.at(0) > 1;

    plan = std::make_unique<PlanType>(nx, ny, comm, is_xslab);
    plan->commit(exec_space);
  } else {
    // Using general API
    auto gin_padded_extents = gin_extents;
    if (KokkosFFT::Impl::is_real_v<in_value_type>) {
      gin_padded_extents = get_padded_extents(gout_extents, non_negative_axes);
    }

    auto [in_extents, in_starts] =
        get_local_extents(gin_extents, in_topology, comm);
    auto [out_extents, out_starts] =
        get_local_extents(gout_extents, out_topology, comm);

    auto [in_padded_extents, in_padded_starts] =
        get_local_extents(gin_padded_extents, in_topology, comm);

    std::array<std::size_t, 2> in_ends, out_ends;
    std::transform(in_starts.begin(), in_starts.end(), in_extents.begin(),
                   in_ends.begin(), std::plus<std::size_t>());
    std::transform(out_starts.begin(), out_starts.end(), out_extents.begin(),
                   out_ends.begin(), std::plus<std::size_t>());

    // Mapping based of axes
    auto mapped_in_extents        = get_mapped_extents(in_extents, map);
    auto mapped_in_padded_extents = get_mapped_extents(in_padded_extents, map);
    auto mapped_out_extents       = get_mapped_extents(out_extents, map);
    auto mapped_in_starts         = get_mapped_extents(in_starts, map);
    auto mapped_out_starts        = get_mapped_extents(out_starts, map);
    auto mapped_in_ends           = get_mapped_extents(in_ends, map);
    auto mapped_out_ends          = get_mapped_extents(out_ends, map);

    auto in_strides  = compute_strides(mapped_in_padded_extents);
    auto out_strides = compute_strides(mapped_out_extents);

    using int_vec_type  = std::vector<int>;
    using long_vec_type = std::vector<long long int>;

    int_vec_type fft_int_extents(fft_extents.begin(), fft_extents.end());
    long_vec_type lower_input(mapped_in_starts.begin(), mapped_in_starts.end()),
        upper_input(mapped_in_ends.begin(), mapped_in_ends.end()),
        lower_output(mapped_out_starts.begin(), mapped_out_starts.end()),
        upper_output(mapped_out_ends.begin(), mapped_out_ends.end()),
        strides_input(in_strides.begin(), in_strides.end()),
        strides_output(out_strides.begin(), out_strides.end());

    plan = std::make_unique<PlanType>(fft_int_extents, lower_input, upper_input,
                                      lower_output, upper_output, strides_input,
                                      strides_output, comm);
    plan->commit(exec_space);
  }

  return fft_size;
}

// 3D transform
template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType>
std::size_t create_plan(const ExecutionSpace& exec_space,
                        std::unique_ptr<PlanType>& plan, const InViewType& in,
                        const OutViewType& out,
                        const KokkosFFT::axis_type<3>& axes,
                        const KokkosFFT::axis_type<3>& map,
                        const KokkosFFT::shape_type<3>& in_topology,
                        const KokkosFFT::shape_type<3>& out_topology,
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

  KOKKOSFFT_THROW_IF(in_topology == out_topology,
                     "Input and output topologies must not be identical");

  Kokkos::Profiling::ScopedRegion region("KokkosFFT::create_plan[TPL_CufftMp]");

  auto gin_extents       = get_global_shape(in, in_topology, comm);
  auto gout_extents      = get_global_shape(out, out_topology, comm);
  auto non_negative_axes = KokkosFFT::Impl::convert_base_int_type<std::size_t>(
      KokkosFFT::Impl::convert_negative_axes(axes, 3));
  auto fft_extents =
      get_fft_extents(gin_extents, gout_extents, non_negative_axes);
  const auto [nx, ny, nz] = fft_extents;
  std::size_t fft_size = std::accumulate(fft_extents.begin(), fft_extents.end(),
                                         1, std::multiplies<std::size_t>());

  // In case of slab geometry, we need to check that the first dimension is
  // ready
  auto last_axis          = axes.back();
  auto in_first_dim       = in_topology.at(last_axis),
       out_first_dim      = out_topology.at(last_axis);
  bool is_first_dim_ready = in_first_dim == 1 && out_first_dim == 1;
  bool is_slab =
      is_slab_topology(in_topology) && is_slab_topology(out_topology);

  if (is_slab && is_first_dim_ready) {
    auto in_mapped_topology = get_mapped_extents(in_topology, map);
    bool is_xslab           = in_mapped_topology.at(0) > 1;

    plan = std::make_unique<PlanType>(nx, ny, nz, comm, is_xslab);
    plan->commit(exec_space);
  } else {
    // Using general API
    auto gin_padded_extents = gin_extents;
    if (KokkosFFT::Impl::is_real_v<in_value_type>) {
      gin_padded_extents = get_padded_extents(gout_extents, non_negative_axes);
    }

    auto [in_extents, in_starts] =
        get_local_extents(gin_extents, in_topology, comm);
    auto [out_extents, out_starts] =
        get_local_extents(gout_extents, out_topology, comm);

    auto [in_padded_extents, in_padded_starts] =
        get_local_extents(gin_padded_extents, in_topology, comm);

    std::array<std::size_t, 3> in_ends, out_ends;
    std::transform(in_starts.begin(), in_starts.end(), in_extents.begin(),
                   in_ends.begin(), std::plus<std::size_t>());
    std::transform(out_starts.begin(), out_starts.end(), out_extents.begin(),
                   out_ends.begin(), std::plus<std::size_t>());

    // Mapping based of axes
    auto mapped_in_extents        = get_mapped_extents(in_extents, map);
    auto mapped_in_padded_extents = get_mapped_extents(in_padded_extents, map);
    auto mapped_out_extents       = get_mapped_extents(out_extents, map);
    auto mapped_in_starts         = get_mapped_extents(in_starts, map);
    auto mapped_out_starts        = get_mapped_extents(out_starts, map);
    auto mapped_in_ends           = get_mapped_extents(in_ends, map);
    auto mapped_out_ends          = get_mapped_extents(out_ends, map);

    auto in_strides  = compute_strides(mapped_in_padded_extents);
    auto out_strides = compute_strides(mapped_out_extents);

    using int_vec_type  = std::vector<int>;
    using long_vec_type = std::vector<long long int>;

    int_vec_type fft_int_extents(fft_extents.begin(), fft_extents.end());
    long_vec_type lower_input(mapped_in_starts.begin(), mapped_in_starts.end()),
        upper_input(mapped_in_ends.begin(), mapped_in_ends.end()),
        lower_output(mapped_out_starts.begin(), mapped_out_starts.end()),
        upper_output(mapped_out_ends.begin(), mapped_out_ends.end()),
        strides_input(in_strides.begin(), in_strides.end()),
        strides_output(out_strides.begin(), out_strides.end());

    plan = std::make_unique<PlanType>(fft_int_extents, lower_input, upper_input,
                                      lower_output, upper_output, strides_input,
                                      strides_output, comm);
    plan->commit(exec_space);
  }

  return fft_size;
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
