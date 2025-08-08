#ifndef CUFFT_MP_PLAN_HPP
#define CUFFT_MP_PLAN_HPP

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "MPI_Helper.hpp"
#include "Topologies.hpp"
#include "Utils.hpp"

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

  bool is_slab   = is_slab_topology(in_topo) && is_slab_topology(out_topo);
  bool is_pencil = is_pencil_topology(in_topo) && is_pencil_topology(out_topo);

  if constexpr (InViewType::rank() == 2 && DIM == 2) {
    if (is_slab) return true;
  } else if constexpr (InViewType::rank() == 3 && DIM == 3) {
    if (is_slab || is_pencil) return true;
  }

  return false;
}

// General interface
template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType, std::size_t DIM>
std::size_t create_plan(
    const ExecutionSpace& exec_space, std::unique_ptr<PlanType>& plan,
    const InViewType& in, const OutViewType& out,
    const KokkosFFT::axis_type<DIM>& axes,
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
  using LayoutType     = typename InViewType::array_layout;

  Kokkos::Profiling::ScopedRegion region("KokkosFFT::create_plan[TPL_CufftMp]");

  auto gin_extents       = get_global_shape(in, in_topology, comm);
  auto gout_extents      = get_global_shape(out, out_topology, comm);
  auto non_negative_axes = convert_negative_axes<std::size_t, int, 2, 2>(axes);
  auto fft_extents =
      get_fft_extents(gin_extents, gout_extents, non_negative_axes);
  const auto [nx, ny]  = fft_extents;
  std::size_t fft_size = std::accumulate(fft_extents.begin(), fft_extents.end(),
                                         1, std::multiplies<std::size_t>());

  KOKKOSFFT_THROW_IF(in_topology == out_topology,
                     "Input and output topologies must not be identical");
  bool is_xslab = in_topology.at(0) > 1;

  plan = std::make_unique<PlanType>(nx, ny, comm, is_xslab);
  plan->commit(exec_space);

  return fft_size;
}

// 3D transform
template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType>
std::size_t create_plan(const ExecutionSpace& exec_space,
                        std::unique_ptr<PlanType>& plan, const InViewType& in,
                        const OutViewType& out,
                        const KokkosFFT::axis_type<3>& axes,
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

  /*
  Kokkos::Profiling::ScopedRegion region("KokkosFFT::create_plan[TPL_CufftMp]");
  auto type = KokkosFFT::Impl::transform_type<ExecutionSpace, in_value_type,
                                              out_value_type>::type();

  auto gin_extents  = get_global_shape(in, in_topology, comm);
  auto gout_extents = get_global_shape(out, out_topology, comm);

  auto non_negative_axes =
        convert_negative_axes<std::size_t, int, 3, 3>(axes);
  auto fft_extents = get_fft_extents(gin_extents, gout_extents,
  non_negative_axes); const auto [nx, ny, nz] = fft_extents; std::size_t
  fft_size = std::accumulate(fft_extents.begin(), fft_extents.end(), 1,
                                 std::multiplies<std::size_t>());

  KOKKOSFFT_THROW_IF(in_topology == out_topology, "Input and output topologies
  must not be identical");

  auto mapped_in_topology = get_mapped_extents(in_topology, axes);

  // In case of slab geometry, we need to check that the first dimension is
  ready auto last_axis = axes.back(); auto in_first_dim =
  in_topology.at(last_axis), out_first_dim = out_topology.at(last_axis); bool
  is_first_dim_ready = in_first_dim==1 && out_first_dim==1; bool is_slab =
        is_slab_topology(in_topology) && is_slab_topology(out_topology);

  if (is_slab && is_first_dim_ready) {
    // SlabPlan is available
    bool is_xslab = mapped_in_topology.at(0) > 1;
    plan          = std::make_unique<PlanType>(nx, ny, nz, type, comm,
  is_xslab); } else {
    // PencilPlan is available
    bool is_pencil =
        is_pencil_topology(in_topology) && is_pencil_topology(out_topology);
    KOKKOSFFT_THROW_IF(!is_pencil,
                       "CufftMp only supports slab or pencil topologies for 3D
  transforms.");
    //plan          = std::make_unique<PlanType>(fft_extents, lower_input,
  upper_input,
    //                                           lower_output, upper_output,
  strides_input, strides_output,
    //                                            type, comm, is_xslab);
  }

  plan->commit(exec_space);
  return fft_size;
  */

  return 1;
}

#endif
