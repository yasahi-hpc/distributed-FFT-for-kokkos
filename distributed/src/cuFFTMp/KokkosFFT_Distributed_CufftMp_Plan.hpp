#ifndef KOKKOSFFT_DISTRIBUTED_CUFFT_MP_PLAN_HPP
#define KOKKOSFFT_DISTRIBUTED_CUFFT_MP_PLAN_HPP

#include <algorithm>
#include <array>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_MPI_Extents.hpp"
#include "KokkosFFT_Distributed_Topologies.hpp"
#include "KokkosFFT_Distributed_Extents.hpp"
#include "KokkosFFT_Distributed_TplAxes.hpp"
#include "nvshmem/KokkosFFT_Distributed_NVSHMEM_Init.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief Check if cuFFTMp TPL can be used for the given transform parameters
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam InViewType Kokkos View type for input data
/// \tparam OutViewType Kokkos View type for output data
/// \tparam DIM Number of transform dimensions
/// \tparam InLayoutType Layout type of input topology
/// \tparam OutLayoutType Layout type of output topology
///
/// \param[in] exec_space Kokkos execution space
/// \param[in] in Input data view
/// \param[in] out Output data view
/// \param[in] axes Axes along which to perform the FFT
/// \param[in] in_topology Topology of input data distribution
/// \param[in] out_topology Topology of output data distribution
/// \return true if cuFFTMp TPL can be used, false otherwise
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM, typename InLayoutType = Kokkos::LayoutRight,
          typename OutLayoutType = Kokkos::LayoutRight>
bool is_tpl_available(
    const ExecutionSpace& exec_space, const InViewType& in,
    const OutViewType& out, const KokkosFFT::axis_type<DIM>& axes,
    const Topology<std::size_t, InViewType::rank(), InLayoutType>& in_topology,
    const Topology<std::size_t, OutViewType::rank(), OutLayoutType>&
        out_topology) {
  // LayoutLeft Views run either on the reversed LayoutRight Views (no copy)
  // or through a copy that changes the layout (is_tpl_permutation_needed);
  // the Views must have the same layout
  if constexpr (!std::is_same_v<typename InViewType::array_layout,
                                typename OutViewType::array_layout>) {
    return false;
  }

  // The axes in the order the plan is made with (canonical_tpl_axes): no
  // transpose when only the order of the axes differs, nor for a LayoutLeft
  // View that needs only a change of layout or runs reversed
  if (is_tpl_permutation_needed<InViewType>(axes)) return false;

  if (in_topology.array() == out_topology.array()) return false;

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

/// \brief Check if cuFFTMp TPL can be used for the given transform parameters
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam InViewType Kokkos View type for input data
/// \tparam OutViewType Kokkos View type for output data
/// \tparam DIM Number of transform dimensions
///
/// \param[in] exec_space Kokkos execution space
/// \param[in] in Input data view
/// \param[in] out Output data view
/// \param[in] axes Axes along which to perform the FFT
/// \param[in] in_topology Topology of input data distribution
/// \param[in] out_topology Topology of output data distribution
/// \return true if cuFFTMp TPL can be used, false otherwise
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

/// \brief Initialize NVSHMEM for cuFFTMp (once per process)
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam T Base floating point type
///
/// \param[in] comm MPI communicator
template <typename ExecutionSpace, typename T,
          std::enable_if_t<std::is_same_v<ExecutionSpace, Kokkos::Cuda>,
                           std::nullptr_t> = nullptr>
void setup(MPI_Comm comm) {
  nvshmem_setup(comm);
}

/// \brief Size in bytes, on this rank, of the NVSHMEM data buffer of a
/// cuFFTMp plan. The transform is in place in the buffer, and the buffer
/// serves both directions, so it must hold the local input (with the padding
/// of the last FFT axis for real transforms) and the local output, whichever
/// is larger: with uneven splits, the local output can be the larger one on
/// some ranks. ScopedCufftMpPlan::commit takes the maximum over the ranks.
/// \tparam InViewType Kokkos View type for input data
/// \tparam OutViewType Kokkos View type for output data
/// \tparam DIM Number of transform dimensions
///
/// \param[in] in Input data view
/// \param[in] out Output data view
/// \param[in] axes Axes along which to perform the FFT
/// \param[in] in_topology Topology of input data distribution
/// \param[in] out_topology Topology of output data distribution
/// \param[in] comm MPI communicator
/// \return The size of the buffer in bytes on this rank
template <typename InViewType, typename OutViewType, std::size_t DIM,
          typename InLayoutType, typename OutLayoutType>
std::size_t buffer_size_in_bytes(
    const InViewType& in, const OutViewType& out,
    const KokkosFFT::axis_type<DIM>& axes,
    const Topology<std::size_t, InViewType::rank(), InLayoutType>& in_topology,
    const Topology<std::size_t, OutViewType::rank(), OutLayoutType>&
        out_topology,
    const MPI_Comm& comm) {
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  auto gin_extents        = compute_global_extents(in, in_topology, comm);
  auto gout_extents       = compute_global_extents(out, out_topology, comm);
  auto gin_padded_extents = gin_extents;
  if constexpr (KokkosFFT::Impl::is_real_v<in_value_type>) {
    auto non_negative_axes =
        KokkosFFT::Impl::convert_base_int_type<std::size_t>(
            KokkosFFT::Impl::convert_negative_axes(axes, InViewType::rank()));
    gin_padded_extents = KokkosFFT::Impl::compute_padded_extents<in_value_type>(
        gout_extents, non_negative_axes);
  }
  auto [in_padded_extents, in_padded_starts] =
      compute_local_extents_and_starts(gin_padded_extents, in_topology, comm);
  auto [out_extents, out_starts] =
      compute_local_extents_and_starts(gout_extents, out_topology, comm);

  const std::size_t in_size =
      KokkosFFT::Impl::total_size(in_padded_extents) * sizeof(in_value_type);
  const std::size_t out_size =
      KokkosFFT::Impl::total_size(out_extents) * sizeof(out_value_type);
  return std::max(in_size, out_size);
}

/// \brief buffer_size_in_bytes with LayoutRight topologies
template <typename InViewType, typename OutViewType, std::size_t DIM>
std::size_t buffer_size_in_bytes(
    const InViewType& in, const OutViewType& out,
    const KokkosFFT::axis_type<DIM>& axes,
    const std::array<std::size_t, InViewType::rank()>& in_topology,
    const std::array<std::size_t, OutViewType::rank()>& out_topology,
    const MPI_Comm& comm) {
  return buffer_size_in_bytes(
      in, out, axes, Topology<std::size_t, InViewType::rank()>(in_topology),
      Topology<std::size_t, OutViewType::rank()>(out_topology), comm);
}

/// \brief General interface to create a cuFFTMp plan for distributed FFT
/// Should not be called directly
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
    const MPI_Comm& comm) {
  KOKKOSFFT_THROW_IF(true, "Plan can be made for 2D or 3D Views only");
}

/// \brief Create a cuFFTMp plan with LayoutRight topologies
template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType, std::size_t DIM>
void create_plan(
    const ExecutionSpace& exec_space, std::unique_ptr<PlanType>& plan,
    const InViewType& in, const OutViewType& out,
    const KokkosFFT::axis_type<DIM>& axes, const KokkosFFT::axis_type<DIM>& map,
    const std::array<std::size_t, InViewType::rank()>& in_topology,
    const std::array<std::size_t, OutViewType::rank()>& out_topology,
    const MPI_Comm& comm) {
  create_plan(exec_space, plan, in, out, axes, map,
              Topology<std::size_t, InViewType::rank()>(in_topology),
              Topology<std::size_t, OutViewType::rank()>(out_topology), comm);
}

/// \brief Create a 2D cuFFTMp plan for distributed FFT
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam PlanType ScopedCufftMpPlan type
/// \tparam InViewType Kokkos View type for input data
/// \tparam OutViewType Kokkos View type for output data
/// \tparam DIM Number of transform dimensions
///
/// \param[in] exec_space Kokkos execution space
/// \param[out] plan Unique pointer to the cuFFTMp plan to be created
/// \param[in] in Input data view
/// \param[in] out Output data view
/// \param[in] axes Axes along which to perform the FFT
/// \param[in] map Axis mapping for data distribution
/// \param[in] in_topology Topology of input data distribution
/// \param[in] out_topology Topology of output data distribution
/// \param[in] comm MPI communicator
template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType, typename InLayoutType, typename OutLayoutType>
void create_plan(const ExecutionSpace& exec_space,
                 std::unique_ptr<PlanType>& plan, const InViewType& in,
                 const OutViewType& out, const KokkosFFT::axis_type<2>& axes,
                 const KokkosFFT::axis_type<2>& map,
                 const Topology<std::size_t, 2, InLayoutType>& in_topology,
                 const Topology<std::size_t, 2, OutLayoutType>& out_topology,
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

  KOKKOSFFT_THROW_IF(in_topology.array() == out_topology.array(),
                     "Input and output topologies must not be identical");

  Kokkos::Profiling::ScopedRegion region("KokkosFFT::create_plan[TPL_cuFFTMp]");

  auto gin_extents       = compute_global_extents(in, in_topology, comm);
  auto gout_extents      = compute_global_extents(out, out_topology, comm);
  auto non_negative_axes = KokkosFFT::Impl::convert_base_int_type<std::size_t>(
      KokkosFFT::Impl::convert_negative_axes(axes, 2));
  auto fft_extents =
      compute_fft_extents(gin_extents, gout_extents, non_negative_axes);
  const auto [nx, ny] = fft_extents;

  if (KokkosFFT::Impl::is_real_v<in_value_type>) {
    // Using general API
    auto gin_padded_extents =
        KokkosFFT::Impl::compute_padded_extents<in_value_type>(
            gout_extents, non_negative_axes);
    auto [in_extents, in_starts] =
        compute_local_extents_and_starts(gin_extents, in_topology, comm);
    auto [out_extents, out_starts] =
        compute_local_extents_and_starts(gout_extents, out_topology, comm);
    auto [in_padded_extents, in_padded_starts] =
        compute_local_extents_and_starts(gin_padded_extents, in_topology, comm);

    std::array<std::size_t, 2> in_ends, out_ends;
    std::transform(in_starts.begin(), in_starts.end(), in_extents.begin(),
                   in_ends.begin(), std::plus<std::size_t>());
    std::transform(out_starts.begin(), out_starts.end(), out_extents.begin(),
                   out_ends.begin(), std::plus<std::size_t>());

    // Mapping based of axes
    auto mapped_in_extents =
        KokkosFFT::Impl::compute_mapped_extents(in_extents, map);
    auto mapped_in_padded_extents =
        KokkosFFT::Impl::compute_mapped_extents(in_padded_extents, map);
    auto mapped_out_extents =
        KokkosFFT::Impl::compute_mapped_extents(out_extents, map);
    auto mapped_in_starts =
        KokkosFFT::Impl::compute_mapped_extents(in_starts, map);
    auto mapped_out_starts =
        KokkosFFT::Impl::compute_mapped_extents(out_starts, map);
    auto mapped_in_ends = KokkosFFT::Impl::compute_mapped_extents(in_ends, map);
    auto mapped_out_ends =
        KokkosFFT::Impl::compute_mapped_extents(out_ends, map);

    auto in_strides = KokkosFFT::Impl::reversed(
        KokkosFFT::Impl::compute_strides(mapped_in_padded_extents));
    auto out_strides = KokkosFFT::Impl::reversed(
        KokkosFFT::Impl::compute_strides(mapped_out_extents));

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
    plan->commit(
        exec_space,
        buffer_size_in_bytes(in, out, axes, in_topology, out_topology, comm),
        comm);

  } else {
    auto in_mapped_topology =
        KokkosFFT::Impl::compute_mapped_extents(in_topology.array(), map);
    bool is_xslab = in_mapped_topology.at(0) > 1;
    plan          = std::make_unique<PlanType>(nx, ny, comm, is_xslab);
    plan->commit(
        exec_space,
        buffer_size_in_bytes(in, out, axes, in_topology, out_topology, comm),
        comm);
  }
}

/// \brief Create a 3D cuFFTMp plan for distributed FFT
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam PlanType ScopedCufftMpPlan type
/// \tparam InViewType Kokkos View type for input data
/// \tparam OutViewType Kokkos View type for output data
/// \tparam DIM Number of transform dimensions
///
/// \param[in] exec_space Kokkos execution space
/// \param[out] plan Unique pointer to the cuFFTMp plan to be created
/// \param[in] in Input data view
/// \param[in] out Output data view
/// \param[in] axes Axes along which to perform the FFT
/// \param[in] map Axis mapping for data distribution
/// \param[in] in_topology Topology of input data distribution
/// \param[in] out_topology Topology of output data distribution
/// \param[in] comm MPI communicator
template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType, typename InLayoutType, typename OutLayoutType>
void create_plan(const ExecutionSpace& exec_space,
                 std::unique_ptr<PlanType>& plan, const InViewType& in,
                 const OutViewType& out, const KokkosFFT::axis_type<3>& axes,
                 const KokkosFFT::axis_type<3>& map,
                 const Topology<std::size_t, 3, InLayoutType>& in_topology,
                 const Topology<std::size_t, 3, OutLayoutType>& out_topology,
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

  KOKKOSFFT_THROW_IF(in_topology.array() == out_topology.array(),
                     "Input and output topologies must not be identical");

  Kokkos::Profiling::ScopedRegion region("KokkosFFT::create_plan[TPL_cuFFTMp]");

  auto gin_extents       = compute_global_extents(in, in_topology, comm);
  auto gout_extents      = compute_global_extents(out, out_topology, comm);
  auto non_negative_axes = KokkosFFT::Impl::convert_base_int_type<std::size_t>(
      KokkosFFT::Impl::convert_negative_axes(axes, 3));
  auto fft_extents =
      compute_fft_extents(gin_extents, gout_extents, non_negative_axes);
  const auto [nx, ny, nz] = fft_extents;

  // In case of slab geometry, we need to check that the first dimension is
  // ready
  auto last_axis          = axes.back();
  auto in_first_dim       = in_topology.at(last_axis),
       out_first_dim      = out_topology.at(last_axis);
  bool is_first_dim_ready = in_first_dim == 1 && out_first_dim == 1;
  bool is_slab =
      are_specified_topologies(TopologyType::Slab, in_topology, out_topology);

  if (is_slab && is_first_dim_ready) {
    auto in_mapped_topology =
        KokkosFFT::Impl::compute_mapped_extents(in_topology.array(), map);
    bool is_xslab = in_mapped_topology.at(0) > 1;

    plan = std::make_unique<PlanType>(nx, ny, nz, comm, is_xslab);
    plan->commit(
        exec_space,
        buffer_size_in_bytes(in, out, axes, in_topology, out_topology, comm),
        comm);
  } else {
    // Using general API
    auto gin_padded_extents = gin_extents;
    if (KokkosFFT::Impl::is_real_v<in_value_type>) {
      gin_padded_extents =
          KokkosFFT::Impl::compute_padded_extents<in_value_type>(
              gout_extents, non_negative_axes);
    }

    auto [in_extents, in_starts] =
        compute_local_extents_and_starts(gin_extents, in_topology, comm);
    auto [out_extents, out_starts] =
        compute_local_extents_and_starts(gout_extents, out_topology, comm);
    auto [in_padded_extents, in_padded_starts] =
        compute_local_extents_and_starts(gin_padded_extents, in_topology, comm);

    std::array<std::size_t, 3> in_ends, out_ends;
    std::transform(in_starts.begin(), in_starts.end(), in_extents.begin(),
                   in_ends.begin(), std::plus<std::size_t>());
    std::transform(out_starts.begin(), out_starts.end(), out_extents.begin(),
                   out_ends.begin(), std::plus<std::size_t>());

    // Mapping based of axes
    auto mapped_in_extents =
        KokkosFFT::Impl::compute_mapped_extents(in_extents, map);
    auto mapped_in_padded_extents =
        KokkosFFT::Impl::compute_mapped_extents(in_padded_extents, map);
    auto mapped_out_extents =
        KokkosFFT::Impl::compute_mapped_extents(out_extents, map);
    auto mapped_in_starts =
        KokkosFFT::Impl::compute_mapped_extents(in_starts, map);
    auto mapped_out_starts =
        KokkosFFT::Impl::compute_mapped_extents(out_starts, map);
    auto mapped_in_ends = KokkosFFT::Impl::compute_mapped_extents(in_ends, map);
    auto mapped_out_ends =
        KokkosFFT::Impl::compute_mapped_extents(out_ends, map);

    auto in_strides = KokkosFFT::Impl::reversed(
        KokkosFFT::Impl::compute_strides(mapped_in_padded_extents));
    auto out_strides = KokkosFFT::Impl::reversed(
        KokkosFFT::Impl::compute_strides(mapped_out_extents));

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
    plan->commit(
        exec_space,
        buffer_size_in_bytes(in, out, axes, in_topology, out_topology, comm),
        comm);
  }
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
