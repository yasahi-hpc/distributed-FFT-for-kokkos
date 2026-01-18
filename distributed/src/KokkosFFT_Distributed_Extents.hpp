#ifndef KOKKOSFFT_DISTRIBUTED_EXTENTS_HPP
#define KOKKOSFFT_DISTRIBUTED_EXTENTS_HPP

#include <algorithm>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_Mapping.hpp"
#include "KokkosFFT_Distributed_Types.hpp"
#include "KokkosFFT_Distributed_Utils.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief Compute padded extents from the extents in Fourier space
///
/// Example
/// in extents: (8, 7, 8)
/// out extents: (8, 7, 5)
///
/// \tparam DIM The number of dimensions of the extents.
///
/// \param[in] in_extents Extents of the global input View.
/// \param[in] out_extents Extents of the global output View.
/// \param[in] axes Axes of the transform
/// \return A extents of the permuted view
template <std::size_t DIM>
auto compute_padded_extents(const std::array<std::size_t, DIM> &extents,
                            const std::array<std::size_t, DIM> &axes) {
  std::array<std::size_t, DIM> padded_extents = extents;
  auto last_axis                              = axes.back();
  padded_extents.at(last_axis) = padded_extents.at(last_axis) * 2;

  return padded_extents;
}

/// \brief Calculate the buffer extents based on the global extents,
/// the in-topology, and the out-topology.
///
/// Example
/// Global View extents (n0, n1, n2, n3)
/// in-topology = {1, p0, p1, 1} // X-pencil
/// out-topology = {p0, 1, p1, 1} // Y-pencil
/// Buffer View (p0, n0/p0, n1/p0, n2/p1, n3)
///
/// \tparam LayoutType The layout type of the view (e.g., Kokkos::LayoutRight).
/// \tparam iType The integer type used for extents and topology.
/// \tparam DIM The number of dimensions of the extents.
///
/// \param[in] extents Extents of the global View.
/// \param[in] in_topology A topology representing the distribution of the input
/// data.
/// \param[in] out_topology A topology representing the distribution of
/// the output data.
/// \return A buffer extents of the view needed for the pencil
/// transformation.
template <typename LayoutType, typename iType, std::size_t DIM>
auto compute_buffer_extents(const std::array<iType, DIM> &extents,
                            const std::array<iType, DIM> &in_topology,
                            const std::array<iType, DIM> &out_topology) {
  std::array<iType, DIM + 1> buffer_extents;
  auto merged_topology = merge_topology(in_topology, out_topology);
  auto p0 = diff_topology(merged_topology, in_topology);  // return 1 or p0
  if (std::is_same_v<LayoutType, Kokkos::LayoutRight>) {
    buffer_extents.at(0) = p0;
    for (std::size_t i = 0; i < extents.size(); i++) {
      buffer_extents.at(i + 1) =
          (extents.at(i) - 1) / merged_topology.at(i) + 1;
    }
  } else {
    for (std::size_t i = 0; i < extents.size(); i++) {
      buffer_extents.at(i) = (extents.at(i) - 1) / merged_topology.at(i) + 1;
    }
    buffer_extents.back() = p0;
  }
  return buffer_extents;
}

/// \brief Calculate the buffer extents based on the global extents,
/// the in-topology, and the out-topology.
///
/// Example
/// Global View extents (n0, n1, n2, n3)
/// in-topology = {1, p0, p1, 1} // X-pencil
/// out-topology = {p0, 1, p1, 1} // Y-pencil
/// Buffer View (p0, n0/p0, n1/p0, n2/p1, n3)
///
/// \tparam LayoutType The layout type of the view (e.g., Kokkos::LayoutRight).
/// \tparam iType The integer type used for extents and topology.
/// \tparam DIM The number of dimensions of the extents.
/// \tparam InLayoutType The layout type of the in-topology (e.g.,
/// Kokkos::LayoutRight).
/// \tparam OutLayoutType The layout type of the out-topology (e.g.,
/// Kokkos::LayoutRight).
///
/// \param[in] extents Extents of the global View.
/// \param[in] in_topology A topology representing the distribution of the input
/// data.
/// \param[in] out_topology A topology representing the distribution of
/// the output data.
/// \return A buffer extents of the view needed for the pencil
/// transformation.
template <typename LayoutType, typename iType, std::size_t DIM = 1,
          typename InLayoutType  = Kokkos::LayoutRight,
          typename OutLayoutType = Kokkos::LayoutRight>
auto compute_buffer_extents(
    const std::array<iType, DIM> &extents,
    const Topology<iType, DIM, InLayoutType> &in_topology,
    const Topology<iType, DIM, OutLayoutType> &out_topology) {
  return compute_buffer_extents<LayoutType>(extents, in_topology.array(),
                                            out_topology.array());
}

/// \brief Calculate the permuted extents based on the map
///
/// Example
/// View extents: (n0, n1, n2, n3)
/// map: (0, 2, 3, 1)
/// Next extents: (n0, n2, n3, n1)
///
/// \tparam ContainerType The container type
/// \tparam iType The integer type used for extents
/// \tparam DIM The number of dimensions of the extents.
///
/// \param[in] extents Extents of the View.
/// \param[in] map A map representing how the data is permuted
/// \return A extents of the permuted view
/// \throws std::runtime_error if the size of map is not equal to DIM
template <typename ContainerType, typename iType, std::size_t DIM>
auto compute_mapped_extents(const std::array<iType, DIM> &extents,
                            const ContainerType &map) {
  using value_type =
      std::remove_cv_t<std::remove_reference_t<decltype(map.at(0))>>;
  static_assert(std::is_integral_v<value_type>,
                "compute_mapped_extents: Map container value type must be an "
                "integral type");
  KOKKOSFFT_THROW_IF(
      map.size() != DIM,
      "compute_mapped_extents: extents size must be equal to map size.");
  std::array<iType, DIM> mapped_extents;
  std::transform(
      map.begin(), map.end(), mapped_extents.begin(),
      [&](std::size_t mapped_idx) { return extents.at(mapped_idx); });

  return mapped_extents;
}

/// \brief Compute the larger extents. Larger one corresponds to
/// the extents to FFT library. This is a helper for vendor library
/// which supports 2D or 3D non-batched FFTs.
///
/// Example
/// in extents: (8, 7, 8)
/// out extents: (8, 7, 5)
///
/// \tparam iType The integer type used for extents
/// \tparam DIM The number of dimensions of the extents.
/// \tparam FFT_DIM The number of dimensions of the FFT.
///
/// \param[in] in_extents Extents of the global input View.
/// \param[in] out_extents Extents of the global output View.
/// \return A extents of the permuted view
template <typename iType, std::size_t DIM, std::size_t FFT_DIM>
auto compute_fft_extents(const std::array<iType, DIM> &in_extents,
                         const std::array<iType, DIM> &out_extents,
                         const std::array<iType, FFT_DIM> &axes) {
  static_assert(std::is_integral_v<iType>,
                "compute_fft_extents: iType must be an integral type");
  static_assert(
      FFT_DIM >= 1 && FFT_DIM <= KokkosFFT::MAX_FFT_DIM,
      "compute_fft_extents: the Rank of FFT axes must be between 1 and 3");
  static_assert(
      DIM >= FFT_DIM,
      "compute_fft_extents: View rank must be larger than or equal to "
      "the Rank of FFT axes");

  std::array<iType, FFT_DIM> fft_extents;
  std::transform(axes.begin(), axes.end(), fft_extents.begin(),
                 [&](iType axis) {
                   return std::max(in_extents.at(axis), out_extents.at(axis));
                 });

  return fft_extents;
}

/**
 * @brief Check if input and output views have valid extents for the given axes,
 * FFT topologies, and MPI communicator.
 *
 * @param in The input view.
 * @param out The output view.
 * @param axes The axes over which the FFT is performed.
 * @param in_topology The FFT topology for the input view.
 * @param out_topology The FFT topology for the output view.
 * @param comm The MPI communicator. Default is MPI_COMM_WORLD.
 *
 * @return true if the input and output views have valid extents, false
 * otherwise.
 *
 * Note: This function does not check if the FFT topologies are valid or not.
 * It only checks the compatibility of the FFT topologies with the input and
 * output views. Please use KokkosFFT::Impl::are_valid_axes to check if the FFT
 * axes are valid for the input view.
 *
 * For C2C transform:
 *   - The input and output views must have the same extent in all dimensions.
 *
 * For R2C transform:
 *   - The 'output extent' of the transform must be equal to 'input extent'/2
 * + 1.
 *
 * For C2R transform:
 *   - The 'input extent' of the transform must be equal to 'output extent' / 2
 * + 1.
 */
template <typename InViewType, typename OutViewType, typename SizeType,
          std::size_t DIM = 1, typename InLayoutType = Kokkos::LayoutRight,
          typename OutLayoutType = Kokkos::LayoutRight>
bool are_valid_extents(
    const InViewType &in, const OutViewType &out,
    KokkosFFT::axis_type<DIM> axes,
    const Topology<SizeType, InViewType::rank(), InLayoutType> &in_topology,
    const Topology<SizeType, OutViewType::rank(), OutLayoutType> &out_topology,
    const MPI_Comm &comm = MPI_COMM_WORLD) {
  using in_value_type     = typename InViewType::non_const_value_type;
  using out_value_type    = typename OutViewType::non_const_value_type;
  using array_layout_type = typename InViewType::array_layout;

  static_assert(!(KokkosFFT::Impl::is_real_v<in_value_type> &&
                  KokkosFFT::Impl::is_real_v<out_value_type>),
                "are_valid_extents: real to real transform is not supported");

  KOKKOSFFT_THROW_IF(!KokkosFFT::Impl::are_valid_axes(in, axes),
                     "input axes are not valid for the view");

  constexpr std::size_t rank = InViewType::rank;
  [[maybe_unused]] std::size_t inner_most_axis =
      std::is_same_v<array_layout_type, typename Kokkos::LayoutLeft>
          ? 0
          : (rank - 1);

  // index map after transpose over axis
  auto [map, map_inv] = KokkosFFT::Impl::get_map_axes(in, axes);

  // Get global shape to define buffer and next shape
  auto gin_extents  = compute_global_extents(in, in_topology, comm);
  auto gout_extents = compute_global_extents(out, out_topology, comm);

  auto in_extents  = compute_mapped_extents(gin_extents, map);
  auto out_extents = compute_mapped_extents(gout_extents, map);

  auto mismatched_extents = [&in, &out, &in_extents,
                             &out_extents]() -> std::string {
    std::string message;
    message += in.label();
    message += "(";
    message += std::to_string(in_extents.at(0));
    for (std::size_t r = 1; r < in_extents.size(); r++) {
      message += ",";
      message += std::to_string(in_extents.at(r));
    }
    message += "), ";
    message += out.label();
    message += "(";
    message += std::to_string(out_extents.at(0));
    for (std::size_t r = 1; r < out_extents.size(); r++) {
      message += ",";
      message += std::to_string(out_extents.at(r));
    }
    message += ")";
    return message;
  };

  for (std::size_t i = 0; i < rank; i++) {
    // The requirement for inner_most_axis is different for transform type
    if (i == inner_most_axis) continue;
    KOKKOSFFT_THROW_IF(in_extents.at(i) != out_extents.at(i),
                       "input and output extents must be the same except for "
                       "the transform axis: " +
                           mismatched_extents());
  }

  if constexpr (KokkosFFT::Impl::is_complex_v<in_value_type> &&
                KokkosFFT::Impl::is_complex_v<out_value_type>) {
    // Then C2C
    KOKKOSFFT_THROW_IF(
        in_extents.at(inner_most_axis) != out_extents.at(inner_most_axis),
        "input and output extents must be the same for C2C transform: " +
            mismatched_extents());
  }

  if constexpr (KokkosFFT::Impl::is_real_v<in_value_type>) {
    // Then R2C
    KOKKOSFFT_THROW_IF(
        out_extents.at(inner_most_axis) !=
            in_extents.at(inner_most_axis) / 2 + 1,
        "For R2C, the 'output extent' of transform must be equal to "
        "'input extent'/2 + 1: " +
            mismatched_extents());
  }

  if constexpr (KokkosFFT::Impl::is_real_v<out_value_type>) {
    // Then C2R
    KOKKOSFFT_THROW_IF(
        in_extents.at(inner_most_axis) !=
            out_extents.at(inner_most_axis) / 2 + 1,
        "For C2R, the 'input extent' of transform must be equal to "
        "'output extent' / 2 + 1: " +
            mismatched_extents());
  }
  return true;
}

/**
 * @brief Check if input and output views have valid extents for the given axes,
 * FFT topologies, and MPI communicator.
 *
 * @param in The input view.
 * @param out The output view.
 * @param axes The axes over which the FFT is performed.
 * @param in_topology The FFT topology for the input view.
 * @param out_topology The FFT topology for the output view.
 * @param comm The MPI communicator. Default is MPI_COMM_WORLD.
 *
 * @return true if the input and output views have valid extents, false
 * otherwise.
 *
 * Note: This function does not check if the FFT topologies are valid or not.
 * It only checks the compatibility of the FFT topologies with the input and
 * output views. Please use KokkosFFT::Impl::are_valid_axes to check if the FFT
 * axes are valid for the input view.
 *
 * For C2C transform:
 *   - The input and output views must have the same extent in all dimensions.
 *
 * For R2C transform:
 *   - The 'output extent' of the transform must be equal to 'input extent'/2
 * + 1.
 *
 * For C2R transform:
 *   - The 'input extent' of the transform must be equal to 'output extent' / 2
 * + 1.
 */
template <typename InViewType, typename OutViewType, typename SizeType,
          std::size_t DIM = 1>
bool are_valid_extents(
    const InViewType &in, const OutViewType &out,
    KokkosFFT::axis_type<DIM> axes,
    const std::array<SizeType, InViewType::rank()> &in_topology,
    const std::array<SizeType, OutViewType::rank()> &out_topology,
    const MPI_Comm &comm = MPI_COMM_WORLD) {
  return are_valid_extents(
      in, out, axes, Topology<SizeType, InViewType::rank()>(in_topology),
      Topology<SizeType, OutViewType::rank()>(out_topology), comm);
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
