#ifndef KOKKOSFFT_DISTRIBUTED_EXTENTS_HPP
#define KOKKOSFFT_DISTRIBUTED_EXTENTS_HPP

#include <algorithm>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_Mapping.hpp"
#include "KokkosFFT_Distributed_Types.hpp"
#include "KokkosFFT_Distributed_Utils.hpp"

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
/// \tparam DIM The number of dimensions of the extents.
///
/// \param[in] extents Extents of the global View.
/// \param[in] in_topology A topology representing the distribution of the input
/// data.
/// \param[in] out_topology A topology representing the distribution of
/// the output data.
/// \return A buffer extents of the view needed for the pencil
/// transformation.
template <typename LayoutType, std::size_t DIM = 1>
auto get_buffer_extents(const std::array<std::size_t, DIM> &extents,
                        const std::array<std::size_t, DIM> &in_topology,
                        const std::array<std::size_t, DIM> &out_topology) {
  std::array<std::size_t, DIM + 1> buffer_extents;
  auto merged_topology = merge_topology(in_topology, out_topology);
  auto p0 = diff_toplogy(merged_topology, in_topology);  // return 1 or p0
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
/// \tparam DIM The number of dimensions of the extents.
/// \tparam InLayoutType The layout type of the in-topology (e.g.,
/// Kokkos::LayoutRight). \tparam OutLayoutType The layout type of the
/// out-topology (e.g., Kokkos::LayoutRight).
///
/// \param[in] extents Extents of the global View.
/// \param[in] in_topology A topology representing the distribution of the input
/// data.
/// \param[in] out_topology A topology representing the distribution of
/// the output data.
/// \return A buffer extents of the view needed for the pencil
/// transformation.
template <typename LayoutType, std::size_t DIM = 1,
          typename InLayoutType  = Kokkos::LayoutRight,
          typename OutLayoutType = Kokkos::LayoutRight>
auto get_buffer_extents(
    const std::array<std::size_t, DIM> &extents,
    const Topology<std::size_t, DIM, InLayoutType> &in_topology,
    const Topology<std::size_t, DIM, OutLayoutType> &out_topology) {
  return get_buffer_extents<LayoutType>(extents, in_topology.array(),
                                        out_topology.array());
}

/// \brief Calculate the next extents based on the global extents,
/// the topology, and the mapping.
///
/// Example
/// Global View extents: (n0, n1, n2, n3)
/// topology: {p0, 1, p1, 1} // Y-pencil
/// map: (0, 2, 3, 1)
/// Next extents: ((n0-1)/p0+1, (n2-1)/p1+1, n3, n1)
///
/// \tparam DIM The number of dimensions of the extents.
///
/// \param[in] extents Extents of the global View.
/// \param[in] topology A topology representing the distribution of the data.
/// \param[in] map A map representing how the data is permuted
/// \return A extents of the view after the pencil transformation.

template <std::size_t DIM = 1>
auto get_next_extents(const std::array<std::size_t, DIM> &extents,
                      const std::array<std::size_t, DIM> &topology,
                      const std::array<std::size_t, DIM> &map) {
  std::array<std::size_t, DIM> next_extents;

  for (std::size_t i = 0; i < extents.size(); i++) {
    std::size_t mapped_idx = map.at(i);
    next_extents.at(i) =
        (extents.at(mapped_idx) - 1) / topology.at(mapped_idx) + 1;
  }

  return next_extents;
}

/// \brief From the list of extents, calculate the required allocation size
/// that is big enough to represent all of the extents.
/// \tparam DIM The number of dimensions of the extents.
///
/// \param[in] extents A vector of extents, each represented as an array of size
/// DIM.
/// \return The total size required for the allocation.
template <std::size_t DIM = 1>
auto get_required_allocation_size(
    const std::vector<std::array<std::size_t, DIM>> &extents) {
  std::vector<std::size_t> sizes;
  for (const auto &extent : extents) {
    sizes.push_back(get_size(extent));
  }
  return *std::max_element(sizes.begin(), sizes.end());
}

/// \brief From the list of extents, calculate the required allocation size
/// that is big enough to represent all of the extents.
/// \tparam DIM The number of dimensions of the extents.
///
/// \param[in] extents A vector of extents, each represented as an array of size
/// DIM.
/// \param[in] byte_sizes A vector of bytes of sizes
/// \return The total byte size required for the allocation.
template <std::size_t DIM = 1>
auto get_required_allocation_size(
    const std::vector<std::array<std::size_t, DIM>> &extents,
    std::vector<std::size_t> &byte_sizes) {
  KOKKOSFFT_THROW_IF(extents.size() != byte_sizes.size(),
                     "extents and byte_sizes must have the same size.");
  std::vector<std::size_t> sizes;
  for (std::size_t i = 0; i < extents.size(); i++) {
    sizes.push_back(get_size(extents.at(i)) * byte_sizes.at(i));
  }
  return *std::max_element(sizes.begin(), sizes.end());
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
  auto gin_extents  = get_global_shape(in, in_topology, comm);
  auto gout_extents = get_global_shape(out, out_topology, comm);

  auto in_extents  = get_mapped_extents(gin_extents, map);
  auto out_extents = get_mapped_extents(gout_extents, map);

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

#endif
