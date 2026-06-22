#ifndef KOKKOSFFT_DISTRIBUTED_TOPOLOGIES_HPP
#define KOKKOSFFT_DISTRIBUTED_TOPOLOGIES_HPP

#include <algorithm>
#include <array>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Asserts.hpp"
#include "KokkosFFT_Distributed_MPI_Extents.hpp"
#include "KokkosFFT_Distributed_Types.hpp"
#include "KokkosFFT_Distributed_ContainerAnalyses.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief Get the topology type from the given topology container
/// Empty topology: at least one element is 0
/// Shared topology: no elements differ from 1
/// Slab topology: exactly 1 element differs from 1
/// Pencil topology: exactly 2 elements differ from 1
/// Brick topology: exactly 3 elements differ from 1
/// Invalid topology: more than 3 elements differ from 1
///
/// \tparam ContainerType Topology container type (std::array or Topology)
/// \param[in] topology Topology container
/// \return TopologyType enum value representing the topology type
template <typename ContainerType>
inline auto to_topology_type(const ContainerType& topology) {
  static_assert(
      (is_allowed_topology_v<ContainerType>),
      "to_topology_type: topologies must be either in std::array or Topology");
  using value_type =
      std::remove_cv_t<std::remove_reference_t<decltype(*topology.begin())>>;
  static_assert(
      std::is_integral_v<value_type>,
      "to_topology_type: Container value type must be an integral type");

  for (const auto& value : topology) {
    if (value == 0) return TopologyType::Empty;
  }

  switch (count_non_ones(topology)) {
    case 0: return TopologyType::Shared;
    case 1: return TopologyType::Slab;
    case 2: return TopologyType::Pencil;
    case 3: return TopologyType::Brick;
    default: return TopologyType::Invalid;
  }
}

/// \brief Check if all given topologies are of specified type
/// \tparam Topologies Variadic template parameter for topology container types
/// \param[in] topology_type a topology type of interest
/// \param[in] topologies Topology containers
/// \return true if all topologies are of the specified type, false otherwise
template <class... Topologies>
inline bool are_specified_topologies(const TopologyType topology_type,
                                     const Topologies&... topologies) {
  static_assert(
      sizeof...(Topologies) > 0,
      "are_specified_topologies: at least one topology must be provided");
  static_assert((are_allowed_topologies_v<Topologies...>),
                "are_specified_topologies: topologies must be either in "
                "std::array or Topology");
  auto is_specified_topology = [topology_type](const auto& topology) {
    return to_topology_type(topology) == topology_type;
  };
  return (is_specified_topology(topologies) && ...);
}

/// \brief Get the topology type from the given topology containers
///
/// \tparam FirstTopology The type of the first topology container
/// \tparam RestTopologies Variadic template parameter for the rest of the
/// topology container types
/// \param[in] first_topology The first topology container
/// \param[in] rest_topologies The rest of the topology containers
/// \return TopologyType::Empty if any topology is empty; otherwise the common
/// topology type if all topologies have the same non-empty type; otherwise
/// TopologyType::Invalid
template <class FirstTopology, class... RestTopologies>
inline auto get_common_topology_type(const FirstTopology& first_topology,
                                     const RestTopologies&... rest_topologies) {
  static_assert((are_allowed_topologies_v<FirstTopology, RestTopologies...>),
                "get_common_topology_type: topologies must be either in "
                "std::array or Topology");

  const auto common_topology_type = to_topology_type(first_topology);
  if (common_topology_type == TopologyType::Empty) {
    return TopologyType::Empty;
  }

  if constexpr (sizeof...(RestTopologies) > 0) {
    const bool has_empty =
        ((to_topology_type(rest_topologies) == TopologyType::Empty) || ...);
    if (has_empty) {
      return TopologyType::Empty;
    }
  }

  bool has_mismatch        = false;
  auto check_topology_type = [&](const auto& topology) {
    const auto topology_type = to_topology_type(topology);
    if (topology_type != common_topology_type) {
      has_mismatch = true;
    }
  };
  if constexpr (sizeof...(RestTopologies) > 0) {
    (check_topology_type(rest_topologies), ...);
  }

  return has_mismatch ? TopologyType::Invalid : common_topology_type;
}

/// \brief Get the axes of the input and output slab topologies that are
/// different
/// Example
/// (1, P) -> (P, 1): y-slab to x-slab
/// (P, 1) -> (1, p): x-slab to y-slab
/// (1, 1, P) -> (1, P, 1): z-slab to y-slab
/// (P, 1, 1) -> (1, P, 1): x-slab to y-slab
///
/// \tparam iType The type of the index in the topology.
/// \tparam DIM The number of dimensions of the topology.
/// \param[in] in_topology The input topology.
/// \param[in] out_topology The output topology.
/// \return A tuple of two size_t representing the axes that are different
/// \throws std::runtime_error if the input and output topologies do not have
/// the same size or if they are identical
/// \throws std::runtime_error if the input and output topologies are not slab
/// topologies
template <typename iType, std::size_t DIM>
auto slab_in_out_axes(const std::array<iType, DIM>& in_topology,
                      const std::array<iType, DIM>& out_topology) {
  auto in_size  = KokkosFFT::Impl::total_size(in_topology);
  auto out_size = KokkosFFT::Impl::total_size(out_topology);

  KOKKOSFFT_THROW_IF(in_size != out_size,
                     "Input and output topologies must have the same size.");

  KOKKOSFFT_THROW_IF(in_topology == out_topology,
                     "Input and output topologies must be different.");

  bool is_slab =
      are_specified_topologies(TopologyType::Slab, in_topology, out_topology);
  KOKKOSFFT_THROW_IF(!is_slab,
                     "Input and output topologies must be slab topologies.");

  std::size_t in_axis = 0, out_axis = 0;
  for (std::size_t i = 0; i < DIM; ++i) {
    if (in_topology.at(i) > 1 && out_topology.at(i) == 1) {
      out_axis = i;
    }
    if (in_topology.at(i) == 1 && out_topology.at(i) > 1) {
      in_axis = i;
    }
  }

  return std::make_tuple(in_axis, out_axis);
}

/// \brief Get the axes of the input and output topologies that are different
///
/// Example
/// (1, Px, Py, 1) -> (Px, 1, Py, 1): 0-pencil to 1-pencil
/// (1, 1, P) -> (1, P, 1): 1-pencil to 2-pencil
/// (P, 1, 1) -> (1, P, 1): 1-pencil to 0-pencil
///
/// \tparam iType The type of the index in the topology.
/// \tparam DIM The number of dimensions of the topology.
///
/// \param[in] in_topology The input topology.
/// \param[in] out_topology The output topology.
/// \return A tuple of two size_t representing the axes that are different
/// \throws std::runtime_error if the input and output topologies do not have
/// at least one non-trivial dimension
/// \throws std::runtime_error if the input and output topologies do not have
/// the same size
template <typename iType, std::size_t DIM>
auto pencil_in_out_axes(const std::array<iType, DIM>& in_topology,
                        const std::array<iType, DIM>& out_topology) {
  // Extract topology that is common between in_topology and out_topology
  auto in_size  = KokkosFFT::Impl::total_size(in_topology);
  auto out_size = KokkosFFT::Impl::total_size(out_topology);

  KOKKOSFFT_THROW_IF(in_size == 1 || out_size == 1,
                     "Input and output topologies must have at least one "
                     "non-trivial dimension.");

  KOKKOSFFT_THROW_IF(in_size != out_size,
                     "Input and output topologies must have the same size.");

  std::size_t in_axis = 0, out_axis = 0;
  for (std::size_t i = 0; i < DIM; ++i) {
    if (in_topology.at(i) != out_topology.at(i)) {
      if (in_topology.at(i) == 1) in_axis = i;
      if (out_topology.at(i) == 1) out_axis = i;
    }
  }

  return std::make_tuple(in_axis, out_axis);
}

/// \brief Get an intermediate topology by swapping two non-one elements
///        between input and output topologies. Used to propose intermediate
///        topology for slab/pencil decompositions if direct conversion is not
///        possible.
///
/// \tparam iType The index type used for the topology.
/// \tparam DIM The dimensionality of the topology.
///
/// \param[in] in The input topology.
/// \param[in] out The output topology.
/// \return An intermediate topology obtained by swapping two non-one elements.
/// \throws std::runtime_error if the input and output topologies do not differ
/// exactly three positions
template <typename iType, std::size_t DIM>
std::array<iType, DIM> propose_mid_array(const std::array<iType, DIM>& in,
                                         const std::array<iType, DIM>& out) {
  auto diff_indices         = extract_different_indices(in, out);
  auto diff_value_set       = extract_different_value_set(in, out);
  auto diff_non_one_indices = extract_non_one_indices(in, out);

  KOKKOSFFT_THROW_IF(diff_non_one_indices.size() < 3,
                     "The total number of non-one elements either in Input and "
                     "output topologies must be three.");
  KOKKOSFFT_THROW_IF(
      diff_indices.size() < 3 && diff_value_set.size() == 3,
      "Input and output topologies must differ exactly three positions.");

  // Only copy the exchangeable indices from original arrays in and out
  std::array<iType, DIM> in_trimmed{}, out_trimmed{};
  for (auto diff_idx : diff_indices) {
    in_trimmed.at(diff_idx)  = in.at(diff_idx);
    out_trimmed.at(diff_idx) = out.at(diff_idx);
  }

  iType idx_one_in  = KokkosFFT::Impl::get_index(in_trimmed, iType(1));
  iType idx_one_out = KokkosFFT::Impl::get_index(out_trimmed, iType(1));

  // Try all combinations of 2 indices for a single valid swap
  for (std::size_t i = 0; i < diff_non_one_indices.size(); ++i) {
    for (std::size_t j = i + 1; j < diff_non_one_indices.size(); ++j) {
      iType idx_in  = diff_non_one_indices.at(i);
      iType idx_out = diff_non_one_indices.at(j);

      std::array<iType, DIM> mid = swap_elements(in, idx_in, idx_out);
      iType idx_one_mid          = KokkosFFT::Impl::get_index(mid, iType(1));

      auto mid_in_diff_indices  = extract_different_indices(mid, in);
      auto mid_out_diff_indices = extract_different_indices(mid, out);
      if ((mid_in_diff_indices.size() == 2) &&
          (mid_out_diff_indices.size() == 2) &&
          !(idx_one_mid == idx_one_in || idx_one_mid == idx_one_out)) {
        // Do not allow exchange two non-one elements
        auto mid_in_diff0  = mid.at(mid_in_diff_indices.at(0));
        auto mid_in_diff1  = mid.at(mid_in_diff_indices.at(1));
        auto mid_out_diff0 = mid.at(mid_out_diff_indices.at(0));
        auto mid_out_diff1 = mid.at(mid_out_diff_indices.at(1));
        if ((mid_in_diff0 == 1 || mid_in_diff1 == 1) &&
            (mid_out_diff0 == 1 || mid_out_diff1 == 1)) {
          return mid;
        }
      }
    }
  }

  return out;
}

/// \brief Decompose the FFT axes into vectors
///        The first vector includes the axes for FFT without transpose
///        The second vector includes the axes for FFT after transpose
///        The third vector includes the axes for remaining FFT
///
/// \tparam iType The index type used for the topology.
/// \tparam DIM The dimensionality of the topology.
/// \tparam FFT_DIM The dimensionality of the FFT axes.
///
/// \param[in] topologies The vector of topologies.
/// \param[in] axes The axes along which the FFT is performed.
/// \return A vector of vectors of axes.
/// \throws std::runtime_error if the total size of decomposed axes does not
/// match the original axes size
template <typename iType, std::size_t DIM, std::size_t FFT_DIM>
std::vector<std::vector<iType>> decompose_axes(
    const std::vector<std::array<std::size_t, DIM>>& topologies,
    const std::array<iType, FFT_DIM>& axes) {
  auto non_negative_axes = KokkosFFT::Impl::convert_base_int_type<std::size_t>(
      KokkosFFT::Impl::convert_negative_axes(axes, DIM));

  // Reverse the axes e.g. {0, 2, 1} -> {1, 2, 0}
  std::vector<std::size_t> axes_reversed =
      KokkosFFT::Impl::reversed(KokkosFFT::Impl::to_vector(non_negative_axes));

  std::vector<std::vector<iType>> all_axes{};
  for (auto topology : topologies) {
    std::vector<iType> ready_axes;
    for (auto axis : axes_reversed) {
      if (topology.at(axis) > 1) break;
      ready_axes.push_back(axis);
    }
    // We need to reverse the axes again
    // i.e. {1, 2} -> {2, 1}
    all_axes.push_back(KokkosFFT::Impl::reversed(ready_axes));

    // Remove already registered axes
    for (auto axis : ready_axes) {
      auto it = std::find(axes_reversed.begin(), axes_reversed.end(), axis);
      if (it != axes_reversed.end()) {
        axes_reversed.erase(it);
      }
    }
  }

  auto error_msg = [&axes, &all_axes,
                    &topologies](std::string_view details) -> std::string {
    std::string msg(details);
    msg += KokkosFFT::Impl::container_to_string(" Input axes: ", axes);
    msg += "\n";
    msg += "Decomposed axes: \n";
    for (std::size_t i = 0; i < all_axes.size(); ++i) {
      auto topology = topologies.at(i);
      msg += "at ";
      msg += KokkosFFT::Impl::container_to_string("topology: ", topology);
      msg += ": Ready axes: ";
      if (all_axes.at(i).empty()) {
        msg += "None";
      } else {
        msg += KokkosFFT::Impl::container_to_string("", all_axes.at(i));
      }
      msg += "\n";
    }
    return msg;
  };

  std::size_t total_axes = 0;
  for (auto ready_axes : all_axes) {
    total_axes += ready_axes.size();
  }

  KOKKOSFFT_THROW_IF(total_axes != axes.size(),
                     error_msg("Axes are not decomposed correctly:"));

  return all_axes;
}

/// \brief Compute the axis to transpose to convert one topology to another
/// Example
/// (1, Px, Py, 1) -> (Px, 1, Py, 1). Transpose axis is Px (0)
/// (1, Px, Py, 1) -> (1, Px, 1, Py). Transpose axis is Py (1)
///
/// \tparam iType The index type
/// \tparam DIM The dimension
///
/// \param[in] in_topology The input topology
/// \param[in] out_topology The output topology
/// \param[in] first_non_one The first non-one element in the input or output
/// \return The axis to transpose (0 or 1)
/// \throws std::runtime_error if the input and output topologies do not have
/// exactly two non-one elements
/// \throws std::runtime_error if the input and output topologies have identical
/// non-one elements
/// \throws std::runtime_error if the input and output topologies do not differ
/// in exactly two positions
template <typename iType, std::size_t DIM>
auto compute_trans_axis(const std::array<iType, DIM>& in_topology,
                        const std::array<iType, DIM>& out_topology,
                        iType first_non_one) {
  auto in_non_ones  = extract_non_one_values(in_topology);
  auto out_non_ones = extract_non_one_values(out_topology);

  auto error_msg = [&in_topology,
                    &out_topology](std::string_view details) -> std::string {
    std::string message(details);
    message +=
        KokkosFFT::Impl::container_to_string("in_topology: ", in_topology);
    message += ", ";
    message +=
        KokkosFFT::Impl::container_to_string("out_topology: ", out_topology);
    return message;
  };

  KOKKOSFFT_THROW_IF(in_non_ones.size() != 2 || out_non_ones.size() != 2,
                     error_msg("Input and output topologies must have exactly "
                               "two non-one elements."));
  KOKKOSFFT_THROW_IF(has_identical_non_ones(in_non_ones) ||
                         has_identical_non_ones(out_non_ones),
                     error_msg("Input and output topologies must not have "
                               "identical non-one elements."));
  auto diff_indices = extract_different_indices(in_topology, out_topology);
  KOKKOSFFT_THROW_IF(
      diff_indices.size() != 2,
      error_msg(
          "Input and output topologies must differ exactly two positions"));
  iType exchange_non_one = 0;
  for (auto diff_idx : diff_indices) {
    if (in_topology.at(diff_idx) > 1) {
      exchange_non_one = in_topology.at(diff_idx);
      break;
    }
  }
  iType trans_axis = !(exchange_non_one == first_non_one);
  return trans_axis;
}

/// \brief Divide the extents by the topology to get the local extents for each
/// process This is not exactly same as the local extents, but is used to check
/// whether the local extents include 0 or not.
/// \tparam ExtentsType The extents type (std::array)
/// \tparam TopologyContainerType The topology container type (std::array or
/// TopologyType)
/// \param[in] extents The extents
/// \param[in] topology The topology
/// \return The local extents for each process
template <typename ExtentsType, typename TopologyContainerType>
auto divide_by_topology(const ExtentsType& extents,
                        const TopologyContainerType& topology) {
  ExtentsType result;
  for (std::size_t i = 0; i < result.size(); ++i) {
    result.at(i) = extents.at(i) / topology.at(i);
  }
  return result;
}

/// \brief Append a topology to the vector of topologies if it is not already
/// present. No op if the topology is the same as the last topology in the
/// vector. This is used to avoid duplicate topologies in the vector.
/// \tparam iType The index type
/// \tparam DIM The dimension
/// \param[in,out] topologies The vector of topologies
/// \param[in] topology The topology to append
template <typename iType, std::size_t DIM>
void append_topology(std::vector<std::array<iType, DIM>>& topologies,
                     const std::array<iType, DIM>& topology) {
  if (topologies.empty() || topologies.back() != topology) {
    topologies.push_back(topology);
  }
}

/// \brief Check if the axes are ready for FFT
/// \tparam iType The index type
/// \tparam DIM The dimension
/// \param[in] topology The topology
/// \param[in] axes The axes along which the FFT is performed
/// \return A pair of vectors of (remaining axes, ready axes). The first vector
/// contains the axes that are not ready for FFT. The second vector contains the
/// axes that are ready for FFT. remaining axes can be empty
template <typename iType, std::size_t DIM>
auto decompose_fft_axes(const std::array<std::size_t, DIM>& topology,
                        const std::vector<iType>& axes) {
  auto reversed_axes                = KokkosFFT::Impl::reversed(axes);
  std::vector<iType> remaining_axes = reversed_axes;
  std::vector<iType> ready_axes;

  for (const auto axis : reversed_axes) {
    if (topology.at(axis) > 1) break;
    ready_axes.push_back(axis);

    auto it = std::find(remaining_axes.begin(), remaining_axes.end(), axis);
    if (it != remaining_axes.end()) {
      remaining_axes.erase(it);
    }
  }

  return std::make_pair(KokkosFFT::Impl::reversed(remaining_axes),
                        KokkosFFT::Impl::reversed(ready_axes));
}

/// \brief Check if the topology is ready for FFT along the given axes
/// \tparam iType The index type
/// \tparam DIM The dimension
/// \param[in] topology The topology
/// \param[in] axes The axes along which the FFT is performed
/// \return True if the topology is ready for FFT along the given axes, false
/// otherwise
template <typename iType, std::size_t DIM>
bool is_fft_ready(const std::array<std::size_t, DIM>& topology,
                  const std::vector<iType>& axes) {
  return std::none_of(axes.begin(), axes.end(), [&topology](iType axis) {
    return topology.at(axis) > 1;
  });
}

/// \brief Find the transposed topology for a given topology and FFT extents
/// A new topology is distributed along an axis that is not an FFT axis and is
/// not identical to the original topology
///
/// Example
/// topology: (1, P, 1), fft_extents: (N, M, K), axes: {1, 2}
/// -> transposed_topology: (P, 1, 1)
/// topology: (1, P, 1), fft_extents: (1, M, K), axes: {1, 2}
/// -> transposed_topology: nullopt (no transposed topology found)
/// \tparam iType The index type
/// \tparam DIM The dimension
/// \param[in] topology Input topology
/// \param[in] fft_extents Global FFT extents
/// \param[in] axes The axes along which the FFT is performed
/// \return The transposed topology if it is a slab topology, std::nullopt
/// otherwise
template <typename iType, std::size_t DIM>
std::optional<std::array<std::size_t, DIM>> find_transposed_topology(
    const std::array<std::size_t, DIM>& topology,
    const std::array<std::size_t, DIM>& fft_extents,
    const std::vector<iType>& axes) {
  auto p = KokkosFFT::Impl::total_size(topology);
  std::array<std::size_t, DIM> transposed_topology;
  transposed_topology.fill(1);
  for (std::size_t axis = 0; axis < DIM; ++axis) {
    // Skip if the axis is already an FFT axis or if the topology is identical
    // to the original topology
    if (KokkosFFT::Impl::is_found(axes, static_cast<iType>(axis)) ||
        topology.at(axis) > 1)
      continue;
    if (fft_extents.at(axis) / p > 0) {
      transposed_topology.at(axis) = p;
      break;
    }
  }

  // If a candidate found, transposed topology is a slab geometry
  bool is_slab =
      are_specified_topologies(TopologyType::Slab, transposed_topology);
  return is_slab ? std::optional(transposed_topology) : std::nullopt;
}

/// \brief Find the next slab topology for a given topology and FFT extents
/// A new topology is distributed along an axis that is not an FFT axis and is
/// not identical
/// \tparam iType The index type
/// \tparam DIM The dimension
/// \param[in] topology Input topology
/// \param[in] fft_extents Global FFT extents
/// \param[in] axes The axes along which the FFT is performed
/// \return The next slab topology
/// \throws std::runtime_error if no valid next slab topology is found
template <typename iType, std::size_t DIM>
auto find_next_slab_topology(const std::array<iType, DIM>& topology,
                             const std::array<std::size_t, DIM>& fft_extents,
                             const std::vector<iType>& axes) {
  auto current_axes = axes;
  std::array<iType, DIM> next_topology{};

  while (true) {
    auto transposed_topology =
        find_transposed_topology(topology, fft_extents, current_axes);
    if (transposed_topology.has_value()) {
      next_topology = transposed_topology.value();
      break;
    } else {
      // We should have found a slab topology
      if (current_axes.size() == 1) break;

      // We cannot manipulate all_axes one time, make sub-axes by suppressing
      // the first element
      current_axes.erase(current_axes.begin());
    }
  }

  KOKKOSFFT_THROW_IF(KokkosFFT::Impl::total_size(next_topology) == 0,
                     "No valid next slab topology found. Check if the input "
                     "topology is valid and if the FFT axes are correct.");

  return next_topology;
}

/// \brief Get all slab topologies for a given input and output topology
///
/// Example: 3D case
/// In topology: (1, 1, P)
/// Out topology: (1, 1, P)
/// axes: {0, 1, 2}
/// Output: {(1, 1, P), (P, 1, 1), (1, 1, P)}
/// Operation:
/// Transpose -> FFT2 ax = {1, 2} -> Transpose -> FFT1 ax = {0}
///
/// \tparam iType The index type used for the topology.
/// \tparam DIM The dimensionality of the topology.
/// \tparam FFT_DIM The dimensionality of the FFT axes.
///
/// \param[in] gin_extents The global input extents of the data.
/// \param[in] gout_extents The global output extents of the data.
/// \param[in] in_topology The input topology.
/// \param[in] out_topology The output topology.
/// \param[in] axes The axes along which the FFT is performed.
/// \return A vector of all possible slab topologies that can be formed
/// from the input and output topologies, considering the FFT axes.
/// \throws std::runtime_error
/// 1. if the input and output topologies are not slab topologies
/// 2. if the input and output topologies do not have the same size
/// 3. if the input and output extents include 0
/// 4. if valid next slab topology cannot be found for the given input topology
/// and FFT axes
template <typename iType, std::size_t DIM, std::size_t FFT_DIM>
auto get_all_slab_topologies(const std::array<std::size_t, DIM>& gin_extents,
                             const std::array<std::size_t, DIM>& gout_extents,
                             const std::array<std::size_t, DIM>& in_topology,
                             const std::array<std::size_t, DIM>& out_topology,
                             const std::array<iType, FFT_DIM>& axes) {
  static_assert(FFT_DIM >= 1 && FFT_DIM <= 3, "FFT_DIM must be in [1, 3]");
  static_assert(DIM >= 2 && DIM >= FFT_DIM, "DIM >= 2 and DIM >= FFT_DIM");
  static_assert(std::is_unsigned_v<iType>,
                "get_all_slab_topologies: axes must be unsigned");

  // Firstly, check if input topologies are slabs with the same size
  bool is_slab =
      are_specified_topologies(TopologyType::Slab, in_topology, out_topology);

  KOKKOSFFT_THROW_IF(!is_slab,
                     "Input and output topologies must be slab topologies.");

  auto in_topology_size  = KokkosFFT::Impl::total_size(in_topology);
  auto out_topology_size = KokkosFFT::Impl::total_size(out_topology);

  KOKKOSFFT_THROW_IF(in_topology_size != out_topology_size,
                     "Input and output topologies must have the same size.");

  // Check if local input and output sizes are not zero
  auto in_extents  = divide_by_topology(gin_extents, in_topology);
  auto out_extents = divide_by_topology(gout_extents, out_topology);

  auto in_size  = KokkosFFT::Impl::total_size(in_extents);
  auto out_size = KokkosFFT::Impl::total_size(out_extents);

  KOKKOSFFT_THROW_IF(in_size == 0 || out_size == 0,
                     "Input and output extents must not include 0.");

  // Secondly, check if we can perform FFTs without transpose
  auto current_axes     = KokkosFFT::Impl::to_vector(axes);
  auto current_topology = in_topology;

  std::vector<std::array<std::size_t, DIM>> all_topologies;
  std::vector<std::vector<std::size_t>> all_axes;

  auto [remaining_axes, ready_axes] =
      decompose_fft_axes(current_topology, current_axes);

  append_topology(all_topologies, current_topology);
  all_axes.push_back(ready_axes);

  while (!remaining_axes.empty()) {
    // Check if we can perform FFT on output topology without transpose
    if (is_fft_ready(out_topology, remaining_axes)) break;

    // Remaining axes must require transpose
    bool is_transpose_only = remaining_axes == current_axes;
    auto fft_extents       = is_transpose_only ? gin_extents : gout_extents;
    auto next_topology =
        find_next_slab_topology(current_topology, fft_extents, remaining_axes);
    auto [next_remaining_axes, next_ready_axes] =
        decompose_fft_axes(next_topology, remaining_axes);

    // Add new topology and axes
    append_topology(all_topologies, next_topology);
    all_axes.push_back(next_ready_axes);

    current_topology = next_topology;
    remaining_axes   = next_remaining_axes;
  }

  append_topology(all_topologies, out_topology);
  if (all_topologies.size() != all_axes.size()) {
    all_axes.push_back({});
  }
  return all_topologies;
}

/// \brief Get all pencil topologies for a given input and output topology
///
/// Example: 3D case
/// In topology: (1, Px, Py)
/// Out topology: (Px, Py, 1)
/// axes: {0, 1, 2}
/// Output: {(1, Px, Py), (Py, Px, 1), (Py, 1, Px), (Py, Px, 1), (1, Px, Py),}
/// Operation:
/// Transpose to Topology2 -> FFT ax = {2} -> Transpose to Topology4 -> FFT1 ax
/// = {1} Transpose to Topology2 -> Transpose to Topology0 -> FFT1 ax = {0}
/// Topology0: {1, Px, Py}, Topology2: {Py, Px, 1}, Topology4: {Py, 1, Px}
///
/// \tparam iType The index type used for the topology.
/// \tparam DIM The dimensionality of the topology.
/// \tparam FFT_DIM The dimensionality of the FFT axes.
///
/// \param[in] in_topology The input topology.
/// \param[in] out_topology The output topology.
/// \param[in] axes The axes along which the FFT is performed.
/// \param[in] is_same_order If true, the in/out topologies are considered in
/// the same order.
/// \return A vector of all possible slab topologies that can be formed from the
/// input and output topologies, considering the FFT axes.
template <typename iType, std::size_t DIM, std::size_t FFT_DIM,
          typename InLayoutType  = Kokkos::LayoutRight,
          typename OutLayoutType = Kokkos::LayoutRight>
auto get_all_pencil_topologies(
    const Topology<std::size_t, DIM, InLayoutType>& in_topology,
    const Topology<std::size_t, DIM, OutLayoutType>& out_topology,
    const std::array<iType, FFT_DIM>& axes) {
  static_assert(FFT_DIM >= 1 && FFT_DIM <= 3, "FFT_DIM must be in [1, 3]");
  static_assert(DIM >= 3 && DIM >= FFT_DIM, "DIM >= 3 and DIM >= FFT_DIM");
  static_assert(std::is_unsigned_v<iType>,
                "get_all_pencil_topologies: axes must be unsigned");

  using topology_type   = std::array<std::size_t, DIM>;
  using topologies_type = std::vector<topology_type>;
  using axes_type       = std::vector<std::size_t>;
  using layouts_type    = std::vector<std::size_t>;

  bool is_pencil =
      are_specified_topologies(TopologyType::Pencil, in_topology, out_topology);
  KOKKOSFFT_THROW_IF(!is_pencil,
                     "Input and output topologies must be pencil topologies.");

  auto axes_reversed             = KokkosFFT::Impl::to_vector(axes);
  auto non_ones                  = extract_non_one_values(in_topology.array());
  bool has_same_non_one_elements = has_identical_non_ones(non_ones);

  auto in_topology_tmp  = in_topology.array();
  auto out_topology_tmp = out_topology.array();

  if (has_same_non_one_elements) {
    // If the elements are the same, the following strategy does not work
    // Thus, we replace the elements by dummies to manipulate
    std::array<std::size_t, 2> dummies = {2, 3};
    int count                          = 0;
    for (std::size_t i = 0; i < DIM; i++) {
      if (in_topology_tmp.at(i) > 1) {
        in_topology_tmp.at(i) = dummies.at(count);
        count++;
      }
    }
    count = 0;
    if (!std::is_same_v<InLayoutType, OutLayoutType>) {
      std::reverse(dummies.begin(), dummies.end());
    }
    for (std::size_t i = 0; i < DIM; i++) {
      if (out_topology_tmp.at(i) > 1) {
        out_topology_tmp.at(i) = dummies.at(count);
        count++;
      }
    }
  }

  // If LayoutRight, (1, px, py, 1): first_non_one is px
  // If LayoutLeft, (1, py, px, 1): first_non_one is px
  auto first_non_one = std::is_same_v<InLayoutType, Kokkos::LayoutRight>
                           ? extract_non_one_values(in_topology_tmp).at(0)
                           : extract_non_one_values(in_topology_tmp).at(1);

  auto to_original_topologies = [&](const topologies_type& topologies,
                                    const axes_type& trans_axes,
                                    const layouts_type& layouts) {
    if (has_same_non_one_elements) {
      auto non_one             = non_ones.at(0);
      auto original_topologies = topologies;
      for (auto& topology : original_topologies) {
        for (std::size_t i = 0; i < DIM; i++) {
          if (topology.at(i) > 1) topology.at(i) = non_one;
        }
      }
      return std::make_tuple(original_topologies, trans_axes, layouts);
    } else {
      return std::make_tuple(topologies, trans_axes, layouts);
    }
  };

  auto get_layout = [&](const topology_type& topology) {
    // If this condition is satisfied, it means layout right
    std::size_t is_layout_right =
        extract_non_one_values(topology).at(0) == first_non_one;
    return is_layout_right;
  };

  topologies_type topologies;
  axes_type trans_axes;
  layouts_type layouts;

  topologies.push_back(in_topology_tmp);
  layouts.push_back(get_layout(in_topology_tmp));

  auto add_topology = [&](const topology_type& topo) {
    if (topologies.back() != topo) {
      trans_axes.push_back(
          compute_trans_axis(topologies.back(), topo, first_non_one));
      topologies.push_back(topo);
      layouts.push_back(get_layout(topo));
    }
  };

  auto try_add_mid_topology = [&]() {
    try {
      auto mid_topology =
          propose_mid_array(topologies.back(), out_topology_tmp);
      add_topology(mid_topology);
    } catch (std::runtime_error& e) {
    }
  };

  auto finalize = [&]() {
    add_topology(out_topology_tmp);
    return to_original_topologies(topologies, trans_axes, layouts);
  };

  auto is_topology_ready = [&](const std::array<std::size_t, DIM>& topo,
                               const std::vector<iType>& current_axes) {
    for (const auto& axis : current_axes) {
      if (topo.at(axis) != 1) return false;
    }
    return true;
  };

  // Batched case
  // If input or output is ready, we can skip the rest of the logic
  if (is_topology_ready(in_topology_tmp, axes_reversed) ||
      is_topology_ready(out_topology_tmp, axes_reversed)) {
    try_add_mid_topology();
    return finalize();
  }

  std::reverse(axes_reversed.begin(), axes_reversed.end());
  std::array<std::size_t, DIM> shuffled_topology = in_topology_tmp;

  auto last_axis  = axes_reversed.front();
  auto first_axis = axes_reversed.back();
  auto first_dim  = in_topology_tmp.at(last_axis);
  auto last_dim   = out_topology_tmp.at(first_axis);
  if (first_dim == 1) axes_reversed.erase(axes_reversed.begin());
  if (last_dim == 1 && !axes_reversed.empty()) axes_reversed.pop_back();

  for (const auto& axis : axes_reversed) {
    std::size_t swap_idx = 0;

    if (in_topology_tmp == out_topology_tmp) {
      for (std::size_t idx = 0; idx < DIM; idx++) {
        if (shuffled_topology.at(idx) == 1 && idx != axis) {
          swap_idx = idx;
          break;
        }
      }
    } else {
      auto diff_non_one_indices =
          extract_non_one_indices(in_topology_tmp, out_topology_tmp);
      for (auto diff_idx : diff_non_one_indices) {
        if (shuffled_topology.at(diff_idx) == 1 && diff_idx != axis) {
          swap_idx = diff_idx;
          break;
        }
      }
    }
    shuffled_topology = swap_elements(shuffled_topology, axis, swap_idx);

    add_topology(shuffled_topology);
  }

  if (topologies.back() == out_topology_tmp) {
    return to_original_topologies(topologies, trans_axes, layouts);
  }

  try_add_mid_topology();
  return finalize();
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
