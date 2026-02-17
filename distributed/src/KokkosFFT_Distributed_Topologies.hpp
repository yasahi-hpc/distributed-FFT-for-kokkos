#ifndef KOKKOSFFT_DISTRIBUTED_TOPOLOGIES_HPP
#define KOKKOSFFT_DISTRIBUTED_TOPOLOGIES_HPP

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_MPI_Extents.hpp"
#include "KokkosFFT_Distributed_Types.hpp"
#include "KokkosFFT_Distributed_ContainerAnalyses.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief Get the topology type from the given topology container
/// Empty topology: 0 is included in the topology
/// Shared topology: non-one element is not included in the topology
/// Slab topology: 1 non-one element is included in the topology
/// Pencil topology: 2 non-one elements are included in the topology
/// Brick topology: 3 non-one elements are included in the topology
/// Invalid topology: more than 3 non-one elements are included in the topology
///
/// \tparam ContainerType Topology container type (std::array or Topology)
/// \param[in] topology Topology container
/// \return TopologyType enum value representing the topology type
template <typename ContainerType>
inline auto to_topology_type(const ContainerType& topology) {
  static_assert(
      (is_allowed_topology_v<ContainerType>),
      "to_topology_type: topologies must be either in std::array or Topology");

  auto size = KokkosFFT::Impl::total_size(topology);
  if (size == 0) return TopologyType::Empty;

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
/// \return true if all topologies are Shared type, false otherwise
template <class... Topologies>
inline bool are_specified_topologies(const TopologyType topology_type,
                                     const Topologies&... topologies) {
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
/// \tparam Topologies Variadic template parameter for topology container types
/// \param[in] topology Topology container
/// \return TopologyType enum value representing the topology type
template <class... Topologies>
inline auto get_common_topology_type(const Topologies&... topologies) {
  static_assert((are_allowed_topologies_v<Topologies...>),
                "are_brick_topologies: topologies must be either in "
                "std::array or Topology");

  // Quick return if empty topology is found
  auto is_empty = [](const auto& topology) {
    return to_topology_type(topology) == TopologyType::Empty;
  };
  if ((is_empty(topologies) || ...)) {
    return TopologyType::Empty;
  }

  const std::array<TopologyType, 4> all_topology_types = {
      TopologyType::Shared, TopologyType::Slab, TopologyType::Pencil,
      TopologyType::Brick};
  for (TopologyType t : all_topology_types) {
    if (are_specified_topologies(t, topologies...)) {
      return t;
    }
  }

  return TopologyType::Invalid;
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
/// the same size
/// \throws std::runtime_error if the input and output topologies are not slab
/// topologies
template <typename iType, std::size_t DIM>
auto slab_in_out_axes(const std::array<iType, DIM>& in_topology,
                      const std::array<iType, DIM>& out_topology) {
  auto in_size  = KokkosFFT::Impl::total_size(in_topology);
  auto out_size = KokkosFFT::Impl::total_size(out_topology);

  KOKKOSFFT_THROW_IF(in_size != out_size,
                     "Input and output topologies must have the same size.");

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
std::array<iType, DIM> get_mid_array(const std::array<iType, DIM>& in,
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
  for (size_t i = 0; i < diff_non_one_indices.size(); ++i) {
    for (size_t j = i + 1; j < diff_non_one_indices.size(); ++j) {
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
    msg += " Input axes: ";
    for (auto axis : axes) {
      msg += std::to_string(axis) + " ";
    }
    msg += "\n";
    msg += "Decomposed axes: \n";
    for (std::size_t i = 0; i < all_axes.size(); ++i) {
      auto topology = topologies.at(i);
      msg += "at topology (";
      msg += std::to_string(topology.at(0));
      for (std::size_t j = 1; j < topology.size(); ++j) {
        msg += ", " + std::to_string(topology.at(j));
      }
      msg += "): Ready axes: ";
      if (all_axes.at(i).empty()) {
        msg += "None";
      } else {
        auto axis = all_axes.at(i);
        msg += "(";
        msg += std::to_string(axis.at(0));
        for (std::size_t j = 1; j < axis.size(); ++j) {
          msg += ", " + std::to_string(axis.at(j));
        }
        msg += ")";
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
/// \return The axis to transpose (0 or 1)
/// \throws std::runtime_error if the input and output topologies do not have
/// exactly two non-one elements
/// \throws std::runtime_error if the input and output topologies have identical
/// non-one elements
/// \throws std::runtime_error if the input and output topologies differ exactly
/// two positions
template <typename iType, std::size_t DIM>
auto compute_trans_axis(const std::array<iType, DIM>& in_topology,
                        const std::array<iType, DIM>& out_topology,
                        iType first_non_one) {
  auto in_non_ones  = extract_non_one_values(in_topology);
  auto out_non_ones = extract_non_one_values(out_topology);
  KOKKOSFFT_THROW_IF(
      in_non_ones.size() != 2 || out_non_ones.size() != 2,
      "Input and output topologies must have exactly two non-one "
      "elements.");
  KOKKOSFFT_THROW_IF(has_identical_non_ones(in_non_ones) ||
                         has_identical_non_ones(out_non_ones),
                     "Input and output topologies must not have identical "
                     "non-one elements.");

  auto diff_indices = extract_different_indices(in_topology, out_topology);
  KOKKOSFFT_THROW_IF(
      diff_indices.size() != 2,
      "Input and output topologies must differ exactly two positions");

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
/// \param[in] in_topology The input topology.
/// \param[in] out_topology The output topology.
/// \param[in] axes The axes along which the FFT is performed.
/// \return A vector of all possible slab topologies that can be formed
/// from the input and output topologies, considering the FFT axes.
template <typename iType, std::size_t DIM, std::size_t FFT_DIM>
std::vector<std::array<std::size_t, DIM>> get_all_slab_topologies(
    const std::array<std::size_t, DIM>& in_topology,
    const std::array<std::size_t, DIM>& out_topology,
    const std::array<iType, FFT_DIM>& axes) {
  static_assert(FFT_DIM >= 1 && FFT_DIM <= 3, "FFT_DIM must be in [1, 3]");
  static_assert(DIM >= 2 && DIM >= FFT_DIM, "DIM >= 2 and DIM >= FFT_DIM");
  static_assert(std::is_unsigned_v<iType>,
                "get_all_slab_topologies: axes must be unsigned");

  bool is_slab =
      are_specified_topologies(TopologyType::Slab, in_topology, out_topology);
  KOKKOSFFT_THROW_IF(!is_slab,
                     "Input and output topologies must be slab topologies.");

  std::vector<std::array<std::size_t, DIM>> topologies;
  topologies.push_back(in_topology);
  auto p = KokkosFFT::Impl::total_size(in_topology);

  auto add_topology_at = [&](std::size_t axis) {
    std::array<std::size_t, DIM> t;
    t.fill(1);
    t.at(axis) = p;
    if (topologies.back() != t) {
      topologies.push_back(t);
    }
  };

  auto add_topology_at_first_one = [&](std::array<std::size_t, DIM> topo) {
    for (std::size_t i = 0; i < DIM; ++i) {
      if (topo.at(i) == 1) {
        add_topology_at(i);
        break;
      }
    }
  };

  auto finalize = [&]() {
    if (topologies.back() != out_topology) {
      topologies.push_back(out_topology);
    }
    return topologies;
  };

  auto axes_vec = KokkosFFT::Impl::to_vector(axes);

  // 2D case
  if constexpr (DIM == 2 && FFT_DIM == 2) {
    if (in_topology == out_topology) {
      add_topology_at_first_one(in_topology);
    } else {
      // In case one transpose needed for first fft
      // then we need to add two swapped topologies
      auto last_axis = axes.back();
      auto first_dim = in_topology.at(last_axis);
      if (first_dim != 1) {
        add_topology_at_first_one(in_topology);
        if (topologies.back() != in_topology) {
          topologies.push_back(in_topology);
        }
      }
    }
    return finalize();
  }

  // 3D case
  if constexpr (DIM == 3 && FFT_DIM == 3) {
    if (in_topology == out_topology) {
      auto last_axis = axes.back();
      auto first_dim = in_topology.at(last_axis);
      if (first_dim != 1) {
        add_topology_at(axes.at(0));
      } else {
        add_topology_at(axes.back());
      }
    } else {
      auto last_axis = axes.back();
      auto first_dim = in_topology.at(last_axis);
      if (first_dim != 1) {
        add_topology_at(axes.at(0));
        auto last_dim = out_topology.at(axes.front());
        if (last_dim != 1) {
          add_topology_at_first_one(topologies.back());
        }
      } else {
        auto mid_dim = in_topology.at(axes.at(1));
        if (mid_dim != 1) {
          add_topology_at(axes.back());
        } else {
          auto last_dim = out_topology.at(axes.front());
          if (last_dim != 1) {
            add_topology_at_first_one(topologies.back());
          }
        }
      }
    }
    return finalize();
  }

  // Batched case
  // If input or output is ready, we can skip the rest of the logic
  auto axes_reversed     = axes_vec;
  auto is_topology_ready = [&](const std::array<std::size_t, DIM>& topo) {
    bool is_ready = true;
    for (const auto& axis : axes_reversed) {
      if (topo.at(axis) > 1) is_ready = false;
    }
    return is_ready;
  };

  if (is_topology_ready(in_topology) || is_topology_ready(out_topology)) {
    return finalize();
  }

  // If the conditions above are not satisfied, we need a
  // intermediate topology
  if constexpr (FFT_DIM == 1) {
    for (std::size_t i = 0; i < DIM; ++i) {
      if (!KokkosFFT::Impl::is_found(axes_reversed, i)) {
        add_topology_at(i);
        break;
      }
    }

    return finalize();
  } else if constexpr (DIM > 3 && FFT_DIM == 3) {
    // First, remove the already ready axes
    auto sorted_axes = axes_vec;
    std::sort(sorted_axes.begin(), sorted_axes.end());
    std::reverse(axes_reversed.begin(), axes_reversed.end());

    // Get axes ready for transform
    std::vector<iType> ready_axes;
    for (auto axis : axes_reversed) {
      if (in_topology.at(axis) > 1) break;
      ready_axes.push_back(axis);
    }

    if (ready_axes.size() == 0) {
      add_topology_at(axes_reversed.back());
      for (auto axis : axes_reversed) {
        if (topologies.back().at(axis) > 1) break;
        ready_axes.push_back(axis);
      }
    }

    // Remove already registered axes
    for (auto axis : ready_axes) {
      auto it = std::find(axes_reversed.begin(), axes_reversed.end(), axis);
      if (it != axes_reversed.end()) {
        axes_reversed.erase(it);
      }
    }
    // test if output is ready
    bool is_ready = is_topology_ready(out_topology);
    if (!is_ready) {
      // Need to find a new topology
      for (auto axis : sorted_axes) {
        if (!KokkosFFT::Impl::is_found(axes_reversed, axis)) {
          add_topology_at(axis);
          break;
        }
      }
    }

    return finalize();
  } else {
    // First, remove the already ready axes
    std::reverse(axes_reversed.begin(), axes_reversed.end());

    // Get axes ready for transform
    std::vector<iType> ready_axes;
    for (auto axis : axes_reversed) {
      if (in_topology.at(axis) > 1) break;
      ready_axes.push_back(axis);
    }

    for (auto axis : ready_axes) {
      auto it = std::find(axes_reversed.begin(), axes_reversed.end(), axis);
      if (it != axes_reversed.end()) {
        axes_reversed.erase(it);
      }
    }

    // test if output is ready
    bool is_ready = is_topology_ready(out_topology);
    if (!is_ready) {
      // Need to find a new topology
      for (std::size_t i = 0; i < DIM; ++i) {
        if (!KokkosFFT::Impl::is_found(axes_reversed, i)) {
          add_topology_at(i);
          break;
        }
      }
    }

    return finalize();
  }
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
/// the same order. \return A vector of all possible slab topologies that can be
/// formed from the input and output topologies, considering the FFT axes.
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
      auto mid_topology = get_mid_array(topologies.back(), out_topology_tmp);
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
