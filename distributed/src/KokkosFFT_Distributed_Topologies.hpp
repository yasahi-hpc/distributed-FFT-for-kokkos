#ifndef KOKKOSFFT_DISTRIBUTED_TOPOLOGIES_HPP
#define KOKKOSFFT_DISTRIBUTED_TOPOLOGIES_HPP

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_MPI_Helper.hpp"
#include "KokkosFFT_Distributed_Utils.hpp"
#include "KokkosFFT_Distributed_Types.hpp"

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
inline auto get_topology_type(const ContainerType& topology) {
  static_assert(
      (is_allowed_topology_v<ContainerType>),
      "get_topology_type: topologies must be either in std::array or Topology");
  TopologyType topology_type = TopologyType::Invalid;

  auto size = KokkosFFT::Impl::total_size(topology);
  if (size == 0) {
    topology_type = TopologyType::Empty;
    return topology_type;
  }

  auto non_one_count = count_non_ones(topology);
  if (non_one_count == 0) {
    topology_type = TopologyType::Shared;
  } else if (non_one_count == 1) {
    topology_type = TopologyType::Slab;
  } else if (non_one_count == 2) {
    topology_type = TopologyType::Pencil;
  } else if (non_one_count == 3) {
    topology_type = TopologyType::Brick;
  }

  return topology_type;
}

/// \brief Check if all given topologies are of Shared type
/// \tparam Topologies Variadic template parameter for topology container types
/// \param[in] topologies Topology containers
/// \return true if all topologies are Shared type, false otherwise
template <class... Topologies>
inline bool are_shared_topologies(const Topologies&... topologies) {
  static_assert((are_allowed_topologies_v<Topologies...>),
                "are_shared_topologies: topologies must be either in "
                "std::array or Topology");
  auto is_shared_topology = [](const auto& topology) {
    return get_topology_type(topology) == TopologyType::Shared;
  };
  return (is_shared_topology(topologies) && ...);
}

/// \brief Check if all given topologies are of Slab type
/// \tparam Topologies Variadic template parameter for topology container types
/// \param[in] topologies Topology containers
/// \return true if all topologies are Slab type, false otherwise
template <class... Topologies>
inline bool are_slab_topologies(const Topologies&... topologies) {
  static_assert((are_allowed_topologies_v<Topologies...>),
                "are_slab_topologies: topologies must be either in std::array "
                "or Topology");
  auto is_slab_topology = [](const auto& topology) {
    return get_topology_type(topology) == TopologyType::Slab;
  };
  return (is_slab_topology(topologies) && ...);
}

/// \brief Check if all given topologies are of Pencil type
/// \tparam Topologies Variadic template parameter for topology container types
/// \param[in] topologies Topology containers
/// \return true if all topologies are Pencil type, false otherwise
template <class... Topologies>
inline bool are_pencil_topologies(const Topologies&... topologies) {
  static_assert((are_allowed_topologies_v<Topologies...>),
                "are_pencil_topologies: topologies must be either in "
                "std::array or Topology");
  auto is_pencil_topology = [](const auto& topology) {
    return get_topology_type(topology) == TopologyType::Pencil;
  };
  return (is_pencil_topology(topologies) && ...);
}

/// \brief Check if all given topologies are of Brick type
/// \tparam Topologies Variadic template parameter for topology container types
/// \param[in] topologies Topology containers
/// \return true if all topologies are Brick type, false otherwise
template <class... Topologies>
inline bool are_brick_topologies(const Topologies&... topologies) {
  static_assert((are_allowed_topologies_v<Topologies...>),
                "are_brick_topologies: topologies must be either in "
                "std::array or Topology");
  auto is_brick_topology = [](const auto& topology) {
    return get_topology_type(topology) == TopologyType::Brick;
  };
  return (is_brick_topology(topologies) && ...);
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
    return get_topology_type(topology) == TopologyType::Empty;
  };
  if ((is_empty(topologies) || ...)) {
    return TopologyType::Empty;
  }

  TopologyType topology_type = TopologyType::Invalid;
  if (are_shared_topologies(topologies...)) {
    topology_type = TopologyType::Shared;
  } else if (are_slab_topologies(topologies...)) {
    topology_type = TopologyType::Slab;
  } else if (are_pencil_topologies(topologies...)) {
    topology_type = TopologyType::Pencil;
  } else if (are_brick_topologies(topologies...)) {
    topology_type = TopologyType::Brick;
  }

  return topology_type;
}

// Can we also check that this is a slab?
// Example
// (1, Px, Py, 1) -> (Px, 1, Py, 1): 0-pencil to 1-pencil
// (1, 1, P) -> (1, P, 1): 1-pencil to 2-pencil
// (P, 1, 1) -> (1, P, 1): 1-pencil to 0-pencil

template <std::size_t DIM>
auto get_pencil(const std::array<std::size_t, DIM>& in_topology,
                const std::array<std::size_t, DIM>& out_topology) {
  // Extract topology that is common between in_topology and out_topology
  // std::array<std::size_t, DIM> common_topology = {};
  auto in_size  = KokkosFFT::Impl::total_size(in_topology);
  auto out_size = KokkosFFT::Impl::total_size(out_topology);
  KOKKOSFFT_THROW_IF(in_size == 1 || out_size == 1,
                     "Input and output topologies must have at least one "
                     "non-trivial dimension.");

  KOKKOSFFT_THROW_IF(in_size != out_size,
                     "Input and output topologies must have the same size.");

  std::size_t in_axis = 0, out_axis = 0;
  for (std::size_t i = 0; i < DIM; ++i) {
    if (in_topology[i] != out_topology[i]) {
      if (in_topology[i] == 1) in_axis = i;
      if (out_topology[i] == 1) out_axis = i;
    }
  }

  std::tuple<std::size_t, std::size_t> pencil_array = {in_axis, out_axis};
  return pencil_array;
}

template <std::size_t DIM, typename LayoutType = Kokkos::LayoutRight>
auto get_pencil(const Topology<std::size_t, DIM, LayoutType>& in_topology,
                const Topology<std::size_t, DIM, LayoutType>& out_topology) {
  return get_pencil(in_topology.array(), out_topology.array());
}

// Example
// (1, P) -> (P, 1): x-slab to y-slab
// (P, 1) -> (1, p): y-slab to x-slab
// (1, 1, P) -> (1, P, 1): xy-slab to xz-slab
// (P, 1, 1) -> (1, P, 1): yz-slab to xz-slab

template <std::size_t DIM>
auto get_slab(const std::array<std::size_t, DIM>& in_topology,
              const std::array<std::size_t, DIM>& out_topology) {
  auto in_size  = KokkosFFT::Impl::total_size(in_topology);
  auto out_size = KokkosFFT::Impl::total_size(out_topology);

  KOKKOSFFT_THROW_IF(in_size != out_size,
                     "Input and output topologies must have the same size.");

  bool is_slab = are_slab_topologies(in_topology, out_topology);
  KOKKOSFFT_THROW_IF(!is_slab,
                     "Input and output topologies must be slab topologies.");

  std::size_t in_axis = 0, out_axis = 0;
  for (std::size_t i = 0; i < DIM; ++i) {
    if (in_topology[i] > 1 && out_topology[i] == 1) {
      out_axis = i;
    }
    if (in_topology[i] == 1 && out_topology[i] > 1) {
      in_axis = i;
    }
  }

  std::tuple<std::size_t, std::size_t> slab_array = {in_axis, out_axis};
  return slab_array;
}

template <typename iType, std::size_t DIM = 1>
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
  std::array<iType, DIM> in_trimmed = {}, out_trimmed = {};
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

/// \brief Decompose the FFT axes of the slab geometry into vectors
///        The first vector includes the axes for FFT without transpose
///        The second vector includes the axes for FFT after transpose
///        The third vector includes the axes for remaining FFT
///
/// \tparam iType The index type used for the topology.
/// \tparam DIM The dimensionality of the topology.
/// \tparam FFT_DIM The dimensionality of the FFT axes.
///
template <typename iType, std::size_t DIM = 1, std::size_t FFT_DIM = 1>
std::vector<std::vector<iType>> decompose_axes(
    const std::vector<std::array<std::size_t, DIM>>& topologies,
    const std::array<iType, FFT_DIM>& axes) {
  auto non_negative_axes = KokkosFFT::Impl::convert_base_int_type<std::size_t>(
      KokkosFFT::Impl::convert_negative_axes(axes, DIM));

  // Reverse the axes e.g. {0, 2, 1} -> {1, 2, 0}
  std::vector<std::size_t> axes_reversed =
      KokkosFFT::Impl::reversed(to_vector(non_negative_axes));

  std::vector<std::vector<iType>> all_axes = {};
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

  std::size_t total_axes = 0;
  for (auto ready_axes : all_axes) {
    total_axes += ready_axes.size();
  }

  KOKKOSFFT_THROW_IF(
      total_axes != axes.size(),
      "Axes are not decomposed correctly:" + std::to_string(total_axes) +
          " != " + std::to_string(axes.size()));

  return all_axes;
}

// \brief Get all slab topologies for a given input and output topology
///
/// \tparam iType The index type used for the topology.
/// \tparam DIM The dimensionality of the topology.
/// \tparam FFT_DIM The dimensionality of the FFT axes.
///
/// \param in_topology The input topology.
/// \param out_topology The output topology.
/// \param axes The axes along which the FFT is performed.
/// \return A vector of all possible slab topologies that can be formed
/// from the input and output topologies, considering the FFT axes.
template <typename iType, std::size_t DIM = 1, std::size_t FFT_DIM = 1>
std::vector<std::array<std::size_t, DIM>> get_all_slab_topologies(
    const std::array<std::size_t, DIM>& in_topology,
    const std::array<std::size_t, DIM>& out_topology,
    const std::array<iType, FFT_DIM>& axes) {
  static_assert(FFT_DIM >= 1 && FFT_DIM <= 3, "FFT_DIM must be in [1, 3]");
  static_assert(DIM >= 2 && DIM >= FFT_DIM, "DIM >= 2 and DIM >= FFT_DIM");
  static_assert(std::is_unsigned_v<iType>,
                "get_all_slab_topologies: axes must be unsigned");

  bool is_slab = are_slab_topologies(in_topology, out_topology);
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

  auto axes_vec = to_vector(axes);

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

// \brief Get all pencil topologies for a given input and output topology
///
/// \tparam iType The index type used for the topology.
/// \tparam DIM The dimensionality of the topology.
/// \tparam FFT_DIM The dimensionality of the FFT axes.
///
/// \param in_topology The input topology.
/// \param out_topology The output topology.
/// \param axes The axes along which the FFT is performed.
/// \param is_same_order If true, the in/out topologies are considered in the
/// same order.
/// \return A vector of all possible slab topologies that can be formed from the
/// input and output topologies, considering the FFT axes.
template <typename iType, std::size_t DIM = 1, std::size_t FFT_DIM = 1,
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

  bool is_pencil = are_pencil_topologies(in_topology, out_topology);

  KOKKOSFFT_THROW_IF(!is_pencil,
                     "Input and output topologies must be pencil topologies.");

  auto axes_reversed             = to_vector(axes);
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

  // 3D case
  std::array<std::size_t, DIM> topology = {};
  topology.fill(1);

  // Batched case
  // If input or output is ready, we can skip the rest of the logic
  bool is_input_ready = true;
  for (const auto& axis : axes_reversed) {
    auto non_negative_axis = KokkosFFT::Impl::convert_negative_axis(axis, DIM);
    std::size_t unsigned_axis = static_cast<std::size_t>(non_negative_axis);
    auto dim                  = in_topology.at(unsigned_axis);
    if (dim != 1) is_input_ready = false;
  }

  bool is_output_ready = true;
  for (const auto& axis : axes_reversed) {
    auto non_negative_axis = KokkosFFT::Impl::convert_negative_axis(axis, DIM);
    std::size_t unsigned_axis = static_cast<std::size_t>(non_negative_axis);
    auto dim                  = out_topology_tmp.at(unsigned_axis);
    if (dim != 1) is_output_ready = false;
  }

  if (is_input_ready || is_output_ready) {
    try {
      auto mid_topology = get_mid_array(topologies.back(), out_topology_tmp);
      trans_axes.push_back(
          get_trans_axis(topologies.back(), mid_topology, first_non_one));
      topologies.push_back(mid_topology);
      layouts.push_back(get_layout(mid_topology));
    } catch (std::runtime_error& e) {
    }
    if (topologies.back() != out_topology_tmp) {
      trans_axes.push_back(
          get_trans_axis(topologies.back(), out_topology_tmp, first_non_one));
      topologies.push_back(out_topology_tmp);
      layouts.push_back(get_layout(out_topology_tmp));
    }
    return to_original_topologies(topologies, trans_axes, layouts);
  }

  std::reverse(axes_reversed.begin(), axes_reversed.end());
  std::array<std::size_t, DIM> shuffled_topology = in_topology_tmp;
  if (in_topology_tmp == out_topology_tmp) {
    auto last_axis  = axes_reversed.front();
    auto first_axis = axes_reversed.back();
    auto first_dim  = in_topology_tmp.at(last_axis);
    auto last_dim   = out_topology_tmp.at(first_axis);
    if (first_dim == 1) axes_reversed.erase(axes_reversed.begin());
    if (last_dim == 1 && !axes_reversed.empty()) axes_reversed.pop_back();
    for (const auto& axis : axes_reversed) {
      std::size_t swap_idx = 0;
      auto non_negative_axis =
          KokkosFFT::Impl::convert_negative_axis(axis, DIM);
      std::size_t unsigned_axis = static_cast<std::size_t>(non_negative_axis);
      for (std::size_t idx = 0; idx < DIM; idx++) {
        if (shuffled_topology.at(idx) == 1 && idx != unsigned_axis) {
          swap_idx = idx;
          break;
        }
      }
      shuffled_topology =
          swap_elements(shuffled_topology, unsigned_axis, swap_idx);

      if (topologies.back() != shuffled_topology) {
        trans_axes.push_back(get_trans_axis(topologies.back(),
                                            shuffled_topology, first_non_one));
        topologies.push_back(shuffled_topology);
        layouts.push_back(get_layout(shuffled_topology));
      }
    }

    if (topologies.back() == out_topology_tmp) {
      return to_original_topologies(topologies, trans_axes, layouts);
    }

    try {
      auto mid_topology = get_mid_array(topologies.back(), out_topology_tmp);
      trans_axes.push_back(
          get_trans_axis(topologies.back(), mid_topology, first_non_one));
      topologies.push_back(mid_topology);
      layouts.push_back(get_layout(mid_topology));
    } catch (std::runtime_error& e) {
    }

    if (topologies.back() == out_topology_tmp)
      return to_original_topologies(topologies, trans_axes, layouts);

    trans_axes.push_back(
        get_trans_axis(topologies.back(), out_topology_tmp, first_non_one));
    topologies.push_back(out_topology_tmp);
    layouts.push_back(get_layout(out_topology_tmp));
    return to_original_topologies(topologies, trans_axes, layouts);
  }

  std::vector<std::size_t> diff_non_one_indices =
      extract_non_one_indices(in_topology_tmp, out_topology_tmp);
  auto last_axis  = axes_reversed.front();
  auto first_axis = axes_reversed.back();
  auto first_dim  = in_topology_tmp.at(last_axis);
  auto last_dim   = out_topology_tmp.at(first_axis);
  if (first_dim == 1) axes_reversed.erase(axes_reversed.begin());
  if (last_dim == 1 && !axes_reversed.empty()) axes_reversed.pop_back();
  for (const auto& axis : axes_reversed) {
    std::size_t swap_idx   = 0;
    auto non_negative_axis = KokkosFFT::Impl::convert_negative_axis(axis, DIM);
    std::size_t unsigned_axis = static_cast<std::size_t>(non_negative_axis);
    for (auto diff_idx : diff_non_one_indices) {
      if (shuffled_topology.at(diff_idx) == 1 && diff_idx != unsigned_axis) {
        swap_idx = diff_idx;
        break;
      }
    }
    shuffled_topology =
        swap_elements(shuffled_topology, unsigned_axis, swap_idx);

    if (topologies.back() != shuffled_topology) {
      trans_axes.push_back(
          get_trans_axis(topologies.back(), shuffled_topology, first_non_one));
      topologies.push_back(shuffled_topology);
      layouts.push_back(get_layout(shuffled_topology));
    }
  }
  if (topologies.back() == out_topology_tmp)
    return to_original_topologies(topologies, trans_axes, layouts);

  try {
    auto mid_topology = get_mid_array(topologies.back(), out_topology_tmp);
    trans_axes.push_back(
        get_trans_axis(topologies.back(), mid_topology, first_non_one));
    topologies.push_back(mid_topology);
    layouts.push_back(get_layout(mid_topology));
  } catch (std::runtime_error& e) {
  }
  if (topologies.back() == out_topology_tmp)
    return to_original_topologies(topologies, trans_axes, layouts);

  trans_axes.push_back(
      get_trans_axis(topologies.back(), out_topology_tmp, first_non_one));
  topologies.push_back(out_topology_tmp);
  layouts.push_back(get_layout(out_topology_tmp));

  return to_original_topologies(topologies, trans_axes, layouts);
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
