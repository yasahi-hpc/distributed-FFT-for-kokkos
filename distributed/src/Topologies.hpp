#ifndef TOPOLOGIES_HPP
#define TOPOLOGIES_HPP

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "MPI_Helper.hpp"
#include "Utils.hpp"
#include "Types.hpp"

template <std::size_t DIM = 1>
inline auto get_topology_type(const std::array<std::size_t, DIM>& topology) {
  TopologyType topology_type = TopologyType::Invalid;

  auto size = get_size(topology);
  KOKKOSFFT_THROW_IF(size == 0, "topology must not be size 0.");
  int non_one_count = countNonOneComponents(topology);
  if (non_one_count == 0) {
    topology_type = TopologyType::Shared;
  } else if (non_one_count == 1) {
    topology_type = TopologyType::Slab;
  } else if (non_one_count == 2) {
    topology_type = TopologyType::Pencil;
  } else {
    KOKKOSFFT_THROW_IF(true,
                       "topology must have at most two non-one elements.");
  }

  return topology_type;
}

template <std::size_t DIM = 1>
inline bool is_shared_topology(const std::array<std::size_t, DIM>& topology) {
  bool is_shared = false;
  try {
    is_shared = get_topology_type(topology) == TopologyType::Shared;
  } catch (std::runtime_error& e) {
  }
  return is_shared;
}

template <std::size_t DIM = 1>
inline bool is_slab_topology(const std::array<std::size_t, DIM>& topology) {
  bool is_slab = false;
  try {
    is_slab = get_topology_type(topology) == TopologyType::Slab;
  } catch (std::runtime_error& e) {
  }
  return is_slab;
}

template <std::size_t DIM = 1>
inline bool is_pencil_topology(const std::array<std::size_t, DIM>& topology) {
  bool is_pencil = false;
  try {
    is_pencil = get_topology_type(topology) == TopologyType::Pencil;
  } catch (std::runtime_error& e) {
  }
  return is_pencil;
}

template <std::size_t DIM = 1, typename LayoutType = Kokkos::LayoutRight>
inline bool is_pencil_topology(
    const Topology<std::size_t, DIM, LayoutType>& topology) {
  return is_pencil_topology(topology.array());
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
  auto in_size  = get_size(in_topology);
  auto out_size = get_size(out_topology);
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
      // out_axis = i;
      // common_topology[i] = in_topology[i];
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
  auto in_size  = get_size(in_topology);
  auto out_size = get_size(out_topology);

  KOKKOSFFT_THROW_IF(in_size != out_size,
                     "Input and output topologies must have the same size.");

  bool is_slab =
      is_slab_topology(in_topology) && is_slab_topology(out_topology);
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
  std::vector<iType> diff_indices  = find_differences(in, out);
  std::set<iType> diffs            = diff_sets(in, out);
  std::vector<iType> diff_non_ones = find_non_ones(in, out);

  // std::cout << "diff_non_ones.size() " << diff_non_ones.size() << std::endl;
  // std::cout << "diff_indices.size() " << diff_indices.size() << std::endl;
  // std::cout << "diffs.size() " << diffs.size() << std::endl;

  KOKKOSFFT_THROW_IF(diff_non_ones.size() < 3,
                     "The total number of non-one elements either in Input and "
                     "output topologies must be three.");
  KOKKOSFFT_THROW_IF(
      diff_indices.size() < 3 && diffs.size() == 3,
      "Input and output topologies must differ exactly three positions.");
  /*
  KOKKOSFFT_THROW_IF(diff_non_ones.size() != 3,
                     "The total number of non-one elements either in Input and "
                     "output topologies must be three.");
  KOKKOSFFT_THROW_IF(
      diff_indices.size() != 3 && diffs.size() == 3,
      "Input and output topologies must differ exactly three positions.");
  */

  // Only copy the exchangable indices from original arrays in and out
  std::array<iType, DIM> in_trimed = {}, out_trimed = {};
  for (auto diff_idx : diff_indices) {
    in_trimed.at(diff_idx)  = in.at(diff_idx);
    out_trimed.at(diff_idx) = out.at(diff_idx);
  }

  iType idx_one_in  = KokkosFFT::Impl::get_index(in_trimed, iType(1));
  iType idx_one_out = KokkosFFT::Impl::get_index(out_trimed, iType(1));

  // Try all combinations of 2 indices for a single valid swap
  for (size_t i = 0; i < diff_non_ones.size(); ++i) {
    for (size_t j = i + 1; j < diff_non_ones.size(); ++j) {
      iType idx_in               = diff_non_ones.at(i);
      iType idx_out              = diff_non_ones.at(j);
      std::array<iType, DIM> mid = swap_elements(in, idx_in, idx_out);
      iType idx_one_mid          = KokkosFFT::Impl::get_index(mid, iType(1));
      if ((find_differences(mid, out).size() == 2) &&
          !(idx_one_mid == idx_one_in || idx_one_mid == idx_one_out)) {
        return mid;
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
  std::vector<std::size_t> axes_reversed;
  for (std::size_t i = 0; i < axes.size(); ++i) {
    auto non_negative_axis =
        KokkosFFT::Impl::convert_negative_axis<int, DIM>(axes.at(i));
    std::size_t unsigned_axis = static_cast<std::size_t>(non_negative_axis);
    axes_reversed.push_back(unsigned_axis);
  }

  // Reverse the axes e.g. {0, 2, 1} -> {1, 2, 0}
  std::reverse(axes_reversed.begin(), axes_reversed.end());

  std::vector<std::vector<iType>> all_axes = {};
  for (auto topology : topologies) {
    // KOKKOSFFT_THROW_IF(!is_slab_topology(topology),
    //                  "Topology must be a slab topology.");
    std::vector<iType> ready_axes;
    for (auto axis : axes_reversed) {
      if (topology.at(axis) > 1) break;
      ready_axes.push_back(axis);
    }
    // We need to reverse the axes again
    // i.e. {1, 2} -> {2, 1}
    std::reverse(ready_axes.begin(), ready_axes.end());
    all_axes.push_back(ready_axes);

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

  bool is_slab =
      is_slab_topology(in_topology) && is_slab_topology(out_topology);
  KOKKOSFFT_THROW_IF(!is_slab,
                     "Input and output topologies must be slab topologies.");

  std::vector<std::array<std::size_t, DIM>> topologies;
  topologies.push_back(in_topology);

  std::vector<std::size_t> axes_reversed;
  for (std::size_t i = 0; i < axes.size(); ++i) {
    auto non_negative_axis =
        KokkosFFT::Impl::convert_negative_axis<int, DIM>(axes.at(i));
    std::size_t unsigned_axis = static_cast<std::size_t>(non_negative_axis);
    axes_reversed.push_back(unsigned_axis);
  }

  // 2D case
  std::array<std::size_t, DIM> topology = {};
  topology.fill(1);

  if constexpr (DIM == 2 && FFT_DIM == 2) {
    if (in_topology == out_topology) {
      auto p = get_size(in_topology);
      for (std::size_t i = 0; i < DIM; ++i) {
        if (in_topology.at(i) == 1) {
          topology.at(i) = p;
          break;
        }
      }
      topologies.push_back(topology);
    } else {
      // In case one transpose needed for first fft
      // then we need to add two swapped topologies
      auto last_axis = axes.back();
      auto first_dim = in_topology.at(last_axis);
      if (first_dim != 1) {
        auto p = get_size(in_topology);
        for (std::size_t i = 0; i < DIM; ++i) {
          if (in_topology.at(i) == 1) {
            topology.at(i) = p;
            break;
          }
        }
        topologies.push_back(topology);
        topologies.push_back(in_topology);
      }
    }

    if (topologies.back() == out_topology) return topologies;
    topologies.push_back(out_topology);
    return topologies;
  }

  // 3D case
  if constexpr (DIM == 3 && FFT_DIM == 3) {
    if (in_topology == out_topology) {
      auto p = get_size(in_topology);
      topology.fill(p);
      auto last_axis = axes.back();
      auto first_dim = in_topology.at(last_axis);
      if (first_dim != 1) {
        for (std::size_t i = 1; i < axes.size(); ++i) {
          topology.at(axes.at(i)) = 1;
        }
      } else {
        for (std::size_t i = 0; i < axes.size() - 1; ++i) {
          topology.at(axes.at(i)) = 1;
        }
      }
      topologies.push_back(topology);
    } else {
      auto p = get_size(in_topology);
      topology.fill(p);
      auto last_axis = axes.back();
      auto first_dim = in_topology.at(last_axis);
      if (first_dim != 1) {
        for (std::size_t i = 1; i < axes.size(); ++i) {
          topology.at(axes.at(i)) = 1;
        }
        topologies.push_back(topology);
        auto last_dim = out_topology.at(axes.front());
        if (last_dim != 1) {
          topology.fill(1);
          for (std::size_t i = 0; i < DIM; ++i) {
            if (topologies.back().at(i) == 1) {
              topology.at(i) = p;
              break;
            }
          }
          topologies.push_back(topology);
        }
      } else {
        auto mid_dim = in_topology.at(axes.at(1));
        if (mid_dim != 1) {
          topology.fill(p);
          for (std::size_t i = 0; i < axes.size() - 1; ++i) {
            topology.at(axes.at(i)) = 1;
          }
          topologies.push_back(topology);
        } else {
          auto last_dim = out_topology.at(axes.front());
          if (last_dim != 1) {
            topology.fill(1);
            for (std::size_t i = 0; i < DIM; ++i) {
              if (topologies.back().at(i) == 1) {
                topology.at(i) = p;
                break;
              }
            }
            topologies.push_back(topology);
          }
        }
      }
    }

    if (topologies.back() == out_topology) return topologies;
    topologies.push_back(out_topology);
    return topologies;
  }

  // Batched case
  // If input or output is ready, we can skip the rest of the logic
  bool is_input_ready = true;
  for (const auto& axis : axes_reversed) {
    auto non_negative_axis =
        KokkosFFT::Impl::convert_negative_axis<int, DIM>(axis);
    std::size_t unsigned_axis = static_cast<std::size_t>(non_negative_axis);
    auto dim                  = in_topology.at(unsigned_axis);
    if (dim != 1) is_input_ready = false;
  }

  bool is_output_ready = true;
  for (const auto& axis : axes_reversed) {
    auto non_negative_axis =
        KokkosFFT::Impl::convert_negative_axis<int, DIM>(axis);
    std::size_t unsigned_axis = static_cast<std::size_t>(non_negative_axis);
    auto dim                  = out_topology.at(unsigned_axis);
    if (dim != 1) is_output_ready = false;
  }

  if (is_input_ready || is_output_ready) {
    if (topologies.back() != out_topology) topologies.push_back(out_topology);
    return topologies;
  }

  // If the conditions above are not satisified, we need a
  // intermediate topology
  if constexpr (DIM > 3 && FFT_DIM == 3) {
    // First, remove the already ready axes
    std::vector<std::size_t> sorted_axes = axes_reversed;
    std::sort(sorted_axes.begin(), sorted_axes.end());
    std::reverse(axes_reversed.begin(), axes_reversed.end());

    // Get axes ready for transform
    std::vector<iType> ready_axes;
    for (auto axis : axes_reversed) {
      if (in_topology.at(axis) > 1) break;
      ready_axes.push_back(axis);
    }

    auto p = get_size(in_topology);
    if (ready_axes.size() == 0) {
      topology.at(axes_reversed.back()) = p;
      topologies.push_back(topology);
      for (auto axis : axes_reversed) {
        if (topology.at(axis) > 1) break;
        ready_axes.push_back(axis);
      }
      topology.fill(1);
    }

    // Remove already registered axes
    for (auto axis : ready_axes) {
      auto it = std::find(axes_reversed.begin(), axes_reversed.end(), axis);
      if (it != axes_reversed.end()) {
        axes_reversed.erase(it);
      }
    }
    // test if output is ready
    bool is_ready = true;
    for (const auto& axis : axes_reversed) {
      if (out_topology.at(axis) > 1) is_ready = false;
    }
    if (is_ready) {
      topologies.push_back(out_topology);
    } else {
      // Need to find a new topology
      for (auto axis : sorted_axes) {
        if (!KokkosFFT::Impl::is_found(axes_reversed, axis)) {
          topology.at(axis) = p;
          topologies.push_back(topology);
          break;
        }
      }
    }

    if (topologies.back() == out_topology) return topologies;
    topologies.push_back(out_topology);

    return topologies;
  } else if constexpr (FFT_DIM == 1) {
    for (std::size_t i = 0; i < DIM; ++i) {
      if (!KokkosFFT::Impl::is_found(axes_reversed, i)) {
        auto p         = get_size(in_topology);
        topology.at(i) = p;
        topologies.push_back(topology);
        break;
      }
    }

    if (topologies.back() == out_topology) return topologies;
    topologies.push_back(out_topology);

    return topologies;
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
    bool is_ready = true;
    for (const auto& axis : axes_reversed) {
      if (out_topology.at(axis) > 1) is_ready = false;
    }
    if (is_ready) {
      topologies.push_back(out_topology);
    } else {
      // Need to find a new topology
      auto p = get_size(in_topology);
      topology.fill(1);
      for (std::size_t i = 0; i < DIM; ++i) {
        if (!KokkosFFT::Impl::is_found(axes_reversed, i)) {
          topology.at(i) = p;
          topologies.push_back(topology);
          break;
        }
      }
    }

    if (topologies.back() == out_topology) return topologies;
    topologies.push_back(out_topology);

    return topologies;
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

  using topologies_type = std::vector<std::array<std::size_t, DIM>>;
  using axes_type       = std::vector<std::size_t>;

  bool is_pencil = is_pencil_topology(in_topology.array()) &&
                   is_pencil_topology(out_topology.array());
  KOKKOSFFT_THROW_IF(!is_pencil,
                     "Input and output topologies must be pencil topologies.");

  std::vector<std::size_t> axes_reversed;
  for (std::size_t i = 0; i < axes.size(); ++i) {
    auto non_negative_axis =
        KokkosFFT::Impl::convert_negative_axis<int, DIM>(axes.at(i));
    std::size_t unsigned_axis = static_cast<std::size_t>(non_negative_axis);
    axes_reversed.push_back(unsigned_axis);
  }

  auto non_ones                  = find_non_ones(in_topology.array());
  bool has_same_non_one_elements = has_identical_non_ones(non_ones);

  auto in_topology_tmp  = in_topology.array();
  auto out_topology_tmp = out_topology.array();

  if (has_same_non_one_elements) {
    // If the elements are the same, the following startegy does not work
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

  auto first_non_one = std::is_same_v<InLayoutType, Kokkos::LayoutRight>
                           ? find_non_ones(in_topology_tmp).at(0)
                           : find_non_ones(in_topology_tmp).at(1);

  auto to_original_topologies = [&](const topologies_type& topologies,
                                    const axes_type& trans_axes) {
    if (has_same_non_one_elements) {
      auto non_one             = non_ones.at(0);
      auto original_topologies = topologies;
      for (auto& topology : original_topologies) {
        for (std::size_t i = 0; i < DIM; i++) {
          if (topology.at(i) > 1) topology.at(i) = non_one;
        }
      }
      return std::make_tuple(original_topologies, trans_axes);
    } else {
      return std::make_tuple(topologies, trans_axes);
    }
  };

  topologies_type topologies;
  axes_type trans_axes;

  topologies.push_back(in_topology_tmp);

  // 3D case
  std::array<std::size_t, DIM> topology = {};
  topology.fill(1);

  // Batched case
  // If input or output is ready, we can skip the rest of the logic
  bool is_input_ready = true;
  for (const auto& axis : axes_reversed) {
    auto non_negative_axis =
        KokkosFFT::Impl::convert_negative_axis<int, DIM>(axis);
    std::size_t unsigned_axis = static_cast<std::size_t>(non_negative_axis);
    auto dim                  = in_topology.at(unsigned_axis);
    if (dim != 1) is_input_ready = false;
  }

  bool is_output_ready = true;
  for (const auto& axis : axes_reversed) {
    auto non_negative_axis =
        KokkosFFT::Impl::convert_negative_axis<int, DIM>(axis);
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
    } catch (std::runtime_error& e) {
    }
    if (topologies.back() != out_topology_tmp) {
      trans_axes.push_back(
          get_trans_axis(topologies.back(), out_topology_tmp, first_non_one));
      topologies.push_back(out_topology_tmp);
    }
    return to_original_topologies(topologies, trans_axes);
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
      // std::cout << "axis" << axis << std::endl;
      std::size_t swap_idx = 0;
      auto non_negative_axis =
          KokkosFFT::Impl::convert_negative_axis<int, DIM>(axis);
      std::size_t unsigned_axis = static_cast<std::size_t>(non_negative_axis);
      for (std::size_t idx = 0; idx < DIM; idx++) {
        if (shuffled_topology.at(idx) == 1 && idx != unsigned_axis) {
          swap_idx = idx;
          break;
        }
      }
      shuffled_topology =
          swap_elements(shuffled_topology, unsigned_axis, swap_idx);

      // std::cout << "shuffled_topology" << std::endl;
      // for (std::size_t t=0; t<shuffled_topology.size(); t++) {
      //  std::cout << "t: " << shuffled_topology.at(t) << std::endl;
      // }
      if (topologies.back() != shuffled_topology) {
        trans_axes.push_back(get_trans_axis(topologies.back(),
                                            shuffled_topology, first_non_one));
        topologies.push_back(shuffled_topology);
      }
    }

    // std::cout << "topologies.back()" << std::endl;
    // auto bac = topologies.back();
    // for (std::size_t t=0; t<bac.size(); t++) {
    //    std::cout << "t: " << bac.at(t) << std::endl;
    // }

    // std::cout << "out_topology_tmp" << std::endl;
    // for (std::size_t t=0; t<out_topology_tmp.size(); t++) {
    //   std::cout << "t: " << out_topology_tmp.at(t) << std::endl;
    // }
    //
    if (topologies.back() == out_topology_tmp) {
      return to_original_topologies(topologies, trans_axes);
    }

    // std::cout << "after check" << std::endl;

    try {
      auto mid_topology = get_mid_array(topologies.back(), out_topology_tmp);
      // std::cout << "mid_topology" << std::endl;
      // for (std::size_t t=0; t<mid_topology.size(); t++) {
      //   std::cout << "t: " << mid_topology.at(t) << std::endl;
      // }
      trans_axes.push_back(
          get_trans_axis(topologies.back(), mid_topology, first_non_one));
      topologies.push_back(mid_topology);
    } catch (std::runtime_error& e) {
    }

    if (topologies.back() == out_topology_tmp)
      return to_original_topologies(topologies, trans_axes);

    trans_axes.push_back(
        get_trans_axis(topologies.back(), out_topology_tmp, first_non_one));
    topologies.push_back(out_topology_tmp);
    return to_original_topologies(topologies, trans_axes);
  }

  std::vector<std::size_t> diff_non_ones =
      find_non_ones(in_topology_tmp, out_topology_tmp);
  auto last_axis  = axes_reversed.front();
  auto first_axis = axes_reversed.back();
  auto first_dim  = in_topology_tmp.at(last_axis);
  auto last_dim   = out_topology_tmp.at(first_axis);
  if (first_dim == 1) axes_reversed.erase(axes_reversed.begin());
  if (last_dim == 1 && !axes_reversed.empty()) axes_reversed.pop_back();
  for (const auto& axis : axes_reversed) {
    std::size_t swap_idx = 0;
    auto non_negative_axis =
        KokkosFFT::Impl::convert_negative_axis<int, DIM>(axis);
    std::size_t unsigned_axis = static_cast<std::size_t>(non_negative_axis);
    for (auto diff_idx : diff_non_ones) {
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
    }
  }
  if (topologies.back() == out_topology_tmp)
    return to_original_topologies(topologies, trans_axes);

  try {
    auto mid_topology = get_mid_array(topologies.back(), out_topology_tmp);
    trans_axes.push_back(
        get_trans_axis(topologies.back(), mid_topology, first_non_one));
    topologies.push_back(mid_topology);
  } catch (std::runtime_error& e) {
  }
  if (topologies.back() == out_topology_tmp)
    return to_original_topologies(topologies, trans_axes);

  trans_axes.push_back(
      get_trans_axis(topologies.back(), out_topology_tmp, first_non_one));
  topologies.push_back(out_topology_tmp);

  return to_original_topologies(topologies, trans_axes);
}

#endif
