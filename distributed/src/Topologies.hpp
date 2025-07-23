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

  KOKKOSFFT_THROW_IF(diff_non_ones.size() != 3,
                     "The total number of non-one elements either in Input and "
                     "output topologies must be three.");
  KOKKOSFFT_THROW_IF(
      diff_indices.size() != 3 && diffs.size() == 3,
      "Input and output topologies must differ exactly three positions.");

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

template <typename iType, std::size_t DIM = 1, std::size_t FFT_DIM = 1>
std::vector<std::array<iType, DIM>> get_shuffled_topologies(
    const std::array<iType, DIM>& in_topology,
    const std::array<iType, DIM>& out_topology,
    const std::array<int, FFT_DIM>& axes) {
  std::vector<iType> diff_non_ones = find_non_ones(in_topology, out_topology);
  KOKKOSFFT_THROW_IF(diff_non_ones.size() != 3,
                     "The total number of non-one elements either in Input and "
                     "output topologies must be three.");
  std::vector<std::array<iType, DIM>> topologies;
  topologies.push_back(in_topology);

  std::vector<int> axes_reversed;
  for (std::size_t i = 0; i < axes.size(); ++i) {
    axes_reversed.push_back(axes.at(i));
  }
  auto last_axis = axes.back();
  auto first_dim = in_topology.at(last_axis);
  if (first_dim == 1) axes_reversed.pop_back();

  std::reverse(axes_reversed.begin(), axes_reversed.end());
  std::array<iType, DIM> shuffled_topology = in_topology;
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
    topologies.push_back(shuffled_topology);
  }
  if (topologies.back() == out_topology) return topologies;

  try {
    auto mid_topology = get_mid_array(topologies.back(), out_topology);
    topologies.push_back(mid_topology);
  } catch (std::runtime_error& e) {
  }
  topologies.push_back(out_topology);

  return topologies;
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
/// \return A vector of all possible slab topologies that can be formed
/// from the input and output topologies, considering the FFT axes.
template <typename iType, std::size_t DIM = 1, std::size_t FFT_DIM = 1>
std::vector<std::array<std::size_t, DIM>> get_all_pencil_topologies(
    const std::array<std::size_t, DIM>& in_topology,
    const std::array<std::size_t, DIM>& out_topology,
    const std::array<iType, FFT_DIM>& axes) {
  static_assert(FFT_DIM >= 1 && FFT_DIM <= 3, "FFT_DIM must be in [1, 3]");
  static_assert(DIM >= 3 && DIM >= FFT_DIM, "DIM >= 3 and DIM >= FFT_DIM");

  bool is_pencil =
      is_pencil_topology(in_topology) && is_pencil_topology(out_topology);
  KOKKOSFFT_THROW_IF(!is_pencil,
                     "Input and output topologies must be pencil topologies.");

  std::vector<std::array<std::size_t, DIM>> topologies;
  topologies.push_back(in_topology);

  std::vector<std::size_t> axes_reversed;
  for (std::size_t i = 0; i < axes.size(); ++i) {
    auto non_negative_axis =
        KokkosFFT::Impl::convert_negative_axis<int, DIM>(axes.at(i));
    std::size_t unsigned_axis = static_cast<std::size_t>(non_negative_axis);
    axes_reversed.push_back(unsigned_axis);
  }

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
    auto dim                  = out_topology.at(unsigned_axis);
    if (dim != 1) is_output_ready = false;
  }

  if (is_input_ready || is_output_ready) {
    try {
      auto mid_topology = get_mid_array(topologies.back(), out_topology);
      topologies.push_back(mid_topology);
    } catch (std::runtime_error& e) {
    }
    if (topologies.back() != out_topology) topologies.push_back(out_topology);
    return topologies;
  }

  std::reverse(axes_reversed.begin(), axes_reversed.end());
  std::array<std::size_t, DIM> shuffled_topology = in_topology;
  if (in_topology == out_topology) {
    auto last_axis = axes_reversed.front();
    auto first_dim = in_topology.at(last_axis);
    if (first_dim == 1) axes_reversed.erase(axes_reversed.begin());
    for (const auto& axis : axes_reversed) {
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
      if (topologies.back() != shuffled_topology)
        topologies.push_back(shuffled_topology);
    }
    if (topologies.back() == out_topology) return topologies;

    try {
      auto mid_topology = get_mid_array(topologies.back(), out_topology);
      topologies.push_back(mid_topology);
    } catch (std::runtime_error& e) {
    }

    if (topologies.back() == out_topology) return topologies;
    topologies.push_back(out_topology);
    return topologies;
  }

  std::vector<std::size_t> diff_non_ones =
      find_non_ones(in_topology, out_topology);
  auto last_axis = axes_reversed.front();
  auto first_dim = in_topology.at(last_axis);
  if (first_dim == 1) axes_reversed.erase(axes_reversed.begin());

  for (const auto& axis : axes_reversed) {
    std::size_t swap_idx = 0;
    auto non_negative_axis =
        KokkosFFT::Impl::convert_negative_axis<int, DIM>(axis);
    std::size_t unsigned_axis = static_cast<std::size_t>(non_negative_axis);
    for (auto diff_idx : diff_non_ones) {
      // std::cout << "axis: " << axis << ", diff_idx: " << diff_idx <<
      // std::endl;
      if (shuffled_topology.at(diff_idx) == 1 && diff_idx != unsigned_axis) {
        swap_idx = diff_idx;
        break;
      }
    }
    shuffled_topology =
        swap_elements(shuffled_topology, unsigned_axis, swap_idx);

    // std::cout << "shuffled" << std::endl;
    // for (std::size_t i = 0; i < shuffled_topology.size(); i++) {
    //   std::cout << shuffled_topology[i] << ", " << std::endl;
    // }
    if (topologies.back() != shuffled_topology)
      topologies.push_back(shuffled_topology);
  }
  if (topologies.back() == out_topology) return topologies;

  try {
    auto mid_topology = get_mid_array(topologies.back(), out_topology);
    topologies.push_back(mid_topology);
  } catch (std::runtime_error& e) {
  }
  if (topologies.back() == out_topology) return topologies;
  topologies.push_back(out_topology);

  return topologies;
}

/*
// \brief Get all slab block info for a given input and output topology
///
/// \tparam InViewType
/// \tparam OutViewType
/// \tparam iType The index type used for the topology.
/// \tparam DIM The dimensionality of the topology.
/// \tparam FFT_DIM The dimensionality of the FFT axes.
///
/// \param in_topology The input topology.
/// \param out_topology The output topology.
/// \param axes The axes along which the FFT is performed.
/// \return A vector of all possible slab topologies that can be formed
/// from the input and output topologies, considering the FFT axes.
template <typename InViewType, typename OutViewType, typename iType, std::size_t
FFT_DIM> std::vector<BlockInfo<InViewType::rank()>> get_all_slab_block_info(
    const InViewType& in,
    const OutViewType& out,
    const std::array<std::size_t, InViewType::rank()>& in_topology,
    const std::array<std::size_t, OutViewType::rank()>& out_topology,
    const std::array<iType, FFT_DIM>& axes) {
  SlabBlockAnalyse<InViewType, OutViewType, iType, FFT_DIM>
    slab_block_analyses(in, out, in_topology, out_topology, axes);
  return slab_block_analyses.get_blocks();
}
*/

#endif
