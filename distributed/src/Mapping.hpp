#ifndef MAPPING_HPP
#define MAPPING_HPP

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "MPI_Helper.hpp"

/// \brief Get the mapping of the destination view from
/// src mapping. In the middle of the parallel FFTs,
/// the axis of the view can be changed which is stored in
/// the src_map. The dst_map is the mapping that is ready
/// for FFTs along the innermost direction.
///
/// E.g. Src Mapping (0, 1, 2) -> (0, 2, 1)
///      This corresponds to the mapping of
///      x -> x, y -> z, z -> y
///
///      Layout Left
///      axis == 0 -> (0, 2, 1)
///      axis == 1 -> (1, 0, 2)
///      axis == 2 -> (2, 0, 1)
///
///      Layout Right
///      axis == 0 -> (2, 1, 0)
///      axis == 1 -> (0, 2, 1)
///      axis == 2 -> (0, 1, 2)
///
/// \tparam LayoutType The layout type of the view
/// \tparam DIM        The dimensionality of the map
///
/// \param[in] src_map The axis map of the input view
/// \param[in] axis    The axis to be merged/split
template <typename LayoutType, std::size_t DIM>
auto get_dst_map(const std::array<std::size_t, DIM>& src_map,
                 std::size_t axis) {
  std::vector<std::size_t> map;
  map.reserve(DIM);
  if (std::is_same_v<LayoutType, Kokkos::LayoutRight>) {
    for (std::size_t i = 0; i < DIM; ++i) {
      if (src_map.at(i) != axis) map.push_back(src_map.at(i));
    }
    map.push_back(axis);
  } else {
    map.push_back(axis);
    for (std::size_t i = 0; i < DIM; ++i) {
      if (src_map.at(i) != axis) map.push_back(src_map.at(i));
    }
  }

  using full_axis_type   = std::array<std::size_t, DIM>;
  full_axis_type dst_map = {};
  std::copy_n(map.begin(), DIM, dst_map.begin());

  return dst_map;
}

/// \brief Get the mapping of the destination view from
/// src mapping. In the middle of the parallel FFTs,
/// the axis of the view can be changed which is stored in
/// the src_map. The dst_map is the mapping that is ready
/// for FFTs along the innermost direction.
///
/// E.g. Src Mapping (0, 2, 1)
///      This corresponds to the mapping of
///      x -> x, y -> z, z -> y
///
///      Layout Left
///      axis == 0 -> (0, 2, 1) (i0 <- i0, i1 <- i1, i2 <- i2)
///      axis == 1 -> (1, 0, 2) (i0 <- i2, i1 <- i0, i2 <- i1)
///      axis == 2 -> (2, 0, 1) (i0 <- i1, i1 <- i0, i2 <- i2)
///
///      Layout Right
///      axis == 0 -> (2, 1, 0) (i0 <- i1, i1 <- i2, i2 <- i0)
///      axis == 1 -> (0, 2, 1) (i0 <- i0, i1 <- i1, i2 <- i2)
///      axis == 2 -> (0, 1, 2) (i0 <- i0, i1 <- i2, i2 <- i1)
///
/// \tparam LayoutType The layout type of the view
/// \tparam DIM        The dimensionality of the map
///
/// \param[in] src_map The axis map of the input view
/// \param[in] axis    The axis to be merged/split
template <typename LayoutType, std::size_t DIM>
auto get_src_dst_map(const std::array<std::size_t, DIM>& src_map,
                     std::size_t axis) {
  auto dst_map = get_dst_map<LayoutType>(src_map, axis);
  std::vector<std::size_t> map;
  map.reserve(DIM);

  for (std::size_t i = 0; i < DIM; ++i) {
    map.push_back(KokkosFFT::Impl::get_index(src_map, dst_map.at(i)));
  }

  using full_axis_type       = std::array<std::size_t, DIM>;
  full_axis_type src_dst_map = {};
  std::copy_n(map.begin(), DIM, src_dst_map.begin());

  return src_dst_map;
}

/*
template <std::size_t DIM>
std::size_t get_size(const std::array<std::size_t, DIM>& topology) {
  return std::accumulate(topology.begin(), topology.end(), 1,
                         std::multiplies<std::size_t>());
}
*/

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

template <typename ArrayType>
int countNonOneComponents(const ArrayType& arr) {
  return std::count_if(arr.begin(), arr.end(),
                       [](int val) { return val != 1; });
}

template <typename iType, std::size_t DIM = 1>
std::vector<iType> find_differences(const std::array<iType, DIM>& a,
                                    const std::array<iType, DIM>& b) {
  std::vector<iType> diffs;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      diffs.push_back(i);
    }
  }
  return diffs;
}

template <typename iType, std::size_t DIM = 1>
std::set<iType> diff_sets(const std::array<iType, DIM>& a,
                          const std::array<iType, DIM>& b) {
  std::vector<iType> diffs;
  for (std::size_t i = 0; i < a.size(); ++i) {
    diffs.push_back(a.at(i));
    diffs.push_back(b.at(i));
  }
  return std::set<iType>(diffs.begin(), diffs.end());
}

template <typename iType, std::size_t DIM = 1>
std::vector<iType> find_non_ones(const std::array<iType, DIM>& a,
                                 const std::array<iType, DIM>& b) {
  std::vector<iType> non_ones;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] != 1 || b[i] != 1) {
      non_ones.push_back(i);
    }
  }
  return non_ones;
}

template <typename iType, std::size_t DIM = 1>
std::array<iType, DIM> swap_elements(const std::array<iType, DIM>& arr, int i,
                                     int j) {
  std::array<iType, DIM> result = arr;
  std::swap(result.at(i), result.at(j));
  return result;
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

  iType idx_one_in  = KokkosFFT::Impl::get_index(in, iType(1));
  iType idx_one_out = KokkosFFT::Impl::get_index(out, iType(1));

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
    for (auto diff_idx : diff_non_ones) {
      if (shuffled_topology.at(diff_idx) == 1 && diff_idx != axis) {
        swap_idx = diff_idx;
        break;
      }
    }
    shuffled_topology = swap_elements(shuffled_topology, axis, swap_idx);
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

template <typename iType, std::size_t DIM = 1, std::size_t FFT_DIM = 1>
std::vector<std::array<iType, DIM>> get_topologies(
    const std::array<iType, DIM>& in_topology,
    const std::array<iType, DIM>& out_topology,
    const std::array<iType, FFT_DIM>& axes, bool is_real_transform = false) {
  // Firstly, check that the first transform is ready or not
  if (is_real_transform) {
    auto last_axis = axes.back();
    auto first_dim = in_topology.at(last_axis);

    // If the first dimension is distributed,
    // we need to transpose to make the first transform
    if (first_dim > 1) {
      auto shuffled_topologies =
          get_shuffled_topologies(in_topology, out_topology, axes);
      return shuffled_topologies;
    }
  }

  std::vector<std::array<iType, DIM>> topologies;
  topologies.push_back(in_topology);
  auto mid_topology = get_mid_array(in_topology, out_topology);
  topologies.push_back(mid_topology);
  topologies.push_back(out_topology);

  return topologies;
}

#endif
