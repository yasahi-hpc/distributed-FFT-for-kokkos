#ifndef MAPPING_HPP
#define MAPPING_HPP

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>

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

template <std::size_t DIM>
std::size_t get_size(const std::array<std::size_t, DIM>& topology) {
  return std::accumulate(topology.begin(), topology.end(), 1,
                         std::multiplies<std::size_t>());
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

#endif
