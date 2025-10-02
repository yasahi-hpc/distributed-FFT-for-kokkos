#ifndef KOKKOSFFT_DISTRIBUTED_MAPPING_HPP
#define KOKKOSFFT_DISTRIBUTED_MAPPING_HPP

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
/// [TO DO] Add a test case with src_map is not
/// in ascending order
/// \tparam LayoutType The layout type of the view
/// \tparam DIM        The dimensionality of the map
///
/// \param[in] src_map The axis map of the input view
/// \param[in] axis    The axis to be merged/split
template <typename LayoutType, typename iType, std::size_t DIM>
auto get_dst_map(const std::array<std::size_t, DIM>& src_map,
                 const std::vector<iType>& axes) {
  std::vector<std::size_t> map;
  map.reserve(DIM);
  if (std::is_same_v<LayoutType, Kokkos::LayoutRight>) {
    for (const auto src_idx : src_map) {
      if (!KokkosFFT::Impl::is_found(axes, src_idx)) {
        map.push_back(src_idx);
      }
    }
    for (auto axis : axes) {
      map.push_back(axis);
    }
  } else {
    // For layout Left, stack innermost axes first
    std::vector<iType> axes_reversed(axes);
    std::reverse(axes_reversed.begin(), axes_reversed.end());
    for (auto axis : axes_reversed) {
      map.push_back(axis);
    }

    // Then stack remaining axes
    for (const auto src_idx : src_map) {
      if (!KokkosFFT::Impl::is_found(axes_reversed, src_idx)) {
        map.push_back(src_idx);
      }
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

#endif
