#ifndef UTILS_HPP
#define UTILS_HPP

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "Types.hpp"

template <typename SizeType, typename IntType, std::size_t DIM,
          std::size_t Rank>
auto convert_negative_axes(const std::array<IntType, DIM>& axes) {
  static_assert(std::is_integral_v<SizeType>,
                "convert_negative_axes: SizeType must be an integer type.");
  static_assert(
      std::is_integral_v<IntType> && std::is_signed_v<IntType>,
      "convert_negative_axes: IntType must be a signed integer type.");
  std::array<SizeType, DIM> non_negative_axes = {};
  try {
    for (std::size_t i = 0; i < axes.size(); i++) {
      int axis = axes.at(i);
      auto non_negative_axis =
          KokkosFFT::Impl::convert_negative_axis<IntType, Rank>(axis);
      std::size_t unsigned_axis = static_cast<SizeType>(non_negative_axis);
      non_negative_axes.at(i)   = unsigned_axis;
    }
  } catch (std::runtime_error& e) {
    KOKKOSFFT_THROW_IF(false, "All axes must be in [-rank, rank-1]");
  }

  return non_negative_axes;
}

template <typename Layout, typename iType, std::size_t DIM, std::size_t FFT_DIM>
auto get_map_axes(const std::array<iType, FFT_DIM>& axes) {
  // Convert the input axes to be in the range of [0, rank-1]
  std::array<iType, FFT_DIM> non_negative_axes = {};
  if constexpr (std::is_signed_v<iType>) {
    for (std::size_t i = 0; i < FFT_DIM; i++) {
      non_negative_axes.at(i) =
          KokkosFFT::Impl::convert_negative_axis<iType, DIM>(axes.at(i));
    }
  } else {
    for (std::size_t i = 0; i < FFT_DIM; i++) {
      non_negative_axes.at(i) = axes.at(i);
    }
  }

  // how indices are map
  // For 5D View and axes are (2,3), map would be (0, 1, 4, 2, 3)
  constexpr iType rank = static_cast<iType>(DIM);
  std::vector<iType> map;
  map.reserve(rank);

  if (std::is_same_v<Layout, Kokkos::LayoutRight>) {
    // Stack axes not specified by axes (0, 1, 4)
    for (iType i = 0; i < rank; i++) {
      if (!KokkosFFT::Impl::is_found(non_negative_axes, i)) {
        map.push_back(i);
      }
    }

    // Stack axes on the map (For layout Right)
    // Then stack (2, 3) to have (0, 1, 4, 2, 3)
    for (auto axis : non_negative_axes) {
      map.push_back(axis);
    }
  } else {
    // For layout Left, stack innermost axes first
    std::reverse(non_negative_axes.begin(), non_negative_axes.end());
    for (auto axis : non_negative_axes) {
      map.push_back(axis);
    }

    // Then stack remaining axes
    for (iType i = 0; i < rank; i++) {
      if (!KokkosFFT::Impl::is_found(non_negative_axes, i)) {
        map.push_back(i);
      }
    }
  }

  using full_axis_type     = std::array<iType, rank>;
  full_axis_type array_map = {}, array_map_inv = {};
  std::copy_n(map.begin(), rank, array_map.begin());

  // Construct inverse map
  for (iType i = 0; i < rank; i++) {
    array_map_inv.at(i) = KokkosFFT::Impl::get_index(array_map, i);
  }

  return std::tuple<full_axis_type, full_axis_type>({array_map, array_map_inv});
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
std::vector<iType> find_non_ones(const std::array<iType, DIM>& a) {
  std::vector<iType> non_ones;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] != 1) {
      non_ones.push_back(a[i]);
    }
  }
  return non_ones;
}

template <typename iType, std::size_t DIM = 1>
std::vector<iType> find_ones(const std::array<iType, DIM>& a) {
  std::vector<iType> ones;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] == 1) {
      ones.push_back(i);
    }
  }
  return ones;
}

template <typename iType>
bool has_identical_non_ones(const std::vector<iType>& non_ones) {
  if (non_ones.size() == 2 &&
      std::set<iType>(non_ones.begin(), non_ones.end()).size() == 1) {
    return true;
  }
  return false;
}

template <typename iType, std::size_t DIM = 1>
std::array<iType, DIM> swap_elements(const std::array<iType, DIM>& arr, int i,
                                     int j) {
  std::array<iType, DIM> result = arr;
  std::swap(result.at(i), result.at(j));
  return result;
}

template <typename ContainerType>
std::size_t get_size(const ContainerType& topology) {
  return std::accumulate(topology.begin(), topology.end(), 1,
                         std::multiplies<std::size_t>());
}

template <typename iType, std::size_t DIM = 1>
auto merge_topology(const std::array<iType, DIM>& in_topology,
                    const std::array<iType, DIM>& out_topology) {
  auto in_size  = get_size(in_topology);
  auto out_size = get_size(out_topology);

  KOKKOSFFT_THROW_IF(in_size != out_size,
                     "Input and output topologies must have the same size.");

  if (in_size == 1) return in_topology;

  auto mismatched_extents =
      [](std::array<iType, DIM> in_topology,
         std::array<iType, DIM> out_topology) -> std::string {
    std::string message;
    message += "in_topology (";
    message += std::to_string(in_topology.at(0));
    for (std::size_t r = 1; r < in_topology.size(); r++) {
      message += ",";
      message += std::to_string(in_topology.at(r));
    }
    message += "), ";
    message += "out_topology (";
    message += std::to_string(out_topology.at(0));
    for (std::size_t r = 1; r < out_topology.size(); r++) {
      message += ",";
      message += std::to_string(out_topology.at(r));
    }
    message += ")";
    return message;
  };

  // Check if two topologies are two convertible pencils
  std::vector<iType> diff_indices = find_differences(in_topology, out_topology);
  KOKKOSFFT_THROW_IF(
      diff_indices.size() != 2,
      "Input and output topologies must differ exactly two positions: " +
          mismatched_extents(in_topology, out_topology));

  std::array<iType, DIM> topology = {};
  for (std::size_t i = 0; i < in_topology.size(); i++) {
    topology.at(i) = std::max(in_topology.at(i), out_topology.at(i));
  }
  return topology;
}

template <typename iType, std::size_t DIM = 1>
auto diff_toplogy(const std::array<iType, DIM>& in_topology,
                  const std::array<iType, DIM>& out_topology) {
  auto in_size  = get_size(in_topology);
  auto out_size = get_size(out_topology);

  if (in_size == 1 && out_size == 1) return iType(1);

  std::vector<iType> diff_indices = find_differences(in_topology, out_topology);
  KOKKOSFFT_THROW_IF(
      diff_indices.size() != 1,
      "Input and output topologies must differ exactly one positions.");
  iType diff_idx = diff_indices.at(0);

  return std::max(in_topology.at(diff_idx), out_topology.at(diff_idx));
}

template <typename iType, std::size_t DIM = 1>
auto get_trans_axis(const std::array<iType, DIM>& in_topology,
                    const std::array<iType, DIM>& out_topology,
                    iType first_non_one) {
  auto in_non_ones  = find_non_ones(in_topology);
  auto out_non_ones = find_non_ones(out_topology);
  KOKKOSFFT_THROW_IF(
      in_non_ones.size() != 2 || out_non_ones.size() != 2,
      "Input and output topologies must have exactly two non-one "
      "elements.");
  KOKKOSFFT_THROW_IF(has_identical_non_ones(in_non_ones) ||
                         has_identical_non_ones(out_non_ones),
                     "Input and output topologies must not have identical "
                     "non-one elements.");

  std::vector<iType> diff_indices = find_differences(in_topology, out_topology);
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
  iType trans_axis = exchange_non_one == first_non_one ? 0 : 1;
  return trans_axis;
}

template <typename iType, std::size_t DIM = 1,
          typename InLayoutType  = Kokkos::LayoutRight,
          typename OutLayoutType = Kokkos::LayoutRight>
auto get_trans_axis(const Topology<iType, DIM, InLayoutType>& in_topology,
                    const Topology<iType, DIM, OutLayoutType>& out_topology) {
  auto in_non_ones  = find_non_ones(in_topology.array());
  auto out_non_ones = find_non_ones(out_topology.array());
  KOKKOSFFT_THROW_IF(
      in_non_ones.size() != 2 || out_non_ones.size() != 2,
      "Input and output topologies must have exactly two non-one "
      "elements.");
  KOKKOSFFT_THROW_IF(has_identical_non_ones(in_non_ones) ||
                         has_identical_non_ones(out_non_ones),
                     "Input and output topologies must not have identical "
                     "non-one elements.");

  std::vector<iType> diff_indices =
      find_differences(in_topology.array(), out_topology.array());
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
  iType first_non_one = std::is_same_v<InLayoutType, Kokkos::LayoutRight>
                            ? in_non_ones.at(0)
                            : in_non_ones.at(1);
  iType trans_axis    = exchange_non_one == first_non_one ? 0 : 1;
  return trans_axis;
}

template <typename ContainerType>
auto get_max(const ContainerType& values) {
  return *std::max_element(values.begin(), values.end());
}

template <typename SizeType, std::size_t DIM>
auto to_vector(const std::array<SizeType, DIM>& arr) {
  std::vector<SizeType> vec(arr.begin(), arr.end());
  return vec;
}

template <typename ArraySizeType, typename VecSizeType, std::size_t DIM>
auto to_array(const std::vector<VecSizeType>& vec) {
  KOKKOSFFT_THROW_IF(
      vec.size() != DIM,
      "to_array: Vector size must match the specified dimension.");
  std::array<ArraySizeType, DIM> arr;
  std::copy(vec.begin(), vec.end(), arr.begin());
  return arr;
}

/// \brief Calculate the permuted extents based on the map
///
/// Example
/// View extents: (n0, n1, n2, n3)
/// map: (0, 2, 3, 1)
/// Next extents: (n0, n2, n3, n1)
///
/// \tparam DIM The number of dimensions of the extents.
///
/// \param[in] extents Extents of the View.
/// \param[in] map A map representing how the data is permuted
/// \return A extents of the permuted view
template <typename ContainerType, std::size_t DIM = 1>
auto get_mapped_extents(const std::array<std::size_t, DIM>& extents,
                        const ContainerType& map) {
  std::array<std::size_t, DIM> mapped_extents;

  for (std::size_t i = 0; i < extents.size(); i++) {
    std::size_t mapped_idx = map.at(i);
    mapped_extents.at(i)   = extents.at(mapped_idx);
  }

  return mapped_extents;
}

/// \brief Calculate the permuted axes based on the map
///
/// Example
/// Axes: (1, 3, 2) for (0, 1, 2, 3)
/// map: (0, 2, 3, 1)
/// Mapped Axes: (3, 2, 1) for (0, 2, 3, 1)
///
/// \tparam DIM The number of dimensions of the extents.
///
/// \param[in] extents Extents of the View.
/// \param[in] map A map representing how the data is permuted
/// \return A extents of the permuted view
template <typename ContainerType, std::size_t DIM = 1>
auto get_mapped_axes(const std::array<std::size_t, DIM>& axes,
                     const ContainerType& map) {
  std::array<std::size_t, DIM> mapped_axes;

  for (std::size_t i = 0; i < axes.size(); i++) {
    std::size_t axis  = axes.at(i);
    mapped_axes.at(i) = KokkosFFT::Impl::get_index(map, axis);
  }

  return mapped_axes;
}

/// \brief Get the larger extents. Larger one corresponds to
/// the extents to FFT library. This is a helper for vendor library
/// which supports 2D or 3D non-batched FFTs.
///
/// Example
/// in extents: (8, 7, 8)
/// out extents: (8, 7, 5)
///
/// \tparam DIM The number of dimensions of the extents.
///
/// \param[in] in_extents Extents of the global input View.
/// \param[in] out_extents Extents of the global output View.
/// \return A extents of the permuted view
template <std::size_t DIM>
auto get_fft_extents(const std::array<std::size_t, DIM>& in_extents,
                     const std::array<std::size_t, DIM>& out_extents,
                     const std::array<std::size_t, DIM>& axes) {
  std::array<std::size_t, DIM> fft_extents;

  for (std::size_t i = 0; i < fft_extents.size(); i++) {
    std::size_t axis  = axes.at(i);
    fft_extents.at(i) = std::max(in_extents.at(axis), out_extents.at(axis));
  }

  return fft_extents;
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t... Is>
void crop_or_pad_impl(const ExecutionSpace& exec_space, const InViewType& in,
                      const OutViewType& out, std::index_sequence<Is...>) {
  constexpr std::size_t rank = InViewType::rank();
  using extents_type         = std::array<std::size_t, rank>;

  extents_type extents;
  for (std::size_t i = 0; i < rank; i++) {
    extents.at(i) = std::min(in.extent(i), out.extent(i));
  }

  auto sub_in = Kokkos::subview(
      in, std::make_pair(std::size_t(0), std::get<Is>(extents))...);
  auto sub_out = Kokkos::subview(
      out, std::make_pair(std::size_t(0), std::get<Is>(extents))...);
  Kokkos::deep_copy(exec_space, sub_out, sub_in);
}

/// \brief Get padded extents from the extents in Fourier space
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
auto get_padded_extents(const std::array<std::size_t, DIM>& extents,
                        const std::array<std::size_t, DIM>& axes) {
  std::array<std::size_t, DIM> padded_extents = extents;
  auto last_axis                              = axes.back();
  padded_extents.at(last_axis) = padded_extents.at(last_axis) * 2;

  return padded_extents;
}

/// \brief Calculate the permuted axes based on the map
///
/// Example
/// Axes: (1, 3, 2) for (0, 1, 2, 3)
/// map: (0, 2, 3, 1)
/// Mapped Axes: (3, 2, 1) for (0, 2, 3, 1)
///
/// \tparam DIM The number of dimensions of the extents.
///
/// \param[in] extents Extents of the View.
/// \param[in] map A map representing how the data is permuted
/// \return A extents of the permuted view
template <typename Layout, typename iType, std::size_t Rank>
auto get_contiguous_axes(const std::vector<iType>& axes) {
  std::vector<iType> contiguous_axes(axes.size());

  if (std::is_same_v<Layout, Kokkos::LayoutLeft>) {
    for (std::size_t i = 0; i < axes.size(); ++i) {
      contiguous_axes[i] = static_cast<iType>(i);
    }
    // Reverse the axes to have the inner most axes first
    // (0, 1, 2) -> (2, 1, 0)
    std::reverse(contiguous_axes.begin(), contiguous_axes.end());
  } else {
    for (std::size_t i = 0; i < axes.size(); ++i) {
      int negative_axis = -int(axes.size()) + int(i);
      contiguous_axes[i] =
          KokkosFFT::Impl::convert_negative_axis<int, Rank>(negative_axis);
    }
  }
  return contiguous_axes;
}

/// \brief Transpose functor for out-of-place transpose operations.
/// This struct implements a functor that applies a transpose on a Kokkos view.
/// Before FFT, the input view is transposed into the order which is expected by
/// the FFT plan. After FFT, the input view is transposed back into the original
/// order.
///
/// \tparam ExecutionSpace The type of Kokkos execution space.
/// \tparam InViewType The input view type
/// \tparam OutViewType The output view type
/// \tparam iType The index type used for the view.
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          typename iType>
struct SafeTranspose {
 private:
  // Since MDRangePolicy is not available for 7D and 8D views, we need to
  // handle them separately. We can use a 6D MDRangePolicy and iterate over
  // the last two dimensions in the operator() function.
  static constexpr std::size_t m_rank_truncated =
      std::min(InViewType::rank(), std::size_t(6));

  using ArrayType = Kokkos::Array<std::size_t, InViewType::rank()>;

  /// \brief Retrieves the policy for the parallel execution.
  /// If the view is 1D, a Kokkos::RangePolicy is used. For higher dimensions up
  /// to 6D, a Kokkos::MDRangePolicy is used. For 7D and 8D views, we use 6D
  /// MDRangePolicy
  /// \param[in] space The Kokkos execution space used to launch the parallel
  /// reduction.
  /// \param[in] x The Kokkos view to be used for determining the policy.
  auto get_policy(const ExecutionSpace space, const InViewType& x) const {
    if constexpr (InViewType::rank() == 1) {
      using range_policy_type =
          Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<iType>>;
      return range_policy_type(space, 0, x.extent(0));
    } else {
      using iterate_type =
          Kokkos::Rank<m_rank_truncated, Kokkos::Iterate::Default,
                       Kokkos::Iterate::Default>;
      using mdrange_policy_type =
          Kokkos::MDRangePolicy<ExecutionSpace, iterate_type,
                                Kokkos::IndexType<iType>>;
      Kokkos::Array<std::size_t, m_rank_truncated> begins = {};
      Kokkos::Array<std::size_t, m_rank_truncated> ends   = {};
      for (std::size_t i = 0; i < m_rank_truncated; ++i) {
        ends[i] = x.extent(i);
      }
      return mdrange_policy_type(space, begins, ends);
    }
  }

 public:
  /// \brief Constructor for the Transpose functor.
  ///
  /// \param[in] in The input Kokkos view to be transposed.
  /// \param[out] out The output Kokkos view after transpose.
  /// \param[in] map The indices mapping of transpose
  /// \param[in] exec_space[in] The Kokkos execution space to be used (defaults
  /// to ExecutionSpace()).
  SafeTranspose(const InViewType& in, const OutViewType& out,
                const ArrayType& map,
                const ExecutionSpace exec_space = ExecutionSpace()) {
    Kokkos::parallel_for("KokkosFFT::transpose", get_policy(exec_space, in),
                         TransposeInternal(in, out, map));
  }

  /// \brief Helper functor to perform the transpose operation
  struct TransposeInternal {
    InViewType m_in;
    OutViewType m_out;
    ArrayType m_map;

    TransposeInternal(const InViewType& in, const OutViewType& out,
                      const ArrayType& map)
        : m_in(in), m_out(out), m_map(map) {}

    template <typename... IndicesType>
    KOKKOS_INLINE_FUNCTION void operator()(const IndicesType... indices) const {
      if constexpr (InViewType::rank() <= 6) {
        iType src_indices[InViewType::rank()] = {
            static_cast<iType>(indices)...};
        transpose_internal(src_indices,
                           std::make_index_sequence<InViewType::rank()>{});
      } else if constexpr (InViewType::rank() == 7) {
        for (iType i6 = 0; i6 < iType(m_in.extent(6)); i6++) {
          iType src_indices[InViewType::rank()] = {
              static_cast<iType>(indices)..., i6};
          transpose_internal(src_indices,
                             std::make_index_sequence<InViewType::rank()>{});
        }
      } else if constexpr (InViewType::rank() == 8) {
        for (iType i6 = 0; i6 < iType(m_in.extent(6)); i6++) {
          for (iType i7 = 0; i7 < iType(m_in.extent(7)); i7++) {
            iType src_indices[InViewType::rank()] = {
                static_cast<iType>(indices)..., i6, i7};
            transpose_internal(src_indices,
                               std::make_index_sequence<InViewType::rank()>{});
          }
        }
      }
    }

    template <std::size_t... Is>
    KOKKOS_INLINE_FUNCTION void transpose_internal(
        iType src_idx[], std::index_sequence<Is...>) const {
      // Bounds check
      bool in_bounds = true;
      for (std::size_t i = 0; i < InViewType::rank(); ++i) {
        if (src_idx[m_map[i]] >= iType(m_out.extent(i))) in_bounds = false;
      }

      if (in_bounds) {
        m_out(src_idx[m_map[Is]]...) = m_in(src_idx[Is]...);
      }
    }
  };
};

/// \brief Make the axis direction to the inner most direction
/// axis should be the range in [-(rank-1), rank-1], where
/// negative number is interpreted as rank + axis.
/// E.g. axis = -1 for rank 3 matrix is interpreted as axis = 2
///
/// E.g.
///      Layout Left
///      A (3, 4, 2) and axis = 1 -> A' (4, 3, 2)
///      B (2, 4, 3, 5) and axis = 2 -> B' (3, 2, 4, 5)
///      C (8, 6, 3) and axis = 0 -> C' (8, 6, 3)
///      D (7, 5) and axis = -1 -> D' (5, 7)
///
///      Layout Right
///      A (3, 4, 2) and axis = 1 -> A' (3, 2, 4)
///      B (2, 4, 3, 5) and axis = 2 -> B' (2, 4, 5, 3)
///      C (8, 6, 3) and axis = 0 -> C' (6, 3, 8)
///      D (5, 7) and axis = -1 -> D' (5, 7)
///
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam InViewType The input view type
/// \tparam OutViewType The output view type
/// \tparam DIM         The dimensionality of the map
///
/// \param[in] exec_space execution space instance
/// \param[in] in The input view
/// \param[out] out The output view
/// \param[in] map The axis map for transpose
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1>
void safe_transpose(const ExecutionSpace& exec_space, const InViewType& in,
                    const OutViewType& out,
                    const KokkosFFT::axis_type<DIM>& map) {
  static_assert(
      KokkosFFT::Impl::is_operatable_view_v<ExecutionSpace, InViewType>,
      "safe_transpose: In View value type must be float, double, "
      "Kokkos::Complex<float>, or Kokkos::Complex<double>. "
      "Layout must be either LayoutLeft or LayoutRight. "
      "ExecutionSpace must be able to access data in ViewType");

  static_assert(
      KokkosFFT::Impl::is_operatable_view_v<ExecutionSpace, OutViewType>,
      "safe_transpose: Out View value type must be float, double, "
      "Kokkos::Complex<float>, or Kokkos::Complex<double>. "
      "Layout must be either LayoutLeft or LayoutRight. "
      "ExecutionSpace must be able to access data in ViewType");

  static_assert(KokkosFFT::Impl::have_same_rank_v<InViewType, OutViewType>,
                "safe_transpose: In and Out View must have the same rank.");

  static_assert(
      KokkosFFT::Impl::have_same_base_floating_point_type_v<InViewType,
                                                            OutViewType>,
      "safe_transpose: In and Out View must have the same base floating point "
      "type.");

  static_assert(InViewType::rank() == DIM,
                "transpose: Rank of View must be equal to Rank of "
                "transpose axes.");

  if (!KokkosFFT::Impl::is_transpose_needed(map)) {
    // Just perform deep_copy (Layout change)
    crop_or_pad_impl(exec_space, in, out,
                     std::make_index_sequence<InViewType::rank()>{});
    return;
  }

  /*
  auto in_extents  = KokkosFFT::Impl::extract_extents(in);
  auto out_extents = KokkosFFT::Impl::extract_extents(out);

  auto mismatched_extents = [&]() -> std::string {
    std::string message;
    message += "in (";
    message += std::to_string(in_extents.at(0));
    for (std::size_t r = 1; r < in_extents.size(); r++) {
      message += ",";
      message += std::to_string(in_extents.at(r));
    }
    message += "), ";
    message += "out (";
    message += std::to_string(out_extents.at(0));
    for (std::size_t r = 1; r < out_extents.size(); r++) {
      message += ",";
      message += std::to_string(out_extents.at(r));
    }
    message += "), with map (";
    message += std::to_string(map.at(0));
    for (std::size_t i = 1; i < map.size(); i++) {
      message += ",";
      message += std::to_string(map.at(i));
    }
    message += ")";
    return message;
  };

  KOKKOSFFT_THROW_IF(get_mapped_extents(in_extents, map) != out_extents,
                     "transpose: input and output extents do not match after
                     " "applying the transpose map: " +
                         mismatched_extents());
  */

  std::array<std::size_t, DIM> non_negative_map =
      convert_negative_axes<std::size_t, int, DIM, InViewType::rank()>(map);
  Kokkos::Array<std::size_t, InViewType::rank()> map_array =
      KokkosFFT::Impl::to_array(non_negative_map);
  if ((in.span() >= std::size_t(std::numeric_limits<int>::max())) ||
      (out.span() >= std::size_t(std::numeric_limits<int>::max()))) {
    SafeTranspose<ExecutionSpace, InViewType, OutViewType, int64_t>(
        in, out, map_array, exec_space);
  } else {
    SafeTranspose<ExecutionSpace, InViewType, OutViewType, int>(
        in, out, map_array, exec_space);
  }
}

#endif
