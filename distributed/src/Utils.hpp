#ifndef UTILS_HPP
#define UTILS_HPP

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>

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

template <typename iType, std::size_t DIM = 1>
std::array<iType, DIM> swap_elements(const std::array<iType, DIM>& arr, int i,
                                     int j) {
  std::array<iType, DIM> result = arr;
  std::swap(result.at(i), result.at(j));
  return result;
}

template <std::size_t DIM>
std::size_t get_size(const std::array<std::size_t, DIM>& topology) {
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

  // Check if two topologies are two convertible pencils
  std::vector<iType> diff_indices = find_differences(in_topology, out_topology);
  KOKKOSFFT_THROW_IF(
      diff_indices.size() != 2,
      "Input and output topologies must differ exactly two positions.");

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
        if (src_idx[m_map[i]] >= m_out.extent(i)) in_bounds = false;
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
                    const OutViewType& out, KokkosFFT::axis_type<DIM> map) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "transpose: InViewType and OutViewType must have the same base floating "
      "point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");

  static_assert(InViewType::rank() == DIM,
                "transpose: Rank of View must be equal to Rank of "
                "transpose axes.");

  KOKKOSFFT_THROW_IF(!KokkosFFT::Impl::is_transpose_needed(map),
                     "transpose: transpose not necessary");

  auto in_extents  = KokkosFFT::Impl::extract_extents(in);
  auto out_extents = KokkosFFT::Impl::extract_extents(out);
  KOKKOSFFT_THROW_IF(get_mapped_extents(in_extents, map) != out_extents,
                     "transpose: input and output extents do not match after "
                     "applying the transpose map");
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
