#ifndef UTILS_HPP
#define UTILS_HPP

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>

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

  using ArrayType = Kokkos::Array<int, InViewType::rank()>;

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

  Kokkos::Array<int, InViewType::rank()> map_array =
      KokkosFFT::Impl::to_array(map);
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
