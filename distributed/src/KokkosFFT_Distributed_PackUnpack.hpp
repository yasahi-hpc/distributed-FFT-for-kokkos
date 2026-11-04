#ifndef KOKKOSFFT_DISTRIBUTED_PACKUNPACK_HPP
#define KOKKOSFFT_DISTRIBUTED_PACKUNPACK_HPP

#include <numeric>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <KokkosFFT.hpp>

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

template <typename T>
struct OutOfBounds {
  static_assert(std::is_signed_v<T>, "OutOfBounds: T must be a signed integer");
  static constexpr auto value = T(-1);
};

template <typename T>
inline constexpr auto OutOfBounds_v = OutOfBounds<T>::value;

/// \brief Helper function to compute the start index and length of a bin
/// when dividing a range [0, N) into `nbins` bins. The `ibin`-th bin
/// will contain the indices from `start` to `start + length - 1`.
/// The bins are distributed as evenly as possible, with the first `remainder`
/// bins containing one extra element if `N` is not perfectly divisible by
/// `nbins`.
/// For example, if N=10 and nbins=3, the bins will be:
/// [0, 4), [4, 7), [7, 10)
/// which corresponds to (start=0, length=4), (start=4, length=3),
/// (start=7, length=3).
/// Note, `nbins` corresponds to the number of processes in the communicator and
/// `ibin` corresponds to the rank of the process.
///
/// \tparam iType Integer type for indices
/// \param[in] N Total number of elements
/// \param[in] nbins Number of bins to divide into
/// \param[in] ibin Index of the bin to compute (0-based)
/// \return A pair containing the start index and length of the `ibin`-th bin
template <typename iType>
KOKKOS_INLINE_FUNCTION auto bin_mapping(iType N, iType nbins, iType ibin) {
  iType base_size = N / nbins, remainder = N % nbins;
  iType length = base_size + (ibin < remainder ? 1 : 0);
  iType start  = ibin * base_size + Kokkos::min(ibin, remainder);
  return Kokkos::pair<iType, iType>{start, length};
}

/// \brief If `axis` is not equal to `merged_axis`, it returns `idx` unchanged.
/// If `axis` is equal to `merged_axis`, it merges the index `idx` with the
/// `start` and `extent` of a bin along the specified `axis`. If `idx` is
/// outside the range of the bin, it returns -1. Otherwise, it returns the
/// global index by adding `start` to `idx`.
/// `start` and `extent` are obtained from `bin_mapping` function defined above
///
/// \tparam iType Integer type for indices
/// \param[in] idx Local index to be merged
/// \param[in] start Start index of the bin
/// \param[in] extent Length of the bin
/// \param[in] axis Axis along which the bin is defined
/// \param[in] merged_axis Axis that is being merged
/// \return The global index if `idx` is within the bin, otherwise -1
template <typename iType>
KOKKOS_INLINE_FUNCTION iType merge_indices(iType idx, iType start, iType extent,
                                           std::size_t axis,
                                           std::size_t merged_axis) {
  static_assert(std::is_signed_v<iType>,
                "merge_indices: iType must be a signed integer");
  if (axis == merged_axis) {
    return idx >= extent ? OutOfBounds_v<iType> : idx + start;
  } else {
    return idx;
  }
}

/// \brief Index arithmetic of Pack. It maps an element of the (N+1)D
/// destination (send buffer) to the process its chunk is sent to, and to the
/// element of the ND source view it holds. It is shared by Pack and by the
/// kernels that write the chunks into the peers' buffers directly.
///
/// \tparam LayoutType Layout of the destination view
/// \tparam iType Integer type for indices
/// \tparam SrcRank Rank of the source view
template <typename LayoutType, typename iType, std::size_t SrcRank>
struct PackIndexMap {
  static_assert(std::is_same_v<LayoutType, Kokkos::LayoutLeft> ||
                    std::is_same_v<LayoutType, Kokkos::LayoutRight>,
                "PackIndexMap: We only accept LayoutLeft or "
                "LayoutRight for the Dst View.");
  using ShapeType = Kokkos::Array<std::size_t, SrcRank>;

  std::size_t m_axis, m_merged_size, m_nprocs = 1;
  ShapeType m_map;
  ShapeType m_src_extents;

  /// \brief Constructor for the PackIndexMap.
  ///
  /// \tparam SrcViewType Kokkos View type of the source data
  /// \tparam DstViewType Kokkos View type of the destination data
  /// \param[in] src The source view
  /// \param[in] dst The destination view (or a view with its extents)
  /// \param[in] map The mapping of source view dimensions
  /// \param[in] axis The axis to be split
  /// \param[in] merged_size The extent of the dimension to be split
  template <typename SrcViewType, typename DstViewType>
  PackIndexMap(const SrcViewType& src, const DstViewType& dst,
               const ShapeType& map, const std::size_t axis,
               const std::size_t merged_size)
      : m_axis(axis), m_merged_size(merged_size), m_map(map) {
    static_assert(
        SrcViewType::rank() == SrcRank && DstViewType::rank() == SrcRank + 1,
        "PackIndexMap: SrcView rank must be SrcRank and DstView "
        "rank must be SrcRank + 1");
    m_nprocs = std::is_same_v<LayoutType, Kokkos::LayoutRight>
                   ? dst.extent(0)
                   : dst.extent(SrcRank);
    for (std::size_t i = 0; i < SrcRank; ++i) {
      m_src_extents[i] = src.extent(i);
    }
  }

  /// \brief The process (index of the chunk) of a destination element
  /// \param[in] dst_idx The indices of the destination element
  /// \return The index of the chunk along the dimension of processes
  KOKKOS_INLINE_FUNCTION iType peer(const iType dst_idx[]) const {
    if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutRight>) {
      return dst_idx[0];
    } else {
      return dst_idx[SrcRank];
    }
  }

  /// \brief Get the indices of the source element corresponding to the given
  /// destination indices.
  /// This function computes the source indices by merging the destination
  /// indices with the start and extent of the bin along the specified axis.
  ///
  /// \param[in] dst_idx The indices of the destination element
  /// \param[out] src_idx The indices of the source element
  /// \return false if any of the source indices is out of bounds (the
  /// destination element is padding), true otherwise
  KOKKOS_INLINE_FUNCTION bool src_indices(const iType dst_idx[],
                                          iType src_idx[]) const {
    constexpr std::size_t offset =
        std::is_same_v<LayoutType, Kokkos::LayoutRight> ? 1 : 0;
    const auto [start, extent] =
        bin_mapping(iType(m_merged_size), iType(m_nprocs), peer(dst_idx));
    iType merged_indices[SrcRank];
    for (std::size_t i = 0; i < SrcRank; ++i) {
      merged_indices[i] =
          merge_indices(dst_idx[i + offset], start, extent, i, m_axis);
    }
    for (std::size_t i = 0; i < SrcRank; ++i) {
      src_idx[i] = merged_indices[m_map[i]];
      if (src_idx[i] >= iType(m_src_extents[i]) ||
          src_idx[i] == OutOfBounds_v<iType>) {
        return false;
      }
    }
    return true;
  }
};

/// \brief Pack data from ND source view to N+1D destination view by splitting
/// along a specified axis. The destination view has an additional dimension
/// representing the number of processes (nprocs). The additional dimension is
/// the outermost dimension, corresponding to the dimension to be transposed.
/// The destination data may be permuted based on the provided mapping.
///
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam SrcViewType Kokkos View type of the source data
/// \tparam DstViewType Kokkos View type of the destination data
/// \tparam iType Integer type for indices
template <typename ExecutionSpace, typename SrcViewType, typename DstViewType,
          typename iType>
struct Pack {
 private:
  // Since MDRangePolicy is not available for 7D and 8D views, we need to
  // handle them separately. We can use a 6D MDRangePolicy and iterate over
  // the last two dimensions in the operator() function.
  static constexpr std::size_t m_rank_truncated =
      std::min(DstViewType::rank(), std::size_t(6));

  using ShapeType = Kokkos::Array<std::size_t, SrcViewType::rank()>;

  /// \brief Retrieves the policy for the parallel execution.
  /// If the view is 1D, a Kokkos::RangePolicy is used. For higher dimensions up
  /// to 6D, a Kokkos::MDRangePolicy is used. For 7D and 8D views, we use 6D
  /// MDRangePolicy
  /// \param[in] space The Kokkos execution space used to launch the parallel
  /// reduction.
  /// \param[in] x The Kokkos view to be used for determining the policy.
  auto get_policy(const ExecutionSpace space, const DstViewType& x) const {
    if constexpr (DstViewType::rank() == 1) {
      using range_policy_type =
          Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<iType>>;
      return range_policy_type(space, 0, x.extent(0));
    } else {
      using LayoutType = typename DstViewType::array_layout;
      constexpr Kokkos::Iterate outer_iteration_pattern =
          KokkosFFT::Impl::layout_iterate_type_selector<
              LayoutType>::outer_iteration_pattern;
      constexpr Kokkos::Iterate inner_iteration_pattern =
          KokkosFFT::Impl::layout_iterate_type_selector<
              LayoutType>::inner_iteration_pattern;
      using iterate_type =
          Kokkos::Rank<m_rank_truncated, outer_iteration_pattern,
                       inner_iteration_pattern>;
      using mdrange_policy_type =
          Kokkos::MDRangePolicy<ExecutionSpace, iterate_type,
                                Kokkos::IndexType<iType>>;
      Kokkos::Array<std::size_t, m_rank_truncated> begins{}, ends{};
      for (std::size_t i = 0; i < m_rank_truncated; ++i) {
        ends[i] = x.extent(i);
      }
      return mdrange_policy_type(space, begins, ends);
    }
  }

 public:
  /// \brief Constructor for the Pack functor.
  ///
  /// \param[in] src The input Kokkos view to be packed
  /// \param[out] dst The output Kokkos view to be packed
  /// \param[in] axis The axis to be split
  /// \param[in] merged_size The extent of the dimension to be split
  /// \param[in] exec_space The Kokkos execution space to be used (defaults to
  /// ExecutionSpace()).
  Pack(const SrcViewType& src, const DstViewType& dst, const ShapeType& map,
       const std::size_t axis, const std::size_t merged_size,
       const ExecutionSpace exec_space = ExecutionSpace()) {
    Kokkos::parallel_for("KokkosFFT::Distributed::Pack",
                         get_policy(exec_space, dst),
                         PackInternal(src, dst, map, axis, merged_size));
  }

  struct PackInternal {
    using LayoutType   = typename DstViewType::array_layout;
    using ValueType    = typename SrcViewType::non_const_value_type;
    using IndexMapType = PackIndexMap<LayoutType, iType, SrcViewType::rank()>;

    SrcViewType m_src;
    DstViewType m_dst;
    IndexMapType m_index_map;

    PackInternal(const SrcViewType& src, const DstViewType& dst,
                 const ShapeType& map, const std::size_t axis,
                 const std::size_t merged_size)
        : m_src(src),
          m_dst(dst),
          m_index_map(src, dst, map, axis, merged_size) {}

    template <typename... IndicesType>
    KOKKOS_INLINE_FUNCTION void operator()(const IndicesType... indices) const {
      if constexpr (DstViewType::rank() <= 6) {
        iType dst_indices[DstViewType::rank()] = {
            static_cast<iType>(indices)...};
        m_dst(indices...) = get_src_value(
            dst_indices, std::make_index_sequence<DstViewType::rank() - 1>{});
      } else if constexpr (DstViewType::rank() == 7) {
        for (iType i6 = 0; i6 < iType(m_dst.extent(6)); i6++) {
          iType dst_indices[DstViewType::rank()] = {
              static_cast<iType>(indices)..., i6};
          m_dst(indices..., i6) = get_src_value(
              dst_indices, std::make_index_sequence<DstViewType::rank() - 1>{});
        }
      } else if constexpr (DstViewType::rank() == 8) {
        for (iType i6 = 0; i6 < iType(m_dst.extent(6)); i6++) {
          for (iType i7 = 0; i7 < iType(m_dst.extent(7)); i7++) {
            iType dst_indices[DstViewType::rank()] = {
                static_cast<iType>(indices)..., i6, i7};
            m_dst(indices..., i6, i7) = get_src_value(
                dst_indices,
                std::make_index_sequence<DstViewType::rank() - 1>{});
          }
        }
      }
    }

    /// \brief Get the value from the source view corresponding to the given
    /// destination indices (PackIndexMap::src_indices).
    /// If any of the computed source indices are out of bounds, it returns 0.
    ///
    template <std::size_t... Is>
    KOKKOS_INLINE_FUNCTION ValueType
    get_src_value(iType dst_idx[], std::index_sequence<Is...>) const {
      iType src_indices[DstViewType::rank() - 1];
      if (!m_index_map.src_indices(dst_idx, src_indices)) {
        // Quick return for out-of-bounds
        return ValueType(0);
      }
      return m_src(src_indices[Is]...);
    }
  };
};

/// \brief Unpack data from N+1D source view to ND destination view by merging
/// along a specified axis. The source view has an additional dimension
/// representing the number of processes (nprocs). The additional dimension is
/// the outermost dimension, corresponding to the dimension to be transposed.
/// The destination data may be permuted based on the provided mapping.
template <typename ExecutionSpace, typename SrcViewType, typename DstViewType,
          typename iType>
struct Unpack {
 private:
  // Since MDRangePolicy is not available for 7D and 8D views, we need to
  // handle them separately. We can use a 6D MDRangePolicy and iterate over
  // the last two dimensions in the operator() function.
  static constexpr std::size_t m_rank_truncated =
      std::min(SrcViewType::rank(), std::size_t(6));
  using ShapeType = Kokkos::Array<std::size_t, DstViewType::rank()>;

  /// \brief Retrieves the policy for the parallel execution.
  /// If the view is 1D, a Kokkos::RangePolicy is used. For higher dimensions up
  /// to 6D, a Kokkos::MDRangePolicy is used. For 7D and 8D views, we use 6D
  /// MDRangePolicy
  /// \param[in] space The Kokkos execution space used to launch the parallel
  /// reduction.
  /// \param[in] x The Kokkos view to be used for determining the policy.
  auto get_policy(const ExecutionSpace space, const SrcViewType& x) const {
    if constexpr (SrcViewType::rank() == 1) {
      using range_policy_type =
          Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<iType>>;
      return range_policy_type(space, 0, x.extent(0));
    } else {
      using LayoutType = typename SrcViewType::array_layout;
      constexpr Kokkos::Iterate outer_iteration_pattern =
          KokkosFFT::Impl::layout_iterate_type_selector<
              LayoutType>::outer_iteration_pattern;
      constexpr Kokkos::Iterate inner_iteration_pattern =
          KokkosFFT::Impl::layout_iterate_type_selector<
              LayoutType>::inner_iteration_pattern;
      using iterate_type =
          Kokkos::Rank<m_rank_truncated, outer_iteration_pattern,
                       inner_iteration_pattern>;
      using mdrange_policy_type =
          Kokkos::MDRangePolicy<ExecutionSpace, iterate_type,
                                Kokkos::IndexType<iType>>;
      Kokkos::Array<std::size_t, m_rank_truncated> begins{}, ends{};
      for (std::size_t i = 0; i < m_rank_truncated; ++i) {
        ends[i] = x.extent(i);
      }
      return mdrange_policy_type(space, begins, ends);
    }
  }

 public:
  /// \brief Constructor for the Unpack functor.
  ///
  /// \param[in] src The input Kokkos view to be unpacked
  /// \param[out] dst The output Kokkos view to be unpacked
  /// \param[in] axis The axis to be merged
  /// \param[in] merged_size The extent of the dimension to be merged
  /// \param[in] exec_space The Kokkos execution space to be used (defaults to
  /// ExecutionSpace()).
  Unpack(const SrcViewType& src, const DstViewType& dst, const ShapeType& map,
         const std::size_t axis, const std::size_t merged_size,
         const ExecutionSpace exec_space = ExecutionSpace()) {
    Kokkos::parallel_for("KokkosFFT::Distributed::Unpack",
                         get_policy(exec_space, src),
                         UnpackInternal(src, dst, map, axis, merged_size));
  }

  struct UnpackInternal {
    using LayoutType = typename SrcViewType::array_layout;
    using ValueType  = typename DstViewType::non_const_value_type;

    static_assert(std::is_same_v<LayoutType, Kokkos::LayoutLeft> ||
                      std::is_same_v<LayoutType, Kokkos::LayoutRight>,
                  "UnpackInternal: We only accept LayoutLeft or "
                  "LayoutRight for the Src View.");
    SrcViewType m_src;
    DstViewType m_dst;
    std::size_t m_axis, m_merged_size, m_nprocs = 1;
    ShapeType m_map;
    ShapeType m_dst_extents;
    ShapeType m_src_extents;

    UnpackInternal(const SrcViewType& src, const DstViewType& dst,
                   const ShapeType& map, const std::size_t axis,
                   const std::size_t merged_size)
        : m_src(src),
          m_dst(dst),
          m_axis(axis),
          m_merged_size(merged_size),
          m_map(map) {
      for (std::size_t i = 0; i < DstViewType::rank(); ++i) {
        m_src_extents[i] = std::is_same_v<LayoutType, Kokkos::LayoutRight>
                               ? src.extent(i + 1)
                               : src.extent(i);
      }
      m_nprocs = std::is_same_v<LayoutType, Kokkos::LayoutRight>
                     ? src.extent(0)
                     : src.extent(SrcViewType::rank() - 1);
      for (std::size_t i = 0; i < DstViewType::rank(); ++i) {
        m_dst_extents[i] = dst.extent(i);
      }
    }

    template <typename... IndicesType>
    KOKKOS_INLINE_FUNCTION void operator()(const IndicesType... indices) const {
      if constexpr (SrcViewType::rank() <= 6) {
        iType src_indices[SrcViewType::rank()] = {
            static_cast<iType>(indices)...};
        auto src_value = m_src(indices...);
        set_dst_value(src_indices, src_value,
                      std::make_index_sequence<SrcViewType::rank() - 1>{});
      } else if constexpr (SrcViewType::rank() == 7) {
        for (iType i6 = 0; i6 < iType(m_dst.extent(6)); i6++) {
          iType src_indices[SrcViewType::rank()] = {
              static_cast<iType>(indices)..., i6};
          auto src_value = m_src(indices..., i6);
          set_dst_value(src_indices, src_value,
                        std::make_index_sequence<SrcViewType::rank() - 1>{});
        }
      } else if constexpr (SrcViewType::rank() == 8) {
        for (iType i6 = 0; i6 < iType(m_dst.extent(6)); i6++) {
          for (iType i7 = 0; i7 < iType(m_dst.extent(7)); i7++) {
            iType src_indices[SrcViewType::rank()] = {
                static_cast<iType>(indices)..., i6, i7};
            auto src_value = m_src(indices..., i6, i7);
            set_dst_value(src_indices, src_value,
                          std::make_index_sequence<SrcViewType::rank() - 1>{});
          }
        }
      }
    }

    template <std::size_t... Is>
    KOKKOS_INLINE_FUNCTION void set_dst_value(
        iType src_idx[], ValueType src_value,
        std::index_sequence<Is...>) const {
      if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutRight>) {
        const iType p = src_idx[0];
        const auto [start, extent] =
            bin_mapping(iType(m_merged_size), iType(m_nprocs), p);
        iType dst_indices[SrcViewType::rank() - 1] = {
            merge_indices(src_idx[Is + 1], start, extent, Is, m_axis)...};
        for (std::size_t i = 0; i < SrcViewType::rank() - 1; ++i) {
          if (dst_indices[m_map[i]] >= iType(m_dst_extents[i]) ||
              dst_indices[m_map[i]] == OutOfBounds_v<iType>) {
            // Quick return for out-of-bounds
            return;
          }
        }
        m_dst(dst_indices[m_map[Is]]...) = src_value;
      } else {
        const iType p = src_idx[SrcViewType::rank() - 1];
        const auto [start, extent] =
            bin_mapping(iType(m_merged_size), iType(m_nprocs), p);
        iType dst_indices[SrcViewType::rank() - 1] = {
            merge_indices(src_idx[Is], start, extent, Is, m_axis)...};
        for (std::size_t i = 0; i < SrcViewType::rank() - 1; ++i) {
          if (dst_indices[m_map[i]] >= iType(m_dst_extents[i]) ||
              dst_indices[m_map[i]] == OutOfBounds_v<iType>) {
            // Quick return for out-of-bounds
            return;
          }
        }
        m_dst(dst_indices[m_map[Is]]...) = src_value;
      }
    }
  };
};

/// \brief Pack data from ND source view to (N+1)D destination view by splitting
/// along a specified axis. The destination view has an additional dimension
/// representing the number of processes (nprocs). The additional dimension is
/// the outermost dimension, corresponding to the dimension to be transposed.
/// The destination data may be permuted based on the provided mapping.
///
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam SrcViewType Kokkos View type of the source data
/// \tparam DstViewType Kokkos View type of the destination data
/// \tparam DIM Dimension of the mapping array
///
/// \param[in] exec_space The Kokkos execution space to be used
/// \param[in] src The input Kokkos view to be packed
/// \param[out] dst The output Kokkos view to be packed
/// \param[in] src_map The mapping of source view dimensions to destination view
/// dimensions
/// \param[in] axis The axis to be split
template <typename ExecutionSpace, typename SrcViewType, typename DstViewType,
          std::size_t DIM>
void pack(const ExecutionSpace& exec_space, const SrcViewType& src,
          const DstViewType& dst, std::array<std::size_t, DIM> src_map,
          std::size_t axis) {
  static_assert(SrcViewType::rank() >= 2,
                "pack: SrcView rank must be larger than or equal to 2");
  static_assert(DstViewType::rank() == SrcViewType::rank() + 1,
                "pack: DstView::rank must be SrcView::rank + 1");
  Kokkos::Array<std::size_t, DIM> src_array =
      KokkosFFT::Impl::to_array(src_map);
  std::size_t merged_size =
      src.extent(KokkosFFT::Impl::get_index(src_map, axis));
  if (src.span() >= std::size_t(std::numeric_limits<int>::max()) ||
      dst.span() >= std::size_t(std::numeric_limits<int>::max())) {
    Pack<ExecutionSpace, SrcViewType, DstViewType, int64_t>(
        src, dst, src_array, axis, merged_size, exec_space);
  } else {
    Pack<ExecutionSpace, SrcViewType, DstViewType, int>(
        src, dst, src_array, axis, merged_size, exec_space);
  }
}

/// \brief Unpack data from (N+1)D source view to ND destination view by merging
/// along a specified axis. The source view has an additional dimension
/// representing the number of processes (nprocs). The additional dimension is
/// the outermost dimension, corresponding to the dimension to be transposed.
/// The destination data may be permuted based on the provided mapping.
///
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam SrcViewType Kokkos View type of the source data
/// \tparam DstViewType Kokkos View type of the destination data
/// \tparam DIM Dimension of the mapping array
/// \param[in] exec_space The Kokkos execution space to be used
/// \param[in] src The input Kokkos view to be unpacked
/// \param[out] dst The output Kokkos view to be unpacked
/// \param[in] dst_map The mapping of destination view dimensions to source view
/// dimensions
/// \param[in] axis The axis to be merged
template <typename ExecutionSpace, typename SrcViewType, typename DstViewType,
          std::size_t DIM>
void unpack(const ExecutionSpace& exec_space, const SrcViewType& src,
            const DstViewType& dst, std::array<std::size_t, DIM> dst_map,
            std::size_t axis) {
  static_assert(DstViewType::rank() >= 2,
                "unpack: DstView rank must be larger than or equal to 2");
  static_assert(SrcViewType::rank() == DstViewType::rank() + 1,
                "SrcView::rank must be DstView::rank + 1");
  Kokkos::Array<std::size_t, DIM> dst_array =
      KokkosFFT::Impl::to_array(dst_map);
  std::size_t merged_size =
      dst.extent(KokkosFFT::Impl::get_index(dst_map, axis));
  if (dst.span() >= std::size_t(std::numeric_limits<int>::max()) ||
      src.span() >= std::size_t(std::numeric_limits<int>::max())) {
    Unpack<ExecutionSpace, SrcViewType, DstViewType, int64_t>(
        src, dst, dst_array, axis, merged_size, exec_space);
  } else {
    Unpack<ExecutionSpace, SrcViewType, DstViewType, int>(
        src, dst, dst_array, axis, merged_size, exec_space);
  }
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
