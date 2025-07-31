#ifndef PACKUNPACK_HPP
#define PACKUNPACK_HPP

#include <numeric>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <KokkosFFT.hpp>
#include "Mapping.hpp"

template <typename iType, typename ArrayType>
KOKKOS_INLINE_FUNCTION iType merge_indices(iType idx, iType p,
                                           const ArrayType& extents,
                                           std::size_t axis,
                                           std::size_t merged_axis) {
  return axis == merged_axis ? idx + p * iType(extents[axis]) : idx;
}

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
  /// \brief Constructor for the Pack functor.
  ///
  /// \param[in] src The input Kokkos view to be packed
  /// \param[out] dst The output Kokkos view to be packed
  /// \param[in] axis The axis to be split
  /// \param[in] exec_space The Kokkos execution space to be used (defaults to
  /// ExecutionSpace()).
  Pack(const SrcViewType& src, const DstViewType& dst, const ShapeType& map,
       const std::size_t axis,
       const ExecutionSpace exec_space = ExecutionSpace()) {
    Kokkos::parallel_for("KokkosFFT::Distributed::Pack",
                         get_policy(exec_space, dst),
                         PackInternal(src, dst, map, axis));
  }

  struct PackInternal {
    using LayoutType = typename DstViewType::array_layout;
    using ValueType  = typename SrcViewType::non_const_value_type;
    SrcViewType m_src;
    DstViewType m_dst;
    std::size_t m_axis;
    ShapeType m_map;
    ShapeType m_dst_extents;
    ShapeType m_src_extents;

    PackInternal(const SrcViewType& src, const DstViewType& dst,
                 const ShapeType& map, const std::size_t axis)
        : m_src(src), m_dst(dst), m_axis(axis), m_map(map) {
      for (std::size_t i = 0; i < SrcViewType::rank(); ++i) {
        m_dst_extents[i] = std::is_same_v<LayoutType, Kokkos::LayoutRight>
                               ? dst.extent(i + 1)
                               : dst.extent(i);
      }
      for (std::size_t i = 0; i < SrcViewType::rank(); ++i) {
        m_src_extents[i] = src.extent(i);
      }
    }

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

    template <std::size_t... Is>
    KOKKOS_INLINE_FUNCTION ValueType
    get_src_value(iType dst_idx[], std::index_sequence<Is...>) const {
      // Bounds check
      bool out_of_bounds = false;

      if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
        const iType p = dst_idx[DstViewType::rank() - 1];
        iType src_indices[DstViewType::rank() - 1] = {
            merge_indices(dst_idx[Is], p, m_dst_extents, Is, m_axis)...};
        for (std::size_t i = 0; i < DstViewType::rank() - 1; ++i) {
          if (src_indices[m_map[i]] >= iType(m_src_extents[i]))
            out_of_bounds = true;
        }
        return out_of_bounds ? ValueType(0) : m_src(src_indices[m_map[Is]]...);
      } else {
        const iType p                              = dst_idx[0];
        iType src_indices[DstViewType::rank() - 1] = {
            merge_indices(dst_idx[Is + 1], p, m_dst_extents, Is, m_axis)...};
        for (std::size_t i = 0; i < DstViewType::rank() - 1; ++i) {
          if (src_indices[m_map[i]] >= iType(m_src_extents[i]))
            out_of_bounds = true;
        }
        return out_of_bounds ? ValueType(0) : m_src(src_indices[m_map[Is]]...);
      }
    }
  };
};

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
  /// \brief Constructor for the Unpack functor.
  ///
  /// \param[in] src The input Kokkos view to be packed
  /// \param[out] dst The output Kokkos view to be packed
  /// \param[in] axis The axis to be split
  /// \param[in] exec_space The Kokkos execution space to be used (defaults to
  /// ExecutionSpace()).
  Unpack(const SrcViewType& src, const DstViewType& dst, const ShapeType& map,
         const std::size_t axis,
         const ExecutionSpace exec_space = ExecutionSpace()) {
    Kokkos::parallel_for("KokkosFFT::Distributed::Unpack",
                         get_policy(exec_space, src),
                         UnpackInternal(src, dst, map, axis));
  }

  struct UnpackInternal {
    using LayoutType = typename DstViewType::array_layout;
    using ValueType  = typename SrcViewType::non_const_value_type;
    SrcViewType m_src;
    DstViewType m_dst;
    std::size_t m_axis;
    ShapeType m_map;
    ShapeType m_dst_extents;
    ShapeType m_src_extents;

    UnpackInternal(const SrcViewType& src, const DstViewType& dst,
                   const ShapeType& map, const std::size_t axis)
        : m_src(src), m_dst(dst), m_axis(axis), m_map(map) {
      for (std::size_t i = 0; i < DstViewType::rank(); ++i) {
        m_src_extents[i] = std::is_same_v<LayoutType, Kokkos::LayoutRight>
                               ? src.extent(i + 1)
                               : src.extent(i);
      }
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
      // Bounds check
      bool in_bounds = true;
      if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
        const iType p = src_idx[SrcViewType::rank() - 1];
        iType dst_indices[SrcViewType::rank() - 1] = {
            merge_indices(src_idx[Is], p, m_src_extents, Is, m_axis)...};
        for (std::size_t i = 0; i < SrcViewType::rank() - 1; ++i) {
          if (dst_indices[m_map[i]] >= iType(m_dst_extents[i]))
            in_bounds = false;
        }
        if (in_bounds) {
          m_dst(dst_indices[m_map[Is]]...) = src_value;
        }
      } else {
        const iType p                              = src_idx[0];
        iType dst_indices[SrcViewType::rank() - 1] = {
            merge_indices(src_idx[Is + 1], p, m_src_extents, Is, m_axis)...};
        for (std::size_t i = 0; i < SrcViewType::rank() - 1; ++i) {
          if (dst_indices[m_map[i]] >= iType(m_dst_extents[i]))
            in_bounds = false;
        }
        if (in_bounds) {
          m_dst(dst_indices[m_map[Is]]...) = src_value;
        }
      }
    }
  };
};

template <typename ExecutionSpace, typename SrcViewType, typename DstViewType,
          std::size_t DIM>
void pack(const ExecutionSpace& exec_space, const SrcViewType& src,
          const DstViewType& dst, std::array<std::size_t, DIM> src_map,
          std::size_t axis) {
  static_assert(SrcViewType::rank() >= 2);
  static_assert(DstViewType::rank() == SrcViewType::rank() + 1);

  Kokkos::Profiling::ScopedRegion region("pack");
  Kokkos::Array<std::size_t, DIM> src_array =
      KokkosFFT::Impl::to_array(src_map);
  if (src.span() >= std::size_t(std::numeric_limits<int>::max()) ||
      dst.span() >= std::size_t(std::numeric_limits<int>::max())) {
    Pack<ExecutionSpace, SrcViewType, DstViewType, int64_t>(src, dst, src_array,
                                                            axis, exec_space);
  } else {
    Pack<ExecutionSpace, SrcViewType, DstViewType, int>(src, dst, src_array,
                                                        axis, exec_space);
  }
}

template <typename ExecutionSpace, typename SrcViewType, typename DstViewType,
          std::size_t DIM>
void unpack(const ExecutionSpace& exec_space, const SrcViewType& src,
            const DstViewType& dst, std::array<std::size_t, DIM> dst_map,
            std::size_t axis) {
  static_assert(DstViewType::rank() >= 2);
  static_assert(SrcViewType::rank() == DstViewType::rank() + 1);
  Kokkos::Profiling::ScopedRegion region("unpack");
  Kokkos::Array<std::size_t, DIM> dst_array =
      KokkosFFT::Impl::to_array(dst_map);
  if (dst.span() >= std::size_t(std::numeric_limits<int>::max()) ||
      src.span() >= std::size_t(std::numeric_limits<int>::max())) {
    Unpack<ExecutionSpace, SrcViewType, DstViewType, int64_t>(
        src, dst, dst_array, axis, exec_space);
  } else {
    Unpack<ExecutionSpace, SrcViewType, DstViewType, int>(src, dst, dst_array,
                                                          axis, exec_space);
  }
}

#endif
