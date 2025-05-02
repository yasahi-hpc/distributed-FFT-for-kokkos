#ifndef PACKUNPACK_HPP
#define PACKUNPACK_HPP

#include <numeric>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "Mapping.hpp"

template <typename ExecutionSpace, typename DstViewType, typename SrcViewType,
          std::size_t Rank, typename iType>
struct Pack;

template <typename ExecutionSpace, typename DstViewType, typename SrcViewType,
          std::size_t Rank, typename iType>
struct Unpack;

template <typename ExecutionSpace, typename DstViewType, typename SrcViewType,
          typename iType>
struct Pack<ExecutionSpace, DstViewType, SrcViewType, 2, iType> {
  using LayoutType  = typename DstViewType::array_layout;
  using policy_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<3, Kokkos::Iterate::Default, Kokkos::Iterate::Default>,
      Kokkos::IndexType<iType>>;

  DstViewType m_dst;
  SrcViewType m_src;
  std::size_t m_axis;
  Kokkos::Array<std::size_t, 2> m_dst_extents;
  Kokkos::Array<std::size_t, 2> m_src_extents;

  /// \brief Constructor for the Roll functor.
  ///
  /// \param[in] in The input Kokkos view to be packed
  /// \param[out] out The output Kokkos view to be packed
  /// \param[in] axis The axis to be split
  /// \param exec_space[in] The Kokkos execution space to be used (defaults to
  /// ExecutionSpace()).
  Pack(const DstViewType& dst, const SrcViewType& src, const std::size_t axis,
       const ExecutionSpace exec_space = ExecutionSpace())
      : m_dst(dst),
        m_src(src),
        m_axis(axis),
        m_dst_extents({dst.extent(0), dst.extent(1)}),
        m_src_extents({src.extent(0), src.extent(1)}) {
    iType n0 = dst.extent(0), n1 = dst.extent(1), n2 = dst.extent(2);
    if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutRight>) {
      m_dst_extents[0] = dst.extent(1);
      m_dst_extents[1] = dst.extent(2);
    }
    policy_type policy(exec_space, {0, 0, 0}, {n0, n1, n2}, {4, 4, 4});
    Kokkos::parallel_for("KokkosFFT::Distributed::Pack-2D", policy, *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType i0, const iType i1, const iType i2) const {
    auto get_src = [&](iType idx_src, iType p, std::size_t axis) {
      return axis == m_axis ? idx_src + p * m_dst_extents[axis] : idx_src;
    };
    if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
      const iType p = i2;
      iType i0_src  = get_src(i0, p, 0);
      iType i1_src  = get_src(i1, p, 1);
      if (i0_src < m_src_extents[0] && i1_src < m_src_extents[1])
        m_dst(i0, i1, p) = m_src(i0_src, i1_src);
    } else {
      const iType p = i0;
      iType i0_src  = get_src(i1, p, 0);
      iType i1_src  = get_src(i2, p, 1);
      if (i0_src < m_src_extents[0] && i1_src < m_src_extents[1])
        m_dst(p, i1, i2) = m_src(i0_src, i1_src);
    }
  }
};

template <typename ExecutionSpace, typename DstViewType, typename SrcViewType,
          typename iType>
struct Pack<ExecutionSpace, DstViewType, SrcViewType, 3, iType> {
  using LayoutType  = typename DstViewType::array_layout;
  using policy_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<4, Kokkos::Iterate::Default, Kokkos::Iterate::Default>,
      Kokkos::IndexType<iType>>;

  DstViewType m_dst;
  SrcViewType m_src;
  std::size_t m_axis;
  Kokkos::Array<std::size_t, 3> m_dst_extents;
  Kokkos::Array<std::size_t, 3> m_src_extents;

  /// \brief Constructor for the Roll functor.
  ///
  /// \param[in] in The input Kokkos view to be packed
  /// \param[out] out The output Kokkos view to be packed
  /// \param[in] axis The axis to be split
  /// \param exec_space[in] The Kokkos execution space to be used (defaults to
  /// ExecutionSpace()).
  Pack(const DstViewType& dst, const SrcViewType& src, const std::size_t axis,
       const ExecutionSpace exec_space = ExecutionSpace())
      : m_dst(dst),
        m_src(src),
        m_axis(axis),
        m_dst_extents({dst.extent(0), dst.extent(1), dst.extent(2)}),
        m_src_extents({src.extent(0), src.extent(1), src.extent(2)}) {
    iType n0 = dst.extent(0), n1 = dst.extent(1), n2 = dst.extent(2),
          n3 = dst.extent(3);
    if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutRight>) {
      m_dst_extents[0] = dst.extent(1);
      m_dst_extents[1] = dst.extent(2);
      m_dst_extents[2] = dst.extent(3);
    }
    policy_type policy(exec_space, {0, 0, 0, 0}, {n0, n1, n2, n3},
                       {4, 4, 4, 1});
    Kokkos::parallel_for("KokkosFFT::Distributed::Pack-3D", policy, *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType i0, const iType i1, const iType i2,
                  const iType i3) const {
    auto get_src = [&](iType idx_src, iType p, std::size_t axis) {
      return axis == m_axis ? idx_src + p * m_dst_extents[axis] : idx_src;
    };
    if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
      const iType p = i3;
      iType i0_src  = get_src(i0, p, 0);
      iType i1_src  = get_src(i1, p, 1);
      iType i2_src  = get_src(i2, p, 2);
      if (i0_src < m_src_extents[0] && i1_src < m_src_extents[1] &&
          i2_src < m_src_extents[2]) {
        m_dst(i0, i1, i2, i3) = m_src(i0_src, i1_src, i2_src);
      }
    } else {
      const iType p = i0;
      iType i0_src  = get_src(i1, p, 0);
      iType i1_src  = get_src(i2, p, 1);
      iType i2_src  = get_src(i3, p, 2);
      if (i0_src < m_src_extents[0] && i1_src < m_src_extents[1] &&
          i2_src < m_src_extents[2]) {
        m_dst(i0, i1, i2, i3) = m_src(i0_src, i1_src, i2_src);
      }
    }
  }
};

template <typename ExecutionSpace, typename DstViewType, typename SrcViewType,
          typename iType>
struct Unpack<ExecutionSpace, DstViewType, SrcViewType, 2, iType> {
  using LayoutType  = typename DstViewType::array_layout;
  using policy_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<3, Kokkos::Iterate::Default, Kokkos::Iterate::Default>,
      Kokkos::IndexType<iType>>;

  DstViewType m_dst;
  SrcViewType m_src;
  std::size_t m_axis;
  Kokkos::Array<std::size_t, 2> m_map;
  Kokkos::Array<std::size_t, 2> m_dst_extents;
  Kokkos::Array<std::size_t, 2> m_src_extents;

  /// \brief Constructor for the Unpack functor.
  ///
  /// \param[out] dst The output Kokkos view to be unpacked
  /// \param[in] src The input Kokkos view to be unpacked
  /// \param[in] axis The axis to be split
  /// \param[in] map The mapping array to be used for the unpacking
  /// \param[in] exec_space The Kokkos execution space to be used (defaults to
  /// ExecutionSpace()).
  Unpack(const DstViewType& dst, const SrcViewType& src,
         const Kokkos::Array<std::size_t, 2>& map, const std::size_t axis,
         const ExecutionSpace exec_space = ExecutionSpace())
      : m_dst(dst),
        m_src(src),
        m_axis(axis),
        m_map(map),
        m_dst_extents({dst.extent(0), dst.extent(1)}),
        m_src_extents({src.extent(0), src.extent(1)}) {
    iType n0 = src.extent(0), n1 = src.extent(1), n2 = src.extent(2);
    if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutRight>) {
      m_src_extents[0] = src.extent(1);
      m_src_extents[1] = src.extent(2);
    }
    policy_type policy(exec_space, {0, 0, 0}, {n0, n1, n2}, {4, 4, 4});
    Kokkos::parallel_for("KokkosFFT::Distributed::Unpack-2D", policy, *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType i0, const iType i1, const iType i2) const {
    auto get_dst = [&](iType idx, iType p, std::size_t axis) {
      return axis == m_axis ? idx + p * iType(m_src_extents[axis]) : idx;
    };
    if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
      const iType p        = i2;
      iType dst_indices[2] = {get_dst(i0, p, 0), get_dst(i1, p, 1)};
      iType i0_dst         = dst_indices[m_map[0]];
      iType i1_dst         = dst_indices[m_map[1]];
      if (i0_dst < m_dst_extents[0] && i1_dst < m_dst_extents[1])
        m_dst(i0_dst, i1_dst) = m_src(i0, i1, i2);
    } else {
      const iType p        = i0;
      iType dst_indices[2] = {get_dst(i1, p, 0), get_dst(i2, p, 1)};
      iType i0_dst         = dst_indices[m_map[0]];
      iType i1_dst         = dst_indices[m_map[1]];
      if (i0_dst < m_dst_extents[0] && i1_dst < m_dst_extents[1])
        m_dst(i0_dst, i1_dst) = m_src(i0, i1, i2);
    }
  }
};

template <typename ExecutionSpace, typename DstViewType, typename SrcViewType,
          typename iType>
struct Unpack<ExecutionSpace, DstViewType, SrcViewType, 3, iType> {
  using LayoutType  = typename DstViewType::array_layout;
  using policy_type = Kokkos::MDRangePolicy<
      ExecutionSpace,
      Kokkos::Rank<4, Kokkos::Iterate::Default, Kokkos::Iterate::Default>,
      Kokkos::IndexType<iType>>;

  DstViewType m_dst;
  SrcViewType m_src;
  std::size_t m_axis;
  Kokkos::Array<std::size_t, 3> m_map;
  Kokkos::Array<std::size_t, 3> m_dst_extents;
  Kokkos::Array<std::size_t, 3> m_src_extents;

  /// \brief Constructor for the Unpack functor.
  ///
  /// \param[out] dst The output Kokkos view to be unpacked
  /// \param[in] src The input Kokkos view to be unpacked
  /// \param[in] axis The axis to be split
  /// \param[in] map The mapping array to be used for the unpacking
  /// \param[in] exec_space The Kokkos execution space to be used (defaults to
  /// ExecutionSpace()).
  Unpack(const DstViewType& dst, const SrcViewType& src,
         const Kokkos::Array<std::size_t, 3>& map, const std::size_t axis,
         const ExecutionSpace exec_space = ExecutionSpace())
      : m_dst(dst),
        m_src(src),
        m_axis(axis),
        m_map(map),
        m_dst_extents({dst.extent(0), dst.extent(1), dst.extent(2)}),
        m_src_extents({src.extent(0), src.extent(1), src.extent(2)}) {
    iType n0 = src.extent(0), n1 = src.extent(1), n2 = src.extent(2),
          n3 = src.extent(3);
    if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutRight>) {
      m_src_extents[0] = src.extent(1);
      m_src_extents[1] = src.extent(2);
      m_src_extents[2] = src.extent(3);
    }

    policy_type policy(exec_space, {0, 0, 0, 0}, {n0, n1, n2, n3},
                       {4, 4, 4, 1});
    Kokkos::parallel_for("KokkosFFT::Distributed::Unpack-3D", policy, *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType i0, const iType i1, const iType i2,
                  const iType i3) const {
    auto get_dst = [&](iType idx, iType p, std::size_t axis) {
      return axis == m_axis ? idx + p * iType(m_src_extents[axis]) : idx;
    };
    if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
      const iType p        = i3;
      iType dst_indices[3] = {get_dst(i0, p, 0), get_dst(i1, p, 1),
                              get_dst(i2, p, 2)};
      iType i0_dst         = dst_indices[m_map[0]];
      iType i1_dst         = dst_indices[m_map[1]];
      iType i2_dst         = dst_indices[m_map[2]];

      if (i0_dst < m_dst_extents[0] && i1_dst < m_dst_extents[1] &&
          i2_dst < m_dst_extents[2])
        m_dst(i0_dst, i1_dst, i2_dst) = m_src(i0, i1, i2, i3);
    } else {
      const iType p        = i0;
      iType dst_indices[3] = {get_dst(i1, p, 0), get_dst(i2, p, 1),
                              get_dst(i3, p, 2)};
      iType i0_dst         = dst_indices[m_map[0]];
      iType i1_dst         = dst_indices[m_map[1]];
      iType i2_dst         = dst_indices[m_map[2]];

      if (i0_dst < m_dst_extents[0] && i1_dst < m_dst_extents[1] &&
          i2_dst < m_dst_extents[2])
        m_dst(i0_dst, i1_dst, i2_dst) = m_src(i0, i1, i2, i3);
    }
  }
};

template <typename ExecutionSpace, typename DstViewType, typename SrcViewType>
void pack(const ExecutionSpace& exec_space, const DstViewType& dst,
          const SrcViewType& src, std::size_t axis) {
  static_assert(SrcViewType::rank() >= 2);
  static_assert(DstViewType::rank() == SrcViewType::rank() + 1);
  // DO shape check here
  if (dst.span() >= std::size_t(std::numeric_limits<int>::max()) ||
      src.span() >= std::size_t(std::numeric_limits<int>::max())) {
    Pack<ExecutionSpace, DstViewType, SrcViewType, SrcViewType::rank(),
         int64_t>(dst, src, axis, exec_space);
  } else {
    Pack<ExecutionSpace, DstViewType, SrcViewType, SrcViewType::rank(), int>(
        dst, src, axis, exec_space);
  }
}

template <typename ExecutionSpace, typename DstViewType, typename SrcViewType,
          std::size_t DIM>
void unpack(const ExecutionSpace& exec_space, const DstViewType& dst,
            const SrcViewType& src, std::array<std::size_t, DIM> src_map,
            std::size_t axis) {
  static_assert(DstViewType::rank() >= 2);
  static_assert(SrcViewType::rank() == DstViewType::rank() + 1);

  Kokkos::Array<std::size_t, DIM> dst_map = KokkosFFT::Impl::to_array(
      get_src_dst_map<typename DstViewType::array_layout>(src_map, axis));
  std::size_t mapped_axis = KokkosFFT::Impl::get_index(src_map, axis);

  // DO shape check here
  if (dst.span() >= std::size_t(std::numeric_limits<int>::max()) ||
      src.span() >= std::size_t(std::numeric_limits<int>::max())) {
    Unpack<ExecutionSpace, DstViewType, SrcViewType, DstViewType::rank(),
           int64_t>(dst, src, dst_map, mapped_axis, exec_space);
  } else {
    Unpack<ExecutionSpace, DstViewType, SrcViewType, DstViewType::rank(), int>(
        dst, src, dst_map, mapped_axis, exec_space);
  }
}

#endif
