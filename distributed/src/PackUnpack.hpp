#ifndef PACKUNPACK_HPP
#define PACKUNPACK_HPP

#include <Kokkos_Core.hpp>

template <typename ExecutionSpace, typename ViewType, std::size_t Rank,
          typename iType>
struct Pack;

template <typename ExecutionSpace, typename ViewType, typename iType>
struct Pack<ExecutionSpace, ViewType, 1, iType> {
  using ArrayType          = Kokkos::Array<std::size_t, 1>;
  using LayoutType         = typename ViewType::array_layout;
  using ManageableViewType = typename manageable_view_type<ViewType>::type;
  using policy_type =
      Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<iType>>;

  ViewType m_x;
  ManageableViewType m_tmp;
  ArrayType m_shifts;

  /// \brief Constructor for the Roll functor.
  ///
  /// \param x[in,out] The input/output Kokkos view to be shifted.
  /// \param shifts[in] The shift amounts for each axis.
  /// \param exec_space[in] The Kokkos execution space to be used (defaults to
  /// ExecutionSpace()).
  Pack(const ViewType& x, const ArrayType& shifts,
       const ExecutionSpace exec_space = ExecutionSpace())
      : m_x(x),
        m_tmp("tmp", create_layout<LayoutType>(extract_extents(x))),
        m_shifts(shifts) {
    Kokkos::parallel_for("KokkosFFT::Distributed::Pack-1D",
                         policy_type(exec_space, 0, m_x.extent(0)), *this);
    Kokkos::deep_copy(exec_space, m_x, m_tmp);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const iType i0) const {
    iType i0_dst  = (i0 + iType(m_shifts[0])) % iType(m_x.extent(0));
    m_tmp(i0_dst) = m_x(i0);
  }
};
  
#endif
