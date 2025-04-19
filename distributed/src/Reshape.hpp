#ifndef RESHAPE_HPP
#define RESHAPE_HPP

#include <type_traits>
#include <Kokkos_Core.hpp>

// Primary template: recurse by adding one pointer and decreasing Rank.
template <typename T, std::size_t Rank>
struct add_pointer_n {
  using type = typename add_pointer_n<T*, Rank - 1>::type;
};

// Base case: Rank == 0, just yield T.
template <typename T>
struct add_pointer_n<T, 0> {
  using type = T;
};

// Convenience alias:
template <typename T, std::size_t Rank>
using add_pointer_n_t = typename add_pointer_n<T, Rank>::type;

/*
// Based on the current View and shape,
// propose a next decomposition based on the DecompositionType
template <typename ViewType, typename DecompositionType, std::size_t DIM>
auto propose_shape(const ViewType view,
                   shape_type<DIM> shape, axis_type<DIM> axes) {
  static_assert(ViewType::rank() >= 2 && ViewType::rank() >= DIM);
  // Shape is a current decomposition
  // DecompositionType can be Slab or Pencil
  // 2D View: 1D decomp case
  // (n0, n1), (p0, a0) -> (n0, p0, n1/p0)
  // 2D View: 2D decomp case
  // (n0, n1), (p0, p1) -> (n0, p0, n1/p0) && (n0, p0, n1/p0)
  // 3D View: 1D decomp case
  // (n0, n1, n2), (p0, a0) -> (n0, n1, p0, n2/p0)
  // 3D View: 2D decomp case
  // (n0, n1, n2), (p0, p1) -> slab (n0, n1, p0, p1, n2/(p0*p1))
  //                        -> pencil (n0, n1, p0, n2/p0) && (n0, n1, p1, n2/p1)
  // 3D View: 3D decomp case
  // (n0, n1, n2), (p0, p1, p2) -> slab (n0, n1, p0, p1, n2/(p0*p1)) -> Pencil
(p2, p0*n0/p2, p1*n1, n2/p2)
  //                            -> pencil (n0, n1, p0, n2/p0) -> Pencil (p1,
p0*n0/p1, n1, n2/p0) -> Pencil (p0*n0/p1, p2, p1*n1/p2, n2/p0)


  return new_shape;
}

template <typename ExecutionSpace, typename ViewType, std::size_t Rank,
          typename iType>
struct Reshape {
  static_assert(Rank >= 1);
  using ArrayType  = Kokkos::Array<std::size_t, Rank>;
  using LayoutType = typename ViewType::array_layout;
  using value_type = ViewType::non_const_value_type;
  using element_type = add_pointer_n_t<value_type, Rank>;
  using ReshapedViewType = Kokkos::View<element_type, LayoutType,
ExecutionSpace>;

  ReshapedViewType m_view;

  Reshape(const ViewType& in, const ArrayType& shape, const ExecutionSpace )
      : m_view(in.data(), create_layout<LayoutType>(m_shape)) {
    // Check size consistency here
      }

  auto operator()() const {
    return m_view;
  }
};
*/

#endif
