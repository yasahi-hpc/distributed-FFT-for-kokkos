#ifndef KOKKOSFFT_DISTRIBUTED_TPL_AXES_HPP
#define KOKKOSFFT_DISTRIBUTED_TPL_AXES_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_Types.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief The FFT axes in the order the vendor plans (cuFFTMp, rocFFT-MPI)
/// are made with, for the same result as the given order:
/// - complex-to-complex: the axes in increasing order. A multidimensional DFT
///   does not depend on the order of its axes;
/// - real-to-complex / complex-to-real: the axes but the last in increasing
///   order, then the last axis. Only the last axis (the one halved) matters.
/// With these axes, the axis map (KokkosFFT::Impl::get_map_axes) is the
/// identity whenever the axes are the last dimensions of a LayoutRight View
/// in some order (and, for real transforms, the last axis is the last
/// dimension), so that no transpose is needed for e.g. {1, 0} (C2C) or
/// {1, 0, 2}.
///
/// \tparam InValueType Value type of the input of the forward transform
/// (real for R2C / C2R, complex for C2C)
/// \tparam DIM Number of FFT axes
/// \param[in] axes Axes of the FFT, possibly negative
/// \param[in] rank Rank of the Views
/// \return The axes, non-negative, in the order above
template <typename InValueType, std::size_t DIM>
KokkosFFT::axis_type<DIM> canonical_tpl_axes(
    const KokkosFFT::axis_type<DIM>& axes, std::size_t rank) {
  auto canonical_axes    = KokkosFFT::Impl::convert_negative_axes(axes, rank);
  constexpr bool is_real = KokkosFFT::Impl::is_real_v<InValueType>;
  auto last = is_real ? canonical_axes.end() - 1 : canonical_axes.end();
  std::sort(canonical_axes.begin(), last);
  return canonical_axes;
}

/// \brief Whether the vendor plans need a transpose for these axes: the map
/// of the canonical axes, in the LayoutRight convention of TplPlan (the data
/// is described in the LayoutRight order of the indices), is not the identity
/// \tparam InValueType Value type of the input of the forward transform
/// \tparam Rank Rank of the Views
/// \tparam DIM Number of FFT axes
/// \param[in] axes Axes of the FFT, possibly negative
template <typename InValueType, std::size_t Rank, std::size_t DIM>
bool is_tpl_transpose_needed(const KokkosFFT::axis_type<DIM>& axes) {
  [[maybe_unused]] auto [map, map_inv] =
      KokkosFFT::Impl::get_map_axes<Kokkos::LayoutRight, Rank>(
          canonical_tpl_axes<InValueType>(axes, Rank));
  return KokkosFFT::Impl::is_transpose_needed(map);
}

/// \brief The axes of the same FFT on the reversed View: a LayoutLeft View of
/// extents (n0, ..., n_{r-1}) has the memory of a LayoutRight View of extents
/// (n_{r-1}, ..., n0), whose axis r - 1 - a is the axis a of the former
/// \tparam DIM Number of FFT axes
/// \param[in] axes Axes of the FFT, possibly negative
/// \param[in] rank Rank of the Views
/// \return The axes on the reversed View, non-negative, in the same order
template <std::size_t DIM>
KokkosFFT::axis_type<DIM> reversed_tpl_axes(
    const KokkosFFT::axis_type<DIM>& axes, std::size_t rank) {
  auto reversed_axes = KokkosFFT::Impl::convert_negative_axes(axes, rank);
  const int irank    = static_cast<int>(rank);
  for (auto& axis : reversed_axes) axis = irank - 1 - axis;
  return reversed_axes;
}

/// \brief The opposite layout of a topology (LayoutRight <-> LayoutLeft)
template <typename LayoutType>
using flipped_topology_layout_t =
    std::conditional_t<std::is_same_v<LayoutType, Kokkos::LayoutRight>,
                       Kokkos::LayoutLeft, Kokkos::LayoutRight>;

/// \brief The topology of the reversed View (reversed_tpl_axes): the process
/// grid with its dimensions reversed, and the opposite layout, so that every
/// rank keeps its block. In a LayoutRight topology the last coordinate varies
/// fastest with the rank; once the dimensions are reversed, it is the first
/// one, which is the LayoutLeft rule (and conversely).
/// \tparam N Rank of the topology
/// \tparam LayoutType Layout of the topology
/// \param[in] topology The topology of the View
/// \return The topology of the reversed View
template <std::size_t N, typename LayoutType>
auto reversed_topology(const Topology<std::size_t, N, LayoutType>& topology) {
  std::array<std::size_t, N> reversed = topology.array();
  std::reverse(reversed.begin(), reversed.end());
  return Topology<std::size_t, N, flipped_topology_layout_t<LayoutType>>(
      reversed);
}

/// \brief The extents of the reversed View
/// \tparam N Rank of the View
/// \param[in] extents The extents of the View
template <std::size_t N>
std::array<std::size_t, N> reversed_extents(
    const std::array<std::size_t, N>& extents) {
  std::array<std::size_t, N> reversed = extents;
  std::reverse(reversed.begin(), reversed.end());
  return reversed;
}

/// \brief Whether a vendor plan on LayoutLeft Views runs on the reversed
/// LayoutRight Views (no copy): when the axes need no transpose there. A
/// LayoutRight View never does.
/// \tparam ViewType Kokkos View type of the input
/// \tparam DIM Number of FFT axes
/// \param[in] axes Axes of the FFT
template <typename ViewType, std::size_t DIM>
bool is_tpl_reversed(const KokkosFFT::axis_type<DIM>& axes) {
  using value_type           = typename ViewType::non_const_value_type;
  constexpr std::size_t rank = ViewType::rank();
  if constexpr (std::is_same_v<typename ViewType::array_layout,
                               Kokkos::LayoutLeft>) {
    return !is_tpl_transpose_needed<value_type, rank>(
        reversed_tpl_axes(axes, rank));
  } else {
    return false;
  }
}

/// \brief Whether the vendor plans need a transpose (not only a copy that
/// changes the layout): LayoutRight Views when the axes need one; LayoutLeft
/// Views when they need one both as they are (the layout-changing copy into
/// a LayoutRight buffer) and reversed (is_tpl_reversed)
/// \tparam ViewType Kokkos View type of the input
/// \tparam DIM Number of FFT axes
/// \param[in] axes Axes of the FFT
template <typename ViewType, std::size_t DIM>
bool is_tpl_permutation_needed(const KokkosFFT::axis_type<DIM>& axes) {
  using value_type           = typename ViewType::non_const_value_type;
  constexpr std::size_t rank = ViewType::rank();
  return is_tpl_transpose_needed<value_type, rank>(axes) &&
         !is_tpl_reversed<ViewType>(axes);
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
