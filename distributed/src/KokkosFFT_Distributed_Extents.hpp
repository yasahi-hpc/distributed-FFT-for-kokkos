#ifndef KOKKOSFFT_DISTRIBUTED_EXTENTS_HPP
#define KOKKOSFFT_DISTRIBUTED_EXTENTS_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_Types.hpp"
#include "KokkosFFT_Distributed_ContainerAnalyses.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief Calculate the buffer extents based on the global extents,
/// the in-topology, and the out-topology.
///
/// Example
/// Global View extents (n0, n1, n2, n3)
/// in-topology = {1, p0, p1, 1} // X-pencil
/// out-topology = {p0, 1, p1, 1} // Y-pencil
/// Buffer View (p0, n0/p0, n1/p0, n2/p1, n3)
///
/// \tparam LayoutType The layout type of the buffer view (e.g.,
/// Kokkos::LayoutRight).
/// \tparam iType The integer type used for extents and topology.
/// \tparam DIM The number of dimensions of the extents.
///
/// \param[in] extents Extents of the global View.
/// \param[in] in_topology A topology representing the distribution of the input
/// data.
/// \param[in] out_topology A topology representing the distribution of
/// the output data.
/// \return A buffer extents of the view needed for the pencil
/// transformation.
template <typename LayoutType, typename iType, std::size_t DIM>
auto compute_buffer_extents(const std::array<iType, DIM> &extents,
                            const std::array<iType, DIM> &in_topology,
                            const std::array<iType, DIM> &out_topology) {
  static_assert(std::is_same_v<LayoutType, Kokkos::LayoutLeft> ||
                    std::is_same_v<LayoutType, Kokkos::LayoutRight>,
                "compute_buffer_extents: We only accept LayoutLeft or "
                "LayoutRight for the buffer View.");
  std::array<iType, DIM + 1> buffer_extents{};
  auto merged_topology = merge_topology(in_topology, out_topology);
  auto p0 = diff_topology(merged_topology, in_topology);  // return 1 or p0
  if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutRight>) {
    buffer_extents.at(0) = p0;
    for (std::size_t i = 0; i < extents.size(); i++) {
      buffer_extents.at(i + 1) =
          (extents.at(i) - 1) / merged_topology.at(i) + 1;
    }
  } else {
    for (std::size_t i = 0; i < extents.size(); i++) {
      buffer_extents.at(i) = (extents.at(i) - 1) / merged_topology.at(i) + 1;
    }
    buffer_extents.back() = p0;
  }
  return buffer_extents;
}

/// \brief Calculate the buffer extents based on the global extents,
/// the in-topology, and the out-topology.
///
/// Example
/// Global View extents (n0, n1, n2, n3)
/// in-topology = {1, p0, p1, 1} // X-pencil
/// out-topology = {p0, 1, p1, 1} // Y-pencil
/// Buffer View (p0, n0/p0, n1/p0, n2/p1, n3)
///
/// \tparam LayoutType The layout type of the buffer view (e.g.,
/// Kokkos::LayoutRight).
/// \tparam iType The integer type used for extents and topology.
/// \tparam DIM The number of dimensions of the extents.
/// \tparam InLayoutType The layout type of the in-topology (e.g.,
/// Kokkos::LayoutRight).
/// \tparam OutLayoutType The layout type of the out-topology (e.g.,
/// Kokkos::LayoutRight).
///
/// \param[in] extents Extents of the global View.
/// \param[in] in_topology A topology representing the distribution of the input
/// data.
/// \param[in] out_topology A topology representing the distribution of
/// the output data.
/// \return A buffer extents of the view needed for the pencil
/// transformation.
template <typename LayoutType, typename iType, std::size_t DIM = 1,
          typename InLayoutType  = Kokkos::LayoutRight,
          typename OutLayoutType = Kokkos::LayoutRight>
auto compute_buffer_extents(
    const std::array<iType, DIM> &extents,
    const Topology<iType, DIM, InLayoutType> &in_topology,
    const Topology<iType, DIM, OutLayoutType> &out_topology) {
  return compute_buffer_extents<LayoutType>(extents, in_topology.array(),
                                            out_topology.array());
}

/// \brief Compute the larger extents. Larger one corresponds to
/// the extents to FFT library. This is a helper for vendor library
/// which supports 2D or 3D non-batched FFTs.
///
/// Example
/// in extents: (8, 7, 8)
/// out extents: (8, 7, 5)
/// axes: (0, 1, 2)
/// FFT is operated from 2nd axis, so we have
/// fft extents: (8, 7, 8)
///
/// \tparam iType The integer type used for extents
/// \tparam DIM The number of dimensions of the extents.
/// \tparam FFT_DIM The number of dimensions of the FFT.
///
/// \param[in] in_extents Extents of the global input View.
/// \param[in] out_extents Extents of the global output View.
/// \param[in] axes Axes of the transform
/// \return A extents of the permuted view
template <typename iType, std::size_t DIM, std::size_t FFT_DIM>
auto compute_fft_extents(const std::array<iType, DIM> &in_extents,
                         const std::array<iType, DIM> &out_extents,
                         const std::array<iType, FFT_DIM> &axes) {
  static_assert(std::is_integral_v<iType>,
                "compute_fft_extents: iType must be an integral type");
  static_assert(
      FFT_DIM >= 1 && FFT_DIM <= KokkosFFT::MAX_FFT_DIM,
      "compute_fft_extents: the Rank of FFT axes must be between 1 and 3");
  static_assert(
      DIM >= FFT_DIM,
      "compute_fft_extents: View rank must be larger than or equal to "
      "the Rank of FFT axes");

  std::array<iType, FFT_DIM> fft_extents{};
  std::transform(axes.begin(), axes.end(), fft_extents.begin(),
                 [&](iType axis) {
                   return std::max(in_extents.at(axis), out_extents.at(axis));
                 });

  return fft_extents;
}

/// \brief Whether the memory of two Views overlaps. The memory of a View is
/// the byte range [data(), data() + span() * sizeof(value_type)). Views that
/// start at the same address overlap; empty Views never overlap. The data is
/// not accessed, so the Views may live in any memory space.
///
/// \tparam AViewType The type of the first View
/// \tparam BViewType The type of the second View
///
/// \param[in] a The first View
/// \param[in] b The second View
/// \return True if the byte ranges of a and b intersect
template <typename AViewType, typename BViewType>
bool are_overlapping(const AViewType &a, const BViewType &b) {
  using a_value_type        = typename AViewType::non_const_value_type;
  using b_value_type        = typename BViewType::non_const_value_type;
  const std::size_t a_bytes = a.span() * sizeof(a_value_type);
  const std::size_t b_bytes = b.span() * sizeof(b_value_type);
  if (a_bytes == 0 || b_bytes == 0) return false;

  const auto a_begin = reinterpret_cast<std::uintptr_t>(a.data());
  const auto b_begin = reinterpret_cast<std::uintptr_t>(b.data());
  return a_begin < b_begin + b_bytes && b_begin < a_begin + a_bytes;
}

/// \brief Whether the input and output Views of a plan point to the same data,
/// i.e. whether the transform is in-place. Views without data (empty) never
/// do.
///
/// \tparam InViewType The type of the input View
/// \tparam OutViewType The type of the output View
///
/// \param[in] in The input View
/// \param[in] out The output View
/// \return True if in and out start at the same address
/// \throws std::runtime_error if the Views overlap partially, i.e. they
/// overlap but do not start at the same address
template <typename InViewType, typename OutViewType>
bool check_inplace(const InViewType &in, const OutViewType &out) {
  const bool is_aliasing = in.data() != nullptr &&
                           KokkosFFT::Impl::are_aliasing(in.data(), out.data());
  KOKKOSFFT_THROW_IF(!is_aliasing && are_overlapping(in, out),
                     "the input and output Views overlap partially: they "
                     "must either point to the same data (in-place) or not "
                     "overlap (out-of-place)");
  return is_aliasing;
}

}  // namespace Impl

/// \brief Size of the allocation shared by the input and the output Views of
/// an in-place transform on the current rank, in elements of OutValueType.
///
/// The input and the output Views of an in-place transform point to the same
/// data, and each is dense in its own local extents. The allocation must hold
/// the larger of the two on every rank. This is not always the output: with
/// uneven distributions, the input can be larger on some ranks.
///
/// Example (R2C, double): input local extents (8, 2), i.e. 128 bytes, and
/// output local extents (1, 7), i.e. 112 bytes: the allocation needs 8
/// Kokkos::complex<double>.
///
/// \tparam InValueType The value type of the input View
/// \tparam OutValueType The value type of the output View
/// \tparam DIM The rank of the Views
///
/// \param[in] in_local_extents Local extents of the input View
/// \param[in] out_local_extents Local extents of the output View
/// \return The number of OutValueType elements to allocate
template <typename InValueType, typename OutValueType, std::size_t DIM>
std::size_t compute_inplace_allocation_size(
    const std::array<std::size_t, DIM> &in_local_extents,
    const std::array<std::size_t, DIM> &out_local_extents) {
  const std::size_t in_bytes =
      KokkosFFT::Impl::total_size(in_local_extents) * sizeof(InValueType);
  const std::size_t out_bytes =
      KokkosFFT::Impl::total_size(out_local_extents) * sizeof(OutValueType);
  const std::size_t bytes = std::max(in_bytes, out_bytes);
  return (bytes + sizeof(OutValueType) - 1) / sizeof(OutValueType);
}

}  // namespace Distributed
}  // namespace KokkosFFT

#endif
