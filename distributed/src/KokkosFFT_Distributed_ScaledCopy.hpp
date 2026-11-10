#ifndef KOKKOSFFT_DISTRIBUTED_SCALED_COPY_HPP
#define KOKKOSFFT_DISTRIBUTED_SCALED_COPY_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief Functor of transpose_and_scale: out(j) = coef * in(i), where the
/// source index is i[m] = j[map_inv[m]] (the same correspondence as
/// KokkosFFT::Impl::transpose, out(i[map[0]], i[map[1]], ...) = in(i...)).
/// It iterates over out, which may be smaller than in (the R2C padding of
/// the input is cropped), never larger.
/// \tparam InViewType Kokkos View type of the source
/// \tparam OutViewType Kokkos View type of the destination
/// \tparam ScaleType Floating point type of the coefficient
/// \tparam IndexType Integer type of the indices
template <typename InViewType, typename OutViewType, typename ScaleType,
          typename IndexType>
struct ScaledTranspose {
  static constexpr std::size_t rank = InViewType::rank();

  InViewType m_in;
  OutViewType m_out;
  Kokkos::Array<IndexType, rank> m_map_inv;
  ScaleType m_coef;

  ScaledTranspose(const InViewType& in, const OutViewType& out,
                  const Kokkos::Array<IndexType, rank>& map_inv, ScaleType coef)
      : m_in(in), m_out(out), m_map_inv(map_inv), m_coef(coef) {}

  template <typename... IndicesType>
  KOKKOS_INLINE_FUNCTION void operator()(const IndicesType... indices) const {
    const IndexType dst[rank] = {static_cast<IndexType>(indices)...};
    apply(dst, std::make_index_sequence<rank>{});
  }

  template <std::size_t... Is>
  KOKKOS_INLINE_FUNCTION void apply(const IndexType dst[],
                                    std::index_sequence<Is...>) const {
    m_out(dst[Is]...) = m_in(dst[m_map_inv[Is]]...) * m_coef;
  }
};

/// \brief Copy (or transpose, and change the layout) a View into another and
/// multiply by a coefficient, in one pass: the normalization fused into a
/// copy that is needed anyway (the copy-out of cuFFTMp from its NVSHMEM
/// buffer). With a coefficient of 1, it is KokkosFFT::Impl::transpose (a
/// plain copy when the map is the identity).
///
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam InViewType Kokkos View type of the source
/// \tparam OutViewType Kokkos View type of the destination
/// \tparam ScaleType Floating point type of the coefficient
///
/// \param[in] exec_space Kokkos execution space
/// \param[in] in Source View
/// \param[out] out Destination View: out(i[map[0]], i[map[1]], ...) =
/// coef * in(i...); its extents are at most those of in, mapped
/// \param[in] map Axis map, as for KokkosFFT::Impl::transpose
/// \param[in] coef Coefficient
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          typename ScaleType>
void transpose_and_scale(const ExecutionSpace& exec_space, const InViewType& in,
                         const OutViewType& out,
                         const std::array<int, InViewType::rank()>& map,
                         ScaleType coef) {
  static_assert(InViewType::rank() == OutViewType::rank(),
                "transpose_and_scale: In and Out View must have the same rank");
  static_assert(InViewType::rank() >= 1 && InViewType::rank() <= 6,
                "transpose_and_scale: the rank must be between 1 and 6");
  using out_value_type = typename OutViewType::non_const_value_type;
  using float_type = KokkosFFT::Impl::base_floating_point_type<out_value_type>;
  constexpr std::size_t rank = InViewType::rank();

  if (coef == ScaleType(1)) {
    KokkosFFT::Impl::transpose(exec_space, in, out, map, true);
    return;
  }

  auto launch = [&](auto index) {
    using IndexType = decltype(index);
    Kokkos::Array<IndexType, rank> map_inv{};
    for (std::size_t i = 0; i < rank; ++i) {
      map_inv[static_cast<std::size_t>(map[i])] = static_cast<IndexType>(i);
    }
    Kokkos::parallel_for(
        "KokkosFFT::Distributed::transpose_and_scale",
        KokkosFFT::Impl::get_mdpolicy<IndexType>(exec_space, out),
        ScaledTranspose<InViewType, OutViewType, float_type, IndexType>(
            in, out, map_inv, static_cast<float_type>(coef)));
  };
  if ((in.span() >= std::size_t(std::numeric_limits<int>::max())) ||
      (out.span() >= std::size_t(std::numeric_limits<int>::max()))) {
    launch(std::int64_t{0});
  } else {
    launch(int{0});
  }
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
