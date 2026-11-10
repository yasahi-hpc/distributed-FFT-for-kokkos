#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <utility>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <Kokkos_Core.hpp>

template <typename ViewType>
void display(ViewType& a) {
  auto label   = a.label();
  const auto n = a.size();

  auto h_a = Kokkos::create_mirror_view(a);
  Kokkos::deep_copy(h_a, a);
  auto* data = h_a.data();

  std::cout << std::scientific << std::setprecision(16) << std::flush;
  for (std::size_t i = 0; i < n; i++) {
    std::cout << label + "[" << i << "]: " << i << ", " << data[i] << std::endl;
  }
  std::cout << std::resetiosflags(std::ios_base::floatfield);
}

template <typename ExecutionSpace, typename AViewType, typename BViewType>
bool allclose(const ExecutionSpace& exec, const AViewType& a,
              const BViewType& b, double rtol = 1.e-5, double atol = 1.e-8) {
  constexpr std::size_t rank = AViewType::rank;
  for (std::size_t i = 0; i < rank; i++) {
    assert(a.extent(i) == b.extent(i));
  }
  const auto n = a.size();

  auto* ptr_a = a.data();
  auto* ptr_b = b.data();

  int error = 0;
  Kokkos::parallel_reduce(
      "KokkosFFT::Test::allclose",
      Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::size_t>>(exec,
                                                                          0, n),
      KOKKOS_LAMBDA(const std::size_t& i, int& err) {
        auto tmp_a = ptr_a[i];
        auto tmp_b = ptr_b[i];
        bool not_close =
            Kokkos::abs(tmp_a - tmp_b) > (atol + rtol * Kokkos::abs(tmp_b));
        err += static_cast<int>(not_close);
      },
      error);

  if (error != 0) {
    // Report the size of the mismatch, to choose tolerances from data: the
    // largest |a - b|, and the smallest atol that would pass with this rtol
    double max_diff = 0, needed_atol = 0;
    Kokkos::parallel_reduce(
        "KokkosFFT::Test::allclose_report",
        Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::size_t>>(
            exec, 0, n),
        KOKKOS_LAMBDA(const std::size_t& i, double& diff, double& need) {
          const double d = Kokkos::abs(ptr_a[i] - ptr_b[i]);
          const double e = d - rtol * Kokkos::abs(ptr_b[i]);
          if (d > diff) diff = d;
          if (e > need) need = e;
        },
        Kokkos::Max<double>(max_diff), Kokkos::Max<double>(needed_atol));
    std::cerr << std::scientific << std::setprecision(3) << "allclose("
              << a.label() << ", " << b.label() << "): " << error << " of " << n
              << " not close (rtol " << rtol << ", atol " << atol
              << "), max |a - b| " << max_diff << ", atol needed "
              << needed_atol << std::endl;
    std::cerr << std::resetiosflags(std::ios_base::floatfield)
              << std::setprecision(6);
  }

  return error == 0;
}

template <typename LayoutType, typename iType, std::size_t Rank>
inline auto get_buffer_shape(const std::array<iType, Rank>& shape,
                             iType nprocs) {
  std::array<iType, Rank + 1> buffer_shape;
  if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    for (std::size_t i = 0; i < Rank; i++) {
      buffer_shape[i] = shape[i];
    }
    buffer_shape[Rank] = nprocs;
  } else {
    buffer_shape[0] = nprocs;
    for (std::size_t i = 0; i < Rank; i++) {
      buffer_shape[i + 1] = shape[i];
    }
  }

  return buffer_shape;
}

template <typename iType>
inline auto distribute_extents(iType n, iType r, iType p) {
  iType base_size = n / p, remainder = n % p;
  iType length = base_size + (r < remainder ? 1 : 0);
  iType start  = r * base_size + std::min(r, remainder);
  return std::pair<iType, iType>{start, length};
};

#endif
