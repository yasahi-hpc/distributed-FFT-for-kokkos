#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <Kokkos_Core.hpp>
#include <iomanip>

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

  return error == 0;
}

inline auto get_r2c_shape(std::size_t len, bool is_R2C) {
  return is_R2C ? (len / 2 + 1) : len;
}

#endif
