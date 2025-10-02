#ifndef KOKKOSFFT_DISTRIBUTED_HELPER_HPP
#define KOKKOSFFT_DISTRIBUTED_HELPER_HPP

#include <Kokkos_Core.hpp>
#include <iomanip>
#include <sstream>
#include <iostream>

template <typename ViewType>
void display_sum(ViewType& a) {
  auto label   = a.label();
  const auto n = a.size();

  auto h_a = Kokkos::create_mirror_view(a);
  Kokkos::deep_copy(h_a, a);
  auto* data       = h_a.data();
  using value_type = typename ViewType::non_const_value_type;
  value_type sum   = 0;
  for (std::size_t i = 0; i < n; i++) {
    sum += data[i];
  }

  std::stringstream ss;
  ss << std::scientific << std::setprecision(16) << label << ", sum: " << sum;
  std::cout << ss.str() << std::endl;
  std::cout << std::resetiosflags(std::ios_base::floatfield);
}

template <typename ViewType>
void nd_display(ViewType& a) {
  auto label = a.label();

  using value_type      = typename ViewType::non_const_value_type;
  using execution_space = typename ViewType::execution_space;
  using elem_type =
      KokkosFFT::Impl::add_pointer_n_t<value_type, ViewType::rank()>;

  auto extents = KokkosFFT::Impl::extract_extents(a);

  Kokkos::View<elem_type, Kokkos::LayoutRight, execution_space> a_contiguous(
      "a", KokkosFFT::Impl::create_layout<Kokkos::LayoutRight>(extents));
  Kokkos::deep_copy(a_contiguous, a);

  auto h_a = Kokkos::create_mirror_view(a_contiguous);
  Kokkos::deep_copy(h_a, a_contiguous);

  std::cout << std::scientific << std::setprecision(16) << std::flush;

  if constexpr (ViewType::rank == 1) {
    for (std::size_t i = 0; i < a.extent(0); i++) {
      std::cout << label + "(" << i << "): " << h_a(i) << std::endl;
    }
    std::cout << std::resetiosflags(std::ios_base::floatfield);
  } else if constexpr (ViewType::rank == 2) {
    for (std::size_t i = 0; i < a.extent(0); i++) {
      for (std::size_t j = 0; j < a.extent(1); j++) {
        std::cout << label + "(" << i << ", " << j << "): " << h_a(i, j)
                  << std::endl;
      }
    }
    std::cout << std::resetiosflags(std::ios_base::floatfield);
  } else if constexpr (ViewType::rank == 3) {
    for (std::size_t i = 0; i < a.extent(0); i++) {
      for (std::size_t j = 0; j < a.extent(1); j++) {
        for (std::size_t k = 0; k < a.extent(2); k++) {
          std::cout << label + "(" << i << ", " << j << ", " << k
                    << "): " << h_a(i, j, k) << std::endl;
        }
      }
    }
    std::cout << std::resetiosflags(std::ios_base::floatfield);
  } else if constexpr (ViewType::rank == 4) {
    for (std::size_t i = 0; i < a.extent(0); i++) {
      for (std::size_t j = 0; j < a.extent(1); j++) {
        for (std::size_t k = 0; k < a.extent(2); k++) {
          for (std::size_t l = 0; l < a.extent(3); l++) {
            std::cout << label + "(" << i << ", " << j << ", " << k << ", " << l
                      << "): " << h_a(i, j, k, l) << std::endl;
          }
        }
      }
    }
    std::cout << std::resetiosflags(std::ios_base::floatfield);
  } else {
    std::cerr << "Unsupported rank: " << ViewType::rank
              << ". Only 1D, 2D, 3D, and 4D views are supported." << std::endl;
  }
}

#endif
