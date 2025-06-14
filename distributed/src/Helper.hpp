#ifndef HELPER_HPP
#define HELPER_HPP

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

#endif
