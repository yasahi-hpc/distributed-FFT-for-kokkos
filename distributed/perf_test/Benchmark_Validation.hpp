#ifndef KOKKOSFFT_BENCHMARK_VALIDATION_HPP
#define KOKKOSFFT_BENCHMARK_VALIDATION_HPP

#include <array>
#include <type_traits>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>

namespace KokkosFFT {
namespace Distributed {
namespace Benchmark {

/// \brief Tolerances of the checks of a benchmark: the forward result of a
/// plane wave, relative to its peak, and the round trip of random data in
/// [0, 1), absolute
template <typename T>
struct Tolerances {
  static constexpr double forward   = std::is_same_v<T, float> ? 1e-4 : 1e-10;
  static constexpr double roundtrip = std::is_same_v<T, float> ? 1e-4 : 1e-10;
};

/// \brief The wave numbers of the plane wave of the forward check: 1, 2, 3
/// along the dimensions, below n / 2 for n >= 8, so that the mode -k of a
/// cosine is not stored in the half spectrum of an R2C
inline std::array<std::size_t, 3> plane_wave_modes() { return {1, 2, 3}; }

/// \brief The value of the plane wave at a phase: exp(i phase) (complex) or
/// cos(phase) (real)
template <typename ValueType>
KOKKOS_INLINE_FUNCTION ValueType plane_wave_value(double phase) {
  using float_type = KokkosFFT::Impl::base_floating_point_type<ValueType>;
  if constexpr (KokkosFFT::Impl::is_complex_v<ValueType>) {
    return ValueType(static_cast<float_type>(Kokkos::cos(phase)),
                     static_cast<float_type>(Kokkos::sin(phase)));
  } else {
    return static_cast<float_type>(Kokkos::cos(phase));
  }
}

/// \brief Fill a local input View with a plane wave of the global extents:
/// exp(2 pi i k.x / n) (complex input) or cos(2 pi k.x / n) (real input),
/// x the global index (starts + local index)
template <typename ExecutionSpace, typename ViewType>
void fill_plane_wave(const ExecutionSpace& exec, const ViewType& u,
                     const std::array<std::size_t, 3>& global_extents,
                     const std::array<std::size_t, 3>& starts) {
  using value_type    = typename ViewType::non_const_value_type;
  const auto k        = plane_wave_modes();
  const double two_pi = 2.0 * Kokkos::numbers::pi_v<double>;
  const double c0     = two_pi * static_cast<double>(k[0]) /
                    static_cast<double>(global_extents[0]),
               c1 = two_pi * static_cast<double>(k[1]) /
                    static_cast<double>(global_extents[1]),
               c2 = two_pi * static_cast<double>(k[2]) /
                    static_cast<double>(global_extents[2]);
  const std::size_t s0 = starts[0], s1 = starts[1], s2 = starts[2];
  Kokkos::parallel_for(
      "fill_plane_wave",
      Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<3>>(
          exec, {0, 0, 0}, {u.extent(0), u.extent(1), u.extent(2)}),
      KOKKOS_LAMBDA(std::size_t i0, std::size_t i1, std::size_t i2) {
        const double phase = c0 * static_cast<double>(s0 + i0) +
                             c1 * static_cast<double>(s1 + i1) +
                             c2 * static_cast<double>(s2 + i2);
        u(i0, i1, i2) = plane_wave_value<value_type>(phase);
      });
}

/// \brief The largest error of the forward transform of fill_plane_wave,
/// over every rank of comm, relative to the peak: the transform is n0 n1 n2
/// at the mode k (complex input) or n0 n1 n2 / 2 (real input: the mode -k
/// is not stored), and zero elsewhere
/// \param[in] exec Execution space
/// \param[in] u_hat Local output View of the forward transform
/// \param[in] global_in_extents Global extents of the input
/// \param[in] out_starts Global starts of the local output
/// \param[in] is_real_input Whether the transform is an R2C
/// \param[in] comm MPI communicator
template <typename ExecutionSpace, typename ViewType>
double plane_wave_error(const ExecutionSpace& exec, const ViewType& u_hat,
                        const std::array<std::size_t, 3>& global_in_extents,
                        const std::array<std::size_t, 3>& out_starts,
                        bool is_real_input, MPI_Comm comm) {
  const auto k      = plane_wave_modes();
  const double peak = static_cast<double>(global_in_extents[0]) *
                      static_cast<double>(global_in_extents[1]) *
                      static_cast<double>(global_in_extents[2]) *
                      (is_real_input ? 0.5 : 1.0);
  const std::size_t s0 = out_starts[0], s1 = out_starts[1], s2 = out_starts[2];
  const std::size_t k0 = k[0], k1 = k[1], k2 = k[2];
  double error = 0.0;
  Kokkos::parallel_reduce(
      "plane_wave_error",
      Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<3>>(
          exec, {0, 0, 0}, {u_hat.extent(0), u_hat.extent(1), u_hat.extent(2)}),
      KOKKOS_LAMBDA(std::size_t i0, std::size_t i1, std::size_t i2,
                    double& max_error) {
        const bool is_mode =
            (s0 + i0 == k0) && (s1 + i1 == k1) && (s2 + i2 == k2);
        const auto v = u_hat(i0, i1, i2);
        const double re =
            static_cast<double>(v.real()) - (is_mode ? peak : 0.0);
        const double im = static_cast<double>(v.imag());
        const double e  = Kokkos::sqrt(re * re + im * im);
        if (e > max_error || e != e) {
          // NaN as infinity, so that it fails the check on every rank
          max_error = (e != e) ? Kokkos::Experimental::infinity_v<double> : e;
        }
      },
      Kokkos::Max<double>(error));
  double global_error = 0.0;
  ::MPI_Allreduce(&error, &global_error, 1, MPI_DOUBLE, MPI_MAX, comm);
  return global_error / peak;
}

/// \brief Fill a local View with random numbers in [0, 1), different on
/// each rank
template <typename ExecutionSpace, typename ViewType>
void fill_random(const ExecutionSpace& exec, const ViewType& u, int rank) {
  Kokkos::Random_XorShift64_Pool<ExecutionSpace> pool(12345 + rank);
  Kokkos::fill_random(exec, u, pool, 1.0);
}

/// \brief The largest absolute difference between two local Views of the
/// same extents, over every rank of comm (NaN as infinity)
template <typename ExecutionSpace, typename ViewType, typename RefViewType>
double max_difference(const ExecutionSpace& exec, const ViewType& a,
                      const RefViewType& b, MPI_Comm comm) {
  double error = 0.0;
  Kokkos::parallel_reduce(
      "max_difference",
      Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<3>>(
          exec, {0, 0, 0}, {a.extent(0), a.extent(1), a.extent(2)}),
      KOKKOS_LAMBDA(std::size_t i0, std::size_t i1, std::size_t i2,
                    double& max_error) {
        const double e =
            static_cast<double>(Kokkos::abs(a(i0, i1, i2) - b(i0, i1, i2)));
        if (e > max_error || e != e) {
          max_error = (e != e) ? Kokkos::Experimental::infinity_v<double> : e;
        }
      },
      Kokkos::Max<double>(error));
  double global_error = 0.0;
  ::MPI_Allreduce(&error, &global_error, 1, MPI_DOUBLE, MPI_MAX, comm);
  return global_error;
}

}  // namespace Benchmark
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
