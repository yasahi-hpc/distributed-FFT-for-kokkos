#include <vector>
#include <iostream>
#include <memory>
#include <iomanip>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>
#include "io_utils.hpp"

#include <iomanip>

template <typename ViewType>
void display(ViewType& a, int rank) {
  auto label   = a.label() + std::to_string(rank);
  const auto n = a.size();

  auto h_a = Kokkos::create_mirror_view(a);
  Kokkos::deep_copy(h_a, a);
  auto* data = h_a.data();

  std::cout << std::scientific << std::setprecision(16) << std::flush;
  for (std::size_t i = 0; i < n; i++) {
    std::cout << label + "[" << i << "]: " << i << ", " << data[i] << std::endl;
  }
}

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL)
constexpr int TILE0 = 4;
constexpr int TILE1 = 4;
constexpr int TILE2 = 32;
#else
constexpr int TILE0 = 4;
constexpr int TILE1 = 4;
constexpr int TILE2 = 4;
#endif

using execution_space = Kokkos::DefaultExecutionSpace;
template <typename T>
using View1D = Kokkos::View<T*, Kokkos::LayoutRight, execution_space>;
template <typename T>
using View2D = Kokkos::View<T**, Kokkos::LayoutRight, execution_space>;
template <typename T>
using View3D = Kokkos::View<T***, Kokkos::LayoutRight, execution_space>;
template <typename T>
using View4D = Kokkos::View<T****, Kokkos::LayoutRight, execution_space>;

void compute_derivative(const std::size_t nx, const std::size_t ny,
                        const std::size_t nz, double& seconds);

// \brief Initialize the grid, wavenumbers, and the test function values
// u = sin(2 * x) + cos(3 * y) + sin(4 * z)
// Data is distributed in XZ-slab -> XY-slab
//
// \tparam RealView1DType Type for 1D grids in real space
// \tparam RealView3DType Type for 3D values in real space
// \tparam ComplexView2DType Type for 2D values in Fourier space
//
// \param[in] rank Rank of the process
// \param[out] x 1D grid in x direction (nx)
// \param[out] y 1D grid in y direction (ny)
// \param[out] z 1D grid in y direction (nz)
// \param[out] ikx 3D grid in Fourier space for x direction (nx/2+1)
// \param[out] iky 3D grid in Fourier space for y direction (ny/p)
// \param[out] ikz 3D grid in Fourier space for z direction (nz)
// \param[out] u 3D field in real space (nz, ny/p, nx)
template <typename RealView1DType, typename ComplexView1DType,
          typename RealView3DType>
void initialize(const RealView1DType& x, const RealView1DType& y, const RealView1DType& z, 
                const ComplexView1DType& ikx, const ComplexView1DType& iky, const ComplexView1DType& ikz, 
                const RealView3DType& u) {
  using value_type    = typename RealView1DType::non_const_value_type;
  const auto pi       = Kokkos::numbers::pi_v<double>;
  const value_type Lx = 2.0 * pi, Ly = 2.0 * pi, Lz = 2.0 * pi;
  const int nx = u.extent(2), ny = u.extent(1), nz = u.extent(0);
  const int nxh = ikx.extent(0);
  const value_type dx = Lx / static_cast<value_type>(nx),
                   dy = Ly / static_cast<value_type>(ny),
                   dz = Lz / static_cast<value_type>(nz);

  // Initialize grids
  auto h_x = Kokkos::create_mirror_view(x);
  auto h_y = Kokkos::create_mirror_view(y);
  auto h_z = Kokkos::create_mirror_view(z);
  for (int ix = 0; ix < nx; ++ix) h_x(ix) = static_cast<value_type>(ix) * dx;
  for (int iy = 0; iy < ny; ++iy) h_y(iy) = static_cast<value_type>(iy) * dy;
  for (int iz = 0; iz < nz; ++iz) h_z(iz) = static_cast<value_type>(iz) * dz;

  // Initialize wave numbers
  const Kokkos::complex<value_type> z0(0.0, 1.0);  // Imaginary unit
  auto h_ikx = Kokkos::create_mirror_view(ikx);
  auto h_iky = Kokkos::create_mirror_view(iky);
  auto h_ikz = Kokkos::create_mirror_view(ikz);

  for (int ix = 0; ix < nxh; ++ix) {
    auto tmp_ikx  = z0 * 2.0 * pi * static_cast<value_type>(ix) / Lx;
    h_ikx(ix) = tmp_ikx;
  }

  for (int iy = 0; iy < ny; ++iy) {
    auto tmp_iy   = iy < ny / 2 ? iy : iy - ny;
    auto tmp_iky  = z0 * 2.0 * pi * static_cast<value_type>(tmp_iy) / Ly;
    h_iky(iy) = tmp_iky;
  }

  for (int iz = 0; iz < nz; ++iz) {
    auto tmp_iz   = iz < nz / 2 ? iz : iz - nz;
    auto tmp_ikz  = z0 * 2.0 * pi * static_cast<value_type>(tmp_iz) / Lz;
    h_ikz(iz) = tmp_ikz;
  }

  // Initialize field
  auto h_u = Kokkos::create_mirror_view(u);
  for (int jz = 0; jz < nz; jz++) {
    for (int jy = 0; jy < ny; jy++) {
      for (int jx = 0; jx < nx; jx++) {
        h_u(jz, jy, jx) = std::sin(2.0 * h_x(jx)) + std::cos(3.0 * h_y(jy)) + std::sin(4 * h_z(jz));
      }
    }
  }

  Kokkos::deep_copy(x, h_x);
  Kokkos::deep_copy(y, h_y);
  Kokkos::deep_copy(z, h_z);
  Kokkos::deep_copy(ikx, h_ikx);
  Kokkos::deep_copy(iky, h_iky);
  Kokkos::deep_copy(ikz, h_ikz);
  Kokkos::deep_copy(u, h_u);
}

// \brief Compute analytical solution of the derivative
// du/dx + du/dy + du/dz = 2 * cos(2 * x) - 3 * sin(3 * y) + 4 * cos(4 * z)
// Data is distributed in X-pencil layout
//
// \tparam RealView1DType Type for 1D grids in real space
// \tparam RealView3DType Type for 3D values in real space
//
// \param[in] rank Rank of the process
// \param[in] x 1D grid in x direction (nx)
// \param[in] y 1D grid in y direction (ny)
// \param[in] z 1D grid in z direction (nz)
// \param[out] dudxy 3D field of the analytical derivative value (nz, ny/p, nx)
template <typename RealView1DType, typename RealView3DType>
void analytical_solution(const RealView1DType& x, const RealView1DType& y,
                         const RealView1DType& z, const RealView3DType& dudxy) {
  // Copy grids to host
  auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
  auto h_y = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
  auto h_z = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), z);

  // Compute the analytical solution on host
  const int nx = dudxy.extent(2), ny = dudxy.extent(1),
            nz = dudxy.extent(0);
  auto h_dudxy = Kokkos::create_mirror_view(dudxy);
  for (int iz = 0; iz < nz; iz++) {
    for (int iy = 0; iy < ny; iy++) {
      for (int ix = 0; ix < nx; ix++) {
        h_dudxy(iz, iy, ix) =
            2.0 * std::cos(2.0 * h_x(ix)) - 3.0 * std::sin(3.0 * h_y(iy)) + 4.0 * std::cos(4.0 * h_z(iz));
      }
    }
  }

  Kokkos::deep_copy(dudxy, h_dudxy);
}

// \brief Compute the derivative of a function using FFT-based methods and
// compare with the analytical solution
// \param[in] nx Number of grid points in the x-direction
// \param[in] ny Number of grid points in the y-direction
// \param[in] nz Number of grid points in the z-direction
// \param[out] seconds Time taken to compute the derivatives (in seconds)
void compute_derivative(const int nx, const int ny,
                        const int nz, double& seconds) {
  // View types
  using RealView1D    = View1D<double>;
  using RealView3D    = View3D<double>;
  using ComplexView1D = View1D<Kokkos::complex<double>>;
  using ComplexView3D = View3D<Kokkos::complex<double>>;

  // Declare grids
  RealView1D x("x", nx), y("y", ny), z("z", nz);
  ComplexView1D ikx("ikx", nx / 2 + 1), iky("iky", ny), ikz("ikz", nz);

  // Variables to be transformed
  RealView3D u("u", nz, ny, nx), dudxy("dudxy", nz, ny, nx);
  ComplexView3D u_hat("u_hat", nz, ny, nx / 2 + 1);

  initialize(x, y, z, ikx, iky, ikz, u);
  analytical_solution(x, y, z, dudxy);

  // MDRanges used in the kernels
  using range3D_type = Kokkos::MDRangePolicy<
      execution_space,
      Kokkos::Rank<3, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;
  using tile3D_type  = typename range3D_type::tile_type;
  using point3D_type = typename range3D_type::point_type;

  execution_space exec;

  // kokkos-fft plans
  KokkosFFT::Plan r2c_plan(exec, u, u_hat, KokkosFFT::Direction::forward,
                           KokkosFFT::axis_type<3>({-3, -2, -1}));
  KokkosFFT::Plan c2r_plan(exec, u_hat, u, KokkosFFT::Direction::backward,
                           KokkosFFT::axis_type<3>({-3, -2, -1}));

  // Start computation
  Kokkos::Timer timer;

  // Forward transform u -> u_hat (=FFT (u))
  KokkosFFT::execute(r2c_plan, u, u_hat);

  range3D_type range3d(exec, point3D_type{{0, 0, 0}},
                       point3D_type{{nz, ny, nx / 2 + 1}},
                       tile3D_type{{TILE0, TILE1, TILE2}});

  // Compute derivatives by multiplications in Fourier space
  Kokkos::parallel_for(
      "ComputeDerivative", range3d, KOKKOS_LAMBDA(const int iz, const int iy, const int ix) {
        auto ikx_tmp = ikx(ix), iky_tmp = iky(iy), ikz_tmp = ikz(iz);
        u_hat(iz, iy, ix) =
              (ikx_tmp * u_hat(iz, iy, ix) + iky_tmp * u_hat(iz, iy, ix) + ikz_tmp * u_hat(iz, iy, ix));
      });

  // Backward transform u_hat -> u (=IFFT (u_hat))
  KokkosFFT::execute(c2r_plan, u_hat, u);  // normalization is made here
  exec.fence();

  seconds = timer.seconds();

  // Check results
  auto h_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u);
  auto h_dudxy =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dudxy);

  const double epsilon = 1.e-8;
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        if (std::abs(h_dudxy(iz, iy, ix)) <= epsilon) continue;
        auto relative_error = std::abs(h_dudxy(iz, iy, ix) - h_u(iz, iy, ix)) /
                              std::abs(h_dudxy(iz, iy, ix));
        if (relative_error > epsilon) {
          std::cerr << "Error: " << h_dudxy(iz, iy, ix)
                    << " != " << h_u(iz, iy, ix) << std::endl;
          return;
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const int nx = 16, ny = 16, nz = 16;
    double seconds = 0.0;
    compute_derivative(nx, ny, nz, seconds);
    std::cout << "3D derivative with FFT took: " << seconds << " [s]"
              << std::endl;
  }
  Kokkos::finalize();

  return 0;
}
