#include <vector>
#include <iostream>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>
#include "PackUnpack.hpp"
#include "All2All.hpp"

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

void compute_derivative(const int nx, const int ny, const int nz,
                        double& seconds);

// \brief Initialize the grid, wavenumbers, and the test function values
// u = sin(2 * x) + cos(3 * y)
// Data is distributed in X-pencil layout
//
// \tparam RealView1DType: Type for 1D grids in real space
// \tparam RealView3DType: Type for 3D values in real space
// \tparam ComplexView2DType: Type for 2D values in Fourier space
//
// \param rank [in]: Rank of the process
// \param x [out]: 1D grid in x direction
// \param y [out]: 1D grid in y direction
// \param ikx [out]: 2D grid in Fourier space for x direction
// \param iky [out]: 2D grid in Fourier space for y direction
// \param u [out]: 3D field in real space
template <typename RealView1DType, typename ComplexView2DType,
          typename RealView3DType>
void initialize(const int rank, RealView1DType& x, RealView1DType& y,
                ComplexView2DType& ikx, ComplexView2DType& iky,
                RealView3DType& u) {
  using value_type    = typename RealView1DType::non_const_value_type;
  const auto pi       = Kokkos::numbers::pi_v<double>;
  const value_type Lx = 2.0 * pi, Ly = 2.0 * pi;
  const int nx = u.extent(2), ny_local = u.extent(1), nz = u.extent(0);
  const int ny        = y.extent(0);
  const int nx_local  = ikx.extent(0);
  const value_type dx = Lx / static_cast<value_type>(nx),
                   dy = Ly / static_cast<value_type>(ny);

  // Initialize grids
  auto h_x = Kokkos::create_mirror_view(x);
  auto h_y = Kokkos::create_mirror_view(y);
  for (int ix = 0; ix < nx; ++ix) h_x(ix) = static_cast<value_type>(ix) * dx;
  for (int iy = 0; iy < ny; ++iy) h_y(iy) = static_cast<value_type>(iy) * dy;

  // Initialize wave numbers
  const Kokkos::complex<value_type> z(0.0, 1.0);  // Imaginary unit
  auto h_ikx = Kokkos::create_mirror_view(ikx);
  auto h_iky = Kokkos::create_mirror_view(iky);
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx_local; ++ix) {
      int gix       = ix + rank * nx_local;
      auto tmp_ikx  = z * 2.0 * pi * static_cast<value_type>(gix) / Lx;
      h_ikx(ix, iy) = gix < (nx / 2 + 1) ? tmp_ikx : 0.0;
    }
  }

  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx_local; ++ix) {
      int gix       = ix + rank * nx_local;
      auto tmp_iy   = iy < ny / 2 ? iy : iy - ny;
      auto tmp_iky  = z * 2.0 * pi * static_cast<value_type>(tmp_iy) / Ly;
      h_iky(ix, iy) = gix < (nx / 2 + 1) ? tmp_iky : 0.0;
    }
  }

  // Initialize field
  auto h_u = Kokkos::create_mirror_view(u);
  for (int jz = 0; jz < nz; jz++) {
    for (int jy = 0; jy < ny_local; jy++) {
      for (int jx = 0; jx < nx; jx++) {
        int gjy         = jy + rank * ny_local;
        h_u(jz, jy, jx) = std::sin(2.0 * h_x(jx)) + std::cos(3.0 * h_y(gjy));
      }
    }
  }

  Kokkos::deep_copy(x, h_x);
  Kokkos::deep_copy(y, h_y);
  Kokkos::deep_copy(ikx, h_ikx);
  Kokkos::deep_copy(iky, h_iky);
  Kokkos::deep_copy(u, h_u);
}

// \brief Compute analytical solution of the derivative
// du/dx + du/dy = 2 * cos(2 * x) - 3 * sin(3 * y)
// Data is distributed in X-pencil layout
//
// \tparam RealView1DType: Type for 1D grids in real space
// \tparam RealView3DType: Type for 3D values in real space
//
// \param rank [in]: Rank of the process
// \param x [in]: 1D grid in x direction
// \param y [in]: 1D grid in y direction
// \param dudxy [out]: 3D field of the analytical derivative value
template <typename RealView1DType, typename RealView3DType>
void analytical_solution(const int rank, RealView1DType& x, RealView1DType& y,
                         RealView3DType& dudxy) {
  // Copy grids to host
  auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
  auto h_y = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);

  // Compute the analytical solution on host
  const int nx = dudxy.extent(2), ny_local = dudxy.extent(1),
            nz = dudxy.extent(0);
  auto h_dudxy = Kokkos::create_mirror_view(dudxy);
  for (int iz = 0; iz < nz; iz++) {
    for (int iy = 0; iy < ny_local; iy++) {
      for (int ix = 0; ix < nx; ix++) {
        int giy = iy + rank * ny_local;
        h_dudxy(iz, iy, ix) =
            2.0 * std::cos(2.0 * h_x(ix)) - 3.0 * std::sin(3.0 * h_y(giy));
      }
    }
  }

  Kokkos::deep_copy(dudxy, h_dudxy);
}

// \brief Compute the derivative of a function using FFT-based methods and
// compare with the analytical solution
// \param nx [in]: Number of grid points in the x-direction
// \param ny [in]: Number of grid points in the y-direction
// \param nz [in]: Number of grid points in the z-direction
// \param seconds [out]: Time taken to compute the derivatives (in seconds)
void compute_derivative(const int nx, const int ny, const int nz,
                        double& seconds) {
  int rank, nprocs;
  ::MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // Check ny is divisible by nprocs
  if (ny % nprocs != 0) {
    std::cerr << "Error: ny must be divisible by nprocs" << std::endl;
    return;
  }

  // View types
  using RealView1D    = View1D<double>;
  using RealView3D    = View3D<double>;
  using ComplexView2D = View2D<Kokkos::complex<double>>;
  using ComplexView3D = View3D<Kokkos::complex<double>>;
  using ComplexView4D = View4D<Kokkos::complex<double>>;

  // Declare grids
  RealView1D x("x", nx), y("y", ny);

  // Variables to be transformed
  // Distribute the data in X-pencil layout
  int ny_local = ny / nprocs;

  RealView3D u("u", nz, ny_local, nx), dudxy("dudxy", nz, ny_local, nx);
  ComplexView3D ux_hat("ux_hat", nz, ny_local, nx / 2 + 1);

  // After transpose the data will be in Y-pencil layout
  // Also y direction is placed in the innermost dimension
  int nx_local = ((nx / 2 + 1) - 1) / nprocs + 1;
  ComplexView3D uxy_hat("uxy_hat", nz, nx_local, ny);
  ComplexView2D ikx("ikx", nx_local, ny), iky("iky", nx_local, ny);

  // Buffers used for MPI communications
  ComplexView4D send_buffer("send_buffer", nprocs, nz, ny_local, nx_local);
  ComplexView4D recv_buffer("recv_buffer", nprocs, nz, ny_local, nx_local);

  initialize(rank, x, y, ikx, iky, u);
  analytical_solution(rank, x, y, dudxy);

  execution_space exec;

  // kokkos-fft plans on X-pencil layout
  KokkosFFT::Plan r2c_plan(exec, u, ux_hat, KokkosFFT::Direction::forward, -1);
  KokkosFFT::Plan c2r_plan(exec, ux_hat, u, KokkosFFT::Direction::backward, -1);

  // kokkos-fft plans on Y-pencil layout using in-place transform
  KokkosFFT::Plan c2c_forward_plan(exec, uxy_hat, uxy_hat,
                                   KokkosFFT::Direction::forward, -1);
  KokkosFFT::Plan c2c_backward_plan(exec, uxy_hat, uxy_hat,
                                    KokkosFFT::Direction::backward, -1);

  // Start computation
  Kokkos::Timer timer;

  // Forward transform u -> ux_hat (=RFFT (u))
  KokkosFFT::execute(r2c_plan, u, ux_hat);
  using map_type = std::array<std::size_t, 3>;

  // (z, y, gx) -> (p, z, y, x) (LayoutRight)
  map_type src_xpencil_map = {0, 1, 2};
  pack(exec, ux_hat, send_buffer, src_xpencil_map, 2);
  exec.fence();
  All2All<execution_space, ComplexView4D> all2all(send_buffer, recv_buffer);
  all2all(send_buffer, recv_buffer);
  unpack(exec, recv_buffer, uxy_hat, src_xpencil_map, 1);

  // Forward transform uxy_hat -> uxy_hat (=FFT (ux_hat))
  KokkosFFT::execute(c2c_forward_plan, uxy_hat, uxy_hat);

  // MDRanges used in the kernels
  using range2D_type = Kokkos::MDRangePolicy<
      execution_space,
      Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;
  using tile2D_type  = typename range2D_type::tile_type;
  using point2D_type = typename range2D_type::point_type;

  range2D_type range2d(exec, point2D_type{{0, 0}}, point2D_type{{nx_local, ny}},
                       tile2D_type{{TILE0, TILE2}});

  // Compute derivatives by multiplications in Fourier space
  Kokkos::parallel_for(
      "ComputeDerivative", range2d, KOKKOS_LAMBDA(const int ix, const int iy) {
        auto ikx_tmp = ikx(ix, iy), iky_tmp = iky(ix, iy);
        for (int iz = 0; iz < nz; ++iz) {
          uxy_hat(iz, ix, iy) =
              (ikx_tmp * uxy_hat(iz, ix, iy) + iky_tmp * uxy_hat(iz, ix, iy));
        }
      });

  // Backward transform uxy_hat -> uxy_hat (=IFFT (u_hat))
  KokkosFFT::execute(c2c_backward_plan, uxy_hat,
                     uxy_hat);  // normalization is made here

  // Pack data into send buffer
  pack(exec, uxy_hat, send_buffer, src_xpencil_map, 1);
  exec.fence();
  // All2All<execution_space, ComplexView4D> all2all(send_buffer, recv_buffer);
  all2all(send_buffer, recv_buffer);

  unpack(exec, recv_buffer, ux_hat, src_xpencil_map, 2);

  // Backward transform ux_hat -> u (=IFFT (u_hat))
  KokkosFFT::execute(c2r_plan, ux_hat, u);  // normalization is made here

  exec.fence();
  seconds = timer.seconds();

  // Check results
  auto h_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u);
  auto h_dudxy =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dudxy);

  const double epsilon = 1.e-8;
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny_local; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        if (std::abs(h_dudxy(iz, iy, ix)) <= epsilon) continue;
        auto relative_error = std::abs(h_dudxy(iz, iy, ix) - h_u(iz, iy, ix)) /
                              std::abs(h_dudxy(iz, iy, ix));
        if (relative_error > epsilon) {
          std::cerr << "Error (ix, iy, iz): " << ix << ", " << iy << ", " << iz
                    << " @ rank" << rank << ", " << h_dudxy(iz, iy, ix)
                    << " != " << h_u(iz, iy, ix) << std::endl;
          return;
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  ::MPI_Init(&argc, &argv);

  Kokkos::initialize(argc, argv);
  {
    const int nx = 8, ny = 8, nz = 2;
    double seconds = 0.0;
    compute_derivative(nx, ny, nz, seconds);
    std::cout << "2D derivative with FFT took: " << seconds << " [s]"
              << std::endl;
  }
  Kokkos::finalize();
  ::MPI_Finalize();

  return 0;
}
