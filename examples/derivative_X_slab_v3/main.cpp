#include <vector>
#include <iostream>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>
#include "Block.hpp"
#include "Mapping.hpp"
#include "MPI_Helper.hpp"
#include "Extents.hpp"

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
  // using value_type = typename ViewType::non_const_value_type;
  // value_type sum   = 0.0;
  // for (std::size_t i = 0; i < n; i++) {
  //   sum += data[i];
  // }
  // std::cout << label << ": " << sum << std::endl;
  // std::cout << std::resetiosflags(std::ios_base::floatfield);
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
// u = sin(2 * x) + cos(3 * y)
// Data is distributed in X-pencil layout
//
// \tparam RealView1DType Type for 1D grids in real space
// \tparam RealView3DType Type for 3D values in real space
// \tparam ComplexView2DType Type for 2D values in Fourier space
//
// \param[in] rank Rank of the process
// \param[out] x 1D grid in x direction (nx)
// \param[out] y 1D grid in y direction (ny)
// \param[out] ikx 2D grid in Fourier space for x direction (nx/p, ny)
// \param[out] iky 2D grid in Fourier space for y direction (nx/p, ny)
// \param[out] u 3D field in real space (nz, ny/p, nx)
template <typename RealView1DType, typename ComplexView2DType,
          typename RealView3DType>
void initialize(const int rank, const int nprocs, RealView1DType& x,
                RealView1DType& y, ComplexView2DType& ikx,
                ComplexView2DType& iky, RealView3DType& u) {
  using value_type    = typename RealView1DType::non_const_value_type;
  const auto pi       = Kokkos::numbers::pi_v<double>;
  const value_type Lx = 2.0 * pi, Ly = 2.0 * pi;
  const int nx_local = u.extent(2), ny = u.extent(1), nz = u.extent(0);
  const int nx        = x.extent(0);
  const int ny_local  = ikx.extent(0);
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
  for (int iy = 0; iy < ny_local; ++iy) {
    for (int ix = 0; ix < nx / 2 + 1; ++ix) {
      int giy       = iy + rank * ny_local;
      auto tmp_ikx  = z * 2.0 * pi * static_cast<value_type>(ix) / Lx;
      h_ikx(iy, ix) = giy < ny ? tmp_ikx : 0.0;
    }
  }

  for (int iy = 0; iy < ny_local; ++iy) {
    for (int ix = 0; ix < nx / 2 + 1; ++ix) {
      int giy       = iy + rank * ny_local;
      auto tmp_iy   = giy < ny / 2 ? giy : giy - ny;
      auto tmp_iky  = z * 2.0 * pi * static_cast<value_type>(tmp_iy) / Ly;
      h_iky(iy, ix) = giy < ny ? tmp_iky : 0.0;
    }
  }

  // Initialize field
  auto h_u = Kokkos::create_mirror_view(u);
  for (int jz = 0; jz < nz; jz++) {
    for (int jy = 0; jy < ny; jy++) {
      for (int jx = 0; jx < nx_local; jx++) {
        int gjx         = jx + rank * nx_local;
        h_u(jz, jy, jx) = std::sin(2.0 * h_x(gjx)) + std::cos(3.0 * h_y(jy));
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
// \tparam RealView1DType Type for 1D grids in real space
// \tparam RealView3DType Type for 3D values in real space
//
// \param[in] rank Rank of the process
// \param[in] x 1D grid in x direction (nx)
// \param[in] y 1D grid in y direction (ny)
// \param[out] dudxy 3D field of the analytical derivative value (nz, ny/p, nx)
template <typename RealView1DType, typename RealView3DType>
void analytical_solution(const int rank, RealView1DType& x, RealView1DType& y,
                         RealView3DType& dudxy) {
  // Copy grids to host
  auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
  auto h_y = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);

  // Compute the analytical solution on host
  const int nx_local = dudxy.extent(2), ny = dudxy.extent(1),
            nz = dudxy.extent(0);
  auto h_dudxy = Kokkos::create_mirror_view(dudxy);
  for (int iz = 0; iz < nz; iz++) {
    for (int iy = 0; iy < ny; iy++) {
      for (int ix = 0; ix < nx_local; ix++) {
        int gix = ix + rank * nx_local;
        h_dudxy(iz, iy, ix) =
            2.0 * std::cos(2.0 * h_x(gix)) - 3.0 * std::sin(3.0 * h_y(iy));
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
void compute_derivative(const std::size_t nx, const std::size_t ny,
                        const std::size_t nz, double& seconds) {
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
  using RealView4D    = View4D<double>;
  using ComplexView2D = View2D<Kokkos::complex<double>>;
  using ComplexView3D = View3D<Kokkos::complex<double>>;
  using ComplexView4D = View4D<Kokkos::complex<double>>;

  using map_type     = std::array<std::size_t, 3>;
  using extents_type = std::array<std::size_t, 3>;
  using LayoutType   = typename RealView3D::array_layout;

  std::size_t P             = nprocs;
  extents_type in_topology  = {1, 1, P};  // YZ-slab
  extents_type out_topology = {1, P, 1};  // XZ-slab
  map_type src_map          = {0, 1, 2};

  // Declare grids
  RealView1D x("x", nx), y("y", ny);

  // Variables to be transformed
  // Distribute the data in X-pencil layout
  // Global shape (nz, ny, nx) -> (nz, ny, nx/2+1)
  auto [nz_in, ny_in, nx_in] =
      get_local_shape(extents_type({nz, ny, nx}), in_topology, MPI_COMM_WORLD);
  auto [nz_trans, ny_trans, nx_trans] =
      get_local_shape(extents_type({nz, ny, nx}), out_topology, MPI_COMM_WORLD);
  auto [nz_out, ny_out, nx_out] = get_local_shape(
      extents_type({nz, ny, nx / 2 + 1}), in_topology, MPI_COMM_WORLD);
  auto [nz_x, ny_x, nx_x] = get_local_shape(extents_type({nz, ny, nx / 2 + 1}),
                                            out_topology, MPI_COMM_WORLD);

  auto [n3, n2, n1, n0] = get_buffer_extents<LayoutType>(
      extents_type({nz, ny, nx}), in_topology, out_topology);
  auto [m3, m2, m1, m0] = get_buffer_extents<LayoutType>(
      extents_type({nz, ny, nx / 2 + 1}), in_topology, out_topology);

  // Data in YZ-slab layout (nz, ny, nx/px)
  RealView3D u("u", nz_in, ny_in, nx_in), dudxy("dudxy", nz_in, ny_in, nx_in);

  // Data in XZ-slab layout (nz, ny/px, nx)
  RealView3D u_trans("u_trans", nz_trans, ny_trans, nx_trans);
  ComplexView3D Xslab("Xslab", nz_x, ny_x, nx_x);
  ComplexView2D ikx("ikx", ny_x, nx_x), iky("iky", ny_x, nx_x);

  // Data in YZ-slab layout (nz, (nx/2+1)/px, ny)
  ComplexView3D Yslab("Yslab", nz_out, nx_out, ny_out);

  // Buffers used for MPI communications
  ComplexView4D send_buffer("send_buffer", m3, m2, m1, m0);
  ComplexView4D recv_buffer("recv_buffer", m3, m2, m1, m0);
  RealView4D send_buffer_0(reinterpret_cast<double*>(send_buffer.data()), n3,
                           n2, n1, n0);
  RealView4D recv_buffer_0(reinterpret_cast<double*>(recv_buffer.data()), n3,
                           n2, n1, n0);

  initialize(rank, nprocs, x, y, ikx, iky, u);
  analytical_solution(rank, x, y, dudxy);

  execution_space exec;

  // Start computation
  Kokkos::Timer timer;

  // YZ-slab -> XZ-slab transpose
  // (nz, ny, nx/px) -> (nz, ny/px, nx)
  Block block_y2x(exec, u, u_trans, send_buffer_0, recv_buffer_0, src_map, 1,
                  src_map, 2, MPI_COMM_WORLD);
  block_y2x(u, u_trans);

  // Forward transform u -> ux_hat (=RFFT (u))
  KokkosFFT::rfft(exec, u_trans, Xslab);

  int in_axis      = 2;
  int out_axis     = 1;
  map_type dst_map = {0, 2, 1};

  // XZ-slab -> YZ-slab transpose + FFT (along Y direction)
  // (nz, ny/px, nx) -> (nz, nx/px, ny)
  FFTForwardBlock fft_block_x2y(exec, Xslab, Yslab, Yslab, send_buffer,
                                recv_buffer, src_map, in_axis, dst_map,
                                out_axis, MPI_COMM_WORLD);
  fft_block_x2y(Xslab, Yslab);

  // YZ-slab -> XZ-slab transpose
  // (nz, nx/px, ny) -> (nz, ny/px, nx)
  Block block_y2x2(exec, Yslab, Xslab, send_buffer, recv_buffer, dst_map,
                   out_axis, src_map, in_axis, MPI_COMM_WORLD);
  block_y2x2(Yslab, Xslab);

  // MDRanges used in the kernels
  using range2D_type = Kokkos::MDRangePolicy<
      execution_space,
      Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;
  using tile2D_type  = typename range2D_type::tile_type;
  using point2D_type = typename range2D_type::point_type;

  range2D_type range2d(exec, point2D_type{{0, 0}},
                       point2D_type{{iky.extent(0), iky.extent(1)}},
                       tile2D_type{{TILE0, TILE2}});

  // Compute derivatives by multiplications in Fourier space
  Kokkos::parallel_for(
      "ComputeDerivative", range2d, KOKKOS_LAMBDA(const int iy, const int ix) {
        auto ikx_tmp = ikx(iy, ix), iky_tmp = iky(iy, ix);
        for (int iz = 0; iz < nz; ++iz) {
          Xslab(iz, iy, ix) =
              (ikx_tmp * Xslab(iz, iy, ix) + iky_tmp * Xslab(iz, iy, ix));
        }
      });

  // XZ-slab -> YZ-slab transpose
  // (nz, ny/px, nx) -> (nz, nx/px, ny)
  Block block_x2y2(exec, Xslab, Yslab, send_buffer, recv_buffer, src_map,
                   in_axis, dst_map, out_axis, MPI_COMM_WORLD);
  block_x2y2(Xslab, Yslab);

  FFTBackwardBlock fft_block_y2x(exec, Yslab, Yslab, Xslab, send_buffer,
                                 recv_buffer, dst_map, out_axis, src_map,
                                 in_axis, MPI_COMM_WORLD);
  fft_block_y2x(Yslab, Xslab);

  // Do you local backward FFT
  KokkosFFT::irfft(exec, Xslab, u_trans);  // normalization is made here

  // Y -> X transpose
  Block block_x2y(exec, u_trans, u, send_buffer_0, recv_buffer_0, src_map, 2,
                  src_map, 1, MPI_COMM_WORLD);
  block_x2y(u_trans, u);

  exec.fence();
  seconds = timer.seconds();

  // Check results
  auto h_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u);
  auto h_dudxy =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dudxy);

  const double epsilon = 1.e-8;
  for (int iz = 0; iz < h_dudxy.extent(0); ++iz) {
    for (int iy = 0; iy < h_dudxy.extent(1); ++iy) {
      for (int ix = 0; ix < h_dudxy.extent(2); ++ix) {
        auto abs_error = std::abs(h_dudxy(iz, iy, ix) - h_u(iz, iy, ix));
        if (abs_error > epsilon) {
          std::cout << "Error (ix, iy, iz): " << ix << ", " << iy << ", " << iz
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
    const int nx = 16, ny = 16, nz = 2;
    double seconds = 0.0;
    compute_derivative(nx, ny, nz, seconds);
    std::cout << "2D derivative with FFT took: " << seconds << " [s]"
              << std::endl;
  }
  Kokkos::finalize();
  ::MPI_Finalize();

  return 0;
}
