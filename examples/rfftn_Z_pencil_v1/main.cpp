#include <vector>
#include <iostream>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>
#include "Helper.hpp"
#include "Block.hpp"

using execution_space = Kokkos::DefaultExecutionSpace;
template <typename T>
using View1D = Kokkos::View<T*, Kokkos::LayoutRight, execution_space>;
template <typename T>
using View2D = Kokkos::View<T**, Kokkos::LayoutRight, execution_space>;
template <typename T>
using View3D = Kokkos::View<T***, Kokkos::LayoutRight, execution_space>;
template <typename T>
using View4D = Kokkos::View<T****, Kokkos::LayoutRight, execution_space>;
template <typename T>
using View5D = Kokkos::View<T*****, Kokkos::LayoutRight, execution_space>;

void distributed_fft();

void distributed_fft() {
  int rank, nprocs;
  ::MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // 1) Define your global dims and process grid
  const int nx = 128, ny = 128, nz = 128;
  int dims[2] = {0, 0};
  ::MPI_Dims_create(nprocs, 2, dims);  // choose Px × Py
  int periods[2] = {1, 1};             // Periodic in all directions
  ::MPI_Comm cart_comm;
  ::MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, /*reorder=*/1,
                    &cart_comm);

  // get my coords (px, py)
  int coords[2];
  ::MPI_Cart_coords(cart_comm, rank, 2, coords);

  // split into row‐ and col‐ communicators
  ::MPI_Comm row_comm, col_comm;

  int remain_dims[2];

  // keep Y‐axis for row_comm (all procs with same px)
  remain_dims[0] = 1;
  remain_dims[1] = 0;
  ::MPI_Cart_sub(cart_comm, remain_dims, &row_comm);

  // keep X‐axis for col_comm (all procs with same py)
  remain_dims[0] = 0;
  remain_dims[1] = 1;
  ::MPI_Cart_sub(cart_comm, remain_dims, &col_comm);

  // 2) Local block sizes for Z‐pencils
  int Px        = dims[0];
  int Py        = dims[1];
  int nx_local  = nx / Px;
  int ny_local  = ny / Py;
  int nz_local  = ((nz / 2 + 1) - 1) / Px + 1;  // slice Z across Py
  int nx_local2 = nx / Py;
  int nbatch    = 5;
  int nzh       = nz / 2 + 1;  // half size in X for real-to-complex FFT

  using ComplexView3D = View3D<Kokkos::complex<double>>;
  using ComplexView4D = View4D<Kokkos::complex<double>>;
  using ComplexView5D = View5D<Kokkos::complex<double>>;
  using RealView3D    = View3D<double>;
  using RealView4D    = View4D<double>;

  // Start: Z-Pencil, End: Y-Pencil
  RealView4D in("in", nx_local, ny_local, nz, nbatch),
      in_ref("in_ref", nx_local, ny_local, nz, nbatch);
  ComplexView4D out("out", nx_local, ny_local, nzh, nbatch);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  const double range = 1.0;
  execution_space exec;
  Kokkos::fill_random(exec, in, random_pool, range);
  Kokkos::deep_copy(in_ref, in);

  // FFT on Z-pencil
  // do your local 1D FFTs along Z:
  KokkosFFT::rfft(exec, in, out, KokkosFFT::Normalization::backward, 2);

  // --- First transpose: Z‐pencils -> X‐pencils ---
  // (Nx/px, Ny/py, Nz) -> (Nz/px, Ny/py, Nx)
  // a) prepare for all‐to‐all in the row_comm

  // We reuse these buffers for each all to all communications
  ComplexView5D send_z2x("send_z2x", Px, nx_local, ny_local, nz_local, nbatch);
  ComplexView5D recv_z2x("recv_z2x", Px, nx_local, ny_local, nz_local, nbatch);

  using map_type   = std::array<std::size_t, 4>;
  map_type src_map = {0, 1, 2, 3},
           dst_map = {1, 2, 3, 0};  // Z-pencil to X-pencil

  // (x, y, gz, b) -> (y, z, b, gx)
  ComplexView4D in_Xpencil("in_Xpencil", ny_local, nz_local, nbatch, nx);

  Block block_z2x(exec, out, in_Xpencil, send_z2x, recv_z2x, src_map, 2,
                  dst_map, 0, col_comm);
  block_z2x(out, in_Xpencil);

  // do your local 1D FFTs along X:
  KokkosFFT::fft(exec, in_Xpencil, in_Xpencil,
                 KokkosFFT::Normalization::backward, -1);

  // --- Second transpose: X‐pencils -> Y‐pencils ---
  // (Nz/px, Ny/py, Nx) -> (Nz/px, Nx/py, Ny)
  // a) prepare for all-to-all in the col_comm
  // (y, z, b, gx) -> (z, b, x, gy)
  ComplexView4D in_Ypencil("in_Ypencil", nz_local, nbatch, nx_local2, ny);

  ComplexView5D send_x2y(send_z2x.data(), Px, nx_local, ny_local, nz_local,
                         nbatch);
  ComplexView5D recv_x2y(recv_z2x.data(), Px, nx_local, ny_local, nz_local,
                         nbatch);

  map_type ypencil_map = {2, 3, 0, 1};

  Block block_x2y(exec, in_Xpencil, in_Ypencil, send_x2y, recv_x2y, dst_map, 0,
                  ypencil_map, 1, row_comm);
  block_x2y(in_Xpencil, in_Ypencil);

  // do your local 1D FFTs along Y:
  KokkosFFT::fft(exec, in_Ypencil, in_Ypencil,
                 KokkosFFT::Normalization::backward, -1);

  // Now, we will start the backward transforms
  KokkosFFT::ifft(exec, in_Ypencil, in_Ypencil,
                  KokkosFFT::Normalization::backward, -1);

  Block block_y2x(exec, in_Ypencil, in_Xpencil, send_x2y, recv_x2y, ypencil_map,
                  1, dst_map, 0, row_comm);
  block_y2x(in_Ypencil, in_Xpencil);

  // Do your local 1D FFTs along X:
  KokkosFFT::ifft(exec, in_Xpencil, in_Xpencil,
                  KokkosFFT::Normalization::backward, -1);

  Block block_x2z(exec, in_Xpencil, out, send_z2x, recv_z2x, dst_map, 0,
                  src_map, 2, col_comm);
  block_x2z(in_Xpencil, out);

  // do your local 1D FFTs along Z:
  KokkosFFT::irfft(exec, out, in, KokkosFFT::Normalization::backward, 2);
  exec.fence();

  // Check results
  auto h_in = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), in);
  auto h_in_ref =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), in_ref);
  const double epsilon = 1.e-8;
  for (int ib = 0; ib < h_in.extent(3); ++ib) {
    for (int iz = 0; iz < h_in.extent(2); ++iz) {
      for (int iy = 0; iy < h_in.extent(1); ++iy) {
        for (int ix = 0; ix < h_in.extent(0); ++ix) {
          if (Kokkos::abs(h_in(ix, iy, iz, ib)) <= epsilon) continue;
          auto relative_error =
              Kokkos::abs(h_in(ix, iy, iz, ib) - h_in_ref(ix, iy, iz, ib)) /
              Kokkos::abs(h_in(ix, iy, iz, ib));
          if (relative_error > epsilon) {
            std::cout << "Error (ix, iy, iz, ib): " << ix << ", " << iy << ", "
                      << iz << " @ rank" << rank << ", " << h_in(ix, iy, iz, ib)
                      << " != " << h_in_ref(ix, iy, iz, ib) << std::endl;
            return;
          }
        }
      }
    }
  }

  if (rank == 0) {
    std::cout << "Distributed Z-pencil rFFT v1 completed successfully!"
              << std::endl;
  }

  ::MPI_Comm_free(&row_comm);
  ::MPI_Comm_free(&col_comm);
  ::MPI_Comm_free(&cart_comm);
}

int main(int argc, char** argv) {
  ::MPI_Init(&argc, &argv);

  Kokkos::initialize(argc, argv);
  { distributed_fft(); }
  Kokkos::finalize();
  ::MPI_Finalize();

  return 0;
}
