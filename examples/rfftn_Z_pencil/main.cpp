#include <vector>
#include <iostream>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>
#include "Helper.hpp"

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
  ComplexView4D out("out", nx_local, ny_local, nz / 2 + 1, nbatch);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  const double range = 1.0;
  execution_space exec;
  Kokkos::fill_random(exec, in, random_pool, range);
  Kokkos::deep_copy(in_ref, in);

  // FFT on Z-pencil
  // do your local 1D FFTs along Z:
  KokkosFFT::rfft(exec, in, out, KokkosFFT::Normalization::backward, 2);

  std::cout << "in.extents: " << in.extent(0) << ", " << in.extent(1) << ", "
            << in.extent(2) << ", " << in.extent(3) << std::endl;
  std::cout << "out.extents: " << out.extent(0) << ", " << out.extent(1) << ", "
            << out.extent(2) << ", " << out.extent(3) << std::endl;

  // --- First transpose: Z‐pencils -> X‐pencils ---
  // (Nx/px, Ny/py, Nz) -> (Nz/px, Ny/py, Nx)
  // a) prepare for all‐to‐all in the row_comm
  int send_count =
      nx_local * ny_local * nz_local * nbatch;  // each recv/send block size

  // We reuse these buffers for each all to all communications
  ComplexView5D send_z2x("send_z2x", Px, nx_local, ny_local, nz_local, nbatch);
  ComplexView5D recv_z2x("recv_z2x", Px, nx_local, ny_local, nz_local, nbatch);

  using policy_type = Kokkos::MDRangePolicy<
      execution_space,
      Kokkos::Rank<4, Kokkos::Iterate::Default, Kokkos::Iterate::Default>,
      Kokkos::IndexType<int>>;

  policy_type policy_z2x(exec, {0, 0, 0, 0},
                         {nx_local, ny_local, nz_local, nbatch}, {4, 4, 4, 1});

  Kokkos::parallel_for(
      "pack-z2x", policy_z2x, KOKKOS_LAMBDA(int ix, int iy, int iz, int ib) {
        for (int p = 0; p < Px; p++) {
          int giz                     = iz + p * nz_local;
          send_z2x(p, ix, iy, iz, ib) = giz < nzh
                                            ? out(ix, iy, giz, ib)
                                            : Kokkos::complex<double>(0.0, 0.0);
        }
      });

  // b) exchange blocks
  ::MPI_Alltoall(send_z2x.data(), send_count, MPI_DOUBLE_COMPLEX,
                 recv_z2x.data(), send_count, MPI_DOUBLE_COMPLEX, col_comm);

  // c) reshape back into in_X
  ComplexView4D in_Xpencil("in_Xpencil", nz_local, ny_local, nbatch, nx);

  Kokkos::parallel_for(
      "unpack-z2x", policy_z2x, KOKKOS_LAMBDA(int ix, int iy, int iz, int ib) {
        for (int p = 0; p < Px; p++) {
          int gix                     = ix + p * nx_local;
          in_Xpencil(iz, iy, ib, gix) = recv_z2x(p, ix, iy, iz, ib);
        }
      });

  // do your local 1D FFTs along X:
  KokkosFFT::fft(exec, in_Xpencil, in_Xpencil,
                 KokkosFFT::Normalization::backward, -1);

  // --- Second transpose: X‐pencils -> Y‐pencils ---
  // (Nz/px, Ny/py, Nx) -> (Nz/px, Nx/py, Ny)
  // a) prepare for all-to-all in the col_comm
  int send_count2 = nx_local2 * ny_local * nz_local * nbatch;

  ComplexView5D send_x2y("send_x2y", Py, nx_local2, ny_local, nz_local, nbatch);
  ComplexView5D recv_x2y("recv_x2y", Py, nx_local2, ny_local, nz_local, nbatch);

  exec.fence();

  policy_type policy_x2y(exec, {0, 0, 0, 0},
                         {nx_local2, ny_local, nz_local, nbatch}, {4, 4, 4, 1});

  Kokkos::parallel_for(
      "pack-x2y", policy_x2y, KOKKOS_LAMBDA(int ix, int iy, int iz, int ib) {
        for (int p = 0; p < Py; p++) {
          int gix                     = ix + p * nx_local;
          send_x2y(p, ix, iy, iz, ib) = in_Xpencil(iz, iy, ib, gix);
        }
      });

  // b) exchange blocks
  ::MPI_Alltoall(send_x2y.data(), send_count2, MPI_DOUBLE_COMPLEX,
                 recv_x2y.data(), send_count2, MPI_DOUBLE_COMPLEX, row_comm);

  // c) reshape back into Y-pencil
  ComplexView4D in_Ypencil("in_Ypencil", nz_local, nx_local2, nbatch, ny);

  Kokkos::parallel_for(
      "unpack-x2y", policy_x2y, KOKKOS_LAMBDA(int ix, int iy, int iz, int ib) {
        for (int p = 0; p < Py; p++) {
          int giy                     = iy + p * ny_local;
          in_Ypencil(iz, ix, ib, giy) = recv_x2y(p, ix, iy, iz, ib);
        }
      });

  // do your local 1D FFTs along Y:
  KokkosFFT::fft(exec, in_Ypencil, in_Ypencil,
                 KokkosFFT::Normalization::backward, -1);

  // Now, we will start the backward transforms
  KokkosFFT::ifft(exec, in_Ypencil, in_Ypencil,
                  KokkosFFT::Normalization::backward, -1);

  // --- Third transpose: Y‐pencils -> X‐pencils ---
  // (Nz/px, Nx/py, Ny) -> (Nz/px, Ny/py, Nx)
  // a) prepare for all-to-all in the col_comm
  ComplexView5D send_y2x("send_y2x", Py, nx_local2, ny_local, nz_local, nbatch);
  ComplexView5D recv_y2x("recv_y2x", Py, nx_local2, ny_local, nz_local, nbatch);

  policy_type policy_y2x(exec, {0, 0, 0, 0},
                         {nx_local2, ny_local, nz_local, nbatch}, {4, 4, 4, 1});

  Kokkos::parallel_for(
      "pack-y2x", policy_y2x, KOKKOS_LAMBDA(int ix, int iy, int iz, int ib) {
        for (int p = 0; p < Py; p++) {
          int giy                     = iy + p * ny_local;
          send_y2x(p, ix, iy, iz, ib) = in_Ypencil(iz, ix, ib, giy);
        }
      });

  exec.fence();

  // b) exchange blocks
  ::MPI_Alltoall(send_y2x.data(), send_count2, MPI_DOUBLE_COMPLEX,
                 recv_y2x.data(), send_count2, MPI_DOUBLE_COMPLEX, row_comm);

  // c) reshape back into in_x
  // ComplexView4D in_Ypencil2("in_Ypencil2", nx_local, nz_local, nbatch, ny);

  Kokkos::parallel_for(
      "unpack-y2x", policy_y2x, KOKKOS_LAMBDA(int ix, int iy, int iz, int ib) {
        for (int p = 0; p < Py; p++) {
          int gix                     = ix + p * nx_local;
          in_Xpencil(iz, iy, ib, gix) = recv_y2x(p, ix, iy, iz, ib);
        }
      });

  // do your local 1D FFTs along X:
  KokkosFFT::ifft(exec, in_Xpencil, in_Xpencil,
                  KokkosFFT::Normalization::backward, -1);

  // --- Forth transpose: X‐pencils -> Z‐pencils ---
  // (Nz/px, Ny/py, Nx) -> (Nx/px, Ny/py, Nz)
  // a) prepare for all‐to‐all in the row_comm
  ComplexView5D send_x2z("send_x2z", Px, nx_local, ny_local, nz_local, nbatch);
  ComplexView5D recv_x2z("recv_x2z", Px, nx_local, ny_local, nz_local, nbatch);

  policy_type policy_x2z(exec, {0, 0, 0, 0},
                         {nx_local, ny_local, nz_local, nbatch}, {4, 4, 4, 1});
  Kokkos::parallel_for(
      "pack-x2z", policy_x2z, KOKKOS_LAMBDA(int ix, int iy, int iz, int ib) {
        for (int p = 0; p < Px; p++) {
          int gix                     = ix + p * nx_local;
          send_x2z(p, ix, iy, iz, ib) = in_Xpencil(iz, iy, ib, gix);
        }
      });

  exec.fence();

  // b) exchange blocks
  ::MPI_Alltoall(send_x2z.data(), send_count, MPI_DOUBLE_COMPLEX,
                 recv_x2z.data(), send_count, MPI_DOUBLE_COMPLEX, col_comm);

  Kokkos::parallel_for(
      "unpack-x2z", policy_x2z, KOKKOS_LAMBDA(int ix, int iy, int iz, int ib) {
        for (int p = 0; p < Px; p++) {
          int giz = iz + p * nz_local;
          if (giz < nzh) {
            out(ix, iy, giz, ib) = recv_x2z(p, ix, iy, iz, ib);
          }
        }
      });

  // do your local 1D FFTs along Z:
  KokkosFFT::irfft(exec, out, in, KokkosFFT::Normalization::backward, 2);
  exec.fence();

  // Check results
  auto h_in = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), in);
  auto h_in_ref =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), in_ref);
  const double epsilon = 1.e-8;
  for (int ib = 0; ib < nbatch; ++ib) {
    for (int iz = 0; iz < nz; ++iz) {
      for (int iy = 0; iy < ny_local; ++iy) {
        for (int ix = 0; ix < nx_local; ++ix) {
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
    std::cout << "Distributed Z-pencil rFFT v0 completed successfully!"
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
