#include <vector>
#include <iostream>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>

using execution_space = Kokkos::DefaultExecutionSpace;
template <typename T>
using View1D = Kokkos::View<T*, Kokkos::LayoutRight, execution_space>;
template <typename T>
using View2D = Kokkos::View<T**, Kokkos::LayoutRight, execution_space>;
template <typename T>
using View3D = Kokkos::View<T***, Kokkos::LayoutRight, execution_space>;
template <typename T>
using View4D = Kokkos::View<T****, Kokkos::LayoutRight, execution_space>;

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
  int px = coords[0], py = coords[1];

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
  int nx_local  = nx / dims[0];
  int ny_local  = ny / dims[1];
  int Py        = dims[1];
  int nz_local  = nz / Py;  // slice Z across Py
  int Px        = dims[0];
  int ny_local2 = ny / Px;

  using ComplexView3D = View3D<Kokkos::complex<double>>;
  using ComplexView4D = View4D<Kokkos::complex<double>>;

  // Start: Z-Pencil, End: X-Pencil
  ComplexView3D in("in", nx_local, ny_local, nz),
      in_ref("in_ref", nx_local, ny_local, nz);
  ComplexView3D out("out", nz_local, ny_local2, nx);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  const Kokkos::complex<double> z(1.0, 1.0);
  execution_space exec;
  Kokkos::fill_random(exec, in, random_pool, z);
  Kokkos::deep_copy(in_ref, in);

  // FFT on Z-pencil
  // do your local 1D FFTs along Z:
  KokkosFFT::fft(exec, in, in, KokkosFFT::Normalization::backward, -1);

  // --- First transpose: Z‐pencils -> Y‐pencils ---
  // (Nx/px, Ny/py, Nz) -> (Nx/px, Nz/py, Ny)
  // a) prepare for all‐to‐all in the row_comm
  int send_count = nx_local * ny_local * nz_local;  // each recv/send block size

  // We reuse these buffers for each all to all communications
  ComplexView3D send_buffer("send_buffer", nx_local, ny_local, nz);
  ComplexView3D recv_buffer("recv_buffer", nx_local, ny_local, nz);

  ComplexView4D send_z2y(in.data(), nx_local, ny_local, nz_local, Py);
  ComplexView4D recv_z2y(recv_buffer.data(), nx_local, ny_local, nz_local, Py);

  exec.fence();

  // b) exchange blocks
  ::MPI_Alltoall(send_z2y.data(), send_count, MPI_DOUBLE, recv_z2y.data(),
                 send_count, MPI_DOUBLE, row_comm);

  // c) reshape back into in_y
  ComplexView3D in_Ypencil(in.data(), nx_local, nz_local, ny);

  using policy_type = Kokkos::MDRangePolicy<
      execution_space,
      Kokkos::Rank<3, Kokkos::Iterate::Default, Kokkos::Iterate::Default>,
      Kokkos::IndexType<int>>;

  policy_type policy_z2y(exec, {0, 0, 0}, {nx_local, ny_local, nz_local},
                         {4, 4, 4});
  Kokkos::parallel_for(
      "transpose-z2y", policy_z2y, KOKKOS_LAMBDA(int ix, int iy, int iz) {
        for (int p = 0; p < Py; p++) {
          int giy                 = iy + p * ny_local;
          in_Ypencil(ix, iz, giy) = recv_z2y(ix, iy, iz, p);
        }
      });

  // do your local 1D FFTs along Y:
  KokkosFFT::fft(exec, in_Ypencil, in_Ypencil,
                 KokkosFFT::Normalization::backward, -1);

  // --- Second transpose: Y‐pencils -> X‐pencils ---
  // (Nx/px, Nz/py, Ny) -> (Nz/py, Ny/px, Nx)
  // a) prepare for all-to-all in the col_comm
  int send_count2 = nx_local * ny_local2 * nz_local;

  ComplexView4D send_y2x(in.data(), nx_local, nz_local, ny_local2, Px);
  ComplexView4D recv_y2x(recv_buffer.data(), nx_local, nz_local, ny_local2, Px);

  exec.fence();

  // b) exchange blocks
  ::MPI_Alltoall(send_y2x.data(), send_count2, MPI_DOUBLE, recv_y2x.data(),
                 send_count2, MPI_DOUBLE, col_comm);

  // c) reshape back into in_x
  ComplexView3D in_Xpencil(in.data(), nz_local, ny_local2, nx);

  policy_type policy_y2x(exec, {0, 0, 0}, {nx_local, nz_local, ny_local2},
                         {4, 4, 4});
  Kokkos::parallel_for(
      "transpose-y2x", policy_y2x, KOKKOS_LAMBDA(int ix, int iz, int iy) {
        for (int p = 0; p < Px; p++) {
          int gix                 = ix + p * nx_local;
          in_Xpencil(iz, iy, gix) = recv_y2x(ix, iz, iy, p);
        }
      });

  // do your local 1D FFTs along X:
  KokkosFFT::fft(exec, in_Xpencil, out, KokkosFFT::Normalization::backward, -1);

  // Now, we will start the backward transforms
  Kokkos::deep_copy(in_Xpencil, out);

  KokkosFFT::ifft(exec, in_Xpencil, in_Xpencil,
                  KokkosFFT::Normalization::backward, -1);

  // --- Third transpose: X‐pencils -> Y‐pencils ---
  // (Nz/py, Ny/px, Nx) -> (Nz/py, Nx/px, Ny)
  // a) prepare for all-to-all in the col_comm
  ComplexView4D send_x2y(in.data(), nz_local, ny_local2, nx_local, Px);
  ComplexView4D recv_x2y(recv_buffer.data(), nz_local, ny_local2, nx_local, Px);

  exec.fence();

  // b) exchange blocks
  ::MPI_Alltoall(send_x2y.data(), send_count2, MPI_DOUBLE, recv_x2y.data(),
                 send_count2, MPI_DOUBLE, col_comm);

  // c) reshape back into in_x
  ComplexView3D in_Ypencil2(in.data(), nz_local, nx_local, ny);
  policy_type policy_x2y(exec, {0, 0, 0}, {nz_local, ny_local2, nx_local},
                         {4, 4, 4});
  Kokkos::parallel_for(
      "transpose-x2y", policy_x2y, KOKKOS_LAMBDA(int iz, int iy, int ix) {
        for (int p = 0; p < Py; p++) {
          int giy                  = iy + p * ny_local;
          in_Ypencil2(iz, ix, giy) = recv_x2y(iz, iy, ix, p);
        }
      });

  // do your local 1D FFTs along Y:
  KokkosFFT::ifft(exec, in_Ypencil2, in_Ypencil2,
                  KokkosFFT::Normalization::backward, -1);

  // --- Forth transpose: Y‐pencils -> Z‐pencils ---
  // (Nz/py, Nx/px, Ny) -> (Nx/px, Ny/py, Nz)
  // a) prepare for all‐to‐all in the row_comm
  ComplexView4D send_y2z(in.data(), nz_local, nx_local, ny_local, Py);
  ComplexView4D recv_y2z(recv_buffer.data(), nz_local, nx_local, ny_local, Py);

  exec.fence();

  // b) exchange blocks
  ::MPI_Alltoall(send_y2z.data(), send_count, MPI_DOUBLE, recv_y2z.data(),
                 send_count, MPI_DOUBLE, row_comm);

  policy_type policy_y2z(exec, {0, 0, 0}, {nz_local, nx_local, ny_local},
                         {4, 4, 4});
  Kokkos::parallel_for(
      "transpose-y2z", policy_y2z, KOKKOS_LAMBDA(int iz, int ix, int iy) {
        for (int p = 0; p < Py; p++) {
          int giz         = iz + p * nz_local;
          in(ix, iy, giz) = recv_y2z(iz, ix, iy, p);
        }
      });

  // do your local 1D FFTs along Z:
  KokkosFFT::ifft(exec, in, in, KokkosFFT::Normalization::backward, -1);

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
