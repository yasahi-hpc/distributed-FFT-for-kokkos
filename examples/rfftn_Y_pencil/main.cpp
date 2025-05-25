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
  int Px        = dims[0];
  int Py        = dims[1];
  int nx_local  = nx / Px;
  int ny_local  = ((ny / 2 + 1) - 1) / Py + 1;
  int nz_local  = nz / Py;  // slice Z across Py
  int nz_local2 = ny / Px;
  int nbatch    = 5;
  int nyh       = ny / 2 + 1;  // half size in Y for real-to-complex FFT

  using ComplexView3D = View3D<Kokkos::complex<double>>;
  using ComplexView4D = View4D<Kokkos::complex<double>>;
  using ComplexView5D = View5D<Kokkos::complex<double>>;
  using RealView3D    = View3D<double>;
  using RealView4D    = View4D<double>;

  // Start: Y-Pencil, End: X-Pencil
  RealView4D in("in", nx_local, ny, nz_local, nbatch),
      in_ref("in_ref", nx_local, ny, nz_local, nbatch);
  ComplexView4D out("out", nx_local, ny / 2 + 1, nz_local, nbatch);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  const double range = 1.0;
  execution_space exec;
  Kokkos::fill_random(exec, in, random_pool, range);
  Kokkos::deep_copy(in_ref, in);

  // FFT on Y-pencil
  // do your local 1D FFTs along Y:
  KokkosFFT::rfft(exec, in, out, KokkosFFT::Normalization::backward, 1);

  // --- First transpose: Y‐pencils -> Z‐pencils ---
  // (Nx/px, Ny, Nz/py) -> (Nx/px, Ny/py, Nz)
  // a) prepare for all‐to‐all in the row_comm
  int send_count =
      nx_local * ny_local * nz_local * nbatch;  // each recv/send block size

  // We reuse these buffers for each all to all communications
  ComplexView5D send_y2z("send_y2z", Py, nx_local, ny_local, nz_local, nbatch);
  ComplexView5D recv_y2z("recv_y2z", Py, nx_local, ny_local, nz_local, nbatch);

  using policy_type = Kokkos::MDRangePolicy<
      execution_space,
      Kokkos::Rank<4, Kokkos::Iterate::Default, Kokkos::Iterate::Default>,
      Kokkos::IndexType<int>>;

  policy_type policy_y2z(exec, {0, 0, 0, 0},
                         {nx_local, ny_local, nz_local, nbatch}, {4, 4, 4, 1});

  Kokkos::parallel_for(
      "pack-y2z", policy_y2z, KOKKOS_LAMBDA(int ix, int iy, int iz, int ib) {
        for (int p = 0; p < Py; p++) {
          int giy                     = iy + p * ny_local;
          send_y2z(p, ix, iy, iz, ib) = giy < nyh
                                            ? out(ix, giy, iz, ib)
                                            : Kokkos::complex<double>(0.0, 0.0);
        }
      });

  // b) exchange blocks
  ::MPI_Alltoall(send_y2z.data(), send_count, MPI_DOUBLE_COMPLEX,
                 recv_y2z.data(), send_count, MPI_DOUBLE_COMPLEX, row_comm);

  // c) reshape back into in_z
  ComplexView4D in_Zpencil("in_Zpencil", nx_local, ny_local, nbatch, nz);

  Kokkos::parallel_for(
      "unpack-y2z", policy_y2z, KOKKOS_LAMBDA(int ix, int iy, int iz, int ib) {
        for (int p = 0; p < Py; p++) {
          int giz                     = iz + p * nz_local;
          in_Zpencil(ix, iy, ib, giz) = recv_y2z(p, ix, iy, iz, ib);
        }
      });

  // do your local 1D FFTs along Z:
  KokkosFFT::fft(exec, in_Zpencil, in_Zpencil,
                 KokkosFFT::Normalization::backward, -1);

  // --- Second transpose: Z‐pencils -> X‐pencils ---
  // (Nx/px, Ny/py, Nz) -> (Nz/px, Ny/py, Nx)
  // a) prepare for all-to-all in the col_comm
  int send_count2 = nx_local * ny_local * nz_local2 * nbatch;

  ComplexView5D send_z2x("send_z2x", Px, nx_local, ny_local, nz_local2, nbatch);
  ComplexView5D recv_z2x("recv_z2x", Px, nx_local, ny_local, nz_local2, nbatch);

  exec.fence();

  policy_type policy_z2x(exec, {0, 0, 0, 0},
                         {nx_local, ny_local, nz_local2, nbatch}, {4, 4, 4, 1});

  Kokkos::parallel_for(
      "pack-z2x", policy_z2x, KOKKOS_LAMBDA(int ix, int iy, int iz, int ib) {
        for (int p = 0; p < Px; p++) {
          int giz                     = iz + p * nz_local;
          send_z2x(p, ix, iy, iz, ib) = in_Zpencil(ix, iy, ib, giz);
        }
      });

  // b) exchange blocks
  ::MPI_Alltoall(send_z2x.data(), send_count2, MPI_DOUBLE_COMPLEX,
                 recv_z2x.data(), send_count2, MPI_DOUBLE_COMPLEX, col_comm);

  // c) reshape back into X-pencil
  ComplexView4D in_Xpencil("in_Xpencil", nz_local2, ny_local, nbatch, nx);

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

  // Now, we will start the backward transforms
  KokkosFFT::ifft(exec, in_Xpencil, in_Xpencil,
                  KokkosFFT::Normalization::backward, -1);

  // --- Third transpose: X‐pencils -> Z‐pencils ---
  // (Nz/px, Ny/py, Nx) -> (Nx/px, Ny/py, Nz)
  // a) prepare for all-to-all in the col_comm
  ComplexView5D send_x2z("send_x2z", Px, nx_local, ny_local, nz_local2, nbatch);
  ComplexView5D recv_x2z("recv_x2z", Px, nx_local, ny_local, nz_local2, nbatch);

  policy_type policy_x2z(exec, {0, 0, 0, 0},
                         {nx_local, ny_local, nz_local2, nbatch}, {4, 4, 4, 1});

  Kokkos::parallel_for(
      "pack-x2z", policy_x2z, KOKKOS_LAMBDA(int ix, int iy, int iz, int ib) {
        for (int p = 0; p < Px; p++) {
          int gix                     = ix + p * nx_local;
          send_x2z(p, ix, iy, iz, ib) = in_Xpencil(iz, iy, ib, gix);
        }
      });

  exec.fence();

  // b) exchange blocks
  ::MPI_Alltoall(send_x2z.data(), send_count2, MPI_DOUBLE_COMPLEX,
                 recv_x2z.data(), send_count2, MPI_DOUBLE_COMPLEX, col_comm);

  // c) reshape back into in_y
  // ComplexView4D in_Ypencil2("in_Ypencil2", nx_local, nz_local, nbatch, ny);

  Kokkos::parallel_for(
      "unpack-x2z", policy_x2z, KOKKOS_LAMBDA(int ix, int iy, int iz, int ib) {
        for (int p = 0; p < Px; p++) {
          int giz                     = iz + p * nz_local;
          in_Zpencil(ix, iy, ib, giz) = recv_x2z(p, ix, iy, iz, ib);
        }
      });

  // do your local 1D FFTs along Z:
  KokkosFFT::ifft(exec, in_Zpencil, in_Zpencil,
                  KokkosFFT::Normalization::backward, -1);

  // --- Forth transpose: Z‐pencils -> Y‐pencils ---
  // (Nx/px, Ny/py, Nz) -> (Nx/px, Ny, Nz/py)
  // a) prepare for all‐to‐all in the row_comm
  ComplexView5D send_z2y("send_z2y", Py, nx_local, ny_local, nz_local, nbatch);
  ComplexView5D recv_z2y("recv_z2y", Py, nx_local, ny_local, nz_local, nbatch);

  policy_type policy_z2y(exec, {0, 0, 0, 0},
                         {nx_local, ny_local, nz_local, nbatch}, {4, 4, 4, 1});
  Kokkos::parallel_for(
      "pack-z2y", policy_z2y, KOKKOS_LAMBDA(int ix, int iy, int iz, int ib) {
        for (int p = 0; p < Py; p++) {
          int giz                     = iz + p * nz_local;
          send_z2y(p, ix, iy, iz, ib) = in_Zpencil(ix, iy, ib, giz);
        }
      });

  exec.fence();

  // b) exchange blocks
  ::MPI_Alltoall(send_z2y.data(), send_count, MPI_DOUBLE_COMPLEX,
                 recv_z2y.data(), send_count, MPI_DOUBLE_COMPLEX, row_comm);

  Kokkos::parallel_for(
      "unpack-z2y", policy_z2y, KOKKOS_LAMBDA(int ix, int iy, int iz, int ib) {
        for (int p = 0; p < Py; p++) {
          int giy = iy + p * ny_local;
          if (giy < nyh) {
            out(ix, giy, iz, ib) = recv_z2y(p, ix, iy, iz, ib);
          }
        }
      });

  // do your local 1D FFTs along Y:
  KokkosFFT::irfft(exec, out, in, KokkosFFT::Normalization::backward, 1);
  exec.fence();

  // Check results
  auto h_in = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), in);
  auto h_in_ref =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), in_ref);
  const double epsilon = 1.e-8;
  for (int ib = 0; ib < nbatch; ++ib) {
    for (int iz = 0; iz < nz_local; ++iz) {
      for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx_local; ++ix) {
          if (Kokkos::abs(h_in(ix, iy, iz, ib)) <= epsilon) continue;
          auto relative_error =
              Kokkos::abs(h_in(ix, iy, iz, ib) - h_in_ref(ix, iy, iz, ib)) /
              Kokkos::abs(h_in(ix, iy, iz, ib));
          if (relative_error > epsilon) {
            std::cerr << "Error (ix, iy, iz, ib): " << ix << ", " << iy << ", "
                      << iz << " @ rank" << rank << ", " << h_in(ix, iy, iz, ib)
                      << " != " << h_in_ref(ix, iy, iz, ib) << std::endl;
            return;
          }
        }
      }
    }
  }

  if (rank == 0) {
    std::cout << "Distributed Y-pencil rFFT completed successfully!"
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
