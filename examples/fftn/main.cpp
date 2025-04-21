#include <vector>
#include <iostream>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>

using execution_space = Kokkos::DefaultExecutionSpace;

void distributed_fft();

void distributed_fft() {
  int rank, nprocs;
  ::MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // 1) Define your global dims and process grid
  const int NX = 128, NY = 128, NZ = 128;
  int dims[2] = {0, 0};
  ::MPI_Dims_create(nprocs, 2, dims);  // choose Px × Py
  int periods[2] = {1, 1};             // Periodic in all directions
  ::MPI_Comm cart_comm;
  ::MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, /*reorder=*/1,
                    &cart_comm);

  // get my coords (px, py)
  int coords[2];
  MPI_Cart_coords(cart_comm, rank, 2, coords);
  int px = coords[0], py = coords[1];

  // split into row‐ and col‐ communicators
  MPI_Comm row_comm, col_comm;

  int remain_dims[2];

  // keep Y‐axis for row_comm (all procs with same px)
  remain_dims[0] = 1;
  remain_dims[1] = 0;
  MPI_Cart_sub(cart_comm, remain_dims, &row_comm);

  // keep X‐axis for col_comm (all procs with same py)
  remain_dims[0] = 0;
  remain_dims[1] = 1;
  MPI_Cart_sub(cart_comm, remain_dims, &col_comm);

  // 2) Local block sizes for Z‐pencils
  int nx = NX / dims[0];
  int ny = NY / dims[1];
  int nz = NZ;  // full Z

  using View3DType = Kokkos::View<Kokkos::complex<double>***, execution_space>;
  using View4DType = Kokkos::View<Kokkos::complex<double>****, execution_space>;
  View3DType in("in", nx, ny, nz);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  const Kokkos::complex<double> z(1.0, 1.0);
  execution_space exec;
  Kokkos::fill_random(exec, in, random_pool, z);

  // --- First transpose: Z‐pencils → Y‐pencils ---
  // (Nx/px, Ny/py, Nz) -> (Nx/px, Ny, Nz/py)
  // a) do your local 1D FFTs along Z:
  KokkosFFT::fft(exec, in, in, KokkosFFT::Normalization::backward, -1);

  // b) prepare for all‐to‐all in the row_comm
  int Py         = dims[1];
  int seg_z      = nz / Py;          // slice Z across Py
  int send_count = nx * ny * seg_z;  // each recv/send block size
  View3DType send_buffer("send_buffer", nx, ny, nz);
  View3DType recv_buffer("recv_buffer", nx, ny, nz);

  View4DType send_z2y(in.data(), nx, ny, seg_z, Py);
  View4DType recv_z2y(recv_buffer.data(), nx, ny, seg_z, Py);

  // c) exchange blocks
  MPI_Alltoall(send_z2y.data(), send_count, MPI_DOUBLE, recv_z2y.data(),
               send_count, MPI_DOUBLE, row_comm);

  // d) reshape back into in_y
  View3DType in_gy(in.data(), nx, ny * Py, seg_z);

  using policy_type = Kokkos::MDRangePolicy<
      execution_space,
      Kokkos::Rank<3, Kokkos::Iterate::Default, Kokkos::Iterate::Default>,
      Kokkos::IndexType<int>>;

  policy_type policy_z2y(exec, {0, 0, 0}, {nx, ny, seg_z}, {4, 4, 4});
  Kokkos::parallel_for(
      "transpose-z2y", policy_z2y, KOKKOS_LAMBDA(int ix, int iy, int iz) {
        for (int p = 0; p < Py; p++) {
          int giy            = iy + p * ny;
          in_gy(ix, giy, iz) = recv_z2y(ix, iy, iz, p);
        }
      });

  // a) (after Y‐FFTs) FFT along Y for each (i,k)
  KokkosFFT::fft(exec, in_gy, in_gy, KokkosFFT::Normalization::backward, 1);

  // --- Second transpose: Y‐pencils → X‐pencils ---
  // (Nx/px, Ny, Nz/py) -> (Nx, Ny/px, Nz/py)

  // b) prepare for all-to-all in the col_comm
  int Px          = dims[0];
  int seg_y       = ny * Py / Px;
  int send_count2 = nx * seg_y * seg_z;

  View4DType send_y2x(send_buffer.data(), nx, seg_y, seg_z, Px);
  View4DType recv_y2x(recv_buffer.data(), nx, seg_y, seg_z, Px);

  // c) exchange blocks
  MPI_Alltoall(send_y2x.data(), send_count2, MPI_DOUBLE, recv_y2x.data(),
               send_count2, MPI_DOUBLE, col_comm);

  /*
      // d) reshape: (nx*Px) × seg_y × nz
      View3DType in_x(in.data(), nx * Px, seg_y, seg_z);

      Kokkos::parallel_for("transpose-3D", policy,
        KOKKOS_LAMBDA(int ix, int iy, int iz) {
          for(int p=0; p<Px; p++) {
            int giy = iy + p * seg_y;
            send_y(ix, iy, iz, p) = in_y(ix, giy, iz);
          }
        });

      policy_type policy(exec_space, {0, 0, 0}, {nx, seg_y, seg_z}, {4, 4, 4});
      Kokkos::parallel_for("transpose-3D", policy,
        KOKKOS_LAMBDA(int ix, int iy, int iz) {
          for(int p=0; p<Px; p++) {
            int gix = ix + p * nx;
            in_x(gix, iy, iz) = recv(ix, iy, iz, p);
          }
        });

      // … now you have X‐pencils: do your final X‐FFTs …
      KokkosFFT::fft(in_x, in_x, 0);

      // --- Reverse 1 transpose: X‐pencils → Y‐pencils ---
      int Px = dims[0];
      int seg_y      = ny * Py / Px;
      int send_count2 = nx * seg_y * seg_z;
      View4DType send_x(send_y.data(), nx, seg_y, seg_z, Px);
      View4DType recv_x(recv.data(), nx, seg_y, seg_z, Px);

      Kokkos::parallel_for("transpose-3D", policy,
        KOKKOS_LAMBDA(int ix, int iy, int iz) {
          for(int p=0; p<Px; p++) {
            int gix = ix + p * nx;
            send_x(ix, iy, iz, p) = in_x(gix, iy, iz);
          }
        });

      // c) exchange blocks
      MPI_Alltoall(send_x.data(), send_count2, MPI_DOUBLE,
                   recv_x.data(), send_count2, MPI_DOUBLE,
                   col_comm);

      policy_type policy(exec_space, {0, 0, 0}, {nx, seg_y, seg_z}, {4, 4, 4});
      Kokkos::parallel_for("transpose-3D", policy,
        KOKKOS_LAMBDA(int ix, int iy, int iz) {
          for(int p=0; p<Px; p++) {
            send_y(ix, iy, iz, p) = recv_x(ix, iy, iz, p);
          }
        });

      // --- Reverse 2 transpose: Y‐pencils → Z‐pencils ---
      int Px = dims[0];
      int seg_y      = ny * Py / Px;
      int send_count2 = nx * seg_y * seg_z;
      View4DType send_yz(send.data(), nx, seg_y, seg_z, Py);
      View4DType recv_yz(in.data(), nx, seg_y, seg_z, Py);

      Kokkos::parallel_for("transpose-3D", policy,
        KOKKOS_LAMBDA(int ix, int iy, int iz) {
          for(int p=0; p<Px; p++) {
            int gix = ix + p * nx;
            send_x(ix, iy, iz, p) = in_x(gix, iy, iz);
          }
        });

      // c) exchange blocks
      MPI_Alltoall(send_yz.data(), send_count2, MPI_DOUBLE,
                   recv_yz.data(), send_count2, MPI_DOUBLE,
                   row_comm);
  */
  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);
  MPI_Comm_free(&cart_comm);
}

int main(int argc, char** argv) {
  ::MPI_Init(&argc, &argv);

  Kokkos::initialize(argc, argv);
  { distributed_fft(); }

  Kokkos::finalize();
  ::MPI_Finalize();

  return 0;
}
