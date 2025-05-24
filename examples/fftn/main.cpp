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

  // 2) Local block sizes for Z‐pencils
  int nx_local = nx / nprocs;
  int ny_local = ny / nprocs;

  using ComplexView3D = View3D<Kokkos::complex<double>>;
  using ComplexView4D = View4D<Kokkos::complex<double>>;

  // Start: X-slab, End: Y-slab
  ComplexView3D in("in", nx_local, ny, nz), in_ref("in_ref", nx_local, ny, nz);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  const Kokkos::complex<double> z(1.0, 1.0);
  execution_space exec;
  Kokkos::fill_random(exec, in, random_pool, z);
  Kokkos::deep_copy(in_ref, in);

  using axes_type = KokkosFFT::axis_type<2>;
  axes_type axes  = {-2, -1};

  // X-slab (nx, ny, nz) FFT2 along Y and Z
  KokkosFFT::fft2(exec, in, in, KokkosFFT::Normalization::backward, axes);

  // --- First transpose: X‐pencils -> Y‐pencils ---
  // (Nx/px, Ny, Nz) -> (Nx/px, Nz/py, Ny)
  // a) prepare for all‐to‐all in the row_comm

  // We reuse these buffers for each all to all communications
  ComplexView4D send_x2y("send_x2y", nprocs, nx_local, ny_local, nz);
  ComplexView4D recv_x2y("recv_x2y", nprocs, nx_local, ny_local, nz);

  using policy_type = Kokkos::MDRangePolicy<
      execution_space,
      Kokkos::Rank<3, Kokkos::Iterate::Default, Kokkos::Iterate::Default>,
      Kokkos::IndexType<int>>;

  policy_type policy_x2y(exec, {0, 0, 0}, {nx_local, ny_local, nz}, {4, 4, 4});

  Kokkos::parallel_for(
      "pack-x2y", policy_x2y, KOKKOS_LAMBDA(int ix, int iy, int iz) {
        for (int p = 0; p < nprocs; p++) {
          int giy                 = iy + p * ny_local;
          send_x2y(p, ix, iy, iz) = in(ix, giy, iz);
        }
      });

  exec.fence();

  // b) exchange blocks
  int send_count = nx_local * ny_local * nz;  // each recv/send block size
  ::MPI_Alltoall(send_x2y.data(), send_count, MPI_DOUBLE_COMPLEX,
                 recv_x2y.data(), send_count, MPI_DOUBLE_COMPLEX,
                 MPI_COMM_WORLD);

  // c) reshape back into in_y
  ComplexView3D in_Yslab("in_Yslab", ny_local, nz, nx);

  Kokkos::parallel_for(
      "unpack-x2y", policy_x2y, KOKKOS_LAMBDA(int ix, int iy, int iz) {
        for (int p = 0; p < nprocs; p++) {
          int gix               = ix + p * nx_local;
          in_Yslab(iy, iz, gix) = recv_x2y(p, ix, iy, iz);
        }
      });

  // do your local 1D FFTs along X:
  KokkosFFT::fft(exec, in_Yslab, in_Yslab, KokkosFFT::Normalization::backward,
                 -1);

  // Now, we will start the backward transforms
  // along X
  KokkosFFT::ifft(exec, in_Yslab, in_Yslab, KokkosFFT::Normalization::backward,
                  -1);

  // --- Second transpose: Y‐pencils -> X‐pencils ---
  // (Nx/px, Nz/py, Ny) -> (Nz/py, Ny/px, Nx)
  // a) prepare for all-to-all in the col_comm

  ComplexView4D send_y2x("send_y2x", nprocs, nx_local, ny_local, nz);
  ComplexView4D recv_y2x("recv_y2x", nprocs, nx_local, ny_local, nz);

  policy_type policy_y2x(exec, {0, 0, 0}, {nx_local, ny_local, nz}, {4, 4, 4});

  Kokkos::parallel_for(
      "pack-y2x", policy_y2x, KOKKOS_LAMBDA(int ix, int iy, int iz) {
        for (int p = 0; p < nprocs; p++) {
          int gix                 = ix + p * nx_local;
          send_y2x(p, ix, iy, iz) = in_Yslab(iy, iz, gix);
        }
      });

  exec.fence();

  // b) exchange blocks
  ::MPI_Alltoall(send_y2x.data(), send_count, MPI_DOUBLE_COMPLEX,
                 recv_y2x.data(), send_count, MPI_DOUBLE_COMPLEX,
                 MPI_COMM_WORLD);

  // c) reshape back into in_x
  ComplexView3D in_Xslab("in_Xslab", nx_local, ny, nz);

  Kokkos::parallel_for(
      "unpack-y2x", policy_y2x, KOKKOS_LAMBDA(int ix, int iy, int iz) {
        for (int p = 0; p < nprocs; p++) {
          int giy               = iy + p * ny_local;
          in_Xslab(ix, giy, iz) = recv_y2x(p, ix, iy, iz);
        }
      });

  // do your local FFT2 along Y and Z
  KokkosFFT::ifft2(exec, in_Xslab, in, KokkosFFT::Normalization::backward,
                   axes);
  exec.fence();

  // Check results
  auto h_in = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), in);
  auto h_in_ref =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), in_ref);
  const double epsilon = 1.e-8;
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx_local; ++ix) {
        if (Kokkos::abs(h_in(ix, iy, iz)) <= epsilon) continue;
        auto relative_error =
            Kokkos::abs(h_in(ix, iy, iz) - h_in_ref(ix, iy, iz)) /
            Kokkos::abs(h_in(ix, iy, iz));
        if (relative_error > epsilon) {
          std::cerr << "Error (ix, iy, iz): " << ix << ", " << iy << ", " << iz
                    << " @ rank" << rank << ", " << h_in(ix, iy, iz)
                    << " != " << h_in_ref(ix, iy, iz) << std::endl;
          return;
        }
      }
    }
  }
  if (rank == 0) {
    std::cout << "Distributed FFT completed successfully!" << std::endl;
  }
}

int main(int argc, char** argv) {
  ::MPI_Init(&argc, &argv);

  Kokkos::initialize(argc, argv);
  { distributed_fft(); }
  Kokkos::finalize();
  ::MPI_Finalize();

  return 0;
}
