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

  // Transforming from (Nr, Ntheta, Nphi/nprocs) -> (Nr, Ntheta/nprocs, Nphi/2)
  // in_topology (1, 1, n) -> (1, n, 1)
  // View types
  using RealView3D    = View3D<double>;
  using RealView4D    = View4D<double>;
  using ComplexView3D = View3D<Kokkos::complex<double>>;
  using ComplexView4D = View4D<Kokkos::complex<double>>;

  int Nr = 16, Ntheta = 16, Nphi = 8;
  int Ntheta_local = Ntheta / nprocs;
  int Nphi_local   = Nphi / nprocs;
  RealView3D phi("phi", Nr, Ntheta, Nphi_local),
      phi_full("phi_full", Nr, Ntheta_local, Nphi),
      phi_ref("phi_ref", Nr, Ntheta, Nphi_local);
  ComplexView3D phi_hat("phi_hat", Nr, Ntheta_local, Nphi / 2 + 1);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  const double range = 1.0;
  execution_space exec;
  Kokkos::fill_random(exec, phi, random_pool, range);
  Kokkos::deep_copy(phi_ref, phi);

  // First transpose to full Nphi layout
  RealView4D send_y2z("send_y2z", nprocs, Nr, Ntheta_local, Nphi_local);
  RealView4D recv_y2z("recv_y2z", nprocs, Nr, Ntheta_local, Nphi_local);

  // Pack
  using policy_type = Kokkos::MDRangePolicy<
      execution_space,
      Kokkos::Rank<3, Kokkos::Iterate::Default, Kokkos::Iterate::Default>,
      Kokkos::IndexType<int>>;

  policy_type policy_y2z(exec, {0, 0, 0}, {Nr, Ntheta_local, Nphi_local},
                         {4, 4, 4});

  Kokkos::parallel_for(
      "pack-y2z", policy_y2z, KOKKOS_LAMBDA(int ix, int iy, int iz) {
        for (int p = 0; p < nprocs; p++) {
          int giy                 = iy + p * Ntheta_local;
          send_y2z(p, ix, iy, iz) = giy < Ntheta ? phi(ix, giy, iz) : 0.0;
        }
      });

  exec.fence();

  int send_count = Nr * Ntheta_local * Nphi_local;
  ::MPI_Alltoall(send_y2z.data(), send_count, MPI_DOUBLE, recv_y2z.data(),
                 send_count, MPI_DOUBLE, MPI_COMM_WORLD);

  Kokkos::parallel_for(
      "unpack-y2z", policy_y2z, KOKKOS_LAMBDA(int ix, int iy, int iz) {
        for (int p = 0; p < nprocs; p++) {
          int giz = iz + p * Nphi_local;
          if (giz < Nphi) {
            phi_full(ix, iy, giz) = recv_y2z(p, ix, iy, iz);
          }
        }
      });

  // Do FFT along phi direction
  KokkosFFT::rfft(exec, phi_full, phi_hat, KokkosFFT::Normalization::backward,
                  -1);

  // Do some operation here
  KokkosFFT::irfft(exec, phi_hat, phi_full, KokkosFFT::Normalization::backward,
                   -1);

  // Pack
  RealView4D send_z2y("send_z2y", nprocs, Nr, Ntheta_local, Nphi_local);
  RealView4D recv_z2y("recv_z2y", nprocs, Nr, Ntheta_local, Nphi_local);

  policy_type policy_z2y(exec, {0, 0, 0}, {Nr, Ntheta_local, Nphi_local},
                         {4, 4, 4});

  Kokkos::parallel_for(
      "pack-z2y", policy_z2y, KOKKOS_LAMBDA(int ix, int iy, int iz) {
        for (int p = 0; p < nprocs; p++) {
          int giz                 = iz + p * Nphi_local;
          send_z2y(p, ix, iy, iz) = giz < Nphi ? phi_full(ix, iy, giz) : 0.0;
        }
      });

  exec.fence();

  ::MPI_Alltoall(send_z2y.data(), send_count, MPI_DOUBLE, recv_z2y.data(),
                 send_count, MPI_DOUBLE, MPI_COMM_WORLD);

  Kokkos::parallel_for(
      "unpack-z2y", policy_z2y, KOKKOS_LAMBDA(int ix, int iy, int iz) {
        for (int p = 0; p < nprocs; p++) {
          int giy = iy + p * Ntheta_local;
          if (giy < Ntheta) {
            phi(ix, giy, iz) = recv_z2y(p, ix, iy, iz);
          }
        }
      });

  exec.fence();

  // Check results
  auto h_phi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), phi);
  auto h_phi_ref =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), phi_ref);
  const double epsilon = 1.e-8;

  for (int iz = 0; iz < Nphi_local; ++iz) {
    for (int iy = 0; iy < Ntheta; ++iy) {
      for (int ix = 0; ix < Nr; ++ix) {
        if (Kokkos::abs(h_phi(ix, iy, iz)) <= epsilon) continue;
        auto relative_error =
            Kokkos::abs(h_phi(ix, iy, iz) - h_phi_ref(ix, iy, iz)) /
            Kokkos::abs(h_phi(ix, iy, iz));
        if (relative_error > epsilon) {
          std::cerr << "Error phi (ix, iy, iz): " << ix << ", " << iy << ", "
                    << iz << " @ rank" << rank << ", " << h_phi(ix, iy, iz)
                    << " != " << h_phi_ref(ix, iy, iz) << std::endl;
          return;
        }
      }
    }
  }

  if (rank == 0) {
    std::cout << "Distributed 1D-batched FFT completed successfully!"
              << std::endl;
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
