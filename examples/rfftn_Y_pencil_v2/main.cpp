#include <vector>
#include <iostream>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>
#include "Extents.hpp"
#include "Plan.hpp"

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

  // 2) Local block sizes for Y‐pencils
  std::size_t Px     = dims[0];
  std::size_t Py     = dims[1];
  std::size_t nbatch = 5;

  using ComplexView3D = View3D<Kokkos::complex<double>>;
  using ComplexView4D = View4D<Kokkos::complex<double>>;
  using RealView3D    = View3D<double>;
  using RealView4D    = View4D<double>;

  using map_type        = std::array<std::size_t, 4>;
  using extents_type    = std::array<std::size_t, 4>;
  using axes_type       = std::array<int, 3>;
  using LayoutType      = typename RealView4D::array_layout;
  map_type in_topology  = {Px, 1, Py, 1};  // Y-pencil
  map_type out_topology = {1, Py, Px, 1};  // X-pencil
  axes_type axes{0, 2, 1};                 // axes for the FFTs

  // Start: Y-Pencil, End: X-Pencil
  // Global shape (nx, ny, nz, nbatch) -> (nx, ny/2+1, nz, nbatch)
  auto [nx_in, ny_in, nz_in, nbatch_in] = get_local_shape(
      extents_type({nx, ny, nz, nbatch}), in_topology, MPI_COMM_WORLD);
  auto [nx_out, ny_out, nz_out, nbatch_out] = get_local_shape(
      extents_type({nx, ny / 2 + 1, nz, nbatch}), out_topology, MPI_COMM_WORLD);
  RealView4D in("in", nx_in, ny_in, nz_in, nbatch_in),
      in_ref("in_ref", nx_in, ny_in, nz_in, nbatch_in);
  ComplexView4D out("out", nx_out, ny_out, nz_out, nbatch_out);

  // Initialize random input data
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  const double range = 1.0;
  execution_space exec;
  Kokkos::fill_random(exec, in, random_pool, range);
  Kokkos::deep_copy(in_ref, in);

  Plan distributed_plan(exec, in, out, axes, in_topology, out_topology,
                        MPI_COMM_WORLD);

  distributed_plan.forward(in, out);
  distributed_plan.backward(out, in);
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
    std::cout << "Distributed Y-pencil rFFT_v2 completed successfully!"
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
