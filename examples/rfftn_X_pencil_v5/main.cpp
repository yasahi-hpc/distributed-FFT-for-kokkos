#include <vector>
#include <iostream>
#include <sstream>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>
#include "Block.hpp"
#include "Mapping.hpp"
#include "MPI_Helper.hpp"
#include "Extents.hpp"
// #include "Plan.hpp"

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
  std::size_t Px     = dims[0];
  std::size_t Py     = dims[1];
  std::size_t nbatch = 5;

  using ComplexView1D = View1D<Kokkos::complex<double>>;
  using ComplexView3D = View3D<Kokkos::complex<double>>;
  using ComplexView4D = View4D<Kokkos::complex<double>>;
  using ComplexView5D = View5D<Kokkos::complex<double>>;
  using RealView3D    = View3D<double>;
  using RealView4D    = View4D<double>;

  using map_type        = std::array<std::size_t, 4>;
  using extents_type    = std::array<std::size_t, 4>;
  using axes_type       = std::array<int, 3>;
  using LayoutType      = typename RealView4D::array_layout;
  map_type in_topology  = {1, Px, Py, 1};  // X-pencil
  map_type out_topology = {Px, Py, 1, 1};  // Z-pencil
  map_type src_map      = {0, 1, 2, 3};

  // Start: X-Pencil, End: Z-Pencil
  // Global shape (nx, ny, nz, nbatch) -> (nx/2+1, ny, nz, nbatch)
  auto [nx_in, ny_in, nz_in, nbatch_in] = get_local_shape(
      extents_type({nx, ny, nz, nbatch}), in_topology, MPI_COMM_WORLD);
  auto [nx_out, ny_out, nz_out, nbatch_out] = get_local_shape(
      extents_type({nx / 2 + 1, ny, nz, nbatch}), out_topology, MPI_COMM_WORLD);
  RealView4D in("in", nx_in, ny_in, nz_in, nbatch_in),
      in_ref("in_ref", nx_in, ny_in, nz_in, nbatch_in);
  ComplexView4D out("out", nx_out, ny_out, nz_out, nbatch_out);

  // Initialize random input data
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  const double range = 1.0;
  execution_space exec;
  Kokkos::fill_random(exec, in, random_pool, range);
  Kokkos::deep_copy(in_ref, in);

  // First get global shape to define buffer and next shape
  auto gin_shape  = get_global_shape(in, in_topology, MPI_COMM_WORLD);
  auto gout_shape = get_global_shape(out, out_topology, MPI_COMM_WORLD);

  auto out_x_extents = get_next_extents(gout_shape, in_topology, src_map);
  ComplexView4D out_x(
      "out_x", KokkosFFT::Impl::create_layout<LayoutType>(out_x_extents));

  // do your local 1D FFTs along X:
  KokkosFFT::rfft(exec, in, out_x, KokkosFFT::Normalization::backward, 0);

  // X-Pencil to Y-pencil transpose + local 1D FFTs along Y
  map_type mid_topology = get_mid_array(in_topology, out_topology);  // Y-pencil
  auto [in_axis01, out_axis01] = get_pencil(in_topology, mid_topology);
  map_type mid_map = get_dst_map<Kokkos::LayoutRight, 4>(src_map, out_axis01);

  auto [in_axis12, out_axis12] = get_pencil(mid_topology, out_topology);
  map_type out_map = get_dst_map<Kokkos::LayoutRight, 4>(mid_map, out_axis12);

  // Allocate buffers
  auto x2y_buffer_extents =
      get_buffer_extents<LayoutType>(gout_shape, in_topology, mid_topology);
  auto y2z_buffer_extents =
      get_buffer_extents<LayoutType>(gout_shape, mid_topology, out_topology);

  // Allocate Y and Z pencils
  auto out_y_extents = get_next_extents(gout_shape, mid_topology, mid_map);
  auto out_z_extents = get_next_extents(gout_shape, out_topology, out_map);

  // Allocate send buffer and Y-pencil
  std::vector<std::array<std::size_t, 5>> buffer_extents = {x2y_buffer_extents,
                                                            y2z_buffer_extents};
  std::vector<std::array<std::size_t, 4>> pencil_extents = {out_y_extents,
                                                            out_z_extents};

  auto buffer_size = get_required_allocation_size(buffer_extents);
  auto pencil_size = get_required_allocation_size(pencil_extents);

  ComplexView1D send_buffer_allocation("send_buffer_allocation", buffer_size),
      recv_buffer_allocation("recv_buffer_allocation", buffer_size);
  ComplexView1D pencil_allocation("pencil_allocation", pencil_size);

  ComplexView5D send_x2y(
      send_buffer_allocation.data(),
      KokkosFFT::Impl::create_layout<LayoutType>(x2y_buffer_extents));
  ComplexView5D recv_x2y(
      recv_buffer_allocation.data(),
      KokkosFFT::Impl::create_layout<LayoutType>(x2y_buffer_extents));

  ComplexView5D send_y2z(
      send_buffer_allocation.data(),
      KokkosFFT::Impl::create_layout<LayoutType>(y2z_buffer_extents));
  ComplexView5D recv_y2z(
      recv_buffer_allocation.data(),
      KokkosFFT::Impl::create_layout<LayoutType>(y2z_buffer_extents));
  ComplexView4D Ypencil(
      pencil_allocation.data(),
      KokkosFFT::Impl::create_layout<LayoutType>(out_y_extents));
  ComplexView4D Zpencil(
      pencil_allocation.data(),
      KokkosFFT::Impl::create_layout<LayoutType>(out_z_extents));

  // X-pencil to Y-pencil transpose + local 1D FFTs along Y
  FFTForwardBlock fft_block_x2y(exec, out_x, Ypencil, Ypencil, send_x2y,
                                recv_x2y, src_map, in_axis01, mid_map,
                                out_axis01, col_comm);
  fft_block_x2y();

  // Y-pencil to Z-pencil transpose + local 1D FFTs along Z
  // Allocate send buffer and Z-pencil
  FFTForwardBlock fft_block_y2z(exec, Ypencil, Zpencil, Zpencil, send_y2z,
                                recv_y2z, mid_map, out_axis01, out_map,
                                out_axis12, row_comm);
  fft_block_y2z();

  // Now, we will start the backward transforms
  // --- Third transpose: Z‐pencils -> Y‐pencils ---
  // IFFT on Z-pencil (Px, Py, 1, 1) -> (Px, 1, Py, 1)
  // local 1D FFTs along Z + Z-pencil to Y-Pencil transpose
  FFTBackwardBlock fft_block_z2y(exec, Zpencil, Zpencil, Ypencil, send_y2z,
                                 recv_y2z, out_map, out_axis12, mid_map,
                                 out_axis01, row_comm);
  fft_block_z2y();

  FFTBackwardBlock fft_block_y2x(exec, Ypencil, Ypencil, out_x, send_x2y,
                                 recv_x2y, mid_map, out_axis01, src_map,
                                 in_axis01, col_comm);
  fft_block_y2x();

  // do your local 1D FFTs along X:
  KokkosFFT::irfft(exec, out_x, in, KokkosFFT::Normalization::backward, 0);
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
    std::cout << "Distributed X-pencil rFFT completed successfully!"
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
