#include <vector>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>
#include "Block.hpp"
#include "Mapping.hpp"
#include "MPI_Helper.hpp"
#include "Extents.hpp"

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

  using map_type            = std::array<std::size_t, 4>;
  using extents_type        = std::array<std::size_t, 4>;
  using buffer_extents_type = std::array<std::size_t, 5>;
  using axes_type           = std::array<int, 3>;
  using pencil_axes_type    = std::tuple<std::size_t, std::size_t>;
  using paired_map_type     = std::tuple<map_type, map_type>;
  using paired_extents_type = std::tuple<extents_type, extents_type>;
  using LayoutType          = typename RealView4D::array_layout;
  map_type in_topology      = {Px, Py, 1, 1};  // Z-pencil
  map_type out_topology     = {1, Px, Py, 1};  // X-pencil
  map_type src_map          = {0, 1, 2, 3};
  axes_type axes{0, 1, 2};  // axes for the FFTs

  using ComplexView3D = View3D<Kokkos::complex<double>>;
  using ComplexView4D = View4D<Kokkos::complex<double>>;
  using ComplexView5D = View5D<Kokkos::complex<double>>;
  using RealView3D    = View3D<double>;
  using RealView4D    = View4D<double>;
  using FFTForwardBlockType =
      FFTForwardBlock<execution_space, ComplexView4D, ComplexView4D,
                      ComplexView4D, ComplexView5D>;
  using FFTBackwardBlockType =
      FFTBackwardBlock<execution_space, ComplexView4D, ComplexView4D,
                       ComplexView4D, ComplexView5D>;

  // Start: Z-Pencil, End: X-Pencil
  // Global shape (nx, ny, nz, nbatch) -> (nx, ny, nz/2+1, nbatch)
  auto [nx_in, ny_in, nz_in, nbatch_in] = get_local_shape(
      extents_type({nx, ny, nz, nbatch}), in_topology, MPI_COMM_WORLD);
  auto [nx_out, ny_out, nz_out, nbatch_out] = get_local_shape(
      extents_type({nx, ny, nz / 2 + 1, nbatch}), out_topology, MPI_COMM_WORLD);
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

  auto all_topologies =
      get_shuffled_topologies(in_topology, out_topology, axes);
  auto in_hat_extents = get_next_extents(gout_shape, in_topology, src_map);

  // Send/Recv buffers for MPI_communications
  extents_type current_extents = in_hat_extents;
  map_type current_map         = src_map;
  std::vector<buffer_extents_type> all_buffer_extents;
  std::vector<extents_type> all_pencil_extents;
  // std::vector<map_type> all_maps;
  std::vector<paired_map_type> all_maps;
  std::vector<pencil_axes_type> all_pencil_axes;
  std::vector<paired_extents_type> all_paired_extents;

  for (std::size_t i = 0; i < all_topologies.size(); i++) {
    // There are valid intermediate topology
    if (i + 2 < all_topologies.size()) {
      map_type mid_topology =
          get_mid_array(all_topologies.at(i), all_topologies.at(i + 2));
      auto [in_axis01, out_axis01] =
          get_pencil(all_topologies.at(i), mid_topology);
      auto [in_axis12, out_axis12] =
          get_pencil(mid_topology, all_topologies.at(i + 2));
      map_type mid_map01 =
          get_dst_map<Kokkos::LayoutRight, 4>(current_map, out_axis01);
      map_type mid_map12 =
          get_dst_map<Kokkos::LayoutRight, 4>(mid_map01, out_axis12);
      all_pencil_axes.push_back(pencil_axes_type{in_axis01, out_axis01});
      all_pencil_axes.push_back(pencil_axes_type{in_axis12, out_axis12});
      all_maps.push_back(paired_map_type{current_map, mid_map01});
      all_maps.push_back(paired_map_type{mid_map01, mid_map12});

      // Evaluate buffer extents based on the topology
      auto buffer_01_extents = get_buffer_extents<LayoutType>(
          gout_shape, all_topologies.at(i), mid_topology);
      auto buffer_12_extents = get_buffer_extents<LayoutType>(
          gout_shape, mid_topology, all_topologies.at(i + 2));
      all_buffer_extents.push_back(buffer_01_extents);
      all_buffer_extents.push_back(buffer_12_extents);

      // Evaluate next pencil extents
      auto next_01_extents =
          get_next_extents(gout_shape, mid_topology, mid_map01);
      auto next_12_extents =
          get_next_extents(gout_shape, all_topologies.at(i + 2), mid_map12);
      all_pencil_extents.push_back(next_01_extents);
      all_pencil_extents.push_back(next_12_extents);

      all_paired_extents.push_back(
          paired_extents_type{current_extents, next_01_extents});
      all_paired_extents.push_back(
          paired_extents_type{next_01_extents, next_12_extents});

      // Update the current topology
      current_map     = mid_map01;
      current_extents = next_01_extents;
    }
  }

  // Get the required buffer and pencil sizes
  auto buffer_size = get_required_allocation_size(all_buffer_extents);
  auto pencil_size = get_required_allocation_size(all_pencil_extents);

  // Allocate buffer views once
  ComplexView1D send_buffer_allocation("send_buffer_allocation", buffer_size),
      recv_buffer_allocation("recv_buffer_allocation", buffer_size);
  ComplexView1D pencil_allocation("pencil_allocation", pencil_size),
      pencil_allocation2("pencil_allocation2", pencil_size);

  // Using the loop to create plans
  std::vector<MPI_Comm> comms = {col_comm, row_comm};

  // Perform FFTs
  // do your local 1D FFTs along Z:
  ComplexView4D in_pencil(
      "in_pencil", KokkosFFT::Impl::create_layout<LayoutType>(in_hat_extents));

  KokkosFFT::rfft(exec, in, in_pencil, KokkosFFT::Normalization::backward, 2);

  auto [in_extents0, out_extents0] = all_paired_extents.at(0);
  auto [in_axis0, out_axis0]       = all_pencil_axes.at(0);
  auto [in_map0, out_map0]         = all_maps.at(0);

  ComplexView4D in_pencil0(
      pencil_allocation.data(),
      KokkosFFT::Impl::create_layout<LayoutType>(in_extents0));
  ComplexView4D out_pencil0(
      pencil_allocation.data(),
      KokkosFFT::Impl::create_layout<LayoutType>(out_extents0));
  ComplexView5D send_buffer0(
      send_buffer_allocation.data(),
      KokkosFFT::Impl::create_layout<LayoutType>(all_buffer_extents.at(0)));
  ComplexView5D recv_buffer0(
      recv_buffer_allocation.data(),
      KokkosFFT::Impl::create_layout<LayoutType>(all_buffer_extents.at(0)));

  FFTForwardBlock fft_block_z2y(exec, in_pencil, out_pencil0, out_pencil0,
                                send_buffer0, recv_buffer0, in_map0, in_axis0,
                                out_map0, out_axis0, comms.at(0));
  FFTBackwardBlock fft_block_y2z(exec, out_pencil0, out_pencil0, in_pencil,
                                 send_buffer0, recv_buffer0, out_map0,
                                 out_axis0, in_map0, in_axis0, comms.at(0));

  auto [in_extents1, out_extents1] = all_paired_extents.at(1);
  auto [in_axis1, out_axis1]       = all_pencil_axes.at(1);
  auto [in_map1, out_map1]         = all_maps.at(1);

  ComplexView4D in_pencil1(
      pencil_allocation.data(),
      KokkosFFT::Impl::create_layout<LayoutType>(in_extents1));
  ComplexView4D out_pencil1(
      pencil_allocation2.data(),
      KokkosFFT::Impl::create_layout<LayoutType>(out_extents1));
  ComplexView5D send_buffer1(
      send_buffer_allocation.data(),
      KokkosFFT::Impl::create_layout<LayoutType>(all_buffer_extents.at(1)));
  ComplexView5D recv_buffer1(
      recv_buffer_allocation.data(),
      KokkosFFT::Impl::create_layout<LayoutType>(all_buffer_extents.at(1)));

  FFTForwardBlock fft_block_y2x(exec, in_pencil1, out_pencil1, out_pencil1,
                                send_buffer1, recv_buffer1, in_map1, in_axis1,
                                out_map1, out_axis1, comms.at(1));
  FFTBackwardBlock fft_block_x2y(exec, out_pencil1, out_pencil1, in_pencil1,
                                 send_buffer1, recv_buffer1, out_map1,
                                 out_axis1, in_map1, in_axis1, comms.at(1));

  fft_block_z2y();
  fft_block_y2x();

  // Now, we will start the backward transforms

  fft_block_x2y();
  fft_block_y2z();

  // do your local 1D FFTs along Z:
  KokkosFFT::irfft(exec, in_pencil, in, KokkosFFT::Normalization::backward, 2);
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
    std::cout << "Distributed Z-pencil rFFT v3 completed successfully!"
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
