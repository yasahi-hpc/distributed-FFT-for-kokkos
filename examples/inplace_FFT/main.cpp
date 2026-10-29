#include <iostream>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed.hpp"

using execution_space = Kokkos::DefaultExecutionSpace;
template <typename T>
using View1D = Kokkos::View<T*, Kokkos::LayoutRight, execution_space>;
template <typename T>
using View3D = Kokkos::View<T***, Kokkos::LayoutRight, execution_space>;
template <typename T>
using UnmanagedView3D = Kokkos::View<T***, Kokkos::LayoutRight, execution_space,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

using extents_type = std::array<std::size_t, 3>;
using range3D_type = Kokkos::MDRangePolicy<
    execution_space,
    Kokkos::Rank<3, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;

// \brief Maximum of |a - b| over all the ranks
// \param[in] a The first View
// \param[in] b The second View
// \param[in] comm MPI communicator
// \return The maximum absolute difference between a and b
template <typename AViewType, typename BViewType>
double max_abs_diff(const AViewType& a, const BViewType& b, MPI_Comm comm) {
  double local_max = 0.0;
  Kokkos::parallel_reduce(
      "max_abs_diff",
      range3D_type({0, 0, 0}, {a.extent(0), a.extent(1), a.extent(2)}),
      KOKKOS_LAMBDA(int i0, int i1, int i2, double& lmax) {
        double diff = Kokkos::abs(a(i0, i1, i2) - b(i0, i1, i2));
        if (diff > lmax) lmax = diff;
      },
      Kokkos::Max<double>(local_max));
  double global_max = 0.0;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);
  return global_max;
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  Kokkos::initialize(argc, argv);
  {
    // 3D real to complex transform along all the axes on slabs:
    // the input is split along x, the output along y.
    // The extents are not divisible by every number of processes, so the
    // local extents can differ between the ranks.
    const std::size_t nx = 16, ny = 15, nz = 14;
    const std::size_t p = static_cast<std::size_t>(nprocs);
    const extents_type in_topology{p, 1, 1}, out_topology{1, p, 1};
    const extents_type gin_extents{nx, ny, nz},
        gout_extents{nx, ny, nz / 2 + 1};
    const KokkosFFT::axis_type<3> axes{0, 1, 2};

    auto [in_extents, in_starts] =
        KokkosFFT::Distributed::compute_local_extents_and_starts(
            gin_extents, in_topology, MPI_COMM_WORLD);
    auto [out_extents, out_starts] =
        KokkosFFT::Distributed::compute_local_extents_and_starts(
            gout_extents, out_topology, MPI_COMM_WORLD);

    // One allocation for both Views. It must hold the larger of the input
    // and the output on this rank, which is not always the output.
    const std::size_t buffer_size =
        KokkosFFT::Distributed::compute_inplace_allocation_size<
            double, Kokkos::complex<double>>(in_extents, out_extents);
    View1D<Kokkos::complex<double>> buffer("buffer", buffer_size);

    // Unmanaged Views on the same data, each in its own local extents:
    // u is (nx/p, ny, nz) real and u_hat is (nx, ny/p, nz/2+1) complex,
    // up to the remainders. Unlike single-node KokkosFFT, the real View is
    // not padded.
    auto [in0, in1, in2]    = in_extents;
    auto [out0, out1, out2] = out_extents;
    UnmanagedView3D<double> u(reinterpret_cast<double*>(buffer.data()), in0,
                              in1, in2);
    UnmanagedView3D<Kokkos::complex<double>> u_hat(buffer.data(), out0, out1,
                                                   out2);

    // Fill the input and keep a copy of it to check the results
    execution_space exec;
    Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345 + rank);
    Kokkos::fill_random(exec, u, random_pool, 1.0);
    View3D<double> u_ref("u_ref", in0, in1, in2);
    Kokkos::deep_copy(exec, u_ref, u);

    // The plan is in-place because u and u_hat point to the same data
    KokkosFFT::Distributed::Plan plan(exec, u, u_hat, axes, in_topology,
                                      out_topology, MPI_COMM_WORLD);
    if (rank == 0) {
      std::cout << "Plan: " << plan.name() << ", in-place: " << std::boolalpha
                << plan.is_inplace() << std::endl;
    }

    // Reference: the same transform out-of-place, on separate Views
    View3D<Kokkos::complex<double>> u_hat_ref("u_hat_ref", out0, out1, out2);
    KokkosFFT::Distributed::Plan plan_ref(exec, u_ref, u_hat_ref, axes,
                                          in_topology, out_topology,
                                          MPI_COMM_WORLD);
    KokkosFFT::Distributed::execute(plan_ref, u_ref, u_hat_ref,
                                    KokkosFFT::Direction::forward);

    // Forward: u is overwritten by u_hat
    KokkosFFT::Distributed::execute(plan, u, u_hat,
                                    KokkosFFT::Direction::forward);
    exec.fence();
    const double fwd_error = max_abs_diff(u_hat, u_hat_ref, MPI_COMM_WORLD);

    // Backward: u_hat is overwritten by u. With the default (backward)
    // normalization, it recovers the input.
    KokkosFFT::Distributed::execute(plan, u_hat, u,
                                    KokkosFFT::Direction::backward);
    exec.fence();
    const double bwd_error = max_abs_diff(u, u_ref, MPI_COMM_WORLD);

    if (rank == 0) {
      std::cout << "Forward error: " << fwd_error
                << ", backward error: " << bwd_error << std::endl;
    }
    if (fwd_error > 1.0e-10 || bwd_error > 1.0e-12) {
      if (rank == 0) std::cerr << "In-place FFT failed" << std::endl;
      Kokkos::finalize();
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
  Kokkos::finalize();
  MPI_Finalize();

  return 0;
}
