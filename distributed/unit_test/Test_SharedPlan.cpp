#include <mpi.h>
#include <gtest/gtest.h>
#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include "SharedPlan.hpp"
#include "MPI_Helper.hpp"
#include "Test_Utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using test_types      = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>,
                                    std::pair<double, Kokkos::LayoutLeft>,
                                    std::pair<double, Kokkos::LayoutRight>>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct TestShared1D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;

  int m_rank   = 0;
  int m_nprocs = 1;

  virtual void SetUp() {
    ::MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &m_nprocs);
  }
};

template <typename T>
struct TestShared2D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;

  int m_rank   = 0;
  int m_nprocs = 1;

  virtual void SetUp() {
    ::MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &m_nprocs);
  }
};

template <typename T>
struct TestShared3D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;

  int m_rank   = 0;
  int m_nprocs = 1;

  virtual void SetUp() {
    ::MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &m_nprocs);
  }
};

template <typename T, typename LayoutType>
void test_shared1D_view2D(std::size_t nprocs) {
  using RealView2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;
  using axes_type     = KokkosFFT::axis_type<1>;
  using extents_type  = std::array<std::size_t, 2>;
  using topology_type = std::array<std::size_t, 2>;

  topology_type topology0{1, nprocs};
  topology_type topology1{nprocs, 1};

  const std::size_t n0 = 8, n1 = 7;
  extents_type global_in_extents{n0, n1},
      global_out_extents_ax0{n0 / 2 + 1, n1},
      global_out_extents_ax1{n0, n1 / 2 + 1};

  auto [in_extents_t0, in_starts_t0] =
      get_local_extents(global_in_extents, topology0, MPI_COMM_WORLD);
  auto [in_extents_t1, in_starts_t1] =
      get_local_extents(global_in_extents, topology1, MPI_COMM_WORLD);
  auto [out_extents_t0_ax0, out_starts_t0_ax0] =
      get_local_extents(global_out_extents_ax0, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax0, out_starts_t1_ax0] =
      get_local_extents(global_out_extents_ax0, topology1, MPI_COMM_WORLD);
  auto [out_extents_t0_ax1, out_starts_t0_ax1] =
      get_local_extents(global_out_extents_ax1, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax1, out_starts_t1_ax1] =
      get_local_extents(global_out_extents_ax1, topology1, MPI_COMM_WORLD);

  // Make reference with a basic-API
  RealView2DType gu("gu", n0, n1), gu_inv("gu_inv", n0, n1);
  ComplexView2DType gu_hat_ax0("gu_hat_ax0", n0 / 2 + 1, n1),
      gu_hat_ax1("gu_hat_ax1", n0, n1 / 2 + 1);

  // Data in Topology 0 (X-slab)
  RealView2DType u_0("u_0",
                     KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t0)),
      u_inv_0("u_inv_0",
              KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t0)),
      ref_u_inv_0("ref_u_inv_0",
                  KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t0));
  ComplexView2DType u_hat_0_ax0(
      "u_hat_0_ax0",
      KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax0)),
      u_hat_0_ax1("u_hat_0_ax1", KokkosFFT::Impl::create_layout<LayoutType>(
                                     out_extents_t0_ax1)),
      ref_u_hat_0_ax0(
          "ref_u_hat_0_ax0",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax0)),
      ref_u_hat_0_ax1(
          "ref_u_hat_0_ax1",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax1));

  // Data in Topology 1 (Y-slab)
  RealView2DType u_1("u_1",
                     KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t1)),
      u_inv_1("u_inv_1",
              KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t1)),
      ref_u_inv_1("ref_u_inv_1",
                  KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t1));
  ComplexView2DType u_hat_1_ax0(
      "u_hat_1_ax0",
      KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax0)),
      u_hat_1_ax1("u_hat_1_ax1", KokkosFFT::Impl::create_layout<LayoutType>(
                                     out_extents_t1_ax1)),
      ref_u_hat_1_ax0(
          "ref_u_hat_1_ax0",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax0)),
      ref_u_hat_1_ax1(
          "ref_u_hat_1_ax1",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax1));

  // Initialization
  execution_space exec;
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::fill_random(gu, random_pool, 1.0);

  KokkosFFT::rfft(exec, gu, gu_hat_ax0, KokkosFFT::Normalization::backward, 0);
  KokkosFFT::rfft(exec, gu, gu_hat_ax1, KokkosFFT::Normalization::backward, 1);

  auto h_gu = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu);
  auto h_gu_hat_ax0 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax0);
  auto h_gu_hat_ax1 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax1);
  auto h_u_0             = Kokkos::create_mirror_view(u_0);
  auto h_u_1             = Kokkos::create_mirror_view(u_1);
  auto h_ref_u_hat_0_ax0 = Kokkos::create_mirror_view(ref_u_hat_0_ax0);
  auto h_ref_u_hat_0_ax1 = Kokkos::create_mirror_view(ref_u_hat_0_ax1);
  auto h_ref_u_hat_1_ax0 = Kokkos::create_mirror_view(ref_u_hat_1_ax0);
  auto h_ref_u_hat_1_ax1 = Kokkos::create_mirror_view(ref_u_hat_1_ax1);

  // Topo 0
  Kokkos::pair<std::size_t, std::size_t> range_gu0(
      in_starts_t0.at(1), in_starts_t0.at(1) + in_extents_t0.at(1));
  auto h_sub_gu_0 = Kokkos::subview(h_gu, Kokkos::ALL, range_gu0);
  Kokkos::deep_copy(h_u_0, h_sub_gu_0);

  // Topo 1
  Kokkos::pair<std::size_t, std::size_t> range_gu1(
      in_starts_t1.at(0), in_starts_t1.at(0) + in_extents_t1.at(0));
  auto h_sub_gu_1 = Kokkos::subview(h_gu, range_gu1, Kokkos::ALL);
  Kokkos::deep_copy(h_u_1, h_sub_gu_1);

  // Topo 0 -> Topo 0 ax = {0}
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax0(
      out_starts_t0_ax0.at(1),
      out_starts_t0_ax0.at(1) + out_extents_t0_ax0.at(1));
  auto h_sub_gu_hat_0_ax0 =
      Kokkos::subview(h_gu_hat_ax0, Kokkos::ALL, range_gu_hat_0_ax0);
  Kokkos::deep_copy(h_ref_u_hat_0_ax0, h_sub_gu_hat_0_ax0);

  // Topo 0 -> Topo 0 ax = {1}
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax1(
      out_starts_t0_ax1.at(1),
      out_starts_t0_ax1.at(1) + out_extents_t0_ax1.at(1));
  auto h_sub_gu_hat_0_ax1 =
      Kokkos::subview(h_gu_hat_ax1, Kokkos::ALL, range_gu_hat_0_ax1);
  Kokkos::deep_copy(h_ref_u_hat_0_ax1, h_sub_gu_hat_0_ax1);

  // Topo 1 -> Topo 1 ax = {0}
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_1_ax0(
      out_starts_t1_ax0.at(0),
      out_starts_t1_ax0.at(0) + out_extents_t1_ax0.at(0));
  auto h_sub_gu_hat_1_ax0 =
      Kokkos::subview(h_gu_hat_ax0, range_gu_hat_1_ax0, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_1_ax0, h_sub_gu_hat_1_ax0);

  // Topo 1 -> Topo 1 ax = {1}
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_1_ax1(
      out_starts_t1_ax1.at(0),
      out_starts_t1_ax1.at(0) + out_extents_t1_ax1.at(0));
  auto h_sub_gu_hat_1_ax1 =
      Kokkos::subview(h_gu_hat_ax1, range_gu_hat_1_ax1, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_1_ax1, h_sub_gu_hat_1_ax1);

  Kokkos::deep_copy(u_0, h_u_0);
  Kokkos::deep_copy(u_1, h_u_1);
  Kokkos::deep_copy(ref_u_hat_0_ax0, h_ref_u_hat_0_ax0);
  Kokkos::deep_copy(ref_u_hat_0_ax1, h_ref_u_hat_0_ax1);
  Kokkos::deep_copy(ref_u_hat_1_ax0, h_ref_u_hat_1_ax0);
  Kokkos::deep_copy(ref_u_hat_1_ax1, h_ref_u_hat_1_ax1);

  // For inverse transform
  Kokkos::deep_copy(ref_u_inv_0, u_0);
  Kokkos::deep_copy(ref_u_inv_1, u_1);

  using SharedPlanType =
      SharedPlan<execution_space, RealView2DType, ComplexView2DType, 1>;

  // topology0 -> topology1
  // (n0, n1/p) -> ((n0/2+1)/p, n1)
  // This is slab
  if (nprocs != 1) {
    ASSERT_THROW(
        {
          SharedPlanType plan_0_1_ax0(exec, u_0, u_hat_1_ax0, axes_type{0},
                                      topology0, topology1, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology1 -> topology0
    // (n0/p, n1) -> (n0/2+1, n1/p)
    // This is slab
    ASSERT_THROW(
        {
          SharedPlanType plan_1_0_ax0(exec, u_1, u_hat_0_ax0, axes_type{0},
                                      topology1, topology0, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology1 -> topology1
    // (n0/p, n1) -> ((n0/2+1)/p, n1)
    // This is slab
    ASSERT_THROW(
        {
          SharedPlanType plan_1_1_ax0(exec, u_1, u_hat_1_ax0, axes_type{0},
                                      topology1, topology1, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology0 (n0, n1/p) -> (n0/2+1, n1/p)
    ASSERT_THROW(
        {
          SharedPlanType plan_0_0_ax1(exec, u_0, u_hat_0_ax1, axes_type{1},
                                      topology0, topology0, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology0 -> topology1
    // (n0, n1/p) -> ((n0/2+1)/p, n1)
    // This is slab
    ASSERT_THROW(
        {
          SharedPlanType plan_0_1_ax1(exec, u_0, u_hat_1_ax1, axes_type{1},
                                      topology0, topology1, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology1 -> topology0
    // (n0/p, n1) -> (n0/2+1, n1/p)
    // This is slab
    ASSERT_THROW(
        {
          SharedPlanType plan_1_0_ax1(exec, u_1, u_hat_0_ax1, axes_type{1},
                                      topology1, topology0, MPI_COMM_WORLD);
        },
        std::runtime_error);
  }

  // topology0 (n0, n1/p) -> (n0/2+1, n1/p)
  SharedPlanType plan_0_0_ax0(exec, u_0, u_hat_0_ax0, axes_type{0}, topology0,
                              topology0, MPI_COMM_WORLD);
  plan_0_0_ax0.forward(u_0, u_hat_0_ax0);
  EXPECT_TRUE(allclose(exec, u_hat_0_ax0, ref_u_hat_0_ax0));

  plan_0_0_ax0.backward(u_hat_0_ax0, u_inv_0);
  EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0));

  // topology1 -> topology1
  // (n0/p, n1) -> ((n0/2+1)/p, n1)
  SharedPlanType plan_1_1_ax1(exec, u_1, u_hat_1_ax1, axes_type{1}, topology1,
                              topology1, MPI_COMM_WORLD);
  plan_1_1_ax1.forward(u_1, u_hat_1_ax1);
  EXPECT_TRUE(allclose(exec, u_hat_1_ax1, ref_u_hat_1_ax1));

  plan_1_1_ax1.backward(u_hat_1_ax1, u_inv_1);
  EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1));
}

template <typename T, typename LayoutType>
void test_shared2D_view3D(std::size_t nprocs) {
  using RealView3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<T>***, LayoutType, execution_space>;
  using axes_type     = KokkosFFT::axis_type<2>;
  using extents_type  = std::array<std::size_t, 3>;
  using topology_type = std::array<std::size_t, 3>;

  topology_type topology0{1, 1, nprocs};
  topology_type topology1{1, nprocs, 1};
  topology_type topology2{nprocs, 1, 1};

  const std::size_t n0 = 8, n1 = 7, n2 = 5;
  extents_type global_in_extents{n0, n1, n2},
      global_out_extents_ax0{n0 / 2 + 1, n1, n2},
      global_out_extents_ax1{n0, n1 / 2 + 1, n2},
      global_out_extents_ax2{n0, n1, n2 / 2 + 1};

  // Available combinations
  // Topology 0 -> Topology 0 with ax01, ax10
  // Topology 1 -> Topology 1 with ax02, ax20
  // Topology 2 -> Topology 2 with ax12, ax21

  auto [in_extents_t0, in_starts_t0] =
      get_local_extents(global_in_extents, topology0, MPI_COMM_WORLD);
  auto [in_extents_t1, in_starts_t1] =
      get_local_extents(global_in_extents, topology1, MPI_COMM_WORLD);
  auto [in_extents_t2, in_starts_t2] =
      get_local_extents(global_in_extents, topology2, MPI_COMM_WORLD);
  auto [out_extents_t0_ax0, out_starts_t0_ax0] =
      get_local_extents(global_out_extents_ax0, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax0, out_starts_t1_ax0] =
      get_local_extents(global_out_extents_ax0, topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax0, out_starts_t2_ax0] =
      get_local_extents(global_out_extents_ax0, topology2, MPI_COMM_WORLD);
  auto [out_extents_t0_ax1, out_starts_t0_ax1] =
      get_local_extents(global_out_extents_ax1, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax1, out_starts_t1_ax1] =
      get_local_extents(global_out_extents_ax1, topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax1, out_starts_t2_ax1] =
      get_local_extents(global_out_extents_ax1, topology2, MPI_COMM_WORLD);
  auto [out_extents_t0_ax2, out_starts_t0_ax2] =
      get_local_extents(global_out_extents_ax2, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax2, out_starts_t1_ax2] =
      get_local_extents(global_out_extents_ax2, topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax2, out_starts_t2_ax2] =
      get_local_extents(global_out_extents_ax2, topology2, MPI_COMM_WORLD);

  // Make reference with a basic-API
  RealView3DType gu("gu", n0, n1, n2);
  ComplexView3DType gu_hat_ax01("gu_hat_ax01", n0, n1 / 2 + 1, n2),
      gu_hat_ax02("gu_hat_ax02", n0, n1, n2 / 2 + 1),
      gu_hat_ax10("gu_hat_ax10", n0 / 2 + 1, n1, n2),
      gu_hat_ax12("gu_hat_ax12", n0, n1, n2 / 2 + 1),
      gu_hat_ax20("gu_hat_ax20", n0 / 2 + 1, n1, n2),
      gu_hat_ax21("gu_hat_ax21", n0, n1 / 2 + 1, n2);

  // Data in Topology 0 (XY-slab)
  RealView3DType u_0("u_0",
                     KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t0)),
      u_inv_0("u_inv_0",
              KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t0)),
      ref_u_inv_0("ref_u_inv_0",
                  KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t0));
  ComplexView3DType u_hat_0_ax0(
      "u_hat_0_ax0",
      KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax0)),
      u_hat_0_ax1("u_hat_0_ax1", KokkosFFT::Impl::create_layout<LayoutType>(
                                     out_extents_t0_ax1)),
      u_hat_0_ax2("u_hat_0_ax2", KokkosFFT::Impl::create_layout<LayoutType>(
                                     out_extents_t0_ax2)),
      ref_u_hat_0_ax0(
          "ref_u_hat_0_ax0",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax0)),
      ref_u_hat_0_ax1(
          "ref_u_hat_0_ax1",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax1)),
      ref_u_hat_0_ax2(
          "ref_u_hat_0_ax2",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax2));

  // Data in Topology 1 (XZ-slab)
  RealView3DType u_1("u_1",
                     KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t1)),
      u_inv_1("u_inv_1",
              KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t1)),
      ref_u_inv_1("ref_u_inv_1",
                  KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t1));
  ComplexView3DType u_hat_1_ax0(
      "u_hat_1_ax0",
      KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax0)),
      u_hat_1_ax1("u_hat_1_ax1", KokkosFFT::Impl::create_layout<LayoutType>(
                                     out_extents_t1_ax1)),
      u_hat_1_ax2("u_hat_1_ax2", KokkosFFT::Impl::create_layout<LayoutType>(
                                     out_extents_t1_ax2)),
      ref_u_hat_1_ax0(
          "ref_u_hat_1_ax0",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax0)),
      ref_u_hat_1_ax1(
          "ref_u_hat_1_ax1",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax1)),
      ref_u_hat_1_ax2(
          "ref_u_hat_1_ax2",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax2));

  // Data in Topology 2 (YZ-slab)
  RealView3DType u_2("u_2",
                     KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t2)),
      u_inv_2("u_inv_2",
              KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t2)),
      ref_u_inv_2("ref_u_inv_2",
                  KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t2));
  ComplexView3DType u_hat_2_ax0(
      "u_hat_2_ax0",
      KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax0)),
      u_hat_2_ax1("u_hat_2_ax1", KokkosFFT::Impl::create_layout<LayoutType>(
                                     out_extents_t2_ax1)),
      u_hat_2_ax2("u_hat_2_ax2", KokkosFFT::Impl::create_layout<LayoutType>(
                                     out_extents_t2_ax2)),
      ref_u_hat_2_ax0(
          "ref_u_hat_2_ax0",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax0)),
      ref_u_hat_2_ax1(
          "ref_u_hat_2_ax1",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax1)),
      ref_u_hat_2_ax2(
          "ref_u_hat_2_ax2",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax2));

  // Initialization
  execution_space exec;
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::fill_random(gu, random_pool, 1.0);

  KokkosFFT::rfft2(exec, gu, gu_hat_ax01, KokkosFFT::Normalization::backward,
                   axes_type{0, 1});
  KokkosFFT::rfft2(exec, gu, gu_hat_ax02, KokkosFFT::Normalization::backward,
                   axes_type{0, 2});
  KokkosFFT::rfft2(exec, gu, gu_hat_ax10, KokkosFFT::Normalization::backward,
                   axes_type{1, 0});
  KokkosFFT::rfft2(exec, gu, gu_hat_ax12, KokkosFFT::Normalization::backward,
                   axes_type{1, 2});
  KokkosFFT::rfft2(exec, gu, gu_hat_ax20, KokkosFFT::Normalization::backward,
                   axes_type{2, 0});
  KokkosFFT::rfft2(exec, gu, gu_hat_ax21, KokkosFFT::Normalization::backward,
                   axes_type{2, 1});

  auto h_gu = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu);
  auto h_gu_hat_ax01 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax01);
  auto h_gu_hat_ax02 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax02);
  auto h_gu_hat_ax10 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax10);
  auto h_gu_hat_ax12 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax12);
  auto h_gu_hat_ax20 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax20);
  auto h_gu_hat_ax21 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax21);
  auto h_u_0             = Kokkos::create_mirror_view(u_0);
  auto h_u_1             = Kokkos::create_mirror_view(u_1);
  auto h_u_2             = Kokkos::create_mirror_view(u_2);
  auto h_ref_u_hat_0_ax0 = Kokkos::create_mirror_view(ref_u_hat_0_ax0);
  auto h_ref_u_hat_0_ax1 = Kokkos::create_mirror_view(ref_u_hat_0_ax1);
  auto h_ref_u_hat_0_ax2 = Kokkos::create_mirror_view(ref_u_hat_0_ax2);
  auto h_ref_u_hat_1_ax0 = Kokkos::create_mirror_view(ref_u_hat_1_ax0);
  auto h_ref_u_hat_1_ax1 = Kokkos::create_mirror_view(ref_u_hat_1_ax1);
  auto h_ref_u_hat_1_ax2 = Kokkos::create_mirror_view(ref_u_hat_1_ax2);
  auto h_ref_u_hat_2_ax0 = Kokkos::create_mirror_view(ref_u_hat_2_ax0);
  auto h_ref_u_hat_2_ax1 = Kokkos::create_mirror_view(ref_u_hat_2_ax1);
  auto h_ref_u_hat_2_ax2 = Kokkos::create_mirror_view(ref_u_hat_2_ax2);

  // Topo 0
  Kokkos::pair<std::size_t, std::size_t> range_gu0(
      in_starts_t0.at(2), in_starts_t0.at(2) + in_extents_t0.at(2));
  auto h_sub_gu_0 = Kokkos::subview(h_gu, Kokkos::ALL, Kokkos::ALL, range_gu0);
  Kokkos::deep_copy(h_u_0, h_sub_gu_0);

  // Topo 1
  Kokkos::pair<std::size_t, std::size_t> range_gu1(
      in_starts_t1.at(1), in_starts_t1.at(1) + in_extents_t1.at(1));
  auto h_sub_gu_1 = Kokkos::subview(h_gu, Kokkos::ALL, range_gu1, Kokkos::ALL);
  Kokkos::deep_copy(h_u_1, h_sub_gu_1);

  // Topo 2
  Kokkos::pair<std::size_t, std::size_t> range_gu2(
      in_starts_t2.at(0), in_starts_t2.at(0) + in_extents_t2.at(0));
  auto h_sub_gu_2 = Kokkos::subview(h_gu, range_gu2, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(h_u_2, h_sub_gu_2);

  // Topo 0 -> Topo 0 ax = {1, 0}
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax0(
      out_starts_t0_ax0.at(2),
      out_starts_t0_ax0.at(2) + out_extents_t0_ax0.at(2));
  auto h_sub_gu_hat_0_ax10 = Kokkos::subview(h_gu_hat_ax10, Kokkos::ALL,
                                             Kokkos::ALL, range_gu_hat_0_ax0);
  Kokkos::deep_copy(h_ref_u_hat_0_ax0, h_sub_gu_hat_0_ax10);

  // Topo 0 -> Topo 0 ax = {0, 1}
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax1(
      out_starts_t0_ax1.at(2),
      out_starts_t0_ax1.at(2) + out_extents_t0_ax1.at(2));
  auto h_sub_gu_hat_0_ax01 = Kokkos::subview(h_gu_hat_ax01, Kokkos::ALL,
                                             Kokkos::ALL, range_gu_hat_0_ax1);
  Kokkos::deep_copy(h_ref_u_hat_0_ax1, h_sub_gu_hat_0_ax01);

  // Topo 0 -> Topo 0 ax = {0, 2}; This is slab
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax2(
      out_starts_t0_ax2.at(2),
      out_starts_t0_ax2.at(2) + out_extents_t0_ax2.at(2));
  auto h_sub_gu_hat_0_ax02 = Kokkos::subview(h_gu_hat_ax02, Kokkos::ALL,
                                             Kokkos::ALL, range_gu_hat_0_ax2);
  Kokkos::deep_copy(h_ref_u_hat_0_ax2, h_sub_gu_hat_0_ax02);

  // Topo 1 -> Topo 1 ax = {2, 0}
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_1_ax0(
      out_starts_t1_ax0.at(1),
      out_starts_t1_ax0.at(1) + out_extents_t1_ax0.at(1));
  auto h_sub_gu_hat_1_ax20 = Kokkos::subview(h_gu_hat_ax20, Kokkos::ALL,
                                             range_gu_hat_1_ax0, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_1_ax0, h_sub_gu_hat_1_ax20);

  // Topo 1 -> Topo 1 ax = {2, 1}; This is slab
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_1_ax1(
      out_starts_t1_ax1.at(1),
      out_starts_t1_ax1.at(1) + out_extents_t1_ax1.at(1));
  auto h_sub_gu_hat_1_ax21 = Kokkos::subview(h_gu_hat_ax21, Kokkos::ALL,
                                             range_gu_hat_1_ax1, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_1_ax1, h_sub_gu_hat_1_ax21);

  // Topo 1 -> Topo 1 ax = {0, 2}
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_1_ax2(
      out_starts_t1_ax2.at(1),
      out_starts_t1_ax2.at(1) + out_extents_t1_ax2.at(1));
  auto h_sub_gu_hat_1_ax02 = Kokkos::subview(h_gu_hat_ax02, Kokkos::ALL,
                                             range_gu_hat_1_ax2, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_1_ax2, h_sub_gu_hat_1_ax02);

  // Topo 2 -> Topo 2 ax = {1, 0}; This is slab
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_2_ax0(
      out_starts_t2_ax0.at(0),
      out_starts_t2_ax0.at(0) + out_extents_t2_ax0.at(0));
  auto h_sub_gu_hat_2_ax10 = Kokkos::subview(h_gu_hat_ax10, range_gu_hat_2_ax0,
                                             Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_2_ax0, h_sub_gu_hat_2_ax10);

  // Topo 2 -> Topo 2 ax = {2, 1}
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_2_ax1(
      out_starts_t2_ax1.at(0),
      out_starts_t2_ax1.at(0) + out_extents_t2_ax1.at(0));
  auto h_sub_gu_hat_2_ax21 = Kokkos::subview(h_gu_hat_ax21, range_gu_hat_2_ax1,
                                             Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_2_ax1, h_sub_gu_hat_2_ax21);

  // Topo 2 -> Topo 2 ax = {1, 2}
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_2_ax2(
      out_starts_t2_ax2.at(0),
      out_starts_t2_ax2.at(0) + out_extents_t2_ax2.at(0));
  auto h_sub_gu_hat_2_ax12 = Kokkos::subview(h_gu_hat_ax12, range_gu_hat_2_ax2,
                                             Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_2_ax2, h_sub_gu_hat_2_ax12);

  Kokkos::deep_copy(u_0, h_u_0);
  Kokkos::deep_copy(u_1, h_u_1);
  Kokkos::deep_copy(u_2, h_u_2);
  Kokkos::deep_copy(ref_u_hat_0_ax0, h_ref_u_hat_0_ax0);
  Kokkos::deep_copy(ref_u_hat_0_ax1, h_ref_u_hat_0_ax1);
  Kokkos::deep_copy(ref_u_hat_0_ax2, h_ref_u_hat_0_ax2);
  Kokkos::deep_copy(ref_u_hat_1_ax0, h_ref_u_hat_1_ax0);
  Kokkos::deep_copy(ref_u_hat_1_ax1, h_ref_u_hat_1_ax1);
  Kokkos::deep_copy(ref_u_hat_1_ax2, h_ref_u_hat_1_ax2);
  Kokkos::deep_copy(ref_u_hat_2_ax0, h_ref_u_hat_2_ax0);
  Kokkos::deep_copy(ref_u_hat_2_ax1, h_ref_u_hat_2_ax1);
  Kokkos::deep_copy(ref_u_hat_2_ax2, h_ref_u_hat_2_ax2);

  // For inverse transform
  Kokkos::deep_copy(ref_u_inv_0, u_0);
  Kokkos::deep_copy(ref_u_inv_1, u_1);
  Kokkos::deep_copy(ref_u_inv_2, u_2);

  using SharedPlanType =
      SharedPlan<execution_space, RealView3DType, ComplexView3DType, 2>;

  if (nprocs != 1) {
    // topology0 -> topology1
    // (n0, n1, n2/p) -> (n0, (n1/2+1)/p, n2)
    // This is slab
    ASSERT_THROW(
        {
          SharedPlanType plan_0_1_ax1(exec, u_0, u_hat_1_ax1, axes_type{0, 1},
                                      topology0, topology1, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology0 -> topology2
    // (n0, n1, n2/p) -> (n0/p, n1/2+1, n2)
    // This is slab
    ASSERT_THROW(
        {
          SharedPlanType plan_0_2_ax1(exec, u_0, u_hat_2_ax1, axes_type{0, 1},
                                      topology0, topology2, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology1 -> topology2
    // (n0, n1/p, n2) -> (n0/p, n1, n2/2+1)
    // This is slab
    ASSERT_THROW(
        {
          SharedPlanType plan_1_2_ax2(exec, u_1, u_hat_2_ax2, axes_type{1, 2},
                                      topology1, topology2, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology0 -> topology0 with ax = {1, 2}
    // (n0, n1, n2/p) -> (n0, n1, (n2/2+1)/p)
    // This is slab
    ASSERT_THROW(
        {
          SharedPlanType plan_0_0_ax0(exec, u_0, u_hat_0_ax2, axes_type{1, 2},
                                      topology0, topology0, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology1 -> topology1 with ax = {1, 2}
    // (n0, n1/p, n2) -> (n0, (n1/2+1)/p, n2)
    // This is slab
    ASSERT_THROW(
        {
          SharedPlanType plan_1_1_ax2(exec, u_1, u_hat_1_ax1, axes_type{2, 1},
                                      topology1, topology1, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology2 -> topology2 with ax = {0, 1}
    // (n0/p, n1, n2) -> (n0/p, (n1/2+1), n2)
    // This is slab
    ASSERT_THROW(
        {
          SharedPlanType plan_2_2_ax1(exec, u_2, u_hat_2_ax1, axes_type{0, 1},
                                      topology2, topology2, MPI_COMM_WORLD);
        },
        std::runtime_error);
  }

  // topology0 (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) axis {1, 0}
  SharedPlanType plan_0_0_ax0(exec, u_0, u_hat_0_ax0, axes_type{1, 0},
                              topology0, topology0, MPI_COMM_WORLD);
  plan_0_0_ax0.forward(u_0, u_hat_0_ax0);
  EXPECT_TRUE(allclose(exec, u_hat_0_ax0, ref_u_hat_0_ax0));

  plan_0_0_ax0.backward(u_hat_0_ax0, u_inv_0);
  EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

  // topology0 (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) axis {0, 1}
  SharedPlanType plan_0_0_ax1(exec, u_0, u_hat_0_ax1, axes_type{0, 1},
                              topology0, topology0, MPI_COMM_WORLD);
  plan_0_0_ax1.forward(u_0, u_hat_0_ax1);
  EXPECT_TRUE(allclose(exec, u_hat_0_ax1, ref_u_hat_0_ax1));

  plan_0_0_ax1.backward(u_hat_0_ax1, u_inv_0);
  EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

  // topology1 (n0, n1/p, n2) -> topology1 (n0/2+1, n1/p, n2) axis {2, 0}
  SharedPlanType plan_1_1_ax0(exec, u_1, u_hat_1_ax0, axes_type{2, 0},
                              topology1, topology1, MPI_COMM_WORLD);
  plan_1_1_ax0.forward(u_1, u_hat_1_ax0);
  EXPECT_TRUE(allclose(exec, u_hat_1_ax0, ref_u_hat_1_ax0));

  plan_1_1_ax0.backward(u_hat_1_ax0, u_inv_1);
  EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

  // topology1 (n0, n1/p, n2) -> topology1 (n0, n1/p, n2/2+1) axis {0, 2}
  SharedPlanType plan_1_1_ax2(exec, u_1, u_hat_1_ax2, axes_type{0, 2},
                              topology1, topology1, MPI_COMM_WORLD);
  plan_1_1_ax2.forward(u_1, u_hat_1_ax2);
  EXPECT_TRUE(allclose(exec, u_hat_1_ax2, ref_u_hat_1_ax2));

  plan_1_1_ax2.backward(u_hat_1_ax2, u_inv_1);
  EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

  // topology2 (n0/p, n1, n2) -> topology2 (n0/p, n1/2+1, n2) axis {2, 1}
  SharedPlanType plan_2_2_ax1(exec, u_2, u_hat_2_ax1, axes_type{2, 1},
                              topology2, topology2, MPI_COMM_WORLD);
  plan_2_2_ax1.forward(u_2, u_hat_2_ax1);
  EXPECT_TRUE(allclose(exec, u_hat_2_ax1, ref_u_hat_2_ax1));

  plan_2_2_ax1.backward(u_hat_2_ax1, u_inv_2);
  EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

  // topology2 (n0/p, n1, n2) -> topology2 (n0/p, n1, n2/2+1) axis {1, 2}
  SharedPlanType plan_2_2_ax2(exec, u_2, u_hat_2_ax2, axes_type{1, 2},
                              topology2, topology2, MPI_COMM_WORLD);
  plan_2_2_ax2.forward(u_2, u_hat_2_ax2);
  EXPECT_TRUE(allclose(exec, u_hat_2_ax2, ref_u_hat_2_ax2));

  plan_2_2_ax2.backward(u_hat_2_ax2, u_inv_2);
  EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));
}

template <typename T, typename LayoutType>
void test_shared3D_view4D(std::size_t nprocs) {
  using RealView4DType = Kokkos::View<T****, LayoutType, execution_space>;
  using ComplexView4DType =
      Kokkos::View<Kokkos::complex<T>****, LayoutType, execution_space>;
  using axes_type     = KokkosFFT::axis_type<3>;
  using extents_type  = std::array<std::size_t, 4>;
  using topology_type = std::array<std::size_t, 4>;

  topology_type topology0{1, 1, 1, nprocs};
  topology_type topology1{1, 1, nprocs, 1};
  topology_type topology2{1, nprocs, 1, 1};
  topology_type topology3{nprocs, 1, 1, 1};

  const std::size_t n0 = 8, n1 = 7, n2 = 5, n3 = 6;
  extents_type global_in_extents{n0, n1, n2, n3},
      global_out_extents_ax0{n0 / 2 + 1, n1, n2, n3},
      global_out_extents_ax1{n0, n1 / 2 + 1, n2, n3},
      global_out_extents_ax2{n0, n1, n2 / 2 + 1, n3},
      global_out_extents_ax3{n0, n1, n2, n3 / 2 + 1};

  // All axes
  axes_type ax012 = {0, 1, 2}, ax013 = {0, 1, 3}, ax021 = {0, 2, 1},
            ax023 = {0, 2, 3}, ax031 = {0, 3, 1}, ax032 = {0, 3, 2},
            ax102 = {1, 0, 2}, ax103 = {1, 0, 3}, ax120 = {1, 2, 0},
            ax123 = {1, 2, 3}, ax130 = {1, 3, 0}, ax132 = {1, 3, 2},
            ax201 = {2, 0, 1}, ax203 = {2, 0, 3}, ax210 = {2, 1, 0},
            ax213 = {2, 1, 3}, ax230 = {2, 3, 0}, ax231 = {2, 3, 1},
            ax301 = {3, 0, 1}, ax302 = {3, 0, 2}, ax310 = {3, 1, 0},
            ax312 = {3, 1, 2}, ax320 = {3, 2, 0}, ax321 = {3, 2, 1};

  // Available combinations
  // Topology 0 -> Topology 0 with ax012, ax201, ...
  // Topology 1 -> Topology 1 with ax013, ax130, ...
  // Topology 2 -> Topology 2 with ax023, ax320, ...
  // Topology 3 -> Topology 3 with ax123, ax321, ...

  auto [in_extents_t0, in_starts_t0] =
      get_local_extents(global_in_extents, topology0, MPI_COMM_WORLD);
  auto [in_extents_t1, in_starts_t1] =
      get_local_extents(global_in_extents, topology1, MPI_COMM_WORLD);
  auto [in_extents_t2, in_starts_t2] =
      get_local_extents(global_in_extents, topology2, MPI_COMM_WORLD);
  auto [in_extents_t3, in_starts_t3] =
      get_local_extents(global_in_extents, topology3, MPI_COMM_WORLD);
  auto [out_extents_t0_ax0, out_starts_t0_ax0] =
      get_local_extents(global_out_extents_ax0, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax0, out_starts_t1_ax0] =
      get_local_extents(global_out_extents_ax0, topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax0, out_starts_t2_ax0] =
      get_local_extents(global_out_extents_ax0, topology2, MPI_COMM_WORLD);
  auto [out_extents_t3_ax0, out_starts_t3_ax0] =
      get_local_extents(global_out_extents_ax0, topology3, MPI_COMM_WORLD);
  auto [out_extents_t0_ax1, out_starts_t0_ax1] =
      get_local_extents(global_out_extents_ax1, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax1, out_starts_t1_ax1] =
      get_local_extents(global_out_extents_ax1, topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax1, out_starts_t2_ax1] =
      get_local_extents(global_out_extents_ax1, topology2, MPI_COMM_WORLD);
  auto [out_extents_t3_ax1, out_starts_t3_ax1] =
      get_local_extents(global_out_extents_ax1, topology3, MPI_COMM_WORLD);
  auto [out_extents_t0_ax2, out_starts_t0_ax2] =
      get_local_extents(global_out_extents_ax2, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax2, out_starts_t1_ax2] =
      get_local_extents(global_out_extents_ax2, topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax2, out_starts_t2_ax2] =
      get_local_extents(global_out_extents_ax2, topology2, MPI_COMM_WORLD);
  auto [out_extents_t3_ax2, out_starts_t3_ax2] =
      get_local_extents(global_out_extents_ax2, topology3, MPI_COMM_WORLD);
  auto [out_extents_t0_ax3, out_starts_t0_ax3] =
      get_local_extents(global_out_extents_ax3, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax3, out_starts_t1_ax3] =
      get_local_extents(global_out_extents_ax3, topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax3, out_starts_t2_ax3] =
      get_local_extents(global_out_extents_ax3, topology2, MPI_COMM_WORLD);
  auto [out_extents_t3_ax3, out_starts_t3_ax3] =
      get_local_extents(global_out_extents_ax3, topology3, MPI_COMM_WORLD);

  // Make reference with a basic-API
  RealView4DType gu("gu", n0, n1, n2, n3);
  ComplexView4DType gu_hat_ax012("gu_hat_ax012", n0, n1, n2 / 2 + 1, n3),
      gu_hat_ax013("gu_hat_ax013", n0, n1, n2, n3 / 2 + 1),
      gu_hat_ax021("gu_hat_ax021", n0, n1 / 2 + 1, n2, n3),
      gu_hat_ax023("gu_hat_ax023", n0, n1, n2, n3 / 2 + 1),
      gu_hat_ax031("gu_hat_ax031", n0, n1 / 2 + 1, n2, n3),
      gu_hat_ax032("gu_hat_ax032", n0, n1, n2 / 2 + 1, n3),
      gu_hat_ax102("gu_hat_ax102", n0, n1, n2 / 2 + 1, n3),
      gu_hat_ax103("gu_hat_ax103", n0, n1, n2, n3 / 2 + 1),
      gu_hat_ax120("gu_hat_ax120", n0 / 2 + 1, n1, n2, n3),
      gu_hat_ax123("gu_hat_ax123", n0, n1, n2, n3 / 2 + 1),
      gu_hat_ax130("gu_hat_ax130", n0 / 2 + 1, n1, n2, n3),
      gu_hat_ax132("gu_hat_ax132", n0, n1, n2 / 2 + 1, n3),
      gu_hat_ax201("gu_hat_ax201", n0, n1 / 2 + 1, n2, n3),
      gu_hat_ax203("gu_hat_ax203", n0, n1, n2, n3 / 2 + 1),
      gu_hat_ax210("gu_hat_ax210", n0 / 2 + 1, n1, n2, n3),
      gu_hat_ax213("gu_hat_ax213", n0, n1, n2, n3 / 2 + 1),
      gu_hat_ax230("gu_hat_ax230", n0 / 2 + 1, n1, n2, n3),
      gu_hat_ax231("gu_hat_ax231", n0, n1 / 2 + 1, n2, n3),
      gu_hat_ax301("gu_hat_ax301", n0, n1 / 2 + 1, n2, n3),
      gu_hat_ax302("gu_hat_ax302", n0, n1, n2 / 2 + 1, n3),
      gu_hat_ax310("gu_hat_ax310", n0 / 2 + 1, n1, n2, n3),
      gu_hat_ax312("gu_hat_ax312", n0, n1, n2 / 2 + 1, n3),
      gu_hat_ax320("gu_hat_ax320", n0 / 2 + 1, n1, n2, n3),
      gu_hat_ax321("gu_hat_ax321", n0, n1 / 2 + 1, n2, n3);

  // Data in Topology 0 (XYZ-slab)
  RealView4DType u_0("u_0",
                     KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t0)),
      u_inv_0("u_inv_0",
              KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t0)),
      ref_u_inv_0("ref_u_inv_0",
                  KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t0));
  ComplexView4DType u_hat_0_ax0(
      "u_hat_0_ax0",
      KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax0)),
      u_hat_0_ax1("u_hat_0_ax1", KokkosFFT::Impl::create_layout<LayoutType>(
                                     out_extents_t0_ax1)),
      u_hat_0_ax2("u_hat_0_ax2", KokkosFFT::Impl::create_layout<LayoutType>(
                                     out_extents_t0_ax2)),
      u_hat_0_ax3("u_hat_0_ax3", KokkosFFT::Impl::create_layout<LayoutType>(
                                     out_extents_t0_ax3)),
      ref_u_hat_0_ax0(
          "ref_u_hat_0_ax0",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax0)),
      ref_u_hat_0_ax1(
          "ref_u_hat_0_ax1",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax1)),
      ref_u_hat_0_ax2(
          "ref_u_hat_0_ax2",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax2)),
      ref_u_hat_0_ax3(
          "ref_u_hat_0_ax3",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax3));

  // Data in Topology 1 (XYW-slab)
  RealView4DType u_1("u_1",
                     KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t1)),
      u_inv_1("u_inv_1",
              KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t1)),
      ref_u_inv_1("ref_u_inv_1",
                  KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t1));
  ComplexView4DType u_hat_1_ax0(
      "u_hat_1_ax0",
      KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax0)),
      u_hat_1_ax1("u_hat_1_ax1", KokkosFFT::Impl::create_layout<LayoutType>(
                                     out_extents_t1_ax1)),
      u_hat_1_ax2("u_hat_1_ax2", KokkosFFT::Impl::create_layout<LayoutType>(
                                     out_extents_t1_ax2)),
      u_hat_1_ax3("u_hat_1_ax3", KokkosFFT::Impl::create_layout<LayoutType>(
                                     out_extents_t1_ax3)),
      ref_u_hat_1_ax0(
          "ref_u_hat_1_ax0",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax0)),
      ref_u_hat_1_ax1(
          "ref_u_hat_1_ax1",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax1)),
      ref_u_hat_1_ax2(
          "ref_u_hat_1_ax2",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax2)),
      ref_u_hat_1_ax3(
          "ref_u_hat_1_ax3",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax3));

  // Data in Topology 2 (XZW-slab)
  RealView4DType u_2("u_2",
                     KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t2)),
      u_inv_2("u_inv_2",
              KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t2)),
      ref_u_inv_2("ref_u_inv_2",
                  KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t2));
  ComplexView4DType u_hat_2_ax0(
      "u_hat_2_ax0",
      KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax0)),
      u_hat_2_ax1("u_hat_2_ax1", KokkosFFT::Impl::create_layout<LayoutType>(
                                     out_extents_t2_ax1)),
      u_hat_2_ax2("u_hat_2_ax2", KokkosFFT::Impl::create_layout<LayoutType>(
                                     out_extents_t2_ax2)),
      u_hat_2_ax3("u_hat_2_ax3", KokkosFFT::Impl::create_layout<LayoutType>(
                                     out_extents_t2_ax3)),
      ref_u_hat_2_ax0(
          "ref_u_hat_2_ax0",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax0)),
      ref_u_hat_2_ax1(
          "ref_u_hat_2_ax1",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax1)),
      ref_u_hat_2_ax2(
          "ref_u_hat_2_ax2",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax2)),
      ref_u_hat_2_ax3(
          "ref_u_hat_2_ax3",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax3));

  // Data in Topology 3 (YZW-slab)
  RealView4DType u_3("u_3",
                     KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t3)),
      u_inv_3("u_inv_3",
              KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t3)),
      ref_u_inv_3("ref_u_inv_3",
                  KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t3));
  ComplexView4DType u_hat_3_ax0(
      "u_hat_3_ax0",
      KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t3_ax0)),
      u_hat_3_ax1("u_hat_3_ax1", KokkosFFT::Impl::create_layout<LayoutType>(
                                     out_extents_t3_ax1)),
      u_hat_3_ax2("u_hat_3_ax2", KokkosFFT::Impl::create_layout<LayoutType>(
                                     out_extents_t3_ax2)),
      u_hat_3_ax3("u_hat_3_ax3", KokkosFFT::Impl::create_layout<LayoutType>(
                                     out_extents_t3_ax3)),
      ref_u_hat_3_ax0(
          "ref_u_hat_3_ax0",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t3_ax0)),
      ref_u_hat_3_ax1(
          "ref_u_hat_3_ax1",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t3_ax1)),
      ref_u_hat_3_ax2(
          "ref_u_hat_3_ax2",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t3_ax2)),
      ref_u_hat_3_ax3(
          "ref_u_hat_3_ax3",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t3_ax3));

  // Initialization
  execution_space exec;
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::fill_random(gu, random_pool, 1.0);

  KokkosFFT::rfftn(exec, gu, gu_hat_ax012, ax012,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax013, ax013,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax021, ax021,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax023, ax023,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax031, ax031,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax032, ax032,
                   KokkosFFT::Normalization::backward);

  KokkosFFT::rfftn(exec, gu, gu_hat_ax102, ax102,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax103, ax103,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax120, ax120,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax123, ax123,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax130, ax130,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax132, ax132,
                   KokkosFFT::Normalization::backward);

  KokkosFFT::rfftn(exec, gu, gu_hat_ax201, ax201,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax203, ax203,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax210, ax210,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax213, ax213,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax230, ax230,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax231, ax231,
                   KokkosFFT::Normalization::backward);

  KokkosFFT::rfftn(exec, gu, gu_hat_ax301, ax301,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax302, ax302,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax310, ax310,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax312, ax312,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax320, ax320,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax321, ax321,
                   KokkosFFT::Normalization::backward);

  auto h_gu = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu);
  auto h_gu_hat_ax012 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax012);
  auto h_gu_hat_ax013 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax013);
  auto h_gu_hat_ax021 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax021);
  auto h_gu_hat_ax023 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax023);
  auto h_gu_hat_ax031 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax031);
  auto h_gu_hat_ax032 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax032);

  auto h_gu_hat_ax102 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax102);
  auto h_gu_hat_ax103 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax103);
  auto h_gu_hat_ax120 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax120);
  auto h_gu_hat_ax123 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax123);
  auto h_gu_hat_ax130 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax130);
  auto h_gu_hat_ax132 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax132);

  auto h_gu_hat_ax201 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax201);
  auto h_gu_hat_ax203 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax203);
  auto h_gu_hat_ax210 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax210);
  auto h_gu_hat_ax213 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax213);
  auto h_gu_hat_ax230 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax230);
  auto h_gu_hat_ax231 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax231);

  auto h_gu_hat_ax301 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax301);
  auto h_gu_hat_ax302 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax302);
  auto h_gu_hat_ax310 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax310);
  auto h_gu_hat_ax312 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax312);
  auto h_gu_hat_ax320 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax320);
  auto h_gu_hat_ax321 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax321);
  auto h_u_0             = Kokkos::create_mirror_view(u_0);
  auto h_u_1             = Kokkos::create_mirror_view(u_1);
  auto h_u_2             = Kokkos::create_mirror_view(u_2);
  auto h_u_3             = Kokkos::create_mirror_view(u_3);
  auto h_ref_u_hat_0_ax0 = Kokkos::create_mirror_view(ref_u_hat_0_ax0);
  auto h_ref_u_hat_0_ax1 = Kokkos::create_mirror_view(ref_u_hat_0_ax1);
  auto h_ref_u_hat_0_ax2 = Kokkos::create_mirror_view(ref_u_hat_0_ax2);
  auto h_ref_u_hat_0_ax3 = Kokkos::create_mirror_view(ref_u_hat_0_ax3);
  auto h_ref_u_hat_1_ax0 = Kokkos::create_mirror_view(ref_u_hat_1_ax0);
  auto h_ref_u_hat_1_ax1 = Kokkos::create_mirror_view(ref_u_hat_1_ax1);
  auto h_ref_u_hat_1_ax2 = Kokkos::create_mirror_view(ref_u_hat_1_ax2);
  auto h_ref_u_hat_1_ax3 = Kokkos::create_mirror_view(ref_u_hat_1_ax3);
  auto h_ref_u_hat_2_ax0 = Kokkos::create_mirror_view(ref_u_hat_2_ax0);
  auto h_ref_u_hat_2_ax1 = Kokkos::create_mirror_view(ref_u_hat_2_ax1);
  auto h_ref_u_hat_2_ax2 = Kokkos::create_mirror_view(ref_u_hat_2_ax2);
  auto h_ref_u_hat_2_ax3 = Kokkos::create_mirror_view(ref_u_hat_2_ax3);
  auto h_ref_u_hat_3_ax0 = Kokkos::create_mirror_view(ref_u_hat_3_ax0);
  auto h_ref_u_hat_3_ax1 = Kokkos::create_mirror_view(ref_u_hat_3_ax1);
  auto h_ref_u_hat_3_ax2 = Kokkos::create_mirror_view(ref_u_hat_3_ax2);
  auto h_ref_u_hat_3_ax3 = Kokkos::create_mirror_view(ref_u_hat_3_ax3);

  // Topo 0
  Kokkos::pair<std::size_t, std::size_t> range_gu0(
      in_starts_t0.at(3), in_starts_t0.at(3) + in_extents_t0.at(3));
  auto h_sub_gu_0 =
      Kokkos::subview(h_gu, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, range_gu0);
  Kokkos::deep_copy(h_u_0, h_sub_gu_0);

  // Topo 1
  Kokkos::pair<std::size_t, std::size_t> range_gu1(
      in_starts_t1.at(2), in_starts_t1.at(2) + in_extents_t1.at(2));
  auto h_sub_gu_1 =
      Kokkos::subview(h_gu, Kokkos::ALL, Kokkos::ALL, range_gu1, Kokkos::ALL);
  Kokkos::deep_copy(h_u_1, h_sub_gu_1);

  // Topo 2
  Kokkos::pair<std::size_t, std::size_t> range_gu2(
      in_starts_t2.at(1), in_starts_t2.at(1) + in_extents_t2.at(1));
  auto h_sub_gu_2 =
      Kokkos::subview(h_gu, Kokkos::ALL, range_gu2, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(h_u_2, h_sub_gu_2);

  // Topo 3
  Kokkos::pair<std::size_t, std::size_t> range_gu3(
      in_starts_t3.at(0), in_starts_t3.at(0) + in_extents_t3.at(0));
  auto h_sub_gu_3 =
      Kokkos::subview(h_gu, range_gu3, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(h_u_3, h_sub_gu_3);

  // Topo 0 -> Topo 0 ax = {2, 1, 0}
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax0(
      out_starts_t0_ax0.at(3),
      out_starts_t0_ax0.at(3) + out_extents_t0_ax0.at(3));
  auto h_sub_gu_hat_0_ax210 =
      Kokkos::subview(h_gu_hat_ax210, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                      range_gu_hat_0_ax0);
  Kokkos::deep_copy(h_ref_u_hat_0_ax0, h_sub_gu_hat_0_ax210);

  // Topo 0 -> Topo 0 ax = {2, 0, 1}
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax1(
      out_starts_t0_ax1.at(3),
      out_starts_t0_ax1.at(3) + out_extents_t0_ax1.at(3));
  auto h_sub_gu_hat_0_ax201 =
      Kokkos::subview(h_gu_hat_ax201, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                      range_gu_hat_0_ax1);
  Kokkos::deep_copy(h_ref_u_hat_0_ax1, h_sub_gu_hat_0_ax201);

  // Topo 0 -> Topo 0 ax = {1, 0, 2}
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax2(
      out_starts_t0_ax2.at(3),
      out_starts_t0_ax2.at(3) + out_extents_t0_ax2.at(3));
  auto h_sub_gu_hat_0_ax102 =
      Kokkos::subview(h_gu_hat_ax102, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                      range_gu_hat_0_ax2);
  Kokkos::deep_copy(h_ref_u_hat_0_ax2, h_sub_gu_hat_0_ax102);

  // Topo 0 -> Topo 0 ax = {1, 2, 3}; This is slab
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax3(
      out_starts_t0_ax3.at(3),
      out_starts_t0_ax3.at(3) + out_extents_t0_ax3.at(3));
  auto h_sub_gu_hat_0_ax123 =
      Kokkos::subview(h_gu_hat_ax123, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                      range_gu_hat_0_ax3);
  Kokkos::deep_copy(h_ref_u_hat_0_ax3, h_sub_gu_hat_0_ax123);

  // Topo 1 -> Topo 1 ax = {3, 1, 0}
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_1_ax0(
      out_starts_t1_ax0.at(2),
      out_starts_t1_ax0.at(2) + out_extents_t1_ax0.at(2));
  auto h_sub_gu_hat_1_ax310 =
      Kokkos::subview(h_gu_hat_ax310, Kokkos::ALL, Kokkos::ALL,
                      range_gu_hat_1_ax0, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_1_ax0, h_sub_gu_hat_1_ax310);

  // Topo 1 -> Topo 1 ax = {3, 0, 1}
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_1_ax1(
      out_starts_t1_ax1.at(2),
      out_starts_t1_ax1.at(2) + out_extents_t1_ax1.at(2));
  auto h_sub_gu_hat_1_ax301 =
      Kokkos::subview(h_gu_hat_ax301, Kokkos::ALL, Kokkos::ALL,
                      range_gu_hat_1_ax1, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_1_ax1, h_sub_gu_hat_1_ax301);

  // Topo 1 -> Topo 1 ax = {1, 0, 3}
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_1_ax3(
      out_starts_t1_ax3.at(2),
      out_starts_t1_ax3.at(2) + out_extents_t1_ax3.at(2));
  auto h_sub_gu_hat_1_ax103 =
      Kokkos::subview(h_gu_hat_ax103, Kokkos::ALL, Kokkos::ALL,
                      range_gu_hat_1_ax3, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_1_ax3, h_sub_gu_hat_1_ax103);

  // Topo 1 -> Topo 1 ax = {1, 3, 2}; This is slab
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_1_ax2(
      out_starts_t1_ax2.at(2),
      out_starts_t1_ax2.at(2) + out_extents_t1_ax2.at(2));
  auto h_sub_gu_hat_1_ax132 =
      Kokkos::subview(h_gu_hat_ax132, Kokkos::ALL, Kokkos::ALL,
                      range_gu_hat_1_ax2, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_1_ax2, h_sub_gu_hat_1_ax132);

  // Topo 2 -> Topo 2 ax = {2, 3, 0}
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_2_ax0(
      out_starts_t2_ax0.at(1),
      out_starts_t2_ax0.at(1) + out_extents_t2_ax0.at(1));
  auto h_sub_gu_hat_2_ax230 =
      Kokkos::subview(h_gu_hat_ax230, Kokkos::ALL, range_gu_hat_2_ax0,
                      Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_2_ax0, h_sub_gu_hat_2_ax230);

  // Topo 2 -> Topo 2 ax = {3, 0, 2}
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_2_ax2(
      out_starts_t2_ax2.at(1),
      out_starts_t2_ax2.at(1) + out_extents_t2_ax2.at(1));
  auto h_sub_gu_hat_2_ax302 =
      Kokkos::subview(h_gu_hat_ax302, Kokkos::ALL, range_gu_hat_2_ax2,
                      Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_2_ax2, h_sub_gu_hat_2_ax302);

  // Topo 2 -> Topo 2 ax = {2, 0, 3}
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_2_ax3(
      out_starts_t2_ax3.at(1),
      out_starts_t2_ax3.at(1) + out_extents_t2_ax3.at(1));
  auto h_sub_gu_hat_2_ax203 =
      Kokkos::subview(h_gu_hat_ax203, Kokkos::ALL, range_gu_hat_2_ax3,
                      Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_2_ax3, h_sub_gu_hat_2_ax203);

  // Topo 2 -> Topo 2 ax = {3, 2, 1}; This is slab
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_2_ax1(
      out_starts_t2_ax1.at(1),
      out_starts_t2_ax1.at(1) + out_extents_t2_ax1.at(1));
  auto h_sub_gu_hat_2_ax321 =
      Kokkos::subview(h_gu_hat_ax321, Kokkos::ALL, range_gu_hat_2_ax1,
                      Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_2_ax1, h_sub_gu_hat_2_ax321);

  // Topo 3 -> Topo 3 ax = {3, 2, 1}
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_3_ax1(
      out_starts_t3_ax1.at(0),
      out_starts_t3_ax1.at(0) + out_extents_t3_ax1.at(0));
  auto h_sub_gu_hat_3_ax321 =
      Kokkos::subview(h_gu_hat_ax321, range_gu_hat_3_ax1, Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_3_ax1, h_sub_gu_hat_3_ax321);

  // Topo 3 -> Topo 3 ax = {3, 1, 2}
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_3_ax2(
      out_starts_t3_ax2.at(0),
      out_starts_t3_ax2.at(0) + out_extents_t3_ax2.at(0));
  auto h_sub_gu_hat_ax312 =
      Kokkos::subview(h_gu_hat_ax312, range_gu_hat_3_ax2, Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_3_ax2, h_sub_gu_hat_ax312);

  // Topo 3 -> Topo 3 ax = {1, 2, 3}
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_3_ax3(
      out_starts_t3_ax3.at(0),
      out_starts_t3_ax3.at(0) + out_extents_t3_ax3.at(0));
  auto h_sub_gu_hat_ax123 =
      Kokkos::subview(h_gu_hat_ax123, range_gu_hat_3_ax3, Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_3_ax3, h_sub_gu_hat_ax123);

  // Topo 3 -> Topo 3 ax = {1, 2, 0}
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_3_ax0(
      out_starts_t3_ax0.at(0),
      out_starts_t3_ax0.at(0) + out_extents_t3_ax0.at(0));
  auto h_sub_gu_hat_ax120 =
      Kokkos::subview(h_gu_hat_ax120, range_gu_hat_3_ax0, Kokkos::ALL,
                      Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_3_ax0, h_sub_gu_hat_ax120);

  Kokkos::deep_copy(u_0, h_u_0);
  Kokkos::deep_copy(u_1, h_u_1);
  Kokkos::deep_copy(u_2, h_u_2);
  Kokkos::deep_copy(u_3, h_u_3);
  Kokkos::deep_copy(ref_u_hat_0_ax0, h_ref_u_hat_0_ax0);
  Kokkos::deep_copy(ref_u_hat_0_ax1, h_ref_u_hat_0_ax1);
  Kokkos::deep_copy(ref_u_hat_0_ax2, h_ref_u_hat_0_ax2);
  Kokkos::deep_copy(ref_u_hat_0_ax3, h_ref_u_hat_0_ax3);
  Kokkos::deep_copy(ref_u_hat_1_ax0, h_ref_u_hat_1_ax0);
  Kokkos::deep_copy(ref_u_hat_1_ax1, h_ref_u_hat_1_ax1);
  Kokkos::deep_copy(ref_u_hat_1_ax2, h_ref_u_hat_1_ax2);
  Kokkos::deep_copy(ref_u_hat_1_ax3, h_ref_u_hat_1_ax3);
  Kokkos::deep_copy(ref_u_hat_2_ax0, h_ref_u_hat_2_ax0);
  Kokkos::deep_copy(ref_u_hat_2_ax1, h_ref_u_hat_2_ax1);
  Kokkos::deep_copy(ref_u_hat_2_ax2, h_ref_u_hat_2_ax2);
  Kokkos::deep_copy(ref_u_hat_2_ax3, h_ref_u_hat_2_ax3);
  Kokkos::deep_copy(ref_u_hat_3_ax0, h_ref_u_hat_3_ax0);
  Kokkos::deep_copy(ref_u_hat_3_ax1, h_ref_u_hat_3_ax1);
  Kokkos::deep_copy(ref_u_hat_3_ax2, h_ref_u_hat_3_ax2);
  Kokkos::deep_copy(ref_u_hat_3_ax3, h_ref_u_hat_3_ax3);

  // For inverse transform
  Kokkos::deep_copy(ref_u_inv_0, u_0);
  Kokkos::deep_copy(ref_u_inv_1, u_1);
  Kokkos::deep_copy(ref_u_inv_2, u_2);
  Kokkos::deep_copy(ref_u_inv_3, u_3);

  using SharedPlanType =
      SharedPlan<execution_space, RealView4DType, ComplexView4DType, 3>;

  if (nprocs != 1) {
    // topology0 -> topology1
    // (n0, n1, n2, n3/p) -> (n0, n1, (n2/2+1)/p, n3)
    ASSERT_THROW(
        {
          SharedPlanType plan_0_1_ax1(exec, u_0, u_hat_1_ax1,
                                      axes_type{0, 1, 2}, topology0, topology1,
                                      MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology0 -> topology2
    // (n0, n1, n2, n3/p) -> (n0, (n1/2+1)/p, n2, n3)
    ASSERT_THROW(
        {
          SharedPlanType plan_0_2_ax1(exec, u_0, u_hat_2_ax1,
                                      axes_type{0, 2, 1}, topology0, topology2,
                                      MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology0 -> topology3
    // (n0, n1, n2, n3/p) -> ((n0/2+1)/p, n1, n2, n3)
    ASSERT_THROW(
        {
          SharedPlanType plan_0_3_ax0(exec, u_0, u_hat_3_ax0,
                                      axes_type{2, 1, 0}, topology0, topology3,
                                      MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology1 -> topology2
    // (n0, n1, n2/p, n3) -> (n0, n1/p, n2, n3/2+1)
    ASSERT_THROW(
        {
          SharedPlanType plan_1_2_ax3(exec, u_1, u_hat_2_ax3,
                                      axes_type{1, 2, 3}, topology1, topology2,
                                      MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology1 -> topology3
    // (n0, n1, n2/p, n3) -> (n0/p, n1, n2, n3/2+1)
    ASSERT_THROW(
        {
          SharedPlanType plan_1_3_ax3(exec, u_1, u_hat_3_ax3,
                                      axes_type{1, 2, 3}, topology1, topology3,
                                      MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology2 -> topology3
    // (n0, n1/p, n2, n3) -> (n0/p, n1, n2, n3/2+1)
    ASSERT_THROW(
        {
          SharedPlanType plan_2_3_ax3(exec, u_2, u_hat_3_ax3,
                                      axes_type{1, 2, 3}, topology1, topology3,
                                      MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology0 -> topology0 with ax = {1, 2, 3}
    // (n0, n1, n2, n3/p) -> (n0, n1, n2, (n3/2+1)/p)
    // This is slab
    ASSERT_THROW(
        {
          SharedPlanType plan_0_0_ax3(exec, u_0, u_hat_0_ax3,
                                      axes_type{1, 2, 3}, topology0, topology0,
                                      MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology1 -> topology1 with ax = {1, 3, 2}
    // (n0, n1, n2/p, n3) -> (n0, n1, (n2/2+1)/p, n3)
    // This is slab
    ASSERT_THROW(
        {
          SharedPlanType plan_1_1_ax2(exec, u_1, u_hat_1_ax2,
                                      axes_type{1, 3, 2}, topology1, topology1,
                                      MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology2 -> topology2 with ax = {0, 2, 1}
    // (n0, n1/p, n2, n3) -> (n0, (n1/2+1)/p, n2, n3)
    // This is slab
    ASSERT_THROW(
        {
          SharedPlanType plan_2_2_ax1(exec, u_2, u_hat_2_ax1,
                                      axes_type{0, 2, 1}, topology2, topology2,
                                      MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology3 -> topology3 with ax = {2, 1, 0}
    // (n0/p, n1, n2, n3) -> ((n0/2+1)/p, n1, n2, n3)
    // This is slab
    ASSERT_THROW(
        {
          SharedPlanType plan_3_3_ax0(exec, u_3, u_hat_3_ax0,
                                      axes_type{2, 1, 0}, topology3, topology3,
                                      MPI_COMM_WORLD);
        },
        std::runtime_error);
  }

  // topology0 (n0, n1, n2, n3/p) -> (n0/2+1, n1, n2, n3/p) axis {2, 1, 0}
  SharedPlanType plan_0_0_ax0(exec, u_0, u_hat_0_ax0, ax210, topology0,
                              topology0, MPI_COMM_WORLD);
  plan_0_0_ax0.forward(u_0, u_hat_0_ax0);
  EXPECT_TRUE(allclose(exec, u_hat_0_ax0, ref_u_hat_0_ax0));

  plan_0_0_ax0.backward(u_hat_0_ax0, u_inv_0);
  EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

  // topology0 (n0, n1, n2, n3/p) -> (n0, n1/2+1, n2, n3/p) axis {2, 0, 1}
  SharedPlanType plan_0_0_ax1(exec, u_0, u_hat_0_ax1, ax201, topology0,
                              topology0, MPI_COMM_WORLD);
  plan_0_0_ax1.forward(u_0, u_hat_0_ax1);
  EXPECT_TRUE(allclose(exec, u_hat_0_ax1, ref_u_hat_0_ax1));

  plan_0_0_ax1.backward(u_hat_0_ax1, u_inv_0);
  EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

  // topology0 (n0, n1, n2, n3/p) -> (n0, n1, n2/2+1, n3/p) axis {0, 1, 2}
  SharedPlanType plan_0_0_ax2(exec, u_0, u_hat_0_ax2, ax012, topology0,
                              topology0, MPI_COMM_WORLD);
  plan_0_0_ax2.forward(u_0, u_hat_0_ax2);
  EXPECT_TRUE(allclose(exec, u_hat_0_ax2, ref_u_hat_0_ax2));

  plan_0_0_ax2.backward(u_hat_0_ax2, u_inv_0);
  EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

  // topology1 (n0, n1, n2/p, n3) -> topology1 (n0/2+1, n1, n2/p, n3) axis {3,
  // 1, 0}
  SharedPlanType plan_1_1_ax0(exec, u_1, u_hat_1_ax0, ax310, topology1,
                              topology1, MPI_COMM_WORLD);
  plan_1_1_ax0.forward(u_1, u_hat_1_ax0);
  EXPECT_TRUE(allclose(exec, u_hat_1_ax0, ref_u_hat_1_ax0));

  plan_1_1_ax0.backward(u_hat_1_ax0, u_inv_1);
  EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

  // topology1 (n0, n1, n2/p, n3) -> topology1 (n0, n1/2+1, n2/p, n3) axis {0,
  // 3, 1}
  SharedPlanType plan_1_1_ax1(exec, u_1, u_hat_1_ax1, ax031, topology1,
                              topology1, MPI_COMM_WORLD);
  plan_1_1_ax1.forward(u_1, u_hat_1_ax1);
  EXPECT_TRUE(allclose(exec, u_hat_1_ax1, ref_u_hat_1_ax1));

  plan_1_1_ax1.backward(u_hat_1_ax1, u_inv_1);
  EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

  // topology1 (n0, n1, n2/p, n3) -> topology1 (n0, n1, n2/p, n3/2+1) axis {0,
  // 1, 3}
  SharedPlanType plan_1_1_ax3(exec, u_1, u_hat_1_ax3, ax013, topology1,
                              topology1, MPI_COMM_WORLD);
  plan_1_1_ax3.forward(u_1, u_hat_1_ax3);
  EXPECT_TRUE(allclose(exec, u_hat_1_ax3, ref_u_hat_1_ax3));

  plan_1_1_ax3.backward(u_hat_1_ax3, u_inv_1);
  EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

  // topology2 (n0, n1/p, n2, n3) -> topology2 (n0/2+1, n1/p, n2, n3) axis {2,
  // 3, 0}
  SharedPlanType plan_2_2_ax0(exec, u_2, u_hat_2_ax0, ax230, topology2,
                              topology2, MPI_COMM_WORLD);
  plan_2_2_ax0.forward(u_2, u_hat_2_ax0);
  EXPECT_TRUE(allclose(exec, u_hat_2_ax0, ref_u_hat_2_ax0));

  plan_2_2_ax0.backward(u_hat_2_ax0, u_inv_2);
  EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

  // topology2 (n0, n1/p, n2, n3) -> topology2 (n0, n1/p, n2/2+1, n3) axis {3,
  // 0, 2}
  SharedPlanType plan_2_2_ax2(exec, u_2, u_hat_2_ax2, ax302, topology2,
                              topology2, MPI_COMM_WORLD);
  plan_2_2_ax2.forward(u_2, u_hat_2_ax2);
  EXPECT_TRUE(allclose(exec, u_hat_2_ax2, ref_u_hat_2_ax2));

  plan_2_2_ax2.backward(u_hat_2_ax2, u_inv_2);
  EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

  // topology2 (n0, n1/p, n2, n3) -> topology2 (n0, n1/p, n2, n3/2+1) axis {2,
  // 0, 3}
  SharedPlanType plan_2_2_ax3(exec, u_2, u_hat_2_ax3, ax203, topology2,
                              topology2, MPI_COMM_WORLD);
  plan_2_2_ax3.forward(u_2, u_hat_2_ax3);
  EXPECT_TRUE(allclose(exec, u_hat_2_ax3, ref_u_hat_2_ax3));

  plan_2_2_ax3.backward(u_hat_2_ax3, u_inv_2);
  EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

  // topology3 (n0/p, n1, n2, n3) -> topology3 (n0/p, n1/2+1, n2, n3) axis {2,
  // 3, 1}
  SharedPlanType plan_3_3_ax1(exec, u_3, u_hat_3_ax1, ax231, topology3,
                              topology3, MPI_COMM_WORLD);
  plan_3_3_ax1.forward(u_3, u_hat_3_ax1);
  EXPECT_TRUE(allclose(exec, u_hat_3_ax1, ref_u_hat_3_ax1));

  plan_3_3_ax1.backward(u_hat_3_ax1, u_inv_3);
  EXPECT_TRUE(allclose(exec, u_inv_3, ref_u_inv_3, 1.0e-5, 1.0e-6));

  // topology3 (n0/p, n1, n2, n3) -> topology3 (n0/p, n1, n2/2+1, n3) axis {3,
  // 1, 2}
  SharedPlanType plan_3_3_ax2(exec, u_3, u_hat_3_ax2, ax312, topology3,
                              topology3, MPI_COMM_WORLD);
  plan_3_3_ax2.forward(u_3, u_hat_3_ax2);
  EXPECT_TRUE(allclose(exec, u_hat_3_ax2, ref_u_hat_3_ax2));

  plan_3_3_ax2.backward(u_hat_3_ax2, u_inv_3);
  EXPECT_TRUE(allclose(exec, u_inv_3, ref_u_inv_3, 1.0e-5, 1.0e-6));

  // topology3 (n0/p, n1, n2, n3) -> topology2 (n0/p, n1, n2, n3/2+1) axis {1,
  // 2, 3}
  SharedPlanType plan_3_3_ax3(exec, u_3, u_hat_3_ax3, ax123, topology3,
                              topology3, MPI_COMM_WORLD);
  plan_3_3_ax3.forward(u_3, u_hat_3_ax3);
  EXPECT_TRUE(allclose(exec, u_hat_3_ax3, ref_u_hat_3_ax3));

  plan_3_3_ax3.backward(u_hat_3_ax3, u_inv_3);
  EXPECT_TRUE(allclose(exec, u_inv_3, ref_u_inv_3, 1.0e-5, 1.0e-6));
}

}  // namespace

TYPED_TEST_SUITE(TestShared1D, test_types);
TYPED_TEST_SUITE(TestShared2D, test_types);
TYPED_TEST_SUITE(TestShared3D, test_types);

TYPED_TEST(TestShared1D, View2D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_shared1D_view2D<float_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestShared2D, View3D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_shared2D_view3D<float_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestShared3D, View4D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_shared3D_view4D<float_type, layout_type>(this->m_nprocs);
}
