#include <mpi.h>
#include <gtest/gtest.h>
#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include "SlabPlan.hpp"
#include "MPI_Helper.hpp"
#include "Test_Utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using test_types      = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>,
                                    std::pair<double, Kokkos::LayoutLeft>,
                                    std::pair<double, Kokkos::LayoutRight>>;

//  Basically the same fixtures, used for labeling tests
template <typename T>
struct TestSlab1D : public ::testing::Test {
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
struct TestSlab2D : public ::testing::Test {
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
struct TestSlab3D : public ::testing::Test {
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
void test_slab1D_view2D(std::size_t nprocs) {
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

  // Topo 0
  Kokkos::pair<std::size_t, std::size_t> range_gu0(
      in_starts_t0.at(1), in_starts_t0.at(1) + in_extents_t0.at(1));
  auto sub_gu_0 = Kokkos::subview(gu, Kokkos::ALL, range_gu0);
  Kokkos::deep_copy(u_0, sub_gu_0);

  // Topo 1
  Kokkos::pair<std::size_t, std::size_t> range_gu1(
      in_starts_t1.at(0), in_starts_t1.at(0) + in_extents_t1.at(0));
  auto sub_gu_1 = Kokkos::subview(gu, range_gu1, Kokkos::ALL);
  Kokkos::deep_copy(u_1, sub_gu_1);

  // Topo 0 -> Topo 0 ax = {0}
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax0(
      out_starts_t0_ax0.at(1),
      out_starts_t0_ax0.at(1) + out_extents_t0_ax0.at(1));
  auto sub_gu_hat_0_ax0 =
      Kokkos::subview(gu_hat_ax0, Kokkos::ALL, range_gu_hat_0_ax0);
  Kokkos::deep_copy(ref_u_hat_0_ax0, sub_gu_hat_0_ax0);

  // Topo 0 -> Topo 0 ax = {1}
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax1(
      out_starts_t0_ax1.at(1),
      out_starts_t0_ax1.at(1) + out_extents_t0_ax1.at(1));
  auto sub_gu_hat_0_ax1 =
      Kokkos::subview(gu_hat_ax1, Kokkos::ALL, range_gu_hat_0_ax1);
  Kokkos::deep_copy(ref_u_hat_0_ax1, sub_gu_hat_0_ax1);

  // Topo 1 -> Topo 1 ax = {0}
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_1_ax0(
      out_starts_t1_ax0.at(0),
      out_starts_t1_ax0.at(0) + out_extents_t1_ax0.at(0));
  auto sub_gu_hat_1_ax0 =
      Kokkos::subview(gu_hat_ax0, range_gu_hat_1_ax0, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_1_ax0, sub_gu_hat_1_ax0);

  // Topo 1 -> Topo 1 ax = {1}
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_1_ax1(
      out_starts_t1_ax1.at(0),
      out_starts_t1_ax1.at(0) + out_extents_t1_ax1.at(0));
  auto sub_gu_hat_1_ax1 =
      Kokkos::subview(gu_hat_ax1, range_gu_hat_1_ax1, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_1_ax1, sub_gu_hat_1_ax1);

  // For inverse transform
  Kokkos::deep_copy(ref_u_inv_0, u_0);
  Kokkos::deep_copy(ref_u_inv_1, u_1);

  using SlabPlanType =
      SlabPlan<execution_space, RealView2DType, ComplexView2DType, 1>;

  // Not a slab geometry
  if (nprocs == 1) {
    ASSERT_THROW(
        {
          SlabPlanType plan_0_1_ax0(exec, u_0, u_hat_1_ax0, axes_type{0},
                                    topology0, topology1, MPI_COMM_WORLD);
        },
        std::runtime_error);
    ASSERT_THROW(
        {
          SlabPlanType plan_0_1_ax1(exec, u_0, u_hat_1_ax1, axes_type{1},
                                    topology0, topology1, MPI_COMM_WORLD);
        },
        std::runtime_error);
    ASSERT_THROW(
        {
          SlabPlanType plan_1_0_ax0(exec, u_1, u_hat_0_ax0, axes_type{0},
                                    topology1, topology0, MPI_COMM_WORLD);
        },
        std::runtime_error);
    ASSERT_THROW(
        {
          SlabPlanType plan_1_0_ax1(exec, u_1, u_hat_0_ax1, axes_type{1},
                                    topology1, topology0, MPI_COMM_WORLD);
        },
        std::runtime_error);
  } else {
    // topo0 -> topo0 with ax = {1}:
    // (n0, n1/p) -> (n0/p, n1) -> ((n0/2+1)/p, n1)
    // Transpose + FFT ax = {1} + Transpose
    SlabPlanType plan_0_0_ax1(exec, u_0, u_hat_0_ax1, axes_type{1}, topology0,
                              topology0, MPI_COMM_WORLD);
    plan_0_0_ax1.forward(u_0, u_hat_0_ax1);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax1, ref_u_hat_0_ax1));

    plan_0_0_ax1.backward(u_hat_0_ax1, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0));

    // topo 0 -> topo 1 with ax = {0}:
    // (n0, n1/p) -> (n0/2+1, n1/p) -> ((n0 / 2 + 1) / p, n1)
    // FFT ax = {0} + Transpose
    SlabPlanType plan_0_1_ax0(exec, u_0, u_hat_1_ax0, axes_type{0}, topology0,
                              topology1, MPI_COMM_WORLD);
    plan_0_1_ax0.forward(u_0, u_hat_1_ax0);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax0, ref_u_hat_1_ax0));

    plan_0_1_ax0.backward(u_hat_1_ax0, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0));

    // topology0 (n0, n1/p), ax=1 -> topology1 (n0/p, n1/2+1)
    // topo0 -> topo1 with ax = {1}: (n0, n1/p) -> (n0/p, n1) -> (n0/p, n1/2+1)
    // Transpose + FFT ax = {1}
    SlabPlanType plan_0_1_ax1(exec, u_0, u_hat_1_ax1, axes_type{1}, topology0,
                              topology1, MPI_COMM_WORLD);
    plan_0_1_ax1.forward(u_0, u_hat_1_ax1);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax1, ref_u_hat_1_ax1));

    plan_0_1_ax1.backward(u_hat_1_ax1, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0));

    // topo1 -> topo0 with ax = {0}: (n0/p, n1) -> (n0/2+1, n1/p)
    // Transpose + FFT ax = {0}
    SlabPlanType plan_1_0_ax0(exec, u_1, u_hat_0_ax0, axes_type{0}, topology1,
                              topology0, MPI_COMM_WORLD);
    plan_1_0_ax0.forward(u_1, u_hat_0_ax0);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax0, ref_u_hat_0_ax0));

    plan_1_0_ax0.backward(u_hat_0_ax0, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1));

    // topo1 -> topo0 with ax = {1}: (n0/p, n1) -> (n0, n1/p)
    // FFT ax = {1} -> Transpose
    SlabPlanType plan_1_0_ax1(exec, u_1, u_hat_0_ax1, axes_type{1}, topology1,
                              topology0, MPI_COMM_WORLD);
    plan_1_0_ax1.forward(u_1, u_hat_0_ax1);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax1, ref_u_hat_0_ax1));

    plan_1_0_ax1.backward(u_hat_0_ax1, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1));

    // topo1 -> topo1 with ax = {0}: (n0/p, n1) -> (n0, n1/p) -> (n0/2+1, n1/p)
    // Transpose + FFT ax = {0} + Transpose
    SlabPlanType plan_1_1_ax0(exec, u_1, u_hat_1_ax0, axes_type{0}, topology1,
                              topology1, MPI_COMM_WORLD);
    plan_1_1_ax0.forward(u_1, u_hat_1_ax0);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax0, ref_u_hat_1_ax0));

    plan_1_1_ax0.backward(u_hat_1_ax0, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1));
  }
}

template <typename T, typename LayoutType>
void test_slab1D_view3D(std::size_t nprocs) {
  using RealView3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<T>***, LayoutType, execution_space>;
  using axes_type     = KokkosFFT::axis_type<1>;
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

  axes_type ax0 = {0}, ax1 = {1}, ax2 = {2};

  // Available combinations
  // Topology 0 -> Topology 1 with all axes
  // Topology 0 -> Topology 2 with all axes
  // Topology 1 -> Topology 2 with all axes

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
  ComplexView3DType gu_hat_ax0("gu_hat_ax0", n0 / 2 + 1, n1, n2),
      gu_hat_ax1("gu_hat_ax1", n0, n1 / 2 + 1, n2),
      gu_hat_ax2("gu_hat_ax2", n0, n1, n2 / 2 + 1);

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

  KokkosFFT::rfft(exec, gu, gu_hat_ax0, KokkosFFT::Normalization::backward, 0);
  KokkosFFT::rfft(exec, gu, gu_hat_ax1, KokkosFFT::Normalization::backward, 1);
  KokkosFFT::rfft(exec, gu, gu_hat_ax2, KokkosFFT::Normalization::backward, 2);

  // Topo 0
  Kokkos::pair<std::size_t, std::size_t> range_gu0(
      in_starts_t0.at(2), in_starts_t0.at(2) + in_extents_t0.at(2));
  auto sub_gu_0 = Kokkos::subview(gu, Kokkos::ALL, Kokkos::ALL, range_gu0);
  Kokkos::deep_copy(u_0, sub_gu_0);

  // Topo 1
  Kokkos::pair<std::size_t, std::size_t> range_gu1(
      in_starts_t1.at(1), in_starts_t1.at(1) + in_extents_t1.at(1));
  auto sub_gu_1 = Kokkos::subview(gu, Kokkos::ALL, range_gu1, Kokkos::ALL);
  Kokkos::deep_copy(u_1, sub_gu_1);

  // Topo 2
  Kokkos::pair<std::size_t, std::size_t> range_gu2(
      in_starts_t2.at(0), in_starts_t2.at(0) + in_extents_t2.at(0));
  auto sub_gu_2 = Kokkos::subview(gu, range_gu2, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(u_2, sub_gu_2);

  // Define ranges for topology 0 (XY-slab)
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax0(
      out_starts_t0_ax0.at(2),
      out_starts_t0_ax0.at(2) + out_extents_t0_ax0.at(2));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax1(
      out_starts_t0_ax1.at(2),
      out_starts_t0_ax1.at(2) + out_extents_t0_ax1.at(2));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax2(
      out_starts_t0_ax2.at(2),
      out_starts_t0_ax2.at(2) + out_extents_t0_ax2.at(2));

  // Topo 0 ax = {0}
  auto sub_gu_hat_0_ax0 =
      Kokkos::subview(gu_hat_ax0, Kokkos::ALL, Kokkos::ALL, range_gu_hat_0_ax0);
  Kokkos::deep_copy(ref_u_hat_0_ax0, sub_gu_hat_0_ax0);

  // Topo 0 ax = {1}
  auto sub_gu_hat_0_ax1 =
      Kokkos::subview(gu_hat_ax1, Kokkos::ALL, Kokkos::ALL, range_gu_hat_0_ax1);
  Kokkos::deep_copy(ref_u_hat_0_ax1, sub_gu_hat_0_ax1);

  // Topo 0 ax = {2}
  auto sub_gu_hat_0_ax2 =
      Kokkos::subview(gu_hat_ax2, Kokkos::ALL, Kokkos::ALL, range_gu_hat_0_ax2);
  Kokkos::deep_copy(ref_u_hat_0_ax2, sub_gu_hat_0_ax2);

  // Define ranges for topology 1 (XZ-slab)
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_1_ax0(
      out_starts_t1_ax0.at(1),
      out_starts_t1_ax0.at(1) + out_extents_t1_ax0.at(1));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_1_ax1(
      out_starts_t1_ax1.at(1),
      out_starts_t1_ax1.at(1) + out_extents_t1_ax1.at(1));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_1_ax2(
      out_starts_t1_ax2.at(1),
      out_starts_t1_ax2.at(1) + out_extents_t1_ax2.at(1));

  // Topo 1 ax = {0}
  auto sub_gu_hat_1_ax0 =
      Kokkos::subview(gu_hat_ax0, Kokkos::ALL, range_gu_hat_1_ax0, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_1_ax0, sub_gu_hat_1_ax0);

  // Topo 1 ax = {1}
  auto sub_gu_hat_1_ax1 =
      Kokkos::subview(gu_hat_ax1, Kokkos::ALL, range_gu_hat_1_ax1, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_1_ax1, sub_gu_hat_1_ax1);

  // Topo 1 ax = {2}
  auto sub_gu_hat_1_ax2 =
      Kokkos::subview(gu_hat_ax2, Kokkos::ALL, range_gu_hat_1_ax2, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_1_ax2, sub_gu_hat_1_ax2);

  // Define ranges for topology 2 (YZ-slab)
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_2_ax0(
      out_starts_t2_ax0.at(0),
      out_starts_t2_ax0.at(0) + out_extents_t2_ax0.at(0));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_2_ax1(
      out_starts_t2_ax1.at(0),
      out_starts_t2_ax1.at(0) + out_extents_t2_ax1.at(0));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_2_ax2(
      out_starts_t2_ax2.at(0),
      out_starts_t2_ax2.at(0) + out_extents_t2_ax2.at(0));

  // Topo 2 ax = {0}
  auto sub_gu_hat_2_ax0 =
      Kokkos::subview(gu_hat_ax0, range_gu_hat_2_ax0, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax0, sub_gu_hat_2_ax0);

  // Topo 2 ax = {1}
  auto sub_gu_hat_2_ax1 =
      Kokkos::subview(gu_hat_ax1, range_gu_hat_2_ax1, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax1, sub_gu_hat_2_ax1);

  // Topo 2 ax = {2}
  auto sub_gu_hat_2_ax2 =
      Kokkos::subview(gu_hat_ax2, range_gu_hat_2_ax2, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax2, sub_gu_hat_2_ax2);

  // For inverse transform
  Kokkos::deep_copy(ref_u_inv_0, u_0);
  Kokkos::deep_copy(ref_u_inv_1, u_1);
  Kokkos::deep_copy(ref_u_inv_2, u_2);

  using SlabPlanType =
      SlabPlan<execution_space, RealView3DType, ComplexView3DType, 1>;

  // Not a slab geometry
  if (nprocs == 1) {
    // topo0 -> topo1 with ax = {0}:
    // (n0, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
    ASSERT_THROW(
        {
          SlabPlanType plan_0_1_ax0(exec, u_0, u_hat_1_ax0, ax0, topology0,
                                    topology1, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topo0 -> topo2 with ax = {1}:
    // (n0, n1, n2/p) -> (n0/p, n1/2+1, n2)
    ASSERT_THROW(
        {
          SlabPlanType plan_0_2_ax1(exec, u_0, u_hat_2_ax1, ax1, topology0,
                                    topology2, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topo1 -> topo2 with ax = {2}:
    // (n0, n1/p, n2) -> (n0/p, n1, n2/2+1)
    ASSERT_THROW(
        {
          SlabPlanType plan_1_2_ax2(exec, u_1, u_hat_2_ax2, ax2, topology1,
                                    topology2, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topo0 -> topo0 with ax = {2}:
    // (n0, n1, n2/p) -> (n0, n1, (n2/2+1)/p)
    ASSERT_THROW(
        {
          SlabPlanType plan_0_0_ax2(exec, u_0, u_hat_0_ax2, ax2, topology0,
                                    topology0, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topo1 -> topo1 with ax = {1}
    // (n0, n1/p, n2) -> (n0, (n1/2+1)/p, n2)
    ASSERT_THROW(
        {
          SlabPlanType plan_1_1_ax1(exec, u_1, u_hat_1_ax1, ax1, topology1,
                                    topology1, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology2 -> topology2 with ax = {1}
    // (n0/p, n1, n2) -> (n0/p, (n1/2+1), n2)
    ASSERT_THROW(
        {
          SlabPlanType plan_2_2_ax1(exec, u_2, u_hat_2_ax1, ax1, topology2,
                                    topology2, MPI_COMM_WORLD);
        },
        std::runtime_error);
  } else {
    // topo 0 -> topo 0 with ax = {0}:
    // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p)
    // FFT ax = {0}
    SlabPlanType plan_0_0_ax0(exec, u_0, u_hat_0_ax0, ax0, topology0, topology0,
                              MPI_COMM_WORLD);
    plan_0_0_ax0.forward(u_0, u_hat_0_ax0);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax0, ref_u_hat_0_ax0));

    plan_0_0_ax0.backward(u_hat_0_ax0, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 0 with ax = {1}:
    // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p)
    // FFT ax = {1}
    SlabPlanType plan_0_0_ax1(exec, u_0, u_hat_0_ax1, ax1, topology0, topology0,
                              MPI_COMM_WORLD);
    plan_0_0_ax1.forward(u_0, u_hat_0_ax1);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax1, ref_u_hat_0_ax1));

    plan_0_0_ax1.backward(u_hat_0_ax1, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 0 with ax = {2}:
    // (n0, n1, n2/p) -> (n0/p, n1, n2) -> (n0, n1, (n2/2+1)/p)
    // Transpose topo 2 -> FFT ax = {2} -> Transpose topo 0
    SlabPlanType plan_0_0_ax2(exec, u_0, u_hat_0_ax2, ax2, topology0, topology0,
                              MPI_COMM_WORLD);
    plan_0_0_ax2.forward(u_0, u_hat_0_ax2);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax2, ref_u_hat_0_ax2));

    plan_0_0_ax2.backward(u_hat_0_ax2, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 1 with ax = {0}:
    // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> (n0/2+1, n1/p, n2)
    // FFT ax = {0} + Transpose 1
    SlabPlanType plan_0_1_ax0(exec, u_0, u_hat_1_ax0, ax0, topology0, topology1,
                              MPI_COMM_WORLD);
    plan_0_1_ax0.forward(u_0, u_hat_1_ax0);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax0, ref_u_hat_1_ax0));

    plan_0_1_ax0.backward(u_hat_1_ax0, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 1 with ax = {1}:
    // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0, (n1/2+1)/p, n2)
    // FFT ax = {1} -> Transpose topo 1
    SlabPlanType plan_0_1_ax1(exec, u_0, u_hat_1_ax1, ax1, topology0, topology1,
                              MPI_COMM_WORLD);
    plan_0_1_ax1.forward(u_0, u_hat_1_ax1);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax1, ref_u_hat_1_ax1));

    plan_0_1_ax1.backward(u_hat_1_ax1, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 1 with ax = {2}:
    // (n0, n1, n2/p) -> (n0, n1/p, n2) -> (n0, n1/p, n2/2+1)
    // Transpose topo 1 ->FFT ax = {2}
    SlabPlanType plan_0_1_ax2(exec, u_0, u_hat_1_ax2, ax2, topology0, topology1,
                              MPI_COMM_WORLD);
    plan_0_1_ax2.forward(u_0, u_hat_1_ax2);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax2, ref_u_hat_1_ax2));

    plan_0_1_ax2.backward(u_hat_1_ax2, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 2 with ax = {0}:
    // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
    // FFT ax = {0} -> Transpose topo 2
    SlabPlanType plan_0_2_ax0(exec, u_0, u_hat_2_ax0, ax0, topology0, topology2,
                              MPI_COMM_WORLD);
    plan_0_2_ax0.forward(u_0, u_hat_2_ax0);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax0, ref_u_hat_2_ax0));

    plan_0_2_ax0.backward(u_hat_2_ax0, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 2 with ax = {1}:
    // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0/p, n1/2+1, n2)
    // FFT ax = {1} -> Transpose topo 2
    SlabPlanType plan_0_2_ax1(exec, u_0, u_hat_2_ax1, ax1, topology0, topology2,
                              MPI_COMM_WORLD);
    plan_0_2_ax1.forward(u_0, u_hat_2_ax1);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax1, ref_u_hat_2_ax1));

    plan_0_2_ax1.backward(u_hat_2_ax1, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 2 with ax = {2}:
    // (n0, n1, n2/p) -> (n0/p, n1, n2) -> (n0/p, n1, n2/2+1)
    // Transpose topo 2 -> FFT ax = {2}
    SlabPlanType plan_0_2_ax2(exec, u_0, u_hat_2_ax2, ax2, topology0, topology2,
                              MPI_COMM_WORLD);
    plan_0_2_ax2.forward(u_0, u_hat_2_ax2);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax2, ref_u_hat_2_ax2));

    plan_0_2_ax2.backward(u_hat_2_ax2, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 0 with ax = {0}:
    // (n0, n1/p, n2) -> (n0/2+1, n1, n2/p) -> (n0/2+1, n1, n2/p)
    // FFT ax = {0} -> Transpose topo 0
    SlabPlanType plan_1_0_ax0(exec, u_1, u_hat_0_ax0, ax0, topology1, topology0,
                              MPI_COMM_WORLD);
    plan_1_0_ax0.forward(u_1, u_hat_0_ax0);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax0, ref_u_hat_0_ax0));

    plan_1_0_ax0.backward(u_hat_0_ax0, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));
  }
}

template <typename T, typename LayoutType>
void test_slab2D_view2D(std::size_t nprocs) {
  using RealView2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;
  using axes_type     = KokkosFFT::axis_type<2>;
  using extents_type  = std::array<std::size_t, 2>;
  using topology_type = std::array<std::size_t, 2>;

  topology_type topology0{1, nprocs};
  topology_type topology1{nprocs, 1};

  const std::size_t n0 = 8, n1 = 7;
  extents_type global_in_extents{n0, n1},
      global_out_extents_ax0{n0 / 2 + 1, n1},
      global_out_extents_ax1{n0, n1 / 2 + 1};

  // All axes
  axes_type ax01 = {0, 1}, ax10 = {1, 0};

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
  ComplexView2DType gu_hat_ax01("gu_hat_ax01", n0, n1 / 2 + 1),
      gu_hat_ax10("gu_hat_ax10", n0 / 2 + 1, n1);

  // Data in Topology 0 (X-slab)
  RealView2DType u_0("u_0",
                     KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t0)),
      u_inv_0("u_inv_0",
              KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t0)),
      ref_u_inv_0("ref_u_inv_0",
                  KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t0));
  ComplexView2DType u_hat_0_ax01(
      "u_hat_0_ax01",
      KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax1)),
      u_hat_0_ax10("u_hat_0_ax10", KokkosFFT::Impl::create_layout<LayoutType>(
                                       out_extents_t0_ax0)),
      ref_u_hat_0_ax01(
          "ref_u_hat_0_ax01",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax1)),
      ref_u_hat_0_ax10(
          "ref_u_hat_0_ax10",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax0));

  // Data in Topology 1 (Y-slab)
  RealView2DType u_1("u_1",
                     KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t1)),
      u_inv_1("u_inv_1",
              KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t1)),
      ref_u_inv_1("ref_u_inv_1",
                  KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t1));
  ComplexView2DType u_hat_1_ax01(
      "u_hat_1_ax01",
      KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax1)),
      u_hat_1_ax10("u_hat_1_ax10", KokkosFFT::Impl::create_layout<LayoutType>(
                                       out_extents_t1_ax0)),
      ref_u_hat_1_ax01(
          "ref_u_hat_1_ax01",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax1)),
      ref_u_hat_1_ax10(
          "ref_u_hat_1_ax10",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax0));

  // Initialization
  execution_space exec;
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::fill_random(gu, random_pool, 1.0);

  KokkosFFT::rfft2(exec, gu, gu_hat_ax01, KokkosFFT::Normalization::backward,
                   ax01);
  KokkosFFT::rfft2(exec, gu, gu_hat_ax10, KokkosFFT::Normalization::backward,
                   ax10);

  // Topo 0
  Kokkos::pair<std::size_t, std::size_t> range_gu0(
      in_starts_t0.at(1), in_starts_t0.at(1) + in_extents_t0.at(1));
  auto sub_gu_0 = Kokkos::subview(gu, Kokkos::ALL, range_gu0);
  Kokkos::deep_copy(u_0, sub_gu_0);

  // Topo 1
  Kokkos::pair<std::size_t, std::size_t> range_gu1(
      in_starts_t1.at(0), in_starts_t1.at(0) + in_extents_t1.at(0));
  auto sub_gu_1 = Kokkos::subview(gu, range_gu1, Kokkos::ALL);
  Kokkos::deep_copy(u_1, sub_gu_1);

  // Define ranges for topology 0 (X-slab)
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax0(
      out_starts_t0_ax0.at(1),
      out_starts_t0_ax0.at(1) + out_extents_t0_ax0.at(1));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax1(
      out_starts_t0_ax1.at(1),
      out_starts_t0_ax1.at(1) + out_extents_t0_ax1.at(1));

  // Topo 0 -> Topo 0 ax = {0, 1}
  auto sub_gu_hat_0_ax01 =
      Kokkos::subview(gu_hat_ax01, Kokkos::ALL, range_gu_hat_0_ax1);
  Kokkos::deep_copy(ref_u_hat_0_ax01, sub_gu_hat_0_ax01);

  // Topo 0 -> Topo 0 ax = {1, 0}
  auto sub_gu_hat_0_ax10 =
      Kokkos::subview(gu_hat_ax10, Kokkos::ALL, range_gu_hat_0_ax0);
  Kokkos::deep_copy(ref_u_hat_0_ax10, sub_gu_hat_0_ax10);

  // Define ranges for topology 1 (Y-slab)
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_1_ax0(
      out_starts_t1_ax0.at(0),
      out_starts_t1_ax0.at(0) + out_extents_t1_ax0.at(0));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_1_ax1(
      out_starts_t1_ax1.at(0),
      out_starts_t1_ax1.at(0) + out_extents_t1_ax1.at(0));

  // Topo 1 -> Topo 1 ax = {0, 1}
  auto sub_gu_hat_1_ax01 =
      Kokkos::subview(gu_hat_ax01, range_gu_hat_1_ax1, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_1_ax01, sub_gu_hat_1_ax01);

  // Topo 1 -> Topo 1 ax = {1, 0}
  auto sub_gu_hat_1_ax10 =
      Kokkos::subview(gu_hat_ax10, range_gu_hat_1_ax0, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_1_ax10, sub_gu_hat_1_ax10);

  // For inverse transform
  Kokkos::deep_copy(ref_u_inv_0, u_0);
  Kokkos::deep_copy(ref_u_inv_1, u_1);

  using SlabPlanType =
      SlabPlan<execution_space, RealView2DType, ComplexView2DType, 2>;

  // Not a slab geometry
  if (nprocs == 1) {
    ASSERT_THROW(
        {
          SlabPlanType plan_0_1_ax01(exec, u_0, u_hat_1_ax01, ax01, topology0,
                                     topology1, MPI_COMM_WORLD);
        },
        std::runtime_error);
    ASSERT_THROW(
        {
          SlabPlanType plan_0_1_ax10(exec, u_0, u_hat_1_ax10, ax10, topology0,
                                     topology1, MPI_COMM_WORLD);
        },
        std::runtime_error);
    ASSERT_THROW(
        {
          SlabPlanType plan_1_0_ax01(exec, u_1, u_hat_0_ax01, ax01, topology1,
                                     topology0, MPI_COMM_WORLD);
        },
        std::runtime_error);
    ASSERT_THROW(
        {
          SlabPlanType plan_1_0_ax10(exec, u_1, u_hat_0_ax10, ax10, topology1,
                                     topology0, MPI_COMM_WORLD);
        },
        std::runtime_error);
  } else {
    // topo0 -> topo0 with ax = {0, 1}:
    // (n0, n1/p) -> (n0/p, n1) -> (n0/p, n1/2+1) -> (n0, (n1/2+1)/p)
    // Transpose topo 1 -> FFT ax = {1} -> Transpose topo 0 -> FFT ax = {0}
    SlabPlanType plan_0_0_ax01(exec, u_0, u_hat_0_ax01, ax01, topology0,
                               topology0, MPI_COMM_WORLD);
    plan_0_0_ax01.forward(u_0, u_hat_0_ax01);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax01, ref_u_hat_0_ax01));

    plan_0_0_ax01.backward(u_hat_0_ax01, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0));

    // topo0 -> topo0 with ax = {1, 0}:
    // (n0, n1/p) -> (n0/2+1, n1/p) -> ((n0/2+1)/p, n1) -> (n0/2+1, n1/p)
    // FFT ax = {0} -> Transpose topo 1 -> FFT ax = {1}
    // -> Transpose topo 0
    SlabPlanType plan_0_0_ax10(exec, u_0, u_hat_0_ax10, ax10, topology0,
                               topology0, MPI_COMM_WORLD);
    plan_0_0_ax10.forward(u_0, u_hat_0_ax10);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax10, ref_u_hat_0_ax10));

    plan_0_0_ax10.backward(u_hat_0_ax10, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0));

    // topo0 -> topo1 with ax = {0, 1}:
    // (n0, n1/p) -> (n0/p, n1) -> (n0/p, n1/2+1) -> (n0, (n1/2+1)/p)
    // -> (n0/p, n1/2+1)
    // Transpose topo 1 -> FFT ax = {1} -> Transpose topo 0 -> FFT ax = {0}
    // -> Transpose topo 1
    SlabPlanType plan_0_1_ax01(exec, u_0, u_hat_1_ax01, ax01, topology0,
                               topology1, MPI_COMM_WORLD);
    plan_0_1_ax01.forward(u_0, u_hat_1_ax01);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax01, ref_u_hat_1_ax01));

    plan_0_1_ax01.backward(u_hat_1_ax01, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0));

    // topo0 -> topo1 with ax = {1, 0}:
    // (n0, n1/p) -> (n0/2+1, n1/p) -> ((n0/2+1)/p, n1)
    // FFT ax = {0} -> Transpose topo 1 -> FFT ax = {1}
    SlabPlanType plan_0_1_ax10(exec, u_0, u_hat_1_ax10, ax10, topology0,
                               topology1, MPI_COMM_WORLD);
    plan_0_1_ax10.forward(u_0, u_hat_1_ax10);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax10, ref_u_hat_1_ax10));

    plan_0_1_ax10.backward(u_hat_1_ax10, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0));

    // topo1 -> topo0 with ax = {0, 1}:
    // (n0/p, n1) -> (n0/p, n1/2+1) -> (n0, (n1/2+1)/p)
    // FFT ax = {1} -> Transpose topo 0 -> FFT ax = {0}
    SlabPlanType plan_1_0_ax01(exec, u_1, u_hat_0_ax01, ax01, topology1,
                               topology0, MPI_COMM_WORLD);
    plan_1_0_ax01.forward(u_1, u_hat_0_ax01);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax01, ref_u_hat_0_ax01));

    plan_1_0_ax01.backward(u_hat_0_ax01, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1));

    // topo1 -> topo0 with ax = {1, 0}:
    // (n0/p, n1) -> (n0, n1/p) -> (n0/2+1, n1/p) -> ((n0/2+1)/p, n1)
    // Transpose topo 0 -> FFT ax = {0} -> Transpose topo 1 -> FFT ax = {1}
    // -> Transpose topo 0
    SlabPlanType plan_1_0_ax10(exec, u_1, u_hat_0_ax10, ax10, topology1,
                               topology0, MPI_COMM_WORLD);
    plan_1_0_ax10.forward(u_1, u_hat_0_ax10);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax10, ref_u_hat_0_ax10));

    plan_1_0_ax10.backward(u_hat_0_ax10, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1));

    // topo1 -> topo1 with ax = {0, 1}:
    // (n0/p, n1) -> (n0/p, n1/2+1) -> (n0, (n1/2+1)/p)
    // FFT ax = {1} -> Transpose topo 0 -> FFT ax = {0} -> Transpose topo 1
    SlabPlanType plan_1_1_ax01(exec, u_1, u_hat_1_ax01, ax01, topology1,
                               topology1, MPI_COMM_WORLD);
    plan_1_1_ax01.forward(u_1, u_hat_1_ax01);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax01, ref_u_hat_1_ax01));

    plan_1_1_ax01.backward(u_hat_1_ax01, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1));

    // topo1 -> topo1 with ax = {1, 0}:
    // (n0/p, n1) -> (n0, n1/p) -> (n0/2+1, n1/p) -> ((n0/2+1)/p, n1)
    // Transpose topo 0 -> FFT ax = {0} -> Transpose topo 1 -> FFT ax = {1}
    SlabPlanType plan_1_1_ax10(exec, u_1, u_hat_1_ax10, ax10, topology1,
                               topology1, MPI_COMM_WORLD);
    plan_1_1_ax10.forward(u_1, u_hat_1_ax10);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax10, ref_u_hat_1_ax10));

    plan_1_1_ax10.backward(u_hat_1_ax10, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1));
  }
}

template <typename T, typename LayoutType>
void test_slab2D_view3D(std::size_t nprocs) {
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

  // All axes
  axes_type ax01 = {0, 1}, ax02 = {0, 2}, ax10 = {1, 0}, ax12 = {1, 2},
            ax20 = {2, 0}, ax21 = {2, 1};

  // Available combinations
  // Topology 0 -> Topology 1 with all axes
  // Topology 0 -> Topology 2 with all axes
  // Topology 1 -> Topology 2 with all axes

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
  ComplexView3DType u_hat_0_ax01(
      "u_hat_0_ax01",
      KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax1)),
      u_hat_0_ax02("u_hat_0_ax02", KokkosFFT::Impl::create_layout<LayoutType>(
                                       out_extents_t0_ax2)),
      u_hat_0_ax10("u_hat_0_ax10", KokkosFFT::Impl::create_layout<LayoutType>(
                                       out_extents_t0_ax0)),
      u_hat_0_ax12("u_hat_0_ax12", KokkosFFT::Impl::create_layout<LayoutType>(
                                       out_extents_t0_ax2)),
      u_hat_0_ax20("u_hat_0_ax20", KokkosFFT::Impl::create_layout<LayoutType>(
                                       out_extents_t0_ax0)),
      u_hat_0_ax21("u_hat_0_ax21", KokkosFFT::Impl::create_layout<LayoutType>(
                                       out_extents_t0_ax1)),
      ref_u_hat_0_ax01(
          "ref_u_hat_0_ax01",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax1)),
      ref_u_hat_0_ax02(
          "ref_u_hat_0_ax02",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax2)),
      ref_u_hat_0_ax10(
          "ref_u_hat_0_ax10",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax0)),
      ref_u_hat_0_ax12(
          "ref_u_hat_0_ax12",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax2)),
      ref_u_hat_0_ax20(
          "ref_u_hat_0_ax20",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax0)),
      ref_u_hat_0_ax21(
          "ref_u_hat_0_ax21",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax1));

  // Data in Topology 1 (XZ-slab)
  RealView3DType u_1("u_1",
                     KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t1)),
      u_inv_1("u_inv_1",
              KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t1)),
      ref_u_inv_1("ref_u_inv_1",
                  KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t1));
  ComplexView3DType u_hat_1_ax01(
      "u_hat_1_ax01",
      KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax1)),
      u_hat_1_ax02("u_hat_1_ax02", KokkosFFT::Impl::create_layout<LayoutType>(
                                       out_extents_t1_ax2)),
      u_hat_1_ax10("u_hat_1_ax10", KokkosFFT::Impl::create_layout<LayoutType>(
                                       out_extents_t1_ax0)),
      u_hat_1_ax12("u_hat_1_ax12", KokkosFFT::Impl::create_layout<LayoutType>(
                                       out_extents_t1_ax2)),
      u_hat_1_ax20("u_hat_1_ax20", KokkosFFT::Impl::create_layout<LayoutType>(
                                       out_extents_t1_ax0)),
      u_hat_1_ax21("u_hat_1_ax21", KokkosFFT::Impl::create_layout<LayoutType>(
                                       out_extents_t1_ax1)),
      ref_u_hat_1_ax01(
          "ref_u_hat_1_ax01",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax1)),
      ref_u_hat_1_ax02(
          "ref_u_hat_1_ax02",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax2)),
      ref_u_hat_1_ax10(
          "ref_u_hat_1_ax10",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax0)),
      ref_u_hat_1_ax12(
          "ref_u_hat_1_ax12",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax2)),
      ref_u_hat_1_ax20(
          "ref_u_hat_1_ax20",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax0)),
      ref_u_hat_1_ax21(
          "ref_u_hat_1_ax21",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax1));

  // Data in Topology 2 (YZ-slab)
  RealView3DType u_2("u_2",
                     KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t2)),
      u_inv_2("u_inv_2",
              KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t2)),
      ref_u_inv_2("ref_u_inv_2",
                  KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t2));
  ComplexView3DType u_hat_2_ax01(
      "u_hat_2_ax01",
      KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax1)),
      u_hat_2_ax02("u_hat_2_ax02", KokkosFFT::Impl::create_layout<LayoutType>(
                                       out_extents_t2_ax2)),
      u_hat_2_ax10("u_hat_2_ax10", KokkosFFT::Impl::create_layout<LayoutType>(
                                       out_extents_t2_ax0)),
      u_hat_2_ax12("u_hat_2_ax12", KokkosFFT::Impl::create_layout<LayoutType>(
                                       out_extents_t2_ax2)),
      u_hat_2_ax20("u_hat_2_ax20", KokkosFFT::Impl::create_layout<LayoutType>(
                                       out_extents_t2_ax0)),
      u_hat_2_ax21("u_hat_2_ax21", KokkosFFT::Impl::create_layout<LayoutType>(
                                       out_extents_t2_ax1)),
      ref_u_hat_2_ax01(
          "ref_u_hat_2_ax01",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax1)),
      ref_u_hat_2_ax02(
          "ref_u_hat_2_ax02",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax2)),
      ref_u_hat_2_ax10(
          "ref_u_hat_2_ax10",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax0)),
      ref_u_hat_2_ax12(
          "ref_u_hat_2_ax12",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax2)),
      ref_u_hat_2_ax20(
          "ref_u_hat_2_ax20",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax0)),
      ref_u_hat_2_ax21(
          "ref_u_hat_2_ax21",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax1));

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

  // Topo 0
  Kokkos::pair<std::size_t, std::size_t> range_gu0(
      in_starts_t0.at(2), in_starts_t0.at(2) + in_extents_t0.at(2));
  auto sub_gu_0 = Kokkos::subview(gu, Kokkos::ALL, Kokkos::ALL, range_gu0);
  Kokkos::deep_copy(u_0, sub_gu_0);

  // Topo 1
  Kokkos::pair<std::size_t, std::size_t> range_gu1(
      in_starts_t1.at(1), in_starts_t1.at(1) + in_extents_t1.at(1));
  auto sub_gu_1 = Kokkos::subview(gu, Kokkos::ALL, range_gu1, Kokkos::ALL);
  Kokkos::deep_copy(u_1, sub_gu_1);

  // Topo 2
  Kokkos::pair<std::size_t, std::size_t> range_gu2(
      in_starts_t2.at(0), in_starts_t2.at(0) + in_extents_t2.at(0));
  auto sub_gu_2 = Kokkos::subview(gu, range_gu2, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(u_2, sub_gu_2);

  // Define ranges for topology 0 (XY-slab)
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax0(
      out_starts_t0_ax0.at(2),
      out_starts_t0_ax0.at(2) + out_extents_t0_ax0.at(2));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax1(
      out_starts_t0_ax1.at(2),
      out_starts_t0_ax1.at(2) + out_extents_t0_ax1.at(2));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax2(
      out_starts_t0_ax2.at(2),
      out_starts_t0_ax2.at(2) + out_extents_t0_ax2.at(2));

  // Topo 0 ax = {0, 1}
  auto sub_gu_hat_0_ax01 = Kokkos::subview(gu_hat_ax01, Kokkos::ALL,
                                           Kokkos::ALL, range_gu_hat_0_ax1);
  Kokkos::deep_copy(ref_u_hat_0_ax01, sub_gu_hat_0_ax01);

  // Topo 0 ax = {0, 2}
  auto sub_gu_hat_0_ax02 = Kokkos::subview(gu_hat_ax02, Kokkos::ALL,
                                           Kokkos::ALL, range_gu_hat_0_ax2);
  Kokkos::deep_copy(ref_u_hat_0_ax02, sub_gu_hat_0_ax02);

  // Topo 0 ax = {1, 0}
  auto sub_gu_hat_0_ax10 = Kokkos::subview(gu_hat_ax10, Kokkos::ALL,
                                           Kokkos::ALL, range_gu_hat_0_ax0);
  Kokkos::deep_copy(ref_u_hat_0_ax10, sub_gu_hat_0_ax10);

  // Topo 0 ax = {1, 2}
  auto sub_gu_hat_0_ax12 = Kokkos::subview(gu_hat_ax12, Kokkos::ALL,
                                           Kokkos::ALL, range_gu_hat_0_ax2);
  Kokkos::deep_copy(ref_u_hat_0_ax12, sub_gu_hat_0_ax12);

  // Topo 0 ax = {2, 0}
  auto sub_gu_hat_0_ax20 = Kokkos::subview(gu_hat_ax20, Kokkos::ALL,
                                           Kokkos::ALL, range_gu_hat_0_ax0);
  Kokkos::deep_copy(ref_u_hat_0_ax20, sub_gu_hat_0_ax20);

  // Topo 0 ax = {2, 1}
  auto sub_gu_hat_0_ax21 = Kokkos::subview(gu_hat_ax21, Kokkos::ALL,
                                           Kokkos::ALL, range_gu_hat_0_ax1);
  Kokkos::deep_copy(ref_u_hat_0_ax21, sub_gu_hat_0_ax21);

  // Define ranges for topology 1 (XZ-slab)
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_1_ax0(
      out_starts_t1_ax0.at(1),
      out_starts_t1_ax0.at(1) + out_extents_t1_ax0.at(1));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_1_ax1(
      out_starts_t1_ax1.at(1),
      out_starts_t1_ax1.at(1) + out_extents_t1_ax1.at(1));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_1_ax2(
      out_starts_t1_ax2.at(1),
      out_starts_t1_ax2.at(1) + out_extents_t1_ax2.at(1));

  // Topo 1 ax = {0, 1}
  auto sub_gu_hat_1_ax01 = Kokkos::subview(gu_hat_ax01, Kokkos::ALL,
                                           range_gu_hat_1_ax1, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_1_ax01, sub_gu_hat_1_ax01);

  // Topo 1 ax = {0, 2}
  auto sub_gu_hat_1_ax02 = Kokkos::subview(gu_hat_ax02, Kokkos::ALL,
                                           range_gu_hat_1_ax2, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_1_ax02, sub_gu_hat_1_ax02);

  // Topo 1 ax = {1, 0}
  auto sub_gu_hat_1_ax10 = Kokkos::subview(gu_hat_ax10, Kokkos::ALL,
                                           range_gu_hat_1_ax0, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_1_ax10, sub_gu_hat_1_ax10);

  // Topo 1 ax = {1, 2}
  auto sub_gu_hat_1_ax12 = Kokkos::subview(gu_hat_ax12, Kokkos::ALL,
                                           range_gu_hat_1_ax2, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_1_ax12, sub_gu_hat_1_ax12);

  // Topo 1 ax = {2, 0}
  auto sub_gu_hat_1_ax20 = Kokkos::subview(gu_hat_ax20, Kokkos::ALL,
                                           range_gu_hat_1_ax0, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_1_ax20, sub_gu_hat_1_ax20);

  // Topo 1 ax = {2, 1}
  auto sub_gu_hat_1_ax21 = Kokkos::subview(gu_hat_ax21, Kokkos::ALL,
                                           range_gu_hat_1_ax1, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_1_ax21, sub_gu_hat_1_ax21);

  // Define ranges for topology 2 (YZ-slab)
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_2_ax0(
      out_starts_t2_ax0.at(0),
      out_starts_t2_ax0.at(0) + out_extents_t2_ax0.at(0));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_2_ax1(
      out_starts_t2_ax1.at(0),
      out_starts_t2_ax1.at(0) + out_extents_t2_ax1.at(0));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_2_ax2(
      out_starts_t2_ax2.at(0),
      out_starts_t2_ax2.at(0) + out_extents_t2_ax2.at(0));

  // Topo 2 ax = {0, 1}
  auto sub_gu_hat_2_ax01 = Kokkos::subview(gu_hat_ax01, range_gu_hat_2_ax1,
                                           Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax01, sub_gu_hat_2_ax01);

  // Topo 2 ax = {0, 2}
  auto sub_gu_hat_2_ax02 = Kokkos::subview(gu_hat_ax02, range_gu_hat_2_ax2,
                                           Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax02, sub_gu_hat_2_ax02);

  // Topo 2 ax = {1, 0}
  auto sub_gu_hat_2_ax10 = Kokkos::subview(gu_hat_ax10, range_gu_hat_2_ax0,
                                           Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax10, sub_gu_hat_2_ax10);

  // Topo 2 ax = {1, 2}
  auto sub_gu_hat_2_ax12 = Kokkos::subview(gu_hat_ax12, range_gu_hat_2_ax2,
                                           Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax12, sub_gu_hat_2_ax12);

  // Topo 2 ax = {2, 0}
  auto sub_gu_hat_2_ax20 = Kokkos::subview(gu_hat_ax20, range_gu_hat_2_ax0,
                                           Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax20, sub_gu_hat_2_ax20);

  // Topo 2 ax = {2, 1}
  auto sub_gu_hat_2_ax21 = Kokkos::subview(gu_hat_ax21, range_gu_hat_2_ax1,
                                           Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax21, sub_gu_hat_2_ax21);

  // For inverse transform
  Kokkos::deep_copy(ref_u_inv_0, u_0);
  Kokkos::deep_copy(ref_u_inv_1, u_1);
  Kokkos::deep_copy(ref_u_inv_2, u_2);

  using SlabPlanType =
      SlabPlan<execution_space, RealView3DType, ComplexView3DType, 2>;

  // Not a slab geometry
  if (nprocs == 1) {
    // topology0 -> topology1
    // (n0, n1, n2/p) -> (n0, (n1/2+1)/p, n2)
    ASSERT_THROW(
        {
          SlabPlanType plan_0_1_ax01(exec, u_0, u_hat_1_ax01, ax01, topology0,
                                     topology1, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology0 -> topology2
    // (n0, n1, n2/p) -> (n0/p, n1/2+1, n2)
    ASSERT_THROW(
        {
          SlabPlanType plan_0_2_ax01(exec, u_0, u_hat_2_ax01, ax01, topology0,
                                     topology2, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology1 -> topology2
    // (n0, n1/p, n2) -> (n0/p, n1, n2/2+1)
    ASSERT_THROW(
        {
          SlabPlanType plan_1_2_ax12(exec, u_1, u_hat_2_ax12, ax12, topology1,
                                     topology2, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology0 -> topology0 with ax = {1, 2}
    // (n0, n1, n2/p) -> (n0, n1, (n2/2+1)/p)
    ASSERT_THROW(
        {
          SlabPlanType plan_0_0_ax12(exec, u_0, u_hat_0_ax12, ax12, topology0,
                                     topology0, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology1 -> topology1 with ax = {1, 2}
    // (n0, n1/p, n2) -> (n0, (n1/2+1)/p, n2)
    ASSERT_THROW(
        {
          SlabPlanType plan_1_1_ax21(exec, u_1, u_hat_1_ax21, ax21, topology1,
                                     topology1, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology2 -> topology2 with ax = {0, 1}
    // (n0/p, n1, n2) -> (n0/p, (n1/2+1), n2)
    ASSERT_THROW(
        {
          SlabPlanType plan_2_2_ax01(exec, u_2, u_hat_2_ax01, ax01, topology2,
                                     topology2, MPI_COMM_WORLD);
        },
        std::runtime_error);
  } else {
    // topo 0 -> topo 0 with ax = {0, 1}:
    // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p)
    // FFT2 ax = {0, 1}
    SlabPlanType plan_0_0_ax01(exec, u_0, u_hat_0_ax01, ax01, topology0,
                               topology0, MPI_COMM_WORLD);
    plan_0_0_ax01.forward(u_0, u_hat_0_ax01);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax01, ref_u_hat_0_ax01));

    plan_0_0_ax01.backward(u_hat_0_ax01, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 0 with ax = {0, 2}:
    // (n0, n1, n2/p) -> (n0, n1/p, n2) -> (n0, n1, (n2/2+1)/p)
    // Transpose topo 1 -> FFT2 ax = {0, 2} -> Transpose topo 0
    SlabPlanType plan_0_0_ax02(exec, u_0, u_hat_0_ax02, ax02, topology0,
                               topology0, MPI_COMM_WORLD);
    plan_0_0_ax02.forward(u_0, u_hat_0_ax02);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax02, ref_u_hat_0_ax02));

    plan_0_0_ax02.backward(u_hat_0_ax02, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 0 with ax = {1, 0}:
    // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p)
    // FFT2 ax = {1, 0}
    SlabPlanType plan_0_0_ax10(exec, u_0, u_hat_0_ax10, ax10, topology0,
                               topology0, MPI_COMM_WORLD);
    plan_0_0_ax10.forward(u_0, u_hat_0_ax10);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax10, ref_u_hat_0_ax10));

    plan_0_0_ax10.backward(u_hat_0_ax10, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 0 with ax = {1, 2}:
    // (n0, n1, n2/p) -> (n0/p, n1, n2) -> (n0, n1, (n2/2+1)/p)
    // Transpose topo 2 -> FFT2 ax = {1, 2} -> Transpose topo 0
    SlabPlanType plan_0_0_ax12(exec, u_0, u_hat_0_ax12, ax12, topology0,
                               topology0, MPI_COMM_WORLD);
    plan_0_0_ax12.forward(u_0, u_hat_0_ax12);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax12, ref_u_hat_0_ax12));

    plan_0_0_ax12.backward(u_hat_0_ax12, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 0 with ax = {2, 0}:
    // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> (n0/2+1, n1/p, n2)
    // -> (n0/2+1, n1, n2/p)
    // FFT ax = {0} -> Transpose topo 1 -> FFT ax = {2} -> Transpose topo 0
    SlabPlanType plan_0_0_ax20(exec, u_0, u_hat_0_ax20, ax20, topology0,
                               topology0, MPI_COMM_WORLD);
    plan_0_0_ax20.forward(u_0, u_hat_0_ax20);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax20, ref_u_hat_0_ax20));

    plan_0_0_ax20.backward(u_hat_0_ax20, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 0 with ax = {2, 1}:
    // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0/p, n1/2+1, n2)
    // -> (n0, n1/2+1, n2/p)
    // FFT ax = {1} -> Transpose topo 2 -> FFT ax = {2}
    // -> Transpose topo 0
    SlabPlanType plan_0_0_ax21(exec, u_0, u_hat_0_ax21, ax21, topology0,
                               topology0, MPI_COMM_WORLD);
    plan_0_0_ax21.forward(u_0, u_hat_0_ax21);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax21, ref_u_hat_0_ax21));

    plan_0_0_ax21.backward(u_hat_0_ax21, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 1 with ax = {0, 1}:
    // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0, (n1/2+1)/p, n2)
    // FFT2 ax = {0, 1} -> Transpose topo 1
    SlabPlanType plan_0_1_ax01(exec, u_0, u_hat_1_ax01, ax01, topology0,
                               topology1, MPI_COMM_WORLD);
    plan_0_1_ax01.forward(u_0, u_hat_1_ax01);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax01, ref_u_hat_1_ax01));

    plan_0_1_ax01.backward(u_hat_1_ax01, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 1 with ax = {0, 2}:
    // (n0, n1, n2/p) -> (n0, n1/p, n2) -> (n0, n1/p, n2/2+1)
    // Transpose 1 -> FFT2 ax = {0, 2}
    SlabPlanType plan_0_1_ax02(exec, u_0, u_hat_1_ax02, ax02, topology0,
                               topology1, MPI_COMM_WORLD);
    plan_0_1_ax02.forward(u_0, u_hat_1_ax02);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax02, ref_u_hat_1_ax02));

    plan_0_1_ax02.backward(u_hat_1_ax02, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 1 with ax = {1, 0}:
    // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> (n0/2+1, n1/p, n2)
    // FFT2 ax = {1, 0} + Transpose
    SlabPlanType plan_0_1_ax10(exec, u_0, u_hat_1_ax10, ax10, topology0,
                               topology1, MPI_COMM_WORLD);
    plan_0_1_ax10.forward(u_0, u_hat_1_ax10);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax10, ref_u_hat_1_ax10));

    plan_0_1_ax10.backward(u_hat_1_ax10, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 1 with ax = {1, 2}:
    // (n0, n1, n2/p) -> (n0/p, n1, n2) -> (n0/p, n1, n2/2+1)
    // Transpose topo 2 -> FFT2 ax = {1, 2}
    SlabPlanType plan_0_1_ax12(exec, u_0, u_hat_1_ax12, ax12, topology0,
                               topology1, MPI_COMM_WORLD);
    plan_0_1_ax12.forward(u_0, u_hat_1_ax12);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax12, ref_u_hat_1_ax12));

    plan_0_1_ax12.backward(u_hat_1_ax12, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 1 with ax = {2, 0}:
    // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> (n0/2+1, n1/p, n2)
    // FFT ax = {0} -> Transpose topo 1 -> FFT ax = {2}
    SlabPlanType plan_0_1_ax20(exec, u_0, u_hat_1_ax20, ax20, topology0,
                               topology1, MPI_COMM_WORLD);
    plan_0_1_ax20.forward(u_0, u_hat_1_ax20);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax20, ref_u_hat_1_ax20));

    plan_0_1_ax20.backward(u_hat_1_ax20, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 1 with ax = {2, 1}:
    // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0, (n1/2+1)/p, n2)
    // FFT ax = {1} -> Transpose topo 1 -> FFT ax = {2}
    SlabPlanType plan_0_1_ax21(exec, u_0, u_hat_1_ax21, ax21, topology0,
                               topology1, MPI_COMM_WORLD);
    plan_0_1_ax21.forward(u_0, u_hat_1_ax21);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax21, ref_u_hat_1_ax21));

    plan_0_1_ax21.backward(u_hat_1_ax21, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 2 with ax = {0, 1}:
    // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0/p, n1/2+1, n2)
    // FFT2 ax = {0, 1} -> Transpose topo 2
    SlabPlanType plan_0_2_ax01(exec, u_0, u_hat_2_ax01, ax01, topology0,
                               topology2, MPI_COMM_WORLD);
    plan_0_2_ax01.forward(u_0, u_hat_2_ax01);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax01, ref_u_hat_2_ax01));

    plan_0_2_ax01.backward(u_hat_2_ax01, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 2 with ax = {0, 2}:
    // (n0, n1, n2/p) -> (n0, n1/p, n2) -> (n0, n1/p, n2/2+1)
    // -> (n0/p, n1, n2/2+1)
    // Transpose topo 1 -> FFT2 ax = {0, 2} -> Transpose topo 2
    SlabPlanType plan_0_2_ax02(exec, u_0, u_hat_2_ax02, ax02, topology0,
                               topology2, MPI_COMM_WORLD);
    plan_0_2_ax02.forward(u_0, u_hat_2_ax02);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax02, ref_u_hat_2_ax02));

    plan_0_2_ax02.backward(u_hat_2_ax02, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 2 with ax = {1, 0}:
    // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
    // FFT2 ax = {1, 0} -> Transpose topo 2
    SlabPlanType plan_0_2_ax10(exec, u_0, u_hat_2_ax10, ax10, topology0,
                               topology2, MPI_COMM_WORLD);
    plan_0_2_ax10.forward(u_0, u_hat_2_ax10);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax10, ref_u_hat_2_ax10));

    plan_0_2_ax10.backward(u_hat_2_ax10, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 2 with ax = {1, 2}:
    // (n0, n1, n2/p) -> (n0/p, n1, n2) -> (n0/p, n1, n2/2+1)
    // Transpose topo 2 -> FFT2 ax = {1, 2}
    SlabPlanType plan_0_2_ax12(exec, u_0, u_hat_2_ax12, ax12, topology0,
                               topology2, MPI_COMM_WORLD);
    plan_0_2_ax12.forward(u_0, u_hat_2_ax12);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax12, ref_u_hat_2_ax12));

    plan_0_2_ax12.backward(u_hat_2_ax12, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 2 with ax = {2, 0}:
    // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
    // FFT ax = {0} -> Transpose topo 2 -> FFT ax = {2}
    SlabPlanType plan_0_2_ax20(exec, u_0, u_hat_2_ax20, ax20, topology0,
                               topology2, MPI_COMM_WORLD);
    plan_0_2_ax20.forward(u_0, u_hat_2_ax20);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax20, ref_u_hat_2_ax20));

    plan_0_2_ax20.backward(u_hat_2_ax20, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 2 with ax = {2, 1}:
    // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0/p, n1/2+1, n2)
    // FFT ax = {1} -> Transpose topo 2 -> FFT ax = {2}
    SlabPlanType plan_0_2_ax21(exec, u_0, u_hat_2_ax21, ax21, topology0,
                               topology2, MPI_COMM_WORLD);
    plan_0_2_ax21.forward(u_0, u_hat_2_ax21);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax21, ref_u_hat_2_ax21));

    plan_0_2_ax21.backward(u_hat_2_ax21, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 0 with ax = {0, 1}:
    // (n0, n1/p, n2) -> (n0, n1, n2/p) -> (n0, n1/2+1, n2)
    // Transpose topo 0 -> FFT2 ax = {0, 1}
    SlabPlanType plan_1_0_ax01(exec, u_1, u_hat_0_ax01, ax01, topology1,
                               topology0, MPI_COMM_WORLD);
    plan_1_0_ax01.forward(u_1, u_hat_0_ax01);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax01, ref_u_hat_0_ax01));

    plan_1_0_ax01.backward(u_hat_0_ax01, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 0 with ax = {0, 2}:
    // (n0, n1/p, n2) -> (n0, n1/p, n2/2+1) -> (n0, n1, (n2/2+1)/p)
    // FFT ax = {2} -> Transpose topo 0 -> FFT ax = {0}
    SlabPlanType plan_1_0_ax02(exec, u_1, u_hat_0_ax02, ax02, topology1,
                               topology0, MPI_COMM_WORLD);
    plan_1_0_ax02.forward(u_1, u_hat_0_ax02);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax02, ref_u_hat_0_ax02));

    plan_1_0_ax02.backward(u_hat_0_ax02, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 0 with ax = {1, 0}:
    // (n0, n1/p, n2) -> (n0/2+1, n1/p, n2) -> (n0/2+1, n1, n2/p)
    // FFT ax = {0} -> Transpose topo 0 -> FFT ax = {1}
    // Transpose topo 0 -> FFT2 ax = {1, 0}
    SlabPlanType plan_1_0_ax10(exec, u_1, u_hat_0_ax10, ax10, topology1,
                               topology0, MPI_COMM_WORLD);
    plan_1_0_ax10.forward(u_1, u_hat_0_ax10);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax10, ref_u_hat_0_ax10));

    plan_1_0_ax10.backward(u_hat_0_ax10, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 0 with ax = {1, 2}:
    // (n0, n1/p, n2) -> (n0, n1/p, n2/2+1) -> (n0, n1, (n2/2+1)/p)
    // FFT ax = {2} -> Transpose topo 0 -> FFT ax = {1}
    SlabPlanType plan_1_0_ax12(exec, u_1, u_hat_0_ax12, ax12, topology1,
                               topology0, MPI_COMM_WORLD);
    plan_1_0_ax12.forward(u_1, u_hat_0_ax12);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax12, ref_u_hat_0_ax12));

    plan_1_0_ax12.backward(u_hat_0_ax12, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 0 with ax = {2, 0}:
    // (n0, n1/p, n2) -> (n0/2+1, n1/p, n2) -> (n0/2+1, n1, n2/p)
    // FFT2 ax = {2,0} -> Transpose topo 0
    SlabPlanType plan_1_0_ax20(exec, u_1, u_hat_0_ax20, ax20, topology1,
                               topology0, MPI_COMM_WORLD);
    plan_1_0_ax20.forward(u_1, u_hat_0_ax20);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax20, ref_u_hat_0_ax20));

    plan_1_0_ax20.backward(u_hat_0_ax20, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 0 with ax = {2, 1}:
    // (n0, n1/p, n2) -> (n0/p, n1, n2) -> (n0/p, n1/2+1, n2)
    // -> (n0, n1/2+1, n2/p)
    // Transpose topo 2 -> FFT2 ax = {2, 1} -> Transpose topo 0
    SlabPlanType plan_1_0_ax21(exec, u_1, u_hat_0_ax21, ax21, topology1,
                               topology0, MPI_COMM_WORLD);
    plan_1_0_ax21.forward(u_1, u_hat_0_ax21);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax21, ref_u_hat_0_ax21));

    plan_1_0_ax21.backward(u_hat_0_ax21, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 1 with ax = {0, 1}:
    // (n0, n1/p, n2) -> (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0,
    // (n1/2+1)/p, n2) Transpose topo 0 -> FFT2 ax = {0, 1} -> Transpose topo 1
    SlabPlanType plan_1_1_ax01(exec, u_1, u_hat_1_ax01, ax01, topology1,
                               topology1, MPI_COMM_WORLD);
    plan_1_1_ax01.forward(u_1, u_hat_1_ax01);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax01, ref_u_hat_1_ax01));

    plan_1_1_ax01.backward(u_hat_1_ax01, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 1 with ax = {0, 2}:
    // (n0, n1/p, n2) ->(n0, n1/p, n2/2+1)
    // FFT2 ax = {0, 2}
    SlabPlanType plan_1_1_ax02(exec, u_1, u_hat_1_ax02, ax02, topology1,
                               topology1, MPI_COMM_WORLD);
    plan_1_1_ax02.forward(u_1, u_hat_1_ax02);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax02, ref_u_hat_1_ax02));

    plan_1_1_ax02.backward(u_hat_1_ax02, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 1 with ax = {1, 0}:
    // (n0, n1/p, n2) -> (n0/2+1, n1/p, n2) -> (n0/2+1, n1, n2/p)
    // -> (n0/2+1, n1/p, n2)
    // FFT ax = {0} -> Transpose topo 2 -> FFT ax = {1} -> Transpose topo 1
    SlabPlanType plan_1_1_ax10(exec, u_1, u_hat_1_ax10, ax10, topology1,
                               topology1, MPI_COMM_WORLD);
    plan_1_1_ax10.forward(u_1, u_hat_1_ax10);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax10, ref_u_hat_1_ax10));

    plan_1_1_ax10.backward(u_hat_1_ax10, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 1 with ax = {1, 2}:
    // (n0, n1/p, n2) -> (n0, n1/p, n2/2+1) -> (n0/p, n1, n2/2+1)
    // -> (n0, n1/p, n2/2+1)
    // FFT ax = {2} -> Transpose topo 2 -> FFT ax = {1} -> Transpose topo 1
    SlabPlanType plan_1_1_ax12(exec, u_1, u_hat_1_ax12, ax12, topology1,
                               topology1, MPI_COMM_WORLD);
    plan_1_1_ax12.forward(u_1, u_hat_1_ax12);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax12, ref_u_hat_1_ax12));

    plan_1_1_ax12.backward(u_hat_1_ax12, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 1 with ax = {2, 0}:
    // (n0, n1/p, n2) -> (n0/2+1, n1/p, n2)
    // FFT2 ax = {2, 0}
    SlabPlanType plan_1_1_ax20(exec, u_1, u_hat_1_ax20, ax20, topology1,
                               topology1, MPI_COMM_WORLD);
    plan_1_1_ax20.forward(u_1, u_hat_1_ax20);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax20, ref_u_hat_1_ax20));

    plan_1_1_ax20.backward(u_hat_1_ax20, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 1 with ax = {2, 1}:
    // (n0, n1/p, n2) -> (n0, n1/2+1, n2/p) -> (n0/p, n1/2+1, n2)
    // -> (n0, n1/p, n2/2+1)
    // Transpose topo 2 -> FFT2 ax = {2, 1} -> Transpose topo 1
    SlabPlanType plan_1_1_ax21(exec, u_1, u_hat_1_ax21, ax21, topology1,
                               topology1, MPI_COMM_WORLD);
    plan_1_1_ax21.forward(u_1, u_hat_1_ax21);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax21, ref_u_hat_1_ax21));

    plan_1_1_ax21.backward(u_hat_1_ax21, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 2 with ax = {0, 1}:
    // (n0, n1/p, n2) -> (n0, n1, n2/p) -> (n0, n1/2+1, n2/p)
    // -> (n0/p, n1/2+1, n2)
    // Transpose topo 0 -> FFT2 ax = {0, 1} -> Transpose topo 2
    SlabPlanType plan_1_2_ax01(exec, u_1, u_hat_2_ax01, ax01, topology1,
                               topology2, MPI_COMM_WORLD);
    plan_1_2_ax01.forward(u_1, u_hat_2_ax01);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax01, ref_u_hat_2_ax01));

    plan_1_2_ax01.backward(u_hat_2_ax01, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 2 with ax = {0, 2}:
    // (n0, n1/p, n2) -> (n0, n1/p, n2/2+1)
    // FFT2 ax = {0, 2}
    SlabPlanType plan_1_2_ax02(exec, u_1, u_hat_2_ax02, ax02, topology1,
                               topology2, MPI_COMM_WORLD);
    plan_1_2_ax02.forward(u_1, u_hat_2_ax02);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax02, ref_u_hat_2_ax02));

    plan_1_2_ax02.backward(u_hat_2_ax02, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 2 with ax = {1, 0}:
    // (n0, n1/p, n2) -> (n0/2+1, n1/p, n2) -> (n0/2+1, n1, n2/p)
    // FFT ax = {0} -> Transpose topo 2 -> FFT ax = {1}
    SlabPlanType plan_1_2_ax10(exec, u_1, u_hat_2_ax10, ax10, topology1,
                               topology2, MPI_COMM_WORLD);
    plan_1_2_ax10.forward(u_1, u_hat_2_ax10);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax10, ref_u_hat_2_ax10));

    plan_1_2_ax10.backward(u_hat_2_ax10, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 2 with ax = {1, 2}:
    // (n0, n1/p, n2) -> (n0, n1/p, n2/2+1) -> (n0/p, n1, n2/2+1)
    // FFT ax = {2} -> Transpose topo 2 -> FFT ax = {1}
    SlabPlanType plan_1_2_ax12(exec, u_1, u_hat_2_ax12, ax12, topology1,
                               topology2, MPI_COMM_WORLD);
    plan_1_2_ax12.forward(u_1, u_hat_2_ax12);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax12, ref_u_hat_2_ax12));

    plan_1_2_ax12.backward(u_hat_2_ax12, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 2 with ax = {2, 0}:
    // (n0, n1/p, n2) -> (n0/2+1, n1/p, n2) -> (n0/2+1, n1, n2/p)
    // FFT2 ax = {2, 0} -> Transpose topo 2
    SlabPlanType plan_1_2_ax20(exec, u_1, u_hat_2_ax20, ax20, topology1,
                               topology2, MPI_COMM_WORLD);
    plan_1_2_ax20.forward(u_1, u_hat_2_ax20);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax20, ref_u_hat_2_ax20));

    plan_1_2_ax20.backward(u_hat_2_ax20, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 2 with ax = {2, 1}:
    // (n0, n1/p, n2) -> (n0/p, n1, n2) -> (n0/p, n1/2+1, n2)
    // Transpose topo 2 -> FFT2 ax = {2, 1}
    SlabPlanType plan_1_2_ax21(exec, u_1, u_hat_2_ax21, ax21, topology1,
                               topology2, MPI_COMM_WORLD);
    plan_1_2_ax21.forward(u_1, u_hat_2_ax21);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax21, ref_u_hat_2_ax21));

    plan_1_2_ax21.backward(u_hat_2_ax21, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 0 with ax = {0, 1}:
    // (n0/p, n1, n2) -> (n0, n1, n2/p) -> (n0, n1/2+1, n2/p)
    // Transpose topo 0 -> FFT2 ax = {0, 1}
    SlabPlanType plan_2_0_ax01(exec, u_2, u_hat_0_ax01, ax01, topology2,
                               topology0, MPI_COMM_WORLD);
    plan_2_0_ax01.forward(u_2, u_hat_0_ax01);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax01, ref_u_hat_0_ax01));

    plan_2_0_ax01.backward(u_hat_0_ax01, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 0 with ax = {0, 2}:
    // (n0/p, n1, n2) -> (n0/p, n1, n2/2+1) -> (n0, n1, (n2/2+1)/p)
    // FFT ax = {2} -> Transpose topo 0 -> FFT ax = {0}
    SlabPlanType plan_2_0_ax02(exec, u_2, u_hat_0_ax02, ax02, topology2,
                               topology0, MPI_COMM_WORLD);
    plan_2_0_ax02.forward(u_2, u_hat_0_ax02);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax02, ref_u_hat_0_ax02));

    plan_2_0_ax02.backward(u_hat_0_ax02, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 0 with ax = {1, 0}:
    // (n0/p, n1, n2) -> (n0, n1, n2/p) -> (n0/2+1, n1, n2/p)
    // Transpose topo 0 -> FFT2 ax = {1, 0}
    SlabPlanType plan_2_0_ax10(exec, u_2, u_hat_0_ax10, ax10, topology2,
                               topology0, MPI_COMM_WORLD);
    plan_2_0_ax10.forward(u_2, u_hat_0_ax10);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax10, ref_u_hat_0_ax10));

    plan_2_0_ax10.backward(u_hat_0_ax10, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 0 with ax = {1, 2}:
    // (n0/p, n1, n2) -> (n0/p, n1, n2/2+1) -> (n0, n1, (n2/2+1)/p)
    // FFT2 ax = {1, 2} -> Transpose topo 0
    SlabPlanType plan_2_0_ax12(exec, u_2, u_hat_0_ax12, ax12, topology2,
                               topology0, MPI_COMM_WORLD);
    plan_2_0_ax12.forward(u_2, u_hat_0_ax12);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax12, ref_u_hat_0_ax12));

    plan_2_0_ax12.backward(u_hat_0_ax12, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 0 with ax = {2, 0}:
    // (n0/p, n1, n2) -> (n0, n1/p, n2) -> (n0/2+1, n1/p, n2)
    // -> (n0/2+1, n1, n2/p)
    // Transpose topo 1 -> FFT2 ax = {2, 0} -> Transpose topo 0
    SlabPlanType plan_2_0_ax20(exec, u_2, u_hat_0_ax20, ax20, topology2,
                               topology0, MPI_COMM_WORLD);
    plan_2_0_ax20.forward(u_2, u_hat_0_ax20);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax20, ref_u_hat_0_ax20));

    plan_2_0_ax20.backward(u_hat_0_ax20, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 0 with ax = {2, 1}:
    // (n0/p, n1, n2) -> (n0/p, n1/2+1, n2) -> (n0, n1/2+1, n2/p)
    // FFT2 ax = {2, 1} -> Transpose topo 0
    SlabPlanType plan_2_0_ax21(exec, u_2, u_hat_0_ax21, ax21, topology2,
                               topology0, MPI_COMM_WORLD);
    plan_2_0_ax21.forward(u_2, u_hat_0_ax21);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax21, ref_u_hat_0_ax21));

    plan_2_0_ax21.backward(u_hat_0_ax21, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 1 with ax = {0, 1}:
    // (n0/p, n1, n2) -> (n0/p, n1/2+1, n2) -> (n0, (n1/2+1)/p, n2)
    //  FFT ax = {1} -> Transpose topo 1 -> FFT ax = {0}
    SlabPlanType plan_2_1_ax01(exec, u_2, u_hat_1_ax01, ax01, topology2,
                               topology1, MPI_COMM_WORLD);
    plan_2_1_ax01.forward(u_2, u_hat_1_ax01);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax01, ref_u_hat_1_ax01));

    plan_2_1_ax01.backward(u_hat_1_ax01, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 1 with ax = {0, 2}:
    // (n0/p, n1, n2) -> (n0/p, n1, n2/2+1) -> (n0, n1/p, n2/2+1)
    // FFT ax = {2} -> Transpose topo 1 -> FFT ax = {0}
    SlabPlanType plan_2_1_ax02(exec, u_2, u_hat_1_ax02, ax02, topology2,
                               topology1, MPI_COMM_WORLD);
    plan_2_1_ax02.forward(u_2, u_hat_1_ax02);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax02, ref_u_hat_1_ax02));

    plan_2_1_ax02.backward(u_hat_1_ax02, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 1 with ax = {1, 0}:
    // (n0/p, n1, n2) -> (n0, n1, n2/p) -> (n0/2+1, n1, n2/p)
    // -> (n0/2+1, n1/p, n2)
    // Transpose topo 0 -> FFT2 ax = {1, 0} -> Transpose topo 1
    SlabPlanType plan_2_1_ax10(exec, u_2, u_hat_1_ax10, ax10, topology2,
                               topology1, MPI_COMM_WORLD);
    plan_2_1_ax10.forward(u_2, u_hat_1_ax10);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax10, ref_u_hat_1_ax10));

    plan_2_1_ax10.backward(u_hat_1_ax10, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 1 with ax = {1, 2}:
    // (n0/p, n1, n2) -> (n0/p, n1, n2/2+1) -> (n0, n1/p, n2/2+1)
    // FFT2 ax = {1, 2} -> Transpose topo 1
    SlabPlanType plan_2_1_ax12(exec, u_2, u_hat_1_ax12, ax12, topology2,
                               topology1, MPI_COMM_WORLD);
    plan_2_1_ax12.forward(u_2, u_hat_1_ax12);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax12, ref_u_hat_1_ax12));

    plan_2_1_ax12.backward(u_hat_1_ax12, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 1 with ax = {2, 0}:
    // (n0/p, n1, n2) -> (n0, n1/p, n2) -> (n0/2+1, n1/p, n2)
    // Transpose topo 1 -> FFT2 ax = {2, 0}
    SlabPlanType plan_2_1_ax20(exec, u_2, u_hat_1_ax20, ax20, topology2,
                               topology1, MPI_COMM_WORLD);
    plan_2_1_ax20.forward(u_2, u_hat_1_ax20);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax20, ref_u_hat_1_ax20));

    plan_2_1_ax20.backward(u_hat_1_ax20, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 1 with ax = {2, 1}:
    // (n0/p, n1, n2) -> (n0/p, n1/2+1, n2) -> (n0, (n1/2+1)/p, n2)
    // FFT2 ax = {2, 1} -> Transpose topo 1
    SlabPlanType plan_2_1_ax21(exec, u_2, u_hat_1_ax21, ax21, topology2,
                               topology1, MPI_COMM_WORLD);
    plan_2_1_ax21.forward(u_2, u_hat_1_ax21);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax21, ref_u_hat_1_ax21));

    plan_2_1_ax21.backward(u_hat_1_ax21, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 2 with ax = {0, 1}:
    // (n0/p, n1, n2) -> (n0/p, n1/2+1, n2) -> (n0, (n1/2+1)/p, n2)
    // -> (n0/p, n1/2+1, n2)
    // FFT ax = {1} -> Transpose topo 1 -> FFT ax = {0} -> Transpose topo 2
    SlabPlanType plan_2_2_ax01(exec, u_2, u_hat_2_ax01, ax01, topology2,
                               topology2, MPI_COMM_WORLD);
    plan_2_2_ax01.forward(u_2, u_hat_2_ax01);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax01, ref_u_hat_2_ax01));

    plan_2_2_ax01.backward(u_hat_2_ax01, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 2 with ax = {0, 2}:
    // (n0/p, n1, n2) -> (n0/p, n1, n2/2+1) -> (n0, n1/p, n2/2+1)
    // -> (n0/p, n1, n2/2+1)
    // FFT ax = {2} -> Transpose topo 1 -> FFT ax = {0} -> Transpose topo 2
    SlabPlanType plan_2_2_ax02(exec, u_2, u_hat_2_ax02, ax02, topology2,
                               topology2, MPI_COMM_WORLD);
    plan_2_2_ax02.forward(u_2, u_hat_2_ax02);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax02, ref_u_hat_2_ax02));

    plan_2_2_ax02.backward(u_hat_2_ax02, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 2 with ax = {1, 0}:
    // (n0/p, n1, n2) -> (n0, n1, n2/p) -> (n0/2+1, n1, n2/p)
    // -> ((n0/2+1)/p, n1, n2)
    // Transpose topo 0 -> FFT2 ax = {1, 0} -> Transpose topo 2
    SlabPlanType plan_2_2_ax10(exec, u_2, u_hat_2_ax10, ax10, topology2,
                               topology2, MPI_COMM_WORLD);
    plan_2_2_ax10.forward(u_2, u_hat_2_ax10);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax10, ref_u_hat_2_ax10));

    plan_2_2_ax10.backward(u_hat_2_ax10, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 2 with ax = {1, 2}:
    // (n0/p, n1, n2) -> (n0/p, n1, n2/2+1)
    // FFT2 ax = {1, 2}
    SlabPlanType plan_2_2_ax12(exec, u_2, u_hat_2_ax12, ax12, topology2,
                               topology2, MPI_COMM_WORLD);
    plan_2_2_ax12.forward(u_2, u_hat_2_ax12);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax12, ref_u_hat_2_ax12));

    plan_2_2_ax12.backward(u_hat_2_ax12, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 2 with ax = {2, 0}:
    // (n0/p, n1, n2) -> (n0, n1/p, n2) -> (n0/2+1, n1/p, n2)
    // -> ((n0/2+1)/p, n1, n2)
    // Transpose topo 1 -> FFT2 ax = {2, 0} -> Transpose topo 2
    SlabPlanType plan_2_2_ax20(exec, u_2, u_hat_2_ax20, ax20, topology2,
                               topology2, MPI_COMM_WORLD);
    plan_2_2_ax20.forward(u_2, u_hat_2_ax20);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax20, ref_u_hat_2_ax20));

    plan_2_2_ax20.backward(u_hat_2_ax20, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 2 with ax = {2, 1}:
    // (n0/p, n1, n2) -> (n0/p, n1/2+1, n2)
    // FFT2 ax = {2, 1}
    SlabPlanType plan_2_2_ax21(exec, u_2, u_hat_2_ax21, ax21, topology2,
                               topology2, MPI_COMM_WORLD);
    plan_2_2_ax21.forward(u_2, u_hat_2_ax21);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax21, ref_u_hat_2_ax21));

    plan_2_2_ax21.backward(u_hat_2_ax21, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));
  }
}

template <typename T, typename LayoutType>
void test_slab3D_view3D(std::size_t nprocs) {
  using RealView3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<T>***, LayoutType, execution_space>;
  using axes_type     = KokkosFFT::axis_type<3>;
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

  // All axes
  axes_type ax012 = {0, 1, 2}, ax021 = {0, 2, 1}, ax102 = {1, 0, 2},
            ax120 = {1, 2, 0}, ax201 = {2, 0, 1}, ax210 = {2, 1, 0};

  // Available combinations
  // Topology 0 -> Topology 1 with all axes
  // Topology 0 -> Topology 2 with all axes
  // Topology 1 -> Topology 2 with all axes

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
  ComplexView3DType gu_hat_ax012("gu_hat_ax012", n0, n1, n2 / 2 + 1),
      gu_hat_ax021("gu_hat_ax021", n0, n1 / 2 + 1, n2),
      gu_hat_ax102("gu_hat_ax102", n0, n1, n2 / 2 + 1),
      gu_hat_ax120("gu_hat_ax120", n0 / 2 + 1, n1, n2),
      gu_hat_ax201("gu_hat_ax201", n0, n1 / 2 + 1, n2),
      gu_hat_ax210("gu_hat_ax210", n0 / 2 + 1, n1, n2);

  // Data in Topology 0 (XY-slab)
  RealView3DType u_0("u_0",
                     KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t0)),
      u_inv_0("u_inv_0",
              KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t0)),
      ref_u_inv_0("ref_u_inv_0",
                  KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t0));
  ComplexView3DType u_hat_0_ax012(
      "u_hat_0_ax012",
      KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax2)),
      u_hat_0_ax021("u_hat_0_ax021", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t0_ax1)),
      u_hat_0_ax102("u_hat_0_ax102", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t0_ax2)),
      u_hat_0_ax120("u_hat_0_ax120", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t0_ax0)),
      u_hat_0_ax201("u_hat_0_ax201", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t0_ax1)),
      u_hat_0_ax210("u_hat_0_ax210", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t0_ax0)),
      ref_u_hat_0_ax012(
          "ref_u_hat_0_ax012",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax2)),
      ref_u_hat_0_ax021(
          "ref_u_hat_0_ax021",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax1)),
      ref_u_hat_0_ax102(
          "ref_u_hat_0_ax102",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax2)),
      ref_u_hat_0_ax120(
          "ref_u_hat_0_ax120",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax0)),
      ref_u_hat_0_ax201(
          "ref_u_hat_0_ax201",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax1)),
      ref_u_hat_0_ax210(
          "ref_u_hat_0_ax210",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax0));

  // Data in Topology 1 (XZ-slab)
  RealView3DType u_1("u_1",
                     KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t1)),
      u_inv_1("u_inv_1",
              KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t1)),
      ref_u_inv_1("ref_u_inv_1",
                  KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t1));
  ComplexView3DType u_hat_1_ax012(
      "u_hat_1_ax012",
      KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax2)),
      u_hat_1_ax021("u_hat_1_ax021", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t1_ax1)),
      u_hat_1_ax102("u_hat_1_ax102", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t1_ax2)),
      u_hat_1_ax120("u_hat_1_ax120", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t1_ax0)),
      u_hat_1_ax201("u_hat_1_ax201", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t1_ax1)),
      u_hat_1_ax210("u_hat_1_ax210", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t1_ax0)),
      ref_u_hat_1_ax012(
          "ref_u_hat_1_ax012",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax2)),
      ref_u_hat_1_ax021(
          "ref_u_hat_1_ax021",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax1)),
      ref_u_hat_1_ax102(
          "ref_u_hat_1_ax102",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax2)),
      ref_u_hat_1_ax120(
          "ref_u_hat_1_ax120",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax0)),
      ref_u_hat_1_ax201(
          "ref_u_hat_1_ax201",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax1)),
      ref_u_hat_1_ax210(
          "ref_u_hat_1_ax210",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax0));

  // Data in Topology 2 (YZ-slab)
  RealView3DType u_2("u_2",
                     KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t2)),
      u_inv_2("u_inv_2",
              KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t2)),
      ref_u_inv_2("ref_u_inv_2",
                  KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t2));
  ComplexView3DType u_hat_2_ax012(
      "u_hat_2_ax012",
      KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax2)),
      u_hat_2_ax021("u_hat_2_ax021", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t2_ax1)),
      u_hat_2_ax102("u_hat_2_ax102", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t2_ax2)),
      u_hat_2_ax120("u_hat_2_ax120", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t2_ax0)),
      u_hat_2_ax201("u_hat_2_ax201", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t2_ax1)),
      u_hat_2_ax210("u_hat_2_ax210", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t2_ax0)),
      ref_u_hat_2_ax012(
          "ref_u_hat_2_ax012",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax2)),
      ref_u_hat_2_ax021(
          "ref_u_hat_2_ax021",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax1)),
      ref_u_hat_2_ax102(
          "ref_u_hat_2_ax102",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax2)),
      ref_u_hat_2_ax120(
          "ref_u_hat_2_ax120",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax0)),
      ref_u_hat_2_ax201(
          "ref_u_hat_2_ax201",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax1)),
      ref_u_hat_2_ax210(
          "ref_u_hat_2_ax210",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax0));

  // Initialization
  execution_space exec;
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::fill_random(gu, random_pool, 1.0);

  KokkosFFT::rfftn(exec, gu, gu_hat_ax012, ax012,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax021, ax021,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax102, ax102,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax120, ax120,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax201, ax201,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax210, ax210,
                   KokkosFFT::Normalization::backward);

  auto h_gu = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu);
  auto h_gu_hat_ax012 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax012);
  auto h_gu_hat_ax021 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax021);
  auto h_gu_hat_ax102 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax102);
  auto h_gu_hat_ax120 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax120);
  auto h_gu_hat_ax201 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax201);
  auto h_gu_hat_ax210 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu_hat_ax210);
  auto h_u_0               = Kokkos::create_mirror_view(u_0);
  auto h_u_1               = Kokkos::create_mirror_view(u_1);
  auto h_u_2               = Kokkos::create_mirror_view(u_2);
  auto h_ref_u_hat_0_ax012 = Kokkos::create_mirror_view(ref_u_hat_0_ax012);
  auto h_ref_u_hat_0_ax021 = Kokkos::create_mirror_view(ref_u_hat_0_ax021);
  auto h_ref_u_hat_0_ax102 = Kokkos::create_mirror_view(ref_u_hat_0_ax102);
  auto h_ref_u_hat_0_ax120 = Kokkos::create_mirror_view(ref_u_hat_0_ax120);
  auto h_ref_u_hat_0_ax201 = Kokkos::create_mirror_view(ref_u_hat_0_ax201);
  auto h_ref_u_hat_0_ax210 = Kokkos::create_mirror_view(ref_u_hat_0_ax210);
  auto h_ref_u_hat_1_ax012 = Kokkos::create_mirror_view(ref_u_hat_1_ax012);
  auto h_ref_u_hat_1_ax021 = Kokkos::create_mirror_view(ref_u_hat_1_ax021);
  auto h_ref_u_hat_1_ax102 = Kokkos::create_mirror_view(ref_u_hat_1_ax102);
  auto h_ref_u_hat_1_ax120 = Kokkos::create_mirror_view(ref_u_hat_1_ax120);
  auto h_ref_u_hat_1_ax201 = Kokkos::create_mirror_view(ref_u_hat_1_ax201);
  auto h_ref_u_hat_1_ax210 = Kokkos::create_mirror_view(ref_u_hat_1_ax210);
  auto h_ref_u_hat_2_ax012 = Kokkos::create_mirror_view(ref_u_hat_2_ax012);
  auto h_ref_u_hat_2_ax021 = Kokkos::create_mirror_view(ref_u_hat_2_ax021);
  auto h_ref_u_hat_2_ax102 = Kokkos::create_mirror_view(ref_u_hat_2_ax102);
  auto h_ref_u_hat_2_ax120 = Kokkos::create_mirror_view(ref_u_hat_2_ax120);
  auto h_ref_u_hat_2_ax201 = Kokkos::create_mirror_view(ref_u_hat_2_ax201);
  auto h_ref_u_hat_2_ax210 = Kokkos::create_mirror_view(ref_u_hat_2_ax210);

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

  // Define ranges for topology 0 (XY-slab)
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax0(
      out_starts_t0_ax0.at(2),
      out_starts_t0_ax0.at(2) + out_extents_t0_ax0.at(2));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax1(
      out_starts_t0_ax1.at(2),
      out_starts_t0_ax1.at(2) + out_extents_t0_ax1.at(2));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax2(
      out_starts_t0_ax2.at(2),
      out_starts_t0_ax2.at(2) + out_extents_t0_ax2.at(2));

  // Topo 0 ax = {0, 1, 2}
  auto h_sub_gu_hat_0_ax012 = Kokkos::subview(h_gu_hat_ax012, Kokkos::ALL,
                                              Kokkos::ALL, range_gu_hat_0_ax2);
  Kokkos::deep_copy(h_ref_u_hat_0_ax012, h_sub_gu_hat_0_ax012);

  // Topo 0 ax = {0, 2, 1}
  auto h_sub_gu_hat_0_ax021 = Kokkos::subview(h_gu_hat_ax021, Kokkos::ALL,
                                              Kokkos::ALL, range_gu_hat_0_ax1);
  Kokkos::deep_copy(h_ref_u_hat_0_ax021, h_sub_gu_hat_0_ax021);

  // Topo 0 ax = {1, 0, 2}
  auto h_sub_gu_hat_0_ax102 = Kokkos::subview(h_gu_hat_ax102, Kokkos::ALL,
                                              Kokkos::ALL, range_gu_hat_0_ax2);
  Kokkos::deep_copy(h_ref_u_hat_0_ax102, h_sub_gu_hat_0_ax102);

  // Topo 0 ax = {1, 2, 0}
  auto h_sub_gu_hat_0_ax120 = Kokkos::subview(h_gu_hat_ax120, Kokkos::ALL,
                                              Kokkos::ALL, range_gu_hat_0_ax0);
  Kokkos::deep_copy(h_ref_u_hat_0_ax120, h_sub_gu_hat_0_ax120);

  // Topo 0 ax = {2, 0, 1}
  auto h_sub_gu_hat_0_ax201 = Kokkos::subview(h_gu_hat_ax201, Kokkos::ALL,
                                              Kokkos::ALL, range_gu_hat_0_ax1);
  Kokkos::deep_copy(h_ref_u_hat_0_ax201, h_sub_gu_hat_0_ax201);

  // Topo 0 ax = {2, 1, 0}
  auto h_sub_gu_hat_0_ax210 = Kokkos::subview(h_gu_hat_ax210, Kokkos::ALL,
                                              Kokkos::ALL, range_gu_hat_0_ax0);
  Kokkos::deep_copy(h_ref_u_hat_0_ax210, h_sub_gu_hat_0_ax210);

  // Define ranges for topology 1 (XZ-slab)
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_1_ax0(
      out_starts_t1_ax0.at(1),
      out_starts_t1_ax0.at(1) + out_extents_t1_ax0.at(1));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_1_ax1(
      out_starts_t1_ax1.at(1),
      out_starts_t1_ax1.at(1) + out_extents_t1_ax1.at(1));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_1_ax2(
      out_starts_t1_ax2.at(1),
      out_starts_t1_ax2.at(1) + out_extents_t1_ax2.at(1));

  // Topo 1 ax = {0, 1, 2}
  auto h_sub_gu_hat_1_ax012 = Kokkos::subview(h_gu_hat_ax012, Kokkos::ALL,
                                              range_gu_hat_1_ax2, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_1_ax012, h_sub_gu_hat_1_ax012);

  // Topo 1 ax = {0, 2, 1}
  auto h_sub_gu_hat_1_ax021 = Kokkos::subview(h_gu_hat_ax021, Kokkos::ALL,
                                              range_gu_hat_1_ax1, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_1_ax021, h_sub_gu_hat_1_ax021);

  // Topo 1 ax = {1, 0, 2}
  auto h_sub_gu_hat_1_ax102 = Kokkos::subview(h_gu_hat_ax102, Kokkos::ALL,
                                              range_gu_hat_1_ax2, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_1_ax102, h_sub_gu_hat_1_ax102);

  // Topo 1 ax = {1, 2, 0}
  auto h_sub_gu_hat_1_ax120 = Kokkos::subview(h_gu_hat_ax120, Kokkos::ALL,
                                              range_gu_hat_1_ax0, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_1_ax120, h_sub_gu_hat_1_ax120);

  // Topo 1 ax = {2, 0, 1}
  auto h_sub_gu_hat_1_ax201 = Kokkos::subview(h_gu_hat_ax201, Kokkos::ALL,
                                              range_gu_hat_1_ax1, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_1_ax201, h_sub_gu_hat_1_ax201);

  // Topo 1 ax = {2, 1, 0}
  auto h_sub_gu_hat_1_ax210 = Kokkos::subview(h_gu_hat_ax210, Kokkos::ALL,
                                              range_gu_hat_1_ax0, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_1_ax210, h_sub_gu_hat_1_ax210);

  // Define ranges for topology 2 (YZ-slab)
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_2_ax0(
      out_starts_t2_ax0.at(0),
      out_starts_t2_ax0.at(0) + out_extents_t2_ax0.at(0));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_2_ax1(
      out_starts_t2_ax1.at(0),
      out_starts_t2_ax1.at(0) + out_extents_t2_ax1.at(0));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_2_ax2(
      out_starts_t2_ax2.at(0),
      out_starts_t2_ax2.at(0) + out_extents_t2_ax2.at(0));

  // Topo 2 ax = {0, 1, 2}
  auto h_sub_gu_hat_2_ax012 = Kokkos::subview(
      h_gu_hat_ax012, range_gu_hat_2_ax2, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_2_ax012, h_sub_gu_hat_2_ax012);

  // Topo 2 ax = {0, 2, 1}
  auto h_sub_gu_hat_2_ax021 = Kokkos::subview(
      h_gu_hat_ax021, range_gu_hat_2_ax1, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_2_ax021, h_sub_gu_hat_2_ax021);

  // Topo 2 ax = {1, 0, 2}
  auto h_sub_gu_hat_2_ax102 = Kokkos::subview(
      h_gu_hat_ax102, range_gu_hat_2_ax2, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_2_ax102, h_sub_gu_hat_2_ax102);

  // Topo 2 ax = {1, 2, 0}
  auto h_sub_gu_hat_2_ax120 = Kokkos::subview(
      h_gu_hat_ax120, range_gu_hat_2_ax0, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_2_ax120, h_sub_gu_hat_2_ax120);

  // Topo 2 ax = {2, 0, 1}
  auto h_sub_gu_hat_2_ax201 = Kokkos::subview(
      h_gu_hat_ax201, range_gu_hat_2_ax1, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_2_ax201, h_sub_gu_hat_2_ax201);

  // Topo 2 ax = {2, 1, 0}
  auto h_sub_gu_hat_2_ax210 = Kokkos::subview(
      h_gu_hat_ax210, range_gu_hat_2_ax0, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(h_ref_u_hat_2_ax210, h_sub_gu_hat_2_ax210);

  Kokkos::deep_copy(u_0, h_u_0);
  Kokkos::deep_copy(u_1, h_u_1);
  Kokkos::deep_copy(u_2, h_u_2);

  Kokkos::deep_copy(ref_u_hat_0_ax012, h_ref_u_hat_0_ax012);
  Kokkos::deep_copy(ref_u_hat_0_ax021, h_ref_u_hat_0_ax021);
  Kokkos::deep_copy(ref_u_hat_0_ax102, h_ref_u_hat_0_ax102);
  Kokkos::deep_copy(ref_u_hat_0_ax120, h_ref_u_hat_0_ax120);
  Kokkos::deep_copy(ref_u_hat_0_ax201, h_ref_u_hat_0_ax201);
  Kokkos::deep_copy(ref_u_hat_0_ax210, h_ref_u_hat_0_ax210);

  Kokkos::deep_copy(ref_u_hat_1_ax012, h_ref_u_hat_1_ax012);
  Kokkos::deep_copy(ref_u_hat_1_ax021, h_ref_u_hat_1_ax021);
  Kokkos::deep_copy(ref_u_hat_1_ax102, h_ref_u_hat_1_ax102);
  Kokkos::deep_copy(ref_u_hat_1_ax120, h_ref_u_hat_1_ax120);
  Kokkos::deep_copy(ref_u_hat_1_ax201, h_ref_u_hat_1_ax201);
  Kokkos::deep_copy(ref_u_hat_1_ax210, h_ref_u_hat_1_ax210);

  Kokkos::deep_copy(ref_u_hat_2_ax012, h_ref_u_hat_2_ax012);
  Kokkos::deep_copy(ref_u_hat_2_ax021, h_ref_u_hat_2_ax021);
  Kokkos::deep_copy(ref_u_hat_2_ax102, h_ref_u_hat_2_ax102);
  Kokkos::deep_copy(ref_u_hat_2_ax120, h_ref_u_hat_2_ax120);
  Kokkos::deep_copy(ref_u_hat_2_ax201, h_ref_u_hat_2_ax201);
  Kokkos::deep_copy(ref_u_hat_2_ax210, h_ref_u_hat_2_ax210);

  // For inverse transform
  Kokkos::deep_copy(ref_u_inv_0, u_0);
  Kokkos::deep_copy(ref_u_inv_1, u_1);
  Kokkos::deep_copy(ref_u_inv_2, u_2);

  using SlabPlanType =
      SlabPlan<execution_space, RealView3DType, ComplexView3DType, 3>;

  // Not a slab geometry
  if (nprocs == 1) {
    // topology0 -> topology1
    // (n0, n1, n2/p) -> (n0, (n1/2+1)/p, n2)
    ASSERT_THROW(
        {
          SlabPlanType plan_0_1_ax012(exec, u_0, u_hat_1_ax012, ax012,
                                      topology0, topology1, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology0 -> topology2
    // (n0, n1, n2/p) -> (n0/p, n1/2+1, n2)
    ASSERT_THROW(
        {
          SlabPlanType plan_0_2_ax012(exec, u_0, u_hat_2_ax012, ax012,
                                      topology0, topology2, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology1 -> topology2
    // (n0, n1/p, n2) -> (n0/p, n1, n2/2+1)
    ASSERT_THROW(
        {
          SlabPlanType plan_1_2_ax012(exec, u_1, u_hat_2_ax012, ax012,
                                      topology1, topology2, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology0 -> topology0 with ax = {1, 2}
    // (n0, n1, n2/p) -> (n0, n1, (n2/2+1)/p)
    ASSERT_THROW(
        {
          SlabPlanType plan_0_0_ax012(exec, u_0, u_hat_0_ax012, ax012,
                                      topology0, topology0, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology1 -> topology1 with ax = {1, 2}
    // (n0, n1/p, n2) -> (n0, (n1/2+1)/p, n2)
    ASSERT_THROW(
        {
          SlabPlanType plan_1_1_ax210(exec, u_1, u_hat_1_ax210, ax210,
                                      topology1, topology1, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology2 -> topology2 with ax = {0, 1, 2}
    // (n0/p, n1, n2) -> (n0/p, (n1/2+1), n2)
    ASSERT_THROW(
        {
          SlabPlanType plan_2_2_ax012(exec, u_2, u_hat_2_ax012, ax012,
                                      topology2, topology2, MPI_COMM_WORLD);
        },
        std::runtime_error);
  } else {
    // topo 0 -> topo 0 with ax = {0, 1, 2}:
    // (n0, n1, n2/p) -> (n0/p, n1, n2) -> (n0/p, n1, n2/2+1) -> (n0, n1,
    // (n2/2+1)/p) Transpose -> FFT2 ax = {1, 2} -> Transpose -> FFT ax = {0}
    SlabPlanType plan_0_0_ax012(exec, u_0, u_hat_0_ax012, ax012, topology0,
                                topology0, MPI_COMM_WORLD);
    plan_0_0_ax012.forward(u_0, u_hat_0_ax012);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax012, ref_u_hat_0_ax012));

    plan_0_0_ax012.backward(u_hat_0_ax012, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 0 with ax = {0, 2, 1}:
    // (n0, n1, n2/p) -> (n0/p, n1, n2) -> (n0/p, n1/2+1, n2) -> (n0, n1/2+1,
    // n2/p) FFT ax = {1} -> Transpose topo 1 -> FFT2 ax = {0, 2} -> Transpose
    SlabPlanType plan_0_0_ax021(exec, u_0, u_hat_0_ax021, ax021, topology0,
                                topology0, MPI_COMM_WORLD);
    plan_0_0_ax021.forward(u_0, u_hat_0_ax021);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax021, ref_u_hat_0_ax021));

    plan_0_0_ax021.backward(u_hat_0_ax021, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 0 with ax = {1, 0, 2}:
    // (n0, n1, n2/p) -> (n0, n1/p, n2) -> (n0, n1/p, n2/2+1) -> (n0, n1,
    // (n2/2+1)/p) Transpose -> FFT2 ax = {0, 2} -> Transpose -> FFT ax = {1}
    SlabPlanType plan_0_0_ax102(exec, u_0, u_hat_0_ax102, ax102, topology0,
                                topology0, MPI_COMM_WORLD);
    plan_0_0_ax102.forward(u_0, u_hat_0_ax102);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax102, ref_u_hat_0_ax102));

    plan_0_0_ax102.backward(u_hat_0_ax102, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 0 with ax = {1, 2, 0}:
    // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2) -> (n0/2+1,
    // n1, n2/p) FFT ax = {0} -> Transpose topo 2 -> FFT2 ax = {1, 2} ->
    // Transpose
    SlabPlanType plan_0_0_ax120(exec, u_0, u_hat_0_ax120, ax120, topology0,
                                topology0, MPI_COMM_WORLD);
    plan_0_0_ax120.forward(u_0, u_hat_0_ax120);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax120, ref_u_hat_0_ax120));

    plan_0_0_ax120.backward(u_hat_0_ax120, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 0 with ax = {2, 0, 1}:
    // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0, (n1/2+1)/p, n2) -> (n0,
    // n1/2+1, n2/p) FFT2 ax = {0, 1} -> Transpose topo 1 -> FFT ax = {2} ->
    // Transpose
    SlabPlanType plan_0_0_ax201(exec, u_0, u_hat_0_ax201, ax201, topology0,
                                topology0, MPI_COMM_WORLD);
    plan_0_0_ax201.forward(u_0, u_hat_0_ax201);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax201, ref_u_hat_0_ax201));

    plan_0_0_ax201.backward(u_hat_0_ax201, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 0 with ax = {2, 1, 0}:
    // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2) -> (n0/2+1,
    // n1, n2/p) FFT2 ax = {1, 0} -> Transpose topo 2 -> FFT ax = {2} ->
    // Transpose
    SlabPlanType plan_0_0_ax210(exec, u_0, u_hat_0_ax210, ax210, topology0,
                                topology0, MPI_COMM_WORLD);
    plan_0_0_ax210.forward(u_0, u_hat_0_ax210);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax210, ref_u_hat_0_ax210));

    plan_0_0_ax210.backward(u_hat_0_ax210, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 1 with ax = {0, 1, 2}:
    // (n0, n1, n2/p) -> (n0/p, n1, n2) -> (n0/p, n1, n2/2+1)
    // -> (n0, n1/p, n2/2+1)
    // Transpose -> FFT2 ax = {1, 2} -> Transpose topo 1 -> FFT ax = {1}
    SlabPlanType plan_0_1_ax012(exec, u_0, u_hat_1_ax012, ax012, topology0,
                                topology1, MPI_COMM_WORLD);
    plan_0_1_ax012.forward(u_0, u_hat_1_ax012);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax012, ref_u_hat_1_ax012));

    plan_0_1_ax012.backward(u_hat_1_ax012, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 1 with ax = {0, 2, 1}:
    // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0, (n1/2+1)/p, n2)
    // FFT ax = {1} -> Transpose topo 1 -> FFT2 ax = {0, 2}
    SlabPlanType plan_0_1_ax021(exec, u_0, u_hat_1_ax021, ax021, topology0,
                                topology1, MPI_COMM_WORLD);
    plan_0_1_ax021.forward(u_0, u_hat_1_ax021);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax021, ref_u_hat_1_ax021));

    plan_0_1_ax021.backward(u_hat_1_ax021, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 1 with ax = {1, 0, 2}:
    // (n0, n1, n2/p) -> (n0, n1/p, n2) -> (n0, n1/p, n2/2+1)
    // -> (n0/p, n1, n2/2+1) -> (n0, n1/p, n2/2+1)
    // Transpose topo 1 -> FFT2 ax = {0, 2} -> Transpose topo 2 -> FFT ax = {1}
    // -> Transpose topo 1
    SlabPlanType plan_0_1_ax102(exec, u_0, u_hat_1_ax102, ax102, topology0,
                                topology1, MPI_COMM_WORLD);
    plan_0_1_ax102.forward(u_0, u_hat_1_ax102);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax102, ref_u_hat_1_ax102));

    plan_0_1_ax102.backward(u_hat_1_ax102, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 1 with ax = {1, 2, 0}:
    // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
    // -> (n0/2+1, n1/p, n2)
    // FFT ax = {0} -> Transpose topo 2 -> FFT2 ax = {1, 2} ->
    // Transpose topo 1
    SlabPlanType plan_0_1_ax120(exec, u_0, u_hat_1_ax120, ax120, topology0,
                                topology1, MPI_COMM_WORLD);
    plan_0_1_ax120.forward(u_0, u_hat_1_ax120);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax120, ref_u_hat_1_ax120));

    plan_0_1_ax120.backward(u_hat_1_ax120, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 1 with ax = {2, 0, 1}:
    // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0, (n1/2+1)/p, n2)
    // FFT2 ax = {0, 1} -> Transpose topo 1 -> FFT ax = {2}
    SlabPlanType plan_0_1_ax201(exec, u_0, u_hat_1_ax201, ax201, topology0,
                                topology1, MPI_COMM_WORLD);
    plan_0_1_ax201.forward(u_0, u_hat_1_ax201);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax201, ref_u_hat_1_ax201));

    plan_0_1_ax201.backward(u_hat_1_ax201, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 1 with ax = {2, 1, 0}:
    // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> (n0/2+1, n1/p, n2)
    // FFT2 ax = {1, 0} -> Transpose topo 1 -> FFT ax = {2}
    SlabPlanType plan_0_1_ax210(exec, u_0, u_hat_1_ax210, ax210, topology0,
                                topology1, MPI_COMM_WORLD);
    plan_0_1_ax210.forward(u_0, u_hat_1_ax210);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax210, ref_u_hat_1_ax210));

    plan_0_1_ax210.backward(u_hat_1_ax210, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 2 with ax = {0, 1, 2}:
    // (n0, n1, n2/p) -> (n0/p, n1, n2) -> (n0/p, n1, n2/2+1)
    // -> (n0, n1/p, n2/2+1) -> (n0/p, n1, n2/2+1)
    // Transpose topo 2 -> FFT2 ax = {1, 2} -> Transpose topo 1 -> FFT ax = {2}
    // Transpose topo 2
    SlabPlanType plan_0_2_ax012(exec, u_0, u_hat_2_ax012, ax012, topology0,
                                topology2, MPI_COMM_WORLD);
    plan_0_2_ax012.forward(u_0, u_hat_2_ax012);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax012, ref_u_hat_2_ax012));

    plan_0_2_ax012.backward(u_hat_2_ax012, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 2 with ax = {0, 2, 1}:
    // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0, (n1/2+1)/p, n2)
    // -> (n0/p, n1/2+1, n2)
    // FFT ax = {1} -> Transpose topo 1 -> FFT2 ax = {0, 2} ->
    // Transpose topo 2
    SlabPlanType plan_0_2_ax021(exec, u_0, u_hat_2_ax021, ax021, topology0,
                                topology2, MPI_COMM_WORLD);
    plan_0_2_ax021.forward(u_0, u_hat_2_ax021);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax021, ref_u_hat_2_ax021));

    plan_0_2_ax021.backward(u_hat_2_ax021, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 2 with ax = {1, 0, 2}:
    // (n0, n1, n2/p) -> (n0, n1/p, n2) -> (n0, n1/p, n2/2+1)
    // -> (n0/p, n1, n2/2+1)
    // Transpose topo 1 -> FFT2 ax = {0, 2} -> Transpose topo 2 -> FFT ax = {1}
    SlabPlanType plan_0_2_ax102(exec, u_0, u_hat_2_ax102, ax102, topology0,
                                topology2, MPI_COMM_WORLD);
    plan_0_2_ax102.forward(u_0, u_hat_2_ax102);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax102, ref_u_hat_2_ax102));

    plan_0_2_ax102.backward(u_hat_2_ax102, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 2 with ax = {1, 2, 0}:
    // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
    // FFT ax = {0} -> Transpose topo 2 -> FFT2 ax = {1, 2}
    SlabPlanType plan_0_2_ax120(exec, u_0, u_hat_2_ax120, ax120, topology0,
                                topology2, MPI_COMM_WORLD);
    plan_0_2_ax120.forward(u_0, u_hat_2_ax120);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax120, ref_u_hat_2_ax120));

    plan_0_2_ax120.backward(u_hat_2_ax120, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 2 with ax = {2, 0, 1}:
    // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0/p, n1/2+1, n2)
    // FFT2 ax = {0, 1} -> Transpose topo 2 -> FFT ax = {2}
    SlabPlanType plan_0_2_ax201(exec, u_0, u_hat_2_ax201, ax201, topology0,
                                topology2, MPI_COMM_WORLD);
    plan_0_2_ax201.forward(u_0, u_hat_2_ax201);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax201, ref_u_hat_2_ax201));

    plan_0_2_ax201.backward(u_hat_2_ax201, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 2 with ax = {2, 1, 0}:
    // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
    // FFT2 ax = {0, 1} -> Transpose topo 2 -> FFT ax = {2}
    SlabPlanType plan_0_2_ax210(exec, u_0, u_hat_2_ax210, ax210, topology0,
                                topology2, MPI_COMM_WORLD);
    plan_0_2_ax210.forward(u_0, u_hat_2_ax210);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax210, ref_u_hat_2_ax210));

    plan_0_2_ax210.backward(u_hat_2_ax210, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 1 with ax = {0, 1, 2}:
    // (n0/p, n1, n2) -> (n0/p, n1, n2/2+1) -> (n0, n1/p, n2/2+1)
    // FFT2 ax = {1, 2} -> Transpose -> FFT ax = {0}
    SlabPlanType plan_2_1_ax012(exec, u_2, u_hat_1_ax012, ax012, topology2,
                                topology1, MPI_COMM_WORLD);
    plan_2_1_ax012.forward(u_2, u_hat_1_ax012);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax012, ref_u_hat_1_ax012));

    plan_2_1_ax012.backward(u_hat_1_ax012, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));
  }
}

template <typename T, typename LayoutType>
void test_slab3D_view4D(std::size_t nprocs) {
  using RealView4DType = Kokkos::View<T****, LayoutType, execution_space>;
  using ComplexView4DType =
      Kokkos::View<Kokkos::complex<T>****, LayoutType, execution_space>;
  using axes_type     = KokkosFFT::axis_type<3>;
  using extents_type  = std::array<std::size_t, 4>;
  using topology_type = std::array<std::size_t, 4>;

  topology_type topology0{1, 1, 1, nprocs}, topology1{1, 1, nprocs, 1},
      topology2{1, nprocs, 1, 1}, topology3{nprocs, 1, 1, 1};

  const std::size_t n0 = 8, n1 = 7, n2 = 5, n3 = 6;
  extents_type global_in_extents{n0, n1, n2, n3},
      global_out_extents_ax0{n0 / 2 + 1, n1, n2, n3},
      global_out_extents_ax1{n0, n1 / 2 + 1, n2, n3},
      global_out_extents_ax2{n0, n1, n2 / 2 + 1, n3},
      global_out_extents_ax3{n0, n1, n2, n3 / 2 + 1};

  // All axes
  axes_type ax012 = {0, 1, 2}, ax021 = {0, 2, 1}, ax102 = {1, 0, 2},
            ax120 = {1, 2, 0}, ax201 = {2, 0, 1}, ax210 = {2, 1, 0},
            ax123 = {1, 2, 3}, ax132 = {1, 3, 2}, ax213 = {2, 1, 3};

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
      gu_hat_ax021("gu_hat_ax021", n0, n1 / 2 + 1, n2, n3),
      gu_hat_ax102("gu_hat_ax102", n0, n1, n2 / 2 + 1, n3),
      gu_hat_ax120("gu_hat_ax120", n0 / 2 + 1, n1, n2, n3),
      gu_hat_ax201("gu_hat_ax201", n0, n1 / 2 + 1, n2, n3),
      gu_hat_ax210("gu_hat_ax210", n0 / 2 + 1, n1, n2, n3),
      gu_hat_ax123("gu_hat_ax123", n0, n1, n2, n3 / 2 + 1),
      gu_hat_ax132("gu_hat_ax132", n0, n1, n2 / 2 + 1, n3),
      gu_hat_ax213("gu_hat_ax213", n0, n1, n2, n3 / 2 + 1);

  // Data in Topology 0 (XYZ-slab)
  RealView4DType u_0("u_0",
                     KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t0)),
      u_inv_0("u_inv_0",
              KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t0)),
      ref_u_inv_0("ref_u_inv_0",
                  KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t0));
  ComplexView4DType u_hat_0_ax012(
      "u_hat_0_ax012",
      KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax2)),
      u_hat_0_ax021("u_hat_0_ax021", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t0_ax1)),
      u_hat_0_ax102("u_hat_0_ax102", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t0_ax2)),
      u_hat_0_ax120("u_hat_0_ax120", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t0_ax0)),
      u_hat_0_ax201("u_hat_0_ax201", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t0_ax1)),
      u_hat_0_ax210("u_hat_0_ax210", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t0_ax0)),
      u_hat_0_ax123("u_hat_0_ax123", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t0_ax3)),
      u_hat_0_ax132("u_hat_0_ax132", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t0_ax2)),
      u_hat_0_ax213("u_hat_0_ax213", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t0_ax3)),
      ref_u_hat_0_ax012(
          "ref_u_hat_0_ax012",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax2)),
      ref_u_hat_0_ax021(
          "ref_u_hat_0_ax021",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax1)),
      ref_u_hat_0_ax102(
          "ref_u_hat_0_ax102",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax2)),
      ref_u_hat_0_ax120(
          "ref_u_hat_0_ax120",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax0)),
      ref_u_hat_0_ax201(
          "ref_u_hat_0_ax201",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax1)),
      ref_u_hat_0_ax210(
          "ref_u_hat_0_ax210",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax0)),
      ref_u_hat_0_ax123(
          "ref_u_hat_0_ax123",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax3)),
      ref_u_hat_0_ax132(
          "ref_u_hat_0_ax132",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax2)),
      ref_u_hat_0_ax213(
          "ref_u_hat_0_ax213",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax3));

  // Data in Topology 1 (XYW-slab)
  RealView4DType u_1("u_1",
                     KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t1)),
      u_inv_1("u_inv_1",
              KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t1)),
      ref_u_inv_1("ref_u_inv_1",
                  KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t1));
  ComplexView4DType u_hat_1_ax012(
      "u_hat_1_ax012",
      KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax2)),
      u_hat_1_ax021("u_hat_1_ax021", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t1_ax1)),
      u_hat_1_ax102("u_hat_1_ax102", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t1_ax2)),
      u_hat_1_ax120("u_hat_1_ax120", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t1_ax0)),
      u_hat_1_ax201("u_hat_1_ax201", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t1_ax1)),
      u_hat_1_ax210("u_hat_1_ax210", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t1_ax0)),
      u_hat_1_ax123("u_hat_1_ax123", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t1_ax3)),
      u_hat_1_ax132("u_hat_1_ax132", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t1_ax2)),
      u_hat_1_ax213("u_hat_1_ax213", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t1_ax3)),
      ref_u_hat_1_ax012(
          "ref_u_hat_1_ax012",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax2)),
      ref_u_hat_1_ax021(
          "ref_u_hat_1_ax021",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax1)),
      ref_u_hat_1_ax102(
          "ref_u_hat_1_ax102",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax2)),
      ref_u_hat_1_ax120(
          "ref_u_hat_1_ax120",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax0)),
      ref_u_hat_1_ax201(
          "ref_u_hat_1_ax201",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax1)),
      ref_u_hat_1_ax210(
          "ref_u_hat_1_ax210",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax0)),
      ref_u_hat_1_ax123(
          "ref_u_hat_1_ax123",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax3)),
      ref_u_hat_1_ax132(
          "ref_u_hat_1_ax132",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax2)),
      ref_u_hat_1_ax213(
          "ref_u_hat_1_ax213",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax3));

  // Data in Topology 2 (XZW-slab)
  RealView4DType u_2("u_2",
                     KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t2)),
      u_inv_2("u_inv_2",
              KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t2)),
      ref_u_inv_2("ref_u_inv_2",
                  KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t2));
  ComplexView4DType u_hat_2_ax012(
      "u_hat_2_ax012",
      KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax2)),
      u_hat_2_ax021("u_hat_2_ax021", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t2_ax1)),
      u_hat_2_ax102("u_hat_2_ax102", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t2_ax2)),
      u_hat_2_ax120("u_hat_2_ax120", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t2_ax0)),
      u_hat_2_ax201("u_hat_2_ax201", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t2_ax1)),
      u_hat_2_ax210("u_hat_2_ax210", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t2_ax0)),
      u_hat_2_ax123("u_hat_2_ax123", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t2_ax3)),
      u_hat_2_ax132("u_hat_2_ax132", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t2_ax2)),
      u_hat_2_ax213("u_hat_2_ax213", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t2_ax3)),
      ref_u_hat_2_ax012(
          "ref_u_hat_2_ax012",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax2)),
      ref_u_hat_2_ax021(
          "ref_u_hat_2_ax021",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax1)),
      ref_u_hat_2_ax102(
          "ref_u_hat_2_ax102",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax2)),
      ref_u_hat_2_ax120(
          "ref_u_hat_2_ax120",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax0)),
      ref_u_hat_2_ax201(
          "ref_u_hat_2_ax201",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax1)),
      ref_u_hat_2_ax210(
          "ref_u_hat_2_ax210",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax0)),
      ref_u_hat_2_ax123(
          "ref_u_hat_2_ax123",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax3)),
      ref_u_hat_2_ax132(
          "ref_u_hat_2_ax132",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax2)),
      ref_u_hat_2_ax213(
          "ref_u_hat_2_ax213",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t2_ax3));

  // Data in Topology 3 (YZW-slab)
  RealView4DType u_3("u_3",
                     KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t3)),
      u_inv_3("u_inv_3",
              KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t3)),
      ref_u_inv_3("ref_u_inv_3",
                  KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t3));

  ComplexView4DType u_hat_3_ax012(
      "u_hat_3_ax012",
      KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t3_ax2)),
      u_hat_3_ax021("u_hat_3_ax021", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t3_ax1)),
      u_hat_3_ax102("u_hat_3_ax102", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t3_ax2)),
      u_hat_3_ax120("u_hat_3_ax120", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t3_ax0)),
      u_hat_3_ax201("u_hat_3_ax201", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t3_ax1)),
      u_hat_3_ax210("u_hat_3_ax210", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t3_ax0)),
      u_hat_3_ax123("u_hat_3_ax123", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t3_ax3)),
      u_hat_3_ax132("u_hat_3_ax132", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t3_ax2)),
      u_hat_3_ax213("u_hat_3_ax213", KokkosFFT::Impl::create_layout<LayoutType>(
                                         out_extents_t3_ax3)),
      ref_u_hat_3_ax012(
          "ref_u_hat_3_ax012",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t3_ax2)),
      ref_u_hat_3_ax021(
          "ref_u_hat_3_ax021",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t3_ax1)),
      ref_u_hat_3_ax102(
          "ref_u_hat_3_ax102",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t3_ax2)),
      ref_u_hat_3_ax120(
          "ref_u_hat_3_ax120",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t3_ax0)),
      ref_u_hat_3_ax201(
          "ref_u_hat_3_ax201",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t3_ax1)),
      ref_u_hat_3_ax210(
          "ref_u_hat_3_ax210",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t3_ax0)),
      ref_u_hat_3_ax123(
          "ref_u_hat_3_ax123",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t3_ax3)),
      ref_u_hat_3_ax132(
          "ref_u_hat_3_ax132",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t3_ax2)),
      ref_u_hat_3_ax213(
          "ref_u_hat_3_ax213",
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t3_ax3));

  // Initialization
  execution_space exec;
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed*/ 12345);
  Kokkos::fill_random(gu, random_pool, 1.0);

  KokkosFFT::rfftn(exec, gu, gu_hat_ax012, ax012,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax021, ax021,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax102, ax102,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax120, ax120,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax201, ax201,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax210, ax210,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax123, ax123,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax132, ax132,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::rfftn(exec, gu, gu_hat_ax213, ax213,
                   KokkosFFT::Normalization::backward);

  // Topo 0
  Kokkos::pair<std::size_t, std::size_t> range_gu0(
      in_starts_t0.at(3), in_starts_t0.at(3) + in_extents_t0.at(3));
  auto sub_gu_0 =
      Kokkos::subview(gu, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, range_gu0);
  Kokkos::deep_copy(u_0, sub_gu_0);

  // Topo 1
  Kokkos::pair<std::size_t, std::size_t> range_gu1(
      in_starts_t1.at(2), in_starts_t1.at(2) + in_extents_t1.at(2));
  auto sub_gu_1 =
      Kokkos::subview(gu, Kokkos::ALL, Kokkos::ALL, range_gu1, Kokkos::ALL);
  Kokkos::deep_copy(u_1, sub_gu_1);

  // Topo 2
  Kokkos::pair<std::size_t, std::size_t> range_gu2(
      in_starts_t2.at(1), in_starts_t2.at(1) + in_extents_t2.at(1));
  auto sub_gu_2 =
      Kokkos::subview(gu, Kokkos::ALL, range_gu2, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(u_2, sub_gu_2);

  // Topo 3
  Kokkos::pair<std::size_t, std::size_t> range_gu3(
      in_starts_t3.at(0), in_starts_t3.at(0) + in_extents_t3.at(0));
  auto sub_gu_3 =
      Kokkos::subview(gu, range_gu3, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(u_3, sub_gu_3);

  // For inverse transform
  Kokkos::deep_copy(ref_u_inv_0, u_0);
  Kokkos::deep_copy(ref_u_inv_1, u_1);
  Kokkos::deep_copy(ref_u_inv_2, u_2);
  Kokkos::deep_copy(ref_u_inv_3, u_3);

  // Define ranges for topology 0 (XYZ-slab)
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax0(
      out_starts_t0_ax0.at(3),
      out_starts_t0_ax0.at(3) + out_extents_t0_ax0.at(3));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax1(
      out_starts_t0_ax1.at(3),
      out_starts_t0_ax1.at(3) + out_extents_t0_ax1.at(3));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax2(
      out_starts_t0_ax2.at(3),
      out_starts_t0_ax2.at(3) + out_extents_t0_ax2.at(3));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax3(
      out_starts_t0_ax3.at(3),
      out_starts_t0_ax3.at(3) + out_extents_t0_ax3.at(3));

  // Topo 0 ax = {0, 1, 2}
  auto sub_gu_hat_0_ax012 = Kokkos::subview(
      gu_hat_ax012, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, range_gu_hat_0_ax2);
  Kokkos::deep_copy(ref_u_hat_0_ax012, sub_gu_hat_0_ax012);

  // Topo 0 ax = {0, 2, 1}
  auto sub_gu_hat_0_ax021 = Kokkos::subview(
      gu_hat_ax021, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, range_gu_hat_0_ax1);
  Kokkos::deep_copy(ref_u_hat_0_ax021, sub_gu_hat_0_ax021);

  // Topo 0 ax = {1, 0, 2}
  auto sub_gu_hat_0_ax102 = Kokkos::subview(
      gu_hat_ax102, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, range_gu_hat_0_ax2);
  Kokkos::deep_copy(ref_u_hat_0_ax102, sub_gu_hat_0_ax102);

  // Topo 0 ax = {1, 2, 0}
  auto sub_gu_hat_0_ax120 = Kokkos::subview(
      gu_hat_ax120, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, range_gu_hat_0_ax0);
  Kokkos::deep_copy(ref_u_hat_0_ax120, sub_gu_hat_0_ax120);

  // Topo 0 ax = {2, 0, 1}
  auto sub_gu_hat_0_ax201 = Kokkos::subview(
      gu_hat_ax201, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, range_gu_hat_0_ax1);
  Kokkos::deep_copy(ref_u_hat_0_ax201, sub_gu_hat_0_ax201);

  // Topo 0 ax = {2, 1, 0}
  auto sub_gu_hat_0_ax210 = Kokkos::subview(
      gu_hat_ax210, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, range_gu_hat_0_ax0);
  Kokkos::deep_copy(ref_u_hat_0_ax210, sub_gu_hat_0_ax210);

  // Topo 0 ax = {1, 2, 3}
  auto sub_gu_hat_0_ax123 = Kokkos::subview(
      gu_hat_ax123, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, range_gu_hat_0_ax3);
  Kokkos::deep_copy(ref_u_hat_0_ax123, sub_gu_hat_0_ax123);

  // Topo 0 ax = {1, 3, 2}
  auto sub_gu_hat_0_ax132 = Kokkos::subview(
      gu_hat_ax132, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, range_gu_hat_0_ax2);
  Kokkos::deep_copy(ref_u_hat_0_ax132, sub_gu_hat_0_ax132);

  // Topo 0 ax = {2, 1, 3}
  auto sub_gu_hat_0_ax213 = Kokkos::subview(
      gu_hat_ax213, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, range_gu_hat_0_ax3);
  Kokkos::deep_copy(ref_u_hat_0_ax213, sub_gu_hat_0_ax213);

  // Define ranges for topology 1 (XYW-slab)
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_1_ax0(
      out_starts_t1_ax0.at(2),
      out_starts_t1_ax0.at(2) + out_extents_t1_ax0.at(2));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_1_ax1(
      out_starts_t1_ax1.at(2),
      out_starts_t1_ax1.at(2) + out_extents_t1_ax1.at(2));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_1_ax2(
      out_starts_t1_ax2.at(2),
      out_starts_t1_ax2.at(2) + out_extents_t1_ax2.at(2));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_1_ax3(
      out_starts_t1_ax3.at(2),
      out_starts_t1_ax3.at(2) + out_extents_t1_ax3.at(2));

  // Topo 1 ax = {0, 1, 2}
  auto sub_gu_hat_1_ax012 = Kokkos::subview(
      gu_hat_ax012, Kokkos::ALL, Kokkos::ALL, range_gu_hat_1_ax2, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_1_ax012, sub_gu_hat_1_ax012);

  // Topo 1 ax = {0, 2, 1}
  auto sub_gu_hat_1_ax021 = Kokkos::subview(
      gu_hat_ax021, Kokkos::ALL, Kokkos::ALL, range_gu_hat_1_ax1, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_1_ax021, sub_gu_hat_1_ax021);

  // Topo 1 ax = {1, 0, 2}
  auto sub_gu_hat_1_ax102 = Kokkos::subview(
      gu_hat_ax102, Kokkos::ALL, Kokkos::ALL, range_gu_hat_1_ax2, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_1_ax102, sub_gu_hat_1_ax102);

  // Topo 1 ax = {1, 2, 0}
  auto sub_gu_hat_1_ax120 = Kokkos::subview(
      gu_hat_ax120, Kokkos::ALL, Kokkos::ALL, range_gu_hat_1_ax0, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_1_ax120, sub_gu_hat_1_ax120);

  // Topo 1 ax = {2, 0, 1}
  auto sub_gu_hat_1_ax201 = Kokkos::subview(
      gu_hat_ax201, Kokkos::ALL, Kokkos::ALL, range_gu_hat_1_ax1, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_1_ax201, sub_gu_hat_1_ax201);

  // Topo 1 ax = {2, 1, 0}
  auto sub_gu_hat_1_ax210 = Kokkos::subview(
      gu_hat_ax210, Kokkos::ALL, Kokkos::ALL, range_gu_hat_1_ax0, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_1_ax210, sub_gu_hat_1_ax210);

  // Topo 1 ax = {1, 2, 3}
  auto sub_gu_hat_1_ax123 = Kokkos::subview(
      gu_hat_ax123, Kokkos::ALL, Kokkos::ALL, range_gu_hat_1_ax3, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_1_ax123, sub_gu_hat_1_ax123);

  // Topo 1 ax = {1, 3, 2}
  auto sub_gu_hat_1_ax132 = Kokkos::subview(
      gu_hat_ax132, Kokkos::ALL, Kokkos::ALL, range_gu_hat_1_ax2, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_1_ax132, sub_gu_hat_1_ax132);

  // Topo 1 ax = {2, 1, 3}
  auto sub_gu_hat_1_ax213 = Kokkos::subview(
      gu_hat_ax213, Kokkos::ALL, Kokkos::ALL, range_gu_hat_1_ax3, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_1_ax213, sub_gu_hat_1_ax213);

  // Define ranges for topology 2 (XZW-slab)
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_2_ax0(
      out_starts_t2_ax0.at(1),
      out_starts_t2_ax0.at(1) + out_extents_t2_ax0.at(1));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_2_ax1(
      out_starts_t2_ax1.at(1),
      out_starts_t2_ax1.at(1) + out_extents_t2_ax1.at(1));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_2_ax2(
      out_starts_t2_ax2.at(1),
      out_starts_t2_ax2.at(1) + out_extents_t2_ax2.at(1));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_2_ax3(
      out_starts_t2_ax3.at(1),
      out_starts_t2_ax3.at(1) + out_extents_t2_ax3.at(1));

  // Topo 2 ax = {0, 1, 2}
  auto sub_gu_hat_2_ax012 = Kokkos::subview(
      gu_hat_ax012, Kokkos::ALL, range_gu_hat_2_ax2, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax012, sub_gu_hat_2_ax012);

  // Topo 2 ax = {0, 2, 1}
  auto sub_gu_hat_2_ax021 = Kokkos::subview(
      gu_hat_ax021, Kokkos::ALL, range_gu_hat_2_ax1, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax021, sub_gu_hat_2_ax021);

  // Topo 2 ax = {1, 0, 2}
  auto sub_gu_hat_2_ax102 = Kokkos::subview(
      gu_hat_ax102, Kokkos::ALL, range_gu_hat_2_ax2, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax102, sub_gu_hat_2_ax102);

  // Topo 2 ax = {1, 2, 0}
  auto sub_gu_hat_2_ax120 = Kokkos::subview(
      gu_hat_ax120, Kokkos::ALL, range_gu_hat_2_ax0, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax120, sub_gu_hat_2_ax120);

  // Topo 2 ax = {2, 0, 1}
  auto sub_gu_hat_2_ax201 = Kokkos::subview(
      gu_hat_ax201, Kokkos::ALL, range_gu_hat_2_ax1, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax201, sub_gu_hat_2_ax201);

  // Topo 2 ax = {2, 1, 0}
  auto sub_gu_hat_2_ax210 = Kokkos::subview(
      gu_hat_ax210, Kokkos::ALL, range_gu_hat_2_ax0, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax210, sub_gu_hat_2_ax210);

  // Topo 2 ax = {1, 2, 3}
  auto sub_gu_hat_2_ax123 = Kokkos::subview(
      gu_hat_ax123, Kokkos::ALL, range_gu_hat_2_ax3, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax123, sub_gu_hat_2_ax123);

  // Topo 2 ax = {1, 3, 2}
  auto sub_gu_hat_2_ax132 = Kokkos::subview(
      gu_hat_ax132, Kokkos::ALL, range_gu_hat_2_ax2, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax132, sub_gu_hat_2_ax132);

  // Topo 2 ax = {2, 1, 3}
  auto sub_gu_hat_2_ax213 = Kokkos::subview(
      gu_hat_ax213, Kokkos::ALL, range_gu_hat_2_ax3, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax213, sub_gu_hat_2_ax213);

  // Define ranges for topology 3 (YZW-slab)
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_3_ax0(
      out_starts_t3_ax0.at(0),
      out_starts_t3_ax0.at(0) + out_extents_t3_ax0.at(0));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_3_ax1(
      out_starts_t3_ax1.at(0),
      out_starts_t3_ax1.at(0) + out_extents_t3_ax1.at(0));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_3_ax2(
      out_starts_t3_ax2.at(0),
      out_starts_t3_ax2.at(0) + out_extents_t3_ax2.at(0));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_3_ax3(
      out_starts_t3_ax3.at(0),
      out_starts_t3_ax3.at(0) + out_extents_t3_ax3.at(0));

  // Topo 3 ax = {0, 1, 2}
  auto sub_gu_hat_3_ax012 = Kokkos::subview(
      gu_hat_ax012, range_gu_hat_3_ax2, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_3_ax012, sub_gu_hat_3_ax012);

  // Topo 3 ax = {0, 2, 1}
  auto sub_gu_hat_3_ax021 = Kokkos::subview(
      gu_hat_ax021, range_gu_hat_3_ax1, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_3_ax021, sub_gu_hat_3_ax021);

  // Topo 3 ax = {1, 0, 2}
  auto sub_gu_hat_3_ax102 = Kokkos::subview(
      gu_hat_ax102, range_gu_hat_3_ax2, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_3_ax102, sub_gu_hat_3_ax102);

  // Topo 3 ax = {1, 2, 0}
  auto sub_gu_hat_3_ax120 = Kokkos::subview(
      gu_hat_ax120, range_gu_hat_3_ax0, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_3_ax120, sub_gu_hat_3_ax120);

  // Topo 3 ax = {2, 0, 1}
  auto sub_gu_hat_3_ax201 = Kokkos::subview(
      gu_hat_ax201, range_gu_hat_3_ax1, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_3_ax201, sub_gu_hat_3_ax201);

  // Topo 3 ax = {2, 1, 0}
  auto sub_gu_hat_3_ax210 = Kokkos::subview(
      gu_hat_ax210, range_gu_hat_3_ax0, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_3_ax210, sub_gu_hat_3_ax210);

  // Topo 3 ax = {1, 2, 3}
  auto sub_gu_hat_3_ax123 = Kokkos::subview(
      gu_hat_ax123, range_gu_hat_3_ax3, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_3_ax123, sub_gu_hat_3_ax123);

  // Topo 3 ax = {1, 3, 2}
  auto sub_gu_hat_3_ax132 = Kokkos::subview(
      gu_hat_ax132, range_gu_hat_3_ax2, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_3_ax132, sub_gu_hat_3_ax132);

  // Topo 3 ax = {2, 1, 3}
  auto sub_gu_hat_3_ax213 = Kokkos::subview(
      gu_hat_ax213, range_gu_hat_3_ax3, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_3_ax213, sub_gu_hat_3_ax213);

  using SlabPlanType =
      SlabPlan<execution_space, RealView4DType, ComplexView4DType, 3>;

  // Not a slab geometry
  if (nprocs == 1) {
    // topology1 -> topology2 with ax = {0, 1, 2}
    // (n0, n1/p, n2, n3) -> (n0, n1, (n2/2+1)/p, n3)
    ASSERT_THROW(
        {
          SlabPlanType plan_1_2_ax012(exec, u_1, u_hat_2_ax012, ax012,
                                      topology1, topology2, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology0 -> topology0 with ax = {0, 1, 2}
    // (n0, n1, n2, n3/p) -> (n0, n1, n2/2+1, n3/p)
    ASSERT_THROW(
        {
          SlabPlanType plan_0_0_ax012(exec, u_0, u_hat_0_ax012, ax012,
                                      topology0, topology0, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology1 -> topology1 with ax = {2, 1, 0}
    // (n0, n1/p, n2, n3) -> (n0/2+1, n1/p, n2, n3)
    ASSERT_THROW(
        {
          SlabPlanType plan_1_1_ax210(exec, u_1, u_hat_1_ax210, ax210,
                                      topology1, topology1, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology2 -> topology2 with ax = {0, 1, 2}
    // (n0, n1/p, n2, n3) -> (n0, n1/p, n2/2+1, n3)
    ASSERT_THROW(
        {
          SlabPlanType plan_2_2_ax012(exec, u_2, u_hat_2_ax012, ax012,
                                      topology2, topology2, MPI_COMM_WORLD);
        },
        std::runtime_error);
  } else {
    // topo 0 -> topo 0 with ax = {0, 1, 2}:
    // (n0, n1, n2, n3/p) -> (n0, n1, n2/2+1, n3/p)
    // FFT3 ax = {0, 1, 2}
    SlabPlanType plan_0_0_ax012(exec, u_0, u_hat_0_ax012, ax012, topology0,
                                topology0, MPI_COMM_WORLD);
    plan_0_0_ax012.forward(u_0, u_hat_0_ax012);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax012, ref_u_hat_0_ax012));

    plan_0_0_ax012.backward(u_hat_0_ax012, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 0 with ax = {0, 2, 1}:
    // (n0, n1, n2, n3/p) -> (n0, n1/2+1, n2, n3/p)
    // FFT3 ax = {0, 2, 1}
    SlabPlanType plan_0_0_ax021(exec, u_0, u_hat_0_ax021, ax021, topology0,
                                topology0, MPI_COMM_WORLD);
    plan_0_0_ax021.forward(u_0, u_hat_0_ax021);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax021, ref_u_hat_0_ax021));

    plan_0_0_ax021.backward(u_hat_0_ax021, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 0 with ax = {1, 0, 2}:
    // (n0, n1, n2, n3/p) -> (n0, n1, n2/2+1, n3/p)
    // FFT3 ax = {1, 0, 2}
    SlabPlanType plan_0_0_ax102(exec, u_0, u_hat_0_ax102, ax102, topology0,
                                topology0, MPI_COMM_WORLD);
    plan_0_0_ax102.forward(u_0, u_hat_0_ax102);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax102, ref_u_hat_0_ax102));

    plan_0_0_ax102.backward(u_hat_0_ax102, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 0 with ax = {1, 2, 0}:
    // (n0, n1, n2, n3/p) -> (n0/2+1, n1, n2, n3/p)
    // FFT3 ax = {1, 2, 0}
    SlabPlanType plan_0_0_ax120(exec, u_0, u_hat_0_ax120, ax120, topology0,
                                topology0, MPI_COMM_WORLD);
    plan_0_0_ax120.forward(u_0, u_hat_0_ax120);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax120, ref_u_hat_0_ax120));

    plan_0_0_ax120.backward(u_hat_0_ax120, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 0 with ax = {2, 0, 1}:
    // (n0, n1, n2, n3/p) -> (n0, n1/2+1, n2, n3/p)
    // FFT3 ax = {2, 0, 1}
    SlabPlanType plan_0_0_ax201(exec, u_0, u_hat_0_ax201, ax201, topology0,
                                topology0, MPI_COMM_WORLD);
    plan_0_0_ax201.forward(u_0, u_hat_0_ax201);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax201, ref_u_hat_0_ax201));

    plan_0_0_ax201.backward(u_hat_0_ax201, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 0 with ax = {2, 1, 0}:
    // (n0, n1, n2, n3/p) -> (n0/2+1, n1, n2, n3/p)
    // FFT3 ax = {2, 1, 0}
    SlabPlanType plan_0_0_ax210(exec, u_0, u_hat_0_ax210, ax210, topology0,
                                topology0, MPI_COMM_WORLD);
    plan_0_0_ax210.forward(u_0, u_hat_0_ax210);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax210, ref_u_hat_0_ax210));

    plan_0_0_ax210.backward(u_hat_0_ax210, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 0 with ax = {1, 2, 3}:
    // (n0, n1, n2, n3/p) -> (n0, n1/p, n2, n3) -> (n0, n1/p, n2, n3/2+1)
    // -> (n0, n1, n2, (n3/2+1)/p)
    // Transpose topo 2 -> FFT2 ax = {2, 3} -> Transpose topo 0
    // FFT ax = {1}
    SlabPlanType plan_0_0_ax123(exec, u_0, u_hat_0_ax123, ax123, topology0,
                                topology0, MPI_COMM_WORLD);
    plan_0_0_ax123.forward(u_0, u_hat_0_ax123);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax123, ref_u_hat_0_ax123));

    plan_0_0_ax123.backward(u_hat_0_ax123, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 0 with ax = {1, 3, 2}:
    // (n0, n1, n2, n3/p) -> (n0, n1, n2/2+1, n3/p) -> (n0, n1, (n2/2+1)/p, n3)
    // -> (n0, n1, n2/2+1, n3/p)
    // FFT ax = {2} -> Transpose topo 1 -> FFT2 ax = {1, 3} -> Transpose topo 0
    SlabPlanType plan_0_0_ax132(exec, u_0, u_hat_0_ax132, ax132, topology0,
                                topology0, MPI_COMM_WORLD);
    plan_0_0_ax132.forward(u_0, u_hat_0_ax132);
    EXPECT_TRUE(
        allclose(exec, u_hat_0_ax132, ref_u_hat_0_ax132, 1.0e-5, 1.0e-6));

    plan_0_0_ax132.backward(u_hat_0_ax132, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 0 with ax = {2, 1, 3}:
    // (n0, n1, n2, n3/p) -> (n0/p, n1, n2, n3) -> (n0/p, n1, n2, n3/2+1)
    // -> (n0, n1, n2, (n3/2+1)/p)
    // Transpose topo 3 -> FFT3 ax = {2, 1, 3} -> Transpose topo 0
    SlabPlanType plan_0_0_ax213(exec, u_0, u_hat_0_ax213, ax213, topology0,
                                topology0, MPI_COMM_WORLD);
    plan_0_0_ax213.forward(u_0, u_hat_0_ax213);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax213, ref_u_hat_0_ax213));

    plan_0_0_ax213.backward(u_hat_0_ax213, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 1 with ax = {1, 2, 3}:
    // (n0, n1/p, n2, n3) -> (n0, n1/p, n2, n3/2+1) -> (n0, n1, n2/p, n3/2+1)
    // FFT2 ax = {2, 3} -> Transpose -> FFT ax = {1}
    SlabPlanType plan_2_1_ax123(exec, u_2, u_hat_1_ax123, ax123, topology2,
                                topology1, MPI_COMM_WORLD);
    plan_2_1_ax123.forward(u_2, u_hat_1_ax123);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax123, ref_u_hat_1_ax123));

    plan_2_1_ax123.backward(u_hat_1_ax123, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));
  }
}

}  // namespace

TYPED_TEST_SUITE(TestSlab1D, test_types);
TYPED_TEST_SUITE(TestSlab2D, test_types);
TYPED_TEST_SUITE(TestSlab3D, test_types);

/*
TYPED_TEST(TestSlab1D, View2D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_slab1D_view2D<float_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestSlab1D, View3D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_slab1D_view3D<float_type, layout_type>(this->m_nprocs);
}
*/

TYPED_TEST(TestSlab2D, View2D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_slab2D_view2D<float_type, layout_type>(this->m_nprocs);
}

/*
TYPED_TEST(TestSlab2D, View3D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_slab2D_view3D<float_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestSlab3D, View3D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_slab3D_view3D<float_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestSlab3D, View4D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_slab3D_view4D<float_type, layout_type>(this->m_nprocs);
}
*/
