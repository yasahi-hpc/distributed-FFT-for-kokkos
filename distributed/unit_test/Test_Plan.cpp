#include <mpi.h>
#include <gtest/gtest.h>
#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include "KokkosFFT_Distributed_Plan.hpp"
#include "Test_Utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using test_types      = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>,
                                    std::pair<double, Kokkos::LayoutLeft>,
                                    std::pair<double, Kokkos::LayoutRight>>;

//  Basically the same fixtures, used for labeling tests
template <typename T>
struct TestPlan1D : public ::testing::Test {
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
void test_plan1D_view2D(std::size_t nprocs) {
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
      KokkosFFT::Distributed::compute_local_extents(global_in_extents,
                                                    topology0, MPI_COMM_WORLD);
  auto [in_extents_t1, in_starts_t1] =
      KokkosFFT::Distributed::compute_local_extents(global_in_extents,
                                                    topology1, MPI_COMM_WORLD);
  auto [out_extents_t0_ax0, out_starts_t0_ax0] =
      KokkosFFT::Distributed::compute_local_extents(global_out_extents_ax0,
                                                    topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax0, out_starts_t1_ax0] =
      KokkosFFT::Distributed::compute_local_extents(global_out_extents_ax0,
                                                    topology1, MPI_COMM_WORLD);
  auto [out_extents_t0_ax1, out_starts_t0_ax1] =
      KokkosFFT::Distributed::compute_local_extents(global_out_extents_ax1,
                                                    topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax1, out_starts_t1_ax1] =
      KokkosFFT::Distributed::compute_local_extents(global_out_extents_ax1,
                                                    topology1, MPI_COMM_WORLD);

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

  // topo0 -> topo0 with ax = {1}:
  // (n0, n1/p) -> (n0/p, n1) -> ((n0/2+1)/p, n1)
  // Transpose + FFT ax = {1} + Transpose
  KokkosFFT::Distributed::Plan plan_0_0_ax1(exec, u_0, u_hat_0_ax1,
                                            axes_type{1}, topology0, topology0,
                                            MPI_COMM_WORLD);
  execute(plan_0_0_ax1, u_0, u_hat_0_ax1, KokkosFFT::Direction::forward);
  EXPECT_TRUE(allclose(exec, u_hat_0_ax1, ref_u_hat_0_ax1));

  execute(plan_0_0_ax1, u_hat_0_ax1, u_inv_0, KokkosFFT::Direction::backward);
  EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0));

#if !defined(PRIORITIZE_TPL_PLAN_IF_AVAILABLE)
  std::string ref_name = nprocs == 1 ? "SharedPlan" : "SlabPlan";
  EXPECT_EQ(plan_0_0_ax1.name(), ref_name);
#endif

  // topo 0 -> topo 1 with ax = {0}:
  // (n0, n1/p) -> (n0/2+1, n1/p) -> ((n0 / 2 + 1) / p, n1)
  // FFT ax = {0} + Transpose
  KokkosFFT::Distributed::Plan plan_0_1_ax0(exec, u_0, u_hat_1_ax0,
                                            axes_type{0}, topology0, topology1,
                                            MPI_COMM_WORLD);
  execute(plan_0_1_ax0, u_0, u_hat_1_ax0, KokkosFFT::Direction::forward);
  EXPECT_TRUE(allclose(exec, u_hat_1_ax0, ref_u_hat_1_ax0));

  execute(plan_0_1_ax0, u_hat_1_ax0, u_inv_0, KokkosFFT::Direction::backward);
  EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0));

#if !defined(PRIORITIZE_TPL_PLAN_IF_AVAILABLE)
  EXPECT_EQ(plan_0_1_ax0.name(), ref_name);
#endif

  // topology0 (n0, n1/p), ax=1 -> topology1 (n0/p, n1/2+1)
  // topo0 -> topo1 with ax = {1}: (n0, n1/p) -> (n0/p, n1) -> (n0/p, n1/2+1)
  // Transpose + FFT ax = {1}
  KokkosFFT::Distributed::Plan plan_0_1_ax1(exec, u_0, u_hat_1_ax1,
                                            axes_type{1}, topology0, topology1,
                                            MPI_COMM_WORLD);
  execute(plan_0_1_ax1, u_0, u_hat_1_ax1, KokkosFFT::Direction::forward);
  EXPECT_TRUE(allclose(exec, u_hat_1_ax1, ref_u_hat_1_ax1));

  execute(plan_0_1_ax1, u_hat_1_ax1, u_inv_0, KokkosFFT::Direction::backward);
  EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0));

#if !defined(PRIORITIZE_TPL_PLAN_IF_AVAILABLE)
  EXPECT_EQ(plan_0_1_ax1.name(), ref_name);
#endif

  // topo1 -> topo0 with ax = {0}: (n0/p, n1) -> (n0/2+1, n1/p)
  // Transpose + FFT ax = {0}
  KokkosFFT::Distributed::Plan plan_1_0_ax0(exec, u_1, u_hat_0_ax0,
                                            axes_type{0}, topology1, topology0,
                                            MPI_COMM_WORLD);
  execute(plan_1_0_ax0, u_1, u_hat_0_ax0, KokkosFFT::Direction::forward);
  EXPECT_TRUE(allclose(exec, u_hat_0_ax0, ref_u_hat_0_ax0));

  execute(plan_1_0_ax0, u_hat_0_ax0, u_inv_1, KokkosFFT::Direction::backward);
  EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1));

#if !defined(PRIORITIZE_TPL_PLAN_IF_AVAILABLE)
  EXPECT_EQ(plan_1_0_ax0.name(), ref_name);
#endif

  // topo1 -> topo0 with ax = {1}: (n0/p, n1) -> (n0, n1/p)
  // FFT ax = {1} -> Transpose
  KokkosFFT::Distributed::Plan plan_1_0_ax1(exec, u_1, u_hat_0_ax1,
                                            axes_type{1}, topology1, topology0,
                                            MPI_COMM_WORLD);
  execute(plan_1_0_ax1, u_1, u_hat_0_ax1, KokkosFFT::Direction::forward);
  EXPECT_TRUE(allclose(exec, u_hat_0_ax1, ref_u_hat_0_ax1));

  execute(plan_1_0_ax1, u_hat_0_ax1, u_inv_1, KokkosFFT::Direction::backward);
  EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1));

#if !defined(PRIORITIZE_TPL_PLAN_IF_AVAILABLE)
  EXPECT_EQ(plan_1_0_ax1.name(), ref_name);
#endif

  // topo1 -> topo1 with ax = {0}: (n0/p, n1) -> (n0, n1/p) -> (n0/2+1, n1/p)
  // Transpose + FFT ax = {0} + Transpose
  KokkosFFT::Distributed::Plan plan_1_1_ax0(exec, u_1, u_hat_1_ax0,
                                            axes_type{0}, topology1, topology1,
                                            MPI_COMM_WORLD);
  execute(plan_1_1_ax0, u_1, u_hat_1_ax0, KokkosFFT::Direction::forward);
  EXPECT_TRUE(allclose(exec, u_hat_1_ax0, ref_u_hat_1_ax0));

  execute(plan_1_1_ax0, u_hat_1_ax0, u_inv_1, KokkosFFT::Direction::backward);
  EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1));

#if !defined(PRIORITIZE_TPL_PLAN_IF_AVAILABLE)
  EXPECT_EQ(plan_1_1_ax0.name(), ref_name);
#endif
}

}  // namespace

TYPED_TEST_SUITE(TestPlan1D, test_types);

TYPED_TEST(TestPlan1D, View2D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_plan1D_view2D<float_type, layout_type>(this->m_nprocs);
}
