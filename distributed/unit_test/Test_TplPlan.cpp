#include <mpi.h>
#include <gtest/gtest.h>
#include <iostream>
#include <utility>
#include <cmath>
#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>
#include <Kokkos_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include "KokkosFFT_Distributed_TplPlan.hpp"
#include "KokkosFFT_Distributed_Plan.hpp"
#include "KokkosFFT_Distributed_Helper.hpp"
#include "KokkosFFT_Distributed_ScaledCopy.hpp"
#include "Test_Utils.hpp"
#include "Test_PlanUtils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using test_types      = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>,
                                    std::pair<double, Kokkos::LayoutLeft>,
                                    std::pair<double, Kokkos::LayoutRight>>;

// Whether a TplPlan constructed directly works on a single process. cuFFTMp
// throws (its plans need a slab or a pencil decomposition); rocFFT makes its
// plans from bricks, which also describe a single process. is_tpl_available
// rejects a single process for both, so plan_factory uses the internal plans.
#if defined(ENABLE_TPL_ROCFFT_MPI)
constexpr bool is_tpl_plan_on_one_process = true;
#else
constexpr bool is_tpl_plan_on_one_process = false;
#endif

// TplPlan is constructible only from its arguments: not default
// constructible, not copyable, not movable, like the other internal plans.
// Plan holds it through std::unique_ptr, so moving a Plan does not move it.
using TplPlanType = KokkosFFT::Distributed::Impl::TplPlan<
    execution_space, Kokkos::View<double***, execution_space>,
    Kokkos::View<Kokkos::complex<double>***, execution_space>, 3>;
static_assert(!std::is_default_constructible_v<TplPlanType>);
static_assert(!std::is_copy_constructible_v<TplPlanType>);
static_assert(!std::is_copy_assignable_v<TplPlanType>);
static_assert(!std::is_move_constructible_v<TplPlanType>);
static_assert(!std::is_move_assignable_v<TplPlanType>);

//  Basically the same fixtures, used for labeling tests
template <typename T>
struct TestTplPlan1D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;

  int m_rank   = 0;
  int m_nprocs = 1;

  virtual void SetUp() {
    ::MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &m_nprocs);

    if (this->m_nprocs > 4) {
      GTEST_SKIP() << "The number of MPI processes should be smaller or equal "
                      "to 4 for this test";
    }
  }
};

template <typename T>
struct TestTplPlan2D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;

  int m_rank   = 0;
  int m_nprocs = 1;

  virtual void SetUp() {
    ::MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &m_nprocs);

    if (this->m_nprocs > 4) {
      GTEST_SKIP() << "The number of MPI processes should be smaller or equal "
                      "to 4 for this test";
    }
  }
};

template <typename T>
struct TestTplPlan3D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;

  int m_rank   = 0;
  int m_nprocs = 1;
  int m_npx    = 1;

  virtual void SetUp() {
    ::MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &m_nprocs);

    m_npx = std::sqrt(m_nprocs);
    if (this->m_nprocs > 4) {
      GTEST_SKIP() << "The number of MPI processes should be smaller or equal "
                      "to 4 for this test";
    }
  }
};

/// \brief Test if the plan is available for 1D transforms
/// \tparam T Type of the data (float or double)
/// \tparam LayoutType Layout of the data (LayoutLeft or LayoutRight)
///
/// \param[in] nprocs Number of processes in the MPI communicator
template <typename T, typename LayoutType>
void test_tpl1D_is_available(std::size_t nprocs) {
  using RealView2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;
  using axes_type     = KokkosFFT::axis_type<1>;
  using extents_type  = std::array<std::size_t, 2>;
  using topology_type = std::array<std::size_t, 2>;

  topology_type topology0{1, nprocs}, topology1{nprocs, 1};
  axes_type ax0 = {0}, ax1 = {1};

  const std::size_t n0 = 8, n1 = 7;
  extents_type global_in_extents{n0, n1},
      global_out_extents_ax0{n0 / 2 + 1, n1},
      global_out_extents_ax1{n0, n1 / 2 + 1};

  auto [in_extents_t0, in_starts_t0] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_in_extents, topology0, MPI_COMM_WORLD);
  auto [in_extents_t1, in_starts_t1] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_in_extents, topology1, MPI_COMM_WORLD);
  auto [out_extents_t0_ax0, out_starts_t0_ax0] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax0, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax0, out_starts_t1_ax0] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax0, topology1, MPI_COMM_WORLD);
  auto [out_extents_t0_ax1, out_starts_t0_ax1] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax1, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax1, out_starts_t1_ax1] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax1, topology1, MPI_COMM_WORLD);

  // Data in Topology 0 (X-slab)
  RealView2DType u_0("u_0",
                     KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t0));
  ComplexView2DType u_hat_0_ax0(
      "u_hat_0_ax0",
      KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax0)),
      u_hat_0_ax1("u_hat_0_ax1", KokkosFFT::Impl::create_layout<LayoutType>(
                                     out_extents_t0_ax1));

  // Data in Topology 1 (Y-slab)
  RealView2DType u_1("u_1",
                     KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t1));
  ComplexView2DType u_hat_1_ax0(
      "u_hat_1_ax0",
      KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax0)),
      u_hat_1_ax1("u_hat_1_ax1", KokkosFFT::Impl::create_layout<LayoutType>(
                                     out_extents_t1_ax1));

  // Check availability
  // For 1D transform, this is not supported by tpls
  execution_space exec;

  bool is_available_0_0_ax0 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_0, u_hat_0_ax0, ax0, topology0, topology0);
  bool is_available_0_0_ax1 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_0, u_hat_0_ax1, ax1, topology0, topology0);
  bool is_available_1_0_ax0 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_1, u_hat_1_ax0, ax0, topology1, topology1);
  bool is_available_1_0_ax1 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_1, u_hat_1_ax1, ax1, topology1, topology1);

  EXPECT_FALSE(is_available_0_0_ax0);
  EXPECT_FALSE(is_available_0_0_ax1);
  EXPECT_FALSE(is_available_1_0_ax0);
  EXPECT_FALSE(is_available_1_0_ax1);
}

/// \brief Test if the plan is available for 2D transforms on 2D View
/// \tparam T Type of the data (float or double)
/// \tparam LayoutType Layout of the data (LayoutLeft or LayoutRight)
///
/// \param[in] nprocs Number of processes in the MPI communicator
template <typename T, typename LayoutType>
void test_tpl2D_is_available_View2D(std::size_t nprocs) {
  using RealView2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;
  using axes_type     = KokkosFFT::axis_type<2>;
  using extents_type  = std::array<std::size_t, 2>;
  using topology_type = std::array<std::size_t, 2>;

  // 2D Topologies
  topology_type topology0{1, nprocs}, topology1{nprocs, 1};
  axes_type ax01 = {0, 1}, ax10 = {1, 0};

  const std::size_t n0 = 8, n1 = 7;
  extents_type global_in_extents{n0, n1},
      global_out_extents_ax0{n0 / 2 + 1, n1},
      global_out_extents_ax1{n0, n1 / 2 + 1};

  auto [in_extents_t0, in_starts_t0] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_in_extents, topology0, MPI_COMM_WORLD);
  auto [in_extents_t1, in_starts_t1] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_in_extents, topology1, MPI_COMM_WORLD);
  auto [out_extents_t0_ax0, out_starts_t0_ax0] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax0, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax0, out_starts_t1_ax0] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax0, topology1, MPI_COMM_WORLD);
  auto [out_extents_t0_ax1, out_starts_t0_ax1] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax1, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax1, out_starts_t1_ax1] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax1, topology1, MPI_COMM_WORLD);

  // Data in Topology 0 (X-slab)
  RealView2DType u_0("u_0",
                     KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t0));
  ComplexView2DType u_hat_0_ax0(
      "u_hat_0_ax0",
      KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t0_ax0)),
      u_hat_0_ax1("u_hat_0_ax1", KokkosFFT::Impl::create_layout<LayoutType>(
                                     out_extents_t0_ax1));

  // Data in Topology 1 (Y-slab)
  RealView2DType u_1("u_1",
                     KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t1));
  ComplexView2DType u_hat_1_ax0(
      "u_hat_1_ax0",
      KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t1_ax0)),
      u_hat_1_ax1("u_hat_1_ax1", KokkosFFT::Impl::create_layout<LayoutType>(
                                     out_extents_t1_ax1));

  // Check availability
  // These 2D transforms on 2D Views are fully supported
  execution_space exec;
  bool is_available_0_0_ax01 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_0, u_hat_0_ax0, ax01, topology0, topology0);
  bool is_available_0_0_ax10 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_0, u_hat_0_ax1, ax10, topology0, topology0);
  bool is_available_1_0_ax01 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_1, u_hat_0_ax0, ax01, topology1, topology0);
  bool is_available_1_0_ax10 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_1, u_hat_0_ax1, ax10, topology1, topology0);

  // Not a slab geometry
  if (nprocs == 1) {
    EXPECT_FALSE(is_available_0_0_ax01);
    EXPECT_FALSE(is_available_0_0_ax10);
    EXPECT_FALSE(is_available_1_0_ax01);
    EXPECT_FALSE(is_available_1_0_ax10);
  } else {
    // The same distribution in and out (cuFFTMp: identical boxes; rocFFT:
    // identical fields), and any order of the axes (the copies of cuFFTMp,
    // the buffers of rocFFT, transpose)
    EXPECT_TRUE(is_available_0_0_ax01);
    EXPECT_TRUE(is_available_0_0_ax10);
    EXPECT_TRUE(is_available_1_0_ax01);
    EXPECT_TRUE(is_available_1_0_ax10);
  }
}

/// \brief Test if the plan is available for 2D transforms on 3D View
/// \tparam T Type of the data (float or double)
/// \tparam LayoutType Layout of the data (LayoutLeft or LayoutRight)
///
/// \param[in] nprocs Number of processes in the MPI communicator
template <typename T, typename LayoutType>
void test_tpl2D_is_available_View3D(std::size_t nprocs) {
  using RealView3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<T>***, LayoutType, execution_space>;
  using axes_type     = KokkosFFT::axis_type<2>;
  using extents_type  = std::array<std::size_t, 3>;
  using topology_type = std::array<std::size_t, 3>;

  // 3D Topologies
  topology_type topology0{1, 1, nprocs}, topology1{1, nprocs, 1},
      topology2{nprocs, 1, 1};

  axes_type ax01 = {0, 1}, ax02 = {0, 2}, ax10 = {1, 0}, ax12 = {1, 2},
            ax20 = {2, 0}, ax21 = {2, 1};

  const std::size_t n0 = 8, n1 = 7, n2 = 5;
  extents_type global_in_extents{n0, n1, n2},
      global_out_extents_ax0{n0 / 2 + 1, n1, n2},
      global_out_extents_ax1{n0, n1 / 2 + 1, n2},
      global_out_extents_ax2{n0, n1, n2 / 2 + 1};

  auto [in_extents_t0, in_starts_t0] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_in_extents, topology0, MPI_COMM_WORLD);
  auto [in_extents_t1, in_starts_t1] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_in_extents, topology1, MPI_COMM_WORLD);
  auto [in_extents_t2, in_starts_t2] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_in_extents, topology2, MPI_COMM_WORLD);
  auto [out_extents_t0_ax0, out_starts_t0_ax0] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax0, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax0, out_starts_t1_ax0] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax0, topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax0, out_starts_t2_ax0] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax0, topology2, MPI_COMM_WORLD);
  auto [out_extents_t0_ax1, out_starts_t0_ax1] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax1, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax1, out_starts_t1_ax1] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax1, topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax1, out_starts_t2_ax1] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax1, topology2, MPI_COMM_WORLD);
  auto [out_extents_t0_ax2, out_starts_t0_ax2] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax2, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax2, out_starts_t1_ax2] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax2, topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax2, out_starts_t2_ax2] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax2, topology2, MPI_COMM_WORLD);

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
                                       out_extents_t0_ax1));

  // Data in Topology 1 (XZ-slab)
  RealView3DType u_1("u_1",
                     KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t1)),
      u_inv_1("u_inv_1",
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
                                       out_extents_t1_ax1));

  // Data in Topology 2 (YZ-slab)
  RealView3DType u_2("u_2",
                     KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t2)),
      u_inv_2("u_inv_2",
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
                                       out_extents_t2_ax1));

  // Check availability
  // These 2D transforms on 3D+ Views are not supported
  execution_space exec;

  bool is_available_0_0_ax01 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_0, u_hat_0_ax01, ax01, topology0, topology0);
  bool is_available_0_0_ax02 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_0, u_hat_0_ax02, ax02, topology0, topology0);
  bool is_available_0_0_ax10 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_0, u_hat_0_ax10, ax10, topology0, topology0);
  bool is_available_0_0_ax12 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_0, u_hat_0_ax12, ax12, topology0, topology0);
  bool is_available_0_0_ax20 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_0, u_hat_0_ax20, ax20, topology0, topology0);
  bool is_available_0_0_ax21 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_0, u_hat_0_ax21, ax21, topology0, topology0);

  bool is_available_0_1_ax01 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_0, u_hat_0_ax01, ax01, topology0, topology1);
  bool is_available_0_1_ax02 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_0, u_hat_0_ax02, ax02, topology0, topology1);
  bool is_available_0_1_ax10 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_0, u_hat_0_ax10, ax10, topology0, topology1);
  bool is_available_0_1_ax12 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_0, u_hat_0_ax12, ax12, topology0, topology1);
  bool is_available_0_1_ax20 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_0, u_hat_0_ax20, ax20, topology0, topology1);
  bool is_available_0_1_ax21 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_0, u_hat_0_ax21, ax21, topology0, topology1);

  // Batched plans are not supported
  EXPECT_FALSE(is_available_0_0_ax01);
  EXPECT_FALSE(is_available_0_0_ax02);
  EXPECT_FALSE(is_available_0_0_ax10);
  EXPECT_FALSE(is_available_0_0_ax12);
  EXPECT_FALSE(is_available_0_0_ax20);
  EXPECT_FALSE(is_available_0_0_ax21);

  EXPECT_FALSE(is_available_0_1_ax01);
  EXPECT_FALSE(is_available_0_1_ax02);
  EXPECT_FALSE(is_available_0_1_ax10);
  EXPECT_FALSE(is_available_0_1_ax12);
  EXPECT_FALSE(is_available_0_1_ax20);
  EXPECT_FALSE(is_available_0_1_ax21);
}

/// \brief Test if the plan is available for 3D transforms on 3D View
/// \tparam T Type of the data (float or double)
/// \tparam LayoutType Layout of the data (LayoutLeft or LayoutRight)
///
/// \param[in] nprocs Number of processes in the MPI communicator
template <typename T, typename LayoutType>
void test_tpl3D_is_available_View3D(std::size_t nprocs) {
  using RealView3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<T>***, LayoutType, execution_space>;
  using axes_type     = KokkosFFT::axis_type<3>;
  using extents_type  = std::array<std::size_t, 3>;
  using topology_type = std::array<std::size_t, 3>;

  // 3D Topologies
  topology_type topology0{1, 1, nprocs}, topology1{1, nprocs, 1},
      topology2{nprocs, 1, 1};

  axes_type ax012{0, 1, 2}, ax021{0, 2, 1}, ax102{1, 0, 2}, ax120{1, 2, 0},
      ax201{2, 0, 1}, ax210{2, 1, 0};

  const std::size_t n0 = 8, n1 = 7, n2 = 5;
  extents_type global_in_extents{n0, n1, n2},
      global_out_extents_ax0{n0 / 2 + 1, n1, n2},
      global_out_extents_ax1{n0, n1 / 2 + 1, n2},
      global_out_extents_ax2{n0, n1, n2 / 2 + 1};

  auto [in_extents_t0, in_starts_t0] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_in_extents, topology0, MPI_COMM_WORLD);
  auto [in_extents_t1, in_starts_t1] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_in_extents, topology1, MPI_COMM_WORLD);
  auto [in_extents_t2, in_starts_t2] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_in_extents, topology2, MPI_COMM_WORLD);
  auto [out_extents_t0_ax0, out_starts_t0_ax0] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax0, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax0, out_starts_t1_ax0] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax0, topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax0, out_starts_t2_ax0] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax0, topology2, MPI_COMM_WORLD);
  auto [out_extents_t0_ax1, out_starts_t0_ax1] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax1, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax1, out_starts_t1_ax1] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax1, topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax1, out_starts_t2_ax1] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax1, topology2, MPI_COMM_WORLD);
  auto [out_extents_t0_ax2, out_starts_t0_ax2] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax2, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax2, out_starts_t1_ax2] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax2, topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax2, out_starts_t2_ax2] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax2, topology2, MPI_COMM_WORLD);

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
                                         out_extents_t0_ax0));

  // Data in Topology 1 (XZ-slab)
  RealView3DType u_1("u_1",
                     KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t1)),
      u_inv_1("u_inv_1",
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
                                         out_extents_t1_ax0));

  // Data in Topology 2 (YZ-slab)
  RealView3DType u_2("u_2",
                     KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t2)),
      u_inv_2("u_inv_2",
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
                                         out_extents_t2_ax0));

  // Check availability
  // These 3D transforms on 3D+ Views are not supported
  execution_space exec;

  bool is_available_0_0_ax012 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_0, u_hat_0_ax012, ax012, topology0, topology0);
  bool is_available_0_0_ax021 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_0, u_hat_0_ax021, ax021, topology0, topology0);
  bool is_available_0_0_ax102 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_0, u_hat_0_ax102, ax102, topology0, topology0);
  bool is_available_0_0_ax120 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_0, u_hat_0_ax120, ax120, topology0, topology0);
  bool is_available_0_0_ax201 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_0, u_hat_0_ax201, ax201, topology0, topology0);
  bool is_available_0_0_ax210 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_0, u_hat_0_ax210, ax210, topology0, topology0);

  bool is_available_0_1_ax012 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_0, u_hat_1_ax012, ax012, topology0, topology1);
  bool is_available_0_1_ax021 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_0, u_hat_1_ax021, ax021, topology0, topology1);
  bool is_available_0_1_ax102 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_0, u_hat_1_ax102, ax102, topology0, topology1);
  bool is_available_0_1_ax120 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_0, u_hat_1_ax120, ax120, topology0, topology1);
  bool is_available_0_1_ax201 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_0, u_hat_1_ax201, ax201, topology0, topology1);
  bool is_available_0_1_ax210 = KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, u_0, u_hat_1_ax210, ax210, topology0, topology1);

  // The same distribution in and out (cuFFTMp: identical boxes; rocFFT:
  // identical fields), and any order of the axes (the copies of cuFFTMp, the
  // buffers of rocFFT, transpose), on slabs (more than one process)
  const bool is_slab = nprocs > 1;
  EXPECT_EQ(is_available_0_0_ax012, is_slab);
  EXPECT_EQ(is_available_0_0_ax021, is_slab);
  EXPECT_EQ(is_available_0_0_ax102, is_slab);
  EXPECT_EQ(is_available_0_0_ax120, is_slab);
  EXPECT_EQ(is_available_0_0_ax201, is_slab);
  EXPECT_EQ(is_available_0_0_ax210, is_slab);
  EXPECT_EQ(is_available_0_1_ax012, is_slab);
  EXPECT_EQ(is_available_0_1_ax021, is_slab);
  EXPECT_EQ(is_available_0_1_ax102, is_slab);
  EXPECT_EQ(is_available_0_1_ax120, is_slab);
  EXPECT_EQ(is_available_0_1_ax201, is_slab);
  EXPECT_EQ(is_available_0_1_ax210, is_slab);
}

/// \brief The topology of the reversed View: the dimensions reversed and the
/// layout flipped, so that every rank keeps its block (the local extents and
/// starts are the reversed ones)
template <typename LayoutType>
void test_reversed_topology(std::size_t nprocs) {
  using KokkosFFT::Distributed::Impl::reversed_extents;
  using KokkosFFT::Distributed::Impl::reversed_topology;
  using topology_type =
      KokkosFFT::Distributed::Topology<std::size_t, 3, LayoutType>;
  using extents_type = std::array<std::size_t, 3>;

  // Two distributed dimensions when possible, where the rank order matters
  int npx = 1;
  for (int p = static_cast<int>(std::sqrt(nprocs)); p >= 1; --p) {
    if (nprocs % p == 0) {
      npx = p;
      break;
    }
  }
  const std::size_t npy = nprocs / npx;
  const topology_type topology{1, static_cast<std::size_t>(npx), npy};
  const auto reversed = reversed_topology(topology);
  EXPECT_EQ(reversed.array(), reversed_extents(topology.array()));

  const extents_type global_extents{8, 7, 5};
  auto [extents, starts] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_extents, topology, MPI_COMM_WORLD);
  auto [reversed_local_extents, reversed_starts] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          reversed_extents(global_extents), reversed, MPI_COMM_WORLD);
  EXPECT_EQ(reversed_local_extents, reversed_extents(extents));
  EXPECT_EQ(reversed_starts, reversed_extents(starts));
}

/// \brief LayoutLeft Views: the plan runs on the reversed LayoutRight Views
/// (is_reversed) when the axes need no transpose there, and gives the same
/// result as LayoutRight Views holding the same data in the same index order.
/// From the x-slab to the y-slab.
/// \tparam T Value type of the input (float/double or complex)
/// \param[in] nprocs Number of processes in the MPI communicator
template <typename T>
void test_tpl3D_layout_left_View3D(std::size_t nprocs) {
  using float_type   = KokkosFFT::Impl::base_floating_point_type<T>;
  using complex_type = Kokkos::complex<float_type>;
  using InLeftViewType =
      Kokkos::View<T***, Kokkos::LayoutLeft, execution_space>;
  using OutLeftViewType =
      Kokkos::View<complex_type***, Kokkos::LayoutLeft, execution_space>;
  using InRightViewType =
      Kokkos::View<T***, Kokkos::LayoutRight, execution_space>;
  using OutRightViewType =
      Kokkos::View<complex_type***, Kokkos::LayoutRight, execution_space>;
  using axes_type     = KokkosFFT::axis_type<3>;
  using extents_type  = std::array<std::size_t, 3>;
  using topology_type = std::array<std::size_t, 3>;

  constexpr bool is_R2C = KokkosFFT::Impl::is_real_v<T>;
  const topology_type topology0{1, 1, nprocs}, topology1{1, nprocs, 1};
  const std::size_t n0 = 8, n1 = 7, n2 = 5;

  // {axes, whether the LayoutLeft plan runs reversed}
  const std::vector<std::pair<axes_type, bool>> cases =
      is_R2C ? std::vector<std::pair<axes_type, bool>>{{{0, 1, 2}, false},
                                                       {{1, 2, 0}, true},
                                                       {{2, 1, 0}, true}}
             : std::vector<std::pair<axes_type, bool>>{{{0, 1, 2}, true},
                                                       {{2, 1, 0}, true}};
  for (const auto& [axes, ref_is_reversed] : cases) {
    const std::size_t last = axes.back();
    extents_type global_in_extents{n0, n1, n2},
        global_out_extents      = global_in_extents;
    global_out_extents.at(last) = KokkosFFT::Impl::extent_after_transform(
        global_out_extents.at(last), is_R2C);

    auto [in_extents, in_starts] =
        KokkosFFT::Distributed::compute_local_extents_and_starts(
            global_in_extents, topology0, MPI_COMM_WORLD);
    auto [out_extents, out_starts] =
        KokkosFFT::Distributed::compute_local_extents_and_starts(
            global_out_extents, topology1, MPI_COMM_WORLD);

    InLeftViewType u_left(
        "u_left",
        KokkosFFT::Impl::create_layout<Kokkos::LayoutLeft>(in_extents)),
        u_left_ref(
            "u_left_ref",
            KokkosFFT::Impl::create_layout<Kokkos::LayoutLeft>(in_extents)),
        u_left_inv(
            "u_left_inv",
            KokkosFFT::Impl::create_layout<Kokkos::LayoutLeft>(in_extents));
    OutLeftViewType u_hat_left(
        "u_hat_left",
        KokkosFFT::Impl::create_layout<Kokkos::LayoutLeft>(out_extents));
    InRightViewType u_right(
        "u_right",
        KokkosFFT::Impl::create_layout<Kokkos::LayoutRight>(in_extents));
    OutRightViewType u_hat_right(
        "u_hat_right",
        KokkosFFT::Impl::create_layout<Kokkos::LayoutRight>(out_extents));

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
    Kokkos::fill_random(u_left_ref, random_pool, 1.0);
    Kokkos::deep_copy(u_left, u_left_ref);
    Kokkos::deep_copy(u_right, u_left_ref);

    execution_space exec;
    ASSERT_TRUE(KokkosFFT::Distributed::Impl::is_tpl_available(
        exec, u_left, u_hat_left, axes, topology0, topology1));

    KokkosFFT::Distributed::Impl::TplPlan plan_left(
        exec, u_left, u_hat_left, axes, topology0, topology1, MPI_COMM_WORLD);
    EXPECT_EQ(plan_left.is_reversed(), ref_is_reversed);
    plan_left.forward(u_left, u_hat_left);

    // The reference: plan_factory with LayoutRight Views (with cuFFTMp, the
    // vendor plan through the transposing copies instead of the reversed
    // View)
    auto plan_right = KokkosFFT::Distributed::Impl::plan_factory(
        exec, u_right, u_hat_right, axes,
        KokkosFFT::Distributed::Topology<std::size_t, 3>(topology0),
        KokkosFFT::Distributed::Topology<std::size_t, 3>(topology1),
        MPI_COMM_WORLD);
    plan_right->forward(u_right, u_hat_right);

    OutLeftViewType u_hat_right_as_left(
        "u_hat_right_as_left",
        KokkosFFT::Impl::create_layout<Kokkos::LayoutLeft>(out_extents));
    Kokkos::deep_copy(u_hat_right_as_left, u_hat_right);
    EXPECT_TRUE(
        allclose(exec, u_hat_left, u_hat_right_as_left, 1.0e-5, 1.0e-6));

    plan_left.backward(u_hat_left, u_left_inv);
    EXPECT_TRUE(allclose(exec, u_left_inv, u_left_ref, 1.0e-5, 1.0e-6));
  }
}

/// \brief The axes the vendor plans are made with: C2C in increasing order,
/// R2C the axes but the last in increasing order then the last; negative
/// axes made non-negative
void test_canonical_tpl_axes() {
  using KokkosFFT::Distributed::Impl::canonical_tpl_axes;
  using c2c_type   = Kokkos::complex<double>;
  using r2c_type   = double;
  using axes2_type = KokkosFFT::axis_type<2>;
  using axes3_type = KokkosFFT::axis_type<3>;

  EXPECT_EQ((canonical_tpl_axes<c2c_type>(axes2_type{1, 0}, 2)),
            (axes2_type{0, 1}));
  EXPECT_EQ((canonical_tpl_axes<r2c_type>(axes2_type{1, 0}, 2)),
            (axes2_type{1, 0}));
  EXPECT_EQ((canonical_tpl_axes<r2c_type>(axes2_type{0, 1}, 2)),
            (axes2_type{0, 1}));
  EXPECT_EQ((canonical_tpl_axes<c2c_type>(axes2_type{-1, -2}, 2)),
            (axes2_type{0, 1}));

  EXPECT_EQ((canonical_tpl_axes<c2c_type>(axes3_type{2, 0, 1}, 3)),
            (axes3_type{0, 1, 2}));
  EXPECT_EQ((canonical_tpl_axes<c2c_type>(axes3_type{1, 2, 0}, 3)),
            (axes3_type{0, 1, 2}));
  EXPECT_EQ((canonical_tpl_axes<r2c_type>(axes3_type{1, 0, 2}, 3)),
            (axes3_type{0, 1, 2}));
  EXPECT_EQ((canonical_tpl_axes<r2c_type>(axes3_type{2, 0, 1}, 3)),
            (axes3_type{0, 2, 1}));
  EXPECT_EQ((canonical_tpl_axes<r2c_type>(axes3_type{-1, 0, -2}, 3)),
            (axes3_type{0, 2, 1}));

  // Batched: the axes of a 2D FFT on a 3D View
  EXPECT_EQ((canonical_tpl_axes<c2c_type>(axes2_type{2, 1}, 3)),
            (axes2_type{1, 2}));
}

/// \brief Test if the plan is available for 2D transforms on 2D View
/// \tparam T Type of the data (float or double)
/// \tparam LayoutType Layout of the data (LayoutLeft or LayoutRight)
///
/// \param[in] nprocs Number of processes in the MPI communicator
/// \param[in] n0 Global extent in the first dimension
/// \param[in] n1 Global extent in the second dimension
template <typename T, typename LayoutType>
void test_tpl2D_execute_View2D(std::size_t nprocs, std::size_t n0,
                               std::size_t n1) {
  using View2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using float_type = KokkosFFT::Impl::base_floating_point_type<T>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<float_type>**, LayoutType, execution_space>;
  using axes_type     = KokkosFFT::axis_type<2>;
  using extents_type  = std::array<std::size_t, 2>;
  using topology_type = std::array<std::size_t, 2>;

  constexpr bool is_R2C = KokkosFFT::Impl::is_real_v<T>;

  // 2D Topologies
  topology_type topology0{1, nprocs}, topology1{nprocs, 1};
  axes_type ax01{0, 1}, ax10{1, 0};

  const std::size_t n0h = KokkosFFT::Impl::extent_after_transform(n0, is_R2C),
                    n1h = KokkosFFT::Impl::extent_after_transform(n1, is_R2C);
  extents_type global_in_extents{n0, n1}, global_out_extents_ax0{n0h, n1},
      global_out_extents_ax1{n0, n1h};

  auto [in_extents_t0, in_starts_t0] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_in_extents, topology0, MPI_COMM_WORLD);
  auto [in_extents_t1, in_starts_t1] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_in_extents, topology1, MPI_COMM_WORLD);
  auto [out_extents_t0_ax0, out_starts_t0_ax0] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax0, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax0, out_starts_t1_ax0] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax0, topology1, MPI_COMM_WORLD);
  auto [out_extents_t0_ax1, out_starts_t0_ax1] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax1, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax1, out_starts_t1_ax1] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax1, topology1, MPI_COMM_WORLD);

  // Make reference with a basic-API
  View2DType gu("gu", n0, n1), gu_inv("gu_inv", n0, n1);
  ComplexView2DType gu_hat_ax01("gu_hat_ax01", n0, n1h),
      gu_hat_ax10("gu_hat_ax10", n0h, n1);

  // Data in Topology 0 (X-slab)
  View2DType u_0("u_0",
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
  View2DType u_1("u_1",
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

  if constexpr (is_R2C) {
    KokkosFFT::rfft2(exec, gu, gu_hat_ax01, KokkosFFT::Normalization::backward,
                     ax01);
    KokkosFFT::rfft2(exec, gu, gu_hat_ax10, KokkosFFT::Normalization::backward,
                     ax10);
  } else {
    KokkosFFT::fft2(exec, gu, gu_hat_ax01, KokkosFFT::Normalization::backward,
                    ax01);
    KokkosFFT::fft2(exec, gu, gu_hat_ax10, KokkosFFT::Normalization::backward,
                    ax10);
  }

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

  // Not a slab geometry: cuFFTMp throws, rocFFT runs the cases below
  if (nprocs == 1 && !is_tpl_plan_on_one_process) {
    ASSERT_THROW(
        {
          KokkosFFT::Distributed::Impl::TplPlan plan_0_1_ax01(
              exec, u_0, u_hat_1_ax01, ax01, topology0, topology1,
              MPI_COMM_WORLD);
        },
        std::runtime_error);
    ASSERT_THROW(
        {
          KokkosFFT::Distributed::Impl::TplPlan plan_0_1_ax10(
              exec, u_0, u_hat_1_ax10, ax10, topology0, topology1,
              MPI_COMM_WORLD);
        },
        std::runtime_error);
    ASSERT_THROW(
        {
          KokkosFFT::Distributed::Impl::TplPlan plan_1_0_ax01(
              exec, u_1, u_hat_0_ax01, ax01, topology1, topology0,
              MPI_COMM_WORLD);
        },
        std::runtime_error);
    ASSERT_THROW(
        {
          KokkosFFT::Distributed::Impl::TplPlan plan_1_0_ax10(
              exec, u_1, u_hat_0_ax10, ax10, topology1, topology0,
              MPI_COMM_WORLD);
        },
        std::runtime_error);
  } else {
    // topo0 -> topo0 with ax = {0, 1}:
    // (n0, n1/p) -> (n0, (n1/2+1)/p)
    if (KokkosFFT::Distributed::Impl::is_tpl_available(
            exec, u_0, u_hat_0_ax01, ax01, topology0, topology0)) {
      Kokkos::deep_copy(u_0, ref_u_inv_0);
      KokkosFFT::Distributed::Impl::TplPlan plan_0_0_ax01(
          exec, u_0, u_hat_0_ax01, ax01, topology0, topology0, MPI_COMM_WORLD);
      plan_0_0_ax01.forward(u_0, u_hat_0_ax01);
      EXPECT_TRUE(
          allclose(exec, u_hat_0_ax01, ref_u_hat_0_ax01, 1.0e-5, 1.0e-5));

      plan_0_0_ax01.backward(u_hat_0_ax01, u_inv_0);
      EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));
    }

    if (KokkosFFT::Distributed::Impl::is_tpl_available(
            exec, u_0, u_hat_0_ax10, ax10, topology0, topology0)) {
      // topo0 -> topo0 with ax = {1, 0}:
      // (n0, n1/p) -> (n0/2+1, n1/p)
      Kokkos::deep_copy(u_0, ref_u_inv_0);
      KokkosFFT::Distributed::Impl::TplPlan plan_0_0_ax10(
          exec, u_0, u_hat_0_ax10, ax10, topology0, topology0, MPI_COMM_WORLD);
      plan_0_0_ax10.forward(u_0, u_hat_0_ax10);
      EXPECT_TRUE(
          allclose(exec, u_hat_0_ax10, ref_u_hat_0_ax10, 1.0e-5, 1.0e-5));

      plan_0_0_ax10.backward(u_hat_0_ax10, u_inv_0);
      EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));
    }

    {
      // topo0 -> topo1 with ax = {0, 1}:
      // (n0, n1/p) -> (n0/p, n1/2+1)
      Kokkos::deep_copy(u_0, ref_u_inv_0);
      KokkosFFT::Distributed::Impl::TplPlan plan_0_1_ax01(
          exec, u_0, u_hat_1_ax01, ax01, topology0, topology1, MPI_COMM_WORLD);

      plan_0_1_ax01.forward(u_0, u_hat_1_ax01);
      EXPECT_TRUE(allclose(exec, u_hat_1_ax01, ref_u_hat_1_ax01));

      plan_0_1_ax01.backward(u_hat_1_ax01, u_inv_0);
      EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));
    }

    {
      // topo0 -> topo1 with ax = {1, 0}:
      // (n0, n1/p) -> ((n0/2+1)/p, n1)
      Kokkos::deep_copy(u_0, ref_u_inv_0);
      KokkosFFT::Distributed::Impl::TplPlan plan_0_1_ax10(
          exec, u_0, u_hat_1_ax10, ax10, topology0, topology1, MPI_COMM_WORLD);

      plan_0_1_ax10.forward(u_0, u_hat_1_ax10);
      EXPECT_TRUE(
          allclose(exec, u_hat_1_ax10, ref_u_hat_1_ax10, 1.0e-5, 1.0e-6));

      plan_0_1_ax10.backward(u_hat_1_ax10, u_inv_0);
      EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));
    }

    {
      // topo1 -> topo0 with ax = {0, 1}:
      // (n0/p, n1) -> (n0, (n1/2+1)/p)
      Kokkos::deep_copy(u_1, ref_u_inv_1);
      KokkosFFT::Distributed::Impl::TplPlan plan_1_0_ax01(
          exec, u_1, u_hat_0_ax01, ax01, topology1, topology0, MPI_COMM_WORLD);
      plan_1_0_ax01.forward(u_1, u_hat_0_ax01);
      EXPECT_TRUE(
          allclose(exec, u_hat_0_ax01, ref_u_hat_0_ax01, 1.0e-5, 1.0e-6));

      plan_1_0_ax01.backward(u_hat_0_ax01, u_inv_1);
      EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));
    }

    {
      // topo1 -> topo0 with ax = {1, 0}:
      // (n0/p, n1) -> ((n0/2+1)/p, n1)
      Kokkos::deep_copy(u_1, ref_u_inv_1);
      KokkosFFT::Distributed::Impl::TplPlan plan_1_0_ax10(
          exec, u_1, u_hat_0_ax10, ax10, topology1, topology0, MPI_COMM_WORLD);
      plan_1_0_ax10.forward(u_1, u_hat_0_ax10);
      EXPECT_TRUE(allclose(exec, u_hat_0_ax10, ref_u_hat_0_ax10));

      plan_1_0_ax10.backward(u_hat_0_ax10, u_inv_1);
      EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));
    }

    // topo1 -> topo1 with ax = {0, 1}:
    // (n0/p, n1) -> (n0/p, (n1/2+1))
    if (KokkosFFT::Distributed::Impl::is_tpl_available(
            exec, u_1, u_hat_1_ax01, ax01, topology1, topology1)) {
      Kokkos::deep_copy(u_1, ref_u_inv_1);
      KokkosFFT::Distributed::Impl::TplPlan plan_1_1_ax01(
          exec, u_1, u_hat_1_ax01, ax01, topology1, topology1, MPI_COMM_WORLD);
      plan_1_1_ax01.forward(u_1, u_hat_1_ax01);
      EXPECT_TRUE(
          allclose(exec, u_hat_1_ax01, ref_u_hat_1_ax01, 1.0e-5, 1.0e-5));

      plan_1_1_ax01.backward(u_hat_1_ax01, u_inv_1);
      EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));
    }

    // topo1 -> topo1 with ax = {1, 0}:
    // (n0/p, n1) -> ((n0/2+1), n1/p)
    if (KokkosFFT::Distributed::Impl::is_tpl_available(
            exec, u_1, u_hat_1_ax10, ax10, topology1, topology1)) {
      Kokkos::deep_copy(u_1, ref_u_inv_1);
      KokkosFFT::Distributed::Impl::TplPlan plan_1_1_ax10(
          exec, u_1, u_hat_1_ax10, ax10, topology1, topology1, MPI_COMM_WORLD);
      plan_1_1_ax10.forward(u_1, u_hat_1_ax10);
      EXPECT_TRUE(
          allclose(exec, u_hat_1_ax10, ref_u_hat_1_ax10, 1.0e-5, 1.0e-5));

      plan_1_1_ax10.backward(u_hat_1_ax10, u_inv_1);
      EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));
    }
  }
}

template <typename T, typename LayoutType>
void test_tpl3D_execute_View3D(std::size_t nprocs, std::size_t n0,
                               std::size_t n1, std::size_t n2) {
  using View3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using float_type = KokkosFFT::Impl::base_floating_point_type<T>;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<float_type>***, LayoutType, execution_space>;
  using axes_type     = KokkosFFT::axis_type<3>;
  using extents_type  = std::array<std::size_t, 3>;
  using topology_type = std::array<std::size_t, 3>;

  constexpr bool is_R2C = KokkosFFT::Impl::is_real_v<T>;

  topology_type topology0{1, 1, nprocs}, topology1{1, nprocs, 1},
      topology2{nprocs, 1, 1};

  const std::size_t n0h = KokkosFFT::Impl::extent_after_transform(n0, is_R2C),
                    n1h = KokkosFFT::Impl::extent_after_transform(n1, is_R2C),
                    n2h = KokkosFFT::Impl::extent_after_transform(n2, is_R2C);
  extents_type global_in_extents{n0, n1, n2},
      global_out_extents_ax0{n0h, n1, n2}, global_out_extents_ax1{n0, n1h, n2},
      global_out_extents_ax2{n0, n1, n2h};

  // All axes
  axes_type ax012{0, 1, 2}, ax021{0, 2, 1}, ax102{1, 0, 2}, ax120{1, 2, 0},
      ax201{2, 0, 1}, ax210{2, 1, 0};

  auto [in_extents_t0, in_starts_t0] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_in_extents, topology0, MPI_COMM_WORLD);
  auto [in_extents_t1, in_starts_t1] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_in_extents, topology1, MPI_COMM_WORLD);
  auto [in_extents_t2, in_starts_t2] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_in_extents, topology2, MPI_COMM_WORLD);
  auto [out_extents_t0_ax0, out_starts_t0_ax0] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax0, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax0, out_starts_t1_ax0] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax0, topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax0, out_starts_t2_ax0] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax0, topology2, MPI_COMM_WORLD);
  auto [out_extents_t0_ax1, out_starts_t0_ax1] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax1, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax1, out_starts_t1_ax1] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax1, topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax1, out_starts_t2_ax1] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax1, topology2, MPI_COMM_WORLD);
  auto [out_extents_t0_ax2, out_starts_t0_ax2] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax2, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax2, out_starts_t1_ax2] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax2, topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax2, out_starts_t2_ax2] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax2, topology2, MPI_COMM_WORLD);

  // Make reference with a basic-API
  View3DType gu("gu", n0, n1, n2);
  ComplexView3DType gu_hat_ax012("gu_hat_ax012", n0, n1, n2h),
      gu_hat_ax021("gu_hat_ax021", n0, n1h, n2),
      gu_hat_ax102("gu_hat_ax102", n0, n1, n2h),
      gu_hat_ax120("gu_hat_ax120", n0h, n1, n2),
      gu_hat_ax201("gu_hat_ax201", n0, n1h, n2),
      gu_hat_ax210("gu_hat_ax210", n0h, n1, n2);

  // Data in Topology 0 (Z-slab)
  View3DType u_0("u_0",
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

  // Data in Topology 1 (Y-slab)
  View3DType u_1("u_1",
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

  // Data in Topology 2 (X-slab)
  View3DType u_2("u_2",
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

  if constexpr (is_R2C) {
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
  } else {
    KokkosFFT::fftn(exec, gu, gu_hat_ax012, ax012,
                    KokkosFFT::Normalization::backward);
    KokkosFFT::fftn(exec, gu, gu_hat_ax021, ax021,
                    KokkosFFT::Normalization::backward);
    KokkosFFT::fftn(exec, gu, gu_hat_ax102, ax102,
                    KokkosFFT::Normalization::backward);
    KokkosFFT::fftn(exec, gu, gu_hat_ax120, ax120,
                    KokkosFFT::Normalization::backward);
    KokkosFFT::fftn(exec, gu, gu_hat_ax201, ax201,
                    KokkosFFT::Normalization::backward);
    KokkosFFT::fftn(exec, gu, gu_hat_ax210, ax210,
                    KokkosFFT::Normalization::backward);
  }

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

  // Define ranges for topology 0 (Z-slab)
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
  auto sub_gu_hat_0_ax012 = Kokkos::subview(gu_hat_ax012, Kokkos::ALL,
                                            Kokkos::ALL, range_gu_hat_0_ax2);
  Kokkos::deep_copy(ref_u_hat_0_ax012, sub_gu_hat_0_ax012);

  // Topo 0 ax = {0, 2, 1}
  auto sub_gu_hat_0_ax021 = Kokkos::subview(gu_hat_ax021, Kokkos::ALL,
                                            Kokkos::ALL, range_gu_hat_0_ax1);
  Kokkos::deep_copy(ref_u_hat_0_ax021, sub_gu_hat_0_ax021);

  // Topo 0 ax = {1, 0, 2}
  auto sub_gu_hat_0_ax102 = Kokkos::subview(gu_hat_ax102, Kokkos::ALL,
                                            Kokkos::ALL, range_gu_hat_0_ax2);
  Kokkos::deep_copy(ref_u_hat_0_ax102, sub_gu_hat_0_ax102);

  // Topo 0 ax = {1, 2, 0}
  auto sub_gu_hat_0_ax120 = Kokkos::subview(gu_hat_ax120, Kokkos::ALL,
                                            Kokkos::ALL, range_gu_hat_0_ax0);
  Kokkos::deep_copy(ref_u_hat_0_ax120, sub_gu_hat_0_ax120);

  // Topo 0 ax = {2, 0, 1}
  auto sub_gu_hat_0_ax201 = Kokkos::subview(gu_hat_ax201, Kokkos::ALL,
                                            Kokkos::ALL, range_gu_hat_0_ax1);
  Kokkos::deep_copy(ref_u_hat_0_ax201, sub_gu_hat_0_ax201);

  // Topo 0 ax = {2, 1, 0}
  auto sub_gu_hat_0_ax210 = Kokkos::subview(gu_hat_ax210, Kokkos::ALL,
                                            Kokkos::ALL, range_gu_hat_0_ax0);
  Kokkos::deep_copy(ref_u_hat_0_ax210, sub_gu_hat_0_ax210);

  // Define ranges for topology 1 (Y-slab)
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
  auto sub_gu_hat_1_ax012 = Kokkos::subview(gu_hat_ax012, Kokkos::ALL,
                                            range_gu_hat_1_ax2, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_1_ax012, sub_gu_hat_1_ax012);

  // Topo 1 ax = {0, 2, 1}
  auto sub_gu_hat_1_ax021 = Kokkos::subview(gu_hat_ax021, Kokkos::ALL,
                                            range_gu_hat_1_ax1, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_1_ax021, sub_gu_hat_1_ax021);

  // Topo 1 ax = {1, 0, 2}
  auto sub_gu_hat_1_ax102 = Kokkos::subview(gu_hat_ax102, Kokkos::ALL,
                                            range_gu_hat_1_ax2, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_1_ax102, sub_gu_hat_1_ax102);

  // Topo 1 ax = {1, 2, 0}
  auto sub_gu_hat_1_ax120 = Kokkos::subview(gu_hat_ax120, Kokkos::ALL,
                                            range_gu_hat_1_ax0, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_1_ax120, sub_gu_hat_1_ax120);

  // Topo 1 ax = {2, 0, 1}
  auto sub_gu_hat_1_ax201 = Kokkos::subview(gu_hat_ax201, Kokkos::ALL,
                                            range_gu_hat_1_ax1, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_1_ax201, sub_gu_hat_1_ax201);

  // Topo 1 ax = {2, 1, 0}
  auto sub_gu_hat_1_ax210 = Kokkos::subview(gu_hat_ax210, Kokkos::ALL,
                                            range_gu_hat_1_ax0, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_1_ax210, sub_gu_hat_1_ax210);

  // Define ranges for topology 2 (X-slab)
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
  auto sub_gu_hat_2_ax012 = Kokkos::subview(gu_hat_ax012, range_gu_hat_2_ax2,
                                            Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax012, sub_gu_hat_2_ax012);

  // Topo 2 ax = {0, 2, 1}
  auto sub_gu_hat_2_ax021 = Kokkos::subview(gu_hat_ax021, range_gu_hat_2_ax1,
                                            Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax021, sub_gu_hat_2_ax021);

  // Topo 2 ax = {1, 0, 2}
  auto sub_gu_hat_2_ax102 = Kokkos::subview(gu_hat_ax102, range_gu_hat_2_ax2,
                                            Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax102, sub_gu_hat_2_ax102);

  // Topo 2 ax = {1, 2, 0}
  auto sub_gu_hat_2_ax120 = Kokkos::subview(gu_hat_ax120, range_gu_hat_2_ax0,
                                            Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax120, sub_gu_hat_2_ax120);

  // Topo 2 ax = {2, 0, 1}
  auto sub_gu_hat_2_ax201 = Kokkos::subview(gu_hat_ax201, range_gu_hat_2_ax1,
                                            Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax201, sub_gu_hat_2_ax201);

  // Topo 2 ax = {2, 1, 0}
  auto sub_gu_hat_2_ax210 = Kokkos::subview(gu_hat_ax210, range_gu_hat_2_ax0,
                                            Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax210, sub_gu_hat_2_ax210);

  // For inverse transform
  Kokkos::deep_copy(ref_u_inv_0, u_0);
  Kokkos::deep_copy(ref_u_inv_1, u_1);
  Kokkos::deep_copy(ref_u_inv_2, u_2);

  // Not a slab geometry: cuFFTMp throws, rocFFT runs the cases below
  if (nprocs == 1 && !is_tpl_plan_on_one_process) {
    // topology0 -> topology1
    // (n0, n1, n2/p) -> (n0, (n1/2+1)/p, n2)
    ASSERT_THROW(
        {
          KokkosFFT::Distributed::Impl::TplPlan plan_0_1_ax012(
              exec, u_0, u_hat_1_ax012, ax012, topology0, topology1,
              MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology0 -> topology2
    // (n0, n1, n2/p) -> (n0/p, n1/2+1, n2)
    ASSERT_THROW(
        {
          KokkosFFT::Distributed::Impl::TplPlan plan_0_2_ax012(
              exec, u_0, u_hat_2_ax012, ax012, topology0, topology2,
              MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology1 -> topology2
    // (n0, n1/p, n2) -> (n0/p, n1, n2/2+1)
    ASSERT_THROW(
        {
          KokkosFFT::Distributed::Impl::TplPlan plan_1_2_ax012(
              exec, u_1, u_hat_2_ax012, ax012, topology1, topology2,
              MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology0 -> topology0 with ax = {1, 2}
    // (n0, n1, n2/p) -> (n0, n1, (n2/2+1)/p)
    ASSERT_THROW(
        {
          KokkosFFT::Distributed::Impl::TplPlan plan_0_0_ax012(
              exec, u_0, u_hat_0_ax012, ax012, topology0, topology0,
              MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology1 -> topology1 with ax = {1, 2}
    // (n0, n1/p, n2) -> (n0, (n1/2+1)/p, n2)
    ASSERT_THROW(
        {
          KokkosFFT::Distributed::Impl::TplPlan plan_1_1_ax210(
              exec, u_1, u_hat_1_ax210, ax210, topology1, topology1,
              MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology2 -> topology2 with ax = {0, 1, 2}
    // (n0/p, n1, n2) -> (n0/p, (n1/2+1), n2)
    ASSERT_THROW(
        {
          KokkosFFT::Distributed::Impl::TplPlan plan_2_2_ax012(
              exec, u_2, u_hat_2_ax012, ax012, topology2, topology2,
              MPI_COMM_WORLD);
        },
        std::runtime_error);
  } else {
    // topo 0 -> topo 1 with ax = {0, 1, 2}:
    // (n0, n1, n2/p) -> (n0, n1/p, n2/2+1)
    {
      KokkosFFT::Distributed::Impl::TplPlan plan_0_1_ax012(
          exec, u_0, u_hat_1_ax012, ax012, topology0, topology1,
          MPI_COMM_WORLD);
      plan_0_1_ax012.forward(u_0, u_hat_1_ax012);
      EXPECT_TRUE(allclose(exec, u_hat_1_ax012, ref_u_hat_1_ax012))
          << "Failed topo 0 -> topo 1 with ax {0, 1, 2} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_0_1_ax012.forward(u_0, u_hat_1_ax012);
      EXPECT_TRUE(allclose(exec, u_hat_1_ax012, ref_u_hat_1_ax012))
          << "Failed (second execution) topo 0 -> topo 1 with ax {0, 1, 2} (nx "
             "x ny x nz): "
          << n0 << " x " << n1 << " x " << n2;

      plan_0_1_ax012.backward(u_hat_1_ax012, u_inv_0);
      EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 0 -> topo 1 with ax {0, 1, 2} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;

      plan_0_1_ax012.backward(u_hat_1_ax012, u_inv_0);
      EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6))
          << "Failed (second inverse execution) topo 0 -> topo 1 with ax {0, "
             "1, 2} (nx x ny x nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 0 -> topo 1 with ax = {0, 2, 1}:
      // (n0, n1, n2/p) -> (n0, (n1/2+1)/p, n2)
      KokkosFFT::Distributed::Impl::TplPlan plan_0_1_ax021(
          exec, u_0, u_hat_1_ax021, ax021, topology0, topology1,
          MPI_COMM_WORLD);
      plan_0_1_ax021.forward(u_0, u_hat_1_ax021);
      EXPECT_TRUE(
          allclose(exec, u_hat_1_ax021, ref_u_hat_1_ax021, 1.0e-5, 1.0e-5))
          << "Failed topo 0 -> topo 1 with ax {0, 2, 1} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_0_1_ax021.backward(u_hat_1_ax021, u_inv_0);
      EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 0 -> topo 1 with ax {0, 2, 1} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 0 -> topo 1 with ax = {1, 0, 2}:
      // (n0, n1, n2/p) -> (n0, n1/p, n2/2+1)
      KokkosFFT::Distributed::Impl::TplPlan plan_0_1_ax102(
          exec, u_0, u_hat_1_ax102, ax102, topology0, topology1,
          MPI_COMM_WORLD);
      plan_0_1_ax102.forward(u_0, u_hat_1_ax102);
      EXPECT_TRUE(
          allclose(exec, u_hat_1_ax102, ref_u_hat_1_ax102, 1.0e-5, 1.0e-4))
          << "Failed topo 0 -> topo 1 with ax {1, 0, 2} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_0_1_ax102.backward(u_hat_1_ax102, u_inv_0);
      EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 0 -> topo 1 with ax {1, 0, 2} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 0 -> topo 1 with ax = {1, 2, 0}:
      // (n0, n1, n2/p) -> (n0/2+1, n1/p, n2)
      KokkosFFT::Distributed::Impl::TplPlan plan_0_1_ax120(
          exec, u_0, u_hat_1_ax120, ax120, topology0, topology1,
          MPI_COMM_WORLD);
      plan_0_1_ax120.forward(u_0, u_hat_1_ax120);
      EXPECT_TRUE(
          allclose(exec, u_hat_1_ax120, ref_u_hat_1_ax120, 1.0e-5, 5.0e-5))
          << "Failed topo 0 -> topo 1 with ax {1, 2, 0} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_0_1_ax120.backward(u_hat_1_ax120, u_inv_0);
      EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 0 -> topo 1 with ax {1, 2, 0} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 0 -> topo 1 with ax = {2, 0, 1}:
      // (n0, n1, n2/p) -> (n0, (n1/2+1)/p, n2)
      KokkosFFT::Distributed::Impl::TplPlan plan_0_1_ax201(
          exec, u_0, u_hat_1_ax201, ax201, topology0, topology1,
          MPI_COMM_WORLD);
      plan_0_1_ax201.forward(u_0, u_hat_1_ax201);
      EXPECT_TRUE(
          allclose(exec, u_hat_1_ax201, ref_u_hat_1_ax201, 1.0e-5, 1.0e-5))
          << "Failed topo 0 -> topo 1 with ax {2, 0, 1} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_0_1_ax201.backward(u_hat_1_ax201, u_inv_0);
      EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 0 -> topo 1 with ax {2, 0, 1} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 0 -> topo 1 with ax = {2, 1, 0}:
      // (n0, n1, n2/p) -> (n0/2+1, n1/p, n2)
      KokkosFFT::Distributed::Impl::TplPlan plan_0_1_ax210(
          exec, u_0, u_hat_1_ax210, ax210, topology0, topology1,
          MPI_COMM_WORLD);
      plan_0_1_ax210.forward(u_0, u_hat_1_ax210);
      EXPECT_TRUE(
          allclose(exec, u_hat_1_ax210, ref_u_hat_1_ax210, 1.0e-5, 1.0e-5))
          << "Failed topo 0 -> topo 1 with ax {2, 1, 0} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_0_1_ax210.backward(u_hat_1_ax210, u_inv_0);
      EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 0 -> topo 1 with ax {2, 1, 0} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 0 -> topo 2 with ax = {0, 1, 2}:
      // (n0, n1, n2/p) -> (n0/p, n1, n2) -> (n0/p, n1, n2/2+1)
      KokkosFFT::Distributed::Impl::TplPlan plan_0_2_ax012(
          exec, u_0, u_hat_2_ax012, ax012, topology0, topology2,
          MPI_COMM_WORLD);
      plan_0_2_ax012.forward(u_0, u_hat_2_ax012);
      EXPECT_TRUE(
          allclose(exec, u_hat_2_ax012, ref_u_hat_2_ax012, 1.0e-5, 5.0e-5))
          << "Failed topo 0 -> topo 2 with ax {0, 1, 2} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_0_2_ax012.backward(u_hat_2_ax012, u_inv_0);
      EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 0 -> topo 2 with ax {0, 1, 2} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 0 -> topo 2 with ax = {0, 2, 1}:
      // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0, (n1/2+1)/p, n2)
      KokkosFFT::Distributed::Impl::TplPlan plan_0_2_ax021(
          exec, u_0, u_hat_2_ax021, ax021, topology0, topology2,
          MPI_COMM_WORLD);
      plan_0_2_ax021.forward(u_0, u_hat_2_ax021);
      EXPECT_TRUE(
          allclose(exec, u_hat_2_ax021, ref_u_hat_2_ax021, 1.0e-5, 1.0e-5))
          << "Failed topo 0 -> topo 2 with ax {0, 2, 1} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_0_2_ax021.backward(u_hat_2_ax021, u_inv_0);
      EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 0 -> topo 2 with ax {0, 2, 1} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 0 -> topo 2 with ax = {1, 0, 2}:
      // (n0, n1, n2/p) -> (n0, n1/p, n2) -> (n0, n1/p, n2/2+1)
      KokkosFFT::Distributed::Impl::TplPlan plan_0_2_ax102(
          exec, u_0, u_hat_2_ax102, ax102, topology0, topology2,
          MPI_COMM_WORLD);
      plan_0_2_ax102.forward(u_0, u_hat_2_ax102);
      EXPECT_TRUE(allclose(exec, u_hat_2_ax102, ref_u_hat_2_ax102))
          << "Failed topo 0 -> topo 2 with ax {1, 0, 2} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_0_2_ax102.backward(u_hat_2_ax102, u_inv_0);
      EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 0 -> topo 2 with ax {1, 0, 2} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 0 -> topo 2 with ax = {1, 2, 0}:
      // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
      KokkosFFT::Distributed::Impl::TplPlan plan_0_2_ax120(
          exec, u_0, u_hat_2_ax120, ax120, topology0, topology2,
          MPI_COMM_WORLD);
      plan_0_2_ax120.forward(u_0, u_hat_2_ax120);
      EXPECT_TRUE(
          allclose(exec, u_hat_2_ax120, ref_u_hat_2_ax120, 1.0e-5, 5.0e-5))
          << "Failed topo 0 -> topo 2 with ax {1, 2, 0} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_0_2_ax120.backward(u_hat_2_ax120, u_inv_0);
      EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 0 -> topo 2 with ax {1, 2, 0} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 0 -> topo 2 with ax = {2, 0, 1}:
      // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0/p, n1/2+1, n2)
      KokkosFFT::Distributed::Impl::TplPlan plan_0_2_ax201(
          exec, u_0, u_hat_2_ax201, ax201, topology0, topology2,
          MPI_COMM_WORLD);
      plan_0_2_ax201.forward(u_0, u_hat_2_ax201);
      EXPECT_TRUE(
          allclose(exec, u_hat_2_ax201, ref_u_hat_2_ax201, 1.0e-5, 1.0e-5))
          << "Failed topo 0 -> topo 2 with ax {2, 0, 1} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_0_2_ax201.backward(u_hat_2_ax201, u_inv_0);
      EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 0 -> topo 2 with ax {2, 0, 1} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 0 -> topo 2 with ax = {2, 1, 0}:
      // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
      KokkosFFT::Distributed::Impl::TplPlan plan_0_2_ax210(
          exec, u_0, u_hat_2_ax210, ax210, topology0, topology2,
          MPI_COMM_WORLD);
      plan_0_2_ax210.forward(u_0, u_hat_2_ax210);
      EXPECT_TRUE(
          allclose(exec, u_hat_2_ax210, ref_u_hat_2_ax210, 1.0e-5, 1.0e-5))
          << "Failed topo 0 -> topo 2 with ax {2, 1, 0} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_0_2_ax210.backward(u_hat_2_ax210, u_inv_0);
      EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 0 -> topo 2 with ax {2, 1, 0} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 1 -> topo 0 with ax = {0, 1, 2}:
      // (n0p, n1/p, n2) -> (n0, n1, (n2/2+1)/p)
      KokkosFFT::Distributed::Impl::TplPlan plan_1_0_ax012(
          exec, u_1, u_hat_0_ax012, ax012, topology1, topology0,
          MPI_COMM_WORLD);
      plan_1_0_ax012.forward(u_1, u_hat_0_ax012);
      EXPECT_TRUE(
          allclose(exec, u_hat_0_ax012, ref_u_hat_0_ax012, 1.0e-5, 5.0e-5))
          << "Failed topo 1 -> topo 0 with ax {0, 1, 2} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_1_0_ax012.backward(u_hat_0_ax012, u_inv_1);
      EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 1 -> topo 0 with ax {0, 1, 2} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 1 -> topo 0 with ax = {0, 2, 1}:
      // (n0p, n1/p, n2) -> (n0, n1/2+1, n2/p) -> (n0, (n1/2+1)/p, n2)
      KokkosFFT::Distributed::Impl::TplPlan plan_1_0_ax021(
          exec, u_1, u_hat_0_ax021, ax021, topology1, topology0,
          MPI_COMM_WORLD);
      plan_1_0_ax021.forward(u_1, u_hat_0_ax021);
      EXPECT_TRUE(
          allclose(exec, u_hat_0_ax021, ref_u_hat_0_ax021, 1.0e-5, 1.0e-5))
          << "Failed topo 1 -> topo 0 with ax {0, 2, 1} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_1_0_ax021.backward(u_hat_0_ax021, u_inv_1);
      EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 1 -> topo 0 with ax {0, 2, 1} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 1 -> topo 0 with ax = {1, 0, 2}:
      // (n0p, n1/p, n2) -> (n0, n1/2+1, (n2/2+1)/p)
      KokkosFFT::Distributed::Impl::TplPlan plan_1_0_ax102(
          exec, u_1, u_hat_0_ax102, ax102, topology1, topology0,
          MPI_COMM_WORLD);
      plan_1_0_ax102.forward(u_1, u_hat_0_ax102);
      EXPECT_TRUE(
          allclose(exec, u_hat_0_ax102, ref_u_hat_0_ax102, 1.0e-5, 1.0e-5))
          << "Failed topo 1 -> topo 0 with ax {1, 0, 2} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_1_0_ax102.backward(u_hat_0_ax102, u_inv_1);
      EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 1 -> topo 0 with ax {1, 0, 2} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 1 -> topo 0 with ax = {1, 2, 0}:
      // (n0p, n1/p, n2) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
      KokkosFFT::Distributed::Impl::TplPlan plan_1_0_ax120(
          exec, u_1, u_hat_0_ax120, ax120, topology1, topology0,
          MPI_COMM_WORLD);
      plan_1_0_ax120.forward(u_1, u_hat_0_ax120);
      EXPECT_TRUE(
          allclose(exec, u_hat_0_ax120, ref_u_hat_0_ax120, 1.0e-5, 1.0e-5))
          << "Failed topo 1 -> topo 0 with ax {1, 2, 0} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_1_0_ax120.backward(u_hat_0_ax120, u_inv_1);
      EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 1 -> topo 0 with ax {1, 2, 0} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 1 -> topo 0 with ax = {2, 0, 1}:
      // (n0p, n1/p, n2) -> (n0, n1/2+1, n2/p) -> (n0/p, n1/2+1, n2)
      KokkosFFT::Distributed::Impl::TplPlan plan_1_0_ax201(
          exec, u_1, u_hat_0_ax201, ax201, topology1, topology0,
          MPI_COMM_WORLD);
      plan_1_0_ax201.forward(u_1, u_hat_0_ax201);
      EXPECT_TRUE(
          allclose(exec, u_hat_0_ax201, ref_u_hat_0_ax201, 1.0e-5, 1.0e-5))
          << "Failed topo 1 -> topo 0 with ax {2, 0, 1} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_1_0_ax201.backward(u_hat_0_ax201, u_inv_1);
      EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 1 -> topo 0 with ax {2, 0, 1} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 1 -> topo 0 with ax = {2, 1, 0}:
      // (n0p, n1/p, n2) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
      KokkosFFT::Distributed::Impl::TplPlan plan_1_0_ax210(
          exec, u_1, u_hat_0_ax210, ax210, topology1, topology0,
          MPI_COMM_WORLD);
      plan_1_0_ax210.forward(u_1, u_hat_0_ax210);
      EXPECT_TRUE(
          allclose(exec, u_hat_0_ax210, ref_u_hat_0_ax210, 1.0e-5, 5.0e-5))
          << "Failed topo 1 -> topo 0 with ax {2, 1, 0} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_1_0_ax210.backward(u_hat_0_ax210, u_inv_1);
      EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 1 -> topo 0 with ax {2, 1, 0} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 1 -> topo 2 with ax = {0, 1, 2}:
      // (n0, n1/p, n2) -> (n0/p, n1, n2) -> (n0/p, n1, n2/2+1)
      KokkosFFT::Distributed::Impl::TplPlan plan_1_2_ax012(
          exec, u_1, u_hat_2_ax012, ax012, topology1, topology2,
          MPI_COMM_WORLD);
      plan_1_2_ax012.forward(u_1, u_hat_2_ax012);
      EXPECT_TRUE(
          allclose(exec, u_hat_2_ax012, ref_u_hat_2_ax012, 1.0e-5, 5.0e-5))
          << "Failed topo 1 -> topo 2 with ax {0, 1, 2} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_1_2_ax012.backward(u_hat_2_ax012, u_inv_1);
      EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 1 -> topo 2 with ax {0, 1, 2} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 1 -> topo 2 with ax = {0, 2, 1}:
      // (n0, n1/p, n2) -> (n0, n1/2+1, n2/p) -> (n0, (n1/2+1)/p, n2)
      KokkosFFT::Distributed::Impl::TplPlan plan_1_2_ax021(
          exec, u_1, u_hat_2_ax021, ax021, topology1, topology2,
          MPI_COMM_WORLD);
      plan_1_2_ax021.forward(u_1, u_hat_2_ax021);
      EXPECT_TRUE(
          allclose(exec, u_hat_2_ax021, ref_u_hat_2_ax021, 1.0e-5, 1.0e-5))
          << "Failed topo 1 -> topo 2 with ax {0, 2, 1} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_1_2_ax021.backward(u_hat_2_ax021, u_inv_1);
      EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 1 -> topo 2 with ax {0, 2, 1} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 1 -> topo 2 with ax = {1, 0, 2}:
      // (n0, n1/p, n2) -> (n0, n1/2+1, n2/p) -> (n0, n1/p, n2/2+1)
      KokkosFFT::Distributed::Impl::TplPlan plan_1_2_ax102(
          exec, u_1, u_hat_2_ax102, ax102, topology1, topology2,
          MPI_COMM_WORLD);
      plan_1_2_ax102.forward(u_1, u_hat_2_ax102);
      EXPECT_TRUE(
          allclose(exec, u_hat_2_ax102, ref_u_hat_2_ax102, 1.0e-5, 1.0e-5))
          << "Failed topo 1 -> topo 2 with ax {1, 0, 2} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_1_2_ax102.backward(u_hat_2_ax102, u_inv_1);
      EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 1 -> topo 2 with ax {1, 0, 2} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 1 -> topo 2 with ax = {1, 2, 0}:
      // (n0, n1/p, n2) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
      KokkosFFT::Distributed::Impl::TplPlan plan_1_2_ax120(
          exec, u_1, u_hat_2_ax120, ax120, topology1, topology2,
          MPI_COMM_WORLD);
      plan_1_2_ax120.forward(u_1, u_hat_2_ax120);
      EXPECT_TRUE(
          allclose(exec, u_hat_2_ax120, ref_u_hat_2_ax120, 1.0e-5, 1.0e-5))
          << "Failed topo 1 -> topo 2 with ax {1, 2, 0} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_1_2_ax120.backward(u_hat_2_ax120, u_inv_1);
      EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 1 -> topo 2 with ax {1, 2, 0} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 1 -> topo 2 with ax = {2, 0, 1}:
      // (n0, n1/p, n2) -> (n0, n1/2+1, n2/p) -> (n0/p, n1/2+1, n2)
      KokkosFFT::Distributed::Impl::TplPlan plan_1_2_ax201(
          exec, u_1, u_hat_2_ax201, ax201, topology1, topology2,
          MPI_COMM_WORLD);
      plan_1_2_ax201.forward(u_1, u_hat_2_ax201);
      EXPECT_TRUE(allclose(exec, u_hat_2_ax201, ref_u_hat_2_ax201))
          << "Failed topo 1 -> topo 2 with ax {2, 0, 1} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_1_2_ax201.backward(u_hat_2_ax201, u_inv_1);
      EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 1 -> topo 2 with ax {2, 0, 1} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 1 -> topo 2 with ax = {2, 1, 0}:
      // (n0, n1/p, n2) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
      KokkosFFT::Distributed::Impl::TplPlan plan_1_2_ax210(
          exec, u_1, u_hat_2_ax210, ax210, topology1, topology2,
          MPI_COMM_WORLD);
      plan_1_2_ax210.forward(u_1, u_hat_2_ax210);
      EXPECT_TRUE(
          allclose(exec, u_hat_2_ax210, ref_u_hat_2_ax210, 1.0e-5, 5.0e-5))
          << "Failed topo 1 -> topo 2 with ax {2, 1, 0} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_1_2_ax210.backward(u_hat_2_ax210, u_inv_1);
      EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 1 -> topo 2 with ax {2, 1, 0} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 2 -> topo 0 with ax = {0, 1, 2}:
      // (n0/p, n1, n2) -> (n0, n1, n2/2+1) -> (n0, n1, n2/p)
      KokkosFFT::Distributed::Impl::TplPlan plan_2_0_ax012(
          exec, u_2, u_hat_0_ax012, ax012, topology2, topology0,
          MPI_COMM_WORLD);
      plan_2_0_ax012.forward(u_2, u_hat_0_ax012);
      EXPECT_TRUE(
          allclose(exec, u_hat_0_ax012, ref_u_hat_0_ax012, 1.0e-5, 1.0e-5))
          << "Failed topo 2 -> topo 0 with ax {0, 1, 2} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_2_0_ax012.backward(u_hat_0_ax012, u_inv_2);
      EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 2 -> topo 0 with ax {0, 1, 2} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 2 -> topo 0 with ax = {0, 2, 1}:
      // (n0/p, n1, n2) -> (n0, n1/2+1, n2/p) -> (n0, (n1/2+1)/p, n2)
      KokkosFFT::Distributed::Impl::TplPlan plan_2_0_ax021(
          exec, u_2, u_hat_0_ax021, ax021, topology2, topology0,
          MPI_COMM_WORLD);
      plan_2_0_ax021.forward(u_2, u_hat_0_ax021);
      EXPECT_TRUE(
          allclose(exec, u_hat_0_ax021, ref_u_hat_0_ax021, 1.0e-5, 1.0e-5))
          << "Failed topo 2 -> topo 0 with ax {0, 2, 1} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_2_0_ax021.backward(u_hat_0_ax021, u_inv_2);
      EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 2 -> topo 0 with ax {0, 2, 1} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 2 -> topo 0 with ax = {1, 0, 2}:
      // (n0/p, n1, n2) -> (n0, n1/2+1, n2/p) -> (n0, n1/p, n2/2+1)
      KokkosFFT::Distributed::Impl::TplPlan plan_2_0_ax102(
          exec, u_2, u_hat_0_ax102, ax102, topology2, topology0,
          MPI_COMM_WORLD);
      plan_2_0_ax102.forward(u_2, u_hat_0_ax102);
      EXPECT_TRUE(
          allclose(exec, u_hat_0_ax102, ref_u_hat_0_ax102, 1.0e-5, 5.0e-5))
          << "Failed topo 2 -> topo 0 with ax {1, 0, 2} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_2_0_ax102.backward(u_hat_0_ax102, u_inv_2);
      EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 2 -> topo 0 with ax {1, 0, 2} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 2 -> topo 0 with ax = {1, 2, 0}:
      // (n0/p, n1, n2) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
      KokkosFFT::Distributed::Impl::TplPlan plan_2_0_ax120(
          exec, u_2, u_hat_0_ax120, ax120, topology2, topology0,
          MPI_COMM_WORLD);
      plan_2_0_ax120.forward(u_2, u_hat_0_ax120);
      EXPECT_TRUE(
          allclose(exec, u_hat_0_ax120, ref_u_hat_0_ax120, 1.0e-5, 1.0e-5))
          << "Failed topo 2 -> topo 0 with ax {1, 2, 0} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_2_0_ax120.backward(u_hat_0_ax120, u_inv_2);
      EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 2 -> topo 0 with ax {1, 2, 0} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 2 -> topo 0 with ax = {2, 0, 1}:
      // (n0/p, n1, n2) -> (n0, n1/2+1, n2/p) -> (n0/p, n1/2+1, n2)
      KokkosFFT::Distributed::Impl::TplPlan plan_2_0_ax201(
          exec, u_2, u_hat_0_ax201, ax201, topology2, topology0,
          MPI_COMM_WORLD);
      plan_2_0_ax201.forward(u_2, u_hat_0_ax201);
      EXPECT_TRUE(
          allclose(exec, u_hat_0_ax201, ref_u_hat_0_ax201, 1.0e-5, 1.0e-5))
          << "Failed topo 2 -> topo 0 with ax {2, 0, 1} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_2_0_ax201.backward(u_hat_0_ax201, u_inv_2);
      EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 2 -> topo 0 with ax {2, 0, 1} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 2 -> topo 0 with ax = {2, 1, 0}:
      // (n0/p, n1, n2) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
      KokkosFFT::Distributed::Impl::TplPlan plan_2_0_ax210(
          exec, u_2, u_hat_0_ax210, ax210, topology2, topology0,
          MPI_COMM_WORLD);
      plan_2_0_ax210.forward(u_2, u_hat_0_ax210);
      EXPECT_TRUE(
          allclose(exec, u_hat_0_ax210, ref_u_hat_0_ax210, 1.0e-5, 5.0e-5))
          << "Failed topo 2 -> topo 0 with ax {2, 1, 0} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_2_0_ax210.backward(u_hat_0_ax210, u_inv_2);
      EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 2 -> topo 0 with ax {2, 1, 0} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 2 -> topo 1 with ax = {0, 1, 2}:
      // (n0/p, n1, n2) -> (n0/p, n1, n2/2+1) -> (n0, n1/p, n2/2+1)
      KokkosFFT::Distributed::Impl::TplPlan plan_2_1_ax012(
          exec, u_2, u_hat_1_ax012, ax012, topology2, topology1,
          MPI_COMM_WORLD);
      plan_2_1_ax012.forward(u_2, u_hat_1_ax012);
      EXPECT_TRUE(
          allclose(exec, u_hat_1_ax012, ref_u_hat_1_ax012, 1.0e-5, 1.0e-5))
          << "Failed topo 2 -> topo 1 with ax {0, 1, 2} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_2_1_ax012.backward(u_hat_1_ax012, u_inv_2);
      EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 2 -> topo 1 with ax {0, 1, 2} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 2 -> topo 1 with ax = {0, 2, 1}:
      // (n0/p, n1, n2) -> (n0, n1/2+1, n2/p) -> (n0, (n1/2+1)/p, n2)
      KokkosFFT::Distributed::Impl::TplPlan plan_2_1_ax021(
          exec, u_2, u_hat_1_ax021, ax021, topology2, topology1,
          MPI_COMM_WORLD);
      plan_2_1_ax021.forward(u_2, u_hat_1_ax021);
      EXPECT_TRUE(
          allclose(exec, u_hat_1_ax021, ref_u_hat_1_ax021, 1.0e-5, 1.0e-5))
          << "Failed topo 2 -> topo 1 with ax {0, 2, 1} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_2_1_ax021.backward(u_hat_1_ax021, u_inv_2);
      EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 2 -> topo 1 with ax {0, 2, 1} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 2 -> topo 1 with ax = {1, 0, 2}:
      // (n0/p, n1, n2) -> (n0/p, n1, n2/2+1) -> (n0, n1/p, n2/2+1)
      KokkosFFT::Distributed::Impl::TplPlan plan_2_1_ax102(
          exec, u_2, u_hat_1_ax102, ax102, topology2, topology1,
          MPI_COMM_WORLD);
      plan_2_1_ax102.forward(u_2, u_hat_1_ax102);
      EXPECT_TRUE(
          allclose(exec, u_hat_1_ax102, ref_u_hat_1_ax102, 1.0e-5, 5.0e-5));

      plan_2_1_ax102.backward(u_hat_1_ax102, u_inv_2);
      EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 2 -> topo 1 with ax {1, 0, 2} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 2 -> topo 1 with ax = {1, 2, 0}:
      // (n0/p, n1, n2) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
      KokkosFFT::Distributed::Impl::TplPlan plan_2_1_ax120(
          exec, u_2, u_hat_1_ax120, ax120, topology2, topology1,
          MPI_COMM_WORLD);
      plan_2_1_ax120.forward(u_2, u_hat_1_ax120);
      EXPECT_TRUE(
          allclose(exec, u_hat_1_ax120, ref_u_hat_1_ax120, 1.0e-5, 5.0e-5));

      plan_2_1_ax120.backward(u_hat_1_ax120, u_inv_2);
      EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));
    }

    {
      // topo 2 -> topo 1 with ax = {2, 0, 1}:
      // (n0/p, n1, n2) -> (n0, n1/2+1, n2/p) -> (n0/p, n1/2+1, n2)
      KokkosFFT::Distributed::Impl::TplPlan plan_2_1_ax201(
          exec, u_2, u_hat_1_ax201, ax201, topology2, topology1,
          MPI_COMM_WORLD);
      plan_2_1_ax201.forward(u_2, u_hat_1_ax201);
      EXPECT_TRUE(
          allclose(exec, u_hat_1_ax201, ref_u_hat_1_ax201, 1.0e-5, 1.0e-5))
          << "Failed topo 2 -> topo 1 with ax {2, 0, 1} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_2_1_ax201.backward(u_hat_1_ax201, u_inv_2);
      EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 2 -> topo 1 with ax {2, 0, 1} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }

    {
      // topo 2 -> topo 1 with ax = {2, 1, 0}:
      // (n0/p, n1, n2) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
      KokkosFFT::Distributed::Impl::TplPlan plan_2_1_ax210(
          exec, u_2, u_hat_1_ax210, ax210, topology2, topology1,
          MPI_COMM_WORLD);
      plan_2_1_ax210.forward(u_2, u_hat_1_ax210);
      EXPECT_TRUE(allclose(exec, u_hat_1_ax210, ref_u_hat_1_ax210))
          << "Failed topo 2 -> topo 1 with ax {2, 1, 0} (nx x ny x nz): " << n0
          << " x " << n1 << " x " << n2;

      plan_2_1_ax210.backward(u_hat_1_ax210, u_inv_2);
      EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6))
          << "Failed inverse topo 2 -> topo 1 with ax {2, 1, 0} (nx x ny x "
             "nz): "
          << n0 << " x " << n1 << " x " << n2;
    }
  }
}

template <typename T, typename LayoutType>
void test_tpl3D_execute_View3D_pencil(std::size_t npx, std::size_t npy,
                                      std::size_t n0, std::size_t n1,
                                      std::size_t n2) {
  using View3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using float_type = KokkosFFT::Impl::base_floating_point_type<T>;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<float_type>***, LayoutType, execution_space>;
  using axes_type     = KokkosFFT::axis_type<3>;
  using extents_type  = std::array<std::size_t, 3>;
  using topology_type = std::array<std::size_t, 3>;

  constexpr bool is_R2C = KokkosFFT::Impl::is_real_v<T>;

  topology_type topology0{1, npx, npy}, topology1{npx, 1, npy},
      topology2{npx, npy, 1};

  const std::size_t n0h = KokkosFFT::Impl::extent_after_transform(n0, is_R2C),
                    n1h = KokkosFFT::Impl::extent_after_transform(n1, is_R2C),
                    n2h = KokkosFFT::Impl::extent_after_transform(n2, is_R2C);
  extents_type global_in_extents{n0, n1, n2},
      global_out_extents_ax0{n0h, n1, n2}, global_out_extents_ax1{n0, n1h, n2},
      global_out_extents_ax2{n0, n1, n2h};

  // All axes
  axes_type ax012{0, 1, 2}, ax021{0, 2, 1}, ax102{1, 0, 2}, ax120{1, 2, 0},
      ax201{2, 0, 1}, ax210{2, 1, 0};

  auto [in_extents_t0, in_starts_t0] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_in_extents, topology0, MPI_COMM_WORLD);
  auto [in_extents_t1, in_starts_t1] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_in_extents, topology1, MPI_COMM_WORLD);
  auto [in_extents_t2, in_starts_t2] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_in_extents, topology2, MPI_COMM_WORLD);
  auto [out_extents_t0_ax0, out_starts_t0_ax0] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax0, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax0, out_starts_t1_ax0] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax0, topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax0, out_starts_t2_ax0] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax0, topology2, MPI_COMM_WORLD);
  auto [out_extents_t0_ax1, out_starts_t0_ax1] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax1, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax1, out_starts_t1_ax1] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax1, topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax1, out_starts_t2_ax1] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax1, topology2, MPI_COMM_WORLD);
  auto [out_extents_t0_ax2, out_starts_t0_ax2] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax2, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax2, out_starts_t1_ax2] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax2, topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax2, out_starts_t2_ax2] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents_ax2, topology2, MPI_COMM_WORLD);

  // Make reference with a basic-API
  View3DType gu("gu", n0, n1, n2);
  ComplexView3DType gu_hat_ax012("gu_hat_ax012", n0, n1, n2h),
      gu_hat_ax021("gu_hat_ax021", n0, n1h, n2),
      gu_hat_ax102("gu_hat_ax102", n0, n1, n2h),
      gu_hat_ax120("gu_hat_ax120", n0h, n1, n2),
      gu_hat_ax201("gu_hat_ax201", n0, n1h, n2),
      gu_hat_ax210("gu_hat_ax210", n0h, n1, n2);

  // Data in Topology 0 (X-pencil)
  View3DType u_0("u_0",
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

  // Data in Topology 1 (Y-pencil)
  View3DType u_1("u_1",
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

  // Data in Topology 2 (Z-pencil)
  View3DType u_2("u_2",
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

  if constexpr (is_R2C) {
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
  } else {
    KokkosFFT::fftn(exec, gu, gu_hat_ax012, ax012,
                    KokkosFFT::Normalization::backward);
    KokkosFFT::fftn(exec, gu, gu_hat_ax021, ax021,
                    KokkosFFT::Normalization::backward);
    KokkosFFT::fftn(exec, gu, gu_hat_ax102, ax102,
                    KokkosFFT::Normalization::backward);
    KokkosFFT::fftn(exec, gu, gu_hat_ax120, ax120,
                    KokkosFFT::Normalization::backward);
    KokkosFFT::fftn(exec, gu, gu_hat_ax201, ax201,
                    KokkosFFT::Normalization::backward);
    KokkosFFT::fftn(exec, gu, gu_hat_ax210, ax210,
                    KokkosFFT::Normalization::backward);
  }

  // Topo 0
  Kokkos::pair<std::size_t, std::size_t> range_gu0_dim1(
      in_starts_t0.at(1), in_starts_t0.at(1) + in_extents_t0.at(1)),
      range_gu0_dim2(in_starts_t0.at(2),
                     in_starts_t0.at(2) + in_extents_t0.at(2));
  auto sub_gu_0 =
      Kokkos::subview(gu, Kokkos::ALL, range_gu0_dim1, range_gu0_dim2);
  Kokkos::deep_copy(u_0, sub_gu_0);

  // Topo 1
  Kokkos::pair<std::size_t, std::size_t> range_gu1_dim0(
      in_starts_t1.at(0), in_starts_t1.at(0) + in_extents_t1.at(0)),
      range_gu1_dim2(in_starts_t1.at(2),
                     in_starts_t1.at(2) + in_extents_t1.at(2));
  auto sub_gu_1 =
      Kokkos::subview(gu, range_gu1_dim0, Kokkos::ALL, range_gu1_dim2);
  Kokkos::deep_copy(u_1, sub_gu_1);

  // Topo 2
  Kokkos::pair<std::size_t, std::size_t> range_gu2_dim0(
      in_starts_t2.at(0), in_starts_t2.at(0) + in_extents_t2.at(0)),
      range_gu2_dim1(in_starts_t2.at(1),
                     in_starts_t2.at(1) + in_extents_t2.at(1));
  auto sub_gu_2 =
      Kokkos::subview(gu, range_gu2_dim0, range_gu2_dim1, Kokkos::ALL);
  Kokkos::deep_copy(u_2, sub_gu_2);

  // Define ranges for topology 0 (X-pencil)
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax0_dim1(
      out_starts_t0_ax0.at(1),
      out_starts_t0_ax0.at(1) + out_extents_t0_ax0.at(1)),
      range_gu_hat_0_ax0_dim2(
          out_starts_t0_ax0.at(2),
          out_starts_t0_ax0.at(2) + out_extents_t0_ax0.at(2));

  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax1_dim1(
      out_starts_t0_ax1.at(1),
      out_starts_t0_ax1.at(1) + out_extents_t0_ax1.at(1)),
      range_gu_hat_0_ax1_dim2(
          out_starts_t0_ax1.at(2),
          out_starts_t0_ax1.at(2) + out_extents_t0_ax1.at(2));

  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax2_dim1(
      out_starts_t0_ax2.at(1),
      out_starts_t0_ax2.at(1) + out_extents_t0_ax2.at(1)),
      range_gu_hat_0_ax2_dim2(
          out_starts_t0_ax2.at(2),
          out_starts_t0_ax2.at(2) + out_extents_t0_ax2.at(2));

  // Topo 0 ax = {0, 1, 2}
  auto sub_gu_hat_0_ax012 =
      Kokkos::subview(gu_hat_ax012, Kokkos::ALL, range_gu_hat_0_ax2_dim1,
                      range_gu_hat_0_ax2_dim2);
  Kokkos::deep_copy(ref_u_hat_0_ax012, sub_gu_hat_0_ax012);

  // Topo 0 ax = {0, 2, 1}
  auto sub_gu_hat_0_ax021 =
      Kokkos::subview(gu_hat_ax021, Kokkos::ALL, range_gu_hat_0_ax1_dim1,
                      range_gu_hat_0_ax1_dim2);
  Kokkos::deep_copy(ref_u_hat_0_ax021, sub_gu_hat_0_ax021);

  // Topo 0 ax = {1, 0, 2}
  auto sub_gu_hat_0_ax102 =
      Kokkos::subview(gu_hat_ax102, Kokkos::ALL, range_gu_hat_0_ax2_dim1,
                      range_gu_hat_0_ax2_dim2);
  Kokkos::deep_copy(ref_u_hat_0_ax102, sub_gu_hat_0_ax102);

  // Topo 0 ax = {1, 2, 0}
  auto sub_gu_hat_0_ax120 =
      Kokkos::subview(gu_hat_ax120, Kokkos::ALL, range_gu_hat_0_ax0_dim1,
                      range_gu_hat_0_ax0_dim2);
  Kokkos::deep_copy(ref_u_hat_0_ax120, sub_gu_hat_0_ax120);

  // Topo 0 ax = {2, 0, 1}
  auto sub_gu_hat_0_ax201 =
      Kokkos::subview(gu_hat_ax201, Kokkos::ALL, range_gu_hat_0_ax1_dim1,
                      range_gu_hat_0_ax1_dim2);
  Kokkos::deep_copy(ref_u_hat_0_ax201, sub_gu_hat_0_ax201);

  // Topo 0 ax = {2, 1, 0}
  auto sub_gu_hat_0_ax210 =
      Kokkos::subview(gu_hat_ax210, Kokkos::ALL, range_gu_hat_0_ax0_dim1,
                      range_gu_hat_0_ax0_dim2);
  Kokkos::deep_copy(ref_u_hat_0_ax210, sub_gu_hat_0_ax210);

  // Define ranges for topology 1 (Y-pencil)
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_1_ax0_dim0(
      out_starts_t1_ax0.at(0),
      out_starts_t1_ax0.at(0) + out_extents_t1_ax0.at(0)),
      range_gu_hat_1_ax0_dim2(
          out_starts_t1_ax0.at(2),
          out_starts_t1_ax0.at(2) + out_extents_t1_ax0.at(2));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_1_ax1_dim0(
      out_starts_t1_ax1.at(0),
      out_starts_t1_ax1.at(0) + out_extents_t1_ax1.at(0)),
      range_gu_hat_1_ax1_dim2(
          out_starts_t1_ax1.at(2),
          out_starts_t1_ax1.at(2) + out_extents_t1_ax1.at(2));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_1_ax2_dim0(
      out_starts_t1_ax2.at(0),
      out_starts_t1_ax2.at(0) + out_extents_t1_ax2.at(0)),
      range_gu_hat_1_ax2_dim2(
          out_starts_t1_ax2.at(2),
          out_starts_t1_ax2.at(2) + out_extents_t1_ax2.at(2));

  // Topo 1 ax = {0, 1, 2}
  auto sub_gu_hat_1_ax012 =
      Kokkos::subview(gu_hat_ax012, range_gu_hat_1_ax2_dim0, Kokkos::ALL,
                      range_gu_hat_1_ax2_dim2);
  Kokkos::deep_copy(ref_u_hat_1_ax012, sub_gu_hat_1_ax012);

  // Topo 1 ax = {0, 2, 1}
  auto sub_gu_hat_1_ax021 =
      Kokkos::subview(gu_hat_ax021, range_gu_hat_1_ax1_dim0, Kokkos::ALL,
                      range_gu_hat_1_ax1_dim2);
  Kokkos::deep_copy(ref_u_hat_1_ax021, sub_gu_hat_1_ax021);

  // Topo 1 ax = {1, 0, 2}
  auto sub_gu_hat_1_ax102 =
      Kokkos::subview(gu_hat_ax102, range_gu_hat_1_ax2_dim0, Kokkos::ALL,
                      range_gu_hat_1_ax2_dim2);
  Kokkos::deep_copy(ref_u_hat_1_ax102, sub_gu_hat_1_ax102);

  // Topo 1 ax = {1, 2, 0}
  auto sub_gu_hat_1_ax120 =
      Kokkos::subview(gu_hat_ax120, range_gu_hat_1_ax0_dim0, Kokkos::ALL,
                      range_gu_hat_1_ax0_dim2);
  Kokkos::deep_copy(ref_u_hat_1_ax120, sub_gu_hat_1_ax120);

  // Topo 1 ax = {2, 0, 1}
  auto sub_gu_hat_1_ax201 =
      Kokkos::subview(gu_hat_ax201, range_gu_hat_1_ax1_dim0, Kokkos::ALL,
                      range_gu_hat_1_ax1_dim2);
  Kokkos::deep_copy(ref_u_hat_1_ax201, sub_gu_hat_1_ax201);

  // Topo 1 ax = {2, 1, 0}
  auto sub_gu_hat_1_ax210 =
      Kokkos::subview(gu_hat_ax210, range_gu_hat_1_ax0_dim0, Kokkos::ALL,
                      range_gu_hat_1_ax0_dim2);
  Kokkos::deep_copy(ref_u_hat_1_ax210, sub_gu_hat_1_ax210);

  // Define ranges for topology 2 (Z-pencil)
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_2_ax0_dim0(
      out_starts_t2_ax0.at(0),
      out_starts_t2_ax0.at(0) + out_extents_t2_ax0.at(0)),
      range_gu_hat_2_ax0_dim1(
          out_starts_t2_ax0.at(1),
          out_starts_t2_ax0.at(1) + out_extents_t2_ax0.at(1));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_2_ax1_dim0(
      out_starts_t2_ax1.at(0),
      out_starts_t2_ax1.at(0) + out_extents_t2_ax1.at(0)),
      range_gu_hat_2_ax1_dim1(
          out_starts_t2_ax1.at(1),
          out_starts_t2_ax1.at(1) + out_extents_t2_ax1.at(1));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_2_ax2_dim0(
      out_starts_t2_ax2.at(0),
      out_starts_t2_ax2.at(0) + out_extents_t2_ax2.at(0)),
      range_gu_hat_2_ax2_dim1(
          out_starts_t2_ax2.at(1),
          out_starts_t2_ax2.at(1) + out_extents_t2_ax2.at(1));

  // Topo 2 ax = {0, 1, 2}
  auto sub_gu_hat_2_ax012 =
      Kokkos::subview(gu_hat_ax012, range_gu_hat_2_ax2_dim0,
                      range_gu_hat_2_ax2_dim1, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax012, sub_gu_hat_2_ax012);

  // Topo 2 ax = {0, 2, 1}
  auto sub_gu_hat_2_ax021 =
      Kokkos::subview(gu_hat_ax021, range_gu_hat_2_ax1_dim0,
                      range_gu_hat_2_ax1_dim1, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax021, sub_gu_hat_2_ax021);

  // Topo 2 ax = {1, 0, 2}
  auto sub_gu_hat_2_ax102 =
      Kokkos::subview(gu_hat_ax102, range_gu_hat_2_ax2_dim0,
                      range_gu_hat_2_ax2_dim1, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax102, sub_gu_hat_2_ax102);

  // Topo 2 ax = {1, 2, 0}
  auto sub_gu_hat_2_ax120 =
      Kokkos::subview(gu_hat_ax120, range_gu_hat_2_ax0_dim0,
                      range_gu_hat_2_ax0_dim1, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax120, sub_gu_hat_2_ax120);

  // Topo 2 ax = {2, 0, 1}
  auto sub_gu_hat_2_ax201 =
      Kokkos::subview(gu_hat_ax201, range_gu_hat_2_ax1_dim0,
                      range_gu_hat_2_ax1_dim1, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax201, sub_gu_hat_2_ax201);

  // Topo 2 ax = {2, 1, 0}
  auto sub_gu_hat_2_ax210 =
      Kokkos::subview(gu_hat_ax210, range_gu_hat_2_ax0_dim0,
                      range_gu_hat_2_ax0_dim1, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax210, sub_gu_hat_2_ax210);

  // For inverse transform
  Kokkos::deep_copy(ref_u_inv_0, u_0);
  Kokkos::deep_copy(ref_u_inv_1, u_1);
  Kokkos::deep_copy(ref_u_inv_2, u_2);

  // Not a pencil geometry
  if (npx == 1 || npy == 1) {
    // topology0 -> topology1
    // (n0, n1/px, n2/py) -> (n0, (n1/2+1)/p, n2)
    ASSERT_THROW(
        {
          KokkosFFT::Distributed::Impl::TplPlan plan_0_1_ax012(
              exec, u_0, u_hat_1_ax012, ax012, topology0, topology1,
              MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology0 -> topology2
    // (n0, n1, n2/p) -> (n0/p, n1/2+1, n2)
    ASSERT_THROW(
        {
          KokkosFFT::Distributed::Impl::TplPlan plan_0_2_ax012(
              exec, u_0, u_hat_2_ax012, ax012, topology0, topology2,
              MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology1 -> topology2
    // (n0, n1/p, n2) -> (n0/p, n1, n2/2+1)
    ASSERT_THROW(
        {
          KokkosFFT::Distributed::Impl::TplPlan plan_1_2_ax012(
              exec, u_1, u_hat_2_ax012, ax012, topology1, topology2,
              MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology0 -> topology0 with ax = {1, 2}
    // (n0, n1, n2/p) -> (n0, n1, (n2/2+1)/p)
    ASSERT_THROW(
        {
          KokkosFFT::Distributed::Impl::TplPlan plan_0_0_ax012(
              exec, u_0, u_hat_0_ax012, ax012, topology0, topology0,
              MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology1 -> topology1 with ax = {1, 2}
    // (n0, n1/p, n2) -> (n0, (n1/2+1)/p, n2)
    ASSERT_THROW(
        {
          KokkosFFT::Distributed::Impl::TplPlan plan_1_1_ax210(
              exec, u_1, u_hat_1_ax210, ax210, topology1, topology1,
              MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology2 -> topology2 with ax = {0, 1, 2}
    // (n0/p, n1, n2) -> (n0/p, (n1/2+1), n2)
    ASSERT_THROW(
        {
          KokkosFFT::Distributed::Impl::TplPlan plan_2_2_ax012(
              exec, u_2, u_hat_2_ax012, ax012, topology2, topology2,
              MPI_COMM_WORLD);
        },
        std::runtime_error);
  } else {
    {
      // topo 0 -> topo 2 with ax = {0, 1, 2}:
      // (n0, n1/px, n2/py) -> (n0/px, n1/py, n2/2+1)
      KokkosFFT::Distributed::Impl::TplPlan plan_0_2_ax012(
          exec, u_0, u_hat_2_ax012, ax012, topology0, topology2,
          MPI_COMM_WORLD);
      plan_0_2_ax012.forward(u_0, u_hat_2_ax012);
      EXPECT_TRUE(
          allclose(exec, u_hat_2_ax012, ref_u_hat_2_ax012, 1.0e-5, 1.0e-6))
          << "Failed forward topo 0 -> topo 2 with ax = {0, 1, 2} (nx x ny x "
             "nz) = ("
          << n0 << " x " << n1 << " x " << n2 << ") with (px, py) = (" << npx
          << ", " << npy << ")";

      plan_0_2_ax012.forward(u_0, u_hat_2_ax012);
      EXPECT_TRUE(
          allclose(exec, u_hat_2_ax012, ref_u_hat_2_ax012, 1.0e-5, 1.0e-6))
          << "Failed forward topo 0 -> topo 2 with ax = {0, 1, 2} (nx x ny x "
             "nz) = ("
          << n0 << " x " << n1 << " x " << n2 << ") with (px, py) = (" << npx
          << ", " << npy << ")";

      plan_0_2_ax012.backward(u_hat_2_ax012, u_inv_0);
      EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6))
          << "Failed backward topo 0 -> topo 2 with ax = {0, 1, 2} (nx x ny x "
             "nz) = ("
          << n0 << " x " << n1 << " x " << n2 << ") with (px, py) = (" << npx
          << ", " << npy << ")";

      plan_0_2_ax012.backward(u_hat_2_ax012, u_inv_0);
      EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6))
          << "Failed backward topo 0 -> topo 2 with ax = {0, 1, 2} (nx x ny x "
             "nz) = ("
          << n0 << " x " << n1 << " x " << n2 << ") with (px, py) = (" << npx
          << ", " << npy << ")";
    }

    {
      // topo 2 -> topo 0 with ax = {0, 1, 2}:
      // (n0/px, n1/py, n2) -> (n0, n1/px, (n2/2+1)/py)
      KokkosFFT::Distributed::Impl::TplPlan plan_2_0_ax012(
          exec, u_2, u_hat_0_ax012, ax012, topology2, topology0,
          MPI_COMM_WORLD);
      plan_2_0_ax012.forward(u_2, u_hat_0_ax012);
      EXPECT_TRUE(
          allclose(exec, u_hat_0_ax012, ref_u_hat_0_ax012, 1.0e-5, 1.0e-6))
          << "Failed forward topo 2 -> topo 0 with ax = {0, 1, 2} (nx x ny x "
             "nz) = ("
          << n0 << " x " << n1 << " x " << n2 << ") with (px, py) = (" << npx
          << ", " << npy << ")";

      plan_2_0_ax012.forward(u_2, u_hat_0_ax012);
      EXPECT_TRUE(
          allclose(exec, u_hat_0_ax012, ref_u_hat_0_ax012, 1.0e-5, 1.0e-6))
          << "Failed forward topo 2 -> topo 0 with ax = {0, 1, 2} (nx x ny x "
             "nz) = ("
          << n0 << " x " << n1 << " x " << n2 << ") with (px, py) = (" << npx
          << ", " << npy << ")";

      plan_2_0_ax012.backward(u_hat_0_ax012, u_inv_2);
      EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6))
          << "Failed backward topo 2 -> topo 0 with ax = {0, 1, 2} (nx x ny x "
             "nz) = ("
          << n0 << " x " << n1 << " x " << n2 << ") with (px, py) = (" << npx
          << ", " << npy << ")";

      plan_2_0_ax012.backward(u_hat_0_ax012, u_inv_2);
      EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6))
          << "Failed backward topo 2 -> topo 0 with ax = {0, 1, 2} (nx x ny x "
             "nz) = ("
          << n0 << " x " << n1 << " x " << n2 << ") with (px, py) = (" << npx
          << ", " << npy << ")";
    }

    {
      // topo 2 -> topo 1 with ax = {0, 1, 2}:
      // (n0/px, n1/py, n2) -> (n0/px, n1, (n2/2+1)/py)
      KokkosFFT::Distributed::Impl::TplPlan plan_2_1_ax012(
          exec, u_2, u_hat_1_ax012, ax012, topology2, topology1,
          MPI_COMM_WORLD);
      plan_2_1_ax012.forward(u_2, u_hat_1_ax012);
      EXPECT_TRUE(
          allclose(exec, u_hat_1_ax012, ref_u_hat_1_ax012, 1.0e-5, 7.0e-5))
          << "Failed forward topo 2 -> topo 1 with ax = {0, 1, 2} (nx x ny x "
             "nz) = ("
          << n0 << " x " << n1 << " x " << n2 << ") with (px, py) = (" << npx
          << ", " << npy << ")";

      plan_2_1_ax012.forward(u_2, u_hat_1_ax012);
      EXPECT_TRUE(
          allclose(exec, u_hat_1_ax012, ref_u_hat_1_ax012, 1.0e-5, 7.0e-5))
          << "Failed forward topo 2 -> topo 1 with ax = {0, 1, 2} (nx x ny x "
             "nz) = ("
          << n0 << " x " << n1 << " x " << n2 << ") with (px, py) = (" << npx
          << ", " << npy << ")";

      plan_2_1_ax012.backward(u_hat_1_ax012, u_inv_2);
      EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6))
          << "Failed backward topo 2 -> topo 1 with ax = {0, 1, 2} (nx x ny x "
             "nz) = ("
          << n0 << " x " << n1 << " x " << n2 << ") with (px, py) = (" << npx
          << ", " << npy << ")";

      plan_2_1_ax012.backward(u_hat_1_ax012, u_inv_2);
      EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6))
          << "Failed backward topo 2 -> topo 1 with ax = {0, 1, 2} (nx x ny x "
             "nz) = ("
          << n0 << " x " << n1 << " x " << n2 << ") with (px, py) = (" << npx
          << ", " << npy << ")";
    }
  }
}

/// \brief Plan operations (see run_plan_cases) for TplPlan
struct TplPlanOps : public InternalPlanOps {
  template <typename ExecutionSpace, typename InViewType, typename OutViewType,
            typename AxesType, typename InTopologyType,
            typename OutTopologyType>
  auto create(const ExecutionSpace& exec, const InViewType& in,
              const OutViewType& out, const AxesType& axes,
              const InTopologyType& in_topology,
              const OutTopologyType& out_topology, MPI_Comm comm) const {
    return KokkosFFT::Distributed::Impl::TplPlan<
        ExecutionSpace, InViewType, OutViewType, std::tuple_size_v<AxesType>>(
        exec, in, out, axes, in_topology, out_topology, comm);
  }
};

/// \brief In-place 2D FFT of 2D Views with TplPlan (slab decomposition),
/// in-place and out-of-place, against the same references. cuFFTMp copies
/// the input into its own buffer before writing the output; the rocFFT plans
/// are out-of-place and run in-place from a copy of the input (the staging
/// buffer, T3).
template <typename T, typename LayoutType>
void test_tpl2D_inplace_View2D(std::size_t nprocs) {
  using View2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using float_type = KokkosFFT::Impl::base_floating_point_type<T>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<float_type>**, LayoutType, execution_space>;
  using topology_type = std::array<std::size_t, 2>;

  const auto topologies =
      std::make_tuple(topology_type{1, nprocs}, topology_type{nprocs, 1});

  // Two cases of test_tpl2D_execute_View2D
  const Tolerance tol_bwd{1.0e-5, 1.0e-6};
  const std::vector<PlanCase<2>> cases = {
      // topo0 -> topo1 with ax = {0, 1}
      {0, 1, {0, 1}, Expect::Success, {}, tol_bwd},
      // topo1 -> topo0 with ax = {1, 0}
      {1, 0, {1, 0}, Expect::Success, {}, tol_bwd},
  };

  const std::array<std::size_t, 2> global_extents{8, 7};
  auto gu = make_global_input<View2DType>(global_extents);

  const std::string name = "Test_TplPlan.cpp test_tpl2D_inplace_View2D";
  run_plan_cases_all_placements<ComplexView2DType>(
      name, execution_space(), gu, topologies, cases, TplPlanOps{});
}

#if defined(ENABLE_TPL_CUFFT_MP)
/// \brief The NVSHMEM data buffer and the shared scratch of a cuFFTMp plan
/// have the same size on every rank, and the buffer holds the local input
/// (padded for real transforms) and the local output of this rank. With
/// (13, 7, 17) from the x-pencil to the y-pencil, the local output is larger
/// than the local input on some ranks (C2C): the case the buffer used to be
/// too small for, when it was sized from the local input only.
/// \tparam T Value type of the input (float/double or complex)
/// \param[in] npx Number of processes in the x direction
/// \param[in] npy Number of processes in the y direction
template <typename T>
void test_tpl3D_buffer_size_View3D_pencil(std::size_t npx, std::size_t npy) {
  using float_type   = KokkosFFT::Impl::base_floating_point_type<T>;
  using complex_type = Kokkos::complex<float_type>;
  using InViewType   = Kokkos::View<T***, Kokkos::LayoutRight, execution_space>;
  using OutViewType =
      Kokkos::View<complex_type***, Kokkos::LayoutRight, execution_space>;
  using PlanType =
      KokkosFFT::Distributed::Impl::ScopedCufftMpPlan<execution_space, T,
                                                      complex_type>;
  using axes_type     = KokkosFFT::axis_type<3>;
  using extents_type  = std::array<std::size_t, 3>;
  using topology_type = std::array<std::size_t, 3>;

  constexpr bool is_R2C = KokkosFFT::Impl::is_real_v<T>;
  const topology_type topology0{1, npx, npy}, topology1{npx, 1, npy};
  const axes_type axes{0, 1, 2};
  const std::size_t n0 = 13, n1 = 7, n2 = 17;
  const extents_type global_in_extents{n0, n1, n2},
      global_out_extents{n0, n1,
                         KokkosFFT::Impl::extent_after_transform(n2, is_R2C)};

  auto [in_extents, in_starts] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_in_extents, topology0, MPI_COMM_WORLD);
  auto [out_extents, out_starts] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents, topology1, MPI_COMM_WORLD);
  InViewType in(
      "in", KokkosFFT::Impl::create_layout<Kokkos::LayoutRight>(in_extents));
  OutViewType out(
      "out", KokkosFFT::Impl::create_layout<Kokkos::LayoutRight>(out_extents));

  execution_space exec;
  ASSERT_TRUE(KokkosFFT::Distributed::Impl::is_tpl_available(
      exec, in, out, axes, topology0, topology1));

  KokkosFFT::Distributed::Impl::setup<execution_space, float_type>(
      MPI_COMM_WORLD);
  [[maybe_unused]] auto [map, map_inv] =
      KokkosFFT::Impl::get_map_axes(in, axes);
  std::unique_ptr<PlanType> plan;
  KokkosFFT::Distributed::Impl::create_plan(
      exec, plan, in, out, axes, map, topology0, topology1, MPI_COMM_WORLD);

  // The same sizes on every rank (nvshmem_malloc is symmetric)
  unsigned long long sizes[2] = {plan->buffer_size(), plan->workspace_size()};
  unsigned long long min_sizes[2] = {0, 0}, max_sizes[2] = {0, 0};
  ::MPI_Allreduce(sizes, min_sizes, 2, MPI_UNSIGNED_LONG_LONG, MPI_MIN,
                  MPI_COMM_WORLD);
  ::MPI_Allreduce(sizes, max_sizes, 2, MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                  MPI_COMM_WORLD);
  EXPECT_EQ(min_sizes[0], max_sizes[0]);
  EXPECT_EQ(min_sizes[1], max_sizes[1]);

  // Large enough for the input (padded) and the output of this rank
  const std::size_t in_bytes  = in.size() * sizeof(T);
  const std::size_t out_bytes = out.size() * sizeof(complex_type);
  EXPECT_GE(plan->buffer_size(),
            KokkosFFT::Distributed::Impl::buffer_size_in_bytes(
                in, out, axes, topology0, topology1, MPI_COMM_WORLD));
  EXPECT_GE(plan->buffer_size(), in_bytes);
  EXPECT_GE(plan->buffer_size(), out_bytes);

  // The case this test is about: some rank has a larger local output
  if constexpr (!is_R2C) {
    int is_output_larger = out_bytes > in_bytes, any_output_larger = 0;
    ::MPI_Allreduce(&is_output_larger, &any_output_larger, 1, MPI_INT, MPI_LOR,
                    MPI_COMM_WORLD);
    EXPECT_TRUE(any_output_larger);
  }
}
#endif

/// \brief Permuted axes that canonical_tpl_axes maps to the identity: the
/// vendor plan is made for them, and gives the same result as with the axes
/// in increasing order. C2C: every order; R2C: {1, 0, 2} (the last axis is
/// kept). From the x-slab to the y-slab.
/// \tparam T Value type of the input (float/double or complex)
/// \param[in] nprocs Number of processes in the MPI communicator
template <typename T>
void test_tpl3D_permuted_axes_View3D(std::size_t nprocs) {
  using float_type   = KokkosFFT::Impl::base_floating_point_type<T>;
  using complex_type = Kokkos::complex<float_type>;
  using InViewType   = Kokkos::View<T***, Kokkos::LayoutRight, execution_space>;
  using OutViewType =
      Kokkos::View<complex_type***, Kokkos::LayoutRight, execution_space>;
  using axes_type     = KokkosFFT::axis_type<3>;
  using extents_type  = std::array<std::size_t, 3>;
  using topology_type = std::array<std::size_t, 3>;

  constexpr bool is_R2C = KokkosFFT::Impl::is_real_v<T>;
  const topology_type topology0{1, 1, nprocs}, topology1{1, nprocs, 1};
  const std::size_t n0 = 8, n1 = 7, n2 = 5;
  const extents_type global_in_extents{n0, n1, n2},
      global_out_extents{n0, n1,
                         KokkosFFT::Impl::extent_after_transform(n2, is_R2C)};

  auto [in_extents, in_starts] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_in_extents, topology0, MPI_COMM_WORLD);
  auto [out_extents, out_starts] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents, topology1, MPI_COMM_WORLD);
  const auto in_layout =
      KokkosFFT::Impl::create_layout<Kokkos::LayoutRight>(in_extents);
  const auto out_layout =
      KokkosFFT::Impl::create_layout<Kokkos::LayoutRight>(out_extents);

  InViewType u("u", in_layout), u_ref("u_ref", in_layout),
      u_inv("u_inv", in_layout);
  OutViewType u_hat("u_hat", out_layout), u_hat_ref("u_hat_ref", out_layout);
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::fill_random(u_ref, random_pool, 1.0);

  execution_space exec;
  const axes_type ax012{0, 1, 2};
  Kokkos::deep_copy(u, u_ref);
  KokkosFFT::Distributed::Impl::TplPlan plan_ref(
      exec, u, u_hat_ref, ax012, topology0, topology1, MPI_COMM_WORLD);
  plan_ref.forward(u, u_hat_ref);

  const std::vector<axes_type> permuted_axes =
      is_R2C ? std::vector<axes_type>{{1, 0, 2}}
             : std::vector<axes_type>{
                   {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  for (const auto& axes : permuted_axes) {
    EXPECT_TRUE(KokkosFFT::Distributed::Impl::is_tpl_available(
        exec, u, u_hat, axes, topology0, topology1));

    // The input of an out-of-place transform may be overwritten
    Kokkos::deep_copy(u, u_ref);
    KokkosFFT::Distributed::Impl::TplPlan plan(exec, u, u_hat, axes, topology0,
                                               topology1, MPI_COMM_WORLD);
    plan.forward(u, u_hat);
    EXPECT_TRUE(allclose(exec, u_hat, u_hat_ref));

    plan.backward(u_hat, u_inv);
    EXPECT_TRUE(allclose(exec, u_inv, u_ref, 1.0e-5, 1.0e-6));
  }
}

/// \brief transpose_and_scale against KokkosFFT::Impl::transpose followed by
/// a scaling pass: the identity and a permuted map, a layout change, a
/// destination smaller than the source (the cropped R2C padding), and a
/// coefficient of 1 (the plain copy)
/// \tparam T Value type (float/double or complex)
/// \tparam InLayoutType Layout of the source
/// \tparam OutLayoutType Layout of the destination
template <typename T, typename InLayoutType, typename OutLayoutType>
void test_transpose_and_scale() {
  using InViewType   = Kokkos::View<T***, InLayoutType, execution_space>;
  using OutViewType  = Kokkos::View<T***, OutLayoutType, execution_space>;
  using map_type     = std::array<int, 3>;
  using extents_type = std::array<std::size_t, 3>;

  execution_space exec;
  const extents_type in_extents{4, 5, 7};
  InViewType in("in", KokkosFFT::Impl::create_layout<InLayoutType>(in_extents));
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::fill_random(in, random_pool, 1.0);

  for (const auto& map : {map_type{0, 1, 2}, map_type{2, 0, 1}}) {
    for (const bool is_cropped : {false, true}) {
      for (const double coef : {1.0, 0.25}) {
        extents_type out_extents{};
        for (std::size_t i = 0; i < 3; ++i) {
          out_extents.at(i) = in_extents.at(static_cast<std::size_t>(map[i]));
        }
        if (is_cropped) out_extents.at(2) -= 2;
        const auto out_layout =
            KokkosFFT::Impl::create_layout<OutLayoutType>(out_extents);
        OutViewType out("out", out_layout), ref("ref", out_layout);

        KokkosFFT::Impl::transpose(exec, in, ref, map, true);
        using float_type = KokkosFFT::Impl::base_floating_point_type<T>;
        KokkosFFT::Impl::normalize_impl(exec, ref,
                                        static_cast<float_type>(coef));
        KokkosFFT::Distributed::Impl::transpose_and_scale(exec, in, out, map,
                                                          coef);
        EXPECT_TRUE(allclose(exec, out, ref))
            << "map {" << map[0] << ", " << map[1] << ", " << map[2]
            << "}, cropped " << is_cropped << ", coef " << coef;
      }
    }
  }
}

/// \brief TplPlan with every normalization: the forward transform against
/// the single-node FFT with the same normalization, and the backward
/// transform back to the input. With cuFFTMp, the normalization is applied
/// in the copy out of the NVSHMEM buffer (scales_in_transform). From the
/// x-slab to the y-slab, the axes in increasing and reversed order (a
/// transpose for LayoutRight R2C, the reversed View for LayoutLeft).
/// \tparam T Value type of the input (float/double or complex)
/// \tparam LayoutType Layout of the Views
/// \param[in] nprocs Number of processes in the MPI communicator
template <typename T, typename LayoutType>
void test_tpl3D_normalization_View3D(std::size_t nprocs) {
  using float_type   = KokkosFFT::Impl::base_floating_point_type<T>;
  using complex_type = Kokkos::complex<float_type>;
  using InViewType   = Kokkos::View<T***, LayoutType, execution_space>;
  using OutViewType =
      Kokkos::View<complex_type***, LayoutType, execution_space>;
  using axes_type     = KokkosFFT::axis_type<3>;
  using topology_type = std::array<std::size_t, 3>;

  constexpr bool is_R2C = KokkosFFT::Impl::is_real_v<T>;
  const topology_type topology0{1, 1, nprocs}, topology1{1, nprocs, 1};
  const std::array<std::size_t, 3> global_extents{8, 7, 5};
  const MPI_Comm comm = MPI_COMM_WORLD;
  execution_space exec;

  auto gu              = make_global_input<InViewType>(global_extents);
  auto u_ref           = extract_local(gu, topology0, comm);
  const auto in_layout = KokkosFFT::Impl::create_layout<LayoutType>(
      KokkosFFT::Impl::extract_extents(u_ref));

  for (const auto& axes : {axes_type{0, 1, 2}, axes_type{2, 1, 0}}) {
    for (const auto norm :
         {KokkosFFT::Normalization::forward, KokkosFFT::Normalization::backward,
          KokkosFFT::Normalization::ortho, KokkosFFT::Normalization::none}) {
      auto out_extents = global_out_extents(global_extents, axes, is_R2C);
      OutViewType gu_hat(
          "gu_hat", KokkosFFT::Impl::create_layout<LayoutType>(out_extents));
      if constexpr (is_R2C) {
        KokkosFFT::rfftn(exec, gu, gu_hat, axes, norm);
      } else {
        KokkosFFT::fftn(exec, gu, gu_hat, axes, norm);
      }
      auto ref_u_hat = extract_local(gu_hat, topology1, comm);

      InViewType u("u", in_layout), u_inv("u_inv", in_layout);
      OutViewType u_hat("u_hat", ref_u_hat.layout());
      Kokkos::deep_copy(u, u_ref);

      KokkosFFT::Distributed::Impl::TplPlan plan(
          exec, u, u_hat, axes, topology0, topology1, comm, norm);
      plan.forward(u, u_hat);
      EXPECT_TRUE(allclose(exec, u_hat, ref_u_hat, 1.0e-5, 1.0e-6))
          << "axes {" << axes[0] << ", " << axes[1] << ", " << axes[2]
          << "}, normalization " << static_cast<int>(norm);

      // Without normalization, the backward transform of the forward one is
      // N times the input (N = n0 * n1 * n2): scaled back by 1/N, so that
      // the comparison is the same as for the other normalizations
      plan.backward(u_hat, u_inv);
      if (norm == KokkosFFT::Normalization::none) {
        const std::size_t n = KokkosFFT::Impl::total_size(global_extents);
        KokkosFFT::Impl::normalize_impl(
            exec, u_inv, float_type(1) / static_cast<float_type>(n));
      }
      EXPECT_TRUE(allclose(exec, u_inv, u_ref, 1.0e-5, 1.0e-6))
          << "inverse, axes {" << axes[0] << ", " << axes[1] << ", " << axes[2]
          << "}, normalization " << static_cast<int>(norm);
    }
  }
}

/// \brief Whether is_tpl_available accepts a plan, for the local Views of
/// this rank
/// \tparam InViewType Type of the input View
/// \tparam OutViewType Type of the output View
/// \param[in] global_extents Global extents of the input
/// \param[in] in_topology Input topology (std::array or Topology)
/// \param[in] out_topology Output topology (std::array or Topology)
/// \param[in] axes FFT axes
template <typename InViewType, typename OutViewType, std::size_t DIM,
          typename InTopologyType, typename OutTopologyType>
bool is_tpl_available_for(const std::array<std::size_t, DIM>& global_extents,
                          const InTopologyType& in_topology,
                          const OutTopologyType& out_topology,
                          const KokkosFFT::axis_type<DIM>& axes) {
  using LayoutType = typename InViewType::array_layout;
  constexpr bool is_R2C =
      KokkosFFT::Impl::is_real_v<typename InViewType::non_const_value_type>;
  auto [in_extents, in_starts] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_extents, in_topology, MPI_COMM_WORLD);
  auto [out_extents, out_starts] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents(global_extents, axes, is_R2C), out_topology,
          MPI_COMM_WORLD);
  InViewType in("in", KokkosFFT::Impl::create_layout<LayoutType>(in_extents));
  OutViewType out("out",
                  KokkosFFT::Impl::create_layout<LayoutType>(out_extents));
  return KokkosFFT::Distributed::Impl::is_tpl_available(
      execution_space(), in, out, axes, in_topology, out_topology);
}

/// \brief Every order of the axes 0, ..., DIM - 1
template <std::size_t DIM>
std::vector<KokkosFFT::axis_type<DIM>> all_axes_orders() {
  KokkosFFT::axis_type<DIM> axes{};
  std::iota(axes.begin(), axes.end(), 0);
  std::vector<KokkosFFT::axis_type<DIM>> orders;
  do {
    orders.push_back(axes);
  } while (std::next_permutation(axes.begin(), axes.end()));
  return orders;
}

/// \brief The layout of a topology type: that of a Topology, LayoutRight for
/// a std::array
template <typename TopologyType>
struct topology_layout {
  using type = Kokkos::LayoutRight;
};

template <std::size_t N, typename LayoutType>
struct topology_layout<
    KokkosFFT::Distributed::Topology<std::size_t, N, LayoutType>> {
  using type = LayoutType;
};

/// \brief Plan operations (see run_plan_cases) for the plan of the public
/// API (plan_factory), which must choose the vendor plan
struct TplPlanFactoryOps {
  template <typename ExecutionSpace, typename InViewType, typename OutViewType,
            typename AxesType, typename InTopologyType,
            typename OutTopologyType>
  auto create(const ExecutionSpace& exec, const InViewType& in,
              const OutViewType& out, const AxesType& axes,
              const InTopologyType& in_topology,
              const OutTopologyType& out_topology, MPI_Comm comm) const {
    return KokkosFFT::Distributed::Plan<
        ExecutionSpace, InViewType, OutViewType, std::tuple_size_v<AxesType>,
        typename topology_layout<InTopologyType>::type,
        typename topology_layout<OutTopologyType>::type>(
        exec, in, out, axes, in_topology, out_topology, comm);
  }

  template <typename PlanType, typename InViewType, typename OutViewType>
  void forward(const PlanType& plan, const InViewType& in,
               const OutViewType& out) const {
    KokkosFFT::Distributed::execute(plan, in, out,
                                    KokkosFFT::Direction::forward);
  }

  template <typename PlanType, typename InViewType, typename OutViewType>
  void backward(const PlanType& plan, const OutViewType& out,
                const InViewType& in) const {
    KokkosFFT::Distributed::execute(plan, out, in,
                                    KokkosFFT::Direction::backward);
  }

  template <typename PlanType>
  void check(const PlanType& plan) const {
#if defined(ENABLE_TPL_CUFFT_MP)
    EXPECT_EQ(plan.name(), "cufftMpPlan");
#else
    EXPECT_EQ(plan.name(), "rocFFTMPIPlan");
#endif
  }
};

/// \brief Vendor plans between pairs of topologies, for the given orders of
/// the axes, out-of-place and in-place, against the single-node FFT.
/// is_tpl_available must accept every case, so that plan_factory chooses the
/// vendor plan.
/// \tparam T Value type of the input (float/double or complex)
/// \tparam LayoutType Layout of the Views
/// \param[in] name Name of the test, printed with the cases
/// \param[in] global_extents Global extents of the input
/// \param[in] topologies Topologies (std::array or Topology, in a tuple or an
/// array), indexed by the pairs
/// \param[in] pairs Pairs of indices of the input and output topologies
/// \param[in] axes_list Orders of the axes
/// \param[in] ops Plan operations: TplPlanOps or TplPlanFactoryOps
template <typename T, typename LayoutType, std::size_t DIM,
          typename TopologiesType, typename PlanOps>
void test_tpl_plan_cases(
    const std::string& name, const std::array<std::size_t, DIM>& global_extents,
    const TopologiesType& topologies,
    const std::vector<std::pair<std::size_t, std::size_t>>& pairs,
    const std::vector<KokkosFFT::axis_type<DIM>>& axes_list,
    const PlanOps& ops) {
  using float_type   = KokkosFFT::Impl::base_floating_point_type<T>;
  using complex_type = Kokkos::complex<float_type>;
  using InViewType   = Kokkos::View<KokkosFFT::Impl::add_pointer_n_t<T, DIM>,
                                  LayoutType, execution_space>;
  using OutViewType =
      Kokkos::View<KokkosFFT::Impl::add_pointer_n_t<complex_type, DIM>,
                   LayoutType, execution_space>;

  const Tolerance tol_fwd{1.0e-5, 1.0e-5}, tol_bwd{1.0e-5, 1.0e-6};
  std::vector<PlanCase<DIM>> cases;
  for (const auto& pair : pairs) {
    const std::size_t in_topo = pair.first, out_topo = pair.second;
    for (const auto& axes : axes_list) {
      PlanCase<DIM> plan_case{in_topo,         out_topo, axes,
                              Expect::Success, tol_fwd,  tol_bwd};
      visit_topology(topologies, in_topo, [&](const auto& in_topology) {
        visit_topology(topologies, out_topo, [&](const auto& out_topology) {
          EXPECT_TRUE((is_tpl_available_for<InViewType, OutViewType>(
              global_extents, in_topology, out_topology, axes)))
              << name << ": " << to_string(plan_case);
        });
      });
      cases.push_back(plan_case);
    }
  }

  auto gu = make_global_input<InViewType>(global_extents);
  run_plan_cases_all_placements<OutViewType>(name, execution_space(), gu,
                                             topologies, cases, ops);
}

/// \brief The same distribution in and out (T5): a vendor plan from each
/// topology to itself, for every order of the axes (cuFFTMp: identical
/// boxes, T6; rocFFT: identical fields, T2r, every order through its buffers,
/// T4a)
/// \tparam T Value type of the input (float/double or complex)
/// \tparam LayoutType Layout of the Views
/// \param[in] name Name of the test, printed with the cases
/// \param[in] global_extents Global extents of the input
/// \param[in] topologies Topologies, each used for the input and the output
template <typename T, typename LayoutType, std::size_t DIM,
          std::size_t NTopologies>
void test_tpl_same_distribution(
    const std::string& name, const std::array<std::size_t, DIM>& global_extents,
    const std::array<std::array<std::size_t, DIM>, NTopologies>& topologies) {
  std::vector<std::pair<std::size_t, std::size_t>> pairs;
  for (std::size_t i = 0; i < NTopologies; ++i) pairs.emplace_back(i, i);
  test_tpl_plan_cases<T, LayoutType>(name, global_extents, topologies, pairs,
                                     all_axes_orders<DIM>(), TplPlanOps{});
}

}  // namespace

TYPED_TEST_SUITE(TestTplPlan1D, test_types);
TYPED_TEST_SUITE(TestTplPlan2D, test_types);
TYPED_TEST_SUITE(TestTplPlan3D, test_types);

TYPED_TEST(TestTplPlan1D, IsAvailableView2D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_tpl1D_is_available<float_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestTplPlan2D, IsAvailableView2D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_tpl2D_is_available_View2D<float_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestTplPlan2D, IsAvailableView3D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_tpl2D_is_available_View3D<float_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestTplPlan3D, IsAvailableView3D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_tpl3D_is_available_View3D<float_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestTplPlan2D, ExecuteView2D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  const std::size_t n0_even = 16, n0_odd = 15;
  const std::size_t n1_even = 8, n1_odd = 9;
  test_tpl2D_execute_View2D<float_type, layout_type>(this->m_nprocs, n0_even,
                                                     n1_even);
  test_tpl2D_execute_View2D<float_type, layout_type>(this->m_nprocs, n0_even,
                                                     n1_odd);
  test_tpl2D_execute_View2D<float_type, layout_type>(this->m_nprocs, n0_odd,
                                                     n1_even);
  test_tpl2D_execute_View2D<float_type, layout_type>(this->m_nprocs, n0_odd,
                                                     n1_odd);
}

TYPED_TEST(TestTplPlan2D, ExecuteView2D_C2C) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;

  const std::size_t n0_even = 16, n0_odd = 15;
  const std::size_t n1_even = 8, n1_odd = 9;
  test_tpl2D_execute_View2D<complex_type, layout_type>(this->m_nprocs, n0_even,
                                                       n1_even);
  test_tpl2D_execute_View2D<complex_type, layout_type>(this->m_nprocs, n0_even,
                                                       n1_odd);
  test_tpl2D_execute_View2D<complex_type, layout_type>(this->m_nprocs, n0_odd,
                                                       n1_even);
  test_tpl2D_execute_View2D<complex_type, layout_type>(this->m_nprocs, n0_odd,
                                                       n1_odd);
}

TYPED_TEST(TestTplPlan3D, ExecuteView3D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  const std::size_t n0_even = 16, n0_odd = 15;
  const std::size_t n1_even = 8, n1_odd = 9;
  const std::size_t n2_even = 12, n2_odd = 11;

  test_tpl3D_execute_View3D<float_type, layout_type>(this->m_nprocs, n0_even,
                                                     n1_even, n2_even);
  test_tpl3D_execute_View3D<float_type, layout_type>(this->m_nprocs, n0_even,
                                                     n1_even, n2_odd);
  test_tpl3D_execute_View3D<float_type, layout_type>(this->m_nprocs, n0_even,
                                                     n1_odd, n2_even);
  test_tpl3D_execute_View3D<float_type, layout_type>(this->m_nprocs, n0_even,
                                                     n1_odd, n2_odd);
  test_tpl3D_execute_View3D<float_type, layout_type>(this->m_nprocs, n0_odd,
                                                     n1_even, n2_even);
  test_tpl3D_execute_View3D<float_type, layout_type>(this->m_nprocs, n0_odd,
                                                     n1_even, n2_odd);
  test_tpl3D_execute_View3D<float_type, layout_type>(this->m_nprocs, n0_odd,
                                                     n1_odd, n2_even);
  test_tpl3D_execute_View3D<float_type, layout_type>(this->m_nprocs, n0_odd,
                                                     n1_odd, n2_odd);
}

TYPED_TEST(TestTplPlan3D, ExecuteView3D_C2C) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;

  const std::size_t n0_even = 16, n0_odd = 15;
  const std::size_t n1_even = 8, n1_odd = 9;
  const std::size_t n2_even = 12, n2_odd = 11;

  test_tpl3D_execute_View3D<complex_type, layout_type>(this->m_nprocs, n0_even,
                                                       n1_even, n2_even);
  test_tpl3D_execute_View3D<complex_type, layout_type>(this->m_nprocs, n0_even,
                                                       n1_even, n2_odd);
  test_tpl3D_execute_View3D<complex_type, layout_type>(this->m_nprocs, n0_even,
                                                       n1_odd, n2_even);
  test_tpl3D_execute_View3D<complex_type, layout_type>(this->m_nprocs, n0_even,
                                                       n1_odd, n2_odd);
  test_tpl3D_execute_View3D<complex_type, layout_type>(this->m_nprocs, n0_odd,
                                                       n1_even, n2_even);
  test_tpl3D_execute_View3D<complex_type, layout_type>(this->m_nprocs, n0_odd,
                                                       n1_even, n2_odd);
  test_tpl3D_execute_View3D<complex_type, layout_type>(this->m_nprocs, n0_odd,
                                                       n1_odd, n2_even);
  test_tpl3D_execute_View3D<complex_type, layout_type>(this->m_nprocs, n0_odd,
                                                       n1_odd, n2_odd);
}

TYPED_TEST(TestTplPlan3D, ExecuteView3D_Pencil_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  if (this->m_nprocs == 1 || this->m_npx * this->m_npx != this->m_nprocs) {
    GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                    "for this test";
  }

  const std::size_t n0_even = 14, n0_odd = 13;
  const std::size_t n1_even = 10, n1_odd = 7;
  const std::size_t n2_even = 18, n2_odd = 17;

  test_tpl3D_execute_View3D_pencil<float_type, layout_type>(
      this->m_npx, this->m_npx, n0_even, n1_even, n2_even);
  test_tpl3D_execute_View3D_pencil<float_type, layout_type>(
      this->m_npx, this->m_npx, n0_even, n1_even, n2_odd);
  test_tpl3D_execute_View3D_pencil<float_type, layout_type>(
      this->m_npx, this->m_npx, n0_even, n1_odd, n2_even);
  test_tpl3D_execute_View3D_pencil<float_type, layout_type>(
      this->m_npx, this->m_npx, n0_even, n1_odd, n2_odd);
  test_tpl3D_execute_View3D_pencil<float_type, layout_type>(
      this->m_npx, this->m_npx, n0_odd, n1_even, n2_even);
  test_tpl3D_execute_View3D_pencil<float_type, layout_type>(
      this->m_npx, this->m_npx, n0_odd, n1_even, n2_odd);
  test_tpl3D_execute_View3D_pencil<float_type, layout_type>(
      this->m_npx, this->m_npx, n0_odd, n1_odd, n2_even);
  test_tpl3D_execute_View3D_pencil<float_type, layout_type>(
      this->m_npx, this->m_npx, n0_odd, n1_odd, n2_odd);
}

TYPED_TEST(TestTplPlan3D, ExecuteView3D_Pencil_C2C) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;
  if (this->m_nprocs == 1 || this->m_npx * this->m_npx != this->m_nprocs) {
    GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                    "for this test";
  }
  const std::size_t n0_even = 14, n0_odd = 13;
  const std::size_t n1_even = 10, n1_odd = 7;
  const std::size_t n2_even = 18, n2_odd = 17;

  test_tpl3D_execute_View3D_pencil<complex_type, layout_type>(
      this->m_npx, this->m_npx, n0_even, n1_even, n2_even);
  test_tpl3D_execute_View3D_pencil<complex_type, layout_type>(
      this->m_npx, this->m_npx, n0_even, n1_even, n2_odd);
  test_tpl3D_execute_View3D_pencil<complex_type, layout_type>(
      this->m_npx, this->m_npx, n0_even, n1_odd, n2_even);
  test_tpl3D_execute_View3D_pencil<complex_type, layout_type>(
      this->m_npx, this->m_npx, n0_even, n1_odd, n2_odd);
  test_tpl3D_execute_View3D_pencil<complex_type, layout_type>(
      this->m_npx, this->m_npx, n0_odd, n1_even, n2_even);
  test_tpl3D_execute_View3D_pencil<complex_type, layout_type>(
      this->m_npx, this->m_npx, n0_odd, n1_even, n2_odd);
  test_tpl3D_execute_View3D_pencil<complex_type, layout_type>(
      this->m_npx, this->m_npx, n0_odd, n1_odd, n2_even);
  test_tpl3D_execute_View3D_pencil<complex_type, layout_type>(
      this->m_npx, this->m_npx, n0_odd, n1_odd, n2_odd);
}

#if defined(ENABLE_TPL_CUFFT_MP)
TYPED_TEST(TestTplPlan3D, BufferSizeView3D_Pencil) {
  using float_type   = typename TestFixture::float_type;
  using complex_type = Kokkos::complex<float_type>;
  if (this->m_nprocs == 1 || this->m_npx * this->m_npx != this->m_nprocs) {
    GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                    "for this test";
  }
  test_tpl3D_buffer_size_View3D_pencil<float_type>(this->m_npx, this->m_npx);
  test_tpl3D_buffer_size_View3D_pencil<complex_type>(this->m_npx, this->m_npx);
}
#endif

TEST(TestTplAxes, canonical_tpl_axes) { test_canonical_tpl_axes(); }

TEST(TestScaledCopy, transpose_and_scale) {
  using LL = Kokkos::LayoutLeft;
  using LR = Kokkos::LayoutRight;
  test_transpose_and_scale<float, LR, LR>();
  test_transpose_and_scale<double, LL, LR>();
  test_transpose_and_scale<Kokkos::complex<float>, LR, LL>();
  test_transpose_and_scale<Kokkos::complex<double>, LL, LL>();
}

TYPED_TEST(TestTplPlan3D, NormalizationView3D) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;
  if (this->m_nprocs == 1) {
    GTEST_SKIP() << "The slab plans need more than one process";
  }
  test_tpl3D_normalization_View3D<float_type, layout_type>(this->m_nprocs);
  test_tpl3D_normalization_View3D<complex_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestTplPlan3D, PermutedAxesView3D) {
  using float_type   = typename TestFixture::float_type;
  using complex_type = Kokkos::complex<float_type>;
  if (this->m_nprocs == 1) {
    GTEST_SKIP() << "The slab plans need more than one process";
  }
  test_tpl3D_permuted_axes_View3D<float_type>(this->m_nprocs);
  test_tpl3D_permuted_axes_View3D<complex_type>(this->m_nprocs);
}

TYPED_TEST(TestTplPlan3D, ReversedTopology) {
  test_reversed_topology<Kokkos::LayoutRight>(this->m_nprocs);
  test_reversed_topology<Kokkos::LayoutLeft>(this->m_nprocs);
}

TYPED_TEST(TestTplPlan3D, LayoutLeftView3D) {
  using float_type   = typename TestFixture::float_type;
  using complex_type = Kokkos::complex<float_type>;
  if (this->m_nprocs == 1) {
    GTEST_SKIP() << "The slab plans need more than one process";
  }
  test_tpl3D_layout_left_View3D<float_type>(this->m_nprocs);
  test_tpl3D_layout_left_View3D<complex_type>(this->m_nprocs);
}

TYPED_TEST(TestTplPlan2D, InPlaceView2D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  if (this->m_nprocs == 1) {
    GTEST_SKIP() << "TplPlan needs a slab decomposition (2 processes or more)";
  }
  test_tpl2D_inplace_View2D<float_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestTplPlan2D, InPlaceView2D_C2C) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;
  if (this->m_nprocs == 1) {
    GTEST_SKIP() << "TplPlan needs a slab decomposition (2 processes or more)";
  }
  test_tpl2D_inplace_View2D<complex_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestTplPlan2D, SameDistributionView2D) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;
  if (this->m_nprocs == 1) {
    GTEST_SKIP() << "The slab plans need more than one process";
  }
  const std::size_t p                                        = this->m_nprocs;
  const std::array<std::array<std::size_t, 2>, 2> topologies = {
      {{1, p}, {p, 1}}};
  const std::array<std::size_t, 2> global_extents{8, 7};
  const std::string name = "Test_TplPlan.cpp test_tpl_same_distribution View2D";
  test_tpl_same_distribution<float_type, layout_type>(name, global_extents,
                                                      topologies);
  test_tpl_same_distribution<complex_type, layout_type>(name, global_extents,
                                                        topologies);
}

TYPED_TEST(TestTplPlan3D, SameDistributionView3D) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;
  if (this->m_nprocs == 1) {
    GTEST_SKIP() << "The slab plans need more than one process";
  }
  const std::size_t p                                        = this->m_nprocs;
  const std::array<std::array<std::size_t, 3>, 3> topologies = {
      {{1, 1, p}, {1, p, 1}, {p, 1, 1}}};
  const std::array<std::size_t, 3> global_extents{13, 10, 17};
  const std::string name = "Test_TplPlan.cpp test_tpl_same_distribution View3D";
  test_tpl_same_distribution<float_type, layout_type>(name, global_extents,
                                                      topologies);
  test_tpl_same_distribution<complex_type, layout_type>(name, global_extents,
                                                        topologies);
}

TYPED_TEST(TestTplPlan3D, SameDistributionView3D_Pencil) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;
  if (this->m_nprocs == 1 || this->m_npx * this->m_npx != this->m_nprocs) {
    GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                    "for this test";
  }
  const std::size_t px                                       = this->m_npx;
  const std::array<std::array<std::size_t, 3>, 3> topologies = {
      {{1, px, px}, {px, 1, px}, {px, px, 1}}};
  const std::array<std::size_t, 3> global_extents{13, 10, 17};
  const std::string name =
      "Test_TplPlan.cpp test_tpl_same_distribution View3D pencil";
  test_tpl_same_distribution<float_type, layout_type>(name, global_extents,
                                                      topologies);
  test_tpl_same_distribution<complex_type, layout_type>(name, global_extents,
                                                        topologies);
}

TYPED_TEST(TestTplPlan3D, PermutedAxesView3D_Pencil) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;
  if (this->m_nprocs == 1 || this->m_npx * this->m_npx != this->m_nprocs) {
    GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                    "for this test";
  }
  // Every order of the axes from a pencil to another one, which need a
  // transpose in the copies of cuFFTMp (T6) or the buffers of rocFFT (T4a)
  // for most of them
  const std::size_t px                                       = this->m_npx;
  const std::array<std::array<std::size_t, 3>, 3> topologies = {
      {{1, px, px}, {px, 1, px}, {px, px, 1}}};
  const std::vector<std::pair<std::size_t, std::size_t>> pairs = {
      {0, 1}, {1, 2}, {2, 0}};
  const std::array<std::size_t, 3> global_extents{13, 10, 17};
  const std::string name =
      "Test_TplPlan.cpp test_tpl_plan_cases View3D pencil permuted axes";
  test_tpl_plan_cases<float_type, layout_type>(name, global_extents, topologies,
                                               pairs, all_axes_orders<3>(),
                                               TplPlanOps{});
  test_tpl_plan_cases<complex_type, layout_type>(
      name, global_extents, topologies, pairs, all_axes_orders<3>(),
      TplPlanOps{});
}

TYPED_TEST(TestTplPlan3D, LayoutLeftTopologyView3D_Pencil) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;
  using topology_r_type =
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutRight>;
  using topology_l_type =
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutLeft>;
  if (this->m_nprocs == 1 || this->m_npx * this->m_npx != this->m_nprocs) {
    GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                    "for this test";
  }
  // LayoutLeft topologies (the rank order of the process grid reversed) with
  // two distributed dimensions, through plan_factory, which must choose the
  // vendor plan: the case T2c did not cover. LayoutLeft to LayoutLeft, to
  // and from LayoutRight, and to itself.
  const std::size_t px = this->m_npx;
  const auto topologies =
      std::make_tuple(topology_l_type{1, px, px}, topology_l_type{px, 1, px},
                      topology_r_type{px, px, 1}, topology_l_type{px, px, 1});
  const std::vector<std::pair<std::size_t, std::size_t>> pairs = {
      {0, 1}, {0, 2}, {2, 1}, {1, 3}, {3, 3}};
  const std::vector<KokkosFFT::axis_type<3>> axes_list = {
      {0, 1, 2}, {0, 2, 1}, {2, 1, 0}};
  const std::array<std::size_t, 3> global_extents{13, 10, 17};
  const std::string name =
      "Test_TplPlan.cpp test_tpl_plan_cases View3D LayoutLeft topologies";
  test_tpl_plan_cases<float_type, layout_type>(
      name, global_extents, topologies, pairs, axes_list, TplPlanFactoryOps{});
  test_tpl_plan_cases<complex_type, layout_type>(
      name, global_extents, topologies, pairs, axes_list, TplPlanFactoryOps{});
}

#if defined(ENABLE_TPL_ROCFFT_MPI)
TYPED_TEST(TestTplPlan2D, MixedDecompositionView2D) {
  using float_type     = typename TestFixture::float_type;
  using layout_type    = typename TestFixture::layout_type;
  using complex_type   = Kokkos::complex<float_type>;
  const std::size_t p  = this->m_nprocs;
  const std::size_t px = static_cast<std::size_t>(std::sqrt(p));
  if (p == 1 || px * px != p) {
    GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                    "for this test";
  }
  // rocFFT (R4, T3): a 2D View distributed along both dimensions, to and from
  // slabs and to itself, which are neither slab nor pencil pairs
  const std::array<std::array<std::size_t, 2>, 3> topologies = {
      {{px, px}, {1, p}, {p, 1}}};
  const std::vector<std::pair<std::size_t, std::size_t>> pairs = {
      {0, 1}, {2, 0}, {0, 0}};
  const std::array<std::size_t, 2> global_extents{8, 7};
  const std::string name =
      "Test_TplPlan.cpp test_tpl_plan_cases View2D mixed decompositions";
  test_tpl_plan_cases<float_type, layout_type>(name, global_extents, topologies,
                                               pairs, all_axes_orders<2>(),
                                               TplPlanOps{});
  test_tpl_plan_cases<complex_type, layout_type>(
      name, global_extents, topologies, pairs, all_axes_orders<2>(),
      TplPlanOps{});
}

TYPED_TEST(TestTplPlan3D, MixedDecompositionView3D) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;
  if (this->m_nprocs == 1 || this->m_npx * this->m_npx != this->m_nprocs) {
    GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                    "for this test";
  }
  // rocFFT (R4, T3): from slabs to pencils and back, which are neither slab
  // nor pencil pairs
  const std::size_t px = this->m_npx, p = this->m_nprocs;
  const std::array<std::array<std::size_t, 3>, 4> topologies = {
      {{1, 1, p}, {1, px, px}, {px, px, 1}, {p, 1, 1}}};
  const std::vector<std::pair<std::size_t, std::size_t>> pairs = {
      {0, 1}, {2, 3}, {1, 3}, {3, 2}};
  const std::array<std::size_t, 3> global_extents{13, 10, 17};
  const std::string name =
      "Test_TplPlan.cpp test_tpl_plan_cases View3D mixed decompositions";
  test_tpl_plan_cases<float_type, layout_type>(name, global_extents, topologies,
                                               pairs, all_axes_orders<3>(),
                                               TplPlanOps{});
  test_tpl_plan_cases<complex_type, layout_type>(
      name, global_extents, topologies, pairs, all_axes_orders<3>(),
      TplPlanOps{});
}
#endif
