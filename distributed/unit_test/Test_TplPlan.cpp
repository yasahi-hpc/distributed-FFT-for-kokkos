#include <mpi.h>
#include <gtest/gtest.h>
#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include "KokkosFFT_Distributed_TplPlan.hpp"
#include "KokkosFFT_Distributed_Helper.hpp"
#include "Test_Utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using test_types      = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>,
                                    std::pair<double, Kokkos::LayoutLeft>,
                                    std::pair<double, Kokkos::LayoutRight>>;

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
      KokkosFFT::Distributed::get_local_extents(global_in_extents, topology0,
                                                MPI_COMM_WORLD);
  auto [in_extents_t1, in_starts_t1] =
      KokkosFFT::Distributed::get_local_extents(global_in_extents, topology1,
                                                MPI_COMM_WORLD);
  auto [out_extents_t0_ax0, out_starts_t0_ax0] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax0,
                                                topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax0, out_starts_t1_ax0] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax0,
                                                topology1, MPI_COMM_WORLD);
  auto [out_extents_t0_ax1, out_starts_t0_ax1] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax1,
                                                topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax1, out_starts_t1_ax1] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax1,
                                                topology1, MPI_COMM_WORLD);

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
      KokkosFFT::Distributed::get_local_extents(global_in_extents, topology0,
                                                MPI_COMM_WORLD);
  auto [in_extents_t1, in_starts_t1] =
      KokkosFFT::Distributed::get_local_extents(global_in_extents, topology1,
                                                MPI_COMM_WORLD);
  auto [out_extents_t0_ax0, out_starts_t0_ax0] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax0,
                                                topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax0, out_starts_t1_ax0] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax0,
                                                topology1, MPI_COMM_WORLD);
  auto [out_extents_t0_ax1, out_starts_t0_ax1] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax1,
                                                topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax1, out_starts_t1_ax1] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax1,
                                                topology1, MPI_COMM_WORLD);

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
    EXPECT_FALSE(is_available_0_0_ax01);

    if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
      // LayoutLeft does not support 2D transforms on 2D Views
      EXPECT_FALSE(is_available_1_0_ax01);
    } else {
      // LayoutRight supports 2D transforms on 2D Views
      EXPECT_TRUE(is_available_1_0_ax01);
    }

    // FIXME, disable this plan for now, as it is not supported
    EXPECT_FALSE(is_available_0_0_ax10);
    EXPECT_FALSE(is_available_1_0_ax10);
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
      KokkosFFT::Distributed::get_local_extents(global_in_extents, topology0,
                                                MPI_COMM_WORLD);
  auto [in_extents_t1, in_starts_t1] =
      KokkosFFT::Distributed::get_local_extents(global_in_extents, topology1,
                                                MPI_COMM_WORLD);
  auto [in_extents_t2, in_starts_t2] =
      KokkosFFT::Distributed::get_local_extents(global_in_extents, topology2,
                                                MPI_COMM_WORLD);
  auto [out_extents_t0_ax0, out_starts_t0_ax0] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax0,
                                                topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax0, out_starts_t1_ax0] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax0,
                                                topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax0, out_starts_t2_ax0] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax0,
                                                topology2, MPI_COMM_WORLD);
  auto [out_extents_t0_ax1, out_starts_t0_ax1] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax1,
                                                topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax1, out_starts_t1_ax1] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax1,
                                                topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax1, out_starts_t2_ax1] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax1,
                                                topology2, MPI_COMM_WORLD);
  auto [out_extents_t0_ax2, out_starts_t0_ax2] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax2,
                                                topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax2, out_starts_t1_ax2] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax2,
                                                topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax2, out_starts_t2_ax2] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax2,
                                                topology2, MPI_COMM_WORLD);

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

  axes_type ax012 = {0, 1, 2}, ax021 = {0, 2, 1}, ax102 = {1, 0, 2},
            ax120 = {1, 2, 0}, ax201 = {2, 0, 1}, ax210 = {2, 1, 0};

  const std::size_t n0 = 8, n1 = 7, n2 = 5;
  extents_type global_in_extents{n0, n1, n2},
      global_out_extents_ax0{n0 / 2 + 1, n1, n2},
      global_out_extents_ax1{n0, n1 / 2 + 1, n2},
      global_out_extents_ax2{n0, n1, n2 / 2 + 1};

  auto [in_extents_t0, in_starts_t0] =
      KokkosFFT::Distributed::get_local_extents(global_in_extents, topology0,
                                                MPI_COMM_WORLD);
  auto [in_extents_t1, in_starts_t1] =
      KokkosFFT::Distributed::get_local_extents(global_in_extents, topology1,
                                                MPI_COMM_WORLD);
  auto [in_extents_t2, in_starts_t2] =
      KokkosFFT::Distributed::get_local_extents(global_in_extents, topology2,
                                                MPI_COMM_WORLD);
  auto [out_extents_t0_ax0, out_starts_t0_ax0] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax0,
                                                topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax0, out_starts_t1_ax0] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax0,
                                                topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax0, out_starts_t2_ax0] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax0,
                                                topology2, MPI_COMM_WORLD);
  auto [out_extents_t0_ax1, out_starts_t0_ax1] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax1,
                                                topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax1, out_starts_t1_ax1] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax1,
                                                topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax1, out_starts_t2_ax1] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax1,
                                                topology2, MPI_COMM_WORLD);
  auto [out_extents_t0_ax2, out_starts_t0_ax2] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax2,
                                                topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax2, out_starts_t1_ax2] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax2,
                                                topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax2, out_starts_t2_ax2] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax2,
                                                topology2, MPI_COMM_WORLD);

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

  EXPECT_FALSE(is_available_0_0_ax012);
  EXPECT_FALSE(is_available_0_0_ax021);
  EXPECT_FALSE(is_available_0_0_ax102);
  EXPECT_FALSE(is_available_0_0_ax120);
  EXPECT_FALSE(is_available_0_0_ax201);
  EXPECT_FALSE(is_available_0_0_ax210);

  // Not a slab geometry
  if (nprocs == 1) {
    EXPECT_FALSE(is_available_0_1_ax012);
    EXPECT_FALSE(is_available_0_1_ax021);
    EXPECT_FALSE(is_available_0_1_ax102);
    EXPECT_FALSE(is_available_0_1_ax120);
    EXPECT_FALSE(is_available_0_1_ax201);
    EXPECT_FALSE(is_available_0_1_ax210);
  } else {
    if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
      EXPECT_FALSE(is_available_0_1_ax012);
    } else {
      EXPECT_TRUE(is_available_0_1_ax012);
    }

    // FIXME, disable this plan for now, as it is not supported
    EXPECT_FALSE(is_available_0_1_ax021);
    EXPECT_FALSE(is_available_0_1_ax102);
    EXPECT_FALSE(is_available_0_1_ax120);
    EXPECT_FALSE(is_available_0_1_ax201);
    EXPECT_FALSE(is_available_0_1_ax210);
  }
}

/// \brief Test if the plan is available for 2D transforms on 2D View
/// \tparam T Type of the data (float or double)
/// \tparam LayoutType Layout of the data (LayoutLeft or LayoutRight)
///
/// \param[in] nprocs Number of processes in the MPI communicator
template <typename T, typename LayoutType>
void test_tpl2D_execute_View2D(std::size_t nprocs) {
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
  axes_type ax01 = {0, 1}, ax10 = {1, 0};

  const std::size_t n0 = 8, n1 = 7;
  const std::size_t n0h = get_r2c_shape(n0, is_R2C),
                    n1h = get_r2c_shape(n1, is_R2C);
  extents_type global_in_extents{n0, n1}, global_out_extents_ax0{n0h, n1},
      global_out_extents_ax1{n0, n1h};

  auto [in_extents_t0, in_starts_t0] =
      KokkosFFT::Distributed::get_local_extents(global_in_extents, topology0,
                                                MPI_COMM_WORLD);
  auto [in_extents_t1, in_starts_t1] =
      KokkosFFT::Distributed::get_local_extents(global_in_extents, topology1,
                                                MPI_COMM_WORLD);
  auto [out_extents_t0_ax0, out_starts_t0_ax0] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax0,
                                                topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax0, out_starts_t1_ax0] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax0,
                                                topology1, MPI_COMM_WORLD);
  auto [out_extents_t0_ax1, out_starts_t0_ax1] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax1,
                                                topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax1, out_starts_t1_ax1] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax1,
                                                topology1, MPI_COMM_WORLD);

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

  // Do not support cases where the input/output topologies are the same
  ASSERT_THROW(
      {
        KokkosFFT::Distributed::Impl::TplPlan plan_0_0_ax01(
            exec, u_0, u_hat_0_ax01, ax01, topology0, topology0,
            MPI_COMM_WORLD);
      },
      std::runtime_error);

  ASSERT_THROW(
      {
        KokkosFFT::Distributed::Impl::TplPlan plan_0_0_ax10(
            exec, u_0, u_hat_0_ax10, ax10, topology0, topology0,
            MPI_COMM_WORLD);
      },
      std::runtime_error);

  ASSERT_THROW(
      {
        KokkosFFT::Distributed::Impl::TplPlan plan_1_1_ax01(
            exec, u_1, u_hat_1_ax01, ax01, topology1, topology1,
            MPI_COMM_WORLD);
      },
      std::runtime_error);

  ASSERT_THROW(
      {
        KokkosFFT::Distributed::Impl::TplPlan plan_1_1_ax10(
            exec, u_1, u_hat_1_ax10, ax10, topology1, topology1,
            MPI_COMM_WORLD);
      },
      std::runtime_error);

  // Not a slab geometry
  if (nprocs == 1) {
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
    // topo0 -> topo1 with ax = {0, 1}:
    // (n0, n1/p) -> (n0/p, n1/2+1)
    KokkosFFT::Distributed::Impl::TplPlan plan_0_1_ax01(
        exec, u_0, u_hat_1_ax01, ax01, topology0, topology1, MPI_COMM_WORLD);

    plan_0_1_ax01.forward(u_0, u_hat_1_ax01);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax01, ref_u_hat_1_ax01));

    plan_0_1_ax01.backward(u_hat_1_ax01, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo0 -> topo1 with ax = {1, 0}:
    // (n0, n1/p) -> ((n0/2+1)/p, n1)
    KokkosFFT::Distributed::Impl::TplPlan plan_0_1_ax10(
        exec, u_0, u_hat_1_ax10, ax10, topology0, topology1, MPI_COMM_WORLD);

    plan_0_1_ax10.forward(u_0, u_hat_1_ax10);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax10, ref_u_hat_1_ax10));

    plan_0_1_ax10.backward(u_hat_1_ax10, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo1 -> topo0 with ax = {0, 1}:
    // (n0/p, n1) -> (n0, (n1/2+1)/p)
    KokkosFFT::Distributed::Impl::TplPlan plan_1_0_ax01(
        exec, u_1, u_hat_0_ax01, ax01, topology1, topology0, MPI_COMM_WORLD);
    plan_1_0_ax01.forward(u_1, u_hat_0_ax01);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax01, ref_u_hat_0_ax01));

    plan_1_0_ax01.backward(u_hat_0_ax01, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo1 -> topo0 with ax = {1, 0}:
    // (n0/p, n1) -> ((n0/2+1)/p, n1)
    KokkosFFT::Distributed::Impl::TplPlan plan_1_0_ax10(
        exec, u_1, u_hat_0_ax10, ax10, topology1, topology0, MPI_COMM_WORLD);
    plan_1_0_ax10.forward(u_1, u_hat_0_ax10);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax10, ref_u_hat_0_ax10));

    plan_1_0_ax10.backward(u_hat_0_ax10, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));
  }
}

template <typename T, typename LayoutType>
void test_tpl3D_execute_View3D(std::size_t nprocs) {
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

  const std::size_t n0 = 8, n1 = 7, n2 = 5;
  const std::size_t n0h = get_r2c_shape(n0, is_R2C),
                    n1h = get_r2c_shape(n1, is_R2C),
                    n2h = get_r2c_shape(n2, is_R2C);
  extents_type global_in_extents{n0, n1, n2},
      global_out_extents_ax0{n0h, n1, n2}, global_out_extents_ax1{n0, n1h, n2},
      global_out_extents_ax2{n0, n1, n2h};

  // All axes
  axes_type ax012 = {0, 1, 2}, ax021 = {0, 2, 1}, ax102 = {1, 0, 2},
            ax120 = {1, 2, 0}, ax201 = {2, 0, 1}, ax210 = {2, 1, 0};

  auto [in_extents_t0, in_starts_t0] =
      KokkosFFT::Distributed::get_local_extents(global_in_extents, topology0,
                                                MPI_COMM_WORLD);
  auto [in_extents_t1, in_starts_t1] =
      KokkosFFT::Distributed::get_local_extents(global_in_extents, topology1,
                                                MPI_COMM_WORLD);
  auto [in_extents_t2, in_starts_t2] =
      KokkosFFT::Distributed::get_local_extents(global_in_extents, topology2,
                                                MPI_COMM_WORLD);
  auto [out_extents_t0_ax0, out_starts_t0_ax0] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax0,
                                                topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax0, out_starts_t1_ax0] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax0,
                                                topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax0, out_starts_t2_ax0] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax0,
                                                topology2, MPI_COMM_WORLD);
  auto [out_extents_t0_ax1, out_starts_t0_ax1] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax1,
                                                topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax1, out_starts_t1_ax1] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax1,
                                                topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax1, out_starts_t2_ax1] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax1,
                                                topology2, MPI_COMM_WORLD);
  auto [out_extents_t0_ax2, out_starts_t0_ax2] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax2,
                                                topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax2, out_starts_t1_ax2] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax2,
                                                topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax2, out_starts_t2_ax2] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax2,
                                                topology2, MPI_COMM_WORLD);

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

  // Not a slab geometry
  if (nprocs == 1) {
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
    KokkosFFT::Distributed::Impl::TplPlan plan_0_1_ax012(
        exec, u_0, u_hat_1_ax012, ax012, topology0, topology1, MPI_COMM_WORLD);
    plan_0_1_ax012.forward(u_0, u_hat_1_ax012);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax012, ref_u_hat_1_ax012));

    plan_0_1_ax012.backward(u_hat_1_ax012, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 1 with ax = {0, 2, 1}:
    // (n0, n1, n2/p) -> (n0, (n1/2+1)/p, n2)
    KokkosFFT::Distributed::Impl::TplPlan plan_0_1_ax021(
        exec, u_0, u_hat_1_ax021, ax021, topology0, topology1, MPI_COMM_WORLD);
    plan_0_1_ax021.forward(u_0, u_hat_1_ax021);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax021, ref_u_hat_1_ax021));

    plan_0_1_ax021.backward(u_hat_1_ax021, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 1 with ax = {1, 0, 2}:
    // (n0, n1, n2/p) -> (n0, n1/p, n2/2+1)
    KokkosFFT::Distributed::Impl::TplPlan plan_0_1_ax102(
        exec, u_0, u_hat_1_ax102, ax102, topology0, topology1, MPI_COMM_WORLD);
    plan_0_1_ax102.forward(u_0, u_hat_1_ax102);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax102, ref_u_hat_1_ax102));

    plan_0_1_ax102.backward(u_hat_1_ax102, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 1 with ax = {1, 2, 0}:
    // (n0, n1, n2/p) -> (n0/2+1, n1/p, n2)
    KokkosFFT::Distributed::Impl::TplPlan plan_0_1_ax120(
        exec, u_0, u_hat_1_ax120, ax120, topology0, topology1, MPI_COMM_WORLD);
    plan_0_1_ax120.forward(u_0, u_hat_1_ax120);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax120, ref_u_hat_1_ax120));

    plan_0_1_ax120.backward(u_hat_1_ax120, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 1 with ax = {2, 0, 1}:
    // (n0, n1, n2/p) -> (n0, (n1/2+1)/p, n2)
    KokkosFFT::Distributed::Impl::TplPlan plan_0_1_ax201(
        exec, u_0, u_hat_1_ax201, ax201, topology0, topology1, MPI_COMM_WORLD);
    plan_0_1_ax201.forward(u_0, u_hat_1_ax201);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax201, ref_u_hat_1_ax201));

    // FIXME
    plan_0_1_ax201.backward(u_hat_1_ax201, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 1 with ax = {2, 1, 0}:
    // (n0, n1, n2/p) -> (n0/2+1, n1/p, n2)
    KokkosFFT::Distributed::Impl::TplPlan plan_0_1_ax210(
        exec, u_0, u_hat_1_ax210, ax210, topology0, topology1, MPI_COMM_WORLD);
    plan_0_1_ax210.forward(u_0, u_hat_1_ax210);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax210, ref_u_hat_1_ax210));

    plan_0_1_ax210.backward(u_hat_1_ax210, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 2 with ax = {0, 1, 2}:
    // (n0, n1, n2/p) -> (n0/p, n1, n2) -> (n0/p, n1, n2/2+1)
    KokkosFFT::Distributed::Impl::TplPlan plan_0_2_ax012(
        exec, u_0, u_hat_2_ax012, ax012, topology0, topology2, MPI_COMM_WORLD);
    plan_0_2_ax012.forward(u_0, u_hat_2_ax012);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax012, ref_u_hat_2_ax012));

    plan_0_2_ax012.backward(u_hat_2_ax012, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 2 with ax = {0, 2, 1}:
    // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0, (n1/2+1)/p, n2)
    KokkosFFT::Distributed::Impl::TplPlan plan_0_2_ax021(
        exec, u_0, u_hat_2_ax021, ax021, topology0, topology2, MPI_COMM_WORLD);
    plan_0_2_ax021.forward(u_0, u_hat_2_ax021);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax021, ref_u_hat_2_ax021));

    plan_0_2_ax021.backward(u_hat_2_ax021, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 2 with ax = {1, 0, 2}:
    // (n0, n1, n2/p) -> (n0, n1/p, n2) -> (n0, n1/p, n2/2+1)
    KokkosFFT::Distributed::Impl::TplPlan plan_0_2_ax102(
        exec, u_0, u_hat_2_ax102, ax102, topology0, topology2, MPI_COMM_WORLD);
    plan_0_2_ax102.forward(u_0, u_hat_2_ax102);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax102, ref_u_hat_2_ax102));

    plan_0_2_ax102.backward(u_hat_2_ax102, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 2 with ax = {1, 2, 0}:
    // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
    KokkosFFT::Distributed::Impl::TplPlan plan_0_2_ax120(
        exec, u_0, u_hat_2_ax120, ax120, topology0, topology2, MPI_COMM_WORLD);
    plan_0_2_ax120.forward(u_0, u_hat_2_ax120);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax120, ref_u_hat_2_ax120));

    plan_0_2_ax120.backward(u_hat_2_ax120, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 2 with ax = {2, 0, 1}:
    // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0/p, n1/2+1, n2)
    KokkosFFT::Distributed::Impl::TplPlan plan_0_2_ax201(
        exec, u_0, u_hat_2_ax201, ax201, topology0, topology2, MPI_COMM_WORLD);
    plan_0_2_ax201.forward(u_0, u_hat_2_ax201);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax201, ref_u_hat_2_ax201));

    plan_0_2_ax201.backward(u_hat_2_ax201, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 2 with ax = {2, 1, 0}:
    // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
    KokkosFFT::Distributed::Impl::TplPlan plan_0_2_ax210(
        exec, u_0, u_hat_2_ax210, ax210, topology0, topology2, MPI_COMM_WORLD);
    plan_0_2_ax210.forward(u_0, u_hat_2_ax210);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax210, ref_u_hat_2_ax210));

    plan_0_2_ax210.backward(u_hat_2_ax210, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 0 with ax = {0, 1, 2}:
    // (n0p, n1/p, n2) -> (n0, n1, (n2/2+1)/p)
    KokkosFFT::Distributed::Impl::TplPlan plan_1_0_ax012(
        exec, u_1, u_hat_0_ax012, ax012, topology1, topology0, MPI_COMM_WORLD);
    plan_1_0_ax012.forward(u_1, u_hat_0_ax012);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax012, ref_u_hat_0_ax012));

    plan_1_0_ax012.backward(u_hat_0_ax012, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 0 with ax = {0, 2, 1}:
    // (n0p, n1/p, n2) -> (n0, n1/2+1, n2/p) -> (n0, (n1/2+1)/p, n2)
    KokkosFFT::Distributed::Impl::TplPlan plan_1_0_ax021(
        exec, u_1, u_hat_0_ax021, ax021, topology1, topology0, MPI_COMM_WORLD);
    plan_1_0_ax021.forward(u_1, u_hat_0_ax021);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax021, ref_u_hat_0_ax021));

    plan_1_0_ax021.backward(u_hat_0_ax021, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 0 with ax = {1, 0, 2}:
    // (n0p, n1/p, n2) -> (n0, n1/2+1, (n2/2+1)/p)
    KokkosFFT::Distributed::Impl::TplPlan plan_1_0_ax102(
        exec, u_1, u_hat_0_ax102, ax102, topology1, topology0, MPI_COMM_WORLD);
    plan_1_0_ax102.forward(u_1, u_hat_0_ax102);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax102, ref_u_hat_0_ax102));

    plan_1_0_ax102.backward(u_hat_0_ax102, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 0 with ax = {1, 2, 0}:
    // (n0p, n1/p, n2) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
    KokkosFFT::Distributed::Impl::TplPlan plan_1_0_ax120(
        exec, u_1, u_hat_0_ax120, ax120, topology1, topology0, MPI_COMM_WORLD);
    plan_1_0_ax120.forward(u_1, u_hat_0_ax120);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax120, ref_u_hat_0_ax120));

    plan_1_0_ax120.backward(u_hat_0_ax120, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 0 with ax = {2, 0, 1}:
    // (n0p, n1/p, n2) -> (n0, n1/2+1, n2/p) -> (n0/p, n1/2+1, n2)
    KokkosFFT::Distributed::Impl::TplPlan plan_1_0_ax201(
        exec, u_1, u_hat_0_ax201, ax201, topology1, topology0, MPI_COMM_WORLD);
    plan_1_0_ax201.forward(u_1, u_hat_0_ax201);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax201, ref_u_hat_0_ax201));

    plan_1_0_ax201.backward(u_hat_0_ax201, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 0 with ax = {2, 1, 0}:
    // (n0p, n1/p, n2) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
    KokkosFFT::Distributed::Impl::TplPlan plan_1_0_ax210(
        exec, u_1, u_hat_0_ax210, ax210, topology1, topology0, MPI_COMM_WORLD);
    plan_1_0_ax210.forward(u_1, u_hat_0_ax210);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax210, ref_u_hat_0_ax210));

    plan_1_0_ax210.backward(u_hat_0_ax210, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 2 with ax = {0, 1, 2}:
    // (n0, n1/p, n2) -> (n0/p, n1, n2) -> (n0/p, n1, n2/2+1)
    KokkosFFT::Distributed::Impl::TplPlan plan_1_2_ax012(
        exec, u_1, u_hat_2_ax012, ax012, topology1, topology2, MPI_COMM_WORLD);
    plan_1_2_ax012.forward(u_1, u_hat_2_ax012);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax012, ref_u_hat_2_ax012));

    plan_1_2_ax012.backward(u_hat_2_ax012, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 2 with ax = {0, 2, 1}:
    // (n0, n1/p, n2) -> (n0, n1/2+1, n2/p) -> (n0, (n1/2+1)/p, n2)
    KokkosFFT::Distributed::Impl::TplPlan plan_1_2_ax021(
        exec, u_1, u_hat_2_ax021, ax021, topology1, topology2, MPI_COMM_WORLD);
    plan_1_2_ax021.forward(u_1, u_hat_2_ax021);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax021, ref_u_hat_2_ax021));

    plan_1_2_ax021.backward(u_hat_2_ax021, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 2 with ax = {1, 0, 2}:
    // (n0, n1/p, n2) -> (n0, n1/2+1, n2/p) -> (n0, n1/p, n2/2+1)
    KokkosFFT::Distributed::Impl::TplPlan plan_1_2_ax102(
        exec, u_1, u_hat_2_ax102, ax102, topology1, topology2, MPI_COMM_WORLD);
    plan_1_2_ax102.forward(u_1, u_hat_2_ax102);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax102, ref_u_hat_2_ax102));

    plan_1_2_ax102.backward(u_hat_2_ax102, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 2 with ax = {1, 2, 0}:
    // (n0, n1/p, n2) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
    KokkosFFT::Distributed::Impl::TplPlan plan_1_2_ax120(
        exec, u_1, u_hat_2_ax120, ax120, topology1, topology2, MPI_COMM_WORLD);
    plan_1_2_ax120.forward(u_1, u_hat_2_ax120);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax120, ref_u_hat_2_ax120));

    plan_1_2_ax120.backward(u_hat_2_ax120, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 2 with ax = {2, 0, 1}:
    // (n0, n1/p, n2) -> (n0, n1/2+1, n2/p) -> (n0/p, n1/2+1, n2)
    KokkosFFT::Distributed::Impl::TplPlan plan_1_2_ax201(
        exec, u_1, u_hat_2_ax201, ax201, topology1, topology2, MPI_COMM_WORLD);
    plan_1_2_ax201.forward(u_1, u_hat_2_ax201);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax201, ref_u_hat_2_ax201));

    plan_1_2_ax201.backward(u_hat_2_ax201, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 2 with ax = {2, 1, 0}:
    // (n0, n1/p, n2) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
    KokkosFFT::Distributed::Impl::TplPlan plan_1_2_ax210(
        exec, u_1, u_hat_2_ax210, ax210, topology1, topology2, MPI_COMM_WORLD);
    plan_1_2_ax210.forward(u_1, u_hat_2_ax210);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax210, ref_u_hat_2_ax210));

    plan_1_2_ax210.backward(u_hat_2_ax210, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 0 with ax = {0, 1, 2}:
    // (n0/p, n1, n2) -> (n0, n1, n2/2+1) -> (n0, n1, n2/p)
    KokkosFFT::Distributed::Impl::TplPlan plan_2_0_ax012(
        exec, u_2, u_hat_0_ax012, ax012, topology2, topology0, MPI_COMM_WORLD);
    plan_2_0_ax012.forward(u_2, u_hat_0_ax012);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax012, ref_u_hat_0_ax012));

    plan_2_0_ax012.backward(u_hat_0_ax012, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 0 with ax = {0, 2, 1}:
    // (n0/p, n1, n2) -> (n0, n1/2+1, n2/p) -> (n0, (n1/2+1)/p, n2)
    KokkosFFT::Distributed::Impl::TplPlan plan_2_0_ax021(
        exec, u_2, u_hat_0_ax021, ax021, topology2, topology0, MPI_COMM_WORLD);
    plan_2_0_ax021.forward(u_2, u_hat_0_ax021);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax021, ref_u_hat_0_ax021));

    plan_2_0_ax021.backward(u_hat_0_ax021, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 0 with ax = {1, 0, 2}:
    // (n0/p, n1, n2) -> (n0, n1/2+1, n2/p) -> (n0, n1/p, n2/2+1)
    KokkosFFT::Distributed::Impl::TplPlan plan_2_0_ax102(
        exec, u_2, u_hat_0_ax102, ax102, topology2, topology0, MPI_COMM_WORLD);
    plan_2_0_ax102.forward(u_2, u_hat_0_ax102);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax102, ref_u_hat_0_ax102));

    plan_2_0_ax102.backward(u_hat_0_ax102, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 0 with ax = {1, 2, 0}:
    // (n0/p, n1, n2) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
    KokkosFFT::Distributed::Impl::TplPlan plan_2_0_ax120(
        exec, u_2, u_hat_0_ax120, ax120, topology2, topology0, MPI_COMM_WORLD);
    plan_2_0_ax120.forward(u_2, u_hat_0_ax120);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax120, ref_u_hat_0_ax120));

    plan_2_0_ax120.backward(u_hat_0_ax120, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 0 with ax = {2, 0, 1}:
    // (n0/p, n1, n2) -> (n0, n1/2+1, n2/p) -> (n0/p, n1/2+1, n2)
    KokkosFFT::Distributed::Impl::TplPlan plan_2_0_ax201(
        exec, u_2, u_hat_0_ax201, ax201, topology2, topology0, MPI_COMM_WORLD);
    plan_2_0_ax201.forward(u_2, u_hat_0_ax201);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax201, ref_u_hat_0_ax201));

    plan_2_0_ax201.backward(u_hat_0_ax201, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 0 with ax = {2, 1, 0}:
    // (n0/p, n1, n2) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
    KokkosFFT::Distributed::Impl::TplPlan plan_2_0_ax210(
        exec, u_2, u_hat_0_ax210, ax210, topology2, topology0, MPI_COMM_WORLD);
    plan_2_0_ax210.forward(u_2, u_hat_0_ax210);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax210, ref_u_hat_0_ax210));

    plan_2_0_ax210.backward(u_hat_0_ax210, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 1 with ax = {0, 1, 2}:
    // (n0/p, n1, n2) -> (n0/p, n1, n2/2+1) -> (n0, n1/p, n2/2+1)
    KokkosFFT::Distributed::Impl::TplPlan plan_2_1_ax012(
        exec, u_2, u_hat_1_ax012, ax012, topology2, topology1, MPI_COMM_WORLD);
    plan_2_1_ax012.forward(u_2, u_hat_1_ax012);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax012, ref_u_hat_1_ax012));

    plan_2_1_ax012.backward(u_hat_1_ax012, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 1 with ax = {0, 2, 1}:
    // (n0/p, n1, n2) -> (n0, n1/2+1, n2/p) -> (n0, (n1/2+1)/p, n2)
    KokkosFFT::Distributed::Impl::TplPlan plan_2_1_ax021(
        exec, u_2, u_hat_1_ax021, ax021, topology2, topology1, MPI_COMM_WORLD);
    plan_2_1_ax021.forward(u_2, u_hat_1_ax021);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax021, ref_u_hat_1_ax021));

    plan_2_1_ax021.backward(u_hat_1_ax021, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 1 with ax = {1, 0, 2}:
    // (n0/p, n1, n2) -> (n0/p, n1, n2/2+1) -> (n0, n1/p, n2/2+1)
    KokkosFFT::Distributed::Impl::TplPlan plan_2_1_ax102(
        exec, u_2, u_hat_1_ax102, ax102, topology2, topology1, MPI_COMM_WORLD);
    plan_2_1_ax102.forward(u_2, u_hat_1_ax102);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax102, ref_u_hat_1_ax102));

    plan_2_1_ax102.backward(u_hat_1_ax102, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 1 with ax = {1, 2, 0}:
    // (n0/p, n1, n2) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
    KokkosFFT::Distributed::Impl::TplPlan plan_2_1_ax120(
        exec, u_2, u_hat_1_ax120, ax120, topology2, topology1, MPI_COMM_WORLD);
    plan_2_1_ax120.forward(u_2, u_hat_1_ax120);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax120, ref_u_hat_1_ax120));

    plan_2_1_ax120.backward(u_hat_1_ax120, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 1 with ax = {2, 0, 1}:
    // (n0/p, n1, n2) -> (n0, n1/2+1, n2/p) -> (n0/p, n1/2+1, n2)
    KokkosFFT::Distributed::Impl::TplPlan plan_2_1_ax201(
        exec, u_2, u_hat_1_ax201, ax201, topology2, topology1, MPI_COMM_WORLD);
    plan_2_1_ax201.forward(u_2, u_hat_1_ax201);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax201, ref_u_hat_1_ax201));

    plan_2_1_ax201.backward(u_hat_1_ax201, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 1 with ax = {2, 1, 0}:
    // (n0/p, n1, n2) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
    KokkosFFT::Distributed::Impl::TplPlan plan_2_1_ax210(
        exec, u_2, u_hat_1_ax210, ax210, topology2, topology1, MPI_COMM_WORLD);
    plan_2_1_ax210.forward(u_2, u_hat_1_ax210);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax210, ref_u_hat_1_ax210));

    plan_2_1_ax210.backward(u_hat_1_ax210, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));
  }
}

template <typename T, typename LayoutType>
void test_tpl3D_execute_View3D_pencil(std::size_t npx, std::size_t npy) {
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

  const std::size_t n0 = 5, n1 = 6, n2 = 7;
  const std::size_t n0h = get_r2c_shape(n0, is_R2C),
                    n1h = get_r2c_shape(n1, is_R2C),
                    n2h = get_r2c_shape(n2, is_R2C);
  extents_type global_in_extents{n0, n1, n2},
      global_out_extents_ax0{n0h, n1, n2}, global_out_extents_ax1{n0, n1h, n2},
      global_out_extents_ax2{n0, n1, n2h};

  // All axes
  axes_type ax012 = {0, 1, 2}, ax021 = {0, 2, 1}, ax102 = {1, 0, 2},
            ax120 = {1, 2, 0}, ax201 = {2, 0, 1}, ax210 = {2, 1, 0};

  auto [in_extents_t0, in_starts_t0] =
      KokkosFFT::Distributed::get_local_extents(global_in_extents, topology0,
                                                MPI_COMM_WORLD);
  auto [in_extents_t1, in_starts_t1] =
      KokkosFFT::Distributed::get_local_extents(global_in_extents, topology1,
                                                MPI_COMM_WORLD);
  auto [in_extents_t2, in_starts_t2] =
      KokkosFFT::Distributed::get_local_extents(global_in_extents, topology2,
                                                MPI_COMM_WORLD);
  auto [out_extents_t0_ax0, out_starts_t0_ax0] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax0,
                                                topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax0, out_starts_t1_ax0] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax0,
                                                topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax0, out_starts_t2_ax0] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax0,
                                                topology2, MPI_COMM_WORLD);
  auto [out_extents_t0_ax1, out_starts_t0_ax1] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax1,
                                                topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax1, out_starts_t1_ax1] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax1,
                                                topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax1, out_starts_t2_ax1] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax1,
                                                topology2, MPI_COMM_WORLD);
  auto [out_extents_t0_ax2, out_starts_t0_ax2] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax2,
                                                topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax2, out_starts_t1_ax2] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax2,
                                                topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax2, out_starts_t2_ax2] =
      KokkosFFT::Distributed::get_local_extents(global_out_extents_ax2,
                                                topology2, MPI_COMM_WORLD);

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
    // topo 0 -> topo 2 with ax = {0, 1, 2}:
    // (n0, n1/px, n2/py) -> (n0/px, n1/py, n2/2+1)
    KokkosFFT::Distributed::Impl::TplPlan plan_0_2_ax012(
        exec, u_0, u_hat_2_ax012, ax012, topology0, topology2, MPI_COMM_WORLD);
    plan_0_2_ax012.forward(u_0, u_hat_2_ax012);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax012, ref_u_hat_2_ax012));

    plan_0_2_ax012.backward(u_hat_2_ax012, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 0 with ax = {0, 1, 2}:
    // (n0/px, n1/py, n2) -> (n0, n1/px, (n2/2+1)/py)
    KokkosFFT::Distributed::Impl::TplPlan plan_2_0_ax012(
        exec, u_2, u_hat_0_ax012, ax012, topology2, topology0, MPI_COMM_WORLD);
    plan_2_0_ax012.forward(u_2, u_hat_0_ax012);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax012, ref_u_hat_0_ax012));

    plan_2_0_ax012.backward(u_hat_0_ax012, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 1 with ax = {0, 1, 2}:
    // (n0/px, n1/py, n2) -> (n0/px, n1, (n2/2+1)/py)
    KokkosFFT::Distributed::Impl::TplPlan plan_2_1_ax012(
        exec, u_2, u_hat_1_ax012, ax012, topology2, topology1, MPI_COMM_WORLD);
    plan_2_1_ax012.forward(u_2, u_hat_1_ax012);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax012, ref_u_hat_1_ax012));

    plan_2_1_ax012.backward(u_hat_1_ax012, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));
  }
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

  if (this->m_nprocs != 2) {
    GTEST_SKIP() << "The number of MPI processes should be 2 for this test ";
  }
  test_tpl2D_execute_View2D<float_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestTplPlan2D, ExecuteView2D_C2C) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;
  if (this->m_nprocs != 2) {
    GTEST_SKIP() << "The number of MPI processes should be 2 for this test ";
  }
  test_tpl2D_execute_View2D<complex_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestTplPlan3D, ExecuteView3D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  if (this->m_nprocs != 2) {
    GTEST_SKIP() << "The number of MPI processes should be 2 for this test ";
  }
  test_tpl3D_execute_View3D<float_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestTplPlan3D, ExecuteView3D_C2C) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;

  if (this->m_nprocs != 2) {
    GTEST_SKIP() << "The number of MPI processes should be 2 for this test ";
  }
  test_tpl3D_execute_View3D<complex_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestTplPlan3D, ExecuteView3D_Pencil_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  if (this->m_nprocs == 1 || this->m_npx * this->m_npx != this->m_nprocs) {
    GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                    "for this test";
  }
  test_tpl3D_execute_View3D_pencil<float_type, layout_type>(this->m_npx,
                                                            this->m_npx);
}

TYPED_TEST(TestTplPlan3D, ExecuteView3D_Pencil_C2C) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;

  if (this->m_nprocs == 1 || this->m_npx * this->m_npx != this->m_nprocs) {
    GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                    "for this test";
  }
  test_tpl3D_execute_View3D_pencil<complex_type, layout_type>(this->m_npx,
                                                              this->m_npx);
}
