#include <mpi.h>
#include <gtest/gtest.h>
#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include "PencilPlan.hpp"
#include "MPI_Helper.hpp"
#include "Helper.hpp"
#include "Test_Utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using test_types      = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>,
                                    std::pair<double, Kokkos::LayoutLeft>,
                                    std::pair<double, Kokkos::LayoutRight>>;

//  Basically the same fixtures, used for labeling tests
template <typename T>
struct TestPencil1D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;

  int m_rank   = 0;
  int m_nprocs = 1;
  int m_npx    = 1;

  virtual void SetUp() {
    ::MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &m_nprocs);

    m_npx = std::sqrt(m_nprocs);
    if (this->m_nprocs == 1 || this->m_npx * this->m_npx != this->m_nprocs) {
      GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                      "for this test";
    }
  }
};

template <typename T>
struct TestPencil2D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;

  int m_rank   = 0;
  int m_nprocs = 1;
  int m_npx    = 1;

  virtual void SetUp() {
    ::MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &m_nprocs);

    m_npx = std::sqrt(m_nprocs);
    if (this->m_nprocs == 1 || this->m_npx * this->m_npx != this->m_nprocs) {
      GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                      "for this test";
    }
  }
};

template <typename T>
struct TestPencil3D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;

  int m_rank   = 0;
  int m_nprocs = 1;
  int m_npx    = 1;

  virtual void SetUp() {
    ::MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &m_nprocs);

    m_npx = std::sqrt(m_nprocs);
    if (this->m_nprocs == 1 || this->m_npx * this->m_npx != this->m_nprocs) {
      GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                      "for this test";
    }
  }
};

template <typename T, typename LayoutType>
void test_pencil1D_view3D(std::size_t npx, std::size_t npy) {
  using View3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using float_type = KokkosFFT::Impl::base_floating_point_type<T>;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<float_type>***, LayoutType, execution_space>;
  using axes_type       = KokkosFFT::axis_type<1>;
  using extents_type    = std::array<std::size_t, 3>;
  using topology_r_type = Topology<std::size_t, 3, Kokkos::LayoutRight>;
  using topology_l_type = Topology<std::size_t, 3, Kokkos::LayoutLeft>;

  constexpr bool is_R2C = KokkosFFT::Impl::is_real_v<T>;

  // Define, x-pencil, y-pencil and z-pencil
  topology_r_type topology0{1, npx, npy}, topology1{npx, 1, npy};
  topology_l_type topology2{npy, npx, 1};

  const std::size_t n0 = 8, n1 = 7, n2 = 5;
  const std::size_t n0h = get_r2c_shape(n0, is_R2C),
                    n1h = get_r2c_shape(n1, is_R2C),
                    n2h = get_r2c_shape(n2, is_R2C);
  extents_type global_in_extents{n0, n1, n2},
      global_out_extents_ax0{n0h, n1, n2}, global_out_extents_ax1{n0, n1h, n2},
      global_out_extents_ax2{n0, n1, n2h};

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
  View3DType gu("gu", n0, n1, n2);
  ComplexView3DType gu_hat_ax0("gu_hat_ax0", n0h, n1, n2),
      gu_hat_ax1("gu_hat_ax1", n0, n1h, n2),
      gu_hat_ax2("gu_hat_ax2", n0, n1, n2h);

  // Data in Topology 0 (X-pencil)
  View3DType u_0("u_0",
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

  // Data in Topology 1 (Y-pencil)
  View3DType u_1("u_1",
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

  // Data in Topology 2 (Z-pencil)
  View3DType u_2("u_2",
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

  if constexpr (is_R2C) {
    KokkosFFT::rfft(exec, gu, gu_hat_ax0, KokkosFFT::Normalization::backward,
                    0);
    KokkosFFT::rfft(exec, gu, gu_hat_ax1, KokkosFFT::Normalization::backward,
                    1);
    KokkosFFT::rfft(exec, gu, gu_hat_ax2, KokkosFFT::Normalization::backward,
                    2);
  } else {
    KokkosFFT::fft(exec, gu, gu_hat_ax0, KokkosFFT::Normalization::backward, 0);
    KokkosFFT::fft(exec, gu, gu_hat_ax1, KokkosFFT::Normalization::backward, 1);
    KokkosFFT::fft(exec, gu, gu_hat_ax2, KokkosFFT::Normalization::backward, 2);
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
  ;
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

  // Topo 0 ax = {0}
  auto sub_gu_hat_0_ax0 =
      Kokkos::subview(gu_hat_ax0, Kokkos::ALL, range_gu_hat_0_ax0_dim1,
                      range_gu_hat_0_ax0_dim2);
  Kokkos::deep_copy(ref_u_hat_0_ax0, sub_gu_hat_0_ax0);

  // Topo 0 ax = {1}
  auto sub_gu_hat_0_ax1 =
      Kokkos::subview(gu_hat_ax1, Kokkos::ALL, range_gu_hat_0_ax1_dim1,
                      range_gu_hat_0_ax1_dim2);
  Kokkos::deep_copy(ref_u_hat_0_ax1, sub_gu_hat_0_ax1);

  // Topo 0 ax = {2}
  auto sub_gu_hat_0_ax2 =
      Kokkos::subview(gu_hat_ax2, Kokkos::ALL, range_gu_hat_0_ax2_dim1,
                      range_gu_hat_0_ax2_dim2);
  Kokkos::deep_copy(ref_u_hat_0_ax2, sub_gu_hat_0_ax2);

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

  // Topo 1 ax = {0}
  auto sub_gu_hat_1_ax0 = Kokkos::subview(gu_hat_ax0, range_gu_hat_1_ax0_dim0,
                                          Kokkos::ALL, range_gu_hat_1_ax0_dim2);
  Kokkos::deep_copy(ref_u_hat_1_ax0, sub_gu_hat_1_ax0);

  // Topo 1 ax = {1}
  auto sub_gu_hat_1_ax1 = Kokkos::subview(gu_hat_ax1, range_gu_hat_1_ax1_dim0,
                                          Kokkos::ALL, range_gu_hat_1_ax1_dim2);
  Kokkos::deep_copy(ref_u_hat_1_ax1, sub_gu_hat_1_ax1);

  // Topo 1 ax = {2}
  auto sub_gu_hat_1_ax2 = Kokkos::subview(gu_hat_ax2, range_gu_hat_1_ax2_dim0,
                                          Kokkos::ALL, range_gu_hat_1_ax2_dim2);
  Kokkos::deep_copy(ref_u_hat_1_ax2, sub_gu_hat_1_ax2);

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

  // Topo 2 ax = {0}
  auto sub_gu_hat_2_ax0 = Kokkos::subview(gu_hat_ax0, range_gu_hat_2_ax0_dim0,
                                          range_gu_hat_2_ax0_dim1, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax0, sub_gu_hat_2_ax0);

  // Topo 2 ax = {1}
  auto sub_gu_hat_2_ax1 = Kokkos::subview(gu_hat_ax1, range_gu_hat_2_ax1_dim0,
                                          range_gu_hat_2_ax1_dim1, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax1, sub_gu_hat_2_ax1);

  // Topo 2 ax = {2}
  auto sub_gu_hat_2_ax2 = Kokkos::subview(gu_hat_ax2, range_gu_hat_2_ax2_dim0,
                                          range_gu_hat_2_ax2_dim1, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_2_ax2, sub_gu_hat_2_ax2);

  // For inverse transform
  Kokkos::deep_copy(ref_u_inv_0, u_0);
  Kokkos::deep_copy(ref_u_inv_1, u_1);
  Kokkos::deep_copy(ref_u_inv_2, u_2);

  // Not a pencil geometry
  if (npx == 1 || npy == 1) {
    // topo0 -> topo1 with ax = {0}:
    // (n0, n1/px, n2/py) -> ((n0/2+1)/px, n1, n2/py)
    ASSERT_THROW(
        {
          PencilPlan plan_0_1_ax0(exec, u_0, u_hat_1_ax0, ax0, topology0,
                                  topology1, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topo0 -> topo2 with ax = {1}:
    // (n0, n1/px, n2/py) -> (n0/py, (n1/2+1)/px, n2)
    ASSERT_THROW(
        {
          PencilPlan plan_0_2_ax1(exec, u_0, u_hat_2_ax1, ax1, topology0,
                                  topology2, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topo1 -> topo2 with ax = {2}:
    // (n0/px, n1, n2/py) -> (n0/px, n1/py, n2/2+1)
    ASSERT_THROW(
        {
          PencilPlan plan_1_2_ax2(exec, u_1, u_hat_2_ax2, ax2, topology1,
                                  topology2, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topo0 -> topo0 with ax = {2}:
    // (n0, n1/px, n2/py) -> (n0, n1/px, (n2/2+1)/py)
    ASSERT_THROW(
        {
          PencilPlan plan_0_0_ax2(exec, u_0, u_hat_0_ax2, ax2, topology0,
                                  topology0, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topo1 -> topo1 with ax = {1}
    // (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
    ASSERT_THROW(
        {
          PencilPlan plan_1_1_ax1(exec, u_1, u_hat_1_ax1, ax1, topology1,
                                  topology1, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology2 -> topology2 with ax = {1}
    // (n0/px, n1/py, n2) -> (n0/px, (n1/2+1)/py, n2)
    ASSERT_THROW(
        {
          PencilPlan plan_2_2_ax1(exec, u_2, u_hat_2_ax1, ax1, topology2,
                                  topology2, MPI_COMM_WORLD);
        },
        std::runtime_error);
  } else {
    // topo 0 -> topo 0 with ax = {0}:
    // (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
    // FFT ax = {0}
    PencilPlan plan_0_0_ax0(exec, u_0, u_hat_0_ax0, ax0, topology0, topology0,
                            MPI_COMM_WORLD);
    plan_0_0_ax0.forward(u_0, u_hat_0_ax0);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax0, ref_u_hat_0_ax0));

    plan_0_0_ax0.backward(u_hat_0_ax0, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 0 with ax = {1}:
    // (n0, n1/px, n2/py) -> (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
    // -> (n0, (n1/2+1)/px, n2/py)
    // Transpose topo 1 -> FFT ax = {1} -> Transpose topo 0
    PencilPlan plan_0_0_ax1(exec, u_0, u_hat_0_ax1, ax1, topology0, topology0,
                            MPI_COMM_WORLD);
    plan_0_0_ax1.forward(u_0, u_hat_0_ax1);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax1, ref_u_hat_0_ax1));

    plan_0_0_ax1.backward(u_hat_0_ax1, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 0 with ax = {2}:
    // (n0, n1/px, n2/py) -> (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
    // -> (n0, n1/px, (n2/2+1)/py)
    // Transpose topo 2 -> FFT ax = {2} -> Transpose topo 0
    PencilPlan plan_0_0_ax2(exec, u_0, u_hat_0_ax2, ax2, topology0, topology0,
                            MPI_COMM_WORLD);
    plan_0_0_ax2.forward(u_0, u_hat_0_ax2);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax2, ref_u_hat_0_ax2));

    plan_0_0_ax2.backward(u_hat_0_ax2, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 1 with ax = {0}:
    // (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py) -> ((n0/2+1)/px, n1, n2/py)
    // FFT ax = {0} -> Transpose 1
    PencilPlan plan_0_1_ax0(exec, u_0, u_hat_1_ax0, ax0, topology0, topology1,
                            MPI_COMM_WORLD);
    plan_0_1_ax0.forward(u_0, u_hat_1_ax0);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax0, ref_u_hat_1_ax0));

    plan_0_1_ax0.backward(u_hat_1_ax0, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 1 with ax = {1}:
    // (n0, n1/px, n2/py) -> (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
    // Transpose topo 1 -> FFT ax = {1}
    PencilPlan plan_0_1_ax1(exec, u_0, u_hat_1_ax1, ax1, topology0, topology1,
                            MPI_COMM_WORLD);
    plan_0_1_ax1.forward(u_0, u_hat_1_ax1);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax1, ref_u_hat_1_ax1));

    plan_0_1_ax1.backward(u_hat_1_ax1, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 1 with ax = {2}:
    // (n0, n1/px, n2/py) -> (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
    // -> (n0, n1/px, (n2/2+1)/py) -> (n0/px, n1/px, (n2/2+1)/py)
    // Transpose topo 2 -> FFT ax = {2} -> Transpose topo 0 -> Transpose topo 1
    PencilPlan plan_0_1_ax2(exec, u_0, u_hat_1_ax2, ax2, topology0, topology1,
                            MPI_COMM_WORLD);
    plan_0_1_ax2.forward(u_0, u_hat_1_ax2);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax2, ref_u_hat_1_ax2));

    plan_0_1_ax2.backward(u_hat_1_ax2, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 2 with ax = {0}:
    // (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py) -> ((n0/2+1)/py, n1/px, n2)
    // FFT ax = {0} -> Transpose topo 2
    PencilPlan plan_0_2_ax0(exec, u_0, u_hat_2_ax0, ax0, topology0, topology2,
                            MPI_COMM_WORLD);
    plan_0_2_ax0.forward(u_0, u_hat_2_ax0);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax0, ref_u_hat_2_ax0));

    plan_0_2_ax0.backward(u_hat_2_ax0, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    //  topo 0 -> topo 2 with ax = {1}:
    //  (n0, n1/px, n2/py) -> (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
    //  -> (n0, (n1/2+1)/px, n2/py) -> (n0/py, (n1/2+1)/px, n2)
    //  Transpose topo 1 -> FFT ax = {1} -> Transpose topo 0 -> topo 2
    PencilPlan plan_0_2_ax1(exec, u_0, u_hat_2_ax1, ax1, topology0, topology2,
                            MPI_COMM_WORLD);
    plan_0_2_ax1.forward(u_0, u_hat_2_ax1);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax1, ref_u_hat_2_ax1));

    plan_0_2_ax1.backward(u_hat_2_ax1, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 2 with ax = {2}:
    // (n0, n1/px, n2/py) -> (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
    // Transpose topo 2 -> FFT ax = {2}
    PencilPlan plan_0_2_ax2(exec, u_0, u_hat_2_ax2, ax2, topology0, topology2,
                            MPI_COMM_WORLD);
    plan_0_2_ax2.forward(u_0, u_hat_2_ax2);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax2, ref_u_hat_2_ax2));

    plan_0_2_ax2.backward(u_hat_2_ax2, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 1 -> topo 0 with ax = {0}:
    // (n0/px, n1, n2/py) -> (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
    // Transpose topo 0 -> FFT ax = {0}
    PencilPlan plan_1_0_ax0(exec, u_1, u_hat_0_ax0, ax0, topology1, topology0,
                            MPI_COMM_WORLD);
    plan_1_0_ax0.forward(u_1, u_hat_0_ax0);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax0, ref_u_hat_0_ax0));

    plan_1_0_ax0.backward(u_hat_0_ax0, u_inv_1);
    EXPECT_TRUE(allclose(exec, u_inv_1, ref_u_inv_1, 1.0e-5, 1.0e-6));
  }
}

template <typename T, typename LayoutType>
void test_pencil3D_view3D(std::size_t npx, std::size_t npy) {
  using View3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using float_type = KokkosFFT::Impl::base_floating_point_type<T>;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<float_type>***, LayoutType, execution_space>;
  using axes_type       = KokkosFFT::axis_type<3>;
  using extents_type    = std::array<std::size_t, 3>;
  using topology_r_type = Topology<std::size_t, 3, Kokkos::LayoutRight>;
  using topology_l_type = Topology<std::size_t, 3, Kokkos::LayoutLeft>;

  constexpr bool is_R2C = KokkosFFT::Impl::is_real_v<T>;

  topology_r_type topology0{1, npx, npy}, topology1{npx, 1, npy},
      topology3{npx, npy, 1};
  topology_l_type topology2{npy, npx, 1};

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

  // Make reference with a basic-API
  View3DType gu("gu", n0, n1, n2);
  ComplexView3DType gu_hat_ax012("gu_hat_ax012", n0, n1, n2h),
      gu_hat_ax021("gu_hat_ax021", n0, n1h, n2),
      gu_hat_ax102("gu_hat_ax102", n0, n1, n2h),
      gu_hat_ax120("gu_hat_ax120", n0h, n1, n2),
      gu_hat_ax201("gu_hat_ax201", n0, n1h, n2),
      gu_hat_ax210("gu_hat_ax210", n0h, n1, n2);

  // Data in Topology 0 (XY-slab)
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

  // Data in Topology 1 (XZ-slab)
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

  // Data in Topology 3 (Z-pencil)
  View3DType u_3("u_3",
                 KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t3)),
      u_inv_3("u_inv_3",
              KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t3)),
      ref_u_inv_3("ref_u_inv_3",
                  KokkosFFT::Impl::create_layout<LayoutType>(in_extents_t3));
  ComplexView3DType u_hat_3_ax012(
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
          KokkosFFT::Impl::create_layout<LayoutType>(out_extents_t3_ax0));

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

  // Topo 3
  Kokkos::pair<std::size_t, std::size_t> range_gu3_dim0(
      in_starts_t3.at(0), in_starts_t3.at(0) + in_extents_t3.at(0)),
      range_gu3_dim1(in_starts_t3.at(1),
                     in_starts_t3.at(1) + in_extents_t3.at(1));
  auto sub_gu_3 =
      Kokkos::subview(gu, range_gu3_dim0, range_gu3_dim1, Kokkos::ALL);
  Kokkos::deep_copy(u_3, sub_gu_3);

  // Define ranges for topology 0 (X-pencil)
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_0_ax0_dim1(
      out_starts_t0_ax0.at(1),
      out_starts_t0_ax0.at(1) + out_extents_t0_ax0.at(1)),
      range_gu_hat_0_ax0_dim2(
          out_starts_t0_ax0.at(2),
          out_starts_t0_ax0.at(2) + out_extents_t0_ax0.at(2));
  ;
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

  // Define ranges for topology 3 (Z-pencil)
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_3_ax0_dim0(
      out_starts_t3_ax0.at(0),
      out_starts_t3_ax0.at(0) + out_extents_t3_ax0.at(0)),
      range_gu_hat_3_ax0_dim1(
          out_starts_t3_ax0.at(1),
          out_starts_t3_ax0.at(1) + out_extents_t3_ax0.at(1));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_3_ax1_dim0(
      out_starts_t3_ax1.at(0),
      out_starts_t3_ax1.at(0) + out_extents_t3_ax1.at(0)),
      range_gu_hat_3_ax1_dim1(
          out_starts_t3_ax1.at(1),
          out_starts_t3_ax1.at(1) + out_extents_t3_ax1.at(1));
  Kokkos::pair<std::size_t, std::size_t> range_gu_hat_3_ax2_dim0(
      out_starts_t3_ax2.at(0),
      out_starts_t3_ax2.at(0) + out_extents_t3_ax2.at(0)),
      range_gu_hat_3_ax2_dim1(
          out_starts_t3_ax2.at(1),
          out_starts_t3_ax2.at(1) + out_extents_t3_ax2.at(1));

  // Topo 3 ax = {0, 1, 2}
  auto sub_gu_hat_3_ax012 =
      Kokkos::subview(gu_hat_ax012, range_gu_hat_3_ax2_dim0,
                      range_gu_hat_3_ax2_dim1, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_3_ax012, sub_gu_hat_3_ax012);

  // Topo 3 ax = {0, 2, 1}
  auto sub_gu_hat_3_ax021 =
      Kokkos::subview(gu_hat_ax021, range_gu_hat_3_ax1_dim0,
                      range_gu_hat_3_ax1_dim1, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_3_ax021, sub_gu_hat_3_ax021);

  // Topo 3 ax = {1, 0, 2}
  auto sub_gu_hat_3_ax102 =
      Kokkos::subview(gu_hat_ax102, range_gu_hat_3_ax2_dim0,
                      range_gu_hat_3_ax2_dim1, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_3_ax102, sub_gu_hat_3_ax102);

  // Topo 3 ax = {1, 2, 0}
  auto sub_gu_hat_3_ax120 =
      Kokkos::subview(gu_hat_ax120, range_gu_hat_3_ax0_dim0,
                      range_gu_hat_3_ax0_dim1, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_3_ax120, sub_gu_hat_3_ax120);

  // Topo 3 ax = {2, 0, 1}
  auto sub_gu_hat_3_ax201 =
      Kokkos::subview(gu_hat_ax201, range_gu_hat_3_ax1_dim0,
                      range_gu_hat_3_ax1_dim1, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_3_ax201, sub_gu_hat_3_ax201);

  // Topo 3 ax = {2, 1, 0}
  auto sub_gu_hat_3_ax210 =
      Kokkos::subview(gu_hat_ax210, range_gu_hat_3_ax0_dim0,
                      range_gu_hat_3_ax0_dim1, Kokkos::ALL);
  Kokkos::deep_copy(ref_u_hat_3_ax210, sub_gu_hat_3_ax210);

  // For inverse transform
  Kokkos::deep_copy(ref_u_inv_0, u_0);
  Kokkos::deep_copy(ref_u_inv_1, u_1);
  Kokkos::deep_copy(ref_u_inv_2, u_2);
  Kokkos::deep_copy(ref_u_inv_3, u_3);

  // Not a pencil geometry
  if (npx == 1 || npy == 1) {
    // topology0 -> topology1
    // (n0, n1/px, n2/py) -> (n0, (n1/2+1)/p, n2)
    ASSERT_THROW(
        {
          PencilPlan plan_0_1_ax012(exec, u_0, u_hat_1_ax012, ax012, topology0,
                                    topology1, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology0 -> topology2
    // (n0, n1, n2/p) -> (n0/p, n1/2+1, n2)
    ASSERT_THROW(
        {
          PencilPlan plan_0_2_ax012(exec, u_0, u_hat_2_ax012, ax012, topology0,
                                    topology2, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology1 -> topology2
    // (n0, n1/p, n2) -> (n0/p, n1, n2/2+1)
    ASSERT_THROW(
        {
          PencilPlan plan_1_2_ax012(exec, u_1, u_hat_2_ax012, ax012, topology1,
                                    topology2, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology0 -> topology0 with ax = {1, 2}
    // (n0, n1, n2/p) -> (n0, n1, (n2/2+1)/p)
    ASSERT_THROW(
        {
          PencilPlan plan_0_0_ax012(exec, u_0, u_hat_0_ax012, ax012, topology0,
                                    topology0, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology1 -> topology1 with ax = {1, 2}
    // (n0, n1/p, n2) -> (n0, (n1/2+1)/p, n2)
    ASSERT_THROW(
        {
          PencilPlan plan_1_1_ax210(exec, u_1, u_hat_1_ax210, ax210, topology1,
                                    topology1, MPI_COMM_WORLD);
        },
        std::runtime_error);

    // topology2 -> topology2 with ax = {0, 1, 2}
    // (n0/p, n1, n2) -> (n0/p, (n1/2+1), n2)
    ASSERT_THROW(
        {
          PencilPlan plan_2_2_ax012(exec, u_2, u_hat_2_ax012, ax012, topology2,
                                    topology2, MPI_COMM_WORLD);
        },
        std::runtime_error);
  } else {
    // topo 0 -> topo 0 with ax = {0, 1, 2}:
    // (n0, n1/px, n2/py) -> (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
    // -> (n0/py, n1, (n2/2+1)/px) -> (n0/py, n1/px, n2/2+1)
    // -> (n0, n1/px, (n2/2+1)/py)
    // Transpose topo 2 -> FFT ax = {2} -> Transpose topo 4 -> FFT ax = {1}
    // -> Transpose topo 2 -> Transpose topo 0 -> FFT ax = {0}
    PencilPlan plan_0_0_ax012(exec, u_0, u_hat_0_ax012, ax012, topology0,
                              topology0, MPI_COMM_WORLD);
    plan_0_0_ax012.forward(u_0, u_hat_0_ax012);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax012, ref_u_hat_0_ax012));

    plan_0_0_ax012.backward(u_hat_0_ax012, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 0 with ax = {0, 2, 1}:
    // (n0, n1/px, n2/py) -> (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
    // -> (n0/px, (n1/2+1)/py, n2) -> (n0/px, n1/2+1, n2/py) -> (n0, n1/px,
    // n2/py) Transpose topo 1 -> FFT ax = {1} -> Transpose topo 3 -> FFT ax =
    // {2}
    // -> Transpose topo 1 -> Transpose topo 0 -> FFT ax = {0}
    PencilPlan plan_0_0_ax021(exec, u_0, u_hat_0_ax021, ax021, topology0,
                              topology0, MPI_COMM_WORLD);
    plan_0_0_ax021.forward(u_0, u_hat_0_ax021);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax021, ref_u_hat_0_ax021));

    plan_0_0_ax021.backward(u_hat_0_ax021, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 0 with ax = {1, 0, 2}:
    // (n0, n1/px, n2/py) -> (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
    // -> (n0x, n1/px, (n2/2+1)/py) -> (n0/py, n1/px, n2/2+1)
    // -> (n0x, n1/px, (n2/2+1)/py)
    // Transpose topo 2 -> FFT ax = {2} -> Transpose topo 0
    // -> FFT ax = {0} -> Transpose topo 1 -> FFT ax = {1} -> Transpose topo 0
    PencilPlan plan_0_0_ax102(exec, u_0, u_hat_0_ax102, ax102, topology0,
                              topology0, MPI_COMM_WORLD);
    plan_0_0_ax102.forward(u_0, u_hat_0_ax102);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax102, ref_u_hat_0_ax102));

    plan_0_0_ax102.backward(u_hat_0_ax102, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 0 with ax = {1, 2, 0}:
    // (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py) -> ((n0/2+1)/py, n1/px, n2)
    // -> ((n0/2+1)/py, n1, n2/px) -> ((n0/2+1)/py, n1/px, n2) -> (n0/2+1,
    // n1/px, n2/py)
    // FFT ax = {0} -> Transpose topo 2 -> FFT ax = {2} ->
    // Transpose topo 4 -> FFT ax = {1} -> Transpose topo 2 -> Transpose topo 0
    PencilPlan plan_0_0_ax120(exec, u_0, u_hat_0_ax120, ax120, topology0,
                              topology0, MPI_COMM_WORLD);
    plan_0_0_ax120.forward(u_0, u_hat_0_ax120);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax120, ref_u_hat_0_ax120));

    plan_0_0_ax120.backward(u_hat_0_ax120, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 0 with ax = {2, 0, 1}:
    // (n0, n1/px, n2/py) -> (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
    // -> (n0, (n1/2+1)/px, n2/py) -> (n0/py, (n1/2+1)/px, n2)
    // -> (n0, (n1/2+1)/px, n2/py)
    // Transpose topo 1 -> FFT ax = {1} -> Transpose topo 0 -> FFT ax = {0}
    // -> Transpose topo 2 -> FFT ax = {2} -> Transpose topo 0
    PencilPlan plan_0_0_ax201(exec, u_0, u_hat_0_ax201, ax201, topology0,
                              topology0, MPI_COMM_WORLD);
    plan_0_0_ax201.forward(u_0, u_hat_0_ax201);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax201, ref_u_hat_0_ax201));

    plan_0_0_ax201.backward(u_hat_0_ax201, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 0 with ax = {2, 1, 0}:
    // (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py) -> ((n0/2+1)/px, n1, n2/py)
    // -> ((n0/2+1)/px, n1/py, n2) -> ((n0/2+1)/px, n1, n2/py) -> (n0/2+1,
    // n1/px, n2/py) FFT ax = {0} -> Transpose topo 1 -> FFT ax = {1} ->
    // Transpose topo 3 -> FFT ax = {2} -> Transpose topo 1 -> Transpose topo 0
    PencilPlan plan_0_0_ax210(exec, u_0, u_hat_0_ax210, ax210, topology0,
                              topology0, MPI_COMM_WORLD);
    plan_0_0_ax210.forward(u_0, u_hat_0_ax210);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax210, ref_u_hat_0_ax210));

    plan_0_0_ax210.backward(u_hat_0_ax210, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 1 with ax = {0, 1, 2}:
    // (n0, n1/px, n2/py) -> (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
    // -> (n0/py, n1, (n2/2+1)/px) -> (n0, n1/py, (n2/2+1)/px)
    // -> (n0/px, n1/py, n2/2+1) -> (n0/px, n1, (n2/2+1)/py)
    // Transpose topo 2 -> FFT ax = {2} -> Transpose topo 4 -> FFT ax = {1}
    // -> Transpose topo 5 -> FFT ax = {0} -> Transpose topo 3 -> topo 1
    PencilPlan plan_0_1_ax012(exec, u_0, u_hat_1_ax012, ax012, topology0,
                              topology1, MPI_COMM_WORLD);
    plan_0_1_ax012.forward(u_0, u_hat_1_ax012);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax012, ref_u_hat_1_ax012));

    plan_0_1_ax012.backward(u_hat_1_ax012, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 1 with ax = {0, 2, 1}:
    // (n0, n1/px, n2/py) -> (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
    // -> (n0/px, (n1/2+1)/py, n2) -> (n0, (n1/2+1)/py, n2/px)
    // -> (n0/px, (n1/2+1)/py, n2) -> (n0/px, n1/2+1, n2/py)
    // Transpose topo 1 -> FFT ax = {1} -> Transpose topo 3 -> FFT ax = {2}
    // -> Transpose topo 5 -> FFT ax = {0} -> Transpose topo 3 -> topo 1
    PencilPlan plan_0_1_ax021(exec, u_0, u_hat_1_ax021, ax021, topology0,
                              topology1, MPI_COMM_WORLD);
    plan_0_1_ax021.forward(u_0, u_hat_1_ax021);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax021, ref_u_hat_1_ax021));

    plan_0_1_ax021.backward(u_hat_1_ax021, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 1 with ax = {1, 0, 2}:
    // (n0, n1/px, n2/py) -> (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
    // -> (n0, n1/px, (n2/2+1)/py) -> (n0/px, n1, (n2/2+1)/py)
    // Transpose topo 2 -> FFT ax = {2} -> Transpose topo 0 -> FFT ax = {0}
    // -> Transpose topo 1 -> FFT ax = {1}
    PencilPlan plan_0_1_ax102(exec, u_0, u_hat_1_ax102, ax102, topology0,
                              topology1, MPI_COMM_WORLD);
    plan_0_1_ax102.forward(u_0, u_hat_1_ax102);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax102, ref_u_hat_1_ax102));

    plan_0_1_ax102.backward(u_hat_1_ax102, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 1 with ax = {1, 2, 0}:
    // (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py) -> ((n0/2+1)/py, n1/px, n2)
    // -> (n0/2+1, n1/px, n2/py) -> ((n0/2+1)/px, n1, n2/py)
    // FFT ax = {0} ->Transpose topo 2 -> FFT ax = {2} -> Transpose topo 0
    // -> Transpose topo 1 -> FFT ax = {1}
    PencilPlan plan_0_1_ax120(exec, u_0, u_hat_1_ax120, ax120, topology0,
                              topology1, MPI_COMM_WORLD);
    plan_0_1_ax120.forward(u_0, u_hat_1_ax120);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax120, ref_u_hat_1_ax120));

    plan_0_1_ax120.backward(u_hat_1_ax120, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    /*
    // topo 0 -> topo 1 with ax = {2, 0, 1}:
    // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0, (n1/2+1)/p, n2)
    // FFT2 ax = {0, 1} -> Transpose topo 1 -> FFT ax = {2}
    PencilPlan plan_0_1_ax201(exec, u_0, u_hat_1_ax201, ax201, topology0,
                                topology1, MPI_COMM_WORLD);
    plan_0_1_ax201.forward(u_0, u_hat_1_ax201);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax201, ref_u_hat_1_ax201));

    plan_0_1_ax201.backward(u_hat_1_ax201, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 1 with ax = {2, 1, 0}:
    // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> (n0/2+1, n1/p, n2)
    // FFT2 ax = {1, 0} -> Transpose topo 1 -> FFT ax = {2}
    PencilPlan plan_0_1_ax210(exec, u_0, u_hat_1_ax210, ax210, topology0,
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
    PencilPlan plan_0_2_ax012(exec, u_0, u_hat_2_ax012, ax012, topology0,
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
    PencilPlan plan_0_2_ax021(exec, u_0, u_hat_2_ax021, ax021, topology0,
                                topology2, MPI_COMM_WORLD);
    plan_0_2_ax021.forward(u_0, u_hat_2_ax021);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax021, ref_u_hat_2_ax021));

    plan_0_2_ax021.backward(u_hat_2_ax021, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 2 with ax = {1, 0, 2}:
    // (n0, n1, n2/p) -> (n0, n1/p, n2) -> (n0, n1/p, n2/2+1)
    // -> (n0/p, n1, n2/2+1)
    // Transpose topo 1 -> FFT2 ax = {0, 2} -> Transpose topo 2 -> FFT ax = {1}
    PencilPlan plan_0_2_ax102(exec, u_0, u_hat_2_ax102, ax102, topology0,
                                topology2, MPI_COMM_WORLD);
    plan_0_2_ax102.forward(u_0, u_hat_2_ax102);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax102, ref_u_hat_2_ax102));

    plan_0_2_ax102.backward(u_hat_2_ax102, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 2 with ax = {1, 2, 0}:
    // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
    // FFT ax = {0} -> Transpose topo 2 -> FFT2 ax = {1, 2}
    PencilPlan plan_0_2_ax120(exec, u_0, u_hat_2_ax120, ax120, topology0,
                                topology2, MPI_COMM_WORLD);
    plan_0_2_ax120.forward(u_0, u_hat_2_ax120);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax120, ref_u_hat_2_ax120));

    plan_0_2_ax120.backward(u_hat_2_ax120, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 2 with ax = {2, 0, 1}:
    // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0/p, n1/2+1, n2)
    // FFT2 ax = {0, 1} -> Transpose topo 2 -> FFT ax = {2}
    PencilPlan plan_0_2_ax201(exec, u_0, u_hat_2_ax201, ax201, topology0,
                                topology2, MPI_COMM_WORLD);
    plan_0_2_ax201.forward(u_0, u_hat_2_ax201);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax201, ref_u_hat_2_ax201));

    plan_0_2_ax201.backward(u_hat_2_ax201, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 0 -> topo 2 with ax = {2, 1, 0}:
    // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
    // FFT2 ax = {0, 1} -> Transpose topo 2 -> FFT ax = {2}
    PencilPlan plan_0_2_ax210(exec, u_0, u_hat_2_ax210, ax210, topology0,
                                topology2, MPI_COMM_WORLD);
    plan_0_2_ax210.forward(u_0, u_hat_2_ax210);
    EXPECT_TRUE(allclose(exec, u_hat_2_ax210, ref_u_hat_2_ax210));

    plan_0_2_ax210.backward(u_hat_2_ax210, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 2 -> topo 1 with ax = {0, 1, 2}:
    // (n0/p, n1, n2) -> (n0/p, n1, n2/2+1) -> (n0, n1/p, n2/2+1)
    // FFT2 ax = {1, 2} -> Transpose -> FFT ax = {0}
    PencilPlan plan_2_1_ax012(exec, u_2, u_hat_1_ax012, ax012, topology2,
                                topology1, MPI_COMM_WORLD);
    plan_2_1_ax012.forward(u_2, u_hat_1_ax012);
    EXPECT_TRUE(allclose(exec, u_hat_1_ax012, ref_u_hat_1_ax012));

    plan_2_1_ax012.backward(u_hat_1_ax012, u_inv_2);
    EXPECT_TRUE(allclose(exec, u_inv_2, ref_u_inv_2, 1.0e-5, 1.0e-6));
    */

    // topo 0 -> topo 3 with ax = {0, 1, 2}:
    // (n0, n1/px, n2/py) -> (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
    // -> (n0/py, n1, (n2/2+1)/px) -> (n0, n1/py, (n2/2+1)/px)
    // -> (n0/px, n1/py, n2/2+1)
    // Transpose topo 2 -> FFT ax = {2} -> Transpose topo 4 -> FFT ax = {1}
    // -> Transpose topo 5 -> FFT ax = {0} -> Transpose topo 3
    PencilPlan plan_0_3_ax012(exec, u_0, u_hat_3_ax012, ax012, topology0,
                              topology3, MPI_COMM_WORLD);
    plan_0_3_ax012.forward(u_0, u_hat_3_ax012);
    EXPECT_TRUE(allclose(exec, u_hat_3_ax012, ref_u_hat_3_ax012));

    plan_0_3_ax012.backward(u_hat_3_ax012, u_inv_0);
    EXPECT_TRUE(allclose(exec, u_inv_0, ref_u_inv_0, 1.0e-5, 1.0e-6));

    // topo 3 -> topo 0 with ax = {0, 1, 2}:
    // (n0/px, n1/py, n2) -> (n0/px, n1/py, n2/2+1) -> (n0/px, n1, (n2/2+1)/py)
    // -> (n0, n1/px, (n2/2+1)/py)
    // FFT ax = {2} -> Transpose topo 1 -> FFT ax = {1} ->
    // Transpose topo 0 -> FFT ax = {0}
    PencilPlan plan_3_0_ax012(exec, u_3, u_hat_0_ax012, ax012, topology3,
                              topology0, MPI_COMM_WORLD);
    plan_3_0_ax012.forward(u_3, u_hat_0_ax012);
    EXPECT_TRUE(allclose(exec, u_hat_0_ax012, ref_u_hat_0_ax012));

    plan_3_0_ax012.backward(u_hat_0_ax012, u_inv_3);
    EXPECT_TRUE(allclose(exec, u_inv_3, ref_u_inv_3, 1.0e-5, 1.0e-6));
  }
}

}  // namespace

TYPED_TEST_SUITE(TestPencil1D, test_types);
// TYPED_TEST_SUITE(TestPencil2D, test_types);
TYPED_TEST_SUITE(TestPencil3D, test_types);

TYPED_TEST(TestPencil1D, View3D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  // test_pencil1D_view3D<float_type, layout_type>(this->m_npx, this->m_npx);
}

TYPED_TEST(TestPencil1D, View3D_C2C) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;

  // test_pencil1D_view3D<complex_type, layout_type>(this->m_npx, this->m_npx);
}

TYPED_TEST(TestPencil3D, View3D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_pencil3D_view3D<float_type, layout_type>(this->m_npx, this->m_npx);
}

TYPED_TEST(TestPencil3D, View3D_C2C) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;

  // test_pencil3D_view3D<complex_type, layout_type>(this->m_npx, this->m_npx);
}
