#include <mpi.h>
#include <gtest/gtest.h>
#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include "TransBlock.hpp"
#include "MPI_Helper.hpp"
#include "Helper.hpp"
#include "Extents.hpp"
#include "Test_Utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using test_types      = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>,
                                    std::pair<double, Kokkos::LayoutLeft>,
                                    std::pair<double, Kokkos::LayoutRight>>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct TestTransBlock : public ::testing::Test {
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

template <typename T, typename LayoutType>
void test_trans_block_view2D(std::size_t nprocs, int order = 0) {
  using View2DType    = Kokkos::View<T**, LayoutType, execution_space>;
  using View3DType    = Kokkos::View<T***, LayoutType, execution_space>;
  using map_type      = std::array<std::size_t, 2>;
  using extents_type  = std::array<std::size_t, 2>;
  using topology_type = std::array<std::size_t, 2>;

  map_type src_map{0, 1}, dst_map{1, 0};
  topology_type topology0{1, nprocs}, topology1{nprocs, 1};

  const std::size_t n0 = 8, n1 = 7;
  extents_type global_extents{n0, n1};

  auto [local_extents_t0, local_starts_t0] =
      get_local_extents(global_extents, topology0, MPI_COMM_WORLD);
  auto [local_extents_t1, local_starts_t1] =
      get_local_extents(global_extents, topology1, MPI_COMM_WORLD);

  View2DType gu("gu", n0, n1);

  View2DType u_x("u_x", local_extents_t0.at(0), local_extents_t0.at(1)),
      u_y("u_y", local_extents_t1.at(0), local_extents_t1.at(1)),
      u_x_ref("u_x_ref", local_extents_t0.at(0), local_extents_t0.at(1)),
      u_y_ref("u_y_ref", local_extents_t1.at(0), local_extents_t1.at(1));
  View2DType u_x_T("u_x_T", local_extents_t0.at(1), local_extents_t0.at(0)),
      u_y_T("u_y_T", local_extents_t1.at(1), local_extents_t1.at(0)),
      u_x_T_ref("u_x_T_ref", local_extents_t0.at(1), local_extents_t0.at(0)),
      u_y_T_ref("u_y_T_ref", local_extents_t1.at(1), local_extents_t1.at(0));

  // Prepare buffer data
  auto buffer_01 =
      get_buffer_extents<LayoutType>(global_extents, topology0, topology1);
  View3DType send_buffer("send_buffer",
                         KokkosFFT::Impl::create_layout<LayoutType>(buffer_01));
  View3DType recv_buffer("recv_buffer",
                         KokkosFFT::Impl::create_layout<LayoutType>(buffer_01));

  // Initialization
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::fill_random(gu, random_pool, 1.0);

  auto h_gu    = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gu);
  auto h_u_x   = Kokkos::create_mirror_view(u_x);
  auto h_u_y   = Kokkos::create_mirror_view(u_y);
  auto h_u_x_T = Kokkos::create_mirror_view(u_x_T);
  auto h_u_y_T = Kokkos::create_mirror_view(u_y_T);
  for (std::size_t i0 = 0; i0 < h_u_x.extent(0); i0++) {
    for (std::size_t i1 = 0; i1 < h_u_x.extent(1); i1++) {
      h_u_x(i0, i1)   = h_gu(i0, i1 + local_starts_t0.at(1));
      h_u_x_T(i1, i0) = h_u_x(i0, i1);
    }
  }

  for (std::size_t i0 = 0; i0 < h_u_y.extent(0); i0++) {
    for (std::size_t i1 = 0; i1 < h_u_y.extent(1); i1++) {
      h_u_y(i0, i1)   = h_gu(i0 + local_starts_t1.at(0), i1);
      h_u_y_T(i1, i0) = h_u_y(i0, i1);
    }
  }
  Kokkos::deep_copy(u_x, h_u_x);
  Kokkos::deep_copy(u_y, h_u_y);
  Kokkos::deep_copy(u_x_T, h_u_x_T);
  Kokkos::deep_copy(u_y_T, h_u_y_T);
  Kokkos::deep_copy(u_x_ref, u_x);
  Kokkos::deep_copy(u_y_ref, u_y);
  Kokkos::deep_copy(u_x_T_ref, u_x_T);
  Kokkos::deep_copy(u_y_T_ref, u_y_T);

  execution_space exec;

  if (order == 0) {
    // Order reserved, but distribution changed
    TransBlock trans_block_x2y(exec, buffer_01, src_map, 0, src_map, 1,
                               MPI_COMM_WORLD);
    trans_block_x2y(u_x, u_y, send_buffer, recv_buffer,
                    KokkosFFT::Direction::forward);

    EXPECT_TRUE(allclose(exec, u_y, u_y_ref));

    // Recover u_y from u_y_ref
    Kokkos::deep_copy(u_y, u_y_ref);

    trans_block_x2y(u_y, u_x, send_buffer, recv_buffer,
                    KokkosFFT::Direction::backward);
    EXPECT_TRUE(allclose(exec, u_x, u_x_ref));
  } else {
    // Order changed, and distribution changed
    TransBlock trans_block_x2y(exec, buffer_01, src_map, 0, dst_map, 1,
                               MPI_COMM_WORLD);
    trans_block_x2y(u_x, u_y_T, send_buffer, recv_buffer,
                    KokkosFFT::Direction::forward);

    EXPECT_TRUE(allclose(exec, u_y_T, u_y_T_ref));

    TransBlock trans_block_y2x(exec, buffer_01, src_map, 1, dst_map, 0,
                               MPI_COMM_WORLD);
    trans_block_y2x(u_y, u_x_T, send_buffer, recv_buffer,
                    KokkosFFT::Direction::forward);
    EXPECT_TRUE(allclose(exec, u_x_T, u_x_T_ref));
  }
}

template <typename T, typename LayoutType>
void test_trans_block_view3D(std::size_t npx, std::size_t npy) {
  using View3DType      = Kokkos::View<T***, LayoutType, execution_space>;
  using View4DType      = Kokkos::View<T****, LayoutType, execution_space>;
  using map_type        = std::array<int, 3>;
  using extents_type    = std::array<std::size_t, 3>;
  using topology_r_type = Topology<std::size_t, 3, Kokkos::LayoutRight>;
  using topology_l_type = Topology<std::size_t, 3, Kokkos::LayoutLeft>;

  extents_type map012{0, 1, 2}, map021{0, 2, 1}, map102{1, 0, 2},
      map120{1, 2, 0}, map201{2, 0, 1}, map210{2, 1, 0};
  map_type int_map021{0, 2, 1}, int_map102{1, 0, 2}, int_map120{1, 2, 0},
      int_map201{2, 0, 1}, int_map210{2, 1, 0};

  // Define, x, y and z pencils
  topology_r_type topology0{1, npx, npy}, topology1{npx, 1, npy},
      topology2{npx, npy, 1};
  topology_l_type topology3{npy, npx, 1};

  const std::size_t n0 = 8, n1 = 7, n2 = 5;
  extents_type global_extents{n0, n1, n2};

  auto [local_extents_t0, local_starts_t0] =
      get_local_extents(global_extents, topology0, MPI_COMM_WORLD);
  auto [local_extents_t1, local_starts_t1] =
      get_local_extents(global_extents, topology1, MPI_COMM_WORLD);
  auto [local_extents_t2, local_starts_t2] =
      get_local_extents(global_extents, topology2, MPI_COMM_WORLD);
  auto [local_extents_t3, local_starts_t3] =
      get_local_extents(global_extents, topology3, MPI_COMM_WORLD);

  View3DType gu("gu", n0, n1, n2);

  // Data in Topology 0 (X-pencil)
  View3DType u_0_012(
      "u_0_012", KokkosFFT::Impl::create_layout<LayoutType>(local_extents_t0)),
      u_0_021("u_0_021", KokkosFFT::Impl::create_layout<LayoutType>(
                             get_mapped_extents(local_extents_t0, map021))),
      u_0_102("u_0_102", KokkosFFT::Impl::create_layout<LayoutType>(
                             get_mapped_extents(local_extents_t0, map102))),
      u_0_120("u_0_120", KokkosFFT::Impl::create_layout<LayoutType>(
                             get_mapped_extents(local_extents_t0, map120))),
      u_0_201("u_0_201", KokkosFFT::Impl::create_layout<LayoutType>(
                             get_mapped_extents(local_extents_t0, map201))),
      u_0_210("u_0_210", KokkosFFT::Impl::create_layout<LayoutType>(
                             get_mapped_extents(local_extents_t0, map210))),
      ref_u_0_012("ref_u_0_012",
                  KokkosFFT::Impl::create_layout<LayoutType>(local_extents_t0)),
      ref_u_0_021("ref_u_0_021",
                  KokkosFFT::Impl::create_layout<LayoutType>(
                      get_mapped_extents(local_extents_t0, map021))),
      ref_u_0_102("ref_u_0_102",
                  KokkosFFT::Impl::create_layout<LayoutType>(
                      get_mapped_extents(local_extents_t0, map102))),
      ref_u_0_120("ref_u_0_120",
                  KokkosFFT::Impl::create_layout<LayoutType>(
                      get_mapped_extents(local_extents_t0, map120))),
      ref_u_0_201("ref_u_0_201",
                  KokkosFFT::Impl::create_layout<LayoutType>(
                      get_mapped_extents(local_extents_t0, map201))),
      ref_u_0_210("ref_u_0_210",
                  KokkosFFT::Impl::create_layout<LayoutType>(
                      get_mapped_extents(local_extents_t0, map210)));

  // Data in Topology 1 (Y-pencil)
  View3DType u_1_012(
      "u_1_012", KokkosFFT::Impl::create_layout<LayoutType>(local_extents_t1)),
      u_1_021("u_1_021", KokkosFFT::Impl::create_layout<LayoutType>(
                             get_mapped_extents(local_extents_t1, map021))),
      u_1_102("u_1_102", KokkosFFT::Impl::create_layout<LayoutType>(
                             get_mapped_extents(local_extents_t1, map102))),
      u_1_120("u_1_120", KokkosFFT::Impl::create_layout<LayoutType>(
                             get_mapped_extents(local_extents_t1, map120))),
      u_1_201("u_1_201", KokkosFFT::Impl::create_layout<LayoutType>(
                             get_mapped_extents(local_extents_t1, map201))),
      u_1_210("u_1_210", KokkosFFT::Impl::create_layout<LayoutType>(
                             get_mapped_extents(local_extents_t1, map210))),
      ref_u_1_012("ref_u_1_012",
                  KokkosFFT::Impl::create_layout<LayoutType>(local_extents_t1)),
      ref_u_1_021("ref_u_1_021",
                  KokkosFFT::Impl::create_layout<LayoutType>(
                      get_mapped_extents(local_extents_t1, map021))),
      ref_u_1_102("ref_u_1_102",
                  KokkosFFT::Impl::create_layout<LayoutType>(
                      get_mapped_extents(local_extents_t1, map102))),
      ref_u_1_120("ref_u_1_120",
                  KokkosFFT::Impl::create_layout<LayoutType>(
                      get_mapped_extents(local_extents_t1, map120))),
      ref_u_1_201("ref_u_1_201",
                  KokkosFFT::Impl::create_layout<LayoutType>(
                      get_mapped_extents(local_extents_t1, map201))),
      ref_u_1_210("ref_u_1_210",
                  KokkosFFT::Impl::create_layout<LayoutType>(
                      get_mapped_extents(local_extents_t1, map210)));

  // Data in Topology 2 (Z-pencil)
  View3DType u_2_012(
      "u_2_012", KokkosFFT::Impl::create_layout<LayoutType>(local_extents_t2)),
      u_2_021("u_2_021", KokkosFFT::Impl::create_layout<LayoutType>(
                             get_mapped_extents(local_extents_t2, map021))),
      u_2_102("u_2_102", KokkosFFT::Impl::create_layout<LayoutType>(
                             get_mapped_extents(local_extents_t2, map102))),
      u_2_120("u_2_120", KokkosFFT::Impl::create_layout<LayoutType>(
                             get_mapped_extents(local_extents_t2, map120))),
      u_2_201("u_2_201", KokkosFFT::Impl::create_layout<LayoutType>(
                             get_mapped_extents(local_extents_t2, map201))),
      u_2_210("u_2_210", KokkosFFT::Impl::create_layout<LayoutType>(
                             get_mapped_extents(local_extents_t2, map210))),
      ref_u_2_012("ref_u_2_012",
                  KokkosFFT::Impl::create_layout<LayoutType>(local_extents_t2)),
      ref_u_2_021("ref_u_2_021",
                  KokkosFFT::Impl::create_layout<LayoutType>(
                      get_mapped_extents(local_extents_t2, map021))),
      ref_u_2_102("ref_u_2_102",
                  KokkosFFT::Impl::create_layout<LayoutType>(
                      get_mapped_extents(local_extents_t2, map102))),
      ref_u_2_120("ref_u_2_120",
                  KokkosFFT::Impl::create_layout<LayoutType>(
                      get_mapped_extents(local_extents_t2, map120))),
      ref_u_2_201("ref_u_201",
                  KokkosFFT::Impl::create_layout<LayoutType>(
                      get_mapped_extents(local_extents_t2, map201))),
      ref_u_2_210("ref_u_2_210",
                  KokkosFFT::Impl::create_layout<LayoutType>(
                      get_mapped_extents(local_extents_t2, map210)));

  // Data in Topology 3 (Z-pencil)
  View3DType u_3_012(
      "u_3_012", KokkosFFT::Impl::create_layout<LayoutType>(local_extents_t3)),
      u_3_021("u_3_021", KokkosFFT::Impl::create_layout<LayoutType>(
                             get_mapped_extents(local_extents_t3, map021))),
      u_3_102("u_3_102", KokkosFFT::Impl::create_layout<LayoutType>(
                             get_mapped_extents(local_extents_t3, map102))),
      u_3_120("u_3_120", KokkosFFT::Impl::create_layout<LayoutType>(
                             get_mapped_extents(local_extents_t3, map120))),
      u_3_201("u_3_201", KokkosFFT::Impl::create_layout<LayoutType>(
                             get_mapped_extents(local_extents_t3, map201))),
      u_3_210("u_3_210", KokkosFFT::Impl::create_layout<LayoutType>(
                             get_mapped_extents(local_extents_t3, map210))),
      ref_u_3_012("ref_u_3_012",
                  KokkosFFT::Impl::create_layout<LayoutType>(local_extents_t3)),
      ref_u_3_021("ref_u_3_021",
                  KokkosFFT::Impl::create_layout<LayoutType>(
                      get_mapped_extents(local_extents_t3, map021))),
      ref_u_3_102("ref_u_3_102",
                  KokkosFFT::Impl::create_layout<LayoutType>(
                      get_mapped_extents(local_extents_t3, map102))),
      ref_u_3_120("ref_u_3_120",
                  KokkosFFT::Impl::create_layout<LayoutType>(
                      get_mapped_extents(local_extents_t3, map120))),
      ref_u_3_201("ref_u_3_201",
                  KokkosFFT::Impl::create_layout<LayoutType>(
                      get_mapped_extents(local_extents_t3, map201))),
      ref_u_3_210("ref_u_3_210",
                  KokkosFFT::Impl::create_layout<LayoutType>(
                      get_mapped_extents(local_extents_t3, map210)));

  // Prepare buffer data
  auto buffer_01 =
      get_buffer_extents<LayoutType>(global_extents, topology0, topology1);
  auto buffer_03 =
      get_buffer_extents<LayoutType>(global_extents, topology0, topology3);
  auto buffer_12 =
      get_buffer_extents<LayoutType>(global_extents, topology1, topology2);
  View4DType send_buffer01(
      "send_buffer01", KokkosFFT::Impl::create_layout<LayoutType>(buffer_01));
  View4DType recv_buffer01(
      "recv_buffer01", KokkosFFT::Impl::create_layout<LayoutType>(buffer_01));
  View4DType send_buffer03(
      "send_buffer03", KokkosFFT::Impl::create_layout<LayoutType>(buffer_03));
  View4DType recv_buffer03(
      "recv_buffer03", KokkosFFT::Impl::create_layout<LayoutType>(buffer_03));
  View4DType send_buffer12(
      "send_buffer12", KokkosFFT::Impl::create_layout<LayoutType>(buffer_12));
  View4DType recv_buffer12(
      "recv_buffer12", KokkosFFT::Impl::create_layout<LayoutType>(buffer_12));

  // Initialization
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::fill_random(gu, random_pool, 1.0);

  execution_space exec;

  // Topo 0
  Kokkos::pair<std::size_t, std::size_t> range_gu0_dim1(
      local_starts_t0.at(1), local_starts_t0.at(1) + local_extents_t0.at(1)),
      range_gu0_dim2(local_starts_t0.at(2),
                     local_starts_t0.at(2) + local_extents_t0.at(2));
  auto sub_gu_0 =
      Kokkos::subview(gu, Kokkos::ALL, range_gu0_dim1, range_gu0_dim2);
  Kokkos::deep_copy(u_0_012, sub_gu_0);
  Kokkos::deep_copy(ref_u_0_012, sub_gu_0);
  safe_transpose(exec, u_0_012, ref_u_0_021, int_map021);
  safe_transpose(exec, u_0_012, ref_u_0_102, int_map102);
  safe_transpose(exec, u_0_012, ref_u_0_120, int_map120);
  safe_transpose(exec, u_0_012, ref_u_0_201, int_map201);
  safe_transpose(exec, u_0_012, ref_u_0_210, int_map210);

  // Topo 1
  Kokkos::pair<std::size_t, std::size_t> range_gu1_dim0(
      local_starts_t1.at(0), local_starts_t1.at(0) + local_extents_t1.at(0)),
      range_gu1_dim2(local_starts_t1.at(2),
                     local_starts_t1.at(2) + local_extents_t1.at(2));
  auto sub_gu_1 =
      Kokkos::subview(gu, range_gu1_dim0, Kokkos::ALL, range_gu1_dim2);
  Kokkos::deep_copy(u_1_012, sub_gu_1);
  Kokkos::deep_copy(ref_u_1_012, sub_gu_1);
  safe_transpose(exec, u_1_012, ref_u_1_021, int_map021);
  safe_transpose(exec, u_1_012, ref_u_1_102, int_map102);
  safe_transpose(exec, u_1_012, ref_u_1_120, int_map120);
  safe_transpose(exec, u_1_012, ref_u_1_201, int_map201);
  safe_transpose(exec, u_1_012, ref_u_1_210, int_map210);

  // Topo 2
  Kokkos::pair<std::size_t, std::size_t> range_gu2_dim0(
      local_starts_t2.at(0), local_starts_t2.at(0) + local_extents_t2.at(0)),
      range_gu2_dim1(local_starts_t2.at(1),
                     local_starts_t2.at(1) + local_extents_t2.at(1));
  auto sub_gu_2 =
      Kokkos::subview(gu, range_gu2_dim0, range_gu2_dim1, Kokkos::ALL);
  Kokkos::deep_copy(u_2_012, sub_gu_2);
  Kokkos::deep_copy(ref_u_2_012, sub_gu_2);
  safe_transpose(exec, u_2_012, ref_u_2_021, int_map021);
  safe_transpose(exec, u_2_012, ref_u_2_102, int_map102);
  safe_transpose(exec, u_2_012, ref_u_2_120, int_map120);
  safe_transpose(exec, u_2_012, ref_u_2_201, int_map201);
  safe_transpose(exec, u_2_012, ref_u_2_210, int_map210);

  // Topo 3
  Kokkos::pair<std::size_t, std::size_t> range_gu3_dim0(
      local_starts_t3.at(0), local_starts_t3.at(0) + local_extents_t3.at(0)),
      range_gu3_dim1(local_starts_t3.at(1),
                     local_starts_t3.at(1) + local_extents_t3.at(1));
  auto sub_gu_3 =
      Kokkos::subview(gu, range_gu3_dim0, range_gu3_dim1, Kokkos::ALL);
  Kokkos::deep_copy(u_3_012, sub_gu_3);
  Kokkos::deep_copy(ref_u_3_012, sub_gu_3);
  safe_transpose(exec, u_3_012, ref_u_3_021, int_map021);
  safe_transpose(exec, u_3_012, ref_u_3_102, int_map102);
  safe_transpose(exec, u_3_012, ref_u_3_120, int_map120);
  safe_transpose(exec, u_3_012, ref_u_3_201, int_map201);
  safe_transpose(exec, u_3_012, ref_u_3_210, int_map210);

  // Define cart comm
  std::vector<int> dims;
  for (auto& dim : topology0) {
    if (dim > 1) {
      dims.push_back(static_cast<int>(dim));
    }
  }

  MPI_Comm cart_comm;
  int periods[2] = {1, 1};  // Periodic in all directions
  ::MPI_Cart_create(MPI_COMM_WORLD, 2, dims.data(), periods, 1, &cart_comm);
  // split into row‐ and col‐ communicators

  MPI_Comm row_comm, col_comm;
  int remain_dims[2];

  // keep Y‐axis for row_comm (all procs with same px)
  remain_dims[0] = 1;
  remain_dims[1] = 0;
  ::MPI_Cart_sub(cart_comm, remain_dims, &row_comm);

  // keep X‐axis for col_comm (all procs with same py)
  remain_dims[0] = 0;
  remain_dims[1] = 1;
  ::MPI_Cart_sub(cart_comm, remain_dims, &col_comm);

  {
    TransBlock trans_block_0_1(exec, buffer_01, map012, 0, map021, 1, row_comm);
    trans_block_0_1(u_0_012, u_1_021, send_buffer01, recv_buffer01,
                    KokkosFFT::Direction::forward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_1_021, ref_u_1_021));

    trans_block_0_1(u_1_021, u_0_012, send_buffer01, recv_buffer01,
                    KokkosFFT::Direction::backward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_0_012, ref_u_0_012));
  }

  {
    TransBlock trans_block_0_1(exec, buffer_01, map012, 0, map102, 1, row_comm);
    trans_block_0_1(u_0_012, u_1_102, send_buffer01, recv_buffer01,
                    KokkosFFT::Direction::forward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_1_102, ref_u_1_102));

    trans_block_0_1(u_1_102, u_0_012, send_buffer01, recv_buffer01,
                    KokkosFFT::Direction::backward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_0_012, ref_u_0_012));
  }

  {
    TransBlock trans_block_0_1(exec, buffer_01, map012, 0, map120, 1, row_comm);
    trans_block_0_1(u_0_012, u_1_120, send_buffer01, recv_buffer01,
                    KokkosFFT::Direction::forward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_1_120, ref_u_1_120));

    trans_block_0_1(u_1_120, u_0_012, send_buffer01, recv_buffer01,
                    KokkosFFT::Direction::backward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_0_012, ref_u_0_012));
  }

  {
    TransBlock trans_block_0_1(exec, buffer_01, map012, 0, map201, 1, row_comm);
    trans_block_0_1(u_0_012, u_1_201, send_buffer01, recv_buffer01,
                    KokkosFFT::Direction::forward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_1_201, ref_u_1_201));

    trans_block_0_1(u_1_201, u_0_012, send_buffer01, recv_buffer01,
                    KokkosFFT::Direction::backward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_0_012, ref_u_0_012));
  }

  {
    TransBlock trans_block_0_1(exec, buffer_01, map012, 0, map210, 1, row_comm);
    trans_block_0_1(u_0_012, u_1_210, send_buffer01, recv_buffer01,
                    KokkosFFT::Direction::forward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_1_210, ref_u_1_210));

    trans_block_0_1(u_1_210, u_0_012, send_buffer01, recv_buffer01,
                    KokkosFFT::Direction::backward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_0_012, ref_u_0_012));
  }

  {
    TransBlock trans_block_0_3(exec, buffer_03, map012, 0, map012, 2, col_comm);
    trans_block_0_3(u_0_012, u_3_012, send_buffer03, recv_buffer03,
                    KokkosFFT::Direction::forward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_3_012, ref_u_3_012));

    trans_block_0_3(u_3_012, u_0_012, send_buffer03, recv_buffer03,
                    KokkosFFT::Direction::backward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_0_012, ref_u_0_012));
  }

  {
    TransBlock trans_block_0_3(exec, buffer_03, map012, 0, map021, 2, col_comm);
    trans_block_0_3(u_0_012, u_3_021, send_buffer03, recv_buffer03,
                    KokkosFFT::Direction::forward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_3_021, ref_u_3_021));

    trans_block_0_3(u_3_021, u_0_012, send_buffer03, recv_buffer03,
                    KokkosFFT::Direction::backward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_0_012, ref_u_0_012));
  }

  {
    TransBlock trans_block_0_3(exec, buffer_03, map012, 0, map102, 2, col_comm);
    trans_block_0_3(u_0_012, u_3_102, send_buffer03, recv_buffer03,
                    KokkosFFT::Direction::forward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_3_102, ref_u_3_102));

    trans_block_0_3(u_3_102, u_0_012, send_buffer03, recv_buffer03,
                    KokkosFFT::Direction::backward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_0_012, ref_u_0_012));
  }

  {
    TransBlock trans_block_0_3(exec, buffer_03, map012, 0, map120, 2, col_comm);
    trans_block_0_3(u_0_012, u_3_120, send_buffer03, recv_buffer03,
                    KokkosFFT::Direction::forward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_3_120, ref_u_3_120));

    trans_block_0_3(u_3_120, u_0_012, send_buffer03, recv_buffer03,
                    KokkosFFT::Direction::backward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_0_012, ref_u_0_012));
  }

  {
    TransBlock trans_block_0_3(exec, buffer_03, map012, 0, map201, 2, col_comm);
    trans_block_0_3(u_0_012, u_3_201, send_buffer03, recv_buffer03,
                    KokkosFFT::Direction::forward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_3_201, ref_u_3_201));

    trans_block_0_3(u_3_201, u_0_012, send_buffer03, recv_buffer03,
                    KokkosFFT::Direction::backward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_0_012, ref_u_0_012));
  }

  {
    TransBlock trans_block_0_3(exec, buffer_03, map012, 0, map210, 2, col_comm);
    trans_block_0_3(u_0_012, u_3_210, send_buffer03, recv_buffer03,
                    KokkosFFT::Direction::forward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_3_210, ref_u_3_210));

    trans_block_0_3(u_3_210, u_0_012, send_buffer03, recv_buffer03,
                    KokkosFFT::Direction::backward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_0_012, ref_u_0_012));
  }

  {
    TransBlock trans_block_1_2(exec, buffer_12, map012, 1, map012, 2, col_comm);
    trans_block_1_2(u_1_012, u_2_012, send_buffer12, recv_buffer12,
                    KokkosFFT::Direction::forward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_2_012, ref_u_2_012));

    trans_block_1_2(u_2_012, u_1_012, send_buffer12, recv_buffer12,
                    KokkosFFT::Direction::backward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_1_012, ref_u_1_012));
  }

  {
    TransBlock trans_block_1_2(exec, buffer_12, map012, 1, map021, 2, col_comm);
    trans_block_1_2(u_1_012, u_2_021, send_buffer12, recv_buffer12,
                    KokkosFFT::Direction::forward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_2_021, ref_u_2_021));

    trans_block_1_2(u_2_021, u_1_012, send_buffer12, recv_buffer12,
                    KokkosFFT::Direction::backward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_1_012, ref_u_1_012));
  }

  {
    TransBlock trans_block_1_2(exec, buffer_12, map012, 1, map102, 2, col_comm);
    trans_block_1_2(u_1_012, u_2_102, send_buffer12, recv_buffer12,
                    KokkosFFT::Direction::forward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_2_102, ref_u_2_102));

    trans_block_1_2(u_2_102, u_1_012, send_buffer12, recv_buffer12,
                    KokkosFFT::Direction::backward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_1_012, ref_u_1_012));
  }

  {
    TransBlock trans_block_1_2(exec, buffer_12, map012, 1, map120, 2, col_comm);
    trans_block_1_2(u_1_012, u_2_120, send_buffer12, recv_buffer12,
                    KokkosFFT::Direction::forward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_2_120, ref_u_2_120));

    trans_block_1_2(u_2_120, u_1_012, send_buffer12, recv_buffer12,
                    KokkosFFT::Direction::backward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_1_012, ref_u_1_012));
  }

  {
    TransBlock trans_block_1_2(exec, buffer_12, map012, 1, map201, 2, col_comm);
    trans_block_1_2(u_1_012, u_2_201, send_buffer12, recv_buffer12,
                    KokkosFFT::Direction::forward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_2_201, ref_u_2_201));

    trans_block_1_2(u_2_201, u_1_012, send_buffer12, recv_buffer12,
                    KokkosFFT::Direction::backward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_1_012, ref_u_1_012));
  }

  {
    TransBlock trans_block_1_2(exec, buffer_12, map012, 1, map210, 2, col_comm);
    trans_block_1_2(u_1_012, u_2_210, send_buffer12, recv_buffer12,
                    KokkosFFT::Direction::forward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_2_210, ref_u_2_210));

    trans_block_1_2(u_2_210, u_1_012, send_buffer12, recv_buffer12,
                    KokkosFFT::Direction::backward);
    exec.fence();
    EXPECT_TRUE(allclose(exec, u_1_012, ref_u_1_012));
  }
}

}  // namespace

TYPED_TEST_SUITE(TestTransBlock, test_types);

TYPED_TEST(TestTransBlock, View2D_01) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_trans_block_view2D<float_type, layout_type>(this->m_nprocs, 0);
}

TYPED_TEST(TestTransBlock, View2D_10) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_trans_block_view2D<float_type, layout_type>(this->m_nprocs, 1);
}

TYPED_TEST(TestTransBlock, View3D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  if (this->m_nprocs == 1 || this->m_npx * this->m_npx != this->m_nprocs) {
    GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                    "for this test";
  }

  test_trans_block_view3D<float_type, layout_type>(this->m_npx, this->m_npx);
}
