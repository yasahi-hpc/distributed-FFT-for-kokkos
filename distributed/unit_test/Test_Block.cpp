#include <mpi.h>
#include <gtest/gtest.h>
#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include "Block.hpp"
#include "MPI_Helper.hpp"
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
struct TestBlock : public ::testing::Test {
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
void test_block_view2D(std::size_t nprocs, int order = 0) {
  using View2DType    = Kokkos::View<T**, LayoutType, execution_space>;
  using View3DType    = Kokkos::View<T***, LayoutType, execution_space>;
  using map_type      = std::array<std::size_t, 2>;
  using extents_type  = std::array<std::size_t, 2>;
  using topology_type = std::array<std::size_t, 2>;

  map_type src_map{0, 1}, dst_map{1, 0};
  topology_type topology0{1, nprocs};
  topology_type topology1{nprocs, 1};

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
    Block trans_block_x2y(exec, u_x, u_y, send_buffer, recv_buffer, src_map, 0,
                          src_map, 1, MPI_COMM_WORLD);
    trans_block_x2y(u_x, u_y);

    EXPECT_TRUE(allclose(exec, u_y, u_y_ref));

    // Recover u_y from u_y_ref
    Kokkos::deep_copy(u_y, u_y_ref);

    Block trans_block_y2x(exec, u_y, u_x, send_buffer, recv_buffer, src_map, 1,
                          src_map, 0, MPI_COMM_WORLD);
    trans_block_y2x(u_y, u_x);
    EXPECT_TRUE(allclose(exec, u_x, u_x_ref));
  } else {
    // Order changed, and distribution changed
    Block trans_block_x2y(exec, u_x, u_y_T, send_buffer, recv_buffer, src_map,
                          0, dst_map, 1, MPI_COMM_WORLD);
    trans_block_x2y(u_x, u_y_T);

    EXPECT_TRUE(allclose(exec, u_y_T, u_y_T_ref));

    Block trans_block_y2x(exec, u_y, u_x_T, send_buffer, recv_buffer, src_map,
                          1, dst_map, 0, MPI_COMM_WORLD);
    trans_block_y2x(u_y, u_x_T);
    EXPECT_TRUE(allclose(exec, u_x_T, u_x_T_ref));
  }
}

}  // namespace

TYPED_TEST_SUITE(TestBlock, test_types);

TYPED_TEST(TestBlock, View2D_01) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_block_view2D<float_type, layout_type>(this->m_nprocs, 0);
}

TYPED_TEST(TestBlock, View2D_10) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_block_view2D<float_type, layout_type>(this->m_nprocs, 1);
}
