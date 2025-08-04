#include <mpi.h>
#include <sstream>
#include <iostream>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
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
struct TestMPIHelper : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;

  std::size_t m_rank   = 0;
  std::size_t m_nprocs = 1;
  std::size_t m_npx    = 1;

  virtual void SetUp() {
    int rank, nprocs;
    ::MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    m_rank   = rank;
    m_nprocs = nprocs;
    m_npx    = std::sqrt(m_nprocs);
  }
};

template <typename T, typename LayoutType>
void test_get_global_shape2D(std::size_t rank, std::size_t nprocs) {
  using topology_type = std::array<std::size_t, 2>;
  using ViewType      = Kokkos::View<T**, LayoutType, execution_space>;

  topology_type topology0{1, nprocs}, topology1{nprocs, 1};

  const std::size_t gn0 = 19, gn1 = 32;
  const std::size_t n0_t0           = gn0;
  const std::size_t n1_t0_quotient  = (gn1 - 1) / nprocs + 1;
  const std::size_t n1_t0_remainder = gn1 - n1_t0_quotient * (nprocs - 1);
  const std::size_t n1_t0 =
      rank != (nprocs - 1) ? n1_t0_quotient : n1_t0_remainder;

  const std::size_t n0_t1_quotient  = (gn0 - 1) / nprocs + 1;
  const std::size_t n0_t1_remainder = gn0 - n0_t1_quotient * (nprocs - 1);
  const std::size_t n0_t1 =
      rank != (nprocs - 1) ? n0_t1_quotient : n0_t1_remainder;
  const std::size_t n1_t1 = gn1;

  ViewType v0("v0", n0_t0, n1_t0);
  ViewType v1("v1", n0_t1, n1_t1);

  auto global_shape_t0 = get_global_shape(v0, topology0, MPI_COMM_WORLD);
  auto global_shape_t1 = get_global_shape(v1, topology1, MPI_COMM_WORLD);

  topology_type ref_global_shape{gn0, gn1};

  EXPECT_EQ(global_shape_t0, ref_global_shape);
  EXPECT_EQ(global_shape_t1, ref_global_shape);
}

template <typename T, typename LayoutType>
void test_get_global_shape3D(std::size_t rank, std::size_t npx,
                             std::size_t npy) {
  using extents_type    = std::array<std::size_t, 3>;
  using topology_r_type = Topology<std::size_t, 3, Kokkos::LayoutRight>;
  using topology_l_type = Topology<std::size_t, 3, Kokkos::LayoutLeft>;
  using ViewType        = Kokkos::View<T***, LayoutType, execution_space>;

  topology_r_type topology0{1, npx, npy}, topology1{npx, 1, npy};
  topology_l_type topology2{npy, npx, 1};

  std::size_t rx = rank / npy, ry = rank % npy;

  auto distribute_extents = [&](std::size_t n, std::size_t r, std::size_t t) {
    std::size_t quotient  = n / t;
    std::size_t remainder = n % t;
    return r < remainder ? (quotient + 1) : quotient;
  };

  const std::size_t gn0 = 19, gn1 = 32, gn2 = 25;
  const std::size_t n0_t0 = gn0;
  const std::size_t n1_t0 = distribute_extents(gn1, rx, npx);
  const std::size_t n2_t0 = distribute_extents(gn2, ry, npy);

  const std::size_t n0_t1 = distribute_extents(gn0, rx, npx);
  const std::size_t n1_t1 = gn1;
  const std::size_t n2_t1 = distribute_extents(gn2, ry, npy);

  const std::size_t n0_t2 = distribute_extents(gn0, ry, npy);
  const std::size_t n1_t2 = distribute_extents(gn1, rx, npx);
  const std::size_t n2_t2 = gn2;

  ViewType v0("v0", n0_t0, n1_t0, n2_t0);
  ViewType v1("v1", n0_t1, n1_t1, n2_t1);
  ViewType v2("v2", n0_t2, n1_t2, n2_t2);

  auto global_shape_t0 = get_global_shape(v0, topology0, MPI_COMM_WORLD);
  auto global_shape_t1 = get_global_shape(v1, topology1, MPI_COMM_WORLD);
  auto global_shape_t2 = get_global_shape(v2, topology2, MPI_COMM_WORLD);

  extents_type ref_global_shape{gn0, gn1, gn2};

  EXPECT_EQ(global_shape_t0, ref_global_shape);
  EXPECT_EQ(global_shape_t1, ref_global_shape);
  EXPECT_EQ(global_shape_t2, ref_global_shape);
}

void test_rank_to_coord() {
  using topology_r_1D_type = Topology<std::size_t, 1>;
  using topology_r_2D_type = Topology<std::size_t, 2>;
  using topology_r_3D_type = Topology<std::size_t, 3>;
  using topology_r_4D_type = Topology<std::size_t, 4>;

  using extents_1D_type = std::array<std::size_t, 1>;
  using extents_2D_type = std::array<std::size_t, 2>;
  using extents_3D_type = std::array<std::size_t, 3>;
  using extents_4D_type = std::array<std::size_t, 4>;

  topology_r_1D_type topology1{2};
  topology_r_2D_type topology2{1, 2}, topology2_2{4, 2};
  topology_r_3D_type topology3{1, 2, 1}, topology3_2{4, 2, 1};
  topology_r_4D_type topology4{1, 4, 1, 1}, topology4_2{1, 1, 4, 2};

  extents_1D_type ref_coord1_rank0{0}, ref_coord1_rank1{1};
  extents_2D_type ref_coord2_rank0{0, 0}, ref_coord2_rank1{0, 1};
  extents_2D_type ref_coord2_2_rank0{0, 0}, ref_coord2_2_rank1{0, 1},
      ref_coord2_2_rank2{1, 0}, ref_coord2_2_rank3{1, 1},
      ref_coord2_2_rank4{2, 0}, ref_coord2_2_rank5{2, 1},
      ref_coord2_2_rank6{3, 0}, ref_coord2_2_rank7{3, 1};
  extents_3D_type ref_coord3_rank0{0, 0, 0}, ref_coord3_rank1{0, 1, 0};
  extents_3D_type ref_coord3_2_rank0{0, 0, 0}, ref_coord3_2_rank1{0, 1, 0},
      ref_coord3_2_rank2{1, 0, 0}, ref_coord3_2_rank3{1, 1, 0},
      ref_coord3_2_rank4{2, 0, 0}, ref_coord3_2_rank5{2, 1, 0},
      ref_coord3_2_rank6{3, 0, 0}, ref_coord3_2_rank7{3, 1, 0};
  extents_4D_type ref_coord4_rank0{0, 0, 0, 0}, ref_coord4_rank1{0, 1, 0, 0},
      ref_coord4_rank2{0, 2, 0, 0}, ref_coord4_rank3{0, 3, 0, 0};
  extents_4D_type ref_coord4_2_rank0{0, 0, 0, 0},
      ref_coord4_2_rank1{0, 0, 0, 1}, ref_coord4_2_rank2{0, 0, 1, 0},
      ref_coord4_2_rank3{0, 0, 1, 1}, ref_coord4_2_rank4{0, 0, 2, 0},
      ref_coord4_2_rank5{0, 0, 2, 1}, ref_coord4_2_rank6{0, 0, 3, 0},
      ref_coord4_2_rank7{0, 0, 3, 1};

  auto coord1_rank0   = rank_to_coord(topology1, 0);
  auto coord1_rank1   = rank_to_coord(topology1, 1);
  auto coord2_rank0   = rank_to_coord(topology2, 0);
  auto coord2_rank1   = rank_to_coord(topology2, 1);
  auto coord2_2_rank0 = rank_to_coord(topology2_2, 0);
  auto coord2_2_rank1 = rank_to_coord(topology2_2, 1);
  auto coord2_2_rank2 = rank_to_coord(topology2_2, 2);
  auto coord2_2_rank3 = rank_to_coord(topology2_2, 3);
  auto coord2_2_rank4 = rank_to_coord(topology2_2, 4);
  auto coord2_2_rank5 = rank_to_coord(topology2_2, 5);
  auto coord2_2_rank6 = rank_to_coord(topology2_2, 6);
  auto coord2_2_rank7 = rank_to_coord(topology2_2, 7);

  auto coord3_rank0   = rank_to_coord(topology3, 0);
  auto coord3_rank1   = rank_to_coord(topology3, 1);
  auto coord3_2_rank0 = rank_to_coord(topology3_2, 0);
  auto coord3_2_rank1 = rank_to_coord(topology3_2, 1);
  auto coord3_2_rank2 = rank_to_coord(topology3_2, 2);
  auto coord3_2_rank3 = rank_to_coord(topology3_2, 3);
  auto coord3_2_rank4 = rank_to_coord(topology3_2, 4);
  auto coord3_2_rank5 = rank_to_coord(topology3_2, 5);
  auto coord3_2_rank6 = rank_to_coord(topology3_2, 6);
  auto coord3_2_rank7 = rank_to_coord(topology3_2, 7);

  auto coord4_rank0   = rank_to_coord(topology4, 0);
  auto coord4_rank1   = rank_to_coord(topology4, 1);
  auto coord4_rank2   = rank_to_coord(topology4, 2);
  auto coord4_rank3   = rank_to_coord(topology4, 3);
  auto coord4_2_rank0 = rank_to_coord(topology4_2, 0);
  auto coord4_2_rank1 = rank_to_coord(topology4_2, 1);
  auto coord4_2_rank2 = rank_to_coord(topology4_2, 2);
  auto coord4_2_rank3 = rank_to_coord(topology4_2, 3);
  auto coord4_2_rank4 = rank_to_coord(topology4_2, 4);
  auto coord4_2_rank5 = rank_to_coord(topology4_2, 5);
  auto coord4_2_rank6 = rank_to_coord(topology4_2, 6);
  auto coord4_2_rank7 = rank_to_coord(topology4_2, 7);

  EXPECT_EQ(coord1_rank0, ref_coord1_rank0);
  EXPECT_EQ(coord1_rank1, ref_coord1_rank1);
  EXPECT_EQ(coord2_rank0, ref_coord2_rank0);
  EXPECT_EQ(coord2_rank1, ref_coord2_rank1);
  EXPECT_EQ(coord2_2_rank0, ref_coord2_2_rank0);
  EXPECT_EQ(coord2_2_rank1, ref_coord2_2_rank1);
  EXPECT_EQ(coord2_2_rank2, ref_coord2_2_rank2);
  EXPECT_EQ(coord2_2_rank3, ref_coord2_2_rank3);
  EXPECT_EQ(coord2_2_rank4, ref_coord2_2_rank4);
  EXPECT_EQ(coord2_2_rank5, ref_coord2_2_rank5);
  EXPECT_EQ(coord2_2_rank6, ref_coord2_2_rank6);
  EXPECT_EQ(coord2_2_rank7, ref_coord2_2_rank7);
  EXPECT_EQ(coord3_rank0, ref_coord3_rank0);
  EXPECT_EQ(coord3_rank1, ref_coord3_rank1);
  EXPECT_EQ(coord3_2_rank0, ref_coord3_2_rank0);
  EXPECT_EQ(coord3_2_rank1, ref_coord3_2_rank1);
  EXPECT_EQ(coord3_2_rank2, ref_coord3_2_rank2);
  EXPECT_EQ(coord3_2_rank3, ref_coord3_2_rank3);
  EXPECT_EQ(coord3_2_rank4, ref_coord3_2_rank4);
  EXPECT_EQ(coord3_2_rank5, ref_coord3_2_rank5);
  EXPECT_EQ(coord3_2_rank6, ref_coord3_2_rank6);
  EXPECT_EQ(coord3_2_rank7, ref_coord3_2_rank7);
  EXPECT_EQ(coord4_rank0, ref_coord4_rank0);
  EXPECT_EQ(coord4_rank1, ref_coord4_rank1);
  EXPECT_EQ(coord4_rank2, ref_coord4_rank2);
  EXPECT_EQ(coord4_rank3, ref_coord4_rank3);
  EXPECT_EQ(coord4_2_rank0, ref_coord4_2_rank0);
  EXPECT_EQ(coord4_2_rank1, ref_coord4_2_rank1);
  EXPECT_EQ(coord4_2_rank2, ref_coord4_2_rank2);
  EXPECT_EQ(coord4_2_rank3, ref_coord4_2_rank3);
  EXPECT_EQ(coord4_2_rank4, ref_coord4_2_rank4);
  EXPECT_EQ(coord4_2_rank5, ref_coord4_2_rank5);
  EXPECT_EQ(coord4_2_rank6, ref_coord4_2_rank6);
  EXPECT_EQ(coord4_2_rank7, ref_coord4_2_rank7);
}

template <typename T, typename LayoutType>
void test_get_local_shape2D(std::size_t rank, std::size_t nprocs) {
  using topology_type = std::array<std::size_t, 2>;
  using extents_type  = std::array<std::size_t, 2>;

  topology_type topology0{1, nprocs};
  topology_type topology1{nprocs, 1};

  auto distribute_extents = [&](std::size_t n, std::size_t t) {
    std::size_t quotient  = n / t;
    std::size_t remainder = n % t;
    return rank < remainder ? (quotient + 1) : quotient;
  };

  const std::size_t gn0 = 19, gn1 = 32;
  const std::size_t n0_t0 = gn0;
  const std::size_t n1_t0 = distribute_extents(gn1, nprocs);

  const std::size_t n0_t1 = distribute_extents(gn0, nprocs);
  const std::size_t n1_t1 = gn1;

  extents_type global_shape{gn0, gn1};
  extents_type ref_local_shape_t0{n0_t0, n1_t0},
      ref_local_shape_t1{n0_t1, n1_t1};

  auto local_shape_t0 =
      get_local_shape(global_shape, topology0, MPI_COMM_WORLD);
  auto local_shape_t1 =
      get_local_shape(global_shape, topology1, MPI_COMM_WORLD);

  EXPECT_EQ(local_shape_t0, ref_local_shape_t0);
  EXPECT_EQ(local_shape_t1, ref_local_shape_t1);
}

template <typename T, typename LayoutType>
void test_get_local_extents3D(std::size_t rank, std::size_t npx,
                              std::size_t npy) {
  using extents_type    = std::array<std::size_t, 3>;
  using topology_r_type = Topology<std::size_t, 3>;

  topology_r_type topology0{1, npx, npy}, topology1{npx, 1, npy},
      topology2{npx, npy, 1};

  std::size_t rx = rank / npy, ry = rank % npy;

  auto distribute_extents = [&](std::size_t n, std::size_t r, std::size_t t) {
    std::size_t quotient  = n / t;
    std::size_t remainder = n % t;
    return r < remainder ? (quotient + 1) : quotient;
  };

  const std::size_t gn0 = 8, gn1 = 7, gn2 = 5;
  const std::size_t n0_t0 = gn0;
  const std::size_t n1_t0 = distribute_extents(gn1, rx, npx);
  const std::size_t n2_t0 = distribute_extents(gn2, ry, npy);

  const std::size_t n0_t1 = distribute_extents(gn0, rx, npx);
  const std::size_t n1_t1 = gn1;
  const std::size_t n2_t1 = distribute_extents(gn2, ry, npy);

  const std::size_t n0_t2 = distribute_extents(gn0, rx, npx);
  const std::size_t n1_t2 = distribute_extents(gn1, ry, npy);
  const std::size_t n2_t2 = gn2;

  const std::size_t n1_t0_p = (gn1 - 1) / npx + 1;
  const std::size_t n2_t0_p = (gn2 - 1) / npy + 1;
  const std::size_t n0_t1_p = (gn0 - 1) / npx + 1;
  const std::size_t n2_t1_p = (gn2 - 1) / npy + 1;
  const std::size_t n0_t2_p = (gn0 - 1) / npx + 1;
  const std::size_t n1_t2_p = (gn1 - 1) / npy + 1;

  extents_type global_shape{gn0, gn1, gn2};
  extents_type ref_local_shape_t0{n0_t0, n1_t0, n2_t0},
      ref_local_shape_t1{n0_t1, n1_t1, n2_t1},
      ref_local_shape_t2{n0_t2, n1_t2, n2_t2};
  extents_type ref_local_starts_t0{0, rx * n1_t0_p, ry * n2_t0_p},
      ref_local_starts_t1{rx * n0_t1_p, 0, ry * n2_t1_p},
      ref_local_starts_t2{rx * n0_t2_p, ry * n1_t2_p, 0};

  auto [local_shape_t0, local_starts_t0] =
      get_local_extents(global_shape, topology0, MPI_COMM_WORLD);
  auto [local_shape_t1, local_starts_t1] =
      get_local_extents(global_shape, topology1, MPI_COMM_WORLD);
  auto [local_shape_t2, local_starts_t2] =
      get_local_extents(global_shape, topology2, MPI_COMM_WORLD);

  EXPECT_EQ(local_shape_t0, ref_local_shape_t0);
  EXPECT_EQ(local_shape_t1, ref_local_shape_t1);
  EXPECT_EQ(local_shape_t2, ref_local_shape_t2);

  EXPECT_EQ(local_starts_t0, ref_local_starts_t0);
  EXPECT_EQ(local_starts_t1, ref_local_starts_t1);
  EXPECT_EQ(local_starts_t2, ref_local_starts_t2);
}

template <typename T, typename LayoutType>
void test_get_next_extents2D(std::size_t rank, std::size_t nprocs) {
  using extents_type    = std::array<std::size_t, 2>;
  using topology_r_type = Topology<std::size_t, 2>;
  using map_type        = std::array<std::size_t, 2>;

  topology_r_type topology0{1, nprocs}, topology1{nprocs, 1};
  map_type map0{0, 1}, map1{1, 0};

  auto distribute_extents = [&](std::size_t n, std::size_t t) {
    std::size_t quotient  = n / t;
    std::size_t remainder = n % t;
    return rank < remainder ? (quotient + 1) : quotient;
  };

  const std::size_t gn0 = 19, gn1 = 32;
  const std::size_t n0_t0 = gn0;
  const std::size_t n1_t0 = distribute_extents(gn1, nprocs);

  const std::size_t n0_t1 = distribute_extents(gn0, nprocs);
  const std::size_t n1_t1 = gn1;

  extents_type global_shape{gn0, gn1};
  extents_type ref_next_shape_t0_map0{n0_t0, n1_t0},
      ref_next_shape_t0_map1{n1_t0, n0_t0},
      ref_next_shape_t1_map0{n0_t1, n1_t1},
      ref_next_shape_t1_map1{n1_t1, n0_t1};

  auto next_shape_t0_map0 =
      get_next_extents(global_shape, topology0, map0, MPI_COMM_WORLD);
  auto next_shape_t0_map1 =
      get_next_extents(global_shape, topology0, map1, MPI_COMM_WORLD);
  auto next_shape_t1_map0 =
      get_next_extents(global_shape, topology1, map0, MPI_COMM_WORLD);
  auto next_shape_t1_map1 =
      get_next_extents(global_shape, topology1, map1, MPI_COMM_WORLD);

  EXPECT_EQ(next_shape_t0_map0, ref_next_shape_t0_map0);
  EXPECT_EQ(next_shape_t0_map1, ref_next_shape_t0_map1);
  EXPECT_EQ(next_shape_t1_map0, ref_next_shape_t1_map0);
  EXPECT_EQ(next_shape_t1_map1, ref_next_shape_t1_map1);
}

template <typename T, typename LayoutType>
void test_get_next_extents3D(std::size_t rank, std::size_t npx,
                             std::size_t npy) {
  using extents_type    = std::array<std::size_t, 3>;
  using topology_r_type = Topology<std::size_t, 3, Kokkos::LayoutRight>;
  using topology_l_type = Topology<std::size_t, 3, Kokkos::LayoutLeft>;
  using map_type        = std::array<std::size_t, 3>;

  topology_r_type topology0{1, npx, npy}, topology1{npx, 1, npy},
      topology2{npx, npy, 1};
  topology_l_type topology3{npy, npx, 1};
  map_type map012{0, 1, 2}, map021{0, 2, 1}, map102{1, 0, 2}, map120{1, 2, 0},
      map201{2, 0, 1}, map210{2, 1, 0};

  std::size_t rx = rank / npy, ry = rank % npy;

  auto distribute_extents = [&](std::size_t n, std::size_t r, std::size_t t) {
    std::size_t quotient  = n / t;
    std::size_t remainder = n % t;
    return r < remainder ? (quotient + 1) : quotient;
  };

  const std::size_t gn0 = 8, gn1 = 7, gn2 = 5;
  const std::size_t n0_t0 = gn0;
  const std::size_t n1_t0 = distribute_extents(gn1, rx, npx);
  const std::size_t n2_t0 = distribute_extents(gn2, ry, npy);

  const std::size_t n0_t1 = distribute_extents(gn0, rx, npx);
  const std::size_t n1_t1 = gn1;
  const std::size_t n2_t1 = distribute_extents(gn2, ry, npy);

  const std::size_t n0_t2 = distribute_extents(gn0, rx, npx);
  const std::size_t n1_t2 = distribute_extents(gn1, ry, npy);
  const std::size_t n2_t2 = gn2;

  const std::size_t n0_t3 = distribute_extents(gn0, ry, npy);
  const std::size_t n1_t3 = distribute_extents(gn1, rx, npx);
  const std::size_t n2_t3 = gn2;

  extents_type global_shape{gn0, gn1, gn2};
  extents_type ref_next_shape_t0_map012{n0_t0, n1_t0, n2_t0},
      ref_next_shape_t0_map021{n0_t0, n2_t0, n1_t0},
      ref_next_shape_t0_map102{n1_t0, n0_t0, n2_t0},
      ref_next_shape_t0_map120{n1_t0, n2_t0, n0_t0},
      ref_next_shape_t0_map201{n2_t0, n0_t0, n1_t0},
      ref_next_shape_t0_map210{n2_t0, n1_t0, n0_t0},
      ref_next_shape_t1_map012{n0_t1, n1_t1, n2_t1},
      ref_next_shape_t1_map021{n0_t1, n2_t1, n1_t1},
      ref_next_shape_t1_map102{n1_t1, n0_t1, n2_t1},
      ref_next_shape_t1_map120{n1_t1, n2_t1, n0_t1},
      ref_next_shape_t1_map201{n2_t1, n0_t1, n1_t1},
      ref_next_shape_t1_map210{n2_t1, n1_t1, n0_t1},
      ref_next_shape_t2_map012{n0_t2, n1_t2, n2_t2},
      ref_next_shape_t2_map021{n0_t2, n2_t2, n1_t2},
      ref_next_shape_t2_map102{n1_t2, n0_t2, n2_t2},
      ref_next_shape_t2_map120{n1_t2, n2_t2, n0_t2},
      ref_next_shape_t2_map201{n2_t2, n0_t2, n1_t2},
      ref_next_shape_t2_map210{n2_t2, n1_t2, n0_t2},
      ref_next_shape_t3_map012{n0_t3, n1_t3, n2_t3},
      ref_next_shape_t3_map021{n0_t3, n2_t3, n1_t3},
      ref_next_shape_t3_map102{n1_t3, n0_t3, n2_t3},
      ref_next_shape_t3_map120{n1_t3, n2_t3, n0_t3},
      ref_next_shape_t3_map201{n2_t3, n0_t3, n1_t3},
      ref_next_shape_t3_map210{n2_t3, n1_t3, n0_t3};

  auto next_shape_t0_map012 =
      get_next_extents(global_shape, topology0, map012, MPI_COMM_WORLD);
  auto next_shape_t0_map021 =
      get_next_extents(global_shape, topology0, map021, MPI_COMM_WORLD);
  auto next_shape_t0_map102 =
      get_next_extents(global_shape, topology0, map102, MPI_COMM_WORLD);
  auto next_shape_t0_map120 =
      get_next_extents(global_shape, topology0, map120, MPI_COMM_WORLD);
  auto next_shape_t0_map201 =
      get_next_extents(global_shape, topology0, map201, MPI_COMM_WORLD);
  auto next_shape_t0_map210 =
      get_next_extents(global_shape, topology0, map210, MPI_COMM_WORLD);

  auto next_shape_t1_map012 =
      get_next_extents(global_shape, topology1, map012, MPI_COMM_WORLD);
  auto next_shape_t1_map021 =
      get_next_extents(global_shape, topology1, map021, MPI_COMM_WORLD);
  auto next_shape_t1_map102 =
      get_next_extents(global_shape, topology1, map102, MPI_COMM_WORLD);
  auto next_shape_t1_map120 =
      get_next_extents(global_shape, topology1, map120, MPI_COMM_WORLD);
  auto next_shape_t1_map201 =
      get_next_extents(global_shape, topology1, map201, MPI_COMM_WORLD);
  auto next_shape_t1_map210 =
      get_next_extents(global_shape, topology1, map210, MPI_COMM_WORLD);

  auto next_shape_t2_map012 =
      get_next_extents(global_shape, topology2, map012, MPI_COMM_WORLD);
  auto next_shape_t2_map021 =
      get_next_extents(global_shape, topology2, map021, MPI_COMM_WORLD);
  auto next_shape_t2_map102 =
      get_next_extents(global_shape, topology2, map102, MPI_COMM_WORLD);
  auto next_shape_t2_map120 =
      get_next_extents(global_shape, topology2, map120, MPI_COMM_WORLD);
  auto next_shape_t2_map201 =
      get_next_extents(global_shape, topology2, map201, MPI_COMM_WORLD);
  auto next_shape_t2_map210 =
      get_next_extents(global_shape, topology2, map210, MPI_COMM_WORLD);

  auto next_shape_t3_map012 =
      get_next_extents(global_shape, topology3, map012, MPI_COMM_WORLD);
  auto next_shape_t3_map021 =
      get_next_extents(global_shape, topology3, map021, MPI_COMM_WORLD);
  auto next_shape_t3_map102 =
      get_next_extents(global_shape, topology3, map102, MPI_COMM_WORLD);
  auto next_shape_t3_map120 =
      get_next_extents(global_shape, topology3, map120, MPI_COMM_WORLD);
  auto next_shape_t3_map201 =
      get_next_extents(global_shape, topology3, map201, MPI_COMM_WORLD);
  auto next_shape_t3_map210 =
      get_next_extents(global_shape, topology3, map210, MPI_COMM_WORLD);

  EXPECT_EQ(next_shape_t0_map012, ref_next_shape_t0_map012);
  EXPECT_EQ(next_shape_t0_map021, ref_next_shape_t0_map021);
  EXPECT_EQ(next_shape_t0_map102, ref_next_shape_t0_map102);
  EXPECT_EQ(next_shape_t0_map120, ref_next_shape_t0_map120);
  EXPECT_EQ(next_shape_t0_map201, ref_next_shape_t0_map201);
  EXPECT_EQ(next_shape_t0_map210, ref_next_shape_t0_map210);

  EXPECT_EQ(next_shape_t1_map012, ref_next_shape_t1_map012);
  EXPECT_EQ(next_shape_t1_map021, ref_next_shape_t1_map021);
  EXPECT_EQ(next_shape_t1_map102, ref_next_shape_t1_map102);
  EXPECT_EQ(next_shape_t1_map120, ref_next_shape_t1_map120);
  EXPECT_EQ(next_shape_t1_map201, ref_next_shape_t1_map201);
  EXPECT_EQ(next_shape_t1_map210, ref_next_shape_t1_map210);

  EXPECT_EQ(next_shape_t2_map012, ref_next_shape_t2_map012);
  EXPECT_EQ(next_shape_t2_map021, ref_next_shape_t2_map021);
  EXPECT_EQ(next_shape_t2_map102, ref_next_shape_t2_map102);
  EXPECT_EQ(next_shape_t2_map120, ref_next_shape_t2_map120);
  EXPECT_EQ(next_shape_t2_map201, ref_next_shape_t2_map201);
  EXPECT_EQ(next_shape_t2_map210, ref_next_shape_t2_map210);

  EXPECT_EQ(next_shape_t3_map012, ref_next_shape_t3_map012);
  EXPECT_EQ(next_shape_t3_map021, ref_next_shape_t3_map021);
  EXPECT_EQ(next_shape_t3_map102, ref_next_shape_t3_map102);
  EXPECT_EQ(next_shape_t3_map120, ref_next_shape_t3_map120);
  EXPECT_EQ(next_shape_t3_map201, ref_next_shape_t3_map201);
  EXPECT_EQ(next_shape_t3_map210, ref_next_shape_t3_map210);
}

}  // namespace

TYPED_TEST_SUITE(TestMPIHelper, test_types);

TYPED_TEST(TestMPIHelper, GetGlobalShape2D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_get_global_shape2D<float_type, layout_type>(this->m_rank,
                                                   this->m_nprocs);
}

TYPED_TEST(TestMPIHelper, GetGlobalShape3D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  if (this->m_nprocs == 1 || this->m_npx * this->m_npx != this->m_nprocs) {
    GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                    "for this test";
  }

  test_get_global_shape3D<float_type, layout_type>(this->m_rank, this->m_npx,
                                                   this->m_npx);
}

TEST(TestRankToCoord, 1Dto4D) { test_rank_to_coord(); }

TYPED_TEST(TestMPIHelper, GetLocalShape2D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_get_local_shape2D<float_type, layout_type>(this->m_rank, this->m_nprocs);
}

TYPED_TEST(TestMPIHelper, GetLocalShape3D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  if (this->m_nprocs == 1 || this->m_npx * this->m_npx != this->m_nprocs) {
    GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                    "for this test";
  }

  test_get_local_extents3D<float_type, layout_type>(this->m_rank, this->m_npx,
                                                    this->m_npx);
}

TYPED_TEST(TestMPIHelper, GetNextExtents2D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_get_next_extents2D<float_type, layout_type>(this->m_rank,
                                                   this->m_nprocs);
}

TYPED_TEST(TestMPIHelper, GetNextExtents3D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  if (this->m_nprocs == 1 || this->m_npx * this->m_npx != this->m_nprocs) {
    GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                    "for this test";
  }

  test_get_next_extents3D<float_type, layout_type>(this->m_rank, this->m_npx,
                                                   this->m_npx);
}
