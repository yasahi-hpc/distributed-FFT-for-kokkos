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

  virtual void SetUp() {
    int rank, nprocs;
    ::MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    m_rank   = rank;
    m_nprocs = nprocs;
  }
};

template <typename T, typename LayoutType>
void test_get_global_shape2D(std::size_t rank, std::size_t nprocs) {
  using topology_type = std::array<std::size_t, 2>;
  using ViewType      = Kokkos::View<T**, LayoutType, execution_space>;

  topology_type topology0{1, nprocs};
  topology_type topology1{nprocs, 1};

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
void test_get_global_shape3D(std::size_t rank, std::size_t nprocs) {
  using topology_type = std::array<std::size_t, 3>;
  using ViewType      = Kokkos::View<T***, LayoutType, execution_space>;

  std::size_t nprocs_1D = std::sqrt(nprocs);
  if (nprocs_1D * nprocs_1D != nprocs) {
    GTEST_SKIP() << "The number of MPI ranks should be a perfect square ";
  }

  topology_type topology0{1, nprocs_1D, nprocs_1D};
  topology_type topology1{nprocs_1D, 1, nprocs_1D};
  topology_type topology2{nprocs_1D, nprocs_1D, 1};

  std::size_t rank0 = rank / nprocs_1D;
  std::size_t rank1 = rank % nprocs_1D;

  const std::size_t gn0 = 19, gn1 = 32, gn2 = 25;
  const std::size_t n0_t0           = gn0;
  const std::size_t n1_t0_quotient  = (gn1 - 1) / nprocs_1D + 1;
  const std::size_t n1_t0_remainder = gn1 - n1_t0_quotient * (nprocs_1D - 1);
  const std::size_t n1_t0 =
      rank0 != (nprocs_1D - 1) ? n1_t0_quotient : n1_t0_remainder;
  const std::size_t n2_t0_quotient  = (gn2 - 1) / nprocs_1D + 1;
  const std::size_t n2_t0_remainder = gn2 - n2_t0_quotient * (nprocs_1D - 1);
  const std::size_t n2_t0 =
      rank1 != (nprocs_1D - 1) ? n2_t0_quotient : n2_t0_remainder;

  const std::size_t n0_t1_quotient  = (gn0 - 1) / nprocs_1D + 1;
  const std::size_t n0_t1_remainder = gn0 - n0_t1_quotient * (nprocs_1D - 1);
  const std::size_t n0_t1 =
      rank0 != (nprocs_1D - 1) ? n0_t1_quotient : n0_t1_remainder;
  const std::size_t n1_t1           = gn1;
  const std::size_t n2_t1_quotient  = (gn2 - 1) / nprocs_1D + 1;
  const std::size_t n2_t1_remainder = gn2 - n2_t1_quotient * (nprocs_1D - 1);
  const std::size_t n2_t1 =
      rank1 != (nprocs_1D - 1) ? n2_t1_quotient : n2_t1_remainder;

  const std::size_t n0_t2_quotient  = (gn0 - 1) / nprocs_1D + 1;
  const std::size_t n0_t2_remainder = gn0 - n0_t2_quotient * (nprocs_1D - 1);
  const std::size_t n0_t2 =
      rank0 != (nprocs_1D - 1) ? n0_t2_quotient : n0_t2_remainder;
  const std::size_t n1_t2_quotient  = (gn1 - 1) / nprocs_1D + 1;
  const std::size_t n1_t2_remainder = gn1 - n1_t2_quotient * (nprocs_1D - 1);
  const std::size_t n1_t2 =
      rank1 != (nprocs_1D - 1) ? n1_t2_quotient : n1_t2_remainder;
  const std::size_t n2_t2 = gn2;

  ViewType v0("v0", n0_t0, n1_t0, n2_t0);
  ViewType v1("v1", n0_t1, n1_t1, n2_t1);
  ViewType v2("v2", n0_t2, n1_t2, n2_t2);

  auto global_shape_t0 = get_global_shape(v0, topology0, MPI_COMM_WORLD);
  auto global_shape_t1 = get_global_shape(v1, topology1, MPI_COMM_WORLD);
  auto global_shape_t2 = get_global_shape(v2, topology2, MPI_COMM_WORLD);

  topology_type ref_global_shape{gn0, gn1, gn2};

  EXPECT_EQ(global_shape_t0, ref_global_shape);
  EXPECT_EQ(global_shape_t1, ref_global_shape);
  EXPECT_EQ(global_shape_t2, ref_global_shape);
}

void test_rank_to_coord() {
  using topology1D_type = std::array<std::size_t, 1>;
  using topology2D_type = std::array<std::size_t, 2>;
  using topology3D_type = std::array<std::size_t, 3>;
  using topology4D_type = std::array<std::size_t, 4>;

  topology1D_type topology1{2};
  topology2D_type topology2{1, 2}, topology2_2{4, 2};
  topology3D_type topology3{1, 2, 1}, topology3_2{4, 2, 1};
  topology4D_type topology4{1, 4, 1, 1}, topology4_2{1, 1, 4, 2};

  topology1D_type ref_coord1_rank0{0}, ref_coord1_rank1{1};
  topology2D_type ref_coord2_rank0{0, 0}, ref_coord2_rank1{0, 1};
  topology2D_type ref_coord2_2_rank0{0, 0}, ref_coord2_2_rank1{0, 1},
      ref_coord2_2_rank2{1, 0}, ref_coord2_2_rank3{1, 1},
      ref_coord2_2_rank4{2, 0}, ref_coord2_2_rank5{2, 1},
      ref_coord2_2_rank6{3, 0}, ref_coord2_2_rank7{3, 1};
  topology3D_type ref_coord3_rank0{0, 0, 0}, ref_coord3_rank1{0, 1, 0};
  topology3D_type ref_coord3_2_rank0{0, 0, 0}, ref_coord3_2_rank1{0, 1, 0},
      ref_coord3_2_rank2{1, 0, 0}, ref_coord3_2_rank3{1, 1, 0},
      ref_coord3_2_rank4{2, 0, 0}, ref_coord3_2_rank5{2, 1, 0},
      ref_coord3_2_rank6{3, 0, 0}, ref_coord3_2_rank7{3, 1, 0};
  topology4D_type ref_coord4_rank0{0, 0, 0, 0}, ref_coord4_rank1{0, 1, 0, 0},
      ref_coord4_rank2{0, 2, 0, 0}, ref_coord4_rank3{0, 3, 0, 0};
  topology4D_type ref_coord4_2_rank0{0, 0, 0, 0},
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
  using ViewType      = Kokkos::View<T**, LayoutType, execution_space>;

  topology_type topology0{1, nprocs};
  topology_type topology1{nprocs, 1};

  auto distribute_extents = [&](std::size_t n, std::size_t t) {
    std::size_t quotient  = n / t;
    std::size_t remainder = n % t;
    return rank < remainder ? (quotient + 1) : quotient;
  };

  const std::size_t gn0 = 19, gn1 = 32;
  const std::size_t n0_t0           = gn0;
  const std::size_t n1_t0_quotient  = (gn1 - 1) / nprocs + 1;
  const std::size_t n1_t0_remainder = gn1 - n1_t0_quotient * (nprocs - 1);
  const std::size_t n1_t0           = distribute_extents(gn1, nprocs);
  // const std::size_t n1_t0 =
  //     rank != (nprocs - 1) ? n1_t0_quotient : n1_t0_remainder;

  const std::size_t n0_t1_quotient  = (gn0 - 1) / nprocs + 1;
  const std::size_t n0_t1_remainder = gn0 - n0_t1_quotient * (nprocs - 1);
  // const std::size_t n0_t1 =
  //     rank != (nprocs - 1) ? n0_t1_quotient : n0_t1_remainder;
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

  test_get_global_shape3D<float_type, layout_type>(this->m_rank,
                                                   this->m_nprocs);
}

TEST(TestRankToCoord, 1Dto4D) { test_rank_to_coord(); }

TYPED_TEST(TestMPIHelper, GetLocalShape2D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_get_local_shape2D<float_type, layout_type>(this->m_rank, this->m_nprocs);
}
