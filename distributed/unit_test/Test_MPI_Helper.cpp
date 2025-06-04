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

  int m_rank   = 0;
  int m_nprocs = 1;

  virtual void SetUp() {
    ::MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &m_nprocs);
  }
};

template <typename T, typename LayoutType>
void test_get_global_shape2D(int rank, int nprocs) {
  using topology_type = std::array<std::size_t, 2>;
  using ViewType      = Kokkos::View<T**, LayoutType, execution_space>;

  topology_type topology0{1, nprocs};
  topology_type topology1{nprocs, 1};

  const int gn0 = 19, gn1 = 32;
  const int n0_t0           = gn0;
  const int n1_t0_quotient  = (gn1 - 1) / nprocs + 1;
  const int n1_t0_remainder = gn1 - n1_t0_quotient * (nprocs - 1);
  const int n1_t0 = rank != (nprocs - 1) ? n1_t0_quotient : n1_t0_remainder;

  const int n0_t1_quotient  = (gn0 - 1) / nprocs + 1;
  const int n0_t1_remainder = gn0 - n0_t1_quotient * (nprocs - 1);
  const int n0_t1 = rank != (nprocs - 1) ? n0_t1_quotient : n0_t1_remainder;
  const int n1_t1 = gn1;

  ViewType v0("v0", n0_t0, n1_t0);
  ViewType v1("v1", n0_t1, n1_t1);

  auto global_shape_t0 = get_global_shape(v0, topology0, MPI_COMM_WORLD);
  auto global_shape_t1 = get_global_shape(v1, topology1, MPI_COMM_WORLD);

  topology_type ref_global_shape{gn0, gn1};

  EXPECT_EQ(global_shape_t0, ref_global_shape);
  EXPECT_EQ(global_shape_t1, ref_global_shape);
}

template <typename T, typename LayoutType>
void test_get_global_shape3D(int rank, int nprocs) {
  using topology_type = std::array<std::size_t, 3>;
  using ViewType      = Kokkos::View<T***, LayoutType, execution_space>;

  int nprocs_1D = std::sqrt(nprocs);
  if (nprocs_1D * nprocs_1D != nprocs) {
    GTEST_SKIP() << "The number of MPI ranks should be a perfect square ";
  }

  topology_type topology0{1, nprocs_1D, nprocs_1D};
  topology_type topology1{nprocs_1D, 1, nprocs_1D};
  topology_type topology2{nprocs_1D, nprocs_1D, 1};

  int rank0 = rank / nprocs_1D;
  int rank1 = rank % nprocs_1D;

  const int gn0 = 19, gn1 = 32, gn2 = 25;
  const int n0_t0           = gn0;
  const int n1_t0_quotient  = (gn1 - 1) / nprocs_1D + 1;
  const int n1_t0_remainder = gn1 - n1_t0_quotient * (nprocs_1D - 1);
  const int n1_t0 = rank0 != (nprocs_1D - 1) ? n1_t0_quotient : n1_t0_remainder;
  const int n2_t0_quotient  = (gn2 - 1) / nprocs_1D + 1;
  const int n2_t0_remainder = gn2 - n2_t0_quotient * (nprocs_1D - 1);
  const int n2_t0 = rank1 != (nprocs_1D - 1) ? n2_t0_quotient : n2_t0_remainder;

  const int n0_t1_quotient  = (gn0 - 1) / nprocs_1D + 1;
  const int n0_t1_remainder = gn0 - n0_t1_quotient * (nprocs_1D - 1);
  const int n0_t1 = rank0 != (nprocs_1D - 1) ? n0_t1_quotient : n0_t1_remainder;
  const int n1_t1 = gn1;
  const int n2_t1_quotient  = (gn2 - 1) / nprocs_1D + 1;
  const int n2_t1_remainder = gn2 - n2_t1_quotient * (nprocs_1D - 1);
  const int n2_t1 = rank1 != (nprocs_1D - 1) ? n2_t1_quotient : n2_t1_remainder;

  const int n0_t2_quotient  = (gn0 - 1) / nprocs_1D + 1;
  const int n0_t2_remainder = gn0 - n0_t2_quotient * (nprocs_1D - 1);
  const int n0_t2 = rank0 != (nprocs_1D - 1) ? n0_t2_quotient : n0_t2_remainder;
  const int n1_t2_quotient  = (gn1 - 1) / nprocs_1D + 1;
  const int n1_t2_remainder = gn1 - n1_t2_quotient * (nprocs_1D - 1);
  const int n1_t2 = rank1 != (nprocs_1D - 1) ? n1_t2_quotient : n1_t2_remainder;
  const int n2_t2 = gn2;

  ViewType v0("v0", n0_t0, n1_t0, n2_t0);
  ViewType v1("v1", n0_t1, n1_t1, n2_t1);
  ViewType v2("v2", n0_t2, n1_t2, n2_t2);

  std::stringstream ss;
  ss << "n0_t1, n1_t1, n2_t1, rank: " << n0_t1 << ", " << n1_t1 << ", " << n2_t1
     << ", " << rank << "\n";
  ss << "n0_t2, n1_t2, n2_t2, rank: " << n0_t2 << ", " << n1_t2 << ", " << n2_t2
     << ", " << rank << "\n";

  std::cout << ss.str() << std::endl;

  auto global_shape_t0 = get_global_shape(v0, topology0, MPI_COMM_WORLD);
  auto global_shape_t1 = get_global_shape(v1, topology1, MPI_COMM_WORLD);
  auto global_shape_t2 = get_global_shape(v2, topology2, MPI_COMM_WORLD);

  topology_type ref_global_shape{gn0, gn1, gn2};

  EXPECT_EQ(global_shape_t0, ref_global_shape);
  EXPECT_EQ(global_shape_t1, ref_global_shape);
  EXPECT_EQ(global_shape_t2, ref_global_shape);
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
