#include <mpi.h>
#include <gtest/gtest.h>
#include <vector>
#include <array>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_Utils.hpp"
#include "Test_Utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
class CommonUtilsParamTests : public ::testing::TestWithParam<int> {};

using base_int_types = ::testing::Types<int, std::size_t>;

template <typename T>
struct TestContainerTypes : public ::testing::Test {
  static constexpr std::size_t rank = 5;
  using value_type                  = T;
  using vector_type                 = std::vector<T>;
  using array_type                  = std::array<T, rank>;
};

void test_get_trans_axis(std::size_t nprocs) {
  using topology_type  = std::array<std::size_t, 3>;
  using topology_type2 = std::array<std::size_t, 4>;

  std::size_t np0 = 4;

  // 3D topologies
  topology_type topology0 = {1, nprocs, np0}, topology1 = {nprocs, 1, np0},
                topology2 = {np0, nprocs, 1};

  // 4D topologies
  topology_type2 topology3 = {1, 1, np0, nprocs},
                 topology4 = {1, np0, 1, nprocs};

  if (nprocs == 1) {
    // Failure tests because these are not pencils
    EXPECT_THROW(
        {
          [[maybe_unused]] auto trans_axis =
              KokkosFFT::Distributed::Impl::get_trans_axis(topology0, topology1,
                                                           nprocs);
        },
        std::runtime_error);

  } else if (nprocs == np0) {
    // Failure tests because they include identical non-one elements
    EXPECT_THROW(
        {
          [[maybe_unused]] auto trans_axis =
              KokkosFFT::Distributed::Impl::get_trans_axis(topology0, topology1,
                                                           nprocs);
        },
        std::runtime_error);
  } else {
    auto axis_0_1 = KokkosFFT::Distributed::Impl::get_trans_axis(
        topology0, topology1, nprocs);
    auto axis_0_2 = KokkosFFT::Distributed::Impl::get_trans_axis(
        topology0, topology2, nprocs);
    auto axis_1_0 = KokkosFFT::Distributed::Impl::get_trans_axis(
        topology1, topology0, nprocs);
    auto axis_2_0 = KokkosFFT::Distributed::Impl::get_trans_axis(
        topology2, topology0, nprocs);

    std::size_t ref_axis_0_1 = 0, ref_axis_0_2 = 1, ref_axis_1_0 = 0,
                ref_axis_2_0 = 1;
    EXPECT_EQ(axis_0_1, ref_axis_0_1);
    EXPECT_EQ(axis_0_2, ref_axis_0_2);
    EXPECT_EQ(axis_1_0, ref_axis_1_0);
    EXPECT_EQ(axis_2_0, ref_axis_2_0);

    auto axis_3_4 =
        KokkosFFT::Distributed::Impl::get_trans_axis(topology3, topology4, np0);
    auto axis_4_3 =
        KokkosFFT::Distributed::Impl::get_trans_axis(topology4, topology3, np0);

    std::size_t ref_axis_3_4 = 0, ref_axis_4_3 = 0;
    EXPECT_EQ(axis_3_4, ref_axis_3_4);
    EXPECT_EQ(axis_4_3, ref_axis_4_3);

    // Failure tests because they differ at three positions
    EXPECT_THROW(
        {
          [[maybe_unused]] auto axis_1_2 =
              KokkosFFT::Distributed::Impl::get_trans_axis(topology1, topology2,
                                                           nprocs);
        },
        std::runtime_error);
  }
}

template <typename ContainerType, typename iType>
void test_extract_different_indices(iType nprocs) {
  ContainerType topology0 = {nprocs, 1, 8};
  ContainerType topology1 = {nprocs, 8, 1};
  ContainerType topology2 = {8, nprocs, 1};

  if (nprocs == 1) {
    auto diff01 = KokkosFFT::Distributed::Impl::extract_different_indices(
        topology0, topology1);
    auto diff02 = KokkosFFT::Distributed::Impl::extract_different_indices(
        topology0, topology2);
    auto diff10 = KokkosFFT::Distributed::Impl::extract_different_indices(
        topology1, topology0);
    auto diff12 = KokkosFFT::Distributed::Impl::extract_different_indices(
        topology1, topology2);
    auto diff20 = KokkosFFT::Distributed::Impl::extract_different_indices(
        topology2, topology0);
    auto diff21 = KokkosFFT::Distributed::Impl::extract_different_indices(
        topology2, topology1);

    std::vector<std::size_t> ref_diff01 = {1, 2};
    std::vector<std::size_t> ref_diff02 = {0, 2};
    std::vector<std::size_t> ref_diff10 = {1, 2};
    std::vector<std::size_t> ref_diff12 = {0, 1};
    std::vector<std::size_t> ref_diff20 = {0, 2};
    std::vector<std::size_t> ref_diff21 = {0, 1};

    EXPECT_EQ(diff01, ref_diff01);
    EXPECT_EQ(diff02, ref_diff02);
    EXPECT_EQ(diff10, ref_diff10);
    EXPECT_EQ(diff12, ref_diff12);
    EXPECT_EQ(diff20, ref_diff20);
    EXPECT_EQ(diff21, ref_diff21);
  } else {
    auto diff01 = KokkosFFT::Distributed::Impl::extract_different_indices(
        topology0, topology1);
    auto diff02 = KokkosFFT::Distributed::Impl::extract_different_indices(
        topology0, topology2);
    auto diff10 = KokkosFFT::Distributed::Impl::extract_different_indices(
        topology1, topology0);
    auto diff12 = KokkosFFT::Distributed::Impl::extract_different_indices(
        topology1, topology2);
    auto diff20 = KokkosFFT::Distributed::Impl::extract_different_indices(
        topology2, topology0);
    auto diff21 = KokkosFFT::Distributed::Impl::extract_different_indices(
        topology2, topology1);

    std::vector<std::size_t> ref_diff01 = {1, 2};
    std::vector<std::size_t> ref_diff02 = {0, 1, 2};
    std::vector<std::size_t> ref_diff10 = {1, 2};
    std::vector<std::size_t> ref_diff12 = {0, 1};
    std::vector<std::size_t> ref_diff20 = {0, 1, 2};
    std::vector<std::size_t> ref_diff21 = {0, 1};

    EXPECT_EQ(diff01, ref_diff01);
    EXPECT_EQ(diff02, ref_diff02);
    EXPECT_EQ(diff10, ref_diff10);
    EXPECT_EQ(diff12, ref_diff12);
    EXPECT_EQ(diff20, ref_diff20);
    EXPECT_EQ(diff21, ref_diff21);
  }
}

template <typename ContainerType, typename iType>
void test_extract_different_value_set(iType nprocs) {
  ContainerType topology0 = {nprocs, 1, 8};
  ContainerType topology1 = {nprocs, 8, 1};
  ContainerType topology2 = {8, nprocs, 1};

  auto diff01 = KokkosFFT::Distributed::Impl::extract_different_value_set(
      topology0, topology1);
  auto diff02 = KokkosFFT::Distributed::Impl::extract_different_value_set(
      topology0, topology2);
  auto diff10 = KokkosFFT::Distributed::Impl::extract_different_value_set(
      topology1, topology0);
  auto diff12 = KokkosFFT::Distributed::Impl::extract_different_value_set(
      topology1, topology2);
  auto diff20 = KokkosFFT::Distributed::Impl::extract_different_value_set(
      topology2, topology0);
  auto diff21 = KokkosFFT::Distributed::Impl::extract_different_value_set(
      topology2, topology1);

  std::set<iType> ref_diff =
      (nprocs == 1) ? std::set<iType>{1, 8} : std::set<iType>{1, nprocs, 8};
  EXPECT_EQ(diff01, ref_diff);
  EXPECT_EQ(diff02, ref_diff);
  EXPECT_EQ(diff10, ref_diff);
  EXPECT_EQ(diff12, ref_diff);
  EXPECT_EQ(diff20, ref_diff);
  EXPECT_EQ(diff21, ref_diff);
}

template <typename ContainerType0, typename ContainerType1,
          typename ContainerType2>
void test_count_non_ones() {
  ContainerType0 v0 = {0, 1, 4, 2, 3};
  ContainerType1 v1 = {2, 3, 5};
  ContainerType2 v2 = {1};

  EXPECT_EQ(KokkosFFT::Distributed::Impl::count_non_ones(v0), 4);
  EXPECT_EQ(KokkosFFT::Distributed::Impl::count_non_ones(v1), 3);
  EXPECT_EQ(KokkosFFT::Distributed::Impl::count_non_ones(v2), 0);
}

template <typename ContainerType0, typename ContainerType1,
          typename ContainerType2>
void test_extract_non_one_indices() {
  ContainerType0 a0 = {1, 1, 4, 2, 3}, b0 = {1, 4, 2, 3, 1};
  ContainerType1 a1 = {2, 3, 5}, b1 = {5, 3, 2};
  ContainerType2 a2 = {1}, b2 = {1};

  std::vector<std::size_t> ref_diff_00 = {1, 2, 3, 4};
  std::vector<std::size_t> ref_diff_11 = {0, 1, 2};
  std::vector<std::size_t> ref_diff_22 = {};

  EXPECT_EQ(KokkosFFT::Distributed::Impl::extract_non_one_indices(a0, b0),
            ref_diff_00);
  EXPECT_EQ(KokkosFFT::Distributed::Impl::extract_non_one_indices(a1, b1),
            ref_diff_11);
  EXPECT_EQ(KokkosFFT::Distributed::Impl::extract_non_one_indices(a2, b2),
            ref_diff_22);
}

template <typename ContainerType0, typename ContainerType1,
          typename ContainerType2, typename iType>
void test_extract_non_one_values() {
  ContainerType0 a0 = {1, 1, 4, 2, 3};
  ContainerType1 a1 = {2, 3, 5};
  ContainerType2 a2 = {1};

  std::vector<iType> ref_diff_0 = {4, 2, 3};
  std::vector<iType> ref_diff_1 = {2, 3, 5};
  std::vector<iType> ref_diff_2 = {};

  EXPECT_EQ(KokkosFFT::Distributed::Impl::extract_non_one_values(a0),
            ref_diff_0);
  EXPECT_EQ(KokkosFFT::Distributed::Impl::extract_non_one_values(a1),
            ref_diff_1);
  EXPECT_EQ(KokkosFFT::Distributed::Impl::extract_non_one_values(a2),
            ref_diff_2);
}

}  // namespace

TEST_P(CommonUtilsParamTests, GetTransAxis) {
  int n0 = GetParam();
  test_get_trans_axis(n0);
}

INSTANTIATE_TEST_SUITE_P(CommonUtilsTests, CommonUtilsParamTests,
                         ::testing::Values(1, 2, 3, 4, 5, 6));

TYPED_TEST_SUITE(TestContainerTypes, base_int_types);

TYPED_TEST(TestContainerTypes, find_different_indices_of_vector) {
  using value_type     = typename TestFixture::value_type;
  using container_type = typename TestFixture::vector_type;

  for (value_type nprocs = 1; nprocs <= 6; ++nprocs) {
    test_extract_different_indices<container_type, value_type>(nprocs);
  }
}

TYPED_TEST(TestContainerTypes, find_different_indices_of_array) {
  using value_type = typename TestFixture::value_type;
  using array_type = std::array<value_type, 3>;
  for (value_type nprocs = 1; nprocs <= 6; ++nprocs) {
    test_extract_different_indices<array_type, value_type>(nprocs);
  }
}

TYPED_TEST(TestContainerTypes, find_different_value_set_of_vector) {
  using value_type  = typename TestFixture::value_type;
  using vector_type = typename TestFixture::vector_type;

  for (value_type nprocs = 1; nprocs <= 6; ++nprocs) {
    test_extract_different_value_set<vector_type, value_type>(nprocs);
  }
}

TYPED_TEST(TestContainerTypes, find_different_value_set_of_array) {
  using value_type = typename TestFixture::value_type;
  using array_type = std::array<value_type, 3>;
  for (value_type nprocs = 1; nprocs <= 6; ++nprocs) {
    test_extract_different_value_set<array_type, value_type>(nprocs);
  }
}

TYPED_TEST(TestContainerTypes, test_count_non_ones_of_vector) {
  using container_type = typename TestFixture::vector_type;
  test_count_non_ones<container_type, container_type, container_type>();
}

TYPED_TEST(TestContainerTypes, test_count_non_ones_of_array) {
  using value_type      = typename TestFixture::value_type;
  using container_type0 = std::array<value_type, 5>;
  using container_type1 = std::array<value_type, 3>;
  using container_type2 = std::array<value_type, 1>;
  test_count_non_ones<container_type0, container_type1, container_type2>();
}

TYPED_TEST(TestContainerTypes, test_extract_non_one_indices_of_vector) {
  using container_type = typename TestFixture::vector_type;
  test_extract_non_one_indices<container_type, container_type,
                               container_type>();
}

TYPED_TEST(TestContainerTypes, test_extract_non_one_indices_of_array) {
  using value_type      = typename TestFixture::value_type;
  using container_type0 = std::array<value_type, 5>;
  using container_type1 = std::array<value_type, 3>;
  using container_type2 = std::array<value_type, 1>;
  test_extract_non_one_indices<container_type0, container_type1,
                               container_type2>();
}

TYPED_TEST(TestContainerTypes, test_extract_non_one_values_of_vector) {
  using container_type = typename TestFixture::vector_type;
  using value_type     = typename TestFixture::value_type;
  test_extract_non_one_values<container_type, container_type, container_type,
                              value_type>();
}

TYPED_TEST(TestContainerTypes, test_extract_non_one_values_of_array) {
  using value_type      = typename TestFixture::value_type;
  using container_type0 = std::array<value_type, 5>;
  using container_type1 = std::array<value_type, 3>;
  using container_type2 = std::array<value_type, 1>;
  test_extract_non_one_values<container_type0, container_type1, container_type2,
                              value_type>();
}
