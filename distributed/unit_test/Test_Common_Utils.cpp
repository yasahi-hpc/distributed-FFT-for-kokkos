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

}  // namespace

TEST_P(CommonUtilsParamTests, GetTransAxis) {
  int n0 = GetParam();
  test_get_trans_axis(n0);
}

INSTANTIATE_TEST_SUITE_P(CommonUtilsTests, CommonUtilsParamTests,
                         ::testing::Values(1, 2, 3, 4, 5, 6));

TYPED_TEST_SUITE(TestContainerTypes, base_int_types);

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
