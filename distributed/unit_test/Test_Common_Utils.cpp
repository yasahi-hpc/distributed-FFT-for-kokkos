#include <mpi.h>
#include <gtest/gtest.h>
#include <iostream>
#include <Kokkos_Core.hpp>
#include "Utils.hpp"
#include "Test_Utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
class CommonUtilsParamTests : public ::testing::TestWithParam<int> {};

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
              get_trans_axis(topology0, topology1, nprocs);
        },
        std::runtime_error);

  } else if (nprocs == np0) {
    // Failure tests because they include idential non-one elements
    EXPECT_THROW(
        {
          [[maybe_unused]] auto trans_axis =
              get_trans_axis(topology0, topology1, nprocs);
        },
        std::runtime_error);
  } else {
    auto axis_0_1 = get_trans_axis(topology0, topology1, nprocs);
    auto axis_0_2 = get_trans_axis(topology0, topology2, nprocs);
    auto axis_1_0 = get_trans_axis(topology1, topology0, nprocs);
    auto axis_2_0 = get_trans_axis(topology2, topology0, nprocs);

    std::size_t ref_axis_0_1 = 0, ref_axis_0_2 = 1, ref_axis_1_0 = 0,
                ref_axis_2_0 = 1;
    EXPECT_EQ(axis_0_1, ref_axis_0_1);
    EXPECT_EQ(axis_0_2, ref_axis_0_2);
    EXPECT_EQ(axis_1_0, ref_axis_1_0);
    EXPECT_EQ(axis_2_0, ref_axis_2_0);

    auto axis_3_4 = get_trans_axis(topology3, topology4, np0);
    auto axis_4_3 = get_trans_axis(topology4, topology3, np0);

    std::size_t ref_axis_3_4 = 0, ref_axis_4_3 = 0;
    EXPECT_EQ(axis_3_4, ref_axis_3_4);
    EXPECT_EQ(axis_4_3, ref_axis_4_3);

    // Failure tests because they differ at three positions
    EXPECT_THROW(
        {
          [[maybe_unused]] auto axis_1_2 =
              get_trans_axis(topology1, topology2, nprocs);
        },
        std::runtime_error);
  }
}

}  // namespace

TEST_P(CommonUtilsParamTests, GetTransAxis) {
  int n0 = GetParam();
  test_get_trans_axis(n0);
}

INSTANTIATE_TEST_SUITE_P(CommonUtilsTests, CommonUtilsParamTests,
                         ::testing::Values(1, 2, 3, 4, 5, 6));
