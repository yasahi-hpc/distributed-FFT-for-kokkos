#include <mpi.h>
#include <gtest/gtest.h>
#include <iostream>
#include <Kokkos_Core.hpp>
#include "Topologies.hpp"
#include "Test_Utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
class PencilParamTests : public ::testing::TestWithParam<int> {};

void test_get_pencil_3D(std::size_t nprocs) {
  using topology_type     = std::array<std::size_t, 3>;
  topology_type topology0 = {1, 1, nprocs};
  topology_type topology1 = {1, nprocs, 1};
  topology_type topology2 = {nprocs, 1, 1};
  topology_type topology3 = {nprocs, 1, 2};
  topology_type topology4 = {nprocs, 2, 1};

  if (nprocs == 1) {
    // Failure tests because of size 1 case
    EXPECT_THROW(
        {
          [[maybe_unused]] auto inout_axis01 = get_pencil(topology0, topology1);
        },
        std::runtime_error);
    EXPECT_THROW(
        {
          [[maybe_unused]] auto inout_axis02 = get_pencil(topology0, topology2);
        },
        std::runtime_error);
    EXPECT_THROW(
        {
          [[maybe_unused]] auto inout_axis10 = get_pencil(topology1, topology0);
        },
        std::runtime_error);
    EXPECT_THROW(
        {
          [[maybe_unused]] auto inout_axis12 = get_pencil(topology1, topology2);
        },
        std::runtime_error);
    EXPECT_THROW(
        {
          [[maybe_unused]] auto inout_axis20 = get_pencil(topology2, topology0);
        },
        std::runtime_error);
    EXPECT_THROW(
        {
          [[maybe_unused]] auto inout_axis21 = get_pencil(topology2, topology1);
        },
        std::runtime_error);
  } else {
    // Slab tests
    auto [in_axis01, out_axis01] = get_pencil(topology0, topology1);
    auto [in_axis02, out_axis02] = get_pencil(topology0, topology2);
    auto [in_axis10, out_axis10] = get_pencil(topology1, topology0);
    auto [in_axis12, out_axis12] = get_pencil(topology1, topology2);
    auto [in_axis20, out_axis20] = get_pencil(topology2, topology0);
    auto [in_axis21, out_axis21] = get_pencil(topology2, topology1);

    EXPECT_EQ(in_axis01, 1);
    EXPECT_EQ(out_axis01, 2);
    EXPECT_EQ(in_axis02, 0);
    EXPECT_EQ(out_axis02, 2);
    EXPECT_EQ(in_axis10, 2);
    EXPECT_EQ(out_axis10, 1);
    EXPECT_EQ(in_axis12, 0);
    EXPECT_EQ(out_axis12, 1);
    EXPECT_EQ(in_axis20, 2);
    EXPECT_EQ(out_axis20, 0);
    EXPECT_EQ(in_axis21, 1);
    EXPECT_EQ(out_axis21, 0);

    // Pencil tests
    auto [in_axis34, out_axis34] = get_pencil(topology3, topology4);
    auto [in_axis43, out_axis43] = get_pencil(topology4, topology3);
    EXPECT_EQ(in_axis34, 1);
    EXPECT_EQ(out_axis34, 2);
    EXPECT_EQ(in_axis43, 2);
    EXPECT_EQ(out_axis43, 1);
  }

  // Failure tests because of shape mismatch (or size 1 case)
  EXPECT_THROW(
      {
        [[maybe_unused]] auto inout_axis30 = get_pencil(topology3, topology0);
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        [[maybe_unused]] auto inout_axis31 = get_pencil(topology3, topology1);
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        [[maybe_unused]] auto inout_axis32 = get_pencil(topology3, topology2);
      },
      std::runtime_error);
}

void test_difference_pencil_3D(std::size_t nprocs) {
  using topology_type     = std::array<std::size_t, 3>;
  topology_type topology0 = {nprocs, 1, 8};
  topology_type topology1 = {nprocs, 8, 1};
  topology_type topology2 = {8, nprocs, 1};

  if (nprocs == 1) {
    auto diff01 = find_differences(topology0, topology1);
    auto diff02 = find_differences(topology0, topology2);
    auto diff10 = find_differences(topology1, topology0);
    auto diff12 = find_differences(topology1, topology2);
    auto diff20 = find_differences(topology2, topology0);
    auto diff21 = find_differences(topology2, topology1);

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
    auto diff01 = find_differences(topology0, topology1);
    auto diff02 = find_differences(topology0, topology2);
    auto diff10 = find_differences(topology1, topology0);
    auto diff12 = find_differences(topology1, topology2);
    auto diff20 = find_differences(topology2, topology0);
    auto diff21 = find_differences(topology2, topology1);

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

void test_get_mid_array_pencil_3D(std::size_t nprocs) {
  using topology_type     = std::array<std::size_t, 3>;
  topology_type topology0 = {nprocs, 1, 8};
  topology_type topology1 = {nprocs, 8, 1};
  topology_type topology2 = {8, nprocs, 1};
  topology_type topology3 = {1, 2, nprocs};
  topology_type topology4 = {2, nprocs, 1};

  if (nprocs == 1) {
    // Failure tests because only two elements differ
    EXPECT_THROW(
        { [[maybe_unused]] auto mid01 = get_mid_array(topology0, topology1); },
        std::runtime_error);
    EXPECT_THROW(
        { [[maybe_unused]] auto mid02 = get_mid_array(topology0, topology2); },
        std::runtime_error);
    EXPECT_THROW(
        { [[maybe_unused]] auto mid10 = get_mid_array(topology1, topology0); },
        std::runtime_error);
    EXPECT_THROW(
        { [[maybe_unused]] auto mid12 = get_mid_array(topology1, topology2); },
        std::runtime_error);
    EXPECT_THROW(
        { [[maybe_unused]] auto mid20 = get_mid_array(topology2, topology0); },
        std::runtime_error);
    EXPECT_THROW(
        { [[maybe_unused]] auto mid21 = get_mid_array(topology2, topology1); },
        std::runtime_error);
  } else {
    // Failure tests because only two elements differ
    EXPECT_THROW(
        { [[maybe_unused]] auto mid01 = get_mid_array(topology0, topology1); },
        std::runtime_error);
    EXPECT_THROW(
        { [[maybe_unused]] auto mid10 = get_mid_array(topology1, topology0); },
        std::runtime_error);
    EXPECT_THROW(
        { [[maybe_unused]] auto mid12 = get_mid_array(topology1, topology2); },
        std::runtime_error);
    EXPECT_THROW(
        { [[maybe_unused]] auto mid21 = get_mid_array(topology2, topology1); },
        std::runtime_error);

    auto mid02 = get_mid_array(topology0, topology2);
    auto mid20 = get_mid_array(topology2, topology0);

    topology_type ref_mid02 = {1, nprocs, 8};
    topology_type ref_mid20 = {1, nprocs, 8};

    EXPECT_EQ(mid02, ref_mid02);
    EXPECT_EQ(mid20, ref_mid20);

    auto mid34 = get_mid_array(topology3, topology4);
    auto mid43 = get_mid_array(topology4, topology3);

    topology_type ref_mid34 = {2, 1, nprocs};
    topology_type ref_mid43 = {2, 1, nprocs};

    EXPECT_EQ(mid34, ref_mid34);
    EXPECT_EQ(mid43, ref_mid43);
  }
}

void test_get_shuffled_topologies1D(std::size_t nprocs) {
  using topology_type     = std::array<std::size_t, 3>;
  topology_type topology0 = {1, nprocs, 8};
  topology_type topology1 = {nprocs, 1, 8};
  topology_type topology2 = {8, nprocs, 1};

  using axes_type = std::array<int, 1>;
  axes_type axes0 = {0};
  axes_type axes1 = {1};
  axes_type axes2 = {2};

  std::vector<axes_type> all_axes = {axes0, axes1, axes2};

  if (nprocs == 1) {
    for (const auto& axes : all_axes) {
      // Failure tests because only two elements differ (slabs)
      EXPECT_THROW(
          {
            [[maybe_unused]] auto shuffled_topologies =
                get_shuffled_topologies(topology0, topology1, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto shuffled_topologies =
                get_shuffled_topologies(topology0, topology2, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto shuffled_topologies =
                get_shuffled_topologies(topology1, topology0, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto shuffled_topologies =
                get_shuffled_topologies(topology2, topology0, axes);
          },
          std::runtime_error);
    }
  } else {
    // topology0 to topology0
    // auto shuffled_topologies_0_0_0 = get_shuffled_topologies(topology0,
    // topology0, axes0); auto shuffled_topologies_0_0_1 =
    // get_shuffled_topologies(topology0, topology0, axes1); auto
    // shuffled_topologies_0_0_2 = get_shuffled_topologies(topology0, topology0,
    // axes2); std::vector<topology_type> ref_shuffled_topologies_0_0_0 =
    // {topology0}; EXPECT_EQ(shuffled_topologies_0_0_0,
    // ref_shuffled_topologies_0_0_0); std::vector<topology_type>
    // ref_shuffled_topologies_0_0_1 = {topology0, topology_type{nprocs, 1, 8},
    // topology0}; EXPECT_EQ(shuffled_topologies_0_0_1,
    // ref_shuffled_topologies_0_0_1); std::vector<topology_type>
    // ref_shuffled_topologies_0_0_2 = {topology0, topology_type{8, nprocs, 1},
    // topology0}; EXPECT_EQ(shuffled_topologies_0_0_2,
    // ref_shuffled_topologies_0_0_2);

    // topology0 to topology1
    auto shuffled_topologies_0_1_0 =
        get_shuffled_topologies(topology0, topology1, axes0);
    auto shuffled_topologies_0_1_1 =
        get_shuffled_topologies(topology0, topology1, axes1);
    auto shuffled_topologies_0_1_2 =
        get_shuffled_topologies(topology0, topology1, axes2);

    std::vector<topology_type> ref_shuffled_topologies_0_1_0 = {topology0,
                                                                topology1};
    EXPECT_EQ(shuffled_topologies_0_1_0, ref_shuffled_topologies_0_1_0);
    std::vector<topology_type> ref_shuffled_topologies_0_1_1 = {topology0,
                                                                topology1};
    EXPECT_EQ(shuffled_topologies_0_1_1, ref_shuffled_topologies_0_1_1);
    std::vector<topology_type> ref_shuffled_topologies_0_1_2 = {
        topology0, topology_type{8, nprocs, 1}, topology_type{1, nprocs, 8},
        topology1};
    EXPECT_EQ(shuffled_topologies_0_1_2, ref_shuffled_topologies_0_1_2);

    // topology0 to topology2
    auto shuffled_topologies_0_2_0 =
        get_shuffled_topologies(topology0, topology2, axes0);
    auto shuffled_topologies_0_2_1 =
        get_shuffled_topologies(topology0, topology2, axes1);
    auto shuffled_topologies_0_2_2 =
        get_shuffled_topologies(topology0, topology2, axes2);

    std::vector<topology_type> ref_shuffled_topologies_0_2_0 = {topology0,
                                                                topology2};
    EXPECT_EQ(shuffled_topologies_0_2_0, ref_shuffled_topologies_0_2_0);
    std::vector<topology_type> ref_shuffled_topologies_0_2_1 = {
        topology0, topology_type{nprocs, 1, 8}, topology_type{1, nprocs, 8},
        topology2};
    EXPECT_EQ(shuffled_topologies_0_2_1, ref_shuffled_topologies_0_2_1);
    std::vector<topology_type> ref_shuffled_topologies_0_2_2 = {topology0,
                                                                topology2};
    EXPECT_EQ(shuffled_topologies_0_2_2, ref_shuffled_topologies_0_2_2);

    // topology1 to topology0
    auto shuffled_topologies_1_0_0 =
        get_shuffled_topologies(topology1, topology0, axes0);
    auto shuffled_topologies_1_0_1 =
        get_shuffled_topologies(topology1, topology0, axes1);
    auto shuffled_topologies_1_0_2 =
        get_shuffled_topologies(topology1, topology0, axes2);

    std::vector<topology_type> ref_shuffled_topologies_1_0_0 = {topology1,
                                                                topology0};
    EXPECT_EQ(shuffled_topologies_1_0_0, ref_shuffled_topologies_1_0_0);
    std::vector<topology_type> ref_shuffled_topologies_1_0_1 = {topology1,
                                                                topology0};
    EXPECT_EQ(shuffled_topologies_1_0_1, ref_shuffled_topologies_1_0_1);
    std::vector<topology_type> ref_shuffled_topologies_1_0_2 = {
        topology1, topology_type{nprocs, 8, 1}, topology_type{nprocs, 1, 8},
        topology0};
    EXPECT_EQ(shuffled_topologies_1_0_2, ref_shuffled_topologies_1_0_2);

    //// topology1 to topology1
    // auto shuffled_topologies_1_1_0 = get_shuffled_topologies(topology1,
    // topology1, axes0); auto shuffled_topologies_1_1_1 =
    // get_shuffled_topologies(topology1, topology1, axes1); auto
    // shuffled_topologies_1_1_2 = get_shuffled_topologies(topology1, topology1,
    // axes2); std::vector<topology_type> ref_shuffled_topologies_1_1_1 =
    // {topology1}; EXPECT_EQ(shuffled_topologies_1_1_1,
    // ref_shuffled_topologies_1_1_1);
    ////std::vector<topology_type> ref_shuffled_topologies_1_1_2 = {topology1,
    /// topology_type{nprocs, 8, 1}, topology1};
    ////EXPECT_EQ(shuffled_topologies_1_1_2, ref_shuffled_topologies_1_1_2);

    // topology1 to topology2
    auto shuffled_topologies_1_2_0 =
        get_shuffled_topologies(topology1, topology2, axes0);
    auto shuffled_topologies_1_2_1 =
        get_shuffled_topologies(topology1, topology2, axes1);
    auto shuffled_topologies_1_2_2 =
        get_shuffled_topologies(topology1, topology2, axes2);

    std::vector<topology_type> ref_shuffled_topologies_1_2_0 = {
        topology1, topology_type{1, nprocs, 8}, topology2};
    EXPECT_EQ(shuffled_topologies_1_2_0, ref_shuffled_topologies_1_2_0);
    std::vector<topology_type> ref_shuffled_topologies_1_2_1 = {
        topology1, topology_type{1, nprocs, 8}, topology2};
    EXPECT_EQ(shuffled_topologies_1_2_1, ref_shuffled_topologies_1_2_1);
    std::vector<topology_type> ref_shuffled_topologies_1_2_2 = {
        topology1, topology_type{nprocs, 8, 1}, topology2};
    EXPECT_EQ(shuffled_topologies_1_2_2, ref_shuffled_topologies_1_2_2);
    // std::vector<topology_type> ref_shuffled_topologies_1_2_1 = {topology1,
    // topology0}; EXPECT_EQ(shuffled_topologies_1_0_1,
    // ref_shuffled_topologies_1_0_1); std::vector<topology_type>
    // ref_shuffled_topologies_1_2_2 = {topology1, topology_type{nprocs, 8, 1},
    // topology_type{nprocs, 1, 8}, topology0};
    // EXPECT_EQ(shuffled_topologies_1_0_2, ref_shuffled_topologies_1_0_2);

    // topology2 to topology0
    auto shuffled_topologies_2_0_0 =
        get_shuffled_topologies(topology2, topology0, axes0);
    auto shuffled_topologies_2_0_1 =
        get_shuffled_topologies(topology2, topology0, axes1);
    auto shuffled_topologies_2_0_2 =
        get_shuffled_topologies(topology2, topology0, axes2);

    std::vector<topology_type> ref_shuffled_topologies_2_0_0 = {topology2,
                                                                topology0};
    EXPECT_EQ(shuffled_topologies_2_0_0, ref_shuffled_topologies_2_0_0);
    std::vector<topology_type> ref_shuffled_topologies_2_0_1 = {
        topology2, topology_type{8, 1, nprocs}, topology_type{8, nprocs, 1},
        topology0};
    EXPECT_EQ(shuffled_topologies_2_0_1, ref_shuffled_topologies_2_0_1);
    std::vector<topology_type> ref_shuffled_topologies_2_0_2 = {topology2,
                                                                topology0};
    EXPECT_EQ(shuffled_topologies_2_0_2, ref_shuffled_topologies_2_0_2);

    // topology2 to topology1
    auto shuffled_topologies_2_1_0 =
        get_shuffled_topologies(topology2, topology1, axes0);
    auto shuffled_topologies_2_1_1 =
        get_shuffled_topologies(topology2, topology1, axes1);
    auto shuffled_topologies_2_1_2 =
        get_shuffled_topologies(topology2, topology1, axes2);

    std::vector<topology_type> ref_shuffled_topologies_2_1_0 = {
        topology2, topology_type{1, nprocs, 8}, topology1};
    EXPECT_EQ(shuffled_topologies_2_1_0, ref_shuffled_topologies_2_1_0);
    std::vector<topology_type> ref_shuffled_topologies_2_1_1 = {
        topology2, topology_type{8, 1, nprocs}, topology1};
    EXPECT_EQ(shuffled_topologies_2_1_1, ref_shuffled_topologies_2_1_1);
    std::vector<topology_type> ref_shuffled_topologies_2_1_2 = {
        topology2, topology_type{1, nprocs, 8}, topology1};
    EXPECT_EQ(shuffled_topologies_2_1_2, ref_shuffled_topologies_2_1_2);
  }
}

void test_get_shuffled_topologies3D(std::size_t nprocs) {
  using topology_type     = std::array<std::size_t, 3>;
  topology_type topology0 = {nprocs, 1, 8};
  topology_type topology1 = {nprocs, 8, 1};
  topology_type topology2 = {8, nprocs, 1};

  using axes_type = std::array<int, 3>;
  axes_type axes0 = {0, 1, 2};
  axes_type axes1 = {0, 2, 1};
  axes_type axes2 = {1, 0, 2};
  axes_type axes3 = {1, 2, 0};
  axes_type axes4 = {2, 0, 1};
  axes_type axes5 = {2, 1, 0};

  std::vector<axes_type> all_axes = {axes0, axes1, axes2, axes3, axes4, axes5};

  if (nprocs == 1) {
    for (const auto& axes : all_axes) {
      // Failure tests because only two elements differ (slabs)
      EXPECT_THROW(
          {
            [[maybe_unused]] auto shuffled_topologies =
                get_shuffled_topologies(topology0, topology1, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto shuffled_topologies =
                get_shuffled_topologies(topology0, topology2, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto shuffled_topologies =
                get_shuffled_topologies(topology1, topology0, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto shuffled_topologies =
                get_shuffled_topologies(topology2, topology0, axes);
          },
          std::runtime_error);
    }
  } else {
    // topology0 to topology1
    auto shuffled_topologies_0_1_0 =
        get_shuffled_topologies(topology0, topology1, axes0);
    auto shuffled_topologies_0_1_1 =
        get_shuffled_topologies(topology0, topology1, axes1);
    auto shuffled_topologies_0_1_2 =
        get_shuffled_topologies(topology0, topology1, axes2);
    auto shuffled_topologies_0_1_3 =
        get_shuffled_topologies(topology0, topology1, axes3);
    auto shuffled_topologies_0_1_4 =
        get_shuffled_topologies(topology0, topology1, axes4);
    auto shuffled_topologies_0_1_5 =
        get_shuffled_topologies(topology0, topology1, axes5);
    std::vector<topology_type> ref_shuffled_topologies_0_1_0 = {
        topology0,
        topology_type{nprocs, 8, 1},
        topology_type{nprocs, 1, 8},
        topology_type{1, nprocs, 8},
        topology_type{nprocs, 1, 8},
        topology1};
    EXPECT_EQ(shuffled_topologies_0_1_0, ref_shuffled_topologies_0_1_0);

    // toplogy0 is already Y-pencil
    std::vector<topology_type> ref_shuffled_topologies_0_1_1 = {
        topology0, topology_type{nprocs, 8, 1}, topology_type{1, 8, nprocs},
        topology1};
    EXPECT_EQ(shuffled_topologies_0_1_1, ref_shuffled_topologies_0_1_1);

    std::vector<topology_type> ref_shuffled_topologies_0_1_2 = {
        topology0,
        topology_type{nprocs, 8, 1},
        topology_type{1, 8, nprocs},
        topology_type{8, 1, nprocs},
        topology_type{1, 8, nprocs},
        topology1};
    EXPECT_EQ(shuffled_topologies_0_1_2, ref_shuffled_topologies_0_1_2);

    std::vector<topology_type> ref_shuffled_topologies_0_1_3 = {
        topology0,
        topology_type{1, nprocs, 8},
        topology_type{8, nprocs, 1},
        topology_type{8, 1, nprocs},
        topology_type{1, 8, nprocs},
        topology1};
    EXPECT_EQ(shuffled_topologies_0_1_3, ref_shuffled_topologies_0_1_3);

    // toplogy0 is already Y-pencil
    std::vector<topology_type> ref_shuffled_topologies_0_1_4 = {
        topology0, topology_type{1, nprocs, 8}, topology_type{8, nprocs, 1},
        topology1};
    EXPECT_EQ(shuffled_topologies_0_1_4, ref_shuffled_topologies_0_1_4);

    std::vector<topology_type> ref_shuffled_topologies_0_1_5 = {
        topology_type{nprocs, 1, 8}, topology_type{1, nprocs, 8},
        topology_type{nprocs, 1, 8}, topology_type{nprocs, 8, 1}};
    EXPECT_EQ(shuffled_topologies_0_1_5, ref_shuffled_topologies_0_1_5);

    // topology1 to topology0
    auto shuffled_topologies_1_0_0 =
        get_shuffled_topologies(topology1, topology0, axes0);
    auto shuffled_topologies_1_0_1 =
        get_shuffled_topologies(topology1, topology0, axes1);
    auto shuffled_topologies_1_0_2 =
        get_shuffled_topologies(topology1, topology0, axes2);
    auto shuffled_topologies_1_0_3 =
        get_shuffled_topologies(topology1, topology0, axes3);
    auto shuffled_topologies_1_0_4 =
        get_shuffled_topologies(topology1, topology0, axes4);
    auto shuffled_topologies_1_0_5 =
        get_shuffled_topologies(topology1, topology0, axes5);

    // Topology1 is already Z-pencil
    std::vector<topology_type> ref_shuffled_topologies_1_0_0 = {
        topology1, topology_type{nprocs, 1, 8}, topology_type{1, nprocs, 8},
        topology0};
    EXPECT_EQ(shuffled_topologies_1_0_0, ref_shuffled_topologies_1_0_0);

    std::vector<topology_type> ref_shuffled_topologies_1_0_1 = {
        topology1,
        topology_type{nprocs, 1, 8},
        topology_type{nprocs, 8, 1},
        topology_type{1, 8, nprocs},
        topology_type{nprocs, 8, 1},
        topology0};
    EXPECT_EQ(shuffled_topologies_1_0_1, ref_shuffled_topologies_1_0_1);

    // Topology1 is already Z-pencil
    std::vector<topology_type> ref_shuffled_topologies_1_0_2 = {
        topology1, topology_type{1, 8, nprocs}, topology_type{8, 1, nprocs},
        topology0};
    EXPECT_EQ(shuffled_topologies_1_0_2, ref_shuffled_topologies_1_0_2);

    std::vector<topology_type> ref_shuffled_topologies_1_0_3 = {
        topology1, topology_type{1, 8, nprocs}, topology_type{nprocs, 8, 1},
        topology0};
    EXPECT_EQ(shuffled_topologies_1_0_3, ref_shuffled_topologies_1_0_3);

    std::vector<topology_type> ref_shuffled_topologies_1_0_4 = {
        topology1,
        topology_type{nprocs, 1, 8},
        topology_type{1, nprocs, 8},
        topology_type{8, nprocs, 1},
        topology_type{1, nprocs, 8},
        topology0};
    EXPECT_EQ(shuffled_topologies_1_0_4, ref_shuffled_topologies_1_0_4);

    std::vector<topology_type> ref_shuffled_topologies_1_0_5 = {
        topology1,
        topology_type{1, 8, nprocs},
        topology_type{8, 1, nprocs},
        topology_type{8, nprocs, 1},
        topology_type{1, nprocs, 8},
        topology0};
    EXPECT_EQ(shuffled_topologies_1_0_5, ref_shuffled_topologies_1_0_5);

    // topology0 to topology2
    auto shuffled_topologies_0_2_0 =
        get_shuffled_topologies(topology0, topology2, axes0);
    auto shuffled_topologies_0_2_1 =
        get_shuffled_topologies(topology0, topology2, axes1);
    auto shuffled_topologies_0_2_2 =
        get_shuffled_topologies(topology0, topology2, axes2);
    auto shuffled_topologies_0_2_3 =
        get_shuffled_topologies(topology0, topology2, axes3);
    auto shuffled_topologies_0_2_4 =
        get_shuffled_topologies(topology0, topology2, axes4);
    auto shuffled_topologies_0_2_5 =
        get_shuffled_topologies(topology0, topology2, axes5);

    std::vector<topology_type> ref_shuffled_topologies_0_2_0 = {
        topology0, topology_type{nprocs, 8, 1}, topology_type{nprocs, 1, 8},
        topology_type{1, nprocs, 8}, topology2};
    EXPECT_EQ(shuffled_topologies_0_2_0, ref_shuffled_topologies_0_2_0);

    // Topology0 is already Y-pencil
    std::vector<topology_type> ref_shuffled_topologies_0_2_1 = {
        topology0, topology_type{nprocs, 8, 1}, topology_type{1, 8, nprocs},
        topology_type{8, 1, nprocs}, topology2};
    EXPECT_EQ(shuffled_topologies_0_2_1, ref_shuffled_topologies_0_2_1);

    std::vector<topology_type> ref_shuffled_topologies_0_2_2 = {
        topology0, topology_type{nprocs, 8, 1}, topology_type{1, 8, nprocs},
        topology_type{8, 1, nprocs}, topology2};
    EXPECT_EQ(shuffled_topologies_0_2_2, ref_shuffled_topologies_0_2_2);

    std::vector<topology_type> ref_shuffled_topologies_0_2_3 = {
        topology0, topology_type{1, nprocs, 8}, topology_type{8, nprocs, 1},
        topology_type{8, 1, nprocs}, topology2};
    EXPECT_EQ(shuffled_topologies_0_2_3, ref_shuffled_topologies_0_2_3);

    // Topology0 is already Y-pencil
    std::vector<topology_type> ref_shuffled_topologies_0_2_4 = {
        topology0, topology_type{1, nprocs, 8}, topology2};
    EXPECT_EQ(shuffled_topologies_0_2_4, ref_shuffled_topologies_0_2_4);

    std::vector<topology_type> ref_shuffled_topologies_0_2_5 = {
        topology0, topology_type{1, nprocs, 8}, topology_type{nprocs, 1, 8},
        topology_type{nprocs, 8, 1}, topology2};
    EXPECT_EQ(shuffled_topologies_0_2_5, ref_shuffled_topologies_0_2_5);

    // topology2 to topology0
    auto shuffled_topologies_2_0_0 =
        get_shuffled_topologies(topology2, topology0, axes0);
    auto shuffled_topologies_2_0_1 =
        get_shuffled_topologies(topology2, topology0, axes1);
    auto shuffled_topologies_2_0_2 =
        get_shuffled_topologies(topology2, topology0, axes2);
    auto shuffled_topologies_2_0_3 =
        get_shuffled_topologies(topology2, topology0, axes3);
    auto shuffled_topologies_2_0_4 =
        get_shuffled_topologies(topology2, topology0, axes4);
    auto shuffled_topologies_2_0_5 =
        get_shuffled_topologies(topology2, topology0, axes5);

    // topology2 is already Z-pencil
    std::vector<topology_type> ref_shuffled_topologies_2_0_0 = {
        topology2, topology_type{8, 1, nprocs}, topology_type{1, 8, nprocs},
        topology_type{nprocs, 8, 1}, topology0};
    EXPECT_EQ(shuffled_topologies_2_0_0, ref_shuffled_topologies_2_0_0);

    std::vector<topology_type> ref_shuffled_topologies_2_0_1 = {
        topology2, topology_type{8, 1, nprocs}, topology_type{8, nprocs, 1},
        topology_type{1, nprocs, 8}, topology0};
    EXPECT_EQ(shuffled_topologies_2_0_1, ref_shuffled_topologies_2_0_1);

    // topology2 is already Z-pencil
    std::vector<topology_type> ref_shuffled_topologies_2_0_2 = {
        topology2, topology_type{1, nprocs, 8}, topology0};
    EXPECT_EQ(shuffled_topologies_2_0_2, ref_shuffled_topologies_2_0_2);

    std::vector<topology_type> ref_shuffled_topologies_2_0_3 = {
        topology2, topology_type{1, nprocs, 8}, topology_type{8, nprocs, 1},
        topology_type{8, 1, nprocs}, topology0};
    EXPECT_EQ(shuffled_topologies_2_0_3, ref_shuffled_topologies_2_0_3);

    std::vector<topology_type> ref_shuffled_topologies_2_0_4 = {
        topology2, topology_type{8, 1, nprocs}, topology_type{1, 8, nprocs},
        topology_type{nprocs, 8, 1}, topology0};
    EXPECT_EQ(shuffled_topologies_2_0_4, ref_shuffled_topologies_2_0_4);

    std::vector<topology_type> ref_shuffled_topologies_2_0_5 = {
        topology2, topology_type{1, nprocs, 8}, topology_type{nprocs, 1, 8},
        topology_type{nprocs, 8, 1}, topology0};
    EXPECT_EQ(shuffled_topologies_2_0_5, ref_shuffled_topologies_2_0_5);
  }

  for (const auto& axes : all_axes) {
    // Failure tests because only two elements differ (slabs)
    EXPECT_THROW(
        {
          [[maybe_unused]] auto shuffled_topologies =
              get_shuffled_topologies(topology1, topology2, axes);
        },
        std::runtime_error);
    EXPECT_THROW(
        {
          [[maybe_unused]] auto shuffled_topologies =
              get_shuffled_topologies(topology2, topology1, axes);
        },
        std::runtime_error);
  }
}

}  // namespace

TEST_P(PencilParamTests, GetPencil3D) {
  int n0 = GetParam();
  test_get_pencil_3D(n0);
}

TEST_P(PencilParamTests, FindDifference3D) {
  int n0 = GetParam();
  test_difference_pencil_3D(n0);
}

TEST_P(PencilParamTests, GetMidArray3D) {
  int n0 = GetParam();
  test_get_mid_array_pencil_3D(n0);
}

TEST_P(PencilParamTests, GetShuffledTopologies1D) {
  int n0 = GetParam();
  test_get_shuffled_topologies1D(n0);
}

TEST_P(PencilParamTests, GetShuffledTopologies3D) {
  int n0 = GetParam();
  test_get_shuffled_topologies3D(n0);
}

INSTANTIATE_TEST_SUITE_P(PencilTests, PencilParamTests,
                         ::testing::Values(1, 2, 3, 4, 5, 6));
