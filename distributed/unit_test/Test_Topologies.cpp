#include <mpi.h>
#include <gtest/gtest.h>
#include <iostream>
#include <Kokkos_Core.hpp>
#include "Topologies.hpp"
#include "Test_Utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
class TopologyParamTests : public ::testing::TestWithParam<int> {};
class SlabParamTests : public ::testing::TestWithParam<int> {};
class PencilParamTests : public ::testing::TestWithParam<int> {};

void test_get_slab_2D(std::size_t nprocs) {
  using topology_type     = std::array<std::size_t, 2>;
  topology_type topology0 = {1, nprocs};
  topology_type topology1 = {nprocs, 1};
  topology_type topology2 = {nprocs, 7};
  topology_type topology3 = {1, 1};

  if (nprocs == 1) {
    // Failure tests because of size 1 case
    EXPECT_THROW(
        {
          [[maybe_unused]] auto inout_axis01 = get_slab(topology0, topology1);
        },
        std::runtime_error);
    EXPECT_THROW(
        {
          [[maybe_unused]] auto inout_axis10 = get_slab(topology1, topology0);
        },
        std::runtime_error);
  } else {
    // Slab tests
    auto [in_axis01, out_axis01] = get_slab(topology0, topology1);
    auto [in_axis10, out_axis10] = get_slab(topology1, topology0);

    EXPECT_EQ(in_axis01, 0);
    EXPECT_EQ(out_axis01, 1);
    EXPECT_EQ(in_axis10, 1);
    EXPECT_EQ(out_axis10, 0);
  }

  // Failure tests because of shape mismatch (or size 1 case)
  EXPECT_THROW(
      { [[maybe_unused]] auto inout_axis02 = get_slab(topology0, topology2); },
      std::runtime_error);
  EXPECT_THROW(
      { [[maybe_unused]] auto inout_axis03 = get_slab(topology0, topology3); },
      std::runtime_error);
}

void test_get_slab_3D(std::size_t nprocs) {
  using topology_type     = std::array<std::size_t, 3>;
  topology_type topology0 = {1, 1, nprocs};
  topology_type topology1 = {1, nprocs, 1};
  topology_type topology2 = {nprocs, 1, 1};
  topology_type topology3 = {1, nprocs, 7};
  topology_type topology4 = {1, 1, 1};

  if (nprocs == 1) {
    // Failure tests because of size 1 case
    EXPECT_THROW(
        {
          [[maybe_unused]] auto inout_axis01 = get_slab(topology0, topology1);
        },
        std::runtime_error);
    EXPECT_THROW(
        {
          [[maybe_unused]] auto inout_axis02 = get_slab(topology0, topology2);
        },
        std::runtime_error);
    EXPECT_THROW(
        {
          [[maybe_unused]] auto inout_axis10 = get_slab(topology1, topology0);
        },
        std::runtime_error);
    EXPECT_THROW(
        {
          [[maybe_unused]] auto inout_axis12 = get_slab(topology1, topology2);
        },
        std::runtime_error);
    EXPECT_THROW(
        {
          [[maybe_unused]] auto inout_axis20 = get_slab(topology2, topology0);
        },
        std::runtime_error);
    EXPECT_THROW(
        {
          [[maybe_unused]] auto inout_axis21 = get_slab(topology2, topology1);
        },
        std::runtime_error);
  } else {
    // Slab tests
    auto [in_axis01, out_axis01] = get_slab(topology0, topology1);
    auto [in_axis02, out_axis02] = get_slab(topology0, topology2);
    auto [in_axis10, out_axis10] = get_slab(topology1, topology0);
    auto [in_axis12, out_axis12] = get_slab(topology1, topology2);
    auto [in_axis20, out_axis20] = get_slab(topology2, topology0);
    auto [in_axis21, out_axis21] = get_slab(topology2, topology1);

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
  }

  // Failure tests because of shape mismatch (or size 1 case)
  EXPECT_THROW(
      { [[maybe_unused]] auto inout_axis03 = get_slab(topology0, topology3); },
      std::runtime_error);
  EXPECT_THROW(
      { [[maybe_unused]] auto inout_axis04 = get_slab(topology0, topology4); },
      std::runtime_error);
}

void test_get_all_slab_topologies1D_3DView(std::size_t nprocs) {
  using topology_type     = std::array<std::size_t, 3>;
  topology_type topology0 = {1, 1, nprocs};
  topology_type topology1 = {1, nprocs, 1};
  topology_type topology2 = {nprocs, 1, 1};

  using axes_type = std::array<std::size_t, 1>;
  axes_type axes0 = {0};
  axes_type axes1 = {1};
  axes_type axes2 = {2};

  std::vector<axes_type> all_axes = {axes0, axes1, axes2};

  if (nprocs == 1) {
    for (const auto& axes : all_axes) {
      // Failure tests because these are shared topologies
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology0, topology1, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology0, topology2, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology1, topology0, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology2, topology0, axes);
          },
          std::runtime_error);
    }
  } else {
    // topology0 (XY-slab) to topology0 (XY-slab)
    auto all_slab_topologies_0_0_ax0 =
        get_all_slab_topologies(topology0, topology0, axes0);
    auto all_slab_topologies_0_0_ax1 =
        get_all_slab_topologies(topology0, topology0, axes1);
    auto all_slab_topologies_0_0_ax2 =
        get_all_slab_topologies(topology0, topology0, axes2);

    // [Remark] Not sure which one is better {topology0, topology0} or
    // {topology0}
    std::vector<topology_type> ref_all_slab_topologies_0_0_ax0 = {topology0};
    EXPECT_EQ(all_slab_topologies_0_0_ax0, ref_all_slab_topologies_0_0_ax0);
    std::vector<topology_type> ref_all_slab_topologies_0_0_ax1 = {topology0};
    EXPECT_EQ(all_slab_topologies_0_0_ax1, ref_all_slab_topologies_0_0_ax1);
    std::vector<topology_type> ref_all_slab_topologies_0_0_ax2 = {
        topology0, topology2, topology0};
    EXPECT_EQ(all_slab_topologies_0_0_ax2, ref_all_slab_topologies_0_0_ax2);

    // topology0 (XY-slab) to topology1 (XZ-slab)
    auto all_slab_topologies_0_1_ax0 =
        get_all_slab_topologies(topology0, topology1, axes0);
    auto all_slab_topologies_0_1_ax1 =
        get_all_slab_topologies(topology0, topology1, axes1);
    auto all_slab_topologies_0_1_ax2 =
        get_all_slab_topologies(topology0, topology1, axes2);

    std::vector<topology_type> ref_all_slab_topologies_0_1_ax0 = {topology0,
                                                                  topology1};
    EXPECT_EQ(all_slab_topologies_0_1_ax0, ref_all_slab_topologies_0_1_ax0);
    std::vector<topology_type> ref_all_slab_topologies_0_1_ax1 = {topology0,
                                                                  topology1};
    EXPECT_EQ(all_slab_topologies_0_1_ax1, ref_all_slab_topologies_0_1_ax1);
    std::vector<topology_type> ref_all_slab_topologies_0_1_ax2 = {topology0,
                                                                  topology1};
    EXPECT_EQ(all_slab_topologies_0_1_ax2, ref_all_slab_topologies_0_1_ax2);

    // topology0 (XY-slab) to topology2 (YZ-slab)
    auto all_slab_topologies_0_2_ax0 =
        get_all_slab_topologies(topology0, topology2, axes0);
    auto all_slab_topologies_0_2_ax1 =
        get_all_slab_topologies(topology0, topology2, axes1);
    auto all_slab_topologies_0_2_ax2 =
        get_all_slab_topologies(topology0, topology2, axes2);

    std::vector<topology_type> ref_all_slab_topologies_0_2_ax0 = {topology0,
                                                                  topology2};
    EXPECT_EQ(all_slab_topologies_0_2_ax0, ref_all_slab_topologies_0_2_ax0);
    std::vector<topology_type> ref_all_slab_topologies_0_2_ax1 = {topology0,
                                                                  topology2};
    EXPECT_EQ(all_slab_topologies_0_2_ax1, ref_all_slab_topologies_0_2_ax1);
    std::vector<topology_type> ref_all_slab_topologies_0_2_ax2 = {topology0,
                                                                  topology2};
    EXPECT_EQ(all_slab_topologies_0_2_ax2, ref_all_slab_topologies_0_2_ax2);

    // topology1 (XZ-slab) to topology0 (XY-slab)
    auto all_slab_topologies_1_0_ax0 =
        get_all_slab_topologies(topology1, topology0, axes0);
    auto all_slab_topologies_1_0_ax1 =
        get_all_slab_topologies(topology1, topology0, axes1);
    auto all_slab_topologies_1_0_ax2 =
        get_all_slab_topologies(topology1, topology0, axes2);

    std::vector<topology_type> ref_all_slab_topologies_1_0_ax0 = {topology1,
                                                                  topology0};
    EXPECT_EQ(all_slab_topologies_1_0_ax0, ref_all_slab_topologies_1_0_ax0);
    std::vector<topology_type> ref_all_slab_topologies_1_0_ax1 = {topology1,
                                                                  topology0};
    EXPECT_EQ(all_slab_topologies_1_0_ax1, ref_all_slab_topologies_1_0_ax1);
    std::vector<topology_type> ref_all_slab_topologies_1_0_ax2 = {topology1,
                                                                  topology0};
    EXPECT_EQ(all_slab_topologies_1_0_ax2, ref_all_slab_topologies_1_0_ax2);

    // topology1 (XZ-slab) to topology1 (XZ-slab)
    auto all_slab_topologies_1_1_ax0 =
        get_all_slab_topologies(topology1, topology1, axes0);
    auto all_slab_topologies_1_1_ax1 =
        get_all_slab_topologies(topology1, topology1, axes1);
    auto all_slab_topologies_1_1_ax2 =
        get_all_slab_topologies(topology1, topology1, axes2);

    // Should this be topo1 -> topo2 -> topo1
    std::vector<topology_type> ref_all_slab_topologies_1_1_ax0 = {topology1};
    EXPECT_EQ(all_slab_topologies_1_1_ax0, ref_all_slab_topologies_1_1_ax0);
    std::vector<topology_type> ref_all_slab_topologies_1_1_ax1 = {
        topology1, topology2, topology1};
    EXPECT_EQ(all_slab_topologies_1_1_ax1, ref_all_slab_topologies_1_1_ax1);
    std::vector<topology_type> ref_all_slab_topologies_1_1_ax2 = {topology1};
    EXPECT_EQ(all_slab_topologies_1_1_ax2, ref_all_slab_topologies_1_1_ax2);

    // topology1 (XZ-slab) to topology2 (YZ-slab)
    auto all_slab_topologies_1_2_ax0 =
        get_all_slab_topologies(topology1, topology2, axes0);
    auto all_slab_topologies_1_2_ax1 =
        get_all_slab_topologies(topology1, topology2, axes1);
    auto all_slab_topologies_1_2_ax2 =
        get_all_slab_topologies(topology1, topology2, axes2);

    std::vector<topology_type> ref_all_slab_topologies_1_2_ax0 = {topology1,
                                                                  topology2};
    EXPECT_EQ(all_slab_topologies_1_2_ax0, ref_all_slab_topologies_1_2_ax0);
    std::vector<topology_type> ref_all_slab_topologies_1_2_ax1 = {topology1,
                                                                  topology2};
    EXPECT_EQ(all_slab_topologies_1_2_ax1, ref_all_slab_topologies_1_2_ax1);
    std::vector<topology_type> ref_all_slab_topologies_1_2_ax2 = {topology1,
                                                                  topology2};
    EXPECT_EQ(all_slab_topologies_1_2_ax2, ref_all_slab_topologies_1_2_ax2);

    // topology2 (YZ-slab) to topology0 (XY-slab)
    auto all_slab_topologies_2_0_ax0 =
        get_all_slab_topologies(topology2, topology0, axes0);
    auto all_slab_topologies_2_0_ax1 =
        get_all_slab_topologies(topology2, topology0, axes1);
    auto all_slab_topologies_2_0_ax2 =
        get_all_slab_topologies(topology2, topology0, axes2);

    std::vector<topology_type> ref_all_slab_topologies_2_0_ax0 = {topology2,
                                                                  topology0};
    EXPECT_EQ(all_slab_topologies_2_0_ax0, ref_all_slab_topologies_2_0_ax0);
    std::vector<topology_type> ref_all_slab_topologies_2_0_ax1 = {topology2,
                                                                  topology0};
    EXPECT_EQ(all_slab_topologies_2_0_ax1, ref_all_slab_topologies_2_0_ax1);
    std::vector<topology_type> ref_all_slab_topologies_2_0_ax2 = {topology2,
                                                                  topology0};
    EXPECT_EQ(all_slab_topologies_2_0_ax2, ref_all_slab_topologies_2_0_ax2);

    // topology2 (YZ-slab) to topology1 (XZ-slab)
    auto all_slab_topologies_2_1_ax0 =
        get_all_slab_topologies(topology2, topology1, axes0);
    auto all_slab_topologies_2_1_ax1 =
        get_all_slab_topologies(topology2, topology1, axes1);
    auto all_slab_topologies_2_1_ax2 =
        get_all_slab_topologies(topology2, topology1, axes2);

    std::vector<topology_type> ref_all_slab_topologies_2_1_ax0 = {topology2,
                                                                  topology1};
    EXPECT_EQ(all_slab_topologies_2_1_ax0, ref_all_slab_topologies_2_1_ax0);
    std::vector<topology_type> ref_all_slab_topologies_2_1_ax1 = {topology2,
                                                                  topology1};
    EXPECT_EQ(all_slab_topologies_2_1_ax1, ref_all_slab_topologies_2_1_ax1);
    std::vector<topology_type> ref_all_slab_topologies_2_1_ax2 = {topology2,
                                                                  topology1};
    EXPECT_EQ(all_slab_topologies_2_1_ax2, ref_all_slab_topologies_2_1_ax2);

    // topology2 (YZ-slab) to topology2 (YZ-slab)
    auto all_slab_topologies_2_2_ax0 =
        get_all_slab_topologies(topology2, topology2, axes0);
    auto all_slab_topologies_2_2_ax1 =
        get_all_slab_topologies(topology2, topology2, axes1);
    auto all_slab_topologies_2_2_ax2 =
        get_all_slab_topologies(topology2, topology2, axes2);

    std::vector<topology_type> ref_all_slab_topologies_2_2_ax0 = {
        topology2, topology1, topology2};
    EXPECT_EQ(all_slab_topologies_2_2_ax0, ref_all_slab_topologies_2_2_ax0);
    std::vector<topology_type> ref_all_slab_topologies_2_2_ax1 = {topology2};
    EXPECT_EQ(all_slab_topologies_2_2_ax1, ref_all_slab_topologies_2_2_ax1);
    std::vector<topology_type> ref_all_slab_topologies_2_2_ax2 = {topology2};
    EXPECT_EQ(all_slab_topologies_2_2_ax2, ref_all_slab_topologies_2_2_ax2);
  }
}

void test_get_all_slab_topologies2D_2DView(std::size_t nprocs) {
  using topology_type     = std::array<std::size_t, 2>;
  topology_type topology0 = {1, nprocs};
  topology_type topology1 = {nprocs, 1};

  using axes_type  = std::array<std::size_t, 2>;
  axes_type axes01 = {0, 1};
  axes_type axes10 = {1, 0};

  std::vector<axes_type> all_axes = {axes01, axes10};

  if (nprocs == 1) {
    for (const auto& axes : all_axes) {
      // Failure tests because these are shared topologies
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology0, topology1, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology1, topology0, axes);
          },
          std::runtime_error);
    }
  } else {
    // topology0 (X-slab) to topology0 (X-slab)
    auto all_slab_topologies_0_0_ax01 =
        get_all_slab_topologies(topology0, topology0, axes01);
    auto all_slab_topologies_0_0_ax10 =
        get_all_slab_topologies(topology0, topology0, axes10);

    std::vector<topology_type> ref_all_slab_topologies_0_0_ax01 = {
        topology0, topology1, topology0};
    EXPECT_EQ(all_slab_topologies_0_0_ax01, ref_all_slab_topologies_0_0_ax01);
    std::vector<topology_type> ref_all_slab_topologies_0_0_ax10 = {
        topology0, topology1, topology0};
    EXPECT_EQ(all_slab_topologies_0_0_ax10, ref_all_slab_topologies_0_0_ax10);

    // topology0 (X-slab) to topology1 (Y-slab)
    auto all_slab_topologies_0_1_ax01 =
        get_all_slab_topologies(topology0, topology1, axes01);
    auto all_slab_topologies_0_1_ax10 =
        get_all_slab_topologies(topology0, topology1, axes10);

    std::vector<topology_type> ref_all_slab_topologies_0_1_ax01 = {
        topology0, topology1, topology0, topology1};
    EXPECT_EQ(all_slab_topologies_0_1_ax01, ref_all_slab_topologies_0_1_ax01);
    std::vector<topology_type> ref_all_slab_topologies_0_1_ax10 = {topology0,
                                                                   topology1};
    EXPECT_EQ(all_slab_topologies_0_1_ax10, ref_all_slab_topologies_0_1_ax10);

    // topology1 (Y-slab) to topology0 (X-slab)
    auto all_slab_topologies_1_0_ax01 =
        get_all_slab_topologies(topology1, topology0, axes01);
    auto all_slab_topologies_1_0_ax10 =
        get_all_slab_topologies(topology1, topology0, axes10);

    std::vector<topology_type> ref_all_slab_topologies_1_0_ax01 = {topology1,
                                                                   topology0};
    EXPECT_EQ(all_slab_topologies_1_0_ax01, ref_all_slab_topologies_1_0_ax01);
    std::vector<topology_type> ref_all_slab_topologies_1_0_ax10 = {
        topology1, topology0, topology1, topology0};
    EXPECT_EQ(all_slab_topologies_1_0_ax10, ref_all_slab_topologies_1_0_ax10);

    // topology1 (Y-slab) to topology1 (Y-slab)
    auto all_slab_topologies_1_1_ax01 =
        get_all_slab_topologies(topology1, topology1, axes01);
    auto all_slab_topologies_1_1_ax10 =
        get_all_slab_topologies(topology1, topology1, axes10);

    std::vector<topology_type> ref_all_slab_topologies_1_1_ax01 = {
        topology1, topology0, topology1};
    EXPECT_EQ(all_slab_topologies_1_1_ax01, ref_all_slab_topologies_1_1_ax01);
    std::vector<topology_type> ref_all_slab_topologies_1_1_ax10 = {
        topology1, topology0, topology1};
    EXPECT_EQ(all_slab_topologies_1_1_ax10, ref_all_slab_topologies_1_1_ax10);
  }
}

void test_get_all_slab_topologies2D_3DView(std::size_t nprocs) {
  using topology_type     = std::array<std::size_t, 3>;
  topology_type topology0 = {1, 1, nprocs};
  topology_type topology1 = {1, nprocs, 1};
  topology_type topology2 = {nprocs, 1, 1};

  using axes_type  = std::array<std::size_t, 2>;
  axes_type axes01 = {0, 1};
  axes_type axes02 = {0, 2};
  axes_type axes10 = {1, 0};
  axes_type axes12 = {1, 2};
  axes_type axes20 = {2, 0};
  axes_type axes21 = {2, 1};

  std::vector<axes_type> all_axes = {axes01, axes02, axes10,
                                     axes12, axes20, axes21};

  if (nprocs == 1) {
    for (const auto& axes : all_axes) {
      // Failure tests because these are shared topologies
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology0, topology0, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology0, topology1, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology0, topology2, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology1, topology0, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology1, topology1, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology1, topology2, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology2, topology0, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology2, topology1, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology2, topology2, axes);
          },
          std::runtime_error);
    }
  } else {
    // topology0 (XY-slab) to topology0 (XY-slab)
    auto all_slab_topologies_0_0_ax01 =
        get_all_slab_topologies(topology0, topology0, axes01);
    auto all_slab_topologies_0_0_ax02 =
        get_all_slab_topologies(topology0, topology0, axes02);
    auto all_slab_topologies_0_0_ax10 =
        get_all_slab_topologies(topology0, topology0, axes10);
    auto all_slab_topologies_0_0_ax12 =
        get_all_slab_topologies(topology0, topology0, axes12);
    auto all_slab_topologies_0_0_ax20 =
        get_all_slab_topologies(topology0, topology0, axes20);
    auto all_slab_topologies_0_0_ax21 =
        get_all_slab_topologies(topology0, topology0, axes21);

    std::vector<topology_type> ref_all_slab_topologies_0_0_ax01 = {topology0};
    EXPECT_EQ(all_slab_topologies_0_0_ax01, ref_all_slab_topologies_0_0_ax01);
    std::vector<topology_type> ref_all_slab_topologies_0_0_ax02 = {
        topology0, topology1, topology0};
    EXPECT_EQ(all_slab_topologies_0_0_ax02, ref_all_slab_topologies_0_0_ax02);
    std::vector<topology_type> ref_all_slab_topologies_0_0_ax10 = {topology0};
    EXPECT_EQ(all_slab_topologies_0_0_ax10, ref_all_slab_topologies_0_0_ax10);
    std::vector<topology_type> ref_all_slab_topologies_0_0_ax12 = {
        topology0, topology2, topology0};
    EXPECT_EQ(all_slab_topologies_0_0_ax12, ref_all_slab_topologies_0_0_ax12);
    std::vector<topology_type> ref_all_slab_topologies_0_0_ax20 = {
        topology0, topology2, topology0};
    EXPECT_EQ(all_slab_topologies_0_0_ax20, ref_all_slab_topologies_0_0_ax20);
    std::vector<topology_type> ref_all_slab_topologies_0_0_ax21 = {
        topology0, topology2, topology0};
    EXPECT_EQ(all_slab_topologies_0_0_ax21, ref_all_slab_topologies_0_0_ax21);

    // topology0 (XY-slab) to topology1 (XZ-slab)
    auto all_slab_topologies_0_1_ax01 =
        get_all_slab_topologies(topology0, topology1, axes01);
    auto all_slab_topologies_0_1_ax02 =
        get_all_slab_topologies(topology0, topology1, axes02);
    auto all_slab_topologies_0_1_ax10 =
        get_all_slab_topologies(topology0, topology1, axes10);
    auto all_slab_topologies_0_1_ax12 =
        get_all_slab_topologies(topology0, topology1, axes12);
    auto all_slab_topologies_0_1_ax20 =
        get_all_slab_topologies(topology0, topology1, axes20);
    auto all_slab_topologies_0_1_ax21 =
        get_all_slab_topologies(topology0, topology1, axes21);

    std::vector<topology_type> ref_all_slab_topologies_0_1_ax01 = {topology0,
                                                                   topology1};
    EXPECT_EQ(all_slab_topologies_0_1_ax01, ref_all_slab_topologies_0_1_ax01);
    std::vector<topology_type> ref_all_slab_topologies_0_1_ax02 = {topology0,
                                                                   topology1};
    EXPECT_EQ(all_slab_topologies_0_1_ax02, ref_all_slab_topologies_0_1_ax02);
    std::vector<topology_type> ref_all_slab_topologies_0_1_ax10 = {topology0,
                                                                   topology1};
    EXPECT_EQ(all_slab_topologies_0_1_ax10, ref_all_slab_topologies_0_1_ax10);
    std::vector<topology_type> ref_all_slab_topologies_0_1_ax12 = {
        topology0, topology2, topology1};
    EXPECT_EQ(all_slab_topologies_0_1_ax12, ref_all_slab_topologies_0_1_ax12);
    std::vector<topology_type> ref_all_slab_topologies_0_1_ax20 = {topology0,
                                                                   topology1};
    EXPECT_EQ(all_slab_topologies_0_1_ax20, ref_all_slab_topologies_0_1_ax20);
    std::vector<topology_type> ref_all_slab_topologies_0_1_ax21 = {topology0,
                                                                   topology1};
    EXPECT_EQ(all_slab_topologies_0_1_ax21, ref_all_slab_topologies_0_1_ax21);

    // topology0 (XY-slab) to topology2 (YZ-slab)
    auto all_slab_topologies_0_2_ax01 =
        get_all_slab_topologies(topology0, topology2, axes01);
    auto all_slab_topologies_0_2_ax02 =
        get_all_slab_topologies(topology0, topology2, axes02);
    auto all_slab_topologies_0_2_ax10 =
        get_all_slab_topologies(topology0, topology2, axes10);
    auto all_slab_topologies_0_2_ax12 =
        get_all_slab_topologies(topology0, topology2, axes12);
    auto all_slab_topologies_0_2_ax20 =
        get_all_slab_topologies(topology0, topology2, axes20);
    auto all_slab_topologies_0_2_ax21 =
        get_all_slab_topologies(topology0, topology2, axes21);

    std::vector<topology_type> ref_all_slab_topologies_0_2_ax01 = {topology0,
                                                                   topology2};
    EXPECT_EQ(all_slab_topologies_0_2_ax01, ref_all_slab_topologies_0_2_ax01);
    std::vector<topology_type> ref_all_slab_topologies_0_2_ax02 = {
        topology0, topology1, topology2};
    EXPECT_EQ(all_slab_topologies_0_2_ax02, ref_all_slab_topologies_0_2_ax02);
    std::vector<topology_type> ref_all_slab_topologies_0_2_ax10 = {topology0,
                                                                   topology2};
    EXPECT_EQ(all_slab_topologies_0_2_ax10, ref_all_slab_topologies_0_2_ax10);
    std::vector<topology_type> ref_all_slab_topologies_0_2_ax12 = {topology0,
                                                                   topology2};
    EXPECT_EQ(all_slab_topologies_0_2_ax12, ref_all_slab_topologies_0_2_ax12);
    std::vector<topology_type> ref_all_slab_topologies_0_2_ax20 = {topology0,
                                                                   topology2};
    EXPECT_EQ(all_slab_topologies_0_2_ax20, ref_all_slab_topologies_0_2_ax20);
    std::vector<topology_type> ref_all_slab_topologies_0_2_ax21 = {topology0,
                                                                   topology2};
    EXPECT_EQ(all_slab_topologies_0_2_ax21, ref_all_slab_topologies_0_2_ax21);

    // topology1 (XZ-slab) to topology0 (XY-slab)
    auto all_slab_topologies_1_0_ax01 =
        get_all_slab_topologies(topology1, topology0, axes01);
    auto all_slab_topologies_1_0_ax02 =
        get_all_slab_topologies(topology1, topology0, axes02);
    auto all_slab_topologies_1_0_ax10 =
        get_all_slab_topologies(topology1, topology0, axes10);
    auto all_slab_topologies_1_0_ax12 =
        get_all_slab_topologies(topology1, topology0, axes12);
    auto all_slab_topologies_1_0_ax20 =
        get_all_slab_topologies(topology1, topology0, axes20);
    auto all_slab_topologies_1_0_ax21 =
        get_all_slab_topologies(topology1, topology0, axes21);

    std::vector<topology_type> ref_all_slab_topologies_1_0_ax01 = {topology1,
                                                                   topology0};
    EXPECT_EQ(all_slab_topologies_1_0_ax01, ref_all_slab_topologies_1_0_ax01);
    std::vector<topology_type> ref_all_slab_topologies_1_0_ax02 = {topology1,
                                                                   topology0};
    EXPECT_EQ(all_slab_topologies_1_0_ax02, ref_all_slab_topologies_1_0_ax02);
    std::vector<topology_type> ref_all_slab_topologies_1_0_ax10 = {topology1,
                                                                   topology0};
    EXPECT_EQ(all_slab_topologies_1_0_ax10, ref_all_slab_topologies_1_0_ax10);
    std::vector<topology_type> ref_all_slab_topologies_1_0_ax12 = {topology1,
                                                                   topology0};
    EXPECT_EQ(all_slab_topologies_1_0_ax12, ref_all_slab_topologies_1_0_ax12);
    std::vector<topology_type> ref_all_slab_topologies_1_0_ax20 = {topology1,
                                                                   topology0};
    EXPECT_EQ(all_slab_topologies_1_0_ax20, ref_all_slab_topologies_1_0_ax20);
    std::vector<topology_type> ref_all_slab_topologies_1_0_ax21 = {
        topology1, topology2, topology0};
    EXPECT_EQ(all_slab_topologies_1_0_ax21, ref_all_slab_topologies_1_0_ax21);

    // topology1 (XZ-slab) to topology1 (XZ-slab)
    auto all_slab_topologies_1_1_ax01 =
        get_all_slab_topologies(topology1, topology1, axes01);
    auto all_slab_topologies_1_1_ax02 =
        get_all_slab_topologies(topology1, topology1, axes02);
    auto all_slab_topologies_1_1_ax10 =
        get_all_slab_topologies(topology1, topology1, axes10);
    auto all_slab_topologies_1_1_ax12 =
        get_all_slab_topologies(topology1, topology1, axes12);
    auto all_slab_topologies_1_1_ax20 =
        get_all_slab_topologies(topology1, topology1, axes20);
    auto all_slab_topologies_1_1_ax21 =
        get_all_slab_topologies(topology1, topology1, axes21);

    std::vector<topology_type> ref_all_slab_topologies_1_1_ax01 = {
        topology1, topology0, topology1};
    EXPECT_EQ(all_slab_topologies_1_1_ax01, ref_all_slab_topologies_1_1_ax01);
    std::vector<topology_type> ref_all_slab_topologies_1_1_ax02 = {topology1};
    EXPECT_EQ(all_slab_topologies_1_1_ax02, ref_all_slab_topologies_1_1_ax02);
    std::vector<topology_type> ref_all_slab_topologies_1_1_ax10 = {
        topology1, topology2, topology1};
    EXPECT_EQ(all_slab_topologies_1_1_ax10, ref_all_slab_topologies_1_1_ax10);
    std::vector<topology_type> ref_all_slab_topologies_1_1_ax12 = {
        topology1, topology2, topology1};
    EXPECT_EQ(all_slab_topologies_1_1_ax12, ref_all_slab_topologies_1_1_ax12);
    std::vector<topology_type> ref_all_slab_topologies_1_1_ax20 = {topology1};
    EXPECT_EQ(all_slab_topologies_1_1_ax20, ref_all_slab_topologies_1_1_ax20);
    std::vector<topology_type> ref_all_slab_topologies_1_1_ax21 = {
        topology1, topology2, topology1};
    EXPECT_EQ(all_slab_topologies_1_1_ax21, ref_all_slab_topologies_1_1_ax21);

    // topology1 (XZ-slab) to topology2 (YZ-slab)
    auto all_slab_topologies_1_2_ax01 =
        get_all_slab_topologies(topology1, topology2, axes01);
    auto all_slab_topologies_1_2_ax02 =
        get_all_slab_topologies(topology1, topology2, axes02);
    auto all_slab_topologies_1_2_ax10 =
        get_all_slab_topologies(topology1, topology2, axes10);
    auto all_slab_topologies_1_2_ax12 =
        get_all_slab_topologies(topology1, topology2, axes12);
    auto all_slab_topologies_1_2_ax20 =
        get_all_slab_topologies(topology1, topology2, axes20);
    auto all_slab_topologies_1_2_ax21 =
        get_all_slab_topologies(topology1, topology2, axes21);

    std::vector<topology_type> ref_all_slab_topologies_1_2_ax01 = {
        topology1, topology0, topology2};
    EXPECT_EQ(all_slab_topologies_1_2_ax01, ref_all_slab_topologies_1_2_ax01);
    std::vector<topology_type> ref_all_slab_topologies_1_2_ax02 = {topology1,
                                                                   topology2};
    EXPECT_EQ(all_slab_topologies_1_2_ax02, ref_all_slab_topologies_1_2_ax02);
    std::vector<topology_type> ref_all_slab_topologies_1_2_ax10 = {topology1,
                                                                   topology2};
    EXPECT_EQ(all_slab_topologies_1_2_ax10, ref_all_slab_topologies_1_2_ax10);
    std::vector<topology_type> ref_all_slab_topologies_1_2_ax12 = {topology1,
                                                                   topology2};
    EXPECT_EQ(all_slab_topologies_1_2_ax12, ref_all_slab_topologies_1_2_ax12);
    std::vector<topology_type> ref_all_slab_topologies_1_2_ax20 = {topology1,
                                                                   topology2};
    EXPECT_EQ(all_slab_topologies_1_2_ax20, ref_all_slab_topologies_1_2_ax20);
    std::vector<topology_type> ref_all_slab_topologies_1_2_ax21 = {topology1,
                                                                   topology2};
    EXPECT_EQ(all_slab_topologies_1_2_ax21, ref_all_slab_topologies_1_2_ax21);

    // topology2 (YZ-slab) to topology0 (XY-slab)
    auto all_slab_topologies_2_0_ax01 =
        get_all_slab_topologies(topology2, topology0, axes01);
    auto all_slab_topologies_2_0_ax02 =
        get_all_slab_topologies(topology2, topology0, axes02);
    auto all_slab_topologies_2_0_ax10 =
        get_all_slab_topologies(topology2, topology0, axes10);
    auto all_slab_topologies_2_0_ax12 =
        get_all_slab_topologies(topology2, topology0, axes12);
    auto all_slab_topologies_2_0_ax20 =
        get_all_slab_topologies(topology2, topology0, axes20);
    auto all_slab_topologies_2_0_ax21 =
        get_all_slab_topologies(topology2, topology0, axes21);

    std::vector<topology_type> ref_all_slab_topologies_2_0_ax01 = {topology2,
                                                                   topology0};
    EXPECT_EQ(all_slab_topologies_2_0_ax01, ref_all_slab_topologies_2_0_ax01);
    std::vector<topology_type> ref_all_slab_topologies_2_0_ax02 = {topology2,
                                                                   topology0};
    EXPECT_EQ(all_slab_topologies_2_0_ax02, ref_all_slab_topologies_2_0_ax02);
    std::vector<topology_type> ref_all_slab_topologies_2_0_ax10 = {topology2,
                                                                   topology0};
    EXPECT_EQ(all_slab_topologies_2_0_ax10, ref_all_slab_topologies_2_0_ax10);
    std::vector<topology_type> ref_all_slab_topologies_2_0_ax12 = {topology2,
                                                                   topology0};
    EXPECT_EQ(all_slab_topologies_2_0_ax12, ref_all_slab_topologies_2_0_ax12);
    std::vector<topology_type> ref_all_slab_topologies_2_0_ax20 = {
        topology2, topology1, topology0};
    EXPECT_EQ(all_slab_topologies_2_0_ax20, ref_all_slab_topologies_2_0_ax20);
    std::vector<topology_type> ref_all_slab_topologies_2_0_ax21 = {topology2,
                                                                   topology0};
    EXPECT_EQ(all_slab_topologies_2_0_ax21, ref_all_slab_topologies_2_0_ax21);

    // topology2 (YZ-slab) to topology1 (XZ-slab)
    auto all_slab_topologies_2_1_ax01 =
        get_all_slab_topologies(topology2, topology1, axes01);
    auto all_slab_topologies_2_1_ax02 =
        get_all_slab_topologies(topology2, topology1, axes02);
    auto all_slab_topologies_2_1_ax10 =
        get_all_slab_topologies(topology2, topology1, axes10);
    auto all_slab_topologies_2_1_ax12 =
        get_all_slab_topologies(topology2, topology1, axes12);
    auto all_slab_topologies_2_1_ax20 =
        get_all_slab_topologies(topology2, topology1, axes20);
    auto all_slab_topologies_2_1_ax21 =
        get_all_slab_topologies(topology2, topology1, axes21);

    std::vector<topology_type> ref_all_slab_topologies_2_1_ax01 = {topology2,
                                                                   topology1};
    EXPECT_EQ(all_slab_topologies_2_1_ax01, ref_all_slab_topologies_2_1_ax01);
    std::vector<topology_type> ref_all_slab_topologies_2_1_ax02 = {topology2,
                                                                   topology1};
    EXPECT_EQ(all_slab_topologies_2_1_ax02, ref_all_slab_topologies_2_1_ax02);
    std::vector<topology_type> ref_all_slab_topologies_2_1_ax10 = {
        topology2, topology0, topology1};
    EXPECT_EQ(all_slab_topologies_2_1_ax10, ref_all_slab_topologies_2_1_ax10);
    std::vector<topology_type> ref_all_slab_topologies_2_1_ax12 = {topology2,
                                                                   topology1};
    EXPECT_EQ(all_slab_topologies_2_1_ax12, ref_all_slab_topologies_2_1_ax12);
    std::vector<topology_type> ref_all_slab_topologies_2_1_ax20 = {topology2,
                                                                   topology1};
    EXPECT_EQ(all_slab_topologies_2_1_ax20, ref_all_slab_topologies_2_1_ax20);
    std::vector<topology_type> ref_all_slab_topologies_2_1_ax21 = {topology2,
                                                                   topology1};
    EXPECT_EQ(all_slab_topologies_2_1_ax21, ref_all_slab_topologies_2_1_ax21);

    // topology2 (YZ-slab) to topology2 (YZ-slab)
    auto all_slab_topologies_2_2_ax01 =
        get_all_slab_topologies(topology2, topology2, axes01);
    auto all_slab_topologies_2_2_ax02 =
        get_all_slab_topologies(topology2, topology2, axes02);
    auto all_slab_topologies_2_2_ax10 =
        get_all_slab_topologies(topology2, topology2, axes10);
    auto all_slab_topologies_2_2_ax12 =
        get_all_slab_topologies(topology2, topology2, axes12);
    auto all_slab_topologies_2_2_ax20 =
        get_all_slab_topologies(topology2, topology2, axes20);
    auto all_slab_topologies_2_2_ax21 =
        get_all_slab_topologies(topology2, topology2, axes21);

    std::vector<topology_type> ref_all_slab_topologies_2_2_ax01 = {
        topology2, topology1, topology2};
    EXPECT_EQ(all_slab_topologies_2_2_ax01, ref_all_slab_topologies_2_2_ax01);
    std::vector<topology_type> ref_all_slab_topologies_2_2_ax02 = {
        topology2, topology1, topology2};
    EXPECT_EQ(all_slab_topologies_2_2_ax02, ref_all_slab_topologies_2_2_ax02);
    std::vector<topology_type> ref_all_slab_topologies_2_2_ax10 = {
        topology2, topology0, topology2};
    EXPECT_EQ(all_slab_topologies_2_2_ax10, ref_all_slab_topologies_2_2_ax10);
    std::vector<topology_type> ref_all_slab_topologies_2_2_ax12 = {topology2};
    EXPECT_EQ(all_slab_topologies_2_2_ax12, ref_all_slab_topologies_2_2_ax12);
    std::vector<topology_type> ref_all_slab_topologies_2_2_ax20 = {
        topology2, topology1, topology2};
    EXPECT_EQ(all_slab_topologies_2_2_ax20, ref_all_slab_topologies_2_2_ax20);
    std::vector<topology_type> ref_all_slab_topologies_2_2_ax21 = {topology2};
    EXPECT_EQ(all_slab_topologies_2_2_ax21, ref_all_slab_topologies_2_2_ax21);
  }
}

void test_get_all_slab_topologies3D_3DView(std::size_t nprocs) {
  using topology_type     = std::array<std::size_t, 3>;
  topology_type topology0 = {1, 1, nprocs};
  topology_type topology1 = {1, nprocs, 1};
  topology_type topology2 = {nprocs, 1, 1};

  using axes_type   = std::array<std::size_t, 3>;
  axes_type axes012 = {0, 1, 2};
  axes_type axes021 = {0, 2, 1};
  axes_type axes102 = {1, 0, 2};
  axes_type axes120 = {1, 2, 0};
  axes_type axes201 = {2, 0, 1};
  axes_type axes210 = {2, 1, 0};

  std::vector<axes_type> all_axes = {axes012, axes021, axes102,
                                     axes120, axes201, axes210};

  if (nprocs == 1) {
    for (const auto& axes : all_axes) {
      // Failure tests because these are shared topologies
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology0, topology0, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology0, topology1, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology0, topology2, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology1, topology0, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology1, topology1, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology1, topology2, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology2, topology0, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology2, topology1, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology2, topology2, axes);
          },
          std::runtime_error);
    }
  } else {
    // topology0 (XY-slab) to topology0 (XY-slab)
    auto all_slab_topologies_0_0_ax012 =
        get_all_slab_topologies(topology0, topology0, axes012);
    auto all_slab_topologies_0_0_ax021 =
        get_all_slab_topologies(topology0, topology0, axes021);
    auto all_slab_topologies_0_0_ax102 =
        get_all_slab_topologies(topology0, topology0, axes102);
    auto all_slab_topologies_0_0_ax120 =
        get_all_slab_topologies(topology0, topology0, axes120);
    auto all_slab_topologies_0_0_ax201 =
        get_all_slab_topologies(topology0, topology0, axes201);
    auto all_slab_topologies_0_0_ax210 =
        get_all_slab_topologies(topology0, topology0, axes210);

    std::vector<topology_type> ref_all_slab_topologies_0_0_ax012 = {
        topology0, topology2, topology0};
    EXPECT_EQ(all_slab_topologies_0_0_ax012, ref_all_slab_topologies_0_0_ax012);
    std::vector<topology_type> ref_all_slab_topologies_0_0_ax021 = {
        topology0, topology1, topology0};
    EXPECT_EQ(all_slab_topologies_0_0_ax021, ref_all_slab_topologies_0_0_ax021);
    std::vector<topology_type> ref_all_slab_topologies_0_0_ax102 = {
        topology0, topology1, topology0};
    EXPECT_EQ(all_slab_topologies_0_0_ax102, ref_all_slab_topologies_0_0_ax102);
    std::vector<topology_type> ref_all_slab_topologies_0_0_ax120 = {
        topology0, topology2, topology0};
    EXPECT_EQ(all_slab_topologies_0_0_ax120, ref_all_slab_topologies_0_0_ax120);
    std::vector<topology_type> ref_all_slab_topologies_0_0_ax201 = {
        topology0, topology1, topology0};
    EXPECT_EQ(all_slab_topologies_0_0_ax201, ref_all_slab_topologies_0_0_ax201);
    std::vector<topology_type> ref_all_slab_topologies_0_0_ax210 = {
        topology0, topology2, topology0};
    EXPECT_EQ(all_slab_topologies_0_0_ax210, ref_all_slab_topologies_0_0_ax210);

    // topology0 (XY-slab) to topology1 (XZ-slab)
    auto all_slab_topologies_0_1_ax012 =
        get_all_slab_topologies(topology0, topology1, axes012);
    auto all_slab_topologies_0_1_ax021 =
        get_all_slab_topologies(topology0, topology1, axes021);
    auto all_slab_topologies_0_1_ax102 =
        get_all_slab_topologies(topology0, topology1, axes102);
    auto all_slab_topologies_0_1_ax120 =
        get_all_slab_topologies(topology0, topology1, axes120);
    auto all_slab_topologies_0_1_ax201 =
        get_all_slab_topologies(topology0, topology1, axes201);
    auto all_slab_topologies_0_1_ax210 =
        get_all_slab_topologies(topology0, topology1, axes210);

    std::vector<topology_type> ref_all_slab_topologies_0_1_ax012 = {
        topology0, topology2, topology1};
    EXPECT_EQ(all_slab_topologies_0_1_ax012, ref_all_slab_topologies_0_1_ax012);
    std::vector<topology_type> ref_all_slab_topologies_0_1_ax021 = {topology0,
                                                                    topology1};
    EXPECT_EQ(all_slab_topologies_0_1_ax021, ref_all_slab_topologies_0_1_ax021);
    std::vector<topology_type> ref_all_slab_topologies_0_1_ax102 = {
        topology0, topology1, topology2, topology1};
    EXPECT_EQ(all_slab_topologies_0_1_ax102, ref_all_slab_topologies_0_1_ax102);
    std::vector<topology_type> ref_all_slab_topologies_0_1_ax120 = {
        topology0, topology2, topology1};
    EXPECT_EQ(all_slab_topologies_0_1_ax120, ref_all_slab_topologies_0_1_ax120);
    std::vector<topology_type> ref_all_slab_topologies_0_1_ax201 = {topology0,
                                                                    topology1};
    EXPECT_EQ(all_slab_topologies_0_1_ax201, ref_all_slab_topologies_0_1_ax201);
    std::vector<topology_type> ref_all_slab_topologies_0_1_ax210 = {topology0,
                                                                    topology1};
    EXPECT_EQ(all_slab_topologies_0_1_ax210, ref_all_slab_topologies_0_1_ax210);

    // topology0 (XY-slab) to topology2 (YZ-slab)
    auto all_slab_topologies_0_2_ax012 =
        get_all_slab_topologies(topology0, topology2, axes012);
    auto all_slab_topologies_0_2_ax021 =
        get_all_slab_topologies(topology0, topology2, axes021);
    auto all_slab_topologies_0_2_ax102 =
        get_all_slab_topologies(topology0, topology2, axes102);
    auto all_slab_topologies_0_2_ax120 =
        get_all_slab_topologies(topology0, topology2, axes120);
    auto all_slab_topologies_0_2_ax201 =
        get_all_slab_topologies(topology0, topology2, axes201);
    auto all_slab_topologies_0_2_ax210 =
        get_all_slab_topologies(topology0, topology2, axes210);

    std::vector<topology_type> ref_all_slab_topologies_0_2_ax012 = {
        topology0, topology2, topology1, topology2};
    EXPECT_EQ(all_slab_topologies_0_2_ax012, ref_all_slab_topologies_0_2_ax012);
    std::vector<topology_type> ref_all_slab_topologies_0_2_ax021 = {
        topology0, topology1, topology2};
    EXPECT_EQ(all_slab_topologies_0_2_ax021, ref_all_slab_topologies_0_2_ax021);
    std::vector<topology_type> ref_all_slab_topologies_0_2_ax102 = {
        topology0, topology1, topology2};
    EXPECT_EQ(all_slab_topologies_0_2_ax102, ref_all_slab_topologies_0_2_ax102);
    std::vector<topology_type> ref_all_slab_topologies_0_2_ax120 = {topology0,
                                                                    topology2};
    EXPECT_EQ(all_slab_topologies_0_2_ax120, ref_all_slab_topologies_0_2_ax120);
    std::vector<topology_type> ref_all_slab_topologies_0_2_ax201 = {topology0,
                                                                    topology2};
    EXPECT_EQ(all_slab_topologies_0_2_ax201, ref_all_slab_topologies_0_2_ax201);
    std::vector<topology_type> ref_all_slab_topologies_0_2_ax210 = {topology0,
                                                                    topology2};
    EXPECT_EQ(all_slab_topologies_0_2_ax210, ref_all_slab_topologies_0_2_ax210);

    // topology1 (XZ-slab) to topology0 (XY-slab)
    auto all_slab_topologies_1_0_ax012 =
        get_all_slab_topologies(topology1, topology0, axes012);
    auto all_slab_topologies_1_0_ax021 =
        get_all_slab_topologies(topology1, topology0, axes021);
    auto all_slab_topologies_1_0_ax102 =
        get_all_slab_topologies(topology1, topology0, axes102);
    auto all_slab_topologies_1_0_ax120 =
        get_all_slab_topologies(topology1, topology0, axes120);
    auto all_slab_topologies_1_0_ax201 =
        get_all_slab_topologies(topology1, topology0, axes201);
    auto all_slab_topologies_1_0_ax210 =
        get_all_slab_topologies(topology1, topology0, axes210);

    std::vector<topology_type> ref_all_slab_topologies_1_0_ax012 = {topology1,
                                                                    topology0};
    EXPECT_EQ(all_slab_topologies_1_0_ax012, ref_all_slab_topologies_1_0_ax012);
    std::vector<topology_type> ref_all_slab_topologies_1_0_ax021 = {
        topology1, topology2, topology0};
    EXPECT_EQ(all_slab_topologies_1_0_ax021, ref_all_slab_topologies_1_0_ax021);
    std::vector<topology_type> ref_all_slab_topologies_1_0_ax102 = {topology1,
                                                                    topology0};
    EXPECT_EQ(all_slab_topologies_1_0_ax102, ref_all_slab_topologies_1_0_ax102);
    std::vector<topology_type> ref_all_slab_topologies_1_0_ax120 = {topology1,
                                                                    topology0};
    EXPECT_EQ(all_slab_topologies_1_0_ax120, ref_all_slab_topologies_1_0_ax120);
    std::vector<topology_type> ref_all_slab_topologies_1_0_ax201 = {
        topology1, topology0, topology2, topology0};
    EXPECT_EQ(all_slab_topologies_1_0_ax201, ref_all_slab_topologies_1_0_ax201);
    std::vector<topology_type> ref_all_slab_topologies_1_0_ax210 = {
        topology1, topology2, topology0};
    EXPECT_EQ(all_slab_topologies_1_0_ax210, ref_all_slab_topologies_1_0_ax210);

    // topology1 (XZ-slab) to topology1 (XZ-slab)
    auto all_slab_topologies_1_1_ax012 =
        get_all_slab_topologies(topology1, topology1, axes012);
    auto all_slab_topologies_1_1_ax021 =
        get_all_slab_topologies(topology1, topology1, axes021);
    auto all_slab_topologies_1_1_ax102 =
        get_all_slab_topologies(topology1, topology1, axes102);
    auto all_slab_topologies_1_1_ax120 =
        get_all_slab_topologies(topology1, topology1, axes120);
    auto all_slab_topologies_1_1_ax201 =
        get_all_slab_topologies(topology1, topology1, axes201);
    auto all_slab_topologies_1_1_ax210 =
        get_all_slab_topologies(topology1, topology1, axes210);

    std::vector<topology_type> ref_all_slab_topologies_1_1_ax012 = {
        topology1, topology0, topology1};
    EXPECT_EQ(all_slab_topologies_1_1_ax012, ref_all_slab_topologies_1_1_ax012);
    std::vector<topology_type> ref_all_slab_topologies_1_1_ax021 = {
        topology1, topology2, topology1};
    EXPECT_EQ(all_slab_topologies_1_1_ax021, ref_all_slab_topologies_1_1_ax021);
    std::vector<topology_type> ref_all_slab_topologies_1_1_ax102 = {
        topology1, topology0, topology1};
    EXPECT_EQ(all_slab_topologies_1_1_ax102, ref_all_slab_topologies_1_1_ax102);
    std::vector<topology_type> ref_all_slab_topologies_1_1_ax120 = {
        topology1, topology2, topology1};
    EXPECT_EQ(all_slab_topologies_1_1_ax120, ref_all_slab_topologies_1_1_ax120);
    std::vector<topology_type> ref_all_slab_topologies_1_1_ax201 = {
        topology1, topology0, topology1};
    EXPECT_EQ(all_slab_topologies_1_1_ax201, ref_all_slab_topologies_1_1_ax201);
    std::vector<topology_type> ref_all_slab_topologies_1_1_ax210 = {
        topology1, topology2, topology1};
    EXPECT_EQ(all_slab_topologies_1_1_ax210, ref_all_slab_topologies_1_1_ax210);

    // topology1 (XZ-slab) to topology2 (YZ-slab)
    auto all_slab_topologies_1_2_ax012 =
        get_all_slab_topologies(topology1, topology2, axes012);
    auto all_slab_topologies_1_2_ax021 =
        get_all_slab_topologies(topology1, topology2, axes021);
    auto all_slab_topologies_1_2_ax102 =
        get_all_slab_topologies(topology1, topology2, axes102);
    auto all_slab_topologies_1_2_ax120 =
        get_all_slab_topologies(topology1, topology2, axes120);
    auto all_slab_topologies_1_2_ax201 =
        get_all_slab_topologies(topology1, topology2, axes201);
    auto all_slab_topologies_1_2_ax210 =
        get_all_slab_topologies(topology1, topology2, axes210);

    std::vector<topology_type> ref_all_slab_topologies_1_2_ax012 = {
        topology1, topology0, topology2};
    EXPECT_EQ(all_slab_topologies_1_2_ax012, ref_all_slab_topologies_1_2_ax012);
    std::vector<topology_type> ref_all_slab_topologies_1_2_ax021 = {
        topology1, topology2, topology1, topology2};
    EXPECT_EQ(all_slab_topologies_1_2_ax021, ref_all_slab_topologies_1_2_ax021);
    std::vector<topology_type> ref_all_slab_topologies_1_2_ax102 = {topology1,
                                                                    topology2};
    EXPECT_EQ(all_slab_topologies_1_2_ax102, ref_all_slab_topologies_1_2_ax102);
    std::vector<topology_type> ref_all_slab_topologies_1_2_ax120 = {topology1,
                                                                    topology2};
    EXPECT_EQ(all_slab_topologies_1_2_ax120, ref_all_slab_topologies_1_2_ax120);
    std::vector<topology_type> ref_all_slab_topologies_1_2_ax201 = {
        topology1, topology0, topology2};
    EXPECT_EQ(all_slab_topologies_1_2_ax201, ref_all_slab_topologies_1_2_ax201);
    std::vector<topology_type> ref_all_slab_topologies_1_2_ax210 = {topology1,
                                                                    topology2};
    EXPECT_EQ(all_slab_topologies_1_2_ax210, ref_all_slab_topologies_1_2_ax210);

    // topology2 (YZ-slab) to topology0 (XY-slab)
    auto all_slab_topologies_2_0_ax012 =
        get_all_slab_topologies(topology2, topology0, axes012);
    auto all_slab_topologies_2_0_ax021 =
        get_all_slab_topologies(topology2, topology0, axes021);
    auto all_slab_topologies_2_0_ax102 =
        get_all_slab_topologies(topology2, topology0, axes102);
    auto all_slab_topologies_2_0_ax120 =
        get_all_slab_topologies(topology2, topology0, axes120);
    auto all_slab_topologies_2_0_ax201 =
        get_all_slab_topologies(topology2, topology0, axes201);
    auto all_slab_topologies_2_0_ax210 =
        get_all_slab_topologies(topology2, topology0, axes210);

    std::vector<topology_type> ref_all_slab_topologies_2_0_ax012 = {topology2,
                                                                    topology0};
    EXPECT_EQ(all_slab_topologies_2_0_ax012, ref_all_slab_topologies_2_0_ax012);
    std::vector<topology_type> ref_all_slab_topologies_2_0_ax021 = {topology2,
                                                                    topology0};
    EXPECT_EQ(all_slab_topologies_2_0_ax021, ref_all_slab_topologies_2_0_ax021);
    std::vector<topology_type> ref_all_slab_topologies_2_0_ax102 = {topology2,
                                                                    topology0};
    EXPECT_EQ(all_slab_topologies_2_0_ax102, ref_all_slab_topologies_2_0_ax102);
    std::vector<topology_type> ref_all_slab_topologies_2_0_ax120 = {
        topology2, topology1, topology0};
    EXPECT_EQ(all_slab_topologies_2_0_ax120, ref_all_slab_topologies_2_0_ax120);
    std::vector<topology_type> ref_all_slab_topologies_2_0_ax201 = {
        topology2, topology1, topology0};
    EXPECT_EQ(all_slab_topologies_2_0_ax201, ref_all_slab_topologies_2_0_ax201);
    std::vector<topology_type> ref_all_slab_topologies_2_0_ax210 = {
        topology2, topology0, topology2, topology0};
    EXPECT_EQ(all_slab_topologies_2_0_ax210, ref_all_slab_topologies_2_0_ax210);

    // topology2 (YZ-slab) to topology1 (XZ-slab)
    auto all_slab_topologies_2_1_ax012 =
        get_all_slab_topologies(topology2, topology1, axes012);
    auto all_slab_topologies_2_1_ax021 =
        get_all_slab_topologies(topology2, topology1, axes021);
    auto all_slab_topologies_2_1_ax102 =
        get_all_slab_topologies(topology2, topology1, axes102);
    auto all_slab_topologies_2_1_ax120 =
        get_all_slab_topologies(topology2, topology1, axes120);
    auto all_slab_topologies_2_1_ax201 =
        get_all_slab_topologies(topology2, topology1, axes201);
    auto all_slab_topologies_2_1_ax210 =
        get_all_slab_topologies(topology2, topology1, axes210);

    std::vector<topology_type> ref_all_slab_topologies_2_1_ax012 = {topology2,
                                                                    topology1};
    EXPECT_EQ(all_slab_topologies_2_1_ax012, ref_all_slab_topologies_2_1_ax012);
    std::vector<topology_type> ref_all_slab_topologies_2_1_ax021 = {topology2,
                                                                    topology1};
    EXPECT_EQ(all_slab_topologies_2_1_ax021, ref_all_slab_topologies_2_1_ax021);
    std::vector<topology_type> ref_all_slab_topologies_2_1_ax102 = {
        topology2, topology0, topology1};
    EXPECT_EQ(all_slab_topologies_2_1_ax102, ref_all_slab_topologies_2_1_ax102);
    std::vector<topology_type> ref_all_slab_topologies_2_1_ax120 = {
        topology2, topology1, topology2, topology1};
    EXPECT_EQ(all_slab_topologies_2_1_ax120, ref_all_slab_topologies_2_1_ax120);
    std::vector<topology_type> ref_all_slab_topologies_2_1_ax201 = {topology2,
                                                                    topology1};
    EXPECT_EQ(all_slab_topologies_2_1_ax201, ref_all_slab_topologies_2_1_ax201);
    std::vector<topology_type> ref_all_slab_topologies_2_1_ax210 = {
        topology2, topology0, topology1};
    EXPECT_EQ(all_slab_topologies_2_1_ax210, ref_all_slab_topologies_2_1_ax210);

    // topology2 (YZ-slab) to topology2 (YZ-slab)
    auto all_slab_topologies_2_2_ax012 =
        get_all_slab_topologies(topology2, topology2, axes012);
    auto all_slab_topologies_2_2_ax021 =
        get_all_slab_topologies(topology2, topology2, axes021);
    auto all_slab_topologies_2_2_ax102 =
        get_all_slab_topologies(topology2, topology2, axes102);
    auto all_slab_topologies_2_2_ax120 =
        get_all_slab_topologies(topology2, topology2, axes120);
    auto all_slab_topologies_2_2_ax201 =
        get_all_slab_topologies(topology2, topology2, axes201);
    auto all_slab_topologies_2_2_ax210 =
        get_all_slab_topologies(topology2, topology2, axes210);

    std::vector<topology_type> ref_all_slab_topologies_2_2_ax012 = {
        topology2, topology0, topology2};
    EXPECT_EQ(all_slab_topologies_2_2_ax012, ref_all_slab_topologies_2_2_ax012);
    std::vector<topology_type> ref_all_slab_topologies_2_2_ax021 = {
        topology2, topology1, topology2};
    EXPECT_EQ(all_slab_topologies_2_2_ax021, ref_all_slab_topologies_2_2_ax021);
    std::vector<topology_type> ref_all_slab_topologies_2_2_ax102 = {
        topology2, topology0, topology2};
    EXPECT_EQ(all_slab_topologies_2_2_ax102, ref_all_slab_topologies_2_2_ax102);
    std::vector<topology_type> ref_all_slab_topologies_2_2_ax120 = {
        topology2, topology1, topology2};
    EXPECT_EQ(all_slab_topologies_2_2_ax120, ref_all_slab_topologies_2_2_ax120);
    std::vector<topology_type> ref_all_slab_topologies_2_2_ax201 = {
        topology2, topology1, topology2};
    EXPECT_EQ(all_slab_topologies_2_2_ax201, ref_all_slab_topologies_2_2_ax201);
    std::vector<topology_type> ref_all_slab_topologies_2_2_ax210 = {
        topology2, topology0, topology2};
    EXPECT_EQ(all_slab_topologies_2_2_ax210, ref_all_slab_topologies_2_2_ax210);
  }
}

void test_get_all_slab_topologies3D_4DView(std::size_t nprocs) {
  using topology_type     = std::array<std::size_t, 4>;
  topology_type topology0 = {1, 1, 1, nprocs}, topology1 = {1, 1, nprocs, 1},
                topology2 = {1, nprocs, 1, 1}, topology3 = {nprocs, 1, 1, 1};

  using axes_type   = std::array<std::size_t, 3>;
  axes_type axes012 = {0, 1, 2}, axes021 = {0, 2, 1}, axes102 = {1, 0, 2},
            axes120 = {1, 2, 0}, axes201 = {2, 0, 1}, axes210 = {2, 1, 0},
            axes123 = {1, 2, 3}, axes132 = {1, 3, 2};

  std::vector<axes_type> all_axes = {axes012, axes021, axes102, axes120,
                                     axes201, axes210, axes123, axes132};

  if (nprocs == 1) {
    for (const auto& axes : all_axes) {
      // Failure tests because these are shared topologies
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology0, topology0, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology0, topology1, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology0, topology2, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology1, topology0, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology1, topology1, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology1, topology2, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology2, topology0, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology2, topology1, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto all_slab_topologies =
                get_all_slab_topologies(topology2, topology2, axes);
          },
          std::runtime_error);
    }
  } else {
    // topology0 (XYZ-slab) to topology0 (XYZ-slab)
    auto all_slab_topologies_0_0_ax012 =
        get_all_slab_topologies(topology0, topology0, axes012);
    auto all_slab_topologies_0_0_ax021 =
        get_all_slab_topologies(topology0, topology0, axes021);
    auto all_slab_topologies_0_0_ax102 =
        get_all_slab_topologies(topology0, topology0, axes102);
    auto all_slab_topologies_0_0_ax120 =
        get_all_slab_topologies(topology0, topology0, axes120);
    auto all_slab_topologies_0_0_ax201 =
        get_all_slab_topologies(topology0, topology0, axes201);
    auto all_slab_topologies_0_0_ax210 =
        get_all_slab_topologies(topology0, topology0, axes210);
    auto all_slab_topologies_0_0_ax123 =
        get_all_slab_topologies(topology0, topology0, axes123);
    auto all_slab_topologies_0_0_ax132 =
        get_all_slab_topologies(topology0, topology0, axes132);

    std::vector<topology_type> ref_all_slab_topologies_0_0_ax012 = {topology0};
    EXPECT_EQ(all_slab_topologies_0_0_ax012, ref_all_slab_topologies_0_0_ax012);
    std::vector<topology_type> ref_all_slab_topologies_0_0_ax021 = {topology0};
    EXPECT_EQ(all_slab_topologies_0_0_ax021, ref_all_slab_topologies_0_0_ax021);
    std::vector<topology_type> ref_all_slab_topologies_0_0_ax102 = {topology0};
    EXPECT_EQ(all_slab_topologies_0_0_ax102, ref_all_slab_topologies_0_0_ax102);
    std::vector<topology_type> ref_all_slab_topologies_0_0_ax120 = {topology0};
    EXPECT_EQ(all_slab_topologies_0_0_ax120, ref_all_slab_topologies_0_0_ax120);
    std::vector<topology_type> ref_all_slab_topologies_0_0_ax201 = {topology0};
    EXPECT_EQ(all_slab_topologies_0_0_ax201, ref_all_slab_topologies_0_0_ax201);
    std::vector<topology_type> ref_all_slab_topologies_0_0_ax210 = {topology0};
    EXPECT_EQ(all_slab_topologies_0_0_ax210, ref_all_slab_topologies_0_0_ax210);
    std::vector<topology_type> ref_all_slab_topologies_0_0_ax123 = {
        topology0, topology2, topology0};
    EXPECT_EQ(all_slab_topologies_0_0_ax123, ref_all_slab_topologies_0_0_ax123);
    std::vector<topology_type> ref_all_slab_topologies_0_0_ax132 = {
        topology0, topology1, topology0};
    EXPECT_EQ(all_slab_topologies_0_0_ax132, ref_all_slab_topologies_0_0_ax132);

    // topology0 (XYZ-slab) to topology1 (XZW-slab)
    auto all_slab_topologies_0_1_ax012 =
        get_all_slab_topologies(topology0, topology1, axes012);
    auto all_slab_topologies_0_1_ax021 =
        get_all_slab_topologies(topology0, topology1, axes021);
    auto all_slab_topologies_0_1_ax102 =
        get_all_slab_topologies(topology0, topology1, axes102);
    auto all_slab_topologies_0_1_ax120 =
        get_all_slab_topologies(topology0, topology1, axes120);
    auto all_slab_topologies_0_1_ax201 =
        get_all_slab_topologies(topology0, topology1, axes201);
    auto all_slab_topologies_0_1_ax210 =
        get_all_slab_topologies(topology0, topology1, axes210);

    std::vector<topology_type> ref_all_slab_topologies_0_1_ax012 = {topology0,
                                                                    topology1};
    EXPECT_EQ(all_slab_topologies_0_1_ax012, ref_all_slab_topologies_0_1_ax012);
    std::vector<topology_type> ref_all_slab_topologies_0_1_ax021 = {topology0,
                                                                    topology1};
    EXPECT_EQ(all_slab_topologies_0_1_ax021, ref_all_slab_topologies_0_1_ax021);
    std::vector<topology_type> ref_all_slab_topologies_0_1_ax102 = {topology0,
                                                                    topology1};
    EXPECT_EQ(all_slab_topologies_0_1_ax102, ref_all_slab_topologies_0_1_ax102);
    std::vector<topology_type> ref_all_slab_topologies_0_1_ax120 = {topology0,
                                                                    topology1};
    EXPECT_EQ(all_slab_topologies_0_1_ax120, ref_all_slab_topologies_0_1_ax120);
    std::vector<topology_type> ref_all_slab_topologies_0_1_ax201 = {topology0,
                                                                    topology1};
    EXPECT_EQ(all_slab_topologies_0_1_ax201, ref_all_slab_topologies_0_1_ax201);
    std::vector<topology_type> ref_all_slab_topologies_0_1_ax210 = {topology0,
                                                                    topology1};
    EXPECT_EQ(all_slab_topologies_0_1_ax210, ref_all_slab_topologies_0_1_ax210);

    // topology0 (XYZ-slab) to topology2 (XZW-slab)
    auto all_slab_topologies_0_2_ax012 =
        get_all_slab_topologies(topology0, topology2, axes012);
    auto all_slab_topologies_0_2_ax021 =
        get_all_slab_topologies(topology0, topology2, axes021);
    auto all_slab_topologies_0_2_ax102 =
        get_all_slab_topologies(topology0, topology2, axes102);
    auto all_slab_topologies_0_2_ax120 =
        get_all_slab_topologies(topology0, topology2, axes120);
    auto all_slab_topologies_0_2_ax201 =
        get_all_slab_topologies(topology0, topology2, axes201);
    auto all_slab_topologies_0_2_ax210 =
        get_all_slab_topologies(topology0, topology2, axes210);

    std::vector<topology_type> ref_all_slab_topologies_0_2_ax012 = {topology0,
                                                                    topology2};
    EXPECT_EQ(all_slab_topologies_0_2_ax012, ref_all_slab_topologies_0_2_ax012);
    std::vector<topology_type> ref_all_slab_topologies_0_2_ax021 = {topology0,
                                                                    topology2};
    EXPECT_EQ(all_slab_topologies_0_2_ax021, ref_all_slab_topologies_0_2_ax021);
    std::vector<topology_type> ref_all_slab_topologies_0_2_ax102 = {topology0,
                                                                    topology2};
    EXPECT_EQ(all_slab_topologies_0_2_ax102, ref_all_slab_topologies_0_2_ax102);
    std::vector<topology_type> ref_all_slab_topologies_0_2_ax120 = {topology0,
                                                                    topology2};
    EXPECT_EQ(all_slab_topologies_0_2_ax120, ref_all_slab_topologies_0_2_ax120);
    std::vector<topology_type> ref_all_slab_topologies_0_2_ax201 = {topology0,
                                                                    topology2};
    EXPECT_EQ(all_slab_topologies_0_2_ax201, ref_all_slab_topologies_0_2_ax201);
    std::vector<topology_type> ref_all_slab_topologies_0_2_ax210 = {topology0,
                                                                    topology2};
    EXPECT_EQ(all_slab_topologies_0_2_ax210, ref_all_slab_topologies_0_2_ax210);

    // topology0 (XYZ-slab) to topology3 (YZW-slab)
    auto all_slab_topologies_0_3_ax012 =
        get_all_slab_topologies(topology0, topology3, axes012);
    auto all_slab_topologies_0_3_ax021 =
        get_all_slab_topologies(topology0, topology3, axes021);
    auto all_slab_topologies_0_3_ax102 =
        get_all_slab_topologies(topology0, topology3, axes102);
    auto all_slab_topologies_0_3_ax120 =
        get_all_slab_topologies(topology0, topology3, axes120);
    auto all_slab_topologies_0_3_ax201 =
        get_all_slab_topologies(topology0, topology3, axes201);
    auto all_slab_topologies_0_3_ax210 =
        get_all_slab_topologies(topology0, topology3, axes210);
    auto all_slab_topologies_0_3_ax123 =
        get_all_slab_topologies(topology0, topology3, axes123);
    auto all_slab_topologies_0_3_ax132 =
        get_all_slab_topologies(topology0, topology3, axes132);

    std::vector<topology_type> ref_all_slab_topologies_0_3_ax012 = {topology0,
                                                                    topology3};
    EXPECT_EQ(all_slab_topologies_0_3_ax012, ref_all_slab_topologies_0_3_ax012);
    std::vector<topology_type> ref_all_slab_topologies_0_3_ax021 = {topology0,
                                                                    topology3};
    EXPECT_EQ(all_slab_topologies_0_3_ax021, ref_all_slab_topologies_0_3_ax021);
    std::vector<topology_type> ref_all_slab_topologies_0_3_ax102 = {topology0,
                                                                    topology3};
    EXPECT_EQ(all_slab_topologies_0_3_ax102, ref_all_slab_topologies_0_3_ax102);
    std::vector<topology_type> ref_all_slab_topologies_0_3_ax120 = {topology0,
                                                                    topology3};
    EXPECT_EQ(all_slab_topologies_0_3_ax120, ref_all_slab_topologies_0_3_ax120);
    std::vector<topology_type> ref_all_slab_topologies_0_3_ax201 = {topology0,
                                                                    topology3};
    EXPECT_EQ(all_slab_topologies_0_3_ax201, ref_all_slab_topologies_0_3_ax201);
    std::vector<topology_type> ref_all_slab_topologies_0_3_ax210 = {topology0,
                                                                    topology3};
    EXPECT_EQ(all_slab_topologies_0_3_ax210, ref_all_slab_topologies_0_3_ax210);
    std::vector<topology_type> ref_all_slab_topologies_0_3_ax123 = {topology0,
                                                                    topology3};
    EXPECT_EQ(all_slab_topologies_0_3_ax123, ref_all_slab_topologies_0_3_ax123);
    std::vector<topology_type> ref_all_slab_topologies_0_3_ax132 = {topology0,
                                                                    topology3};
    EXPECT_EQ(all_slab_topologies_0_3_ax132, ref_all_slab_topologies_0_3_ax132);

    // topology1 (XZ-slab) to topology0 (XY-slab)
    auto all_slab_topologies_1_0_ax012 =
        get_all_slab_topologies(topology1, topology0, axes012);
    auto all_slab_topologies_1_0_ax021 =
        get_all_slab_topologies(topology1, topology0, axes021);
    auto all_slab_topologies_1_0_ax102 =
        get_all_slab_topologies(topology1, topology0, axes102);
    auto all_slab_topologies_1_0_ax120 =
        get_all_slab_topologies(topology1, topology0, axes120);
    auto all_slab_topologies_1_0_ax201 =
        get_all_slab_topologies(topology1, topology0, axes201);
    auto all_slab_topologies_1_0_ax210 =
        get_all_slab_topologies(topology1, topology0, axes210);

    std::vector<topology_type> ref_all_slab_topologies_1_0_ax012 = {topology1,
                                                                    topology0};
    EXPECT_EQ(all_slab_topologies_1_0_ax012, ref_all_slab_topologies_1_0_ax012);
    std::vector<topology_type> ref_all_slab_topologies_1_0_ax021 = {topology1,
                                                                    topology0};
    EXPECT_EQ(all_slab_topologies_1_0_ax021, ref_all_slab_topologies_1_0_ax021);
    std::vector<topology_type> ref_all_slab_topologies_1_0_ax102 = {topology1,
                                                                    topology0};
    EXPECT_EQ(all_slab_topologies_1_0_ax102, ref_all_slab_topologies_1_0_ax102);
    std::vector<topology_type> ref_all_slab_topologies_1_0_ax120 = {topology1,
                                                                    topology0};
    EXPECT_EQ(all_slab_topologies_1_0_ax120, ref_all_slab_topologies_1_0_ax120);
    std::vector<topology_type> ref_all_slab_topologies_1_0_ax201 = {topology1,
                                                                    topology0};
    EXPECT_EQ(all_slab_topologies_1_0_ax201, ref_all_slab_topologies_1_0_ax201);
    std::vector<topology_type> ref_all_slab_topologies_1_0_ax210 = {topology1,
                                                                    topology0};
    EXPECT_EQ(all_slab_topologies_1_0_ax210, ref_all_slab_topologies_1_0_ax210);

    // topology1 (XYW-slab) to topology1 (XZW-slab)
    auto all_slab_topologies_1_1_ax012 =
        get_all_slab_topologies(topology1, topology1, axes012);
    auto all_slab_topologies_1_1_ax021 =
        get_all_slab_topologies(topology1, topology1, axes021);
    auto all_slab_topologies_1_1_ax102 =
        get_all_slab_topologies(topology1, topology1, axes102);
    auto all_slab_topologies_1_1_ax120 =
        get_all_slab_topologies(topology1, topology1, axes120);
    auto all_slab_topologies_1_1_ax201 =
        get_all_slab_topologies(topology1, topology1, axes201);
    auto all_slab_topologies_1_1_ax210 =
        get_all_slab_topologies(topology1, topology1, axes210);
    auto all_slab_topologies_1_1_ax123 =
        get_all_slab_topologies(topology1, topology1, axes123);
    auto all_slab_topologies_1_1_ax132 =
        get_all_slab_topologies(topology1, topology1, axes132);

    std::vector<topology_type> ref_all_slab_topologies_1_1_ax012 = {
        topology1, topology3, topology1};
    EXPECT_EQ(all_slab_topologies_1_1_ax012, ref_all_slab_topologies_1_1_ax012);
    std::vector<topology_type> ref_all_slab_topologies_1_1_ax021 = {
        topology1, topology2, topology1};
    EXPECT_EQ(all_slab_topologies_1_1_ax021, ref_all_slab_topologies_1_1_ax021);
    std::vector<topology_type> ref_all_slab_topologies_1_1_ax102 = {
        topology1, topology2, topology1};
    EXPECT_EQ(all_slab_topologies_1_1_ax102, ref_all_slab_topologies_1_1_ax102);
    std::vector<topology_type> ref_all_slab_topologies_1_1_ax120 = {
        topology1, topology3, topology1};
    EXPECT_EQ(all_slab_topologies_1_1_ax120, ref_all_slab_topologies_1_1_ax120);
    std::vector<topology_type> ref_all_slab_topologies_1_1_ax201 = {
        topology1, topology3, topology1};
    EXPECT_EQ(all_slab_topologies_1_1_ax201, ref_all_slab_topologies_1_1_ax201);
    std::vector<topology_type> ref_all_slab_topologies_1_1_ax210 = {
        topology1, topology3, topology1};
    EXPECT_EQ(all_slab_topologies_1_1_ax210, ref_all_slab_topologies_1_1_ax210);
    std::vector<topology_type> ref_all_slab_topologies_1_1_ax123 = {
        topology1, topology0, topology1};
    EXPECT_EQ(all_slab_topologies_1_1_ax123, ref_all_slab_topologies_1_1_ax123);
    std::vector<topology_type> ref_all_slab_topologies_1_1_ax132 = {
        topology1, topology2, topology1};
    EXPECT_EQ(all_slab_topologies_1_1_ax132, ref_all_slab_topologies_1_1_ax132);

    // topology1 (XYW-slab) to topology2 (XZW-slab)
    auto all_slab_topologies_1_2_ax012 =
        get_all_slab_topologies(topology1, topology2, axes012);
    auto all_slab_topologies_1_2_ax021 =
        get_all_slab_topologies(topology1, topology2, axes021);
    auto all_slab_topologies_1_2_ax102 =
        get_all_slab_topologies(topology1, topology2, axes102);
    auto all_slab_topologies_1_2_ax120 =
        get_all_slab_topologies(topology1, topology2, axes120);
    auto all_slab_topologies_1_2_ax201 =
        get_all_slab_topologies(topology1, topology2, axes201);
    auto all_slab_topologies_1_2_ax210 =
        get_all_slab_topologies(topology1, topology2, axes210);
    auto all_slab_topologies_1_2_ax123 =
        get_all_slab_topologies(topology1, topology2, axes123);
    auto all_slab_topologies_1_2_ax132 =
        get_all_slab_topologies(topology1, topology2, axes132);

    std::vector<topology_type> ref_all_slab_topologies_1_2_ax012 = {
        topology1, topology3, topology2};
    EXPECT_EQ(all_slab_topologies_1_2_ax012, ref_all_slab_topologies_1_2_ax012);
    std::vector<topology_type> ref_all_slab_topologies_1_2_ax021 = {topology1,
                                                                    topology2};
    EXPECT_EQ(all_slab_topologies_1_2_ax021, ref_all_slab_topologies_1_2_ax021);
    std::vector<topology_type> ref_all_slab_topologies_1_2_ax102 = {
        topology1, topology2, topology3, topology2};
    EXPECT_EQ(all_slab_topologies_1_2_ax102, ref_all_slab_topologies_1_2_ax102);
    std::vector<topology_type> ref_all_slab_topologies_1_2_ax120 = {
        topology1, topology3, topology2};
    EXPECT_EQ(all_slab_topologies_1_2_ax120, ref_all_slab_topologies_1_2_ax120);
    std::vector<topology_type> ref_all_slab_topologies_1_2_ax201 = {topology1,
                                                                    topology2};
    EXPECT_EQ(all_slab_topologies_1_2_ax201, ref_all_slab_topologies_1_2_ax201);
    std::vector<topology_type> ref_all_slab_topologies_1_2_ax210 = {topology1,
                                                                    topology2};
    EXPECT_EQ(all_slab_topologies_1_2_ax210, ref_all_slab_topologies_1_2_ax210);
    std::vector<topology_type> ref_all_slab_topologies_1_2_ax123 = {
        topology1, topology0, topology2};
    EXPECT_EQ(all_slab_topologies_1_2_ax123, ref_all_slab_topologies_1_2_ax123);
    std::vector<topology_type> ref_all_slab_topologies_1_2_ax132 = {
        topology1, topology2, topology1, topology2};
    EXPECT_EQ(all_slab_topologies_1_2_ax132, ref_all_slab_topologies_1_2_ax132);

    // topology1 (XYW-slab) to topology3 (YZW-slab)
    auto all_slab_topologies_1_3_ax012 =
        get_all_slab_topologies(topology1, topology3, axes012);
    auto all_slab_topologies_1_3_ax021 =
        get_all_slab_topologies(topology1, topology3, axes021);
    auto all_slab_topologies_1_3_ax102 =
        get_all_slab_topologies(topology1, topology3, axes102);
    auto all_slab_topologies_1_3_ax120 =
        get_all_slab_topologies(topology1, topology3, axes120);
    auto all_slab_topologies_1_3_ax201 =
        get_all_slab_topologies(topology1, topology3, axes201);
    auto all_slab_topologies_1_3_ax210 =
        get_all_slab_topologies(topology1, topology3, axes210);
    auto all_slab_topologies_1_3_ax123 =
        get_all_slab_topologies(topology1, topology3, axes123);
    auto all_slab_topologies_1_3_ax132 =
        get_all_slab_topologies(topology1, topology3, axes132);

    std::vector<topology_type> ref_all_slab_topologies_1_3_ax012 = {
        topology1, topology3, topology2, topology3};
    EXPECT_EQ(all_slab_topologies_1_3_ax012, ref_all_slab_topologies_1_3_ax012);
    std::vector<topology_type> ref_all_slab_topologies_1_3_ax021 = {
        topology1, topology2, topology3};
    EXPECT_EQ(all_slab_topologies_1_3_ax021, ref_all_slab_topologies_1_3_ax021);
    std::vector<topology_type> ref_all_slab_topologies_1_3_ax102 = {
        topology1, topology2, topology3};
    EXPECT_EQ(all_slab_topologies_1_3_ax102, ref_all_slab_topologies_1_3_ax102);
    std::vector<topology_type> ref_all_slab_topologies_1_3_ax120 = {topology1,
                                                                    topology3};
    EXPECT_EQ(all_slab_topologies_1_3_ax120, ref_all_slab_topologies_1_3_ax120);
    std::vector<topology_type> ref_all_slab_topologies_1_3_ax201 = {topology1,
                                                                    topology3};
    EXPECT_EQ(all_slab_topologies_1_3_ax201, ref_all_slab_topologies_1_3_ax201);
    std::vector<topology_type> ref_all_slab_topologies_1_3_ax210 = {topology1,
                                                                    topology3};
    EXPECT_EQ(all_slab_topologies_1_3_ax210, ref_all_slab_topologies_1_3_ax210);
    std::vector<topology_type> ref_all_slab_topologies_1_3_ax123 = {topology1,
                                                                    topology3};
    EXPECT_EQ(all_slab_topologies_1_3_ax123, ref_all_slab_topologies_1_3_ax123);
    std::vector<topology_type> ref_all_slab_topologies_1_3_ax132 = {topology1,
                                                                    topology3};
    EXPECT_EQ(all_slab_topologies_1_3_ax132, ref_all_slab_topologies_1_3_ax132);

    // topology2 (XYW-slab) to topology0 (XYZ-slab)
    auto all_slab_topologies_2_0_ax012 =
        get_all_slab_topologies(topology2, topology0, axes012);
    auto all_slab_topologies_2_0_ax021 =
        get_all_slab_topologies(topology2, topology0, axes021);
    auto all_slab_topologies_2_0_ax102 =
        get_all_slab_topologies(topology2, topology0, axes102);
    auto all_slab_topologies_2_0_ax120 =
        get_all_slab_topologies(topology2, topology0, axes120);
    auto all_slab_topologies_2_0_ax201 =
        get_all_slab_topologies(topology2, topology0, axes201);
    auto all_slab_topologies_2_0_ax210 =
        get_all_slab_topologies(topology2, topology0, axes210);
    auto all_slab_topologies_2_0_ax123 =
        get_all_slab_topologies(topology2, topology0, axes123);
    auto all_slab_topologies_2_0_ax132 =
        get_all_slab_topologies(topology2, topology0, axes132);

    std::vector<topology_type> ref_all_slab_topologies_2_0_ax012 = {topology2,
                                                                    topology0};
    EXPECT_EQ(all_slab_topologies_2_0_ax012, ref_all_slab_topologies_2_0_ax012);
    std::vector<topology_type> ref_all_slab_topologies_2_0_ax021 = {topology2,
                                                                    topology0};
    EXPECT_EQ(all_slab_topologies_2_0_ax021, ref_all_slab_topologies_2_0_ax021);
    std::vector<topology_type> ref_all_slab_topologies_2_0_ax102 = {topology2,
                                                                    topology0};
    EXPECT_EQ(all_slab_topologies_2_0_ax102, ref_all_slab_topologies_2_0_ax102);
    std::vector<topology_type> ref_all_slab_topologies_2_0_ax120 = {topology2,
                                                                    topology0};
    EXPECT_EQ(all_slab_topologies_2_0_ax120, ref_all_slab_topologies_2_0_ax120);
    std::vector<topology_type> ref_all_slab_topologies_2_0_ax201 = {topology2,
                                                                    topology0};
    EXPECT_EQ(all_slab_topologies_2_0_ax201, ref_all_slab_topologies_2_0_ax201);
    std::vector<topology_type> ref_all_slab_topologies_2_0_ax210 = {topology2,
                                                                    topology0};
    EXPECT_EQ(all_slab_topologies_2_0_ax210, ref_all_slab_topologies_2_0_ax210);
    std::vector<topology_type> ref_all_slab_topologies_2_0_ax123 = {topology2,
                                                                    topology0};
    EXPECT_EQ(all_slab_topologies_2_0_ax123, ref_all_slab_topologies_2_0_ax123);
    std::vector<topology_type> ref_all_slab_topologies_2_0_ax132 = {topology2,
                                                                    topology0};
    EXPECT_EQ(all_slab_topologies_2_0_ax132, ref_all_slab_topologies_2_0_ax132);

    // topology2 (XYW-slab) to topology1 (XZW-slab)
    auto all_slab_topologies_2_1_ax012 =
        get_all_slab_topologies(topology2, topology1, axes012);
    auto all_slab_topologies_2_1_ax021 =
        get_all_slab_topologies(topology2, topology1, axes021);
    auto all_slab_topologies_2_1_ax102 =
        get_all_slab_topologies(topology2, topology1, axes102);
    auto all_slab_topologies_2_1_ax120 =
        get_all_slab_topologies(topology2, topology1, axes120);
    auto all_slab_topologies_2_1_ax201 =
        get_all_slab_topologies(topology2, topology1, axes201);
    auto all_slab_topologies_2_1_ax210 =
        get_all_slab_topologies(topology2, topology1, axes210);
    auto all_slab_topologies_2_1_ax123 =
        get_all_slab_topologies(topology2, topology1, axes123);
    auto all_slab_topologies_2_1_ax132 =
        get_all_slab_topologies(topology2, topology1, axes132);

    std::vector<topology_type> ref_all_slab_topologies_2_1_ax012 = {topology2,
                                                                    topology1};
    EXPECT_EQ(all_slab_topologies_2_1_ax012, ref_all_slab_topologies_2_1_ax012);
    std::vector<topology_type> ref_all_slab_topologies_2_1_ax021 = {
        topology2, topology3, topology1};
    EXPECT_EQ(all_slab_topologies_2_1_ax021, ref_all_slab_topologies_2_1_ax021);
    std::vector<topology_type> ref_all_slab_topologies_2_1_ax102 = {topology2,
                                                                    topology1};
    EXPECT_EQ(all_slab_topologies_2_1_ax102, ref_all_slab_topologies_2_1_ax102);
    std::vector<topology_type> ref_all_slab_topologies_2_1_ax120 = {topology2,
                                                                    topology1};
    EXPECT_EQ(all_slab_topologies_2_1_ax120, ref_all_slab_topologies_2_1_ax120);
    std::vector<topology_type> ref_all_slab_topologies_2_1_ax201 = {
        topology2, topology1, topology3, topology1};
    EXPECT_EQ(all_slab_topologies_2_1_ax201, ref_all_slab_topologies_2_1_ax201);
    std::vector<topology_type> ref_all_slab_topologies_2_1_ax210 = {
        topology2, topology3, topology1};
    EXPECT_EQ(all_slab_topologies_2_1_ax210, ref_all_slab_topologies_2_1_ax210);
    std::vector<topology_type> ref_all_slab_topologies_2_1_ax123 = {topology2,
                                                                    topology1};
    EXPECT_EQ(all_slab_topologies_2_1_ax123, ref_all_slab_topologies_2_1_ax123);
    std::vector<topology_type> ref_all_slab_topologies_2_1_ax132 = {topology2,
                                                                    topology1};
    EXPECT_EQ(all_slab_topologies_2_1_ax132, ref_all_slab_topologies_2_1_ax132);

    /*
    // topology1 (XZ-slab) to topology2 (YZ-slab)
    auto all_slab_topologies_1_2_ax012 =
        get_all_slab_topologies(topology1, topology2, axes012);
    auto all_slab_topologies_1_2_ax021 =
        get_all_slab_topologies(topology1, topology2, axes021);
    auto all_slab_topologies_1_2_ax102 =
        get_all_slab_topologies(topology1, topology2, axes102);
    auto all_slab_topologies_1_2_ax120 =
        get_all_slab_topologies(topology1, topology2, axes120);
    auto all_slab_topologies_1_2_ax201 =
        get_all_slab_topologies(topology1, topology2, axes201);
    auto all_slab_topologies_1_2_ax210 =
        get_all_slab_topologies(topology1, topology2, axes210);

    std::vector<topology_type> ref_all_slab_topologies_1_2_ax012 = {
        topology1, topology0, topology2};
    EXPECT_EQ(all_slab_topologies_1_2_ax012, ref_all_slab_topologies_1_2_ax012);
    std::vector<topology_type> ref_all_slab_topologies_1_2_ax021 = {
        topology1, topology2, topology1, topology2};
    EXPECT_EQ(all_slab_topologies_1_2_ax021, ref_all_slab_topologies_1_2_ax021);
    std::vector<topology_type> ref_all_slab_topologies_1_2_ax102 = {topology1,
                                                                    topology2};
    EXPECT_EQ(all_slab_topologies_1_2_ax102, ref_all_slab_topologies_1_2_ax102);
    std::vector<topology_type> ref_all_slab_topologies_1_2_ax120 = {topology1,
                                                                    topology2};
    EXPECT_EQ(all_slab_topologies_1_2_ax120, ref_all_slab_topologies_1_2_ax120);
    std::vector<topology_type> ref_all_slab_topologies_1_2_ax201 = {
        topology1, topology0, topology2};
    EXPECT_EQ(all_slab_topologies_1_2_ax201, ref_all_slab_topologies_1_2_ax201);
    std::vector<topology_type> ref_all_slab_topologies_1_2_ax210 = {topology1,
                                                                    topology2};
    EXPECT_EQ(all_slab_topologies_1_2_ax210, ref_all_slab_topologies_1_2_ax210);

    // topology2 (YZ-slab) to topology0 (XY-slab)
    auto all_slab_topologies_2_0_ax012 =
        get_all_slab_topologies(topology2, topology0, axes012);
    auto all_slab_topologies_2_0_ax021 =
        get_all_slab_topologies(topology2, topology0, axes021);
    auto all_slab_topologies_2_0_ax102 =
        get_all_slab_topologies(topology2, topology0, axes102);
    auto all_slab_topologies_2_0_ax120 =
        get_all_slab_topologies(topology2, topology0, axes120);
    auto all_slab_topologies_2_0_ax201 =
        get_all_slab_topologies(topology2, topology0, axes201);
    auto all_slab_topologies_2_0_ax210 =
        get_all_slab_topologies(topology2, topology0, axes210);

    std::vector<topology_type> ref_all_slab_topologies_2_0_ax012 = {topology2,
                                                                    topology0};
    EXPECT_EQ(all_slab_topologies_2_0_ax012, ref_all_slab_topologies_2_0_ax012);
    std::vector<topology_type> ref_all_slab_topologies_2_0_ax021 = {topology2,
                                                                    topology0};
    EXPECT_EQ(all_slab_topologies_2_0_ax021, ref_all_slab_topologies_2_0_ax021);
    std::vector<topology_type> ref_all_slab_topologies_2_0_ax102 = {topology2,
                                                                    topology0};
    EXPECT_EQ(all_slab_topologies_2_0_ax102, ref_all_slab_topologies_2_0_ax102);
    std::vector<topology_type> ref_all_slab_topologies_2_0_ax120 = {
        topology2, topology1, topology0};
    EXPECT_EQ(all_slab_topologies_2_0_ax120, ref_all_slab_topologies_2_0_ax120);
    std::vector<topology_type> ref_all_slab_topologies_2_0_ax201 = {
        topology2, topology1, topology0};
    EXPECT_EQ(all_slab_topologies_2_0_ax201, ref_all_slab_topologies_2_0_ax201);
    std::vector<topology_type> ref_all_slab_topologies_2_0_ax210 = {
        topology2, topology0, topology2, topology0};
    EXPECT_EQ(all_slab_topologies_2_0_ax210, ref_all_slab_topologies_2_0_ax210);

    // topology2 (YZ-slab) to topology1 (XZ-slab)
    auto all_slab_topologies_2_1_ax012 =
        get_all_slab_topologies(topology2, topology1, axes012);
    auto all_slab_topologies_2_1_ax021 =
        get_all_slab_topologies(topology2, topology1, axes021);
    auto all_slab_topologies_2_1_ax102 =
        get_all_slab_topologies(topology2, topology1, axes102);
    auto all_slab_topologies_2_1_ax120 =
        get_all_slab_topologies(topology2, topology1, axes120);
    auto all_slab_topologies_2_1_ax201 =
        get_all_slab_topologies(topology2, topology1, axes201);
    auto all_slab_topologies_2_1_ax210 =
        get_all_slab_topologies(topology2, topology1, axes210);

    std::vector<topology_type> ref_all_slab_topologies_2_1_ax012 = {topology2,
                                                                    topology1};
    EXPECT_EQ(all_slab_topologies_2_1_ax012, ref_all_slab_topologies_2_1_ax012);
    std::vector<topology_type> ref_all_slab_topologies_2_1_ax021 = {topology2,
                                                                    topology1};
    EXPECT_EQ(all_slab_topologies_2_1_ax021, ref_all_slab_topologies_2_1_ax021);
    std::vector<topology_type> ref_all_slab_topologies_2_1_ax102 = {
        topology2, topology0, topology1};
    EXPECT_EQ(all_slab_topologies_2_1_ax102, ref_all_slab_topologies_2_1_ax102);
    std::vector<topology_type> ref_all_slab_topologies_2_1_ax120 = {
        topology2, topology1, topology2, topology1};
    EXPECT_EQ(all_slab_topologies_2_1_ax120, ref_all_slab_topologies_2_1_ax120);
    std::vector<topology_type> ref_all_slab_topologies_2_1_ax201 = {topology2,
                                                                    topology1};
    EXPECT_EQ(all_slab_topologies_2_1_ax201, ref_all_slab_topologies_2_1_ax201);
    std::vector<topology_type> ref_all_slab_topologies_2_1_ax210 = {
        topology2, topology0, topology1};
    EXPECT_EQ(all_slab_topologies_2_1_ax210, ref_all_slab_topologies_2_1_ax210);

    // topology2 (YZ-slab) to topology2 (YZ-slab)
    auto all_slab_topologies_2_2_ax012 =
        get_all_slab_topologies(topology2, topology2, axes012);
    auto all_slab_topologies_2_2_ax021 =
        get_all_slab_topologies(topology2, topology2, axes021);
    auto all_slab_topologies_2_2_ax102 =
        get_all_slab_topologies(topology2, topology2, axes102);
    auto all_slab_topologies_2_2_ax120 =
        get_all_slab_topologies(topology2, topology2, axes120);
    auto all_slab_topologies_2_2_ax201 =
        get_all_slab_topologies(topology2, topology2, axes201);
    auto all_slab_topologies_2_2_ax210 =
        get_all_slab_topologies(topology2, topology2, axes210);

    std::vector<topology_type> ref_all_slab_topologies_2_2_ax012 = {
        topology2, topology0, topology2};
    EXPECT_EQ(all_slab_topologies_2_2_ax012, ref_all_slab_topologies_2_2_ax012);
    std::vector<topology_type> ref_all_slab_topologies_2_2_ax021 = {
        topology2, topology1, topology2};
    EXPECT_EQ(all_slab_topologies_2_2_ax021, ref_all_slab_topologies_2_2_ax021);
    std::vector<topology_type> ref_all_slab_topologies_2_2_ax102 = {
        topology2, topology0, topology2};
    EXPECT_EQ(all_slab_topologies_2_2_ax102, ref_all_slab_topologies_2_2_ax102);
    std::vector<topology_type> ref_all_slab_topologies_2_2_ax120 = {
        topology2, topology1, topology2};
    EXPECT_EQ(all_slab_topologies_2_2_ax120, ref_all_slab_topologies_2_2_ax120);
    std::vector<topology_type> ref_all_slab_topologies_2_2_ax201 = {
        topology2, topology1, topology2};
    EXPECT_EQ(all_slab_topologies_2_2_ax201, ref_all_slab_topologies_2_2_ax201);
    std::vector<topology_type> ref_all_slab_topologies_2_2_ax210 = {
        topology2, topology0, topology2};
    EXPECT_EQ(all_slab_topologies_2_2_ax210, ref_all_slab_topologies_2_2_ax210);
    */
  }
}

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

void test_get_mid_array_pencil_4D(std::size_t nprocs) {
  using topology_type     = std::array<std::size_t, 4>;
  topology_type topology0 = {1, 1, nprocs, 8};
  topology_type topology1 = {1, nprocs, 1, 8};
  topology_type topology2 = {1, 8, nprocs, 1};
  topology_type topology3 = {1, nprocs, 8, 1};
  topology_type topology4 = {1, 8, 1, nprocs};
  topology_type topology5 = {1, 1, 8, nprocs};

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
        { [[maybe_unused]] auto mid02 = get_mid_array(topology0, topology2); },
        std::runtime_error);
    EXPECT_THROW(
        { [[maybe_unused]] auto mid05 = get_mid_array(topology0, topology5); },
        std::runtime_error);
    EXPECT_THROW(
        { [[maybe_unused]] auto mid13 = get_mid_array(topology1, topology3); },
        std::runtime_error);
    EXPECT_THROW(
        { [[maybe_unused]] auto mid14 = get_mid_array(topology1, topology4); },
        std::runtime_error);
    EXPECT_THROW(
        { [[maybe_unused]] auto mid23 = get_mid_array(topology2, topology3); },
        std::runtime_error);
    EXPECT_THROW(
        { [[maybe_unused]] auto mid24 = get_mid_array(topology2, topology4); },
        std::runtime_error);
    EXPECT_THROW(
        { [[maybe_unused]] auto mid35 = get_mid_array(topology3, topology5); },
        std::runtime_error);
    EXPECT_THROW(
        { [[maybe_unused]] auto mid45 = get_mid_array(topology4, topology5); },
        std::runtime_error);

    auto mid03 = get_mid_array(topology0, topology3);
    auto mid04 = get_mid_array(topology0, topology4);
    auto mid12 = get_mid_array(topology1, topology2);
    auto mid15 = get_mid_array(topology1, topology5);
    auto mid25 = get_mid_array(topology2, topology5);
    auto mid34 = get_mid_array(topology3, topology4);
    EXPECT_EQ(mid03, topology1);
    EXPECT_EQ(mid04, topology2);
    EXPECT_EQ(mid12, topology0);
    EXPECT_EQ(mid15, topology3);
    EXPECT_EQ(mid25, topology4);
    EXPECT_EQ(mid34, topology5);
  }
}

void test_merge_topology(std::size_t nprocs) {
  using topology_type     = std::array<std::size_t, 3>;
  topology_type topology0 = {1, 1, nprocs};
  topology_type topology1 = {1, nprocs, 1};
  topology_type topology2 = {nprocs, 1, 1};
  topology_type topology3 = {nprocs, 1, 8};
  topology_type topology4 = {nprocs, 8, 1};

  auto merged01              = merge_topology(topology0, topology1);
  auto merged02              = merge_topology(topology0, topology2);
  auto merged12              = merge_topology(topology1, topology2);
  topology_type ref_merged01 = {1, nprocs, nprocs};
  topology_type ref_merged02 = {nprocs, 1, nprocs};
  topology_type ref_merged12 = {nprocs, nprocs, 1};
  EXPECT_EQ(merged01, ref_merged01);
  EXPECT_EQ(merged02, ref_merged02);
  EXPECT_EQ(merged12, ref_merged12);

  // Failure tests because these do not have same size
  EXPECT_THROW(
      {
        [[maybe_unused]] auto merged03 = merge_topology(topology0, topology3);
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        [[maybe_unused]] auto merged04 = merge_topology(topology0, topology4);
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        [[maybe_unused]] auto merged13 = merge_topology(topology1, topology3);
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        [[maybe_unused]] auto merged14 = merge_topology(topology1, topology4);
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        [[maybe_unused]] auto merged23 = merge_topology(topology2, topology3);
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        [[maybe_unused]] auto merged24 = merge_topology(topology2, topology4);
      },
      std::runtime_error);

  // This case is valid
  topology_type ref_merged34 = {nprocs, 8, 8};
  auto merged34              = merge_topology(topology3, topology4);
  EXPECT_EQ(merged34, ref_merged34);
}

void test_diff_topology(std::size_t nprocs) {
  using topology_type      = std::array<std::size_t, 3>;
  topology_type topology0  = {1, 1, nprocs};
  topology_type topology1  = {1, nprocs, 1};
  topology_type topology2  = {nprocs, 1, 1};
  topology_type topology3  = {nprocs, 1, 8};
  topology_type topology01 = {1, nprocs, nprocs};
  topology_type topology02 = {nprocs, 1, nprocs};
  topology_type topology12 = {nprocs, nprocs, 1};

  std::size_t diff0_01 = diff_toplogy(topology0, topology01);
  std::size_t diff0_02 = diff_toplogy(topology0, topology02);
  std::size_t diff1_12 = diff_toplogy(topology1, topology12);
  std::size_t diff2_12 = diff_toplogy(topology2, topology12);

  std::size_t ref_diff0_01 = nprocs;
  std::size_t ref_diff0_02 = nprocs;
  std::size_t ref_diff1_12 = nprocs;
  std::size_t ref_diff2_12 = nprocs;

  EXPECT_EQ(diff0_01, ref_diff0_01);
  EXPECT_EQ(diff0_02, ref_diff0_02);
  EXPECT_EQ(diff1_12, ref_diff1_12);
  EXPECT_EQ(diff2_12, ref_diff2_12);

  if (nprocs == 1) {
    std::size_t diff03     = diff_toplogy(topology0, topology3);
    std::size_t ref_diff03 = topology3.at(2);
    EXPECT_EQ(diff03, ref_diff03);
  } else {
    // Failure tests because more than two elements are different
    EXPECT_THROW(
        {
          [[maybe_unused]] std::size_t diff03 =
              diff_toplogy(topology0, topology3);
        },
        std::runtime_error);
  }
}

void test_get_topology_type(std::size_t nprocs) {
  using topology1D_type = std::array<std::size_t, 1>;
  using topology2D_type = std::array<std::size_t, 2>;
  using topology3D_type = std::array<std::size_t, 3>;
  using topology4D_type = std::array<std::size_t, 4>;

  const std::size_t p0 = 2, p1 = 3, p2 = 4, p3 = 5;

  topology1D_type topology1{nprocs};
  topology2D_type topology2_1{p0, nprocs}, topology2_2{nprocs, p1};
  topology3D_type topology3_1{p0, p1, nprocs}, topology3_2{p0, nprocs, p2},
      topology3_3{nprocs, p1, p2}, topology3_4{nprocs, nprocs, p2},
      topology3_5{nprocs, nprocs, nprocs};
  topology4D_type topology4_1{p0, p1, nprocs, nprocs},
      topology4_2{p0, nprocs, p2, nprocs}, topology4_3{p0, nprocs, nprocs, p3},
      topology4_4{nprocs, p1, p2, nprocs}, topology4_5{nprocs, p1, nprocs, p3},
      topology4_6{nprocs, nprocs, p2, p3},
      topology4_7{p0, nprocs, nprocs, nprocs},
      topology4_8{nprocs, nprocs, nprocs, nprocs};

  if (nprocs == 1) {
    // 1D topology
    auto topo1 = get_topology_type(topology1);
    EXPECT_EQ(topo1, TopologyType::Shared);

    EXPECT_THROW(
        { [[maybe_unused]] auto topo = get_topology_type(topology1D_type{0}); },
        std::runtime_error);

    auto topo2_1 = get_topology_type(topology2_1);
    auto topo2_2 = get_topology_type(topology2_2);
    EXPECT_EQ(topo2_1, TopologyType::Slab);
    EXPECT_EQ(topo2_2, TopologyType::Slab);

    EXPECT_THROW(
        {
          [[maybe_unused]] auto topo =
              get_topology_type(topology2D_type{0, nprocs});
        },
        std::runtime_error);
    EXPECT_THROW(
        {
          [[maybe_unused]] auto topo =
              get_topology_type(topology2D_type{nprocs, 0});
        },
        std::runtime_error);

    auto topo3_1 = get_topology_type(topology3_1);
    auto topo3_2 = get_topology_type(topology3_2);
    auto topo3_3 = get_topology_type(topology3_3);
    auto topo3_4 = get_topology_type(topology3_4);
    auto topo3_5 = get_topology_type(topology3_5);
    EXPECT_EQ(topo3_1, TopologyType::Pencil);
    EXPECT_EQ(topo3_2, TopologyType::Pencil);
    EXPECT_EQ(topo3_3, TopologyType::Pencil);
    EXPECT_EQ(topo3_4, TopologyType::Slab);
    EXPECT_EQ(topo3_5, TopologyType::Shared);

    auto topo4_1 = get_topology_type(topology4_1);
    auto topo4_2 = get_topology_type(topology4_2);
    auto topo4_3 = get_topology_type(topology4_3);
    auto topo4_4 = get_topology_type(topology4_4);
    auto topo4_5 = get_topology_type(topology4_5);
    auto topo4_6 = get_topology_type(topology4_6);
    auto topo4_7 = get_topology_type(topology4_7);
    auto topo4_8 = get_topology_type(topology4_8);
    EXPECT_EQ(topo4_1, TopologyType::Pencil);
    EXPECT_EQ(topo4_2, TopologyType::Pencil);
    EXPECT_EQ(topo4_3, TopologyType::Pencil);
    EXPECT_EQ(topo4_4, TopologyType::Pencil);
    EXPECT_EQ(topo4_5, TopologyType::Pencil);
    EXPECT_EQ(topo4_6, TopologyType::Pencil);
    EXPECT_EQ(topo4_7, TopologyType::Slab);
    EXPECT_EQ(topo4_8, TopologyType::Shared);

  } else {
    // 1D topology
    auto topo1 = get_topology_type(topology1);
    EXPECT_EQ(topo1, TopologyType::Slab);

    // 2D topology
    auto topo2_1 = get_topology_type(topology2_1);
    auto topo2_2 = get_topology_type(topology2_2);
    EXPECT_EQ(topo2_1, TopologyType::Pencil);
    EXPECT_EQ(topo2_2, TopologyType::Pencil);

    // 3D topology
    EXPECT_THROW(
        { [[maybe_unused]] auto topo3_1 = get_topology_type(topology3_1); },
        std::runtime_error);
    EXPECT_THROW(
        { [[maybe_unused]] auto topo3_2 = get_topology_type(topology3_2); },
        std::runtime_error);
    EXPECT_THROW(
        { [[maybe_unused]] auto topo3_3 = get_topology_type(topology3_3); },
        std::runtime_error);
    EXPECT_THROW(
        { [[maybe_unused]] auto topo3_4 = get_topology_type(topology3_4); },
        std::runtime_error);
    EXPECT_THROW(
        { [[maybe_unused]] auto topo3_5 = get_topology_type(topology3_5); },
        std::runtime_error);

    // 4D topology
    EXPECT_THROW(
        { [[maybe_unused]] auto topo4_1 = get_topology_type(topology4_1); },
        std::runtime_error);
    EXPECT_THROW(
        { [[maybe_unused]] auto topo4_2 = get_topology_type(topology4_2); },
        std::runtime_error);
    EXPECT_THROW(
        { [[maybe_unused]] auto topo4_3 = get_topology_type(topology4_3); },
        std::runtime_error);
    EXPECT_THROW(
        { [[maybe_unused]] auto topo4_4 = get_topology_type(topology4_4); },
        std::runtime_error);
    EXPECT_THROW(
        { [[maybe_unused]] auto topo4_5 = get_topology_type(topology4_5); },
        std::runtime_error);
    EXPECT_THROW(
        { [[maybe_unused]] auto topo4_6 = get_topology_type(topology4_6); },
        std::runtime_error);
    EXPECT_THROW(
        { [[maybe_unused]] auto topo4_7 = get_topology_type(topology4_7); },
        std::runtime_error);
    EXPECT_THROW(
        { [[maybe_unused]] auto topo4_8 = get_topology_type(topology4_8); },
        std::runtime_error);
  }
}

void test_is_topology(std::size_t nprocs) {
  using topology1D_type = std::array<std::size_t, 1>;
  using topology2D_type = std::array<std::size_t, 2>;
  using topology3D_type = std::array<std::size_t, 3>;
  using topology4D_type = std::array<std::size_t, 4>;

  const std::size_t p0 = 2, p1 = 3, p2 = 4, p3 = 5;

  topology1D_type topology1{nprocs};
  topology2D_type topology2_1{p0, nprocs}, topology2_2{nprocs, p1};
  topology3D_type topology3_1{p0, p1, nprocs}, topology3_2{p0, nprocs, p2},
      topology3_3{nprocs, p1, p2}, topology3_4{nprocs, nprocs, p2},
      topology3_5{nprocs, nprocs, nprocs};
  topology4D_type topology4_1{p0, p1, nprocs, nprocs},
      topology4_2{p0, nprocs, p2, nprocs}, topology4_3{p0, nprocs, nprocs, p3},
      topology4_4{nprocs, p1, p2, nprocs}, topology4_5{nprocs, p1, nprocs, p3},
      topology4_6{nprocs, nprocs, p2, p3},
      topology4_7{p0, nprocs, nprocs, nprocs},
      topology4_8{nprocs, nprocs, nprocs, nprocs};

  if (nprocs == 1) {
    // 1D topology is shared
    EXPECT_TRUE(is_shared_topology(topology1));
    EXPECT_FALSE(is_slab_topology(topology1));
    EXPECT_FALSE(is_pencil_topology(topology1));

    // Invalid 1D topology
    EXPECT_FALSE(is_pencil_topology(topology1D_type{0}));

    // 2D topology is slab
    EXPECT_TRUE(is_slab_topology(topology2_1));
    EXPECT_TRUE(is_slab_topology(topology2_2));
    EXPECT_FALSE(is_shared_topology(topology2_1));
    EXPECT_FALSE(is_shared_topology(topology2_2));
    EXPECT_FALSE(is_pencil_topology(topology2_1));
    EXPECT_FALSE(is_pencil_topology(topology2_2));

    // Invalid 2D topology
    EXPECT_FALSE(is_slab_topology(topology2D_type{0, nprocs}));
    EXPECT_FALSE(is_slab_topology(topology2D_type{nprocs, 0}));
    EXPECT_FALSE(is_shared_topology(topology2D_type{0, nprocs}));
    EXPECT_FALSE(is_shared_topology(topology2D_type{nprocs, 0}));
    EXPECT_FALSE(is_pencil_topology(topology2D_type{0, nprocs}));
    EXPECT_FALSE(is_pencil_topology(topology2D_type{nprocs, 0}));

    // 3D case
    // Pencil topologies
    EXPECT_TRUE(is_pencil_topology(topology3_1));
    EXPECT_TRUE(is_pencil_topology(topology3_2));
    EXPECT_TRUE(is_pencil_topology(topology3_3));
    EXPECT_TRUE(is_slab_topology(topology3_4));
    EXPECT_TRUE(is_shared_topology(topology3_5));

    // 4D case
    EXPECT_TRUE(is_pencil_topology(topology4_1));
    EXPECT_TRUE(is_pencil_topology(topology4_2));
    EXPECT_TRUE(is_pencil_topology(topology4_3));
    EXPECT_TRUE(is_pencil_topology(topology4_4));
    EXPECT_TRUE(is_pencil_topology(topology4_5));
    EXPECT_TRUE(is_pencil_topology(topology4_6));
    EXPECT_TRUE(is_slab_topology(topology4_7));
    EXPECT_TRUE(is_shared_topology(topology4_8));
  } else {
    // 1D topology
    EXPECT_TRUE(is_slab_topology(topology1));

    // 2D topology
    EXPECT_TRUE(is_pencil_topology(topology2_1));
    EXPECT_TRUE(is_pencil_topology(topology2_2));

    // 3D topology
    EXPECT_FALSE(is_shared_topology(topology3_1));
    EXPECT_FALSE(is_shared_topology(topology3_2));
    EXPECT_FALSE(is_shared_topology(topology3_3));
    EXPECT_FALSE(is_shared_topology(topology3_4));
    EXPECT_FALSE(is_shared_topology(topology3_5));
    EXPECT_FALSE(is_slab_topology(topology3_1));
    EXPECT_FALSE(is_slab_topology(topology3_2));
    EXPECT_FALSE(is_slab_topology(topology3_3));
    EXPECT_FALSE(is_slab_topology(topology3_4));
    EXPECT_FALSE(is_slab_topology(topology3_5));
    EXPECT_FALSE(is_pencil_topology(topology3_1));
    EXPECT_FALSE(is_pencil_topology(topology3_2));
    EXPECT_FALSE(is_pencil_topology(topology3_3));
    EXPECT_FALSE(is_pencil_topology(topology3_4));
    EXPECT_FALSE(is_pencil_topology(topology3_5));

    // 4D topology
    EXPECT_FALSE(is_shared_topology(topology4_1));
    EXPECT_FALSE(is_shared_topology(topology4_2));
    EXPECT_FALSE(is_shared_topology(topology4_3));
    EXPECT_FALSE(is_shared_topology(topology4_4));
    EXPECT_FALSE(is_shared_topology(topology4_5));
    EXPECT_FALSE(is_shared_topology(topology4_6));
    EXPECT_FALSE(is_shared_topology(topology4_7));
    EXPECT_FALSE(is_shared_topology(topology4_8));
    EXPECT_FALSE(is_slab_topology(topology4_1));
    EXPECT_FALSE(is_slab_topology(topology4_2));
    EXPECT_FALSE(is_slab_topology(topology4_3));
    EXPECT_FALSE(is_slab_topology(topology4_4));
    EXPECT_FALSE(is_slab_topology(topology4_5));
    EXPECT_FALSE(is_slab_topology(topology4_6));
    EXPECT_FALSE(is_slab_topology(topology4_7));
    EXPECT_FALSE(is_slab_topology(topology4_8));
    EXPECT_FALSE(is_pencil_topology(topology4_1));
    EXPECT_FALSE(is_pencil_topology(topology4_2));
    EXPECT_FALSE(is_pencil_topology(topology4_3));
    EXPECT_FALSE(is_pencil_topology(topology4_4));
    EXPECT_FALSE(is_pencil_topology(topology4_5));
    EXPECT_FALSE(is_pencil_topology(topology4_6));
    EXPECT_FALSE(is_pencil_topology(topology4_7));
    EXPECT_FALSE(is_pencil_topology(topology4_8));
  }
}

/*
void test_get_all_pencil_topologies1D_3DView(std::size_t nprocs) {
  using topology_type   = std::array<std::size_t, 3>;
  using topologies_type = std::vector<topology_type>;
  using topology_r_type = Topology<std::size_t, 3, Kokkos::LayoutRight>;
  using topology_l_type = Topology<std::size_t, 3, Kokkos::LayoutLeft>;
  using vec_axis_type   = std::vector<std::size_t>;
  using topologies_and_axes_type = std::tuple<topologies_type, vec_axis_type>;
  std::size_t np0                = 4;

  topology_r_type topology0 = {1, nprocs, np0}, topology1 = {nprocs, 1, np0},
                  topology3 = {nprocs, np0, 1};
  topology_l_type topology2 = {np0, nprocs, 1}, topology4 = {np0, 1, nprocs};
  topology_type ref_topo0 = topology0.array(), ref_topo1 = topology1.array(),
                ref_topo2 = topology2.array(), ref_topo3 = topology3.array(),
                ref_topo4 = topology4.array();

  using axes_type = std::array<int, 1>;
  axes_type axes0 = {0}, axes1 = {1}, axes2 = {2};

  std::vector<axes_type> all_axes = {axes0, axes1, axes2};

  if (nprocs == 1) {
    for (const auto& axes : all_axes) {
      // Failure tests because only two elements differ (slabs)
      EXPECT_THROW(
          {
            [[maybe_unused]] auto topologies_and_axes_0_1 =
                get_all_pencil_topologies(topology0, topology1, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto topologies_and_axes_0_2 =
                get_all_pencil_topologies(topology0, topology2, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto topologies_and_axes_1_0 =
                get_all_pencil_topologies(topology1, topology0, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto topologies_and_axes_2_0 =
                get_all_pencil_topologies(topology2, topology0, axes);
          },
          std::runtime_error);
    }
  } else {
    // topology0 to topology0
    auto topologies_and_axes_0_0_0 =
        get_all_pencil_topologies(topology0, topology0, axes0);
    auto topologies_and_axes_0_0_1 =
        get_all_pencil_topologies(topology0, topology0, axes1);
    auto topologies_and_axes_0_0_2 =
        get_all_pencil_topologies(topology0, topology0, axes2);

    topologies_and_axes_type ref_topologies_and_axes_0_0_0 =
        std::make_tuple(topologies_type{ref_topo0}, vec_axis_type{});
    EXPECT_EQ(topologies_and_axes_0_0_0, ref_topologies_and_axes_0_0_0);

    topologies_and_axes_type ref_topologies_and_axes_0_0_1 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0}, vec_axis_type{0, 0});
    EXPECT_EQ(topologies_and_axes_0_0_1, ref_topologies_and_axes_0_0_1);

    topologies_and_axes_type ref_topologies_and_axes_0_0_2 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2, ref_topo0}, vec_axis_type{1, 1});
    EXPECT_EQ(topologies_and_axes_0_0_2, ref_topologies_and_axes_0_0_2);

    // topology0 to topology1
    auto topologies_and_axes_0_1_0 =
        get_all_pencil_topologies(topology0, topology1, axes0);
    auto topologies_and_axes_0_1_1 =
        get_all_pencil_topologies(topology0, topology1, axes1);
    auto topologies_and_axes_0_1_2 =
        get_all_pencil_topologies(topology0, topology1, axes2);

    topologies_and_axes_type ref_topologies_and_axes_0_1_0 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1}, vec_axis_type{0});
    EXPECT_EQ(topologies_and_axes_0_1_0, ref_topologies_and_axes_0_1_0);

    topologies_and_axes_type ref_topologies_and_axes_0_1_1 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1}, vec_axis_type{0});
    EXPECT_EQ(topologies_and_axes_0_1_1, ref_topologies_and_axes_0_1_1);

    topologies_and_axes_type ref_topologies_and_axes_0_1_2 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2, ref_topo0, ref_topo1},
        vec_axis_type{1, 1, 0});
    EXPECT_EQ(topologies_and_axes_0_1_2, ref_topologies_and_axes_0_1_2);

    // topology0 to topology2
    auto topologies_and_axes_0_2_0 =
        get_all_pencil_topologies(topology0, topology2, axes0);
    auto topologies_and_axes_0_2_1 =
        get_all_pencil_topologies(topology0, topology2, axes1);
    auto topologies_and_axes_0_2_2 =
        get_all_pencil_topologies(topology0, topology2, axes2);

    topologies_and_axes_type ref_topologies_and_axes_0_2_0 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2}, vec_axis_type{1});
    EXPECT_EQ(topologies_and_axes_0_2_0, ref_topologies_and_axes_0_2_0);

    topologies_and_axes_type ref_topologies_and_axes_0_2_1 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{0, 0, 1});
    EXPECT_EQ(topologies_and_axes_0_2_1, ref_topologies_and_axes_0_2_1);

    topologies_and_axes_type ref_topologies_and_axes_0_2_2 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2}, vec_axis_type{1});
    EXPECT_EQ(topologies_and_axes_0_2_2, ref_topologies_and_axes_0_2_2);

    // topology1 to topology0
    auto topologies_and_axes_1_0_0 =
        get_all_pencil_topologies(topology1, topology0, axes0);
    auto topologies_and_axes_1_0_1 =
        get_all_pencil_topologies(topology1, topology0, axes1);
    auto topologies_and_axes_1_0_2 =
        get_all_pencil_topologies(topology1, topology0, axes2);

    topologies_and_axes_type ref_topologies_and_axes_1_0_0 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0}, vec_axis_type{0});
    EXPECT_EQ(topologies_and_axes_1_0_0, ref_topologies_and_axes_1_0_0);

    topologies_and_axes_type ref_topologies_and_axes_1_0_1 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0}, vec_axis_type{0});
    EXPECT_EQ(topologies_and_axes_1_0_1, ref_topologies_and_axes_1_0_1);

    topologies_and_axes_type ref_topologies_and_axes_1_0_2 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo3, ref_topo1, ref_topo0},
        vec_axis_type{1, 1, 0});
    EXPECT_EQ(topologies_and_axes_1_0_2, ref_topologies_and_axes_1_0_2);

    // topology1 to topology1
    auto topologies_and_axes_1_1_0 =
        get_all_pencil_topologies(topology1, topology1, axes0);
    auto topologies_and_axes_1_1_1 =
        get_all_pencil_topologies(topology1, topology1, axes1);
    auto topologies_and_axes_1_1_2 =
        get_all_pencil_topologies(topology1, topology1, axes2);

    topologies_and_axes_type ref_topologies_and_axes_1_1_0 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo1}, vec_axis_type{0, 0});
    EXPECT_EQ(topologies_and_axes_1_1_0, ref_topologies_and_axes_1_1_0);

    topologies_and_axes_type ref_topologies_and_axes_1_1_1 =
        std::make_tuple(topologies_type{ref_topo1}, vec_axis_type{});
    EXPECT_EQ(topologies_and_axes_1_1_1, ref_topologies_and_axes_1_1_1);

    topologies_and_axes_type ref_topologies_and_axes_1_1_2 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo3, ref_topo1}, vec_axis_type{1, 1});
    EXPECT_EQ(topologies_and_axes_1_1_2, ref_topologies_and_axes_1_1_2);

    // topology1 to topology2
    auto topologies_and_axes_1_2_0 =
        get_all_pencil_topologies(topology1, topology2, axes0);
    auto topologies_and_axes_1_2_1 =
        get_all_pencil_topologies(topology1, topology2, axes1);
    auto topologies_and_axes_1_2_2 =
        get_all_pencil_topologies(topology1, topology2, axes2);

    topologies_and_axes_type ref_topologies_and_axes_1_2_0 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo2}, vec_axis_type{0, 1});
    EXPECT_EQ(topologies_and_axes_1_2_0, ref_topologies_and_axes_1_2_0);

    topologies_and_axes_type ref_topologies_and_axes_1_2_1 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo2}, vec_axis_type{0, 1});
    EXPECT_EQ(topologies_and_axes_1_2_1, ref_topologies_and_axes_1_2_1);

    topologies_and_axes_type ref_topologies_and_axes_1_2_2 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo2}, vec_axis_type{0, 1});
    EXPECT_EQ(topologies_and_axes_1_2_2, ref_topologies_and_axes_1_2_2);

    // topology2 to topology0
    auto topologies_and_axes_2_0_0 =
        get_all_pencil_topologies(topology2, topology0, axes0);
    auto topologies_and_axes_2_0_1 =
        get_all_pencil_topologies(topology2, topology0, axes1);
    auto topologies_and_axes_2_0_2 =
        get_all_pencil_topologies(topology2, topology0, axes2);

    topologies_and_axes_type ref_topologies_and_axes_2_0_0 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0}, vec_axis_type{1});
    EXPECT_EQ(topologies_and_axes_2_0_0, ref_topologies_and_axes_2_0_0);

    topologies_and_axes_type ref_topologies_and_axes_2_0_1 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo2, ref_topo0},
        vec_axis_type{0, 0, 1});
    EXPECT_EQ(topologies_and_axes_2_0_1, ref_topologies_and_axes_2_0_1);

    topologies_and_axes_type ref_topologies_and_axes_2_0_2 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0}, vec_axis_type{1});
    EXPECT_EQ(topologies_and_axes_2_0_2, ref_topologies_and_axes_2_0_2);

    // topology2 to topology1
    auto topologies_and_axes_2_1_0 =
        get_all_pencil_topologies(topology2, topology1, axes0);
    auto topologies_and_axes_2_1_1 =
        get_all_pencil_topologies(topology2, topology1, axes1);
    auto topologies_and_axes_2_1_2 =
        get_all_pencil_topologies(topology2, topology1, axes2);

    topologies_and_axes_type ref_topologies_and_axes_2_1_0 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0, ref_topo1}, vec_axis_type{1, 0});
    EXPECT_EQ(topologies_and_axes_2_1_0, ref_topologies_and_axes_2_1_0);

    topologies_and_axes_type ref_topologies_and_axes_2_1_1 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0, ref_topo1}, vec_axis_type{1, 0});
    EXPECT_EQ(topologies_and_axes_2_1_1, ref_topologies_and_axes_2_1_1);

    topologies_and_axes_type ref_topologies_and_axes_2_1_2 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0, ref_topo1}, vec_axis_type{1, 0});
    EXPECT_EQ(topologies_and_axes_2_1_2, ref_topologies_and_axes_2_1_2);

    // topology2 to topology2
    auto topologies_and_axes_2_2_0 =
        get_all_pencil_topologies(topology2, topology2, axes0);
    auto topologies_and_axes_2_2_1 =
        get_all_pencil_topologies(topology2, topology2, axes1);
    auto topologies_and_axes_2_2_2 =
        get_all_pencil_topologies(topology2, topology2, axes2);

    topologies_and_axes_type ref_topologies_and_axes_2_2_0 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0, ref_topo2}, vec_axis_type{1, 1});
    EXPECT_EQ(topologies_and_axes_2_2_0, ref_topologies_and_axes_2_2_0);

    topologies_and_axes_type ref_topologies_and_axes_2_2_1 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo2}, vec_axis_type{0, 0});
    EXPECT_EQ(topologies_and_axes_2_2_1, ref_topologies_and_axes_2_2_1);

    topologies_and_axes_type ref_topologies_and_axes_2_2_2 =
        std::make_tuple(topologies_type{ref_topo2}, vec_axis_type{});
    EXPECT_EQ(topologies_and_axes_2_2_2, ref_topologies_and_axes_2_2_2);
  }
}

void test_get_all_pencil_topologies2D_3DView(std::size_t nprocs) {
  using topology_type   = std::array<std::size_t, 3>;
  using topologies_type = std::vector<topology_type>;
  using topology_r_type = Topology<std::size_t, 3, Kokkos::LayoutRight>;
  using topology_l_type = Topology<std::size_t, 3, Kokkos::LayoutLeft>;
  using vec_axis_type   = std::vector<std::size_t>;
  using topologies_and_axes_type = std::tuple<topologies_type, vec_axis_type>;
  std::size_t np0                = 4;

  topology_r_type topology0 = {1, nprocs, np0}, topology1 = {nprocs, 1, np0},
                  topology3 = {nprocs, np0, 1};
  topology_l_type topology2 = {np0, nprocs, 1}, topology4 = {np0, 1, nprocs},
                  topology5 = {1, np0, nprocs};

  topology_type ref_topo0 = topology0.array(), ref_topo1 = topology1.array(),
                ref_topo2 = topology2.array(), ref_topo3 = topology3.array(),
                ref_topo4 = topology4.array(), ref_topo5 = topology5.array();

  using axes_type  = std::array<int, 2>;
  axes_type axes01 = {0, 1}, axes02 = {0, 2}, axes10 = {1, 0}, axes12 = {1, 2},
            axes20 = {2, 0}, axes21 = {2, 1};

  std::vector<axes_type> all_axes = {axes01, axes02, axes10,
                                     axes12, axes20, axes21};

  if (nprocs == 1) {
    for (const auto& axes : all_axes) {
      // Failure tests because only two elements differ (slabs)
      EXPECT_THROW(
          {
            [[maybe_unused]] auto topologies_and_axes_0_1 =
                get_all_pencil_topologies(topology0, topology1, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto topologies_and_axes_0_2 =
                get_all_pencil_topologies(topology0, topology2, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto topologies_and_axes_1_0 =
                get_all_pencil_topologies(topology1, topology0, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto topologies_and_axes_2_0 =
                get_all_pencil_topologies(topology2, topology0, axes);
          },
          std::runtime_error);
    }
  } else {
    // topology0 to topology0
    auto topologies_and_axes_0_0_01 =
        get_all_pencil_topologies(topology0, topology0, axes01);
    auto topologies_and_axes_0_0_02 =
        get_all_pencil_topologies(topology0, topology0, axes02);
    auto topologies_and_axes_0_0_10 =
        get_all_pencil_topologies(topology0, topology0, axes10);
    auto topologies_and_axes_0_0_12 =
        get_all_pencil_topologies(topology0, topology0, axes12);
    auto topologies_and_axes_0_0_20 =
        get_all_pencil_topologies(topology0, topology0, axes20);
    auto topologies_and_axes_0_0_21 =
        get_all_pencil_topologies(topology0, topology0, axes21);
    topologies_and_axes_type ref_topologies_and_axes_0_0_01 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0}, vec_axis_type{0, 0});
    EXPECT_EQ(topologies_and_axes_0_0_01, ref_topologies_and_axes_0_0_01);

    topologies_and_axes_type ref_topologies_and_axes_0_0_02 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2, ref_topo0}, vec_axis_type{1, 1});
    EXPECT_EQ(topologies_and_axes_0_0_02, ref_topologies_and_axes_0_0_02);

    topologies_and_axes_type ref_topologies_and_axes_0_0_10 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0}, vec_axis_type{0, 0});
    EXPECT_EQ(topologies_and_axes_0_0_10, ref_topologies_and_axes_0_0_10);

    topologies_and_axes_type ref_topologies_and_axes_0_0_12 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2, ref_topo4, ref_topo2, ref_topo0},
        vec_axis_type{1, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_0_0_12, ref_topologies_and_axes_0_0_12);

    topologies_and_axes_type ref_topologies_and_axes_0_0_20 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2, ref_topo0}, vec_axis_type{1, 1});
    EXPECT_EQ(topologies_and_axes_0_0_20, ref_topologies_and_axes_0_0_20);

    topologies_and_axes_type ref_topologies_and_axes_0_0_21 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo3, ref_topo1, ref_topo0},
        vec_axis_type{0, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_0_0_21, ref_topologies_and_axes_0_0_21);

    // topology0 to topology1
    auto topologies_and_axes_0_1_01 =
        get_all_pencil_topologies(topology0, topology1, axes01);
    auto topologies_and_axes_0_1_02 =
        get_all_pencil_topologies(topology0, topology1, axes02);
    auto topologies_and_axes_0_1_10 =
        get_all_pencil_topologies(topology0, topology1, axes10);
    auto topologies_and_axes_0_1_12 =
        get_all_pencil_topologies(topology0, topology1, axes12);
    auto topologies_and_axes_0_1_20 =
        get_all_pencil_topologies(topology0, topology1, axes20);
    auto topologies_and_axes_0_1_21 =
        get_all_pencil_topologies(topology0, topology1, axes21);

    topologies_and_axes_type ref_topologies_and_axes_0_1_01 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo1},
        vec_axis_type{0, 0, 0});
    EXPECT_EQ(topologies_and_axes_0_1_01, ref_topologies_and_axes_0_1_01);

    topologies_and_axes_type ref_topologies_and_axes_0_1_02 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2, ref_topo0, ref_topo1},
        vec_axis_type{1, 1, 0});
    EXPECT_EQ(topologies_and_axes_0_1_02, ref_topologies_and_axes_0_1_02);

    topologies_and_axes_type ref_topologies_and_axes_0_1_10 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1}, vec_axis_type{0});
    EXPECT_EQ(topologies_and_axes_0_1_10, ref_topologies_and_axes_0_1_10);

    topologies_and_axes_type ref_topologies_and_axes_0_1_12 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2, ref_topo0, ref_topo1},
        vec_axis_type{1, 1, 0});
    EXPECT_EQ(topologies_and_axes_0_1_12, ref_topologies_and_axes_0_1_12);

    topologies_and_axes_type ref_topologies_and_axes_0_1_20 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2, ref_topo0, ref_topo1},
        vec_axis_type{1, 1, 0});
    EXPECT_EQ(topologies_and_axes_0_1_20, ref_topologies_and_axes_0_1_20);

    topologies_and_axes_type ref_topologies_and_axes_0_1_21 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo3, ref_topo1},
        vec_axis_type{0, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_1_21, ref_topologies_and_axes_0_1_21);

    // topology0 to topology2
    auto topologies_and_axes_0_2_02 =
        get_all_pencil_topologies(topology0, topology2, axes02);
    auto topologies_and_axes_0_2_01 =
        get_all_pencil_topologies(topology0, topology2, axes01);
    auto topologies_and_axes_0_2_10 =
        get_all_pencil_topologies(topology0, topology2, axes10);
    auto topologies_and_axes_0_2_12 =
        get_all_pencil_topologies(topology0, topology2, axes12);
    auto topologies_and_axes_0_2_20 =
        get_all_pencil_topologies(topology0, topology2, axes20);
    auto topologies_and_axes_0_2_21 =
        get_all_pencil_topologies(topology0, topology2, axes21);

    topologies_and_axes_type ref_topologies_and_axes_0_2_01 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{0, 0, 1});
    EXPECT_EQ(topologies_and_axes_0_2_01, ref_topologies_and_axes_0_2_01);

    topologies_and_axes_type ref_topologies_and_axes_0_2_02 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2, ref_topo0, ref_topo2},
        vec_axis_type{1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_2_02, ref_topologies_and_axes_0_2_02);

    topologies_and_axes_type ref_topologies_and_axes_0_2_10 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{0, 0, 1});
    EXPECT_EQ(topologies_and_axes_0_2_10, ref_topologies_and_axes_0_2_10);

    topologies_and_axes_type ref_topologies_and_axes_0_2_12 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2, ref_topo4, ref_topo2},
        vec_axis_type{1, 0, 0});
    EXPECT_EQ(topologies_and_axes_0_2_12, ref_topologies_and_axes_0_2_12);

    topologies_and_axes_type ref_topologies_and_axes_0_2_20 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2}, vec_axis_type{1});
    EXPECT_EQ(topologies_and_axes_0_2_20, ref_topologies_and_axes_0_2_20);

    topologies_and_axes_type ref_topologies_and_axes_0_2_21 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{0, 0, 1});
    EXPECT_EQ(topologies_and_axes_0_2_21, ref_topologies_and_axes_0_2_21);

    // topology1 to topology0
    auto topologies_and_axes_1_0_01 =
        get_all_pencil_topologies(topology1, topology0, axes01);
    auto topologies_and_axes_1_0_02 =
        get_all_pencil_topologies(topology1, topology0, axes02);
    auto topologies_and_axes_1_0_10 =
        get_all_pencil_topologies(topology1, topology0, axes10);
    auto topologies_and_axes_1_0_12 =
        get_all_pencil_topologies(topology1, topology0, axes12);
    auto topologies_and_axes_1_0_20 =
        get_all_pencil_topologies(topology1, topology0, axes20);
    auto topologies_and_axes_1_0_21 =
        get_all_pencil_topologies(topology1, topology0, axes21);

    topologies_and_axes_type ref_topologies_and_axes_1_0_01 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0}, vec_axis_type{0});
    EXPECT_EQ(topologies_and_axes_1_0_01, ref_topologies_and_axes_1_0_01);

    topologies_and_axes_type ref_topologies_and_axes_1_0_02 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo3, ref_topo1, ref_topo0},
        vec_axis_type{1, 1, 0});
    EXPECT_EQ(topologies_and_axes_1_0_02, ref_topologies_and_axes_1_0_02);

    topologies_and_axes_type ref_topologies_and_axes_1_0_10 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo1, ref_topo0},
        vec_axis_type{0, 0, 0});
    EXPECT_EQ(topologies_and_axes_1_0_10, ref_topologies_and_axes_1_0_10);

    topologies_and_axes_type ref_topologies_and_axes_1_0_12 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo3, ref_topo1, ref_topo0},
        vec_axis_type{1, 1, 0});
    EXPECT_EQ(topologies_and_axes_1_0_12, ref_topologies_and_axes_1_0_12);

    topologies_and_axes_type ref_topologies_and_axes_1_0_20 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo2, ref_topo0},
        vec_axis_type{0, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_0_20, ref_topologies_and_axes_1_0_20);

    topologies_and_axes_type ref_topologies_and_axes_1_0_21 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo3, ref_topo1, ref_topo0},
        vec_axis_type{1, 1, 0});
    EXPECT_EQ(topologies_and_axes_1_0_21, ref_topologies_and_axes_1_0_21);

    // topology1 to topology1
    auto topologies_and_axes_1_1_01 =
        get_all_pencil_topologies(topology1, topology1, axes01);
    auto topologies_and_axes_1_1_02 =
        get_all_pencil_topologies(topology1, topology1, axes02);
    auto topologies_and_axes_1_1_10 =
        get_all_pencil_topologies(topology1, topology1, axes10);
    auto topologies_and_axes_1_1_12 =
        get_all_pencil_topologies(topology1, topology1, axes12);
    auto topologies_and_axes_1_1_20 =
        get_all_pencil_topologies(topology1, topology1, axes20);
    auto topologies_and_axes_1_1_21 =
        get_all_pencil_topologies(topology1, topology1, axes21);

    topologies_and_axes_type ref_topologies_and_axes_1_1_01 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo1}, vec_axis_type{0, 0});
    EXPECT_EQ(topologies_and_axes_1_1_01, ref_topologies_and_axes_1_1_01);

    topologies_and_axes_type ref_topologies_and_axes_1_1_02 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo3, ref_topo5, ref_topo3, ref_topo1},
        vec_axis_type{1, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_1_1_02, ref_topologies_and_axes_1_1_02);

    topologies_and_axes_type ref_topologies_and_axes_1_1_10 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo1}, vec_axis_type{0, 0});
    EXPECT_EQ(topologies_and_axes_1_1_10, ref_topologies_and_axes_1_1_10);

    topologies_and_axes_type ref_topologies_and_axes_1_1_12 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo3, ref_topo1}, vec_axis_type{1, 1});
    EXPECT_EQ(topologies_and_axes_1_1_12, ref_topologies_and_axes_1_1_12);

    topologies_and_axes_type ref_topologies_and_axes_1_1_20 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo2, ref_topo0, ref_topo1},
        vec_axis_type{0, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_1_1_20, ref_topologies_and_axes_1_1_20);

    topologies_and_axes_type ref_topologies_and_axes_1_1_21 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo3, ref_topo1}, vec_axis_type{1, 1});
    EXPECT_EQ(topologies_and_axes_1_1_21, ref_topologies_and_axes_1_1_21);

    // topology1 to topology2
    auto topologies_and_axes_1_2_01 =
        get_all_pencil_topologies(topology1, topology2, axes01);
    auto topologies_and_axes_1_2_02 =
        get_all_pencil_topologies(topology1, topology2, axes02);
    auto topologies_and_axes_1_2_10 =
        get_all_pencil_topologies(topology1, topology2, axes10);
    auto topologies_and_axes_1_2_12 =
        get_all_pencil_topologies(topology1, topology2, axes12);
    auto topologies_and_axes_1_2_20 =
        get_all_pencil_topologies(topology1, topology2, axes20);
    auto topologies_and_axes_1_2_21 =
        get_all_pencil_topologies(topology1, topology2, axes21);

    topologies_and_axes_type ref_topologies_and_axes_1_2_01 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo2}, vec_axis_type{0, 1});
    EXPECT_EQ(topologies_and_axes_1_2_01, ref_topologies_and_axes_1_2_01);

    topologies_and_axes_type ref_topologies_and_axes_1_2_02 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo3, ref_topo5, ref_topo4, ref_topo2},
        vec_axis_type{1, 0, 1, 0});
    EXPECT_EQ(topologies_and_axes_1_2_02, ref_topologies_and_axes_1_2_02);

    topologies_and_axes_type ref_topologies_and_axes_1_2_10 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{0, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_1_2_10, ref_topologies_and_axes_1_2_10);

    topologies_and_axes_type ref_topologies_and_axes_1_2_12 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo3, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{1, 1, 0, 1});
    EXPECT_EQ(topologies_and_axes_1_2_12, ref_topologies_and_axes_1_2_12);

    topologies_and_axes_type ref_topologies_and_axes_1_2_20 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo2}, vec_axis_type{0, 1});
    EXPECT_EQ(topologies_and_axes_1_2_20, ref_topologies_and_axes_1_2_20);

    topologies_and_axes_type ref_topologies_and_axes_1_2_21 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo2}, vec_axis_type{0, 1});
    EXPECT_EQ(topologies_and_axes_1_2_21, ref_topologies_and_axes_1_2_21);

    // topology2 to topology0
    auto topologies_and_axes_2_0_01 =
        get_all_pencil_topologies(topology2, topology0, axes01);
    auto topologies_and_axes_2_0_02 =
        get_all_pencil_topologies(topology2, topology0, axes02);
    auto topologies_and_axes_2_0_10 =
        get_all_pencil_topologies(topology2, topology0, axes10);
    auto topologies_and_axes_2_0_12 =
        get_all_pencil_topologies(topology2, topology0, axes12);
    auto topologies_and_axes_2_0_20 =
        get_all_pencil_topologies(topology2, topology0, axes20);
    auto topologies_and_axes_2_0_21 =
        get_all_pencil_topologies(topology2, topology0, axes21);

    topologies_and_axes_type ref_topologies_and_axes_2_0_01 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo2, ref_topo0},
        vec_axis_type{0, 0, 1});
    EXPECT_EQ(topologies_and_axes_2_0_01, ref_topologies_and_axes_2_0_01);

    topologies_and_axes_type ref_topologies_and_axes_2_0_02 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0}, vec_axis_type{1});
    EXPECT_EQ(topologies_and_axes_2_0_02, ref_topologies_and_axes_2_0_02);

    topologies_and_axes_type ref_topologies_and_axes_2_0_10 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0, ref_topo1, ref_topo0},
        vec_axis_type{1, 0, 0});
    EXPECT_EQ(topologies_and_axes_2_0_10, ref_topologies_and_axes_2_0_10);

    topologies_and_axes_type ref_topologies_and_axes_2_0_12 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo2, ref_topo0},
        vec_axis_type{0, 0, 1});
    EXPECT_EQ(topologies_and_axes_2_0_12, ref_topologies_and_axes_2_0_12);

    topologies_and_axes_type ref_topologies_and_axes_2_0_20 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0, ref_topo2, ref_topo0},
        vec_axis_type{1, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_0_20, ref_topologies_and_axes_2_0_20);

    topologies_and_axes_type ref_topologies_and_axes_2_0_21 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo2, ref_topo0},
        vec_axis_type{0, 0, 1});
    EXPECT_EQ(topologies_and_axes_2_0_21, ref_topologies_and_axes_2_0_21);

    // topology2 to topology1
    auto topologies_and_axes_2_1_01 =
        get_all_pencil_topologies(topology2, topology1, axes01);
    auto topologies_and_axes_2_1_02 =
        get_all_pencil_topologies(topology2, topology1, axes02);
    auto topologies_and_axes_2_1_10 =
        get_all_pencil_topologies(topology2, topology1, axes10);
    auto topologies_and_axes_2_1_12 =
        get_all_pencil_topologies(topology2, topology1, axes12);
    auto topologies_and_axes_2_1_20 =
        get_all_pencil_topologies(topology2, topology1, axes20);
    auto topologies_and_axes_2_1_21 =
        get_all_pencil_topologies(topology2, topology1, axes21);

    topologies_and_axes_type ref_topologies_and_axes_2_1_01 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo5, ref_topo3, ref_topo1},
        vec_axis_type{0, 1, 0, 1});
    EXPECT_EQ(topologies_and_axes_2_1_01, ref_topologies_and_axes_2_1_01);

    topologies_and_axes_type ref_topologies_and_axes_2_1_02 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0, ref_topo1}, vec_axis_type{1, 0});
    EXPECT_EQ(topologies_and_axes_2_1_02, ref_topologies_and_axes_2_1_02);

    topologies_and_axes_type ref_topologies_and_axes_2_1_10 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0, ref_topo1}, vec_axis_type{1, 0});
    EXPECT_EQ(topologies_and_axes_2_1_10, ref_topologies_and_axes_2_1_10);

    topologies_and_axes_type ref_topologies_and_axes_2_1_12 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0, ref_topo1}, vec_axis_type{1, 0});
    EXPECT_EQ(topologies_and_axes_2_1_12, ref_topologies_and_axes_2_1_12);

    topologies_and_axes_type ref_topologies_and_axes_2_1_20 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0, ref_topo2, ref_topo0, ref_topo1},
        vec_axis_type{1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_2_1_20, ref_topologies_and_axes_2_1_20);

    topologies_and_axes_type ref_topologies_and_axes_2_1_21 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo2, ref_topo0, ref_topo1},
        vec_axis_type{0, 0, 1, 0});
    EXPECT_EQ(topologies_and_axes_2_1_21, ref_topologies_and_axes_2_1_21);

    // topology2 to topology2
    auto topologies_and_axes_2_2_01 =
        get_all_pencil_topologies(topology2, topology2, axes01);
    auto topologies_and_axes_2_2_02 =
        get_all_pencil_topologies(topology2, topology2, axes02);
    auto topologies_and_axes_2_2_10 =
        get_all_pencil_topologies(topology2, topology2, axes10);
    auto topologies_and_axes_2_2_12 =
        get_all_pencil_topologies(topology2, topology2, axes12);
    auto topologies_and_axes_2_2_20 =
        get_all_pencil_topologies(topology2, topology2, axes20);
    auto topologies_and_axes_2_2_21 =
        get_all_pencil_topologies(topology2, topology2, axes21);

    topologies_and_axes_type ref_topologies_and_axes_2_2_01 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo5, ref_topo4, ref_topo2},
        vec_axis_type{0, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_2_2_01, ref_topologies_and_axes_2_2_01);

    topologies_and_axes_type ref_topologies_and_axes_2_2_02 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0, ref_topo2}, vec_axis_type{1, 1});
    EXPECT_EQ(topologies_and_axes_2_2_02, ref_topologies_and_axes_2_2_02);

    topologies_and_axes_type ref_topologies_and_axes_2_2_10 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{1, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_2_2_10, ref_topologies_and_axes_2_2_10);

    topologies_and_axes_type ref_topologies_and_axes_2_2_12 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo2}, vec_axis_type{0, 0});
    EXPECT_EQ(topologies_and_axes_2_2_12, ref_topologies_and_axes_2_2_12);

    topologies_and_axes_type ref_topologies_and_axes_2_2_20 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0, ref_topo2}, vec_axis_type{1, 1});
    EXPECT_EQ(topologies_and_axes_2_2_20, ref_topologies_and_axes_2_2_20);

    topologies_and_axes_type ref_topologies_and_axes_2_2_21 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo2}, vec_axis_type{0, 0});
    EXPECT_EQ(topologies_and_axes_2_2_21, ref_topologies_and_axes_2_2_21);
  }
}
*/

void test_get_all_pencil_topologies3D_3DView(std::size_t nprocs) {
  using topology_type   = std::array<std::size_t, 3>;
  using topologies_type = std::vector<topology_type>;
  using topology_r_type = Topology<std::size_t, 3, Kokkos::LayoutRight>;
  using topology_l_type = Topology<std::size_t, 3, Kokkos::LayoutLeft>;
  using vec_axis_type   = std::vector<std::size_t>;
  using layouts_type    = std::vector<std::size_t>;
  std::size_t np0       = 4;

  topology_r_type topology0 = {1, nprocs, np0}, topology1 = {nprocs, 1, np0},
                  topology3 = {nprocs, np0, 1};
  topology_l_type topology2 = {np0, nprocs, 1}, topology4 = {np0, 1, nprocs},
                  topology5 = {1, np0, nprocs};

  topology_type ref_topo0 = topology0.array(), ref_topo1 = topology1.array(),
                ref_topo2 = topology2.array(), ref_topo3 = topology3.array(),
                ref_topo4 = topology4.array(), ref_topo5 = topology5.array();

  using axes_type   = std::array<int, 3>;
  axes_type axes012 = {0, 1, 2}, axes021 = {0, 2, 1}, axes102 = {1, 0, 2},
            axes120 = {1, 2, 0}, axes201 = {2, 0, 1}, axes210 = {2, 1, 0};

  std::vector<axes_type> all_axes = {axes012, axes021, axes102,
                                     axes120, axes201, axes210};

  if (nprocs == 1) {
    for (const auto& axes : all_axes) {
      // Failure tests because only two elements differ (slabs)
      EXPECT_THROW(
          {
            [[maybe_unused]] auto topologies_and_axes_0_1 =
                get_all_pencil_topologies(topology0, topology1, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto topologies_and_axes_0_2 =
                get_all_pencil_topologies(topology0, topology2, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto topologies_and_axes_1_0 =
                get_all_pencil_topologies(topology1, topology0, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto topologies_and_axes_2_0 =
                get_all_pencil_topologies(topology2, topology0, axes);
          },
          std::runtime_error);
    }
  } else {
    // topology0 to topology0
    auto topologies_and_axes_0_0_012 =
        get_all_pencil_topologies(topology0, topology0, axes012);
    auto topologies_and_axes_0_0_021 =
        get_all_pencil_topologies(topology0, topology0, axes021);
    auto topologies_and_axes_0_0_102 =
        get_all_pencil_topologies(topology0, topology0, axes102);
    auto topologies_and_axes_0_0_120 =
        get_all_pencil_topologies(topology0, topology0, axes120);
    auto topologies_and_axes_0_0_201 =
        get_all_pencil_topologies(topology0, topology0, axes201);
    auto topologies_and_axes_0_0_210 =
        get_all_pencil_topologies(topology0, topology0, axes210);

    auto ref_topologies_and_axes_0_0_012 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2, ref_topo4, ref_topo2, ref_topo0},
        vec_axis_type{1, 0, 0, 1}, layouts_type{1, 0, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_0_0_012, ref_topologies_and_axes_0_0_012);

    auto ref_topologies_and_axes_0_0_021 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo3, ref_topo1, ref_topo0},
        vec_axis_type{0, 1, 1, 0}, layouts_type{1, 1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_0_021, ref_topologies_and_axes_0_0_021);

    auto ref_topologies_and_axes_0_0_102 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2, ref_topo0, ref_topo1, ref_topo0},
        vec_axis_type{1, 1, 0, 0}, layouts_type{1, 0, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_0_102, ref_topologies_and_axes_0_0_102);

    auto ref_topologies_and_axes_0_0_120 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2, ref_topo4, ref_topo2, ref_topo0},
        vec_axis_type{1, 0, 0, 1}, layouts_type{1, 0, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_0_0_120, ref_topologies_and_axes_0_0_120);

    auto ref_topologies_and_axes_0_0_201 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo2, ref_topo0},
        vec_axis_type{0, 0, 1, 1}, layouts_type{1, 1, 1, 0, 1});
    EXPECT_EQ(topologies_and_axes_0_0_201, ref_topologies_and_axes_0_0_201);

    auto ref_topologies_and_axes_0_0_210 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo3, ref_topo1, ref_topo0},
        vec_axis_type{0, 1, 1, 0}, layouts_type{1, 1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_0_210, ref_topologies_and_axes_0_0_210);

    // topology0 to topology1
    auto topologies_and_axes_0_1_012 =
        get_all_pencil_topologies(topology0, topology1, axes012);
    auto topologies_and_axes_0_1_021 =
        get_all_pencil_topologies(topology0, topology1, axes021);
    auto topologies_and_axes_0_1_102 =
        get_all_pencil_topologies(topology0, topology1, axes102);
    auto topologies_and_axes_0_1_120 =
        get_all_pencil_topologies(topology0, topology1, axes120);
    auto topologies_and_axes_0_1_201 =
        get_all_pencil_topologies(topology0, topology1, axes201);
    auto topologies_and_axes_0_1_210 =
        get_all_pencil_topologies(topology0, topology1, axes210);

    auto ref_topologies_and_axes_0_1_012 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2, ref_topo4, ref_topo5, ref_topo3,
                        ref_topo1},
        vec_axis_type{1, 0, 1, 0, 1}, layouts_type{1, 0, 0, 0, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_1_012, ref_topologies_and_axes_0_1_012);

    auto ref_topologies_and_axes_0_1_021 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo3, ref_topo5, ref_topo3,
                        ref_topo1},
        vec_axis_type{0, 1, 0, 0, 1}, layouts_type{1, 1, 1, 0, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_1_021, ref_topologies_and_axes_0_1_021);

    auto ref_topologies_and_axes_0_1_102 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2, ref_topo0, ref_topo1},
        vec_axis_type{1, 1, 0}, layouts_type{1, 0, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_1_102, ref_topologies_and_axes_0_1_102);

    auto ref_topologies_and_axes_0_1_120 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2, ref_topo0, ref_topo1},
        vec_axis_type{1, 1, 0}, layouts_type{1, 0, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_1_120, ref_topologies_and_axes_0_1_120);

    auto ref_topologies_and_axes_0_1_201 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo2, ref_topo0,
                        ref_topo1},
        vec_axis_type{0, 0, 1, 1, 0}, layouts_type{1, 1, 1, 0, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_1_201, ref_topologies_and_axes_0_1_201);

    auto ref_topologies_and_axes_0_1_210 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo3, ref_topo1},
        vec_axis_type{0, 1, 1}, layouts_type{1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_1_210, ref_topologies_and_axes_0_1_210);

    // topology0 to topology2
    auto topologies_and_axes_0_2_012 =
        get_all_pencil_topologies(topology0, topology2, axes012);
    auto topologies_and_axes_0_2_021 =
        get_all_pencil_topologies(topology0, topology2, axes021);
    auto topologies_and_axes_0_2_102 =
        get_all_pencil_topologies(topology0, topology2, axes102);
    auto topologies_and_axes_0_2_120 =
        get_all_pencil_topologies(topology0, topology2, axes120);
    auto topologies_and_axes_0_2_201 =
        get_all_pencil_topologies(topology0, topology2, axes201);
    auto topologies_and_axes_0_2_210 =
        get_all_pencil_topologies(topology0, topology2, axes210);

    auto ref_topologies_and_axes_0_2_012 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2, ref_topo4, ref_topo5, ref_topo4,
                        ref_topo2},
        vec_axis_type{1, 0, 1, 1, 0}, layouts_type{1, 0, 0, 0, 0, 0});
    EXPECT_EQ(topologies_and_axes_0_2_012, ref_topologies_and_axes_0_2_012);

    auto ref_topologies_and_axes_0_2_021 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo3, ref_topo5, ref_topo4,
                        ref_topo2},
        vec_axis_type{0, 1, 0, 1, 0}, layouts_type{1, 1, 1, 0, 0, 0});

    EXPECT_EQ(topologies_and_axes_0_2_021, ref_topologies_and_axes_0_2_021);

    auto ref_topologies_and_axes_0_2_102 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2, ref_topo0, ref_topo1, ref_topo0,
                        ref_topo2},
        vec_axis_type{1, 1, 0, 0, 1}, layouts_type{1, 0, 1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_0_2_102, ref_topologies_and_axes_0_2_102);

    auto ref_topologies_and_axes_0_2_120 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2, ref_topo4, ref_topo2},
        vec_axis_type{1, 0, 0}, layouts_type{1, 0, 0, 0});
    EXPECT_EQ(topologies_and_axes_0_2_120, ref_topologies_and_axes_0_2_120);

    auto ref_topologies_and_axes_0_2_201 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{0, 0, 1}, layouts_type{1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_0_2_201, ref_topologies_and_axes_0_2_201);

    auto ref_topologies_and_axes_0_2_210 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{0, 0, 1}, layouts_type{1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_0_2_210, ref_topologies_and_axes_0_2_210);

    // topology0 to topology3
    auto topologies_and_axes_0_3_012 =
        get_all_pencil_topologies(topology0, topology3, axes012);
    auto topologies_and_axes_0_3_021 =
        get_all_pencil_topologies(topology0, topology3, axes021);
    auto topologies_and_axes_0_3_102 =
        get_all_pencil_topologies(topology0, topology3, axes102);
    auto topologies_and_axes_0_3_120 =
        get_all_pencil_topologies(topology0, topology3, axes120);
    auto topologies_and_axes_0_3_201 =
        get_all_pencil_topologies(topology0, topology3, axes201);
    auto topologies_and_axes_0_3_210 =
        get_all_pencil_topologies(topology0, topology3, axes210);

    auto ref_topologies_and_axes_0_3_012 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2, ref_topo4, ref_topo5, ref_topo3},
        vec_axis_type{1, 0, 1, 0}, layouts_type{1, 0, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_0_3_012, ref_topologies_and_axes_0_3_012);

    auto ref_topologies_and_axes_0_3_021 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo3, ref_topo5, ref_topo3},
        vec_axis_type{0, 1, 0, 0}, layouts_type{1, 1, 1, 0, 1});

    EXPECT_EQ(topologies_and_axes_0_3_021, ref_topologies_and_axes_0_3_021);

    auto ref_topologies_and_axes_0_3_102 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2, ref_topo0, ref_topo1, ref_topo3},
        vec_axis_type{1, 1, 0, 1}, layouts_type{1, 0, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_3_102, ref_topologies_and_axes_0_3_102);

    auto ref_topologies_and_axes_0_3_120 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2, ref_topo4, ref_topo5, ref_topo3},
        vec_axis_type{1, 0, 1, 0}, layouts_type{1, 0, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_0_3_120, ref_topologies_and_axes_0_3_120);

    auto ref_topologies_and_axes_0_3_201 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo1, ref_topo3},
        vec_axis_type{0, 0, 0, 1}, layouts_type{1, 1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_3_201, ref_topologies_and_axes_0_3_201);

    auto ref_topologies_and_axes_0_3_210 =
        std::make_tuple(topologies_type{ref_topo0, ref_topo1, ref_topo3},
                        vec_axis_type{0, 1}, layouts_type{1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_3_210, ref_topologies_and_axes_0_3_210);

    // topology1 to topology0
    auto topologies_and_axes_1_0_012 =
        get_all_pencil_topologies(topology1, topology0, axes012);
    auto topologies_and_axes_1_0_021 =
        get_all_pencil_topologies(topology1, topology0, axes021);
    auto topologies_and_axes_1_0_102 =
        get_all_pencil_topologies(topology1, topology0, axes102);
    auto topologies_and_axes_1_0_120 =
        get_all_pencil_topologies(topology1, topology0, axes120);
    auto topologies_and_axes_1_0_201 =
        get_all_pencil_topologies(topology1, topology0, axes201);
    auto topologies_and_axes_1_0_210 =
        get_all_pencil_topologies(topology1, topology0, axes210);

    auto ref_topologies_and_axes_1_0_012 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo3, ref_topo1, ref_topo0},
        vec_axis_type{1, 1, 0}, layouts_type{1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_0_012, ref_topologies_and_axes_1_0_012);

    auto ref_topologies_and_axes_1_0_021 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo3, ref_topo1, ref_topo0},
        vec_axis_type{1, 1, 0}, layouts_type{1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_0_021, ref_topologies_and_axes_1_0_021);

    auto ref_topologies_and_axes_1_0_102 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo3, ref_topo5, ref_topo4, ref_topo2,
                        ref_topo0},
        vec_axis_type{1, 0, 1, 0, 1}, layouts_type{1, 1, 0, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_1_0_102, ref_topologies_and_axes_1_0_102);

    auto ref_topologies_and_axes_1_0_120 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo2, ref_topo4, ref_topo2,
                        ref_topo0},
        vec_axis_type{0, 1, 0, 0, 1}, layouts_type{1, 1, 0, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_1_0_120, ref_topologies_and_axes_1_0_120);

    auto ref_topologies_and_axes_1_0_201 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo2, ref_topo0},
        vec_axis_type{0, 1, 1}, layouts_type{1, 1, 0, 1});
    EXPECT_EQ(topologies_and_axes_1_0_201, ref_topologies_and_axes_1_0_201);

    auto ref_topologies_and_axes_1_0_210 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo1, ref_topo3, ref_topo1,
                        ref_topo0},
        vec_axis_type{0, 0, 1, 1, 0}, layouts_type{1, 1, 1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_0_210, ref_topologies_and_axes_1_0_210);

    // topology1 to topology1
    auto topologies_and_axes_1_1_012 =
        get_all_pencil_topologies(topology1, topology1, axes012);
    auto topologies_and_axes_1_1_021 =
        get_all_pencil_topologies(topology1, topology1, axes021);
    auto topologies_and_axes_1_1_102 =
        get_all_pencil_topologies(topology1, topology1, axes102);
    auto topologies_and_axes_1_1_120 =
        get_all_pencil_topologies(topology1, topology1, axes120);
    auto topologies_and_axes_1_1_201 =
        get_all_pencil_topologies(topology1, topology1, axes201);
    auto topologies_and_axes_1_1_210 =
        get_all_pencil_topologies(topology1, topology1, axes210);

    auto ref_topologies_and_axes_1_1_012 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo3, ref_topo1, ref_topo0, ref_topo1},
        vec_axis_type{1, 1, 0, 0}, layouts_type{1, 1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_1_012, ref_topologies_and_axes_1_1_012);

    auto ref_topologies_and_axes_1_1_021 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo3, ref_topo5, ref_topo3, ref_topo1},
        vec_axis_type{1, 0, 0, 1}, layouts_type{1, 1, 0, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_1_021, ref_topologies_and_axes_1_1_021);

    auto ref_topologies_and_axes_1_1_102 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo3, ref_topo5, ref_topo3, ref_topo1},
        vec_axis_type{1, 0, 0, 1}, layouts_type{1, 1, 0, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_1_102, ref_topologies_and_axes_1_1_102);

    auto ref_topologies_and_axes_1_1_120 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo2, ref_topo0, ref_topo1},
        vec_axis_type{0, 1, 1, 0}, layouts_type{1, 1, 0, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_1_120, ref_topologies_and_axes_1_1_120);

    auto ref_topologies_and_axes_1_1_201 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo2, ref_topo0, ref_topo1},
        vec_axis_type{0, 1, 1, 0}, layouts_type{1, 1, 0, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_1_201, ref_topologies_and_axes_1_1_201);

    auto ref_topologies_and_axes_1_1_210 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo1, ref_topo3, ref_topo1},
        vec_axis_type{0, 0, 1, 1}, layouts_type{1, 1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_1_210, ref_topologies_and_axes_1_1_210);

    // topology1 to topology2
    auto topologies_and_axes_1_2_012 =
        get_all_pencil_topologies(topology1, topology2, axes012);
    auto topologies_and_axes_1_2_021 =
        get_all_pencil_topologies(topology1, topology2, axes021);
    auto topologies_and_axes_1_2_102 =
        get_all_pencil_topologies(topology1, topology2, axes102);
    auto topologies_and_axes_1_2_120 =
        get_all_pencil_topologies(topology1, topology2, axes120);
    auto topologies_and_axes_1_2_201 =
        get_all_pencil_topologies(topology1, topology2, axes201);
    auto topologies_and_axes_1_2_210 =
        get_all_pencil_topologies(topology1, topology2, axes210);

    auto ref_topologies_and_axes_1_2_012 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo3, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{1, 1, 0, 1}, layouts_type{1, 1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_1_2_012, ref_topologies_and_axes_1_2_012);

    auto ref_topologies_and_axes_1_2_021 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo3, ref_topo5, ref_topo4, ref_topo2},
        vec_axis_type{1, 0, 1, 0}, layouts_type{1, 1, 0, 0, 0});
    EXPECT_EQ(topologies_and_axes_1_2_021, ref_topologies_and_axes_1_2_021);

    auto ref_topologies_and_axes_1_2_102 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo3, ref_topo5, ref_topo4, ref_topo2},
        vec_axis_type{1, 0, 1, 0}, layouts_type{1, 1, 0, 0, 0});
    EXPECT_EQ(topologies_and_axes_1_2_102, ref_topologies_and_axes_1_2_102);

    auto ref_topologies_and_axes_1_2_120 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo2, ref_topo4, ref_topo2},
        vec_axis_type{0, 1, 0, 0}, layouts_type{1, 1, 0, 0, 0});
    EXPECT_EQ(topologies_and_axes_1_2_120, ref_topologies_and_axes_1_2_120);

    auto ref_topologies_and_axes_1_2_201 =
        std::make_tuple(topologies_type{ref_topo1, ref_topo0, ref_topo2},
                        vec_axis_type{0, 1}, layouts_type{1, 1, 0});
    EXPECT_EQ(topologies_and_axes_1_2_201, ref_topologies_and_axes_1_2_201);

    auto ref_topologies_and_axes_1_2_210 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{0, 0, 0, 1}, layouts_type{1, 1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_1_2_210, ref_topologies_and_axes_1_2_210);

    // topology2 to topology0
    auto topologies_and_axes_2_0_012 =
        get_all_pencil_topologies(topology2, topology0, axes012);
    auto topologies_and_axes_2_0_021 =
        get_all_pencil_topologies(topology2, topology0, axes021);
    auto topologies_and_axes_2_0_102 =
        get_all_pencil_topologies(topology2, topology0, axes102);
    auto topologies_and_axes_2_0_120 =
        get_all_pencil_topologies(topology2, topology0, axes120);
    auto topologies_and_axes_2_0_201 =
        get_all_pencil_topologies(topology2, topology0, axes201);
    auto topologies_and_axes_2_0_210 =
        get_all_pencil_topologies(topology2, topology0, axes210);

    auto ref_topologies_and_axes_2_0_012 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo2, ref_topo0},
        vec_axis_type{0, 0, 1}, layouts_type{0, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_2_0_012, ref_topologies_and_axes_2_0_012);

    auto ref_topologies_and_axes_2_0_021 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo2, ref_topo0},
        vec_axis_type{0, 0, 1}, layouts_type{0, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_2_0_021, ref_topologies_and_axes_2_0_021);

    auto ref_topologies_and_axes_2_0_102 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0, ref_topo1, ref_topo0},
        vec_axis_type{1, 0, 0}, layouts_type{0, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_0_102, ref_topologies_and_axes_2_0_102);

    auto ref_topologies_and_axes_2_0_120 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0, ref_topo2, ref_topo4, ref_topo2,
                        ref_topo0},
        vec_axis_type{1, 1, 0, 0, 1}, layouts_type{0, 1, 0, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_2_0_120, ref_topologies_and_axes_2_0_120);

    auto ref_topologies_and_axes_2_0_201 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo5, ref_topo3, ref_topo1,
                        ref_topo0},
        vec_axis_type{0, 1, 0, 1, 0}, layouts_type{0, 0, 0, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_0_201, ref_topologies_and_axes_2_0_201);

    auto ref_topologies_and_axes_2_0_210 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0, ref_topo1, ref_topo3, ref_topo1,
                        ref_topo0},
        vec_axis_type{1, 0, 1, 1, 0}, layouts_type{0, 1, 1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_0_210, ref_topologies_and_axes_2_0_210);

    // topology2 to topology1
    auto topologies_and_axes_2_1_012 =
        get_all_pencil_topologies(topology2, topology1, axes012);
    auto topologies_and_axes_2_1_021 =
        get_all_pencil_topologies(topology2, topology1, axes021);
    auto topologies_and_axes_2_1_102 =
        get_all_pencil_topologies(topology2, topology1, axes102);
    auto topologies_and_axes_2_1_120 =
        get_all_pencil_topologies(topology2, topology1, axes120);
    auto topologies_and_axes_2_1_201 =
        get_all_pencil_topologies(topology2, topology1, axes201);
    auto topologies_and_axes_2_1_210 =
        get_all_pencil_topologies(topology2, topology1, axes210);

    auto ref_topologies_and_axes_2_1_012 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo5, ref_topo3, ref_topo1},
        vec_axis_type{0, 1, 0, 1}, layouts_type{0, 0, 0, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_1_012, ref_topologies_and_axes_2_1_012);

    auto ref_topologies_and_axes_2_1_021 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo2, ref_topo0, ref_topo1},
        vec_axis_type{0, 0, 1, 0}, layouts_type{0, 0, 0, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_1_021, ref_topologies_and_axes_2_1_021);

    auto ref_topologies_and_axes_2_1_102 =
        std::make_tuple(topologies_type{ref_topo2, ref_topo0, ref_topo1},
                        vec_axis_type{1, 0}, layouts_type{0, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_1_102, ref_topologies_and_axes_2_1_102);

    auto ref_topologies_and_axes_2_1_120 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0, ref_topo2, ref_topo0, ref_topo1},
        vec_axis_type{1, 1, 1, 0}, layouts_type{0, 1, 0, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_1_120, ref_topologies_and_axes_2_1_120);

    auto ref_topologies_and_axes_2_1_201 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo5, ref_topo3, ref_topo1},
        vec_axis_type{0, 1, 0, 1}, layouts_type{0, 0, 0, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_1_201, ref_topologies_and_axes_2_1_201);

    auto ref_topologies_and_axes_2_1_210 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0, ref_topo1, ref_topo3, ref_topo1},
        vec_axis_type{1, 0, 1, 1}, layouts_type{0, 1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_1_210, ref_topologies_and_axes_2_1_210);

    // topology2 to topology2
    auto topologies_and_axes_2_2_012 =
        get_all_pencil_topologies(topology2, topology2, axes012);
    auto topologies_and_axes_2_2_021 =
        get_all_pencil_topologies(topology2, topology2, axes021);
    auto topologies_and_axes_2_2_102 =
        get_all_pencil_topologies(topology2, topology2, axes102);
    auto topologies_and_axes_2_2_120 =
        get_all_pencil_topologies(topology2, topology2, axes120);
    auto topologies_and_axes_2_2_201 =
        get_all_pencil_topologies(topology2, topology2, axes201);
    auto topologies_and_axes_2_2_210 =
        get_all_pencil_topologies(topology2, topology2, axes210);

    auto ref_topologies_and_axes_2_2_012 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo5, ref_topo4, ref_topo2},
        vec_axis_type{0, 1, 1, 0}, layouts_type{0, 0, 0, 0, 0});
    EXPECT_EQ(topologies_and_axes_2_2_012, ref_topologies_and_axes_2_2_012);

    auto ref_topologies_and_axes_2_2_021 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo2, ref_topo0, ref_topo2},
        vec_axis_type{0, 0, 1, 1}, layouts_type{0, 0, 0, 1, 0});
    EXPECT_EQ(topologies_and_axes_2_2_021, ref_topologies_and_axes_2_2_021);

    auto ref_topologies_and_axes_2_2_102 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{1, 0, 0, 1}, layouts_type{0, 1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_2_2_102, ref_topologies_and_axes_2_2_102);

    auto ref_topologies_and_axes_2_2_120 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0, ref_topo2, ref_topo4, ref_topo2},
        vec_axis_type{1, 1, 0, 0}, layouts_type{0, 1, 0, 0, 0});
    EXPECT_EQ(topologies_and_axes_2_2_120, ref_topologies_and_axes_2_2_120);

    auto ref_topologies_and_axes_2_2_201 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo5, ref_topo4, ref_topo2},
        vec_axis_type{0, 1, 1, 0}, layouts_type{0, 0, 0, 0, 0});
    EXPECT_EQ(topologies_and_axes_2_2_201, ref_topologies_and_axes_2_2_201);

    auto ref_topologies_and_axes_2_2_210 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{1, 0, 0, 1}, layouts_type{0, 1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_2_2_210, ref_topologies_and_axes_2_2_210);

    // topology3 to topology0
    auto topologies_and_axes_3_0_012 =
        get_all_pencil_topologies(topology3, topology0, axes012);
    auto topologies_and_axes_3_0_021 =
        get_all_pencil_topologies(topology3, topology0, axes021);
    auto topologies_and_axes_3_0_102 =
        get_all_pencil_topologies(topology3, topology0, axes102);
    auto topologies_and_axes_3_0_120 =
        get_all_pencil_topologies(topology3, topology0, axes120);
    auto topologies_and_axes_3_0_201 =
        get_all_pencil_topologies(topology3, topology0, axes201);
    auto topologies_and_axes_3_0_210 =
        get_all_pencil_topologies(topology3, topology0, axes210);

    auto ref_topologies_and_axes_3_0_012 =
        std::make_tuple(topologies_type{ref_topo3, ref_topo1, ref_topo0},
                        vec_axis_type{1, 0}, layouts_type{1, 1, 1});
    EXPECT_EQ(topologies_and_axes_3_0_012, ref_topologies_and_axes_3_0_012);

    auto ref_topologies_and_axes_3_0_021 = std::make_tuple(
        topologies_type{ref_topo3, ref_topo1, ref_topo3, ref_topo1, ref_topo0},
        vec_axis_type{1, 1, 1, 0}, layouts_type{1, 1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_3_0_021, ref_topologies_and_axes_3_0_021);

    auto ref_topologies_and_axes_3_0_102 = std::make_tuple(
        topologies_type{ref_topo3, ref_topo5, ref_topo4, ref_topo2, ref_topo0},
        vec_axis_type{0, 1, 0, 1}, layouts_type{1, 0, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_3_0_102, ref_topologies_and_axes_3_0_102);

    auto ref_topologies_and_axes_3_0_120 = std::make_tuple(
        topologies_type{ref_topo3, ref_topo5, ref_topo3, ref_topo1, ref_topo0},
        vec_axis_type{0, 0, 1, 0}, layouts_type{1, 0, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_3_0_120, ref_topologies_and_axes_3_0_120);

    auto ref_topologies_and_axes_3_0_201 = std::make_tuple(
        topologies_type{ref_topo3, ref_topo1, ref_topo0, ref_topo2, ref_topo0},
        vec_axis_type{1, 0, 1, 1}, layouts_type{1, 1, 1, 0, 1});
    EXPECT_EQ(topologies_and_axes_3_0_201, ref_topologies_and_axes_3_0_201);

    auto ref_topologies_and_axes_3_0_210 = std::make_tuple(
        topologies_type{ref_topo3, ref_topo5, ref_topo4, ref_topo2, ref_topo0},
        vec_axis_type{0, 1, 0, 1}, layouts_type{1, 0, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_3_0_210, ref_topologies_and_axes_3_0_210);
  }
}

void test_get_all_pencil_topologies3D_4DView(std::size_t nprocs) {
  using topology_type        = std::array<std::size_t, 4>;
  using topologies_type      = std::vector<topology_type>;
  using topology_r_type      = Topology<std::size_t, 4, Kokkos::LayoutRight>;
  using topology_l_type      = Topology<std::size_t, 4, Kokkos::LayoutLeft>;
  using vec_axis_type        = std::vector<std::size_t>;
  using layouts_type         = std::vector<std::size_t>;
  std::size_t np0            = 4;
  topology_r_type topology0  = {1, 1, nprocs, np0},
                  topology1  = {1, nprocs, 1, np0},
                  topology3  = {1, nprocs, np0, 1},
                  topology6  = {nprocs, 1, 1, np0},
                  topology8  = {nprocs, 1, np0, 1},
                  topology10 = {nprocs, np0, 1, 1};

  topology_l_type topology2 = {1, np0, nprocs, 1},
                  topology4 = {1, np0, 1, nprocs},
                  topology5 = {1, 1, np0, nprocs},
                  topology7 = {np0, 1, nprocs, 1},
                  topology9 = {np0, nprocs, 1, 1};

  topology_type ref_topo0 = topology0.array(), ref_topo1 = topology1.array(),
                ref_topo2 = topology2.array(), ref_topo3 = topology3.array(),
                ref_topo4 = topology4.array(), ref_topo5 = topology5.array(),
                ref_topo6 = topology6.array(), ref_topo7 = topology7.array(),
                ref_topo8 = topology8.array(), ref_topo9 = topology9.array(),
                ref_topo10 = topology10.array();

  using axes_type   = std::array<int, 3>;
  axes_type axes012 = {0, 1, 2}, axes021 = {0, 2, 1}, axes102 = {1, 0, 2},
            axes120 = {1, 2, 0}, axes201 = {2, 0, 1}, axes210 = {2, 1, 0},
            axes123 = {1, 2, 3}, axes132 = {1, 3, 2};

  std::vector<axes_type> all_axes = {axes012, axes021, axes102, axes120,
                                     axes201, axes210, axes123, axes132};

  if (nprocs == 1) {
    for (const auto& axes : all_axes) {
      // Failure tests because only two elements differ (slabs)
      EXPECT_THROW(
          {
            [[maybe_unused]] auto topologies_and_axes_0_1 =
                get_all_pencil_topologies(topology0, topology1, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto topologies_and_axes_0_2 =
                get_all_pencil_topologies(topology0, topology2, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto topologies_and_axes_1_0 =
                get_all_pencil_topologies(topology1, topology0, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto topologies_and_axes_2_0 =
                get_all_pencil_topologies(topology2, topology0, axes);
          },
          std::runtime_error);
    }
  } else {
    /*
    // topology0 to topology0
    auto topologies_and_axes_0_0_012 =
        get_all_pencil_topologies(topology0, topology0, axes012);
    auto topologies_and_axes_0_0_021 =
        get_all_pencil_topologies(topology0, topology0, axes021);
    auto topologies_and_axes_0_0_102 =
        get_all_pencil_topologies(topology0, topology0, axes102);
    auto topologies_and_axes_0_0_120 =
        get_all_pencil_topologies(topology0, topology0, axes120);
    auto topologies_and_axes_0_0_201 =
        get_all_pencil_topologies(topology0, topology0, axes201);
    auto topologies_and_axes_0_0_210 =
        get_all_pencil_topologies(topology0, topology0, axes210);
    auto topologies_and_axes_0_0_123 =
        get_all_pencil_topologies(topology0, topology0, axes123);
    auto topologies_and_axes_0_0_132 =
        get_all_pencil_topologies(topology0, topology0, axes132);

    auto ref_topologies_and_axes_0_0_012 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo6, ref_topo0},
        vec_axis_type{0, 0}, layouts_type{1,1,1});
    EXPECT_EQ(topologies_and_axes_0_0_012, ref_topologies_and_axes_0_0_012);

    auto ref_topologies_and_axes_0_0_021 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo6, ref_topo0},
        vec_axis_type{0, 0}, layouts_type{1,1, 1});
    EXPECT_EQ(topologies_and_axes_0_0_021, ref_topologies_and_axes_0_0_021);

    auto ref_topologies_and_axes_0_0_102 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo6, ref_topo1, ref_topo0},
        vec_axis_type{0, 0, 0}, layouts_type{1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_0_102, ref_topologies_and_axes_0_0_102);

    auto ref_topologies_and_axes_0_0_120 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo6, ref_topo0},
        vec_axis_type{0, 0}, layouts_type{1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_0_120, ref_topologies_and_axes_0_0_120);

    auto ref_topologies_and_axes_0_0_201 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo6, ref_topo0},
        vec_axis_type{0, 0}, layouts_type{1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_0_201, ref_topologies_and_axes_0_0_201);

    auto ref_topologies_and_axes_0_0_210 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo6, ref_topo0},
        vec_axis_type{0, 0}, layouts_type{1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_0_210, ref_topologies_and_axes_0_0_210);

    auto ref_topologies_and_axes_0_0_123 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo7, ref_topo9, ref_topo7, ref_topo0},
        vec_axis_type{1, 0, 0, 1}, layouts_type{1, 0, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_0_0_123, ref_topologies_and_axes_0_0_123);

    auto ref_topologies_and_axes_0_0_132 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo6, ref_topo10, ref_topo6, ref_topo0},
        vec_axis_type{0, 1, 1, 0}, layouts_type{1, 1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_0_132, ref_topologies_and_axes_0_0_132);

    // topology0 to topology1
    auto topologies_and_axes_0_1_012 =
        get_all_pencil_topologies(topology0, topology1, axes012);
    auto topologies_and_axes_0_1_021 =
        get_all_pencil_topologies(topology0, topology1, axes021);
    auto topologies_and_axes_0_1_102 =
        get_all_pencil_topologies(topology0, topology1, axes102);
    auto topologies_and_axes_0_1_120 =
        get_all_pencil_topologies(topology0, topology1, axes120);
    auto topologies_and_axes_0_1_201 =
        get_all_pencil_topologies(topology0, topology1, axes201);
    auto topologies_and_axes_0_1_210 =
        get_all_pencil_topologies(topology0, topology1, axes210);
    auto topologies_and_axes_0_1_123 =
        get_all_pencil_topologies(topology0, topology1, axes123);
    auto topologies_and_axes_0_1_132 =
        get_all_pencil_topologies(topology0, topology1, axes132);

    auto ref_topologies_and_axes_0_1_012 =
        std::make_tuple(
            topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo1},
        vec_axis_type{0, 0, 0}, layouts_type{1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_1_012, ref_topologies_and_axes_0_1_012);

    auto ref_topologies_and_axes_0_1_021 =
    std::make_tuple(
        topologies_type{ref_topo0, ref_topo1},
        vec_axis_type{0}, layouts_type{1, 1});
    EXPECT_EQ(topologies_and_axes_0_1_021, ref_topologies_and_axes_0_1_021);

    auto ref_topologies_and_axes_0_1_102 =
                                                       std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo1},
        vec_axis_type{0, 0, 0}, layouts_type{1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_1_102, ref_topologies_and_axes_0_1_102);

    auto ref_topologies_and_axes_0_1_120 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo1},
        vec_axis_type{0, 0, 0}, layouts_type{1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_1_120, ref_topologies_and_axes_0_1_120);

    auto ref_topologies_and_axes_0_1_201 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1},
        vec_axis_type{0}, layouts_type{1, 1});
    EXPECT_EQ(topologies_and_axes_0_1_201, ref_topologies_and_axes_0_1_201);

    auto ref_topologies_and_axes_0_1_210 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1},
        vec_axis_type{0}, layouts_type{1, 1});
    EXPECT_EQ(topologies_and_axes_0_1_210, ref_topologies_and_axes_0_1_210);

    auto ref_topologies_and_axes_0_1_123 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2, ref_topo4, ref_topo5, ref_topo3,
    ref_topo1}, vec_axis_type{1, 0, 1, 0, 1}, layouts_type{1, 0, 0, 0, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_1_123, ref_topologies_and_axes_0_1_123);

    auto ref_topologies_and_axes_0_1_132 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo3, ref_topo5, ref_topo3,
    ref_topo1}, vec_axis_type{0, 1, 0, 0, 1}, layouts_type{1, 1, 1, 0, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_1_132, ref_topologies_and_axes_0_1_132);

    // topology0 to topology2
    auto topologies_and_axes_0_2_012 =
        get_all_pencil_topologies(topology0, topology2, axes012);
    auto topologies_and_axes_0_2_021 =
        get_all_pencil_topologies(topology0, topology2, axes021);
    auto topologies_and_axes_0_2_102 =
        get_all_pencil_topologies(topology0, topology2, axes102);
    auto topologies_and_axes_0_2_120 =
        get_all_pencil_topologies(topology0, topology2, axes120);
    auto topologies_and_axes_0_2_201 =
        get_all_pencil_topologies(topology0, topology2, axes201);
    auto topologies_and_axes_0_2_210 =
        get_all_pencil_topologies(topology0, topology2, axes210);
    auto topologies_and_axes_0_2_123 =
        get_all_pencil_topologies(topology0, topology2, axes123);
    auto topologies_and_axes_0_2_132 =
        get_all_pencil_topologies(topology0, topology2, axes132);

    auto ref_topologies_and_axes_0_2_012 =
    std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{0, 0, 1}, layouts_type{1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_0_2_012, ref_topologies_and_axes_0_2_012);

    auto ref_topologies_and_axes_0_2_021 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{0, 0, 1}, layouts_type{1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_0_2_021, ref_topologies_and_axes_0_2_021);

    auto ref_topologies_and_axes_0_2_102 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{0, 0, 1}, layouts_type{1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_0_2_102, ref_topologies_and_axes_0_2_102);

    auto ref_topologies_and_axes_0_2_120 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{0, 0, 1}, layouts_type{1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_0_2_120, ref_topologies_and_axes_0_2_120);

        auto ref_topologies_and_axes_0_2_201 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{0, 0, 1}, layouts_type{1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_0_2_201, ref_topologies_and_axes_0_2_201);

    auto ref_topologies_and_axes_0_2_210 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{0, 0, 1}, layouts_type{1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_0_2_210, ref_topologies_and_axes_0_2_210);
    auto ref_topologies_and_axes_0_2_123 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2, ref_topo4, ref_topo5, ref_topo4,
                        ref_topo2},
        vec_axis_type{1, 0, 1, 1, 0}, layouts_type{1, 0, 0, 0, 0, 0});
    EXPECT_EQ(topologies_and_axes_0_2_123, ref_topologies_and_axes_0_2_123);
    auto ref_topologies_and_axes_0_2_132 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo3, ref_topo5, ref_topo4,
                        ref_topo2},
        vec_axis_type{0, 1, 0, 1, 0}, layouts_type{1, 1, 1, 0, 0, 0});
    EXPECT_EQ(topologies_and_axes_0_2_132, ref_topologies_and_axes_0_2_132);


    // topology1 to topology0
    auto topologies_and_axes_1_0_012 =
        get_all_pencil_topologies(topology1, topology0, axes012);
    auto topologies_and_axes_1_0_021 =
        get_all_pencil_topologies(topology1, topology0, axes021);
    auto topologies_and_axes_1_0_102 =
        get_all_pencil_topologies(topology1, topology0, axes102);
    auto topologies_and_axes_1_0_120 =
        get_all_pencil_topologies(topology1, topology0, axes120);
    auto topologies_and_axes_1_0_201 =
        get_all_pencil_topologies(topology1, topology0, axes201);
    auto topologies_and_axes_1_0_210 =
        get_all_pencil_topologies(topology1, topology0, axes210);
    auto topologies_and_axes_1_0_123 =
        get_all_pencil_topologies(topology1, topology0, axes123);
    auto topologies_and_axes_1_0_132 =
        get_all_pencil_topologies(topology1, topology0, axes132);

    auto ref_topologies_and_axes_1_0_012 =
    std::make_tuple(
        topologies_type{ref_topo1, ref_topo0},
        vec_axis_type{0}, layouts_type{1, 1});
    EXPECT_EQ(topologies_and_axes_1_0_012, ref_topologies_and_axes_1_0_012);

    auto ref_topologies_and_axes_1_0_021 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo1, ref_topo0},
        vec_axis_type{0, 0, 0}, layouts_type{1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_0_021, ref_topologies_and_axes_1_0_021);

    auto ref_topologies_and_axes_1_0_102 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0},
        vec_axis_type{0}, layouts_type{1, 1});
    EXPECT_EQ(topologies_and_axes_1_0_102, ref_topologies_and_axes_1_0_102);

    auto ref_topologies_and_axes_1_0_120 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0},
        vec_axis_type{0}, layouts_type{1, 1});
    EXPECT_EQ(topologies_and_axes_1_0_120, ref_topologies_and_axes_1_0_120);

    auto ref_topologies_and_axes_1_0_201 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo1, ref_topo0},
        vec_axis_type{0, 0, 0}, layouts_type{1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_0_201, ref_topologies_and_axes_1_0_201);

    auto ref_topologies_and_axes_1_0_210 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo1, ref_topo0},
        vec_axis_type{0, 0, 0}, layouts_type{1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_0_210, ref_topologies_and_axes_1_0_210);

    auto ref_topologies_and_axes_1_0_123 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo3, ref_topo1, ref_topo0},
        vec_axis_type{1, 1, 0}, layouts_type{1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_0_123, ref_topologies_and_axes_1_0_123);

    auto ref_topologies_and_axes_1_0_132 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo3, ref_topo1, ref_topo0},
        vec_axis_type{1, 1, 0}, layouts_type{1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_0_132, ref_topologies_and_axes_1_0_132);
    */

    // topology1 to topology1
    auto topologies_and_axes_1_1_012 =
        get_all_pencil_topologies(topology1, topology1, axes012);
    auto topologies_and_axes_1_1_021 =
        get_all_pencil_topologies(topology1, topology1, axes021);
    auto topologies_and_axes_1_1_102 =
        get_all_pencil_topologies(topology1, topology1, axes102);
    auto topologies_and_axes_1_1_120 =
        get_all_pencil_topologies(topology1, topology1, axes120);
    auto topologies_and_axes_1_1_201 =
        get_all_pencil_topologies(topology1, topology1, axes201);
    auto topologies_and_axes_1_1_210 =
        get_all_pencil_topologies(topology1, topology1, axes210);
    auto topologies_and_axes_1_1_123 =
        get_all_pencil_topologies(topology1, topology1, axes123);
    auto topologies_and_axes_1_1_132 =
        get_all_pencil_topologies(topology1, topology1, axes132);

    auto ref_topologies_and_axes_1_1_012 =
        std::make_tuple(topologies_type{ref_topo1, ref_topo6, ref_topo1},
                        vec_axis_type{0, 0}, layouts_type{1, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_1_012, ref_topologies_and_axes_1_1_012);

    auto ref_topologies_and_axes_1_1_021 =
        std::make_tuple(topologies_type{ref_topo1, ref_topo6, ref_topo1},
                        vec_axis_type{0, 0}, layouts_type{1, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_1_021, ref_topologies_and_axes_1_1_021);

    auto ref_topologies_and_axes_1_1_102 =
        std::make_tuple(topologies_type{ref_topo1, ref_topo6, ref_topo1},
                        vec_axis_type{0, 0}, layouts_type{1, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_1_102, ref_topologies_and_axes_1_1_102);

    auto ref_topologies_and_axes_1_1_120 =
        std::make_tuple(topologies_type{ref_topo1, ref_topo6, ref_topo1},
                        vec_axis_type{0, 0}, layouts_type{1, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_1_120, ref_topologies_and_axes_1_1_120);

    auto ref_topologies_and_axes_1_1_201 =
        std::make_tuple(topologies_type{ref_topo1, ref_topo6, ref_topo1},
                        vec_axis_type{0, 0}, layouts_type{1, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_1_201, ref_topologies_and_axes_1_1_201);

    auto ref_topologies_and_axes_1_1_210 =
        std::make_tuple(topologies_type{ref_topo1, ref_topo6, ref_topo1},
                        vec_axis_type{0, 0}, layouts_type{1, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_1_210, ref_topologies_and_axes_1_1_210);

    auto ref_topologies_and_axes_1_1_123 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo9, ref_topo7, ref_topo9, ref_topo1},
        vec_axis_type{1, 0, 0, 1}, layouts_type{1, 0, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_1_1_123, ref_topologies_and_axes_1_1_123);

    auto ref_topologies_and_axes_1_1_132 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo9, ref_topo7, ref_topo9, ref_topo1},
        vec_axis_type{1, 0, 0, 1}, layouts_type{1, 0, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_1_1_132, ref_topologies_and_axes_1_1_132);

    // topology1 to topology2
    auto topologies_and_axes_1_2_012 =
        get_all_pencil_topologies(topology1, topology2, axes012);
    auto topologies_and_axes_1_2_021 =
        get_all_pencil_topologies(topology1, topology2, axes021);
    auto topologies_and_axes_1_2_102 =
        get_all_pencil_topologies(topology1, topology2, axes102);
    auto topologies_and_axes_1_2_120 =
        get_all_pencil_topologies(topology1, topology2, axes120);
    auto topologies_and_axes_1_2_201 =
        get_all_pencil_topologies(topology1, topology2, axes201);
    auto topologies_and_axes_1_2_210 =
        get_all_pencil_topologies(topology1, topology2, axes210);
    auto topologies_and_axes_1_2_123 =
        get_all_pencil_topologies(topology1, topology2, axes123);
    auto topologies_and_axes_1_2_132 =
        get_all_pencil_topologies(topology1, topology2, axes132);

    auto ref_topologies_and_axes_1_2_012 =
        std::make_tuple(topologies_type{ref_topo1, ref_topo0, ref_topo2},
                        vec_axis_type{0, 1}, layouts_type{1, 1, 0});
    EXPECT_EQ(topologies_and_axes_1_2_012, ref_topologies_and_axes_1_2_012);

    auto ref_topologies_and_axes_1_2_021 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{0, 0, 0, 1}, layouts_type{1, 1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_1_2_021, ref_topologies_and_axes_1_2_021);

    auto ref_topologies_and_axes_1_2_102 =
        std::make_tuple(topologies_type{ref_topo1, ref_topo0, ref_topo2},
                        vec_axis_type{0, 1}, layouts_type{1, 1, 0});
    EXPECT_EQ(topologies_and_axes_1_2_102, ref_topologies_and_axes_1_2_102);

    auto ref_topologies_and_axes_1_2_120 =
        std::make_tuple(topologies_type{ref_topo1, ref_topo0, ref_topo2},
                        vec_axis_type{0, 1}, layouts_type{1, 1, 0});

    EXPECT_EQ(topologies_and_axes_1_2_120, ref_topologies_and_axes_1_2_120);

    auto ref_topologies_and_axes_1_2_201 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{0, 0, 0, 1}, layouts_type{1, 1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_1_2_201, ref_topologies_and_axes_1_2_201);

    auto ref_topologies_and_axes_1_2_210 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{0, 0, 0, 1}, layouts_type{1, 1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_1_2_210, ref_topologies_and_axes_1_2_210);

    auto ref_topologies_and_axes_1_2_123 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo3, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{1, 1, 0, 1}, layouts_type{1, 1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_1_2_123, ref_topologies_and_axes_1_2_123);

    auto ref_topologies_and_axes_1_2_132 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo3, ref_topo5, ref_topo4, ref_topo2},
        vec_axis_type{1, 0, 1, 0}, layouts_type{1, 1, 0, 0, 0});
    EXPECT_EQ(topologies_and_axes_1_2_132, ref_topologies_and_axes_1_2_132);

    // topology2 to topology0
    auto topologies_and_axes_2_0_012 =
        get_all_pencil_topologies(topology2, topology0, axes012);
    auto topologies_and_axes_2_0_021 =
        get_all_pencil_topologies(topology2, topology0, axes021);
    auto topologies_and_axes_2_0_102 =
        get_all_pencil_topologies(topology2, topology0, axes102);
    auto topologies_and_axes_2_0_120 =
        get_all_pencil_topologies(topology2, topology0, axes120);
    auto topologies_and_axes_2_0_201 =
        get_all_pencil_topologies(topology2, topology0, axes201);
    auto topologies_and_axes_2_0_210 =
        get_all_pencil_topologies(topology2, topology0, axes210);
    auto topologies_and_axes_2_0_123 =
        get_all_pencil_topologies(topology2, topology0, axes123);
    auto topologies_and_axes_2_0_132 =
        get_all_pencil_topologies(topology2, topology0, axes132);

    auto ref_topologies_and_axes_2_0_012 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo5, ref_topo0},
        vec_axis_type{0, 1, 1}, layouts_type{0, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_2_0_012, ref_topologies_and_axes_2_0_012);

    auto ref_topologies_and_axes_2_0_021 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0, ref_topo1, ref_topo0},
        vec_axis_type{1, 0, 0}, layouts_type{0, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_0_021, ref_topologies_and_axes_2_0_021);

    auto ref_topologies_and_axes_2_0_102 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo2, ref_topo0},
        vec_axis_type{0, 0, 1}, layouts_type{0, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_2_0_102, ref_topologies_and_axes_2_0_102);

    auto ref_topologies_and_axes_2_0_120 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo2, ref_topo0},
        vec_axis_type{0, 0, 1}, layouts_type{0, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_2_0_120, ref_topologies_and_axes_2_0_120);

    auto ref_topologies_and_axes_2_0_201 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0, ref_topo1, ref_topo0},
        vec_axis_type{1, 0, 0}, layouts_type{0, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_0_201, ref_topologies_and_axes_2_0_201);

    auto ref_topologies_and_axes_2_0_210 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0, ref_topo1, ref_topo0},
        vec_axis_type{1, 0, 0}, layouts_type{0, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_0_210, ref_topologies_and_axes_2_0_210);

    auto ref_topologies_and_axes_2_0_123 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo2, ref_topo0},
        vec_axis_type{0, 0, 1}, layouts_type{0, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_2_0_123, ref_topologies_and_axes_2_0_123);

    auto ref_topologies_and_axes_2_0_132 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo2, ref_topo0},
        vec_axis_type{0, 0, 1}, layouts_type{0, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_2_0_132, ref_topologies_and_axes_2_0_132);

    // topology2 to topology1
    auto topologies_and_axes_2_1_012 =
        get_all_pencil_topologies(topology2, topology1, axes012);
    auto topologies_and_axes_2_1_021 =
        get_all_pencil_topologies(topology2, topology1, axes021);
    auto topologies_and_axes_2_1_102 =
        get_all_pencil_topologies(topology2, topology1, axes102);
    auto topologies_and_axes_2_1_120 =
        get_all_pencil_topologies(topology2, topology1, axes120);
    auto topologies_and_axes_2_1_201 =
        get_all_pencil_topologies(topology2, topology1, axes201);
    auto topologies_and_axes_2_1_210 =
        get_all_pencil_topologies(topology2, topology1, axes210);
    auto topologies_and_axes_2_1_123 =
        get_all_pencil_topologies(topology2, topology1, axes123);
    auto topologies_and_axes_2_1_132 =
        get_all_pencil_topologies(topology2, topology1, axes132);

    auto ref_topologies_and_axes_2_1_012 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo5, ref_topo3, ref_topo1},
        vec_axis_type{0, 1, 0, 1}, layouts_type{0, 0, 0, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_1_012, ref_topologies_and_axes_2_1_012);

    auto ref_topologies_and_axes_2_1_021 =
        std::make_tuple(topologies_type{ref_topo2, ref_topo0, ref_topo1},
                        vec_axis_type{1, 0}, layouts_type{0, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_1_021, ref_topologies_and_axes_2_1_021);

    auto ref_topologies_and_axes_2_1_102 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo5, ref_topo3, ref_topo1},
        vec_axis_type{0, 1, 0, 1}, layouts_type{0, 0, 0, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_1_102, ref_topologies_and_axes_2_1_102);

    auto ref_topologies_and_axes_2_1_120 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo5, ref_topo3, ref_topo1},
        vec_axis_type{0, 1, 0, 1}, layouts_type{0, 0, 0, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_1_120, ref_topologies_and_axes_2_1_120);

    auto ref_topologies_and_axes_2_1_201 =
        std::make_tuple(topologies_type{ref_topo2, ref_topo0, ref_topo1},
                        vec_axis_type{1, 0}, layouts_type{0, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_1_201, ref_topologies_and_axes_2_1_201);

    auto ref_topologies_and_axes_2_1_210 =
        std::make_tuple(topologies_type{ref_topo2, ref_topo0, ref_topo1},
                        vec_axis_type{1, 0}, layouts_type{0, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_1_210, ref_topologies_and_axes_2_1_210);

    auto ref_topologies_and_axes_2_1_123 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo5, ref_topo3, ref_topo1},
        vec_axis_type{0, 1, 0, 1}, layouts_type{0, 0, 0, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_1_123, ref_topologies_and_axes_2_1_123);

    auto ref_topologies_and_axes_2_1_132 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo2, ref_topo0, ref_topo1},
        vec_axis_type{0, 0, 1, 0}, layouts_type{0, 0, 0, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_1_132, ref_topologies_and_axes_2_1_132);

    // topology2 to topology2
    auto topologies_and_axes_2_2_012 =
        get_all_pencil_topologies(topology2, topology2, axes012);
    auto topologies_and_axes_2_2_021 =
        get_all_pencil_topologies(topology2, topology2, axes021);
    auto topologies_and_axes_2_2_102 =
        get_all_pencil_topologies(topology2, topology2, axes102);
    auto topologies_and_axes_2_2_120 =
        get_all_pencil_topologies(topology2, topology2, axes120);
    auto topologies_and_axes_2_2_201 =
        get_all_pencil_topologies(topology2, topology2, axes201);
    auto topologies_and_axes_2_2_210 =
        get_all_pencil_topologies(topology2, topology2, axes210);
    auto topologies_and_axes_2_2_123 =
        get_all_pencil_topologies(topology2, topology2, axes123);
    auto topologies_and_axes_2_2_132 =
        get_all_pencil_topologies(topology2, topology2, axes132);

    auto ref_topologies_and_axes_2_2_012 =
        std::make_tuple(topologies_type{ref_topo2, ref_topo10, ref_topo8,
                                        ref_topo10, ref_topo2},
                        vec_axis_type{0, 1, 1, 0}, layouts_type{0, 1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_2_2_012, ref_topologies_and_axes_2_2_012);

    auto ref_topologies_and_axes_2_2_021 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo7, ref_topo9, ref_topo7, ref_topo2},
        vec_axis_type{1, 0, 0, 1}, layouts_type{0, 0, 0, 0, 0});
    EXPECT_EQ(topologies_and_axes_2_2_021, ref_topologies_and_axes_2_2_021);

    auto ref_topologies_and_axes_2_2_102 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo10, ref_topo2, ref_topo7, ref_topo2},
        vec_axis_type{0, 0, 1, 1}, layouts_type{0, 1, 0, 0, 0});
    EXPECT_EQ(topologies_and_axes_2_2_102, ref_topologies_and_axes_2_2_102);

    auto ref_topologies_and_axes_2_2_120 =
        std::make_tuple(topologies_type{ref_topo2, ref_topo10, ref_topo8,
                                        ref_topo10, ref_topo2},
                        vec_axis_type{0, 1, 1, 0}, layouts_type{0, 1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_2_2_120, ref_topologies_and_axes_2_2_120);

    auto ref_topologies_and_axes_2_2_201 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo7, ref_topo2, ref_topo10, ref_topo2},
        vec_axis_type{1, 1, 0, 0}, layouts_type{0, 0, 0, 1, 0});
    EXPECT_EQ(topologies_and_axes_2_2_201, ref_topologies_and_axes_2_2_201);

    auto ref_topologies_and_axes_2_2_210 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo7, ref_topo9, ref_topo3, ref_topo2},
        vec_axis_type{0, 1, 0, 1}, layouts_type{0, 0, 0, 1, 0});

    auto ref_topologies_and_axes_2_2_123 =
        std::make_tuple(topologies_type{ref_topo2, ref_topo10, ref_topo8,
                                        ref_topo10, ref_topo2},
                        vec_axis_type{0, 1, 1, 0}, layouts_type{0, 1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_2_2_123, ref_topologies_and_axes_2_2_123);

    auto ref_topologies_and_axes_2_2_132 =
        std::make_tuple(topologies_type{ref_topo2, ref_topo10, ref_topo8,
                                        ref_topo10, ref_topo2},
                        vec_axis_type{0, 1, 1, 0}, layouts_type{0, 1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_2_2_132, ref_topologies_and_axes_2_2_132);

    // topology3 to topology0
    auto topologies_and_axes_3_0_123 =
        get_all_pencil_topologies(topology3, topology0, axes123);

    auto ref_topologies_and_axes_3_0_123 =
        std::make_tuple(topologies_type{ref_topo3, ref_topo1, ref_topo0},
                        vec_axis_type{1, 0}, layouts_type{1, 1, 1});
    EXPECT_EQ(topologies_and_axes_3_0_123, ref_topologies_and_axes_3_0_123);
  }
}

void test_decompose_axes_slab(std::size_t nprocs) {
  using topology_type  = std::array<std::size_t, 3>;
  using topology_type2 = std::array<std::size_t, 4>;

  // 3D topologies
  topology_type topology0 = {1, 1, nprocs}, topology1 = {1, nprocs, 1},
                topology2 = {nprocs, 1, 1};

  // 4D topologies
  topology_type2 topology3 = {1, 1, 1, nprocs}, topology4 = {1, 1, nprocs, 1};

  using axes_type     = std::array<std::size_t, 3>;
  using vec_axes_type = std::vector<std::size_t>;
  axes_type axes012 = {0, 1, 2}, axes021 = {0, 2, 1}, axes102 = {1, 0, 2},
            axes120 = {1, 2, 0}, axes201 = {2, 0, 1}, axes210 = {2, 1, 0};

  std::vector<axes_type> all_axes = {axes012, axes021, axes102,
                                     axes120, axes201, axes210};

  // All topologies
  std::vector<topology_type> topologies_0_1   = {topology0, topology1};
  std::vector<topology_type> topologies_0_2   = {topology0, topology2};
  std::vector<topology_type> topologies_1_2   = {topology1, topology2};
  std::vector<topology_type> topologies_2_0_2 = {topology2, topology0,
                                                 topology2};
  std::vector<topology_type2> topologies_3_4  = {topology3, topology4},
                              topologies_4_3  = {topology4, topology3};

  if (nprocs == 1) {
    for (const auto& axes : all_axes) {
      auto all_axes_0_1   = decompose_axes(topologies_0_1, axes);
      auto all_axes_0_2   = decompose_axes(topologies_0_2, axes);
      auto all_axes_1_2   = decompose_axes(topologies_1_2, axes);
      auto all_axes_2_0_2 = decompose_axes(topologies_2_0_2, axes);
      auto all_axes_3_4   = decompose_axes(topologies_3_4, axes);
      auto all_axes_4_3   = decompose_axes(topologies_4_3, axes);
      std::vector<vec_axes_type> ref_all_axes2 = {to_vector(axes), {}},
                                 ref_all_axes3 = {to_vector(axes), {}, {}},
                                 ref_all_axes4 = {to_vector(axes), {}};
      EXPECT_EQ(all_axes_0_1, ref_all_axes2);
      EXPECT_EQ(all_axes_0_2, ref_all_axes2);
      EXPECT_EQ(all_axes_1_2, ref_all_axes2);
      EXPECT_EQ(all_axes_2_0_2, ref_all_axes3);
      EXPECT_EQ(all_axes_3_4, ref_all_axes4);
      EXPECT_EQ(all_axes_4_3, ref_all_axes4);
    }
  } else {
    auto all_axes_2_0_2     = decompose_axes(topologies_2_0_2, axes021);
    auto all_axes_3_4       = decompose_axes(topologies_3_4, axes012);
    auto all_axes_4_3_ax210 = decompose_axes(topologies_4_3, axes210);
    auto all_axes_4_3_ax012 = decompose_axes(topologies_4_3, axes012);
    std::vector<vec_axes_type> ref_all_axes_2_0_2     = {vec_axes_type{2, 1},
                                                         vec_axes_type{0},
                                                         vec_axes_type{}},
                               ref_all_axes_3_4       = {vec_axes_type{0, 1, 2},
                                                         vec_axes_type{}},
                               ref_all_axes_4_3_ax210 = {vec_axes_type{1, 0},
                                                         vec_axes_type{2}},
                               ref_all_axes_4_3_ax012 = {
                                   vec_axes_type{}, vec_axes_type{0, 1, 2}};
    EXPECT_EQ(all_axes_2_0_2, ref_all_axes_2_0_2);
    EXPECT_EQ(all_axes_3_4, ref_all_axes_3_4);
    EXPECT_EQ(all_axes_4_3_ax210, ref_all_axes_4_3_ax210);
    EXPECT_EQ(all_axes_4_3_ax012, ref_all_axes_4_3_ax012);
  }
}

void test_decompose_axes_pencil(std::size_t nprocs) {
  using topology_type = std::array<std::size_t, 3>;
  std::size_t np0     = 4;

  // 3D topologies
  topology_type topology0 = {1, nprocs, np0}, topology1 = {nprocs, 1, np0},
                topology2 = {np0, nprocs, 1}, topology3 = {nprocs, np0, 1},
                topology4 = {np0, 1, nprocs};

  using axes_type     = std::array<std::size_t, 3>;
  using vec_axes_type = std::vector<std::size_t>;
  axes_type axes012 = {0, 1, 2}, axes021 = {0, 2, 1}, axes102 = {1, 0, 2},
            axes120 = {1, 2, 0}, axes201 = {2, 0, 1}, axes210 = {2, 1, 0};

  std::vector<axes_type> all_axes = {axes012, axes021, axes102,
                                     axes120, axes201, axes210};

  // All topologies
  std::vector<topology_type> topologies_02420 = {topology0, topology2,
                                                 topology4, topology2,
                                                 topology0},
                             topologies_01310 = {topology0, topology1,
                                                 topology3, topology1,
                                                 topology0},
                             topologies_02010 = {topology0, topology2,
                                                 topology0, topology1,
                                                 topology0},
                             topologies_01020 = {topology0, topology1,
                                                 topology0, topology2,
                                                 topology0},
                             topologies_0201 = {topology0, topology2, topology0,
                                                topology1};

  if (nprocs == 1) {
    // Slab geometry
    auto all_axes_02420_axes012 = decompose_axes(topologies_02420, axes012);
    auto all_axes_01310_axes021 = decompose_axes(topologies_01310, axes021);
    auto all_axes_02010_axes102 = decompose_axes(topologies_02010, axes102);
    auto all_axes_01020_axes201 = decompose_axes(topologies_01020, axes201);
    auto all_axes_0201_axes102  = decompose_axes(topologies_0201, axes102);
    std::vector<vec_axes_type>
        ref_all_axes_02420_axes012 = {{},
                                      vec_axes_type{1, 2},
                                      {},
                                      {},
                                      vec_axes_type{0}},
        ref_all_axes_01310_axes021 = {vec_axes_type{1},
                                      {},
                                      vec_axes_type{0, 2},
                                      {},
                                      {}},
        ref_all_axes_02010_axes102 = {{},
                                      vec_axes_type{2},
                                      vec_axes_type{1, 0},
                                      {},
                                      {}},
        ref_all_axes_01020_axes201 = {vec_axes_type{0, 1},
                                      {},
                                      {},
                                      vec_axes_type{2},
                                      {}},
        ref_all_axes_0201_axes102  = {
            {}, vec_axes_type{2}, vec_axes_type{1, 0}, {}};
    EXPECT_EQ(all_axes_02420_axes012, ref_all_axes_02420_axes012);
    EXPECT_EQ(all_axes_01310_axes021, ref_all_axes_01310_axes021);
    EXPECT_EQ(all_axes_02010_axes102, ref_all_axes_02010_axes102);
    EXPECT_EQ(all_axes_01020_axes201, ref_all_axes_01020_axes201);
    EXPECT_EQ(all_axes_0201_axes102, ref_all_axes_0201_axes102);
  } else {
    // Pencil geometry
    auto all_axes_02420_axes012 = decompose_axes(topologies_02420, axes012);
    auto all_axes_01310_axes021 = decompose_axes(topologies_01310, axes021);
    auto all_axes_02010_axes102 = decompose_axes(topologies_02010, axes102);
    auto all_axes_01020_axes201 = decompose_axes(topologies_01020, axes201);
    auto all_axes_0201_axes102  = decompose_axes(topologies_0201, axes102);
    std::vector<vec_axes_type> ref_all_axes_02420_axes012 = {{},
                                                             vec_axes_type{2},
                                                             vec_axes_type{1},
                                                             {},
                                                             vec_axes_type{0}},
                               ref_all_axes_01310_axes021 = {{},
                                                             vec_axes_type{1},
                                                             vec_axes_type{2},
                                                             {},
                                                             vec_axes_type{0}},
                               ref_all_axes_02010_axes102 = {{},
                                                             vec_axes_type{2},
                                                             vec_axes_type{0},
                                                             vec_axes_type{1},
                                                             {}},
                               ref_all_axes_01020_axes201 = {{},
                                                             vec_axes_type{1},
                                                             vec_axes_type{0},
                                                             vec_axes_type{2},
                                                             {}},
                               ref_all_axes_0201_axes102  = {{},
                                                             vec_axes_type{2},
                                                             vec_axes_type{0},
                                                             vec_axes_type{1}};
    EXPECT_EQ(all_axes_02420_axes012, ref_all_axes_02420_axes012);
    EXPECT_EQ(all_axes_01310_axes021, ref_all_axes_01310_axes021);
    EXPECT_EQ(all_axes_02010_axes102, ref_all_axes_02010_axes102);
    EXPECT_EQ(all_axes_01020_axes201, ref_all_axes_01020_axes201);
    EXPECT_EQ(all_axes_0201_axes102, ref_all_axes_0201_axes102);
  }
}

}  // namespace

TEST_P(SlabParamTests, GetSlab2D) {
  int n0 = GetParam();
  test_get_slab_2D(n0);
}

TEST_P(SlabParamTests, GetSlab3D) {
  int n0 = GetParam();
  test_get_slab_3D(n0);
}

TEST_P(SlabParamTests, GetAllSlabTopologies1D_3DView) {
  int n0 = GetParam();
  test_get_all_slab_topologies1D_3DView(n0);
}

TEST_P(SlabParamTests, GetAllSlabTopologies2D_2DView) {
  int n0 = GetParam();
  test_get_all_slab_topologies2D_2DView(n0);
}

TEST_P(SlabParamTests, GetAllSlabTopologies2D_3DView) {
  int n0 = GetParam();
  test_get_all_slab_topologies2D_3DView(n0);
}

TEST_P(SlabParamTests, GetAllSlabTopologies3D_3DView) {
  int n0 = GetParam();
  test_get_all_slab_topologies3D_3DView(n0);
}

TEST_P(SlabParamTests, GetAllSlabTopologies3D_4DView) {
  int n0 = GetParam();
  test_get_all_slab_topologies3D_4DView(n0);
}

INSTANTIATE_TEST_SUITE_P(SlabTests, SlabParamTests,
                         ::testing::Values(1, 2, 3, 4, 5, 6));

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

TEST_P(PencilParamTests, GetMidArray4D) {
  int n0 = GetParam();
  test_get_mid_array_pencil_4D(n0);
}

/*
TEST_P(PencilParamTests, GetAllPencilTopologies1D_3DView) {
  int n0 = GetParam();
  test_get_all_pencil_topologies1D_3DView(n0);
}

TEST_P(PencilParamTests, GetAllPencilTopologies2D_3DView) {
  int n0 = GetParam();
  test_get_all_pencil_topologies2D_3DView(n0);
}
*/

TEST_P(PencilParamTests, GetAllPencilTopologies3D_3DView) {
  int n0 = GetParam();
  test_get_all_pencil_topologies3D_3DView(n0);
}

TEST_P(PencilParamTests, GetAllPencilTopologies3D_4DView) {
  int n0 = GetParam();
  test_get_all_pencil_topologies3D_4DView(n0);
}

INSTANTIATE_TEST_SUITE_P(PencilTests, PencilParamTests,
                         ::testing::Values(1, 2, 3, 4, 5, 6));

TEST_P(TopologyParamTests, MergeTopology) {
  int n0 = GetParam();
  test_merge_topology(n0);
}

TEST_P(TopologyParamTests, DiffTopology) {
  int n0 = GetParam();
  test_diff_topology(n0);
}

TEST_P(TopologyParamTests, GetTopologyType) {
  int n0 = GetParam();
  test_get_topology_type(n0);
}

TEST_P(TopologyParamTests, IsTopology) {
  int n0 = GetParam();
  test_is_topology(n0);
}

TEST_P(TopologyParamTests, DecomposeAxesSlab) {
  int n0 = GetParam();
  test_decompose_axes_slab(n0);
}

TEST_P(TopologyParamTests, DecomposeAxesPencil) {
  int n0 = GetParam();
  test_decompose_axes_pencil(n0);
}

INSTANTIATE_TEST_SUITE_P(TopologyTests, TopologyParamTests,
                         ::testing::Values(1, 2, 3, 4, 5, 6));
