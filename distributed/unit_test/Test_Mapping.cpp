#include <mpi.h>
#include <gtest/gtest.h>
#include <iostream>
#include <Kokkos_Core.hpp>
#include "Mapping.hpp"
#include "Test_Utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using test_types      = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>,
                                    std::pair<double, Kokkos::LayoutLeft>,
                                    std::pair<double, Kokkos::LayoutRight>>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct TestMapping : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;
};

class PencilParamTests : public ::testing::TestWithParam<int> {};

template <typename T, typename LayoutType>
void test_get_dst_map2D_View2D() {
  using map_type = std::array<std::size_t, 2>;

  map_type src_map_01   = {0, 1};
  map_type src_map_10   = {1, 0};
  auto dst_map_01_axis0 = get_dst_map<LayoutType, 2>(src_map_01, 0);
  auto dst_map_10_axis0 = get_dst_map<LayoutType, 2>(src_map_10, 0);
  auto dst_map_01_axis1 = get_dst_map<LayoutType, 2>(src_map_01, 1);
  auto dst_map_10_axis1 = get_dst_map<LayoutType, 2>(src_map_10, 1);

  map_type ref_dst_map_01_axis0, ref_dst_map_10_axis0;
  map_type ref_dst_map_01_axis1, ref_dst_map_10_axis1;
  if (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    ref_dst_map_01_axis0 = {0, 1};
    ref_dst_map_10_axis0 = {0, 1};
    ref_dst_map_01_axis1 = {1, 0};
    ref_dst_map_10_axis1 = {1, 0};
  } else {
    ref_dst_map_01_axis0 = {1, 0};
    ref_dst_map_10_axis0 = {1, 0};
    ref_dst_map_01_axis1 = {0, 1};
    ref_dst_map_10_axis1 = {0, 1};
  }

  EXPECT_TRUE(dst_map_01_axis0 == ref_dst_map_01_axis0);
  EXPECT_TRUE(dst_map_10_axis0 == ref_dst_map_10_axis0);
  EXPECT_TRUE(dst_map_01_axis1 == ref_dst_map_01_axis1);
  EXPECT_TRUE(dst_map_10_axis1 == ref_dst_map_10_axis1);
}

template <typename T, typename LayoutType>
void test_get_dst_map3D_View3D() {
  using map_type = std::array<std::size_t, 3>;

  map_type src_map_012 = {0, 1, 2};
  map_type src_map_021 = {0, 2, 1};
  map_type src_map_102 = {1, 0, 2};
  map_type src_map_120 = {1, 2, 0};
  map_type src_map_201 = {2, 0, 1};
  map_type src_map_210 = {2, 1, 0};

  auto dst_map_012_axis0 = get_dst_map<LayoutType, 3>(src_map_012, 0);
  auto dst_map_021_axis0 = get_dst_map<LayoutType, 3>(src_map_021, 0);
  auto dst_map_102_axis0 = get_dst_map<LayoutType, 3>(src_map_102, 0);
  auto dst_map_120_axis0 = get_dst_map<LayoutType, 3>(src_map_120, 0);
  auto dst_map_201_axis0 = get_dst_map<LayoutType, 3>(src_map_201, 0);
  auto dst_map_210_axis0 = get_dst_map<LayoutType, 3>(src_map_210, 0);

  auto dst_map_012_axis1 = get_dst_map<LayoutType, 3>(src_map_012, 1);
  auto dst_map_021_axis1 = get_dst_map<LayoutType, 3>(src_map_021, 1);
  auto dst_map_102_axis1 = get_dst_map<LayoutType, 3>(src_map_102, 1);
  auto dst_map_120_axis1 = get_dst_map<LayoutType, 3>(src_map_120, 1);
  auto dst_map_201_axis1 = get_dst_map<LayoutType, 3>(src_map_201, 1);
  auto dst_map_210_axis1 = get_dst_map<LayoutType, 3>(src_map_210, 1);

  auto dst_map_012_axis2 = get_dst_map<LayoutType, 3>(src_map_012, 2);
  auto dst_map_021_axis2 = get_dst_map<LayoutType, 3>(src_map_021, 2);
  auto dst_map_102_axis2 = get_dst_map<LayoutType, 3>(src_map_102, 2);
  auto dst_map_120_axis2 = get_dst_map<LayoutType, 3>(src_map_120, 2);
  auto dst_map_201_axis2 = get_dst_map<LayoutType, 3>(src_map_201, 2);
  auto dst_map_210_axis2 = get_dst_map<LayoutType, 3>(src_map_210, 2);

  map_type ref_dst_map_012_axis0, ref_dst_map_021_axis0, ref_dst_map_102_axis0,
      ref_dst_map_120_axis0, ref_dst_map_201_axis0, ref_dst_map_210_axis0;
  map_type ref_dst_map_012_axis1, ref_dst_map_021_axis1, ref_dst_map_102_axis1,
      ref_dst_map_120_axis1, ref_dst_map_201_axis1, ref_dst_map_210_axis1;
  map_type ref_dst_map_012_axis2, ref_dst_map_021_axis2, ref_dst_map_102_axis2,
      ref_dst_map_120_axis2, ref_dst_map_201_axis2, ref_dst_map_210_axis2;

  if (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    ref_dst_map_012_axis0 = {0, 1, 2};
    ref_dst_map_021_axis0 = {0, 2, 1};
    ref_dst_map_102_axis0 = {0, 1, 2};
    ref_dst_map_120_axis0 = {0, 1, 2};
    ref_dst_map_201_axis0 = {0, 2, 1};
    ref_dst_map_210_axis0 = {0, 2, 1};

    ref_dst_map_012_axis1 = {1, 0, 2};
    ref_dst_map_021_axis1 = {1, 0, 2};
    ref_dst_map_102_axis1 = {1, 0, 2};
    ref_dst_map_120_axis1 = {1, 2, 0};
    ref_dst_map_201_axis1 = {1, 2, 0};
    ref_dst_map_210_axis1 = {1, 2, 0};

    ref_dst_map_012_axis2 = {2, 0, 1};
    ref_dst_map_021_axis2 = {2, 0, 1};
    ref_dst_map_102_axis2 = {2, 1, 0};
    ref_dst_map_120_axis2 = {2, 1, 0};
    ref_dst_map_201_axis2 = {2, 0, 1};
    ref_dst_map_210_axis2 = {2, 1, 0};

  } else {
    ref_dst_map_012_axis0 = {1, 2, 0};
    ref_dst_map_021_axis0 = {2, 1, 0};
    ref_dst_map_102_axis0 = {1, 2, 0};
    ref_dst_map_120_axis0 = {1, 2, 0};
    ref_dst_map_201_axis0 = {2, 1, 0};
    ref_dst_map_210_axis0 = {2, 1, 0};

    ref_dst_map_012_axis1 = {0, 2, 1};
    ref_dst_map_021_axis1 = {0, 2, 1};
    ref_dst_map_102_axis1 = {0, 2, 1};
    ref_dst_map_120_axis1 = {2, 0, 1};
    ref_dst_map_201_axis1 = {2, 0, 1};
    ref_dst_map_210_axis1 = {2, 0, 1};

    ref_dst_map_012_axis2 = {0, 1, 2};
    ref_dst_map_021_axis2 = {0, 1, 2};
    ref_dst_map_102_axis2 = {1, 0, 2};
    ref_dst_map_120_axis2 = {1, 0, 2};
    ref_dst_map_201_axis2 = {0, 1, 2};
    ref_dst_map_210_axis2 = {1, 0, 2};
  }

  EXPECT_TRUE(dst_map_012_axis0 == ref_dst_map_012_axis0);
  EXPECT_TRUE(dst_map_021_axis0 == ref_dst_map_021_axis0);
  EXPECT_TRUE(dst_map_102_axis0 == ref_dst_map_102_axis0);
  EXPECT_TRUE(dst_map_120_axis0 == ref_dst_map_120_axis0);
  EXPECT_TRUE(dst_map_201_axis0 == ref_dst_map_201_axis0);
  EXPECT_TRUE(dst_map_210_axis0 == ref_dst_map_210_axis0);

  EXPECT_TRUE(dst_map_012_axis1 == ref_dst_map_012_axis1);
  EXPECT_TRUE(dst_map_021_axis1 == ref_dst_map_021_axis1);
  EXPECT_TRUE(dst_map_102_axis1 == ref_dst_map_102_axis1);
  EXPECT_TRUE(dst_map_120_axis1 == ref_dst_map_120_axis1);
  EXPECT_TRUE(dst_map_201_axis1 == ref_dst_map_201_axis1);
  EXPECT_TRUE(dst_map_210_axis1 == ref_dst_map_210_axis1);

  EXPECT_TRUE(dst_map_012_axis2 == ref_dst_map_012_axis2);
  EXPECT_TRUE(dst_map_021_axis2 == ref_dst_map_021_axis2);
  EXPECT_TRUE(dst_map_102_axis2 == ref_dst_map_102_axis2);
  EXPECT_TRUE(dst_map_120_axis2 == ref_dst_map_120_axis2);
  EXPECT_TRUE(dst_map_201_axis2 == ref_dst_map_201_axis2);
  EXPECT_TRUE(dst_map_210_axis2 == ref_dst_map_210_axis2);
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
    EXPECT_THROW({ auto inout_axis01 = get_pencil(topology0, topology1); },
                 std::runtime_error);
    EXPECT_THROW({ auto inout_axis02 = get_pencil(topology0, topology2); },
                 std::runtime_error);
    EXPECT_THROW({ auto inout_axis10 = get_pencil(topology1, topology0); },
                 std::runtime_error);
    EXPECT_THROW({ auto inout_axis12 = get_pencil(topology1, topology2); },
                 std::runtime_error);
    EXPECT_THROW({ auto inout_axis20 = get_pencil(topology2, topology0); },
                 std::runtime_error);
    EXPECT_THROW({ auto inout_axis21 = get_pencil(topology2, topology1); },
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
  EXPECT_THROW({ auto inout_axis30 = get_pencil(topology3, topology0); },
               std::runtime_error);
  EXPECT_THROW({ auto inout_axis31 = get_pencil(topology3, topology1); },
               std::runtime_error);
  EXPECT_THROW({ auto inout_axis32 = get_pencil(topology3, topology2); },
               std::runtime_error);
}

}  // namespace

TYPED_TEST_SUITE(TestMapping, test_types);

TYPED_TEST(TestMapping, View2D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_get_dst_map2D_View2D<float_type, layout_type>();
}

TYPED_TEST(TestMapping, View3D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_get_dst_map3D_View3D<float_type, layout_type>();
}

TEST_P(PencilParamTests, 3D) {
  int n0 = GetParam();
  test_get_pencil_3D(n0);
}

INSTANTIATE_TEST_SUITE_P(PencilTests, PencilParamTests,
                         ::testing::Values(1, 2, 3, 4, 5, 6));
