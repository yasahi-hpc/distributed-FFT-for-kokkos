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

template <typename T, typename LayoutType>
void test_get_dst_map2D_View2D() {
  using map_type = std::array<std::size_t, 2>;

  map_type src_map_01 = {0, 1};
  map_type src_map_10 = {1, 0};
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
