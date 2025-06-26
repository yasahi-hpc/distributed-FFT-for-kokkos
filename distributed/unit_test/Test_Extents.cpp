#include <mpi.h>
#include <gtest/gtest.h>
#include <iostream>
#include <Kokkos_Core.hpp>
#include "Extents.hpp"
#include "Test_Utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using test_types      = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>,
                                    std::pair<double, Kokkos::LayoutLeft>,
                                    std::pair<double, Kokkos::LayoutRight>>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct TestExtents : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;
};

class TopologyParamTests : public ::testing::TestWithParam<int> {};

void test_merge_topology(std::size_t nprocs) {
  using topology_type     = std::array<std::size_t, 3>;
  topology_type topology0 = {1, 1, nprocs};
  topology_type topology1 = {1, nprocs, 1};
  topology_type topology2 = {nprocs, 1, 1};
  topology_type topology3 = {nprocs, 1, 8};
  topology_type topology4 = {nprocs, 8, 1};

  if (nprocs == 1) {
    // Failure tests because these are identical
    EXPECT_THROW({ auto merged01 = merge_topology(topology0, topology1); },
                 std::runtime_error);
    EXPECT_THROW({ auto merged02 = merge_topology(topology0, topology2); },
                 std::runtime_error);
    EXPECT_THROW({ auto merged12 = merge_topology(topology1, topology2); },
                 std::runtime_error);
  } else {
    auto merged01 = merge_topology(topology0, topology1);
    auto merged02 = merge_topology(topology0, topology2);
    auto merged12 = merge_topology(topology1, topology2);

    topology_type ref_merged01 = {1, nprocs, nprocs};
    topology_type ref_merged02 = {nprocs, 1, nprocs};
    topology_type ref_merged12 = {nprocs, nprocs, 1};

    EXPECT_EQ(merged01, ref_merged01);
    EXPECT_EQ(merged02, ref_merged02);
    EXPECT_EQ(merged12, ref_merged12);
  }

  // Failure tests because these do not have same size
  EXPECT_THROW({ auto merged03 = merge_topology(topology0, topology3); },
               std::runtime_error);
  EXPECT_THROW({ auto merged04 = merge_topology(topology0, topology4); },
               std::runtime_error);
  EXPECT_THROW({ auto merged13 = merge_topology(topology1, topology3); },
               std::runtime_error);
  EXPECT_THROW({ auto merged14 = merge_topology(topology1, topology4); },
               std::runtime_error);
  EXPECT_THROW({ auto merged23 = merge_topology(topology2, topology3); },
               std::runtime_error);
  EXPECT_THROW({ auto merged24 = merge_topology(topology2, topology4); },
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

  if (nprocs == 1) {
    // Failure tests because these are identical
    EXPECT_THROW(
        { std::size_t diff0_01 = diff_toplogy(topology0, topology01); },
        std::runtime_error);
    EXPECT_THROW(
        { std::size_t diff0_02 = diff_toplogy(topology0, topology02); },
        std::runtime_error);
    EXPECT_THROW(
        { std::size_t diff1_12 = diff_toplogy(topology1, topology12); },
        std::runtime_error);
    EXPECT_THROW(
        { std::size_t diff2_12 = diff_toplogy(topology2, topology12); },
        std::runtime_error);
  } else {
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

    // Failure tests because more than two elements are different
    EXPECT_THROW({ std::size_t diff03 = diff_toplogy(topology0, topology3); },
                 std::runtime_error);
  }
}

template <typename T, typename LayoutType>
void test_buffer_extents() {
  using extents_type        = std::array<std::size_t, 4>;
  using buffer_extents_type = std::array<std::size_t, 5>;
  using topology_type       = std::array<std::size_t, 4>;
  using ViewType            = Kokkos::View<T****, LayoutType, execution_space>;
  const std::size_t n0 = 13, n1 = 8, n2 = 17, n3 = 5;
  const std::size_t p0 = 2, p1 = 3;

  // Global View
  extents_type extents{n0, n1, n2, n3};

  // X-pencil
  topology_type topology0 = {1, p0, p1, 1};

  // Y-pencil
  topology_type topology1 = {p0, 1, p1, 1};

  // Z-pencil
  topology_type topology2 = {p0, p1, 1, 1};

  buffer_extents_type ref_buffer_01, ref_buffer_12;
  if (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    ref_buffer_01 = {(n0 - 1) / p0 + 1, (n1 - 1) / p0 + 1, (n2 - 1) / p1 + 1,
                     n3, p0};
    ref_buffer_12 = {(n0 - 1) / p0 + 1, (n1 - 1) / p1 + 1, (n2 - 1) / p1 + 1,
                     n3, p1};
  } else {
    ref_buffer_01 = {p0, (n0 - 1) / p0 + 1, (n1 - 1) / p0 + 1,
                     (n2 - 1) / p1 + 1, n3};
    ref_buffer_12 = {p1, (n0 - 1) / p0 + 1, (n1 - 1) / p1 + 1,
                     (n2 - 1) / p1 + 1, n3};
  }

  buffer_extents_type buffer_01 =
      get_buffer_extents<LayoutType>(extents, topology0, topology1);
  buffer_extents_type buffer_12 =
      get_buffer_extents<LayoutType>(extents, topology1, topology2);

  EXPECT_TRUE(buffer_01 == ref_buffer_01);
  EXPECT_TRUE(buffer_12 == ref_buffer_12);

  // In valid, because you cannot go from X to Z in one exchange
  EXPECT_THROW(
      {
        buffer_extents_type buffer_02 =
            get_buffer_extents<LayoutType>(extents, topology0, topology2);
      },
      std::runtime_error);
}

template <typename T, typename LayoutType>
void test_next_extents() {
  using extents_type  = std::array<std::size_t, 4>;
  using topology_type = std::array<std::size_t, 4>;
  using map_type      = std::array<std::size_t, 4>;

  using ViewType       = Kokkos::View<T****, LayoutType, execution_space>;
  const std::size_t n0 = 13, n1 = 8, n2 = 17, n3 = 5;
  const std::size_t p0 = 2, p1 = 3;

  // Global View
  extents_type extents{n0, n1, n2, n3};

  // X-pencil
  topology_type topology0 = {1, p0, p1, 1};
  map_type map0           = {0, 1, 2, 3};

  // Y-pencil
  topology_type topology1 = {p0, 1, p1, 1};
  map_type map1           = {0, 2, 3, 1};

  // Z-pencil
  topology_type topology2 = {p0, p1, 1, 1};
  map_type map2           = {0, 3, 1, 2};

  extents_type ref_next_0{n0, (n1 - 1) / p0 + 1, (n2 - 1) / p1 + 1, n3};
  extents_type ref_next_1{(n0 - 1) / p0 + 1, (n2 - 1) / p1 + 1, n3, n1};
  extents_type ref_next_2{(n0 - 1) / p0 + 1, n3, (n1 - 1) / p1 + 1, n2};

  extents_type next_0 = get_next_extents(extents, topology0, map0);
  extents_type next_1 = get_next_extents(extents, topology1, map1);
  extents_type next_2 = get_next_extents(extents, topology2, map2);

  EXPECT_TRUE(next_0 == ref_next_0);
  EXPECT_TRUE(next_1 == ref_next_1);
  EXPECT_TRUE(next_2 == ref_next_2);
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

    EXPECT_THROW({ auto topo = get_topology_type(topology1D_type{0}); },
                 std::runtime_error);

    auto topo2_1 = get_topology_type(topology2_1);
    auto topo2_2 = get_topology_type(topology2_2);
    EXPECT_EQ(topo2_1, TopologyType::Slab);
    EXPECT_EQ(topo2_2, TopologyType::Slab);

    EXPECT_THROW(
        {
          auto topo = get_topology_type(topology2D_type{0, nprocs});
        },
        std::runtime_error);
    EXPECT_THROW(
        {
          auto topo = get_topology_type(topology2D_type{nprocs, 0});
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
    EXPECT_THROW({ auto topo3_1 = get_topology_type(topology3_1); },
                 std::runtime_error);
    EXPECT_THROW({ auto topo3_2 = get_topology_type(topology3_2); },
                 std::runtime_error);
    EXPECT_THROW({ auto topo3_3 = get_topology_type(topology3_3); },
                 std::runtime_error);
    EXPECT_THROW({ auto topo3_4 = get_topology_type(topology3_4); },
                 std::runtime_error);
    EXPECT_THROW({ auto topo3_5 = get_topology_type(topology3_5); },
                 std::runtime_error);

    // 4D topology
    EXPECT_THROW({ auto topo4_1 = get_topology_type(topology4_1); },
                 std::runtime_error);
    EXPECT_THROW({ auto topo4_2 = get_topology_type(topology4_2); },
                 std::runtime_error);
    EXPECT_THROW({ auto topo4_3 = get_topology_type(topology4_3); },
                 std::runtime_error);
    EXPECT_THROW({ auto topo4_4 = get_topology_type(topology4_4); },
                 std::runtime_error);
    EXPECT_THROW({ auto topo4_5 = get_topology_type(topology4_5); },
                 std::runtime_error);
    EXPECT_THROW({ auto topo4_6 = get_topology_type(topology4_6); },
                 std::runtime_error);
    EXPECT_THROW({ auto topo4_7 = get_topology_type(topology4_7); },
                 std::runtime_error);
    EXPECT_THROW({ auto topo4_8 = get_topology_type(topology4_8); },
                 std::runtime_error);
  }
}

void test_get_required_allocation_size() {
  using topology1D_type = std::array<std::size_t, 1>;
  using topology2D_type = std::array<std::size_t, 2>;
  using topology3D_type = std::array<std::size_t, 3>;
  using topology4D_type = std::array<std::size_t, 4>;

  topology1D_type topology1{2}, topology1_2{4};
  topology2D_type topology2{1, 2}, topology2_2{4, 2};
  topology3D_type topology3{1, 2, 1}, topology3_2{4, 2, 1};
  topology4D_type topology4{1, 4, 1, 1}, topology4_2{1, 1, 4, 2};

  std::vector<topology1D_type> topology1_vec = {topology1, topology1_2};
  std::vector<topology2D_type> topology2_vec = {topology2, topology2_2};
  std::vector<topology3D_type> topology3_vec = {topology3, topology3_2};
  std::vector<topology4D_type> topology4_vec = {topology4, topology4_2};

  auto size1 = get_required_allocation_size(topology1_vec);
  auto size2 = get_required_allocation_size(topology2_vec);
  auto size3 = get_required_allocation_size(topology3_vec);
  auto size4 = get_required_allocation_size(topology4_vec);

  std::size_t ref_size1 = 4;
  std::size_t ref_size2 = 8;
  std::size_t ref_size3 = 8;
  std::size_t ref_size4 = 8;

  EXPECT_EQ(size1, ref_size1);
  EXPECT_EQ(size2, ref_size2);
  EXPECT_EQ(size3, ref_size3);
  EXPECT_EQ(size4, ref_size4);
}

}  // namespace

TYPED_TEST_SUITE(TestExtents, test_types);

TYPED_TEST(TestExtents, BufferExtents) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_buffer_extents<float_type, layout_type>();
}

TYPED_TEST(TestExtents, NextExtents) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_next_extents<float_type, layout_type>();
}

TEST(TestRequiredAllocationSize, 1Dto4D) {
  test_get_required_allocation_size();
}

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

INSTANTIATE_TEST_SUITE_P(TopologyTests, TopologyParamTests,
                         ::testing::Values(1, 2, 3, 4, 5, 6));
