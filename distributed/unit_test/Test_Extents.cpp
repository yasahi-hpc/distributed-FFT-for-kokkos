#include <mpi.h>
#include <gtest/gtest.h>
#include <iostream>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_Extents.hpp"
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

template <typename T, typename LayoutType>
void test_buffer_extents() {
  using extents_type        = std::array<std::size_t, 4>;
  using buffer_extents_type = std::array<std::size_t, 5>;
  using topology_type       = std::array<std::size_t, 4>;
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
      KokkosFFT::Distributed::Impl::get_buffer_extents<LayoutType>(
          extents, topology0, topology1);
  buffer_extents_type buffer_12 =
      KokkosFFT::Distributed::Impl::get_buffer_extents<LayoutType>(
          extents, topology1, topology2);

  EXPECT_TRUE(buffer_01 == ref_buffer_01);
  EXPECT_TRUE(buffer_12 == ref_buffer_12);

  // In valid, because you cannot go from X to Z in one exchange
  EXPECT_THROW(
      {
        [[maybe_unused]] buffer_extents_type buffer_02 =
            KokkosFFT::Distributed::Impl::get_buffer_extents<LayoutType>(
                extents, topology0, topology2);
      },
      std::runtime_error);
}

template <typename T>
void test_next_extents() {
  using extents_type  = std::array<std::size_t, 4>;
  using topology_type = std::array<std::size_t, 4>;
  using map_type      = std::array<std::size_t, 4>;

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

  extents_type next_0 =
      KokkosFFT::Distributed::Impl::get_next_extents(extents, topology0, map0);
  extents_type next_1 =
      KokkosFFT::Distributed::Impl::get_next_extents(extents, topology1, map1);
  extents_type next_2 =
      KokkosFFT::Distributed::Impl::get_next_extents(extents, topology2, map2);

  EXPECT_TRUE(next_0 == ref_next_0);
  EXPECT_TRUE(next_1 == ref_next_1);
  EXPECT_TRUE(next_2 == ref_next_2);
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

  auto size1 =
      KokkosFFT::Distributed::Impl::get_required_allocation_size(topology1_vec);
  auto size2 =
      KokkosFFT::Distributed::Impl::get_required_allocation_size(topology2_vec);
  auto size3 =
      KokkosFFT::Distributed::Impl::get_required_allocation_size(topology3_vec);
  auto size4 =
      KokkosFFT::Distributed::Impl::get_required_allocation_size(topology4_vec);

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
  using float_type = typename TestFixture::float_type;

  test_next_extents<float_type>();
}

TEST(TestRequiredAllocationSize, 1Dto4D) {
  test_get_required_allocation_size();
}
