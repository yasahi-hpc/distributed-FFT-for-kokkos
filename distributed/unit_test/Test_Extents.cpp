#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_Extents.hpp"

namespace {
using test_types =
    ::testing::Types<std::pair<int, Kokkos::LayoutLeft>,
                     std::pair<int, Kokkos::LayoutRight>,
                     std::pair<std::size_t, Kokkos::LayoutLeft>,
                     std::pair<std::size_t, Kokkos::LayoutRight>>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct TestExtents : public ::testing::Test {
  using int_type    = typename T::first_type;
  using layout_type = typename T::second_type;
};

template <typename IntType, typename LayoutType>
void test_buffer_extents() {
  using extents_type        = std::array<IntType, 4>;
  using buffer_extents_type = std::array<IntType, 5>;
  using topology_type       = std::array<IntType, 4>;
  const IntType n0 = 13, n1 = 8, n2 = 17, n3 = 5;
  const IntType p0 = 2, p1 = 3;

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

template <typename IntType>
void test_next_extents() {
  using extents_type  = std::array<IntType, 4>;
  using topology_type = std::array<IntType, 4>;
  using map_type      = std::array<IntType, 4>;

  const IntType n0 = 13, n1 = 8, n2 = 17, n3 = 5;
  const IntType p0 = 2, p1 = 3;

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

}  // namespace

TYPED_TEST_SUITE(TestExtents, test_types);

TYPED_TEST(TestExtents, BufferExtents) {
  using int_type    = typename TestFixture::int_type;
  using layout_type = typename TestFixture::layout_type;

  test_buffer_extents<int_type, layout_type>();
}

TYPED_TEST(TestExtents, NextExtents) {
  using int_type = typename TestFixture::int_type;

  test_next_extents<int_type>();
}
