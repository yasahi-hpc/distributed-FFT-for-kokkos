#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "Reshape.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using test_types      = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>,
                                    std::pair<double, Kokkos::LayoutLeft>,
                                    std::pair<double, Kokkos::LayoutRight>>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct CompileTestCreateViewType : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;

  virtual void SetUp() {
    GTEST_SKIP() << "Skipping all tests for this fixture";
  }
};

template <typename T, typename LayoutType>
void test_create_view_0D() {
  using RefViewType = Kokkos::View<T, LayoutType, execution_space>;
  using ViewType =
      Kokkos::View<add_pointer_n_t<T, 0>, LayoutType, execution_space>;

  testing::StaticAssertTypeEq<ViewType, RefViewType>();
}

template <typename T, typename LayoutType>
void test_create_view_1D() {
  using RefViewType = Kokkos::View<T*, LayoutType, execution_space>;
  using ViewType =
      Kokkos::View<add_pointer_n_t<T, 1>, LayoutType, execution_space>;

  testing::StaticAssertTypeEq<ViewType, RefViewType>();
}

template <typename T, typename LayoutType>
void test_create_view_2D() {
  using RefViewType = Kokkos::View<T**, LayoutType, execution_space>;
  using ViewType =
      Kokkos::View<add_pointer_n_t<T, 2>, LayoutType, execution_space>;

  testing::StaticAssertTypeEq<ViewType, RefViewType>();
}

template <typename T, typename LayoutType>
void test_create_view_3D() {
  using RefViewType = Kokkos::View<T***, LayoutType, execution_space>;
  using ViewType =
      Kokkos::View<add_pointer_n_t<T, 3>, LayoutType, execution_space>;

  testing::StaticAssertTypeEq<ViewType, RefViewType>();
}

template <typename T, typename LayoutType>
void test_create_view_4D() {
  using RefViewType = Kokkos::View<T****, LayoutType, execution_space>;
  using ViewType =
      Kokkos::View<add_pointer_n_t<T, 4>, LayoutType, execution_space>;

  testing::StaticAssertTypeEq<ViewType, RefViewType>();
}

template <typename T, typename LayoutType>
void test_create_view_5D() {
  using RefViewType = Kokkos::View<T*****, LayoutType, execution_space>;
  using ViewType =
      Kokkos::View<add_pointer_n_t<T, 5>, LayoutType, execution_space>;

  testing::StaticAssertTypeEq<ViewType, RefViewType>();
}

template <typename T, typename LayoutType>
void test_create_view_6D() {
  using RefViewType = Kokkos::View<T******, LayoutType, execution_space>;
  using ViewType =
      Kokkos::View<add_pointer_n_t<T, 6>, LayoutType, execution_space>;

  testing::StaticAssertTypeEq<ViewType, RefViewType>();
}

template <typename T, typename LayoutType>
void test_create_view_7D() {
  using RefViewType = Kokkos::View<T*******, LayoutType, execution_space>;
  using ViewType =
      Kokkos::View<add_pointer_n_t<T, 7>, LayoutType, execution_space>;

  testing::StaticAssertTypeEq<ViewType, RefViewType>();
}

template <typename T, typename LayoutType>
void test_create_view_8D() {
  using RefViewType = Kokkos::View<T********, LayoutType, execution_space>;
  using ViewType =
      Kokkos::View<add_pointer_n_t<T, 8>, LayoutType, execution_space>;

  testing::StaticAssertTypeEq<ViewType, RefViewType>();
}
}  // namespace

TYPED_TEST_SUITE(CompileTestCreateViewType, test_types);

TYPED_TEST(CompileTestCreateViewType, create_view_0D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_create_view_0D<float_type, layout_type>();
}

TYPED_TEST(CompileTestCreateViewType, create_view_1D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_create_view_1D<float_type, layout_type>();
}

TYPED_TEST(CompileTestCreateViewType, create_view_2D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_create_view_2D<float_type, layout_type>();
}

TYPED_TEST(CompileTestCreateViewType, create_view_3D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_create_view_3D<float_type, layout_type>();
}

TYPED_TEST(CompileTestCreateViewType, create_view_4D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_create_view_4D<float_type, layout_type>();
}

TYPED_TEST(CompileTestCreateViewType, create_view_5D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_create_view_5D<float_type, layout_type>();
}

TYPED_TEST(CompileTestCreateViewType, create_view_6D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_create_view_6D<float_type, layout_type>();
}

TYPED_TEST(CompileTestCreateViewType, create_view_7D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_create_view_7D<float_type, layout_type>();
}

TYPED_TEST(CompileTestCreateViewType, create_view_8D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_create_view_8D<float_type, layout_type>();
}
