#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "Transpose.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using test_types      = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>,
                                    std::pair<double, Kokkos::LayoutLeft>,
                                    std::pair<double, Kokkos::LayoutRight>>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct TestTranspose : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;

  int m_rank   = 0;
  int m_nprocs = 1;

  virtual void SetUp() {
    ::MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &m_nprocs);
  }
};

template <typename T, typename LayoutType>
void test_transpose_view2D_XtoY(int rank, int nprocs) {
  using ViewType = Kokkos::View<T**, LayoutType, execution_space>;
  const int n0 = 16, n1 = 15;
  const int n0_local = ((n0 - 1) / nprocs) + 1;
  const int n1_local = ((n1 - 1) / nprocs) + 1;
  ViewType in("in", n0, n1_local), out("out", n0_local, n1),
      ref("ref", n0_local, n1);

  double dx = M_PI * 2.0 / static_cast<double>(n0);
  double dy = M_PI * 2.0 / static_cast<double>(n1);

  auto h_in = Kokkos::create_mirror_view(in);
  for (int i1 = 0; i1 < n1_local; i1++) {
    for (int i0 = 0; i0 < n0; i0++) {
      int gi1      = i1 + n1_local * rank;
      double x     = i0 * dx;
      double y     = gi1 * dy;
      h_in(i0, i1) = std::cos(x) * std::sin(y);
    }
  }

  for (int i1 = 0; i1 < n1; i1++) {
    for (int i0 = 0; i0 < n0_local; i0++) {
      int gi0       = i0 + n0_local * rank;
      double x      = gi0 * dx;
      double y      = i1 * dy;
      h_ref(i0, i1) = std::cos(x) * std::sin(y);
    }
  }
  Kokkos::deep_copy(in, h_in);

  Transpose(in, out)();

  auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  auto h_ref = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ref);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;
  for (int i1 = 0; i1 < n1; i1++) {
    for (int i0 = 0; i0 < n0_local; i0++) {
      auto diff = Kokkos::abs(h_out(i0, i1) - h_ref(i0, i1));
      EXPECT_LE(diff, epsilon * Kokkos::abs(h_ref(i0, i1)));
    }
  }
}

template <typename T, typename LayoutType>
void test_transpose_view2D_Y2X() {
  using ViewType = Kokkos::View<T**, LayoutType, execution_space>;
  // ViewType in("in", );
}

}  // namespace

TYPED_TEST_SUITE(TestTranspose, test_types);

TYPED_TEST(TestTranspose, View2D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_transpose_view2D_XtoY<float_type, layout_type>();
}
