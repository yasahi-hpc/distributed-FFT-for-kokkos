#include <array>
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
  using value_type  = typename T::first_type;
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
  topology_type topology0{1, p0, p1, 1};

  // Y-pencil
  topology_type topology1{p0, 1, p1, 1};

  // Z-pencil
  topology_type topology2{p0, p1, 1, 1};

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

  auto buffer_01 =
      KokkosFFT::Distributed::Impl::compute_buffer_extents<LayoutType>(
          extents, topology0, topology1);
  auto buffer_12 =
      KokkosFFT::Distributed::Impl::compute_buffer_extents<LayoutType>(
          extents, topology1, topology2);

  EXPECT_TRUE(buffer_01 == ref_buffer_01);
  EXPECT_TRUE(buffer_12 == ref_buffer_12);

  // Invalid, because you cannot go from X to Z in one exchange
  EXPECT_THROW(
      {
        [[maybe_unused]] auto buffer_02 =
            KokkosFFT::Distributed::Impl::compute_buffer_extents<LayoutType>(
                extents, topology0, topology2);
      },
      std::runtime_error);
}

template <typename ContainerType, typename iType>
void test_compute_fft_extents(iType nprocs) {
  using extents_type = std::array<iType, 3>;
  extents_type in_extents{nprocs, 3, 8}, out_extents{nprocs, 3, 5};
  ContainerType map012{0, 1, 2}, map021{0, 2, 1}, map102{1, 0, 2},
      map120{1, 2, 0}, map201{2, 0, 1}, map210{2, 1, 0};
  auto fft_extents012 = KokkosFFT::Distributed::Impl::compute_fft_extents(
      in_extents, out_extents, map012);
  auto fft_extents021 = KokkosFFT::Distributed::Impl::compute_fft_extents(
      in_extents, out_extents, map021);
  auto fft_extents102 = KokkosFFT::Distributed::Impl::compute_fft_extents(
      in_extents, out_extents, map102);
  auto fft_extents120 = KokkosFFT::Distributed::Impl::compute_fft_extents(
      in_extents, out_extents, map120);
  auto fft_extents201 = KokkosFFT::Distributed::Impl::compute_fft_extents(
      in_extents, out_extents, map201);
  auto fft_extents210 = KokkosFFT::Distributed::Impl::compute_fft_extents(
      in_extents, out_extents, map210);

  extents_type ref_fft_extents012{nprocs, 3, 8},
      ref_fft_extents021{nprocs, 8, 3}, ref_fft_extents102{3, nprocs, 8},
      ref_fft_extents120{3, 8, nprocs}, ref_fft_extents201{8, nprocs, 3},
      ref_fft_extents210{8, 3, nprocs};

  EXPECT_EQ(fft_extents012, ref_fft_extents012);
  EXPECT_EQ(fft_extents021, ref_fft_extents021);
  EXPECT_EQ(fft_extents102, ref_fft_extents102);
  EXPECT_EQ(fft_extents120, ref_fft_extents120);
  EXPECT_EQ(fft_extents201, ref_fft_extents201);
  EXPECT_EQ(fft_extents210, ref_fft_extents210);
}

/// \brief compute_inplace_allocation_size: the larger of the input and the
/// output in bytes, in elements of the output type, rounded up
void test_compute_inplace_allocation_size() {
  using KokkosFFT::Distributed::compute_inplace_allocation_size;
  using extents_type = std::array<std::size_t, 2>;

  // R2C (8, 7) -> (8, 4): 56 doubles = 448 bytes, 32 complex = 512 bytes
  EXPECT_EQ((compute_inplace_allocation_size<double, Kokkos::complex<double>>(
                extents_type{8, 7}, extents_type{8, 4})),
            std::size_t(32));

  // C2C: the input and the output have the same size
  EXPECT_EQ((compute_inplace_allocation_size<Kokkos::complex<float>,
                                             Kokkos::complex<float>>(
                extents_type{8, 7}, extents_type{7, 8})),
            std::size_t(56));

  // Uneven distribution, where the input is the larger one: input (8, 2),
  // 16 doubles = 128 bytes, output (1, 7), 7 complex = 112 bytes. 8 complex
  // are needed, more than the 7 of the output.
  EXPECT_EQ((compute_inplace_allocation_size<double, Kokkos::complex<double>>(
                extents_type{8, 2}, extents_type{1, 7})),
            std::size_t(8));

  // Rounded up: input (1, 5), 5 floats = 20 bytes, output (1, 1), one
  // complex<double> = 16 bytes. 2 complex<double> are needed.
  EXPECT_EQ((compute_inplace_allocation_size<float, Kokkos::complex<double>>(
                extents_type{1, 5}, extents_type{1, 1})),
            std::size_t(2));

  // Empty local extents on this rank
  EXPECT_EQ((compute_inplace_allocation_size<double, Kokkos::complex<double>>(
                extents_type{0, 7}, extents_type{0, 4})),
            std::size_t(0));
}

/// \brief are_overlapping compares byte ranges, without accessing the data
void test_are_overlapping() {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using RealViewType    = Kokkos::View<double*, execution_space>;
  using ComplexViewType =
      Kokkos::View<Kokkos::complex<double>*, execution_space>;
  using KokkosFFT::Distributed::Impl::are_overlapping;

  RealViewType buffer("buffer", 16), other("other", 16);
  double* ptr = buffer.data();

  RealViewType a(ptr, 8);
  RealViewType a_same(ptr, 4);          // same start, shorter
  RealViewType b_partial(ptr + 4, 8);   // starts inside a
  RealViewType c_adjacent(ptr + 8, 8);  // starts right after a
  ComplexViewType a_complex(reinterpret_cast<Kokkos::complex<double>*>(ptr),
                            4);  // a seen as complex
  RealViewType empty(ptr, 0);

  EXPECT_TRUE(are_overlapping(a, a));
  EXPECT_TRUE(are_overlapping(a, a_same));
  EXPECT_TRUE(are_overlapping(a, b_partial));
  EXPECT_TRUE(are_overlapping(b_partial, a));
  EXPECT_FALSE(are_overlapping(a, c_adjacent));
  EXPECT_FALSE(are_overlapping(c_adjacent, a));
  EXPECT_TRUE(are_overlapping(a, a_complex));
  EXPECT_FALSE(are_overlapping(a_complex, c_adjacent));  // 64 bytes
  EXPECT_FALSE(are_overlapping(a, empty));
  EXPECT_FALSE(are_overlapping(empty, a));
  EXPECT_FALSE(are_overlapping(buffer, other));
}

}  // namespace

TEST(TestInplaceHelpers, compute_inplace_allocation_size) {
  test_compute_inplace_allocation_size();
}

TEST(TestInplaceHelpers, are_overlapping) { test_are_overlapping(); }

TYPED_TEST_SUITE(TestExtents, test_types);

TYPED_TEST(TestExtents, BufferExtents) {
  using value_type  = typename TestFixture::value_type;
  using layout_type = typename TestFixture::layout_type;

  test_buffer_extents<value_type, layout_type>();
}

TYPED_TEST(TestExtents, fft_extents_of_array) {
  using value_type = typename TestFixture::value_type;
  using array_type = std::array<value_type, 3>;
  for (value_type nprocs = 1; nprocs <= 6; ++nprocs) {
    test_compute_fft_extents<array_type, value_type>(nprocs);
  }
}
