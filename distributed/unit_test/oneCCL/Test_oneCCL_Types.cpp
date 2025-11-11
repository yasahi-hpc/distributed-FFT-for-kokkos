#include <mpi.h>
#include <sstream>
#include <iostream>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "oneCCL/KokkosFFT_Distributed_oneCCL_Types.hpp"

namespace {
using test_types =
    ::testing::Types<std::int8_t, std::uint8_t, std::int16_t, std::uint16_t,
                     int, std::int32_t, std::uint32_t, std::int64_t,
                     std::uint64_t, float, double>;

template <typename T>
struct TestoneCCLType : public ::testing::Test {
  using value_type = T;
};

template <typename T>
void test_oneCCL_data_type() {
  ccl::datatype oneCCL_data_type =
      KokkosFFT::Distributed::Impl::oneccl_datatype_v<T>;

  if constexpr (std::is_same_v<T, std::int8_t>) {
    ASSERT_EQ(oneCCL_data_type, ccl::datatype::int8);
  } else if constexpr (std::is_same_v<T, std::uint8_t>) {
    ASSERT_EQ(oneCCL_data_type, ccl::datatype::uint8);
  } else if constexpr (std::is_same_v<T, std::int16_t>) {
    ASSERT_EQ(oneCCL_data_type, ccl::datatype::int16);
  } else if constexpr (std::is_same_v<T, std::uint16_t>) {
    ASSERT_EQ(oneCCL_data_type, ccl::datatype::uint16);
  } else if constexpr (std::is_same_v<T, int>) {
    ASSERT_EQ(oneCCL_data_type, ccl::datatype::int32);
  } else if constexpr (std::is_same_v<T, std::int32_t>) {
    ASSERT_EQ(oneCCL_data_type, ccl::datatype::int32);
  } else if constexpr (std::is_same_v<T, std::uint32_t>) {
    ASSERT_EQ(oneCCL_data_type, ccl::datatype::uint32);
  } else if constexpr (std::is_same_v<T, std::int64_t>) {
    ASSERT_EQ(oneCCL_data_type, ccl::datatype::int64);
  } else if constexpr (std::is_same_v<T, std::uint64_t>) {
    ASSERT_EQ(oneCCL_data_type, ccl::datatype::uint64);
  } else if constexpr (std::is_same_v<T, float>) {
    ASSERT_EQ(oneCCL_data_type, ccl::datatype::float32);
  } else if constexpr (std::is_same_v<T, double>) {
    ASSERT_EQ(oneCCL_data_type, ccl::datatype::float64);
  }
}

}  // namespace

TYPED_TEST_SUITE(TestoneCCLType, test_types);

TYPED_TEST(TestoneCCLType, test_convert_scalar_type) {
  using value_type = typename TestFixture::value_type;
  test_oneCCL_data_type<value_type>();
}
