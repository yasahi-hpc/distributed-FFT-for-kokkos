#include <mpi.h>
#include <sstream>
#include <iostream>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "nccl/KokkosFFT_Distributed_NCCL_Types.hpp"

namespace {
using test_types = ::testing::Types<int, std::int32_t, std::uint32_t,
                                    std::int64_t, std::uint64_t, float, double>;

template <typename T>
struct TestNCCLType : public ::testing::Test {
  using value_type = T;
};

template <typename T>
void test_nccl_data_type() {
  ncclDataType_t nccl_data_type =
      KokkosFFT::Distributed::Impl::nccl_datatype_v<T>;

  if constexpr (std::is_same_v<T, char>) {
    ASSERT_EQ(nccl_data_type, ncclChar);
  } else if constexpr (std::is_same_v<T, std::int8_t>) {
    ASSERT_EQ(nccl_data_type, ncclInt8);
  } else if constexpr (std::is_same_v<T, std::uint8_t>) {
    ASSERT_EQ(nccl_data_type, ncclUint8);
  } else if constexpr (std::is_same_v<T, int>) {
    ASSERT_EQ(nccl_data_type, ncclInt);
  } else if constexpr (std::is_same_v<T, std::int32_t>) {
    ASSERT_EQ(nccl_data_type, ncclInt32);
  } else if constexpr (std::is_same_v<T, std::uint32_t>) {
    ASSERT_EQ(nccl_data_type, ncclUint32);
  } else if constexpr (std::is_same_v<T, std::int64_t>) {
    ASSERT_EQ(nccl_data_type, ncclInt64);
  } else if constexpr (std::is_same_v<T, std::uint64_t>) {
    ASSERT_EQ(nccl_data_type, ncclUint64);
  } else if constexpr (std::is_same_v<T, float>) {
    ASSERT_EQ(nccl_data_type, ncclFloat);
  } else if constexpr (std::is_same_v<T, double>) {
    ASSERT_EQ(nccl_data_type, ncclDouble);
  }
}

}  // namespace

TYPED_TEST_SUITE(TestNCCLType, test_types);

TYPED_TEST(TestNCCLType, test_convert_scalar_type) {
  using value_type = typename TestFixture::value_type;
  test_nccl_data_type<value_type>();
}
