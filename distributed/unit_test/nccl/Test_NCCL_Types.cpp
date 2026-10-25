#include <mpi.h>
#include <type_traits>
#include <utility>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "nccl/KokkosFFT_Distributed_NCCL_Types.hpp"
#include "nccl/KokkosFFT_Distributed_NCCL_Comm.hpp"

namespace {
using test_types =
    ::testing::Types<char, std::int8_t, std::uint8_t, int, std::int32_t,
                     std::uint32_t, std::int64_t, std::uint64_t, float, double>;

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

/// \brief ScopedNCCLComm is movable but not copyable. A move hands over the
/// NCCL communicator and leaves nullptr behind, which the destructor and the
/// move assignment must not destroy.
void test_nccl_comm_move() {
  using CommType = KokkosFFT::Distributed::Impl::ScopedNCCLComm;
  static_assert(!std::is_default_constructible_v<CommType>);
  static_assert(!std::is_copy_constructible_v<CommType>);
  static_assert(!std::is_copy_assignable_v<CommType>);
  static_assert(std::is_nothrow_move_constructible_v<CommType>);
  static_assert(std::is_nothrow_move_assignable_v<CommType>);

  int rank = 0, size = 1;
  ::MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ::MPI_Comm_size(MPI_COMM_WORLD, &size);

  CommType comm0(MPI_COMM_WORLD);
  const ncclComm_t handle = comm0.comm();
  EXPECT_NE(handle, nullptr);

  // Move construction: the communicator is handed over
  CommType comm1(std::move(comm0));
  EXPECT_EQ(comm1.comm(), handle);
  EXPECT_EQ(comm1.rank(), rank);
  EXPECT_EQ(comm1.size(), size);
  EXPECT_EQ(comm0.comm(), nullptr);  // NOLINT(bugprone-use-after-move)

  // Move assignment over a communicator that owns one: the old one is
  // destroyed (collective, on all ranks) and the new one is handed over
  CommType comm2(MPI_COMM_WORLD);
  comm2 = std::move(comm1);
  EXPECT_EQ(comm2.comm(), handle);
  EXPECT_EQ(comm1.comm(), nullptr);  // NOLINT(bugprone-use-after-move)

  // Move assignment from a moved-from communicator: nothing to hand over
  CommType comm3(MPI_COMM_WORLD);
  comm3 = std::move(comm0);
  EXPECT_EQ(comm3.comm(), nullptr);

  // comm0, comm1 and comm3 are destroyed without a communicator
}

}  // namespace

TEST(TestNCCLComm, move) { test_nccl_comm_move(); }

TYPED_TEST_SUITE(TestNCCLType, test_types);

TYPED_TEST(TestNCCLType, test_convert_scalar_type) {
  using value_type = typename TestFixture::value_type;
  test_nccl_data_type<value_type>();
}
