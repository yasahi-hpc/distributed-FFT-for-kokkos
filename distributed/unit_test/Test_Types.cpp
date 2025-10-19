#include <gtest/gtest.h>
#include <array>
#include <algorithm>
#include <type_traits>
#include "KokkosFFT_Distributed_Types.hpp"

// Test fixture for Topology class
template <typename TopologyType>
struct TopologyTest : public ::testing::Test {
  using value_type                  = typename TopologyType::value_type;
  static constexpr std::size_t size = TopologyType{}.size();

  std::array<value_type, size> test_data;

 protected:
  virtual void SetUp() override {
    // Initialize test data
    for (std::size_t i = 0; i < size; ++i) {
      test_data[i] = static_cast<value_type>(i + 1);
    }
  }
};

// Type definitions for parameterized tests
using TopologyTypes =
    ::testing::Types<KokkosFFT::Distributed::Topology<int, 3>,
                     KokkosFFT::Distributed::Topology<double, 4>,
                     KokkosFFT::Distributed::Topology<float, 5>,
                     KokkosFFT::Distributed::Topology<std::size_t, 2> >;

TYPED_TEST_SUITE(TopologyTest, TopologyTypes);

// Test type definitions and aliases
TEST(TopologyTypeTest, TypeDefinitions) {
  using TestTopology = KokkosFFT::Distributed::Topology<int, 3>;

  testing::StaticAssertTypeEq<typename TestTopology::value_type, int>();
  testing::StaticAssertTypeEq<typename TestTopology::size_type, std::size_t>();
  testing::StaticAssertTypeEq<typename TestTopology::difference_type,
                              std::ptrdiff_t>();
  testing::StaticAssertTypeEq<typename TestTopology::reference, int&>();
  testing::StaticAssertTypeEq<typename TestTopology::const_reference,
                              const int&>();
  testing::StaticAssertTypeEq<typename TestTopology::pointer, int*>();
  testing::StaticAssertTypeEq<typename TestTopology::const_pointer,
                              const int*>();
  testing::StaticAssertTypeEq<typename TestTopology::layout_type,
                              Kokkos::LayoutRight>();
}

// Test default constructor
TYPED_TEST(TopologyTest, DefaultConstructor) {
  TypeParam topology;
  EXPECT_EQ(topology.size(), this->test_data.size());
  EXPECT_EQ(topology.empty(), this->test_data.empty());
}

// Test constructor from std::array
TYPED_TEST(TopologyTest, ArrayConstructor) {
  TypeParam topology(this->test_data);

  for (std::size_t i = 0; i < topology.size(); ++i) {
    EXPECT_EQ(topology[i], this->test_data[i]);
  }
}

// Test constructor from initializer list
TEST(TopologyConstructorTest, InitializerListConstructor) {
  KokkosFFT::Distributed::Topology<int, 3> topology{1, 2, 3};

  EXPECT_EQ(topology[0], 1);
  EXPECT_EQ(topology[1], 2);
  EXPECT_EQ(topology[2], 3);
}

// Test initializer list constructor with wrong size
TEST(TopologyConstructorTest, InitializerListWrongSize) {
  EXPECT_THROW(({
                 KokkosFFT::Distributed::Topology<int, 3> topology{
                     1, 2, 3, 4};  // Too many elements
               }),
               std::length_error);

  EXPECT_THROW(({
                 KokkosFFT::Distributed::Topology<int, 3> topology{
                     1, 2};  // Too few elements
               }),
               std::length_error);
}

// Test copy constructor
TYPED_TEST(TopologyTest, CopyConstructor) {
  TypeParam original(this->test_data);
  TypeParam copy(original);

  EXPECT_EQ(original, copy);
  for (std::size_t i = 0; i < copy.size(); ++i) {
    EXPECT_EQ(copy[i], original[i]);
  }
}

// Test move constructor
TYPED_TEST(TopologyTest, MoveConstructor) {
  TypeParam original(this->test_data);
  TypeParam expected = original;
  TypeParam moved(std::move(original));

  EXPECT_EQ(moved, expected);
}

// Test copy assignment
TYPED_TEST(TopologyTest, CopyAssignment) {
  TypeParam original(this->test_data);
  TypeParam assigned;

  assigned = original;
  EXPECT_EQ(assigned, original);
}

// Test move assignment
TYPED_TEST(TopologyTest, MoveAssignment) {
  TypeParam original(this->test_data);
  TypeParam expected = original;
  TypeParam assigned;

  assigned = std::move(original);
  EXPECT_EQ(assigned, expected);
}

// Test element access with bounds checking
TYPED_TEST(TopologyTest, ElementAccessAt) {
  TypeParam topology(this->test_data);

  for (std::size_t i = 0; i < topology.size(); ++i) {
    EXPECT_EQ(topology.at(i), this->test_data[i]);
  }

  // Test const version
  const TypeParam& const_topology = topology;
  for (std::size_t i = 0; i < const_topology.size(); ++i) {
    EXPECT_EQ(const_topology.at(i), this->test_data[i]);
  }
}

// Test bounds checking in at() method
TYPED_TEST(TopologyTest, ElementAccessAtBoundsCheck) {
  TypeParam topology(this->test_data);

  EXPECT_THROW(topology.at(topology.size()), std::out_of_range);
  EXPECT_THROW(topology.at(topology.size() + 1), std::out_of_range);
}

// Test element access without bounds checking
TYPED_TEST(TopologyTest, ElementAccessBrackets) {
  TypeParam topology(this->test_data);

  for (std::size_t i = 0; i < topology.size(); ++i) {
    EXPECT_EQ(topology[i], this->test_data[i]);
  }

  // Test const version
  const TypeParam& const_topology = topology;
  for (std::size_t i = 0; i < const_topology.size(); ++i) {
    EXPECT_EQ(const_topology[i], this->test_data[i]);
  }
}

// Test front() and back() methods
TYPED_TEST(TopologyTest, FrontAndBack) {
  if (TypeParam{}.size() == 0) return;  // Skip for empty arrays

  TypeParam topology(this->test_data);

  EXPECT_EQ(topology.front(), this->test_data.front());
  EXPECT_EQ(topology.back(), this->test_data.back());

  // Test const versions
  const TypeParam& const_topology = topology;
  EXPECT_EQ(const_topology.front(), this->test_data.front());
  EXPECT_EQ(const_topology.back(), this->test_data.back());
}

// Test data() method
TYPED_TEST(TopologyTest, DataAccess) {
  TypeParam topology(this->test_data);

  auto* data_ptr = topology.data();
  EXPECT_NE(data_ptr, nullptr);

  for (std::size_t i = 0; i < topology.size(); ++i) {
    EXPECT_EQ(data_ptr[i], this->test_data[i]);
  }

  // Test const version
  const TypeParam& const_topology = topology;
  const auto* const_data_ptr      = const_topology.data();
  EXPECT_NE(const_data_ptr, nullptr);

  for (std::size_t i = 0; i < const_topology.size(); ++i) {
    EXPECT_EQ(const_data_ptr[i], this->test_data[i]);
  }
}

// Test iterators
TYPED_TEST(TopologyTest, Iterators) {
  TypeParam topology(this->test_data);

  // Test begin/end
  auto it = topology.begin();
  for (std::size_t i = 0; i < topology.size(); ++i, ++it) {
    EXPECT_EQ(*it, this->test_data[i]);
  }
  EXPECT_EQ(it, topology.end());

  // Test const iterators
  const TypeParam& const_topology = topology;
  auto const_it                   = const_topology.begin();
  for (std::size_t i = 0; i < const_topology.size(); ++i, ++const_it) {
    EXPECT_EQ(*const_it, this->test_data[i]);
  }
  EXPECT_EQ(const_it, const_topology.end());

  // Test cbegin/cend
  auto cit = topology.cbegin();
  for (std::size_t i = 0; i < topology.size(); ++i, ++cit) {
    EXPECT_EQ(*cit, this->test_data[i]);
  }
  EXPECT_EQ(cit, topology.cend());
}

// Test reverse iterators
TYPED_TEST(TopologyTest, ReverseIterators) {
  TypeParam topology(this->test_data);

  // Test rbegin/rend
  auto rit = topology.rbegin();
  for (std::size_t i = topology.size(); i > 0; --i, ++rit) {
    EXPECT_EQ(*rit, this->test_data[i - 1]);
  }
  EXPECT_EQ(rit, topology.rend());

  // Test const reverse iterators
  const TypeParam& const_topology = topology;
  auto const_rit                  = const_topology.rbegin();
  for (std::size_t i = const_topology.size(); i > 0; --i, ++const_rit) {
    EXPECT_EQ(*const_rit, this->test_data[i - 1]);
  }
  EXPECT_EQ(const_rit, const_topology.rend());

  // Test crbegin/crend
  auto crit = topology.crbegin();
  for (std::size_t i = topology.size(); i > 0; --i, ++crit) {
    EXPECT_EQ(*crit, this->test_data[i - 1]);
  }
  EXPECT_EQ(crit, topology.crend());
}

// Test capacity methods
TYPED_TEST(TopologyTest, Capacity) {
  TypeParam topology;

  EXPECT_EQ(topology.size(), TypeParam{}.size());
  EXPECT_EQ(topology.max_size(), TypeParam{}.size());
  EXPECT_EQ(topology.empty(), TypeParam{}.size() == 0);
}

// Test fill method
TYPED_TEST(TopologyTest, Fill) {
  TypeParam topology;
  using ValueType      = typename TypeParam::value_type;
  ValueType fill_value = static_cast<ValueType>(42);

  topology.fill(fill_value);

  for (std::size_t i = 0; i < topology.size(); ++i) {
    EXPECT_EQ(topology[i], fill_value);
  }
}

// Test swap method
TYPED_TEST(TopologyTest, Swap) {
  TypeParam topology1(this->test_data);
  TypeParam topology2;

  using ValueType      = typename TypeParam::value_type;
  ValueType fill_value = static_cast<ValueType>(99);
  topology2.fill(fill_value);

  TypeParam expected1 = topology1;
  TypeParam expected2 = topology2;

  topology1.swap(topology2);

  EXPECT_EQ(topology1, expected2);
  EXPECT_EQ(topology2, expected1);
}

// Test comparison operators
TYPED_TEST(TopologyTest, ComparisonOperators) {
  TypeParam topology1(this->test_data);
  TypeParam topology2(this->test_data);
  TypeParam topology3;

  using ValueType      = typename TypeParam::value_type;
  ValueType fill_value = static_cast<ValueType>(99);
  topology3.fill(fill_value);

  // Equality
  EXPECT_TRUE(topology1 == topology2);
  EXPECT_FALSE(topology1 == topology3);

  // Inequality
  EXPECT_FALSE(topology1 != topology2);
  EXPECT_TRUE(topology1 != topology3);

  // Less than
  EXPECT_FALSE(topology1 < topology2);
  EXPECT_TRUE(topology1 < topology3 || topology3 < topology1);

  // Less than or equal
  EXPECT_TRUE(topology1 <= topology2);

  // Greater than
  EXPECT_FALSE(topology1 > topology2);

  // Greater than or equal
  EXPECT_TRUE(topology1 >= topology2);
}

// Test array() method
TYPED_TEST(TopologyTest, ArrayAccess) {
  TypeParam topology(this->test_data);

  const auto& const_array = topology.array();
  EXPECT_EQ(const_array, this->test_data);

  auto& array = topology.array();
  EXPECT_EQ(array, this->test_data);

  // Modify through array reference
  if (array.size() > 0) {
    using ValueType     = typename TypeParam::value_type;
    ValueType new_value = static_cast<ValueType>(999);
    array[0]            = new_value;
    EXPECT_EQ(topology[0], new_value);
  }
}

// Test non-member swap function
TEST(TopologyNonMemberTest, SwapFunction) {
  KokkosFFT::Distributed::Topology<int, 3> topology1{1, 2, 3};
  KokkosFFT::Distributed::Topology<int, 3> topology2{4, 5, 6};

  KokkosFFT::Distributed::Topology<int, 3> expected1 = topology1;
  KokkosFFT::Distributed::Topology<int, 3> expected2 = topology2;

  swap(topology1, topology2);

  EXPECT_EQ(topology1, expected2);
  EXPECT_EQ(topology2, expected1);
}

// Test non-member get functions
TEST(TopologyNonMemberTest, GetFunction) {
  KokkosFFT::Distributed::Topology<int, 3> topology{10, 20, 30};

  // Non-const get
  EXPECT_EQ(KokkosFFT::Distributed::get<0>(topology), 10);
  EXPECT_EQ(KokkosFFT::Distributed::get<1>(topology), 20);
  EXPECT_EQ(KokkosFFT::Distributed::get<2>(topology), 30);

  // Const get
  const auto& const_topology = topology;
  EXPECT_EQ(KokkosFFT::Distributed::get<0>(const_topology), 10);
  EXPECT_EQ(KokkosFFT::Distributed::get<1>(const_topology), 20);
  EXPECT_EQ(KokkosFFT::Distributed::get<2>(const_topology), 30);

  // Rvalue get
  EXPECT_EQ(KokkosFFT::Distributed::get<0>(
                KokkosFFT::Distributed::Topology<int, 3>{10, 20, 30}),
            10);
  EXPECT_EQ(KokkosFFT::Distributed::get<1>(
                KokkosFFT::Distributed::Topology<int, 3>{10, 20, 30}),
            20);
  EXPECT_EQ(KokkosFFT::Distributed::get<2>(
                KokkosFFT::Distributed::Topology<int, 3>{10, 20, 30}),
            30);

  // Const rvalue get
  const KokkosFFT::Distributed::Topology<int, 3> const_rvalue{10, 20, 30};
  EXPECT_EQ(KokkosFFT::Distributed::get<0>(std::move(const_rvalue)), 10);
}

// Test get function with modification
TEST(TopologyNonMemberTest, GetFunctionModification) {
  KokkosFFT::Distributed::Topology<int, 3> topology{10, 20, 30};

  KokkosFFT::Distributed::get<0>(topology) = 100;
  EXPECT_EQ(topology[0], 100);
  EXPECT_EQ(KokkosFFT::Distributed::get<0>(topology), 100);
}

// Test with different layout types
TEST(TopologyLayoutTest, DifferentLayouts) {
  using TopologyRight =
      KokkosFFT::Distributed::Topology<int, 3, Kokkos::LayoutRight>;
  using TopologyLeft =
      KokkosFFT::Distributed::Topology<int, 3, Kokkos::LayoutLeft>;

  TopologyRight right_topology{1, 2, 3};
  TopologyLeft left_topology{1, 2, 3};

  // Both should have the same data
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(right_topology[i], left_topology[i]);
  }

  // But different layout types
  testing::StaticAssertTypeEq<typename TopologyRight::layout_type,
                              Kokkos::LayoutRight>();
  testing::StaticAssertTypeEq<typename TopologyLeft::layout_type,
                              Kokkos::LayoutLeft>();
}

// Test empty topology (size 0)
TEST(TopologySpecialTest, EmptyTopology) {
  KokkosFFT::Distributed::Topology<int, 0> empty_topology;

  EXPECT_EQ(empty_topology.size(), 0);
  EXPECT_EQ(empty_topology.max_size(), 0);
  EXPECT_TRUE(empty_topology.empty());

  EXPECT_EQ(empty_topology.begin(), empty_topology.end());
  EXPECT_EQ(empty_topology.cbegin(), empty_topology.cend());
  EXPECT_EQ(empty_topology.rbegin(), empty_topology.rend());
  EXPECT_EQ(empty_topology.crbegin(), empty_topology.crend());
}

// Test large topology
TEST(TopologySpecialTest, LargeTopology) {
  constexpr std::size_t large_size = 1000;
  KokkosFFT::Distributed::Topology<std::size_t, large_size> large_topology;

  // Fill with indices
  for (std::size_t i = 0; i < large_size; ++i) {
    large_topology[i] = i;
  }

  // Verify
  for (std::size_t i = 0; i < large_size; ++i) {
    EXPECT_EQ(large_topology[i], i);
  }

  EXPECT_EQ(large_topology.size(), large_size);
  EXPECT_FALSE(large_topology.empty());
}

// Test constexpr functionality (C++20 or above)
// Before C++20, std::initializer_list could not be used in constexpr contexts
// TEST(TopologyConstexprTest, ConstexprOperations) {
//  constexpr KokkosFFT::Distributed::Topology<int, 3> topology{1, 2, 3};
//
//  static_assert(topology.size() == 3);
//  static_assert(!topology.empty());
//  static_assert(topology.max_size() == 3);
//
//  // These should compile as constexpr
//  constexpr auto size = topology.size();
//  constexpr auto empty = topology.empty();
//  constexpr auto max_size = topology.max_size();
//
//  EXPECT_EQ(size, 3);
//  EXPECT_FALSE(empty);
//  EXPECT_EQ(max_size, 3);
//}

// Test range-based for loop
TYPED_TEST(TopologyTest, RangeBasedForLoop) {
  TypeParam topology(this->test_data);

  std::size_t index = 0;
  for (const auto& element : topology) {
    EXPECT_EQ(element, this->test_data[index]);
    ++index;
  }
  EXPECT_EQ(index, topology.size());
}

// Test STL algorithm compatibility
TYPED_TEST(TopologyTest, STLAlgorithmCompatibility) {
  TypeParam topology(this->test_data);

  // Test std::find
  auto it = std::find(topology.begin(), topology.end(), this->test_data[0]);
  EXPECT_NE(it, topology.end());
  EXPECT_EQ(*it, this->test_data[0]);

  // Test std::count
  auto count = std::count(topology.begin(), topology.end(), this->test_data[0]);
  EXPECT_EQ(count, 1);

  // Test std::reduce
  std::size_t sum          = std::reduce(topology.begin(), topology.end(),
                                         std::size_t{0}, std::plus<std::size_t>{});
  std::size_t expected_sum = 0;
  for (const auto& val : this->test_data) {
    expected_sum += static_cast<std::size_t>(val);
  }
  EXPECT_EQ(sum, expected_sum);
}
