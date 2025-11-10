#ifndef KOKKOSFFT_DISTRIBUTED_ONECCL_TYPES_HPP
#define KOKKOSFFT_DISTRIBUTED_ONECCL_TYPES_HPP

#include <cstdint>
#include <ccl.hpp>

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

template <typename ValueType>
struct oneCCLDataType {};

template <>
struct oneCCLDataType<int> {
  static inline ccl::datatype type() noexcept { return ccl::datatype::int32; }
};

template <>
struct oneCCLDataType<std::uint32_t> {
  static inline ccl::datatype type() noexcept { return ccl::datatype::uint32; }
};

template <>
struct oneCCLDataType<std::int64_t> {
  static inline ccl::datatype type() noexcept { return ccl::datatype::int64; }
};

template <>
struct oneCCLDataType<std::uint64_t> {
  static inline ccl::datatype type() noexcept { return ccl::datatype::uint64; }
};

template <>
struct oneCCLDataType<float> {
  static inline ccl::datatype type() noexcept { return ccl::datatype::float32; }
};

template <>
struct oneCCLDataType<double> {
  static inline ccl::datatype type() noexcept { return ccl::datatype::float64; }
};

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
