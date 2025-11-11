#ifndef KOKKOSFFT_DISTRIBUTED_ONECCL_TYPES_HPP
#define KOKKOSFFT_DISTRIBUTED_ONECCL_TYPES_HPP

#include <cstdint>
#include <ccl.hpp>

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

template <typename ValueType>
auto oneccl_datatype() -> ccl::datatype {
  using T = std::decay_t<ValueType>;

  if constexpr (std::is_same_v<T, std::int8_t>) {
    return ccl::datatype::int8;
  } else if constexpr (std::is_same_v<T, std::uint8_t>) {
    return ccl::datatype::uint8;
  } else if constexpr (std::is_same_v<T, std::int16_t>) {
    return ccl::datatype::int16;
  } else if constexpr (std::is_same_v<T, std::uint16_t>) {
    return ccl::datatype::uint16;
  } else if constexpr (std::is_same_v<T, int>) {
    return ccl::datatype::int32;
  } else if constexpr (std::is_same_v<T, std::int32_t>) {
    return ccl::datatype::int32;
  } else if constexpr (std::is_same_v<T, std::uint32_t>) {
    return ccl::datatype::uint32;
  } else if constexpr (std::is_same_v<T, std::int64_t>) {
    return ccl::datatype::int64;
  } else if constexpr (std::is_same_v<T, std::uint64_t>) {
    return ccl::datatype::uint64;
  } else if constexpr (std::is_same_v<T, float>) {
    return ccl::datatype::float32;
  } else if constexpr (std::is_same_v<T, double>) {
    return ccl::datatype::float64;
  } else {
    // TO DO float16/bfloat16 support
    static_assert(
        std::is_void_v<T>,
        "KokkosFFT::Distributed::Impl::oneccl_datatype: unsupported data type");
    return ccl::datatype::int8;  // unreachable
  }
}

template <typename ValueType>
inline onecclDataType_t oneccl_datatype_v = oneccl_datatype<ValueType>();

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
