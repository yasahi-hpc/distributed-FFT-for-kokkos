#ifndef KOKKOSFFT_DISTRIBUTED_NCCL_TYPES_HPP
#define KOKKOSFFT_DISTRIBUTED_NCCL_TYPES_HPP

#include <cstdint>

#if defined(KOKKOS_ENABLE_CUDA)
#include <nccl.h>
#elif defined(KOKKOS_ENABLE_HIP)
#include <rccl/rccl.h>
#else
static_assert(false,
              "You need to enable CUDA (HIP) backend to use NCCL (RCCL).");
#endif

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

template <typename ValueType>
auto nccl_datatype() -> ncclDataType_t {
  using T = std::decay_t<ValueType>;

  if constexpr (std::is_same_v<T, char>) {
    return ncclChar;
  } else if constexpr (std::is_same_v<T, std::int8_t>) {
    return ncclInt8;
  } else if constexpr (std::is_same_v<T, std::uint8_t>) {
    return ncclUint8;
  } else if constexpr (std::is_same_v<T, int>) {
    return ncclInt;
  } else if constexpr (std::is_same_v<T, std::int32_t>) {
    return ncclInt32;
  } else if constexpr (std::is_same_v<T, std::uint32_t>) {
    return ncclUint32;
  } else if constexpr (std::is_same_v<T, std::int64_t>) {
    return ncclInt64;
  } else if constexpr (std::is_same_v<T, std::uint64_t>) {
    return ncclUint64;
  } else if constexpr (std::is_same_v<T, float>) {
    return ncclFloat;
  } else if constexpr (std::is_same_v<T, double>) {
    return ncclDouble;
  } else {
    static_assert(
        std::is_void_v<T>,
        "KokkosFFT::Distributed::Impl::nccl_datatype: unsupported data type");
    return ncclChar;  // unreachable
  }
}

template <typename ValueType>
inline ncclDataType_t nccl_datatype_v = nccl_datatype<ValueType>();

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
