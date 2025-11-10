#ifndef KOKKOSFFT_DISTRIBUTED_NCCL_TYPES_HPP
#define KOKKOSFFT_DISTRIBUTED_NCCL_TYPES_HPP

#include <cstdint>
#include <Kokkos_Core.hpp>

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
struct NCCLDataType {};

template <>
struct NCCLDataType<int> {
  static inline ncclDataType_t type() noexcept { return ncclInt; }
};

template <>
struct NCCLDataType<std::uint32_t> {
  static inline ncclDataType_t type() noexcept { return ncclUint32; }
};

template <>
struct NCCLDataType<std::int64_t> {
  static inline ncclDataType_t type() noexcept { return ncclInt64; }
};

template <>
struct NCCLDataType<std::uint64_t> {
  static inline ncclDataType_t type() noexcept { return ncclUint64; }
};

template <>
struct NCCLDataType<float> {
  static inline ncclDataType_t type() noexcept { return ncclFloat; }
};

template <>
struct NCCLDataType<double> {
  static inline ncclDataType_t type() noexcept { return ncclDouble; }
};

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
