#ifndef KOKKOSFFT_DISTRIBUTED_NVSHMEM_TYPES_HPP
#define KOKKOSFFT_DISTRIBUTED_NVSHMEM_TYPES_HPP

#include <string>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Asserts.hpp"

#if defined(KOKKOS_ENABLE_CUDA)
#include <nvshmem.h>
#include <nvshmemx.h>
#else
static_assert(false, "You need to enable CUDA backend to use NVSHMEM.");
#endif

/// \brief Throw if an NVSHMEM call returning an int status does not return 0
#define KOKKOSFFT_CHECK_NVSHMEM_CALL(call)                               \
  do {                                                                   \
    const int kokkosfft_nvshmem_status = (call);                         \
    KOKKOSFFT_THROW_IF(kokkosfft_nvshmem_status != 0,                    \
                       std::string(#call) + " failed with error code " + \
                           std::to_string(kokkosfft_nvshmem_status));    \
  } while (0)

#endif
