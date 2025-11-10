#ifndef KOKKOSFFT_DISTRIBUTED_TPLS_VERSIONS_HPP
#define KOKKOSFFT_DISTRIBUTED_TPLS_VERSIONS_HPP

#include <sstream>
#include <iostream>
#include "KokkosFFT_Distributed_config.hpp"

#if defined(KOKKOSFFT_ENABLE_TPL_CUFFT)
#include "cufft.h"
#endif

#if defined(KOKKOSFFT_ENABLE_TPL_ROCFFT)
#include <rocfft/rocfft.h>
#endif

#if defined(KOKKOSFFT_ENABLE_TPL_HIPFFT)
#include <hipfft/hipfft.h>
#endif

#if defined(KOKKOSFFT_DISTRIBUTED_ENABLE_TPL_NCCL)
#include <nccl.h>
#endif

#if defined(KOKKOSFFT_DISTRIBUTED_ENABLE_TPL_RCCL)
#include <rccl/rccl.h>
#endif

namespace KokkosFFT {
namespace Distributed {
#if defined(KOKKOSFFT_ENABLE_TPL_CUFFT)
inline std::string cufft_version_string() {
  // Print version
  std::stringstream ss;

  ss << CUFFT_VER_MAJOR << "." << CUFFT_VER_MINOR << "." << CUFFT_VER_PATCH;

  return ss.str();
}
#endif

#if defined(KOKKOSFFT_ENABLE_TPL_ROCFFT)
inline std::string rocfft_version_string() {
  // Print version
  std::stringstream ss;
  constexpr std::size_t len = 50;
  char version_string[len];
  rocfft_get_version_string(version_string, len);

  ss << version_string;

  return ss.str();
}
#endif

#if defined(KOKKOSFFT_ENABLE_TPL_HIPFFT)
inline std::string hipfft_version_string() {
  // Print version
  std::stringstream ss;

  ss << hipfftVersionMajor << "." << hipfftVersionMinor << "."
     << hipfftVersionPatch;

  return ss.str();
}
#endif

#if defined(KOKKOSFFT_DISTRIBUTED_ENABLE_TPL_NCCL) || \
    defined(KOKKOSFFT_DISTRIBUTED_ENABLE_TPL_RCCL)
inline std::string nccl_version_string() {
  int version;
  ncclResult_t result = ncclGetVersion(&version);
  if (result != ncclSuccess) {
    std::cerr << "Failed to get NCCL version: " << ncclGetErrorString(result)
              << std::endl;
    exit(EXIT_FAILURE);
  }

  int major = version / 10000;
  int minor = (version % 10000) / 100;
  int patch = version % 100;

  // Print version
  std::stringstream ss;

  ss << major << "." << minor << "." << patch;

  return ss.str();
}
#endif

}  // namespace Distributed
}  // namespace KokkosFFT
#endif
