#ifndef KOKKOSFFT_DISTRIBUTED_ROCSHMEM_TYPES_HPP
#define KOKKOSFFT_DISTRIBUTED_ROCSHMEM_TYPES_HPP

#include <string>
#include <string_view>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Asserts.hpp"

#if defined(KOKKOS_ENABLE_HIP)
#include <rocshmem/rocshmem.hpp>
#else
static_assert(false, "You need to enable HIP backend to use rocSHMEM.");
#endif

// rocSHMEM 3.4 (TheRock 7.13) or later is needed: rocSHMEM 3.3 sets the world
// team of the IPC conduit up before its sync arrays, so the collectives on
// ROCSHMEM_TEAM_WORLD (rocshmem_alltoallmem_on_stream) fault.
// ROCSHMEM_VENDOR_MAJOR_VERSION and ROCSHMEM_VENDOR_MINOR_VERSION (in
// rocshmem_config.h) exist from rocSHMEM 3.4.
#if !defined(ROCSHMEM_VENDOR_MAJOR_VERSION) || \
    !defined(ROCSHMEM_VENDOR_MINOR_VERSION) || \
    (ROCSHMEM_VENDOR_MAJOR_VERSION < 3) ||     \
    (ROCSHMEM_VENDOR_MAJOR_VERSION == 3 && ROCSHMEM_VENDOR_MINOR_VERSION < 4)
static_assert(false,
              "rocSHMEM 3.4 or later is needed for the rocSHMEM backend "
              "(collectives on ROCSHMEM_TEAM_WORLD fault with rocSHMEM 3.3)");
#endif

// rocSHMEM 3.5 (TheRock 7.14) has a barrier on a stream for any team
#if ROCSHMEM_VENDOR_MAJOR_VERSION > 3 || \
    (ROCSHMEM_VENDOR_MAJOR_VERSION == 3 && ROCSHMEM_VENDOR_MINOR_VERSION >= 5)
#define KOKKOSFFT_ROCSHMEM_HAS_TEAM_BARRIER_ON_STREAM
#endif

/// \brief Throw if a rocSHMEM call returning an int status does not return 0
#define KOKKOSFFT_CHECK_ROCSHMEM_CALL(call)                              \
  do {                                                                   \
    const int kokkosfft_rocshmem_status = (call);                        \
    KOKKOSFFT_THROW_IF(kokkosfft_rocshmem_status != 0,                   \
                       std::string(#call) + " failed with error code " + \
                           std::to_string(kokkosfft_rocshmem_status));   \
  } while (0)

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief The number at position index of a version string "major.minor..."
/// (0 for major, 1 for minor), e.g. ROCSHMEM_VERSION or rocshmem::VERSION.
/// Kept for the configuration printed by the benchmarks (perf_test).
constexpr int parse_version_number(std::string_view version,
                                   std::size_t index) {
  int number = 0;
  for (char c : version) {
    if (c == '.') {
      if (index == 0) return number;
      --index;
      number = 0;
    } else if (c >= '0' && c <= '9') {
      number = number * 10 + (c - '0');
    }
  }
  return number;
}

/// \brief The rocSHMEM version (3.4 or later, checked above)
inline constexpr int rocshmem_version_major = ROCSHMEM_VENDOR_MAJOR_VERSION;
inline constexpr int rocshmem_version_minor = ROCSHMEM_VENDOR_MINOR_VERSION;

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
