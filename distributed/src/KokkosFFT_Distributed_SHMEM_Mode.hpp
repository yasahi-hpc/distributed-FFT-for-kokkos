#ifndef KOKKOSFFT_DISTRIBUTED_SHMEM_MODE_HPP
#define KOKKOSFFT_DISTRIBUTED_SHMEM_MODE_HPP

#include <cstdlib>
#include <string>
#include <mpi.h>
#include "KokkosFFT_Asserts.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief How the transposes of a SHMEM communicator wrapper exchange the
/// data
enum class SHMEMCommMode {
  /// The host enqueues a barrier and an all-to-all on the stream (all2all)
  host,
  /// A kernel puts the chunks into the peers' receive buffers, with signals
  /// (Exchange). Only with ENABLE_TPL_SHMEM_DEVICE_COMM.
  device
};

/// \brief The default mode: device if the device mode is compiled
/// (ENABLE_TPL_SHMEM_DEVICE_COMM), host otherwise. The environment variable
/// KOKKOSFFT_SHMEM_COMM ("host" or "device") overrides it.
/// \throws std::runtime_error if KOKKOSFFT_SHMEM_COMM has another value
inline SHMEMCommMode default_shmem_comm_mode() {
  const char* env = std::getenv("KOKKOSFFT_SHMEM_COMM");
#if defined(ENABLE_TPL_SHMEM_DEVICE_COMM)
  SHMEMCommMode mode = SHMEMCommMode::device;
#else
  SHMEMCommMode mode = SHMEMCommMode::host;
#endif
  if (env == nullptr) return mode;

  const std::string value(env);
  KOKKOSFFT_THROW_IF(value != "host" && value != "device",
                     "KOKKOSFFT_SHMEM_COMM must be \"host\" or \"device\", "
                     "not \"" +
                         value + "\"");
  return value == "host" ? SHMEMCommMode::host : SHMEMCommMode::device;
}

/// \brief Whether the device mode fuses the pack with the exchange: the pack
/// kernel stores the chunks of the peers reachable by load/store directly
/// into their receive buffers. On by default; KOKKOSFFT_SHMEM_FUSED_PACK=0
/// turns it off, so that every chunk is packed locally and put.
/// Each process may choose on its own: both variants signal the peers in the
/// same way.
/// \throws std::runtime_error if KOKKOSFFT_SHMEM_FUSED_PACK is not "0" or "1"
inline bool default_shmem_fused_pack() {
  const char* env = std::getenv("KOKKOSFFT_SHMEM_FUSED_PACK");
  if (env == nullptr) return true;

  const std::string value(env);
  KOKKOSFFT_THROW_IF(value != "0" && value != "1",
                     "KOKKOSFFT_SHMEM_FUSED_PACK must be \"0\" or \"1\", "
                     "not \"" +
                         value + "\"");
  return value == "1";
}

/// \brief Check a SHMEM communication mode: it must be compiled, and the same
/// on every rank of comm
/// \param[in] mode The mode of this rank
/// \param[in] comm MPI communicator over which the mode must agree
/// \throws std::runtime_error on every rank of comm if the ranks disagree, or
/// if they ask for the device mode while it is not compiled
inline void check_shmem_comm_mode(SHMEMCommMode mode, MPI_Comm comm) {
  int mine = static_cast<int>(mode), min_mode = 0, max_mode = 0;
  ::MPI_Allreduce(&mine, &min_mode, 1, MPI_INT, MPI_MIN, comm);
  ::MPI_Allreduce(&mine, &max_mode, 1, MPI_INT, MPI_MAX, comm);
  KOKKOSFFT_THROW_IF(min_mode != max_mode,
                     "The SHMEM communication mode (KOKKOSFFT_SHMEM_COMM) "
                     "must be the same on every process");
#if !defined(ENABLE_TPL_SHMEM_DEVICE_COMM)
  KOKKOSFFT_THROW_IF(mode == SHMEMCommMode::device,
                     "The SHMEM device mode needs ENABLE_SHMEM_DEVICE_COMM");
#endif
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
