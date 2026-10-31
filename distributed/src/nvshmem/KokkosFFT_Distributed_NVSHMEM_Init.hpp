#ifndef KOKKOSFFT_DISTRIBUTED_NVSHMEM_INIT_HPP
#define KOKKOSFFT_DISTRIBUTED_NVSHMEM_INIT_HPP

#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief The communicator NVSHMEM is initialized with (the SHMEM
/// communicator). It is a duplicate of the communicator passed to the first
/// call of nvshmem_setup, and MPI_COMM_NULL before that call and after
/// Kokkos::finalize.
inline MPI_Comm& nvshmem_comm() {
  static MPI_Comm comm = MPI_COMM_NULL;
  return comm;
}

/// \brief Initialize NVSHMEM once per process with an MPI communicator
/// This is shared by every user of NVSHMEM in the library (cuFFTMp and the
/// NVSHMEM communication backend).
/// - The first call initializes NVSHMEM with comm, unless the user already
///   initialized it, and registers nvshmem_finalize as a Kokkos finalize
///   hook. It keeps a duplicate of comm as the SHMEM communicator.
/// - Every call checks that comm contains the same processes, in the same
///   order, as the SHMEM communicator.
///
/// \param[in] comm MPI communicator
/// \throws std::runtime_error if comm is neither identical nor congruent to
/// the SHMEM communicator
inline void nvshmem_setup(MPI_Comm comm) {
  [[maybe_unused]] static bool once = [comm] {
    if (!(Kokkos::is_initialized() || Kokkos::is_finalized())) {
      Kokkos::abort(
          "Error: KokkosFFT APIs must not be called before initializing "
          "Kokkos.\n");
    }
    if (Kokkos::is_finalized()) {
      Kokkos::abort(
          "Error: KokkosFFT APIs must not be called after finalizing "
          "Kokkos.\n");
    }

    ::MPI_Comm_dup(comm, &nvshmem_comm());

    // If the user already initialized NVSHMEM, the user finalizes it
    const bool initialize =
        nvshmemx_init_status() == NVSHMEM_STATUS_NOT_INITIALIZED;
    if (initialize) {
      nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
      attr.mpi_comm             = (void*)&nvshmem_comm();
      nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    }

    // Register cleanup function as a hook in Kokkos::finalize
    Kokkos::push_finalize_hook([initialize]() {
      if (initialize) nvshmem_finalize();
      ::MPI_Comm_free(&nvshmem_comm());
    });

    return true;
  }();

  int result = MPI_UNEQUAL;
  ::MPI_Comm_compare(comm, nvshmem_comm(), &result);
  KOKKOSFFT_THROW_IF(result != MPI_IDENT && result != MPI_CONGRUENT,
                     "NVSHMEM requires the plan communicator to contain the "
                     "same processes as the communicator it was initialized "
                     "with");
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
