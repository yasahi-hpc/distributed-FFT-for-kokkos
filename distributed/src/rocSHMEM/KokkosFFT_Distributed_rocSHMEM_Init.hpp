#ifndef KOKKOSFFT_DISTRIBUTED_ROCSHMEM_INIT_HPP
#define KOKKOSFFT_DISTRIBUTED_ROCSHMEM_INIT_HPP

#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_rocSHMEM_Types.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief The communicator rocSHMEM is initialized with (the SHMEM
/// communicator). It is a duplicate of the communicator passed to the first
/// call of rocshmem_setup, and MPI_COMM_NULL before that call and after
/// Kokkos::finalize.
inline MPI_Comm& rocshmem_comm() {
  static MPI_Comm comm = MPI_COMM_NULL;
  return comm;
}

/// \brief Initialize rocSHMEM once per process with an MPI communicator
/// - The first call initializes rocSHMEM with a unique id, broadcast over
///   comm: the PE id of each rank is its rank in comm. This works with every
///   heap allocator, unlike the initialization with an MPI communicator
///   (ROCSHMEM_HEAP_ALLOCATOR_TYPE=vmm_posix). It registers
///   rocshmem_finalize as a Kokkos finalize hook, and keeps a duplicate of
///   comm as the SHMEM communicator.
/// - Every call checks that comm contains the same processes, in the same
///   order, as the SHMEM communicator.
///
/// rocSHMEM has no query for its initialization state, so it must not be
/// initialized by the user as well.
///
/// \param[in] comm MPI communicator
/// \throws std::runtime_error if comm is neither identical nor congruent to
/// the SHMEM communicator, or if the initialization fails
inline void rocshmem_setup(MPI_Comm comm) {
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

    ::MPI_Comm_dup(comm, &rocshmem_comm());

    int rank = 0, nranks = 1;
    ::MPI_Comm_rank(rocshmem_comm(), &rank);
    ::MPI_Comm_size(rocshmem_comm(), &nranks);

    rocshmem::rocshmem_uniqueid_t uid{};
    if (rank == 0) {
      KOKKOSFFT_CHECK_ROCSHMEM_CALL(rocshmem::rocshmem_get_uniqueid(&uid));
    }
    ::MPI_Bcast(uid.data(), static_cast<int>(uid.size()), MPI_BYTE, 0,
                rocshmem_comm());

    rocshmem::rocshmem_init_attr_t attr{};
    KOKKOSFFT_CHECK_ROCSHMEM_CALL(
        rocshmem::rocshmem_set_attr_uniqueid_args(rank, nranks, &uid, &attr));
    KOKKOSFFT_CHECK_ROCSHMEM_CALL(rocshmem::rocshmem_init_attr(
        rocshmem::ROCSHMEM_INIT_WITH_UNIQUEID, &attr));

    // Register cleanup function as a hook in Kokkos::finalize
    Kokkos::push_finalize_hook([]() {
      rocshmem::rocshmem_finalize();
      ::MPI_Comm_free(&rocshmem_comm());
    });

    return true;
  }();

  int result = MPI_UNEQUAL;
  ::MPI_Comm_compare(comm, rocshmem_comm(), &result);
  KOKKOSFFT_THROW_IF(result != MPI_IDENT && result != MPI_CONGRUENT,
                     "rocSHMEM requires the plan communicator to contain the "
                     "same processes as the communicator it was initialized "
                     "with");
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
