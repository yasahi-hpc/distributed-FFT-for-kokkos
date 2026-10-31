#ifndef KOKKOSFFT_DISTRIBUTED_MPI_TEAMS_HPP
#define KOKKOSFFT_DISTRIBUTED_MPI_TEAMS_HPP

#include <algorithm>
#include <vector>
#include <mpi.h>
#include "KokkosFFT_Asserts.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief A strided set of processing elements (PEs) of a parent team: the
/// PEs start, start + stride, ..., start + (size - 1) * stride. This is the
/// form a SHMEM team split takes.
struct PETriplet {
  int start  = 0;
  int stride = 1;
  int size   = 1;

  bool operator==(const PETriplet& other) const {
    return start == other.start && stride == other.stride && size == other.size;
  }
  bool operator!=(const PETriplet& other) const { return !(*this == other); }
};

/// \brief The PEs of the members of comm, in rank order, as a strided
/// triplet. Collective over comm.
///
/// \param[in] comm MPI communicator
/// \param[in] my_world_pe The PE id of the calling rank in the parent team
/// \return The triplet of the PEs of comm. A communicator of size 1 has the
/// stride 1.
/// \throws std::runtime_error if the PEs, in rank order, are not an
/// increasing arithmetic progression
inline PETriplet compute_pe_triplet(MPI_Comm comm, int my_world_pe) {
  int size = 0;
  ::MPI_Comm_size(comm, &size);

  std::vector<int> pes(size);
  ::MPI_Allgather(&my_world_pe, 1, MPI_INT, pes.data(), 1, MPI_INT, comm);

  PETriplet triplet{pes.at(0), size > 1 ? pes.at(1) - pes.at(0) : 1, size};
  bool is_strided = triplet.stride > 0;
  for (int i = 0; i < size; ++i) {
    is_strided = is_strided && pes.at(i) == triplet.start + i * triplet.stride;
  }
  KOKKOSFFT_THROW_IF(!is_strided,
                     "compute_pe_triplet: the PEs of the communicator, in "
                     "rank order, must be strided (start, start + stride, "
                     "...) with a positive stride");
  return triplet;
}

/// \brief The distinct triplets of all the ranks of world, in the order of
/// the first rank that has each. Every rank gets the same list. Collective
/// over world.
///
/// \param[in] world MPI communicator of the parent team
/// \param[in] mine The triplet of the calling rank
/// \return The distinct triplets
inline std::vector<PETriplet> gather_pe_triplets(MPI_Comm world,
                                                 const PETriplet& mine) {
  int size = 0;
  ::MPI_Comm_size(world, &size);

  const int send[3] = {mine.start, mine.stride, mine.size};
  std::vector<int> recv(3 * size);
  ::MPI_Allgather(send, 3, MPI_INT, recv.data(), 3, MPI_INT, world);

  std::vector<PETriplet> triplets;
  for (int i = 0; i < size; ++i) {
    PETriplet triplet{recv.at(3 * i), recv.at(3 * i + 1), recv.at(3 * i + 2)};
    if (std::find(triplets.begin(), triplets.end(), triplet) ==
        triplets.end()) {
      triplets.push_back(triplet);
    }
  }
  return triplets;
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
