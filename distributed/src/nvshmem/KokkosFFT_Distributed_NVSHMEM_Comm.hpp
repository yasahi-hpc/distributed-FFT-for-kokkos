#ifndef KOKKOSFFT_DISTRIBUTED_NVSHMEM_COMM_HPP
#define KOKKOSFFT_DISTRIBUTED_NVSHMEM_COMM_HPP

#include <stdexcept>
#include <vector>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_MPI_Comm.hpp"
#include "KokkosFFT_Distributed_MPI_Teams.hpp"
#include "KokkosFFT_Distributed_NVSHMEM_Types.hpp"
#include "KokkosFFT_Distributed_NVSHMEM_Init.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief Scoped wrapper for an NVSHMEM team made from an MPI communicator
///
/// The team has the same processes, in the same order, as the communicator:
/// - a communicator congruent to the SHMEM communicator (nvshmem_comm()) is
///   the world team, which is neither split nor destroyed;
/// - a sub-communicator of the SHMEM communicator is a strided split of the
///   world team. Its processes, in rank order, must be strided in the SHMEM
///   communicator, as the row and column communicators of a cartesian
///   communicator created without reordering are.
///
/// NVSHMEM is set up with the communicator of the first wrapper if it is
/// not set up yet (nvshmem_setup). A plan sets it up with its own
/// communicator first (TplCommSetup), so that it can then make wrappers of
/// its sub-communicators.
///
/// Constructing or destroying a wrapper of a sub-communicator is collective
/// over the SHMEM communicator: every rank must construct one at the same
/// time (a team split is collective over the parent team), and every rank of
/// the team must destroy it.
struct ScopedNVSHMEMComm {
 private:
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  ExecutionSpace m_exec_space;
  MPI_Comm m_comm       = MPI_COMM_NULL;
  nvshmem_team_t m_team = NVSHMEM_TEAM_INVALID;
  int m_rank            = 0;
  int m_size            = 1;

 public:
  /// \brief Construct a ScopedNVSHMEMComm
  /// \param[in] comm MPI communicator: the SHMEM communicator or a strided
  /// sub-communicator of it. Not owned: it must outlive the wrapper.
  /// \param[in] exec_space Kokkos execution space
  /// \throws std::runtime_error if the processes of comm are not strided in
  /// the SHMEM communicator, on every rank of the SHMEM communicator
  explicit ScopedNVSHMEMComm(MPI_Comm comm, const ExecutionSpace &exec_space)
      : m_exec_space(exec_space), m_comm(comm) {
    ::MPI_Comm_rank(comm, &m_rank);
    ::MPI_Comm_size(comm, &m_size);
    m_team = make_team(comm);

    KOKKOSFFT_THROW_IF(nvshmem_team_my_pe(m_team) != m_rank ||
                           nvshmem_team_n_pes(m_team) != m_size,
                       "ScopedNVSHMEMComm: the NVSHMEM team must have the "
                       "ranks and the size of the MPI communicator");
  }

  /// \brief Construct a ScopedNVSHMEMComm
  /// \param[in] comm MPI communicator
  explicit ScopedNVSHMEMComm(MPI_Comm comm)
      : ScopedNVSHMEMComm(comm, ExecutionSpace{}) {}

  ScopedNVSHMEMComm() = delete;

  // Delete copy semantics
  ScopedNVSHMEMComm(const ScopedNVSHMEMComm &)            = delete;
  ScopedNVSHMEMComm &operator=(const ScopedNVSHMEMComm &) = delete;

  // Allow move semantics
  ScopedNVSHMEMComm(ScopedNVSHMEMComm &&other) noexcept
      : m_exec_space(other.m_exec_space),
        m_comm(other.m_comm),
        m_team(other.m_team),
        m_rank(other.m_rank),
        m_size(other.m_size) {
    other.m_comm = MPI_COMM_NULL;
    other.m_team = NVSHMEM_TEAM_INVALID;
  }

  // Move assignment operator. Destroying the old team is collective over it.
  ScopedNVSHMEMComm &operator=(ScopedNVSHMEMComm &&other) noexcept {
    if (this != &other) {
      destroy_team();
      m_exec_space = other.m_exec_space;
      m_comm       = other.m_comm;
      m_team       = other.m_team;
      m_rank       = other.m_rank;
      m_size       = other.m_size;
      other.m_comm = MPI_COMM_NULL;
      other.m_team = NVSHMEM_TEAM_INVALID;
    }
    return *this;
  }

  ~ScopedNVSHMEMComm() { destroy_team(); }

  /// \brief The NVSHMEM team, or NVSHMEM_TEAM_INVALID if moved from
  nvshmem_team_t team() const { return m_team; }

  /// \brief The MPI communicator the team was made from
  MPI_Comm comm() const { return m_comm; }

  ExecutionSpace exec_space() const { return m_exec_space; }
  int rank() const { return m_rank; }
  int size() const { return m_size; }

 private:
  /// \brief The team of comm: the world team, or a strided split of it
  static nvshmem_team_t make_team(MPI_Comm comm) {
    if (nvshmem_comm() == MPI_COMM_NULL) nvshmem_setup(comm);
    MPI_Comm world = nvshmem_comm();

    int result = MPI_UNEQUAL;
    ::MPI_Comm_compare(comm, world, &result);
    if (result == MPI_IDENT || result == MPI_CONGRUENT) {
      return NVSHMEM_TEAM_WORLD;
    }

    // The PE ids of the world team are the ranks of the SHMEM communicator
    const int my_world_pe = nvshmem_my_pe();
    int world_rank        = 0;
    ::MPI_Comm_rank(world, &world_rank);
    KOKKOSFFT_THROW_IF(world_rank != my_world_pe,
                       "ScopedNVSHMEMComm: the NVSHMEM PE ids must be the "
                       "ranks of the communicator NVSHMEM was set up with");

    // Every rank goes on to the splits, which are collective over the world
    // team, or every rank throws
    PETriplet mine;
    int is_strided = 1;
    try {
      mine = compute_pe_triplet(comm, my_world_pe);
    } catch (const std::runtime_error &) {
      is_strided = 0;
    }
    int all_strided = 0;
    ::MPI_Allreduce(&is_strided, &all_strided, 1, MPI_INT, MPI_LAND, world);
    KOKKOSFFT_THROW_IF(!all_strided,
                       "ScopedNVSHMEMComm: the processes of every "
                       "sub-communicator, in rank order, must be strided in "
                       "the communicator NVSHMEM was set up with");

    // Split the world team for each distinct triplet, in the same order on
    // every rank, and keep the team this rank belongs to
    nvshmem_team_t team = NVSHMEM_TEAM_INVALID;
    for (const auto &triplet : gather_pe_triplets(world, mine)) {
      nvshmem_team_t new_team = NVSHMEM_TEAM_INVALID;
      KOKKOSFFT_CHECK_NVSHMEM_CALL(nvshmem_team_split_strided(
          NVSHMEM_TEAM_WORLD, triplet.start, triplet.stride, triplet.size,
          nullptr, 0, &new_team));
      if (triplet == mine) team = new_team;
    }
    return team;
  }

  void destroy_team() noexcept {
    if (m_team != NVSHMEM_TEAM_INVALID && m_team != NVSHMEM_TEAM_WORLD) {
      nvshmem_team_destroy(m_team);
    }
    m_team = NVSHMEM_TEAM_INVALID;
  }
};

/// \brief Set NVSHMEM up with the communicator of a plan, before the plan
/// makes the wrappers of its sub-communicators
template <>
struct TplCommSetup<ScopedNVSHMEMComm> {
  static void setup(MPI_Comm comm) { nvshmem_setup(comm); }
};

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
