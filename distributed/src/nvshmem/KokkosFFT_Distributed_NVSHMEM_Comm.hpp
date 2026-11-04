#ifndef KOKKOSFFT_DISTRIBUTED_NVSHMEM_COMM_HPP
#define KOKKOSFFT_DISTRIBUTED_NVSHMEM_COMM_HPP

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_MPI_Comm.hpp"
#include "KokkosFFT_Distributed_MPI_Teams.hpp"
#include "KokkosFFT_Distributed_SHMEM_Mode.hpp"
#include "KokkosFFT_Distributed_NVSHMEM_Types.hpp"
#include "KokkosFFT_Distributed_NVSHMEM_Init.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief The addresses of a symmetric buffer on the PEs of a team, as seen
/// from this GPU (nvshmem_ptr), for the fused pack of the device mode
struct NVSHMEMPeerTable {
  /// The local address of the buffer
  const void *m_buffer = nullptr;
  /// The address of the buffer on team PE p, or 0 if p is not reachable by
  /// load/store (NVLink or PCIe P2P)
  Kokkos::View<std::uintptr_t *, Kokkos::CudaSpace> m_peers;
  /// Whether a PE other than this one is reachable by load/store
  bool m_has_peer = false;
};

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
/// The mode (SHMEMCommMode) selects how the transposes exchange the data:
/// all2all in host mode, the puts of Exchange<ScopedNVSHMEMComm> in device
/// mode. The device mode adds a signal word on the symmetric heap, which the
/// peers increment when their chunks have arrived, and the number of
/// exchanges done (the epoch).
///
/// Constructing or destroying a wrapper is collective over the SHMEM
/// communicator: every rank must construct one at the same time (a team
/// split is collective over the parent team, the mode is checked and the
/// signal word allocated over the SHMEM communicator), and every rank must
/// destroy it.
struct ScopedNVSHMEMComm {
 private:
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  ExecutionSpace m_exec_space;
  MPI_Comm m_comm       = MPI_COMM_NULL;
  nvshmem_team_t m_team = NVSHMEM_TEAM_INVALID;
  int m_rank            = 0;
  int m_size            = 1;
  SHMEMCommMode m_mode  = SHMEMCommMode::host;
  bool m_fused_pack     = true;

  // The PE ids of the team in the world team
  PETriplet m_triplet;

  // Device mode: the signal word on the symmetric heap, and the number of
  // exchanges done (mutable: the exchanges take a const wrapper)
  std::uint64_t *m_signal       = nullptr;
  mutable std::uint64_t m_epoch = 0;

  // Device mode: the peer tables of the receive buffers used so far
  // (a plan alternates between two buffers)
  mutable std::vector<NVSHMEMPeerTable> m_peer_tables;
  static constexpr std::size_t m_max_peer_tables = 8;

 public:
  /// \brief Construct a ScopedNVSHMEMComm
  /// \param[in] comm MPI communicator: the SHMEM communicator or a strided
  /// sub-communicator of it. Not owned: it must outlive the wrapper.
  /// \param[in] exec_space Kokkos execution space
  /// \param[in] mode The communication mode, the same on every rank
  /// \throws std::runtime_error, on every rank of the SHMEM communicator, if
  /// the mode is not the same on every rank or not compiled, or if the
  /// processes of comm are not strided in the SHMEM communicator
  explicit ScopedNVSHMEMComm(MPI_Comm comm, const ExecutionSpace &exec_space,
                             SHMEMCommMode mode)
      : m_exec_space(exec_space),
        m_comm(comm),
        m_mode(mode),
        m_fused_pack(default_shmem_fused_pack()) {
    ::MPI_Comm_rank(comm, &m_rank);
    ::MPI_Comm_size(comm, &m_size);

    // Before the team split, so that a wrong mode does not leave a team
    if (nvshmem_comm() == MPI_COMM_NULL) nvshmem_setup(comm);
    check_shmem_comm_mode(m_mode, nvshmem_comm());
    m_team = make_team(comm, m_triplet);

    KOKKOSFFT_THROW_IF(nvshmem_team_my_pe(m_team) != m_rank ||
                           nvshmem_team_n_pes(m_team) != m_size,
                       "ScopedNVSHMEMComm: the NVSHMEM team must have the "
                       "ranks and the size of the MPI communicator");

    if (m_mode == SHMEMCommMode::device) allocate_signal();
  }

  /// \brief Construct a ScopedNVSHMEMComm in the default mode
  /// (default_shmem_comm_mode)
  /// \param[in] comm MPI communicator
  /// \param[in] exec_space Kokkos execution space
  explicit ScopedNVSHMEMComm(MPI_Comm comm, const ExecutionSpace &exec_space)
      : ScopedNVSHMEMComm(comm, exec_space, default_shmem_comm_mode()) {}

  /// \brief Construct a ScopedNVSHMEMComm in the default mode
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
        m_size(other.m_size),
        m_mode(other.m_mode),
        m_fused_pack(other.m_fused_pack),
        m_triplet(other.m_triplet),
        m_signal(other.m_signal),
        m_epoch(other.m_epoch),
        m_peer_tables(std::move(other.m_peer_tables)) {
    other.m_comm   = MPI_COMM_NULL;
    other.m_team   = NVSHMEM_TEAM_INVALID;
    other.m_signal = nullptr;
  }

  // Move assignment operator. Destroying the old team is collective over it,
  // freeing the old signal word over the SHMEM communicator.
  ScopedNVSHMEMComm &operator=(ScopedNVSHMEMComm &&other) noexcept {
    if (this != &other) {
      destroy();
      m_exec_space   = other.m_exec_space;
      m_comm         = other.m_comm;
      m_team         = other.m_team;
      m_rank         = other.m_rank;
      m_size         = other.m_size;
      m_mode         = other.m_mode;
      m_fused_pack   = other.m_fused_pack;
      m_triplet      = other.m_triplet;
      m_signal       = other.m_signal;
      m_epoch        = other.m_epoch;
      m_peer_tables  = std::move(other.m_peer_tables);
      other.m_comm   = MPI_COMM_NULL;
      other.m_team   = NVSHMEM_TEAM_INVALID;
      other.m_signal = nullptr;
    }
    return *this;
  }

  ~ScopedNVSHMEMComm() { destroy(); }

  /// \brief The NVSHMEM team, or NVSHMEM_TEAM_INVALID if moved from
  nvshmem_team_t team() const { return m_team; }

  /// \brief The MPI communicator the team was made from
  MPI_Comm comm() const { return m_comm; }

  ExecutionSpace exec_space() const { return m_exec_space; }
  int rank() const { return m_rank; }
  int size() const { return m_size; }

  /// \brief The communication mode
  SHMEMCommMode mode() const { return m_mode; }

  /// \brief The PE id, in the world team, of the PE pe of the team
  int world_pe(int pe) const { return m_triplet.start + pe * m_triplet.stride; }

  /// \brief The PE ids of the team in the world team
  const PETriplet &triplet() const { return m_triplet; }

  /// \brief The signal word on the symmetric heap (device mode), or nullptr
  /// in host mode or if moved from
  std::uint64_t *signal() const { return m_signal; }

  /// \brief The number of exchanges done (device mode)
  std::uint64_t epoch() const { return m_epoch; }

  /// \brief Whether the device mode fuses the pack with the exchange
  /// (default_shmem_fused_pack, read at construction)
  bool fused_pack() const { return m_fused_pack; }

  /// \brief The addresses of a symmetric buffer on the PEs of the team
  /// (device mode). Computed with nvshmem_ptr on the first call for a
  /// buffer and kept: nvshmem_ptr depends only on the address, so a table
  /// stays valid for whatever buffer is later allocated at that address.
  /// \param[in] buffer A local address on the symmetric heap. For any other
  /// address, every entry is 0.
  NVSHMEMPeerTable peer_table(const void *buffer) const {
    for (const auto &table : m_peer_tables) {
      if (table.m_buffer == buffer) return table;
    }
    if (m_peer_tables.size() >= m_max_peer_tables) m_peer_tables.clear();

    NVSHMEMPeerTable table;
    table.m_buffer = buffer;
    table.m_peers  = Kokkos::View<std::uintptr_t *, Kokkos::CudaSpace>(
        "KokkosFFT::Distributed::NVSHMEMPeerTable", m_size);
    auto h_peers = Kokkos::create_mirror_view(table.m_peers);
    for (int pe = 0; pe < m_size; ++pe) {
      h_peers(pe) =
          reinterpret_cast<std::uintptr_t>(nvshmem_ptr(buffer, world_pe(pe)));
      if (pe != m_rank && h_peers(pe) != 0) table.m_has_peer = true;
    }
    Kokkos::deep_copy(table.m_peers, h_peers);
    m_peer_tables.push_back(table);
    return table;
  }

  /// \brief Start an exchange (device mode)
  /// \return The value of the signal word once every PE of the team has put
  /// its chunk of this exchange
  std::uint64_t next_epoch() const {
    ++m_epoch;
    return m_epoch * static_cast<std::uint64_t>(m_size);
  }

 private:
  /// \brief The team of comm: the world team, or a strided split of it
  /// \param[in] comm MPI communicator
  /// \param[out] mine The PE ids of the team in the world team
  static nvshmem_team_t make_team(MPI_Comm comm, PETriplet &mine) {
    MPI_Comm world = nvshmem_comm();

    int result = MPI_UNEQUAL;
    ::MPI_Comm_compare(comm, world, &result);
    if (result == MPI_IDENT || result == MPI_CONGRUENT) {
      mine = PETriplet{0, 1, nvshmem_n_pes()};
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

  /// \brief Allocate the signal word on the symmetric heap (collective over
  /// the SHMEM communicator) and zero it, before any peer can signal it: the
  /// peers signal after a barrier on the team (Exchange)
  void allocate_signal() {
    m_signal =
        static_cast<std::uint64_t *>(nvshmem_malloc(sizeof(std::uint64_t)));
    if (m_signal == nullptr) {
      destroy_team();
      KOKKOSFFT_THROW_IF(true,
                         "ScopedNVSHMEMComm: nvshmem_malloc of the signal "
                         "word failed (NVSHMEM_SYMMETRIC_SIZE)");
    }
    Kokkos::View<std::uint64_t, Kokkos::CudaSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        signal(m_signal);
    Kokkos::deep_copy(m_exec_space, signal, std::uint64_t(0));
    m_exec_space.fence();
  }

  void destroy_team() noexcept {
    if (m_team != NVSHMEM_TEAM_INVALID && m_team != NVSHMEM_TEAM_WORLD) {
      nvshmem_team_destroy(m_team);
    }
    m_team = NVSHMEM_TEAM_INVALID;
  }

  /// \brief Free the signal word, after the exchanges enqueued on the stream
  /// that wait on it, and destroy the team
  void destroy() noexcept {
    m_peer_tables.clear();
    if (m_signal != nullptr) {
      m_exec_space.fence();
      nvshmem_free(m_signal);
      m_signal = nullptr;
    }
    destroy_team();
  }
};

/// \brief Set NVSHMEM up with the communicator of a plan, before the plan
/// makes the wrappers of its sub-communicators
template <>
struct TplCommSetup<ScopedNVSHMEMComm> {
  static void setup(MPI_Comm comm) { nvshmem_setup(comm); }
};

// Host execution spaces keep MPI
template <typename ExecutionSpace>
using TplComm =
    std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::Cuda>,
                       ScopedNVSHMEMComm, ScopedMPIComm<ExecutionSpace>>;

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
