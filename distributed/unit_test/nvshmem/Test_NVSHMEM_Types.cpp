#include <mpi.h>
#include <cstdlib>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "nvshmem/KokkosFFT_Distributed_NVSHMEM_Types.hpp"
#include "nvshmem/KokkosFFT_Distributed_NVSHMEM_Comm.hpp"
#include "nvshmem/KokkosFFT_Distributed_NVSHMEM_Buffer.hpp"
#include "nvshmem/KokkosFFT_Distributed_NVSHMEM_All2All.hpp"
#if defined(ENABLE_TPL_SHMEM_DEVICE_COMM)
#include <Kokkos_Random.hpp>
#include "KokkosFFT_Extents.hpp"
#include "KokkosFFT_Distributed_Extents.hpp"
#include "KokkosFFT_Distributed_MPI_Extents.hpp"
#include "KokkosFFT_Distributed_Exchange.hpp"
#include "nvshmem/KokkosFFT_Distributed_NVSHMEM_Exchange.hpp"
#include "../Test_CommBuffers.hpp"
#endif

namespace {
using CommType = KokkosFFT::Distributed::Impl::ScopedNVSHMEMComm;
using KokkosFFT::Distributed::Impl::SHMEMCommMode;

struct TestNVSHMEMComm : public ::testing::Test {
  int m_rank   = 0;
  int m_nprocs = 1;

  virtual void SetUp() {
    ::MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &m_nprocs);

    // Set NVSHMEM up with the world communicator, as a plan on it does,
    // whichever test runs first
    KokkosFFT::Distributed::Impl::TplCommSetup<CommType>::setup(MPI_COMM_WORLD);
  }
};

#if defined(ENABLE_TPL_SHMEM_DEVICE_COMM)
using exchange_types =
    ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                     std::pair<float, Kokkos::LayoutRight>,
                     std::pair<double, Kokkos::LayoutLeft>,
                     std::pair<double, Kokkos::LayoutRight>,
                     std::pair<Kokkos::complex<float>, Kokkos::LayoutLeft>,
                     std::pair<Kokkos::complex<float>, Kokkos::LayoutRight>,
                     std::pair<Kokkos::complex<double>, Kokkos::LayoutLeft>,
                     std::pair<Kokkos::complex<double>, Kokkos::LayoutRight>>;

/// \brief Fixture for the exchanges of a transpose. value_type is the value
/// type of the Views, layout_type their layout (and the buffers').
template <typename T>
struct TestNVSHMEMExchange : public TestNVSHMEMComm {
  using value_type  = typename T::first_type;
  using layout_type = typename T::second_type;
};
#endif

/// \brief Check that the team of a wrapper has the processes of its MPI
/// communicator, in the same order
void check_team(const CommType& comm) {
  int rank = 0, size = 1, world_rank = 0;
  ::MPI_Comm_rank(comm.comm(), &rank);
  ::MPI_Comm_size(comm.comm(), &size);
  ::MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  EXPECT_EQ(comm.rank(), rank);
  EXPECT_EQ(comm.size(), size);
  EXPECT_EQ(nvshmem_team_my_pe(comm.team()), rank);
  EXPECT_EQ(nvshmem_team_n_pes(comm.team()), size);

  std::vector<int> world_ranks(size);
  ::MPI_Allgather(&world_rank, 1, MPI_INT, world_ranks.data(), 1, MPI_INT,
                  comm.comm());
  for (int pe = 0; pe < size; ++pe) {
    EXPECT_EQ(nvshmem_team_translate_pe(comm.team(), pe, NVSHMEM_TEAM_WORLD),
              world_ranks.at(pe));
  }
}

/// \brief NVSHMEM is set up with the world communicator. A second setup
/// with a congruent communicator succeeds, with a sub-communicator it
/// throws.
void test_setup(int rank, int nprocs) {
  using KokkosFFT::Distributed::Impl::nvshmem_comm;
  using KokkosFFT::Distributed::Impl::nvshmem_setup;

  int result = MPI_UNEQUAL;
  ::MPI_Comm_compare(nvshmem_comm(), MPI_COMM_WORLD, &result);
  EXPECT_EQ(result, MPI_CONGRUENT);
  EXPECT_EQ(nvshmem_my_pe(), rank);
  EXPECT_EQ(nvshmem_n_pes(), nprocs);

  EXPECT_NO_THROW(nvshmem_setup(MPI_COMM_WORLD));

  MPI_Comm dup;
  ::MPI_Comm_dup(MPI_COMM_WORLD, &dup);
  EXPECT_NO_THROW(nvshmem_setup(dup));
  ::MPI_Comm_free(&dup);

  MPI_Comm half;
  ::MPI_Comm_split(MPI_COMM_WORLD, rank % 2, rank, &half);
  if (nprocs > 1) {
    EXPECT_THROW(nvshmem_setup(half), std::runtime_error);
  } else {
    EXPECT_NO_THROW(nvshmem_setup(half));
  }
  ::MPI_Comm_free(&half);
}

/// \brief The world communicator, and a congruent one, give the world team
void test_world() {
  CommType comm(MPI_COMM_WORLD);
  EXPECT_EQ(comm.team(), NVSHMEM_TEAM_WORLD);
  EXPECT_EQ(comm.comm(), MPI_COMM_WORLD);
  check_team(comm);

  MPI_Comm dup;
  ::MPI_Comm_dup(MPI_COMM_WORLD, &dup);
  {
    CommType comm_dup(dup);
    EXPECT_EQ(comm_dup.team(), NVSHMEM_TEAM_WORLD);
    check_team(comm_dup);
  }
  ::MPI_Comm_free(&dup);
}

/// \brief The row and column communicators of a cartesian communicator,
/// created without reordering as in create_pencil_communicators, give
/// strided teams. A communicator of all the processes gives the world team.
void test_cartesian(int nprocs) {
  int dims[2] = {0, 0}, periods[2] = {1, 1};
  ::MPI_Dims_create(nprocs, 2, dims);

  MPI_Comm cart_comm;
  ::MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

  MPI_Comm row_comm, col_comm;
  int remain_row[2] = {1, 0}, remain_col[2] = {0, 1};
  ::MPI_Cart_sub(cart_comm, remain_row, &row_comm);
  ::MPI_Cart_sub(cart_comm, remain_col, &col_comm);

  {
    // Every rank constructs the row wrapper, then the column wrapper
    CommType row(row_comm), col(col_comm);
    check_team(row);
    check_team(col);
    EXPECT_EQ(row.team() == NVSHMEM_TEAM_WORLD, row.size() == nprocs);
    EXPECT_EQ(col.team() == NVSHMEM_TEAM_WORLD, col.size() == nprocs);
  }

  ::MPI_Comm_free(&row_comm);
  ::MPI_Comm_free(&col_comm);
  ::MPI_Comm_free(&cart_comm);
}

/// \brief Teams are destroyed with their wrappers: making many in a row does
/// not run out of teams
void test_many_teams(int rank) {
  MPI_Comm half;
  ::MPI_Comm_split(MPI_COMM_WORLD, rank % 2, rank, &half);
  for (int i = 0; i < 64; ++i) {
    CommType comm(half);
    check_team(comm);
  }
  ::MPI_Comm_free(&half);
}

/// \brief Communicators that are not strided in the world communicator
/// throw, on every rank
void test_not_strided(int rank, int nprocs) {
  // Decreasing order
  MPI_Comm reversed;
  ::MPI_Comm_split(MPI_COMM_WORLD, 0, nprocs - rank, &reversed);
  if (nprocs > 1) {
    EXPECT_THROW({ CommType comm(reversed); }, std::runtime_error);
  } else {
    EXPECT_NO_THROW({ CommType comm(reversed); });
  }
  ::MPI_Comm_free(&reversed);

  // Ranks {0, 1, 3} and {2} of 4: the rank of {2} throws too
  if (nprocs == 4) {
    MPI_Comm comm013;
    ::MPI_Comm_split(MPI_COMM_WORLD, rank == 2, rank, &comm013);
    EXPECT_THROW({ CommType comm(comm013); }, std::runtime_error);
    ::MPI_Comm_free(&comm013);
  }
}

/// \brief ScopedNVSHMEMComm is movable but not copyable. A move hands over
/// the team and leaves NVSHMEM_TEAM_INVALID behind, which the destructor and
/// the move assignment must not destroy.
void test_move(int rank) {
  static_assert(!std::is_default_constructible_v<CommType>);
  static_assert(!std::is_copy_constructible_v<CommType>);
  static_assert(!std::is_copy_assignable_v<CommType>);
  static_assert(std::is_nothrow_move_constructible_v<CommType>);
  static_assert(std::is_nothrow_move_assignable_v<CommType>);

  MPI_Comm half;
  ::MPI_Comm_split(MPI_COMM_WORLD, rank % 2, rank, &half);

  CommType comm0(half);
  const nvshmem_team_t team = comm0.team();
  EXPECT_NE(team, NVSHMEM_TEAM_INVALID);

  // Move construction: the team is handed over
  CommType comm1(std::move(comm0));
  EXPECT_EQ(comm1.team(), team);
  EXPECT_EQ(comm1.comm(), half);
  EXPECT_EQ(comm0.team(),
            NVSHMEM_TEAM_INVALID);  // NOLINT(bugprone-use-after-move)
  check_team(comm1);

  // Move assignment over a wrapper that owns a team: the old one is
  // destroyed (collective, on all ranks) and the new one is handed over
  CommType comm2(half);
  comm2 = std::move(comm1);
  EXPECT_EQ(comm2.team(), team);
  EXPECT_EQ(comm1.team(),
            NVSHMEM_TEAM_INVALID);  // NOLINT(bugprone-use-after-move)
  check_team(comm2);

  // Move assignment from a moved-from wrapper: nothing to hand over
  CommType comm3(MPI_COMM_WORLD);
  comm3 = std::move(comm0);
  EXPECT_EQ(comm3.team(), NVSHMEM_TEAM_INVALID);

  // comm0, comm1 and comm3 are destroyed without a team; comm2 destroys it
  // before half is freed
  comm2 = std::move(comm3);
  ::MPI_Comm_free(&half);
}

/// \brief An exchanged CommBuffer is on the symmetric heap, with the same size
/// on every rank (the maximum of the requested sizes). A local one is a
/// plain View.
void test_comm_buffer(int rank, int nprocs) {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using BufferType =
      KokkosFFT::Distributed::Impl::CommBuffer<CommType, execution_space,
                                               Kokkos::complex<double>>;
  using KokkosFFT::Distributed::Impl::is_nvshmem_symmetric;

  static_assert(!std::is_default_constructible_v<BufferType>);
  static_assert(!std::is_copy_constructible_v<BufferType>);
  static_assert(!std::is_move_constructible_v<BufferType>);

  CommType comm(MPI_COMM_WORLD);
  const std::size_t n = 10 + rank;

  {
    BufferType exchanged("exchanged", n, true);
    EXPECT_EQ(exchanged.size(), std::size_t(10 + nprocs - 1));
    EXPECT_TRUE(exchanged.is_symmetric());
    EXPECT_TRUE(is_nvshmem_symmetric(exchanged.data()));

    BufferType local("local", n, false);
    EXPECT_EQ(local.size(), n);
    EXPECT_FALSE(local.is_symmetric());
    EXPECT_NE(local.data(), nullptr);
    EXPECT_FALSE(is_nvshmem_symmetric(local.data()));
  }

  // Nothing to allocate on any rank
  {
    BufferType empty("empty", 0, true);
    EXPECT_EQ(empty.size(), std::size_t(0));
    EXPECT_EQ(empty.data(), nullptr);
  }

  // The symmetric allocations are freed with the buffers
  for (int i = 0; i < 64; ++i) {
    BufferType exchanged("exchanged", std::size_t(1) << 20, true);
    EXPECT_TRUE(exchanged.is_symmetric());
  }
}

/// \brief The all2all of the wrapper on a sub-communicator (strided team),
/// with symmetric buffers: rank r of the team receives value s * size + r
/// from rank s
void test_all2all_team(int rank) {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using ViewType = Kokkos::View<double**, Kokkos::LayoutRight, execution_space>;
  using BufferType =
      KokkosFFT::Distributed::Impl::CommBuffer<CommType, execution_space,
                                               double>;

  MPI_Comm half;
  ::MPI_Comm_split(MPI_COMM_WORLD, rank % 2, rank, &half);
  {
    CommType comm(half);
    const int size = comm.size(), me = comm.rank();
    const std::size_t count = 5;
    BufferType send_buffer("send", size * count, true),
        recv_buffer("recv", size * count, true);
    ViewType send(send_buffer.data(), size, count),
        recv(recv_buffer.data(), size, count);

    auto h_send = Kokkos::create_mirror_view(send);
    for (int p = 0; p < size; ++p) {
      for (std::size_t i = 0; i < count; ++i) {
        h_send(p, i) = (me * size + p) * 100.0 + i;
      }
    }
    Kokkos::deep_copy(send, h_send);

    // Twice, with the buffers swapped, as consecutive transposes do
    KokkosFFT::Distributed::Impl::all2all(send, recv, comm);
    KokkosFFT::Distributed::Impl::all2all(recv, send, comm);
    auto h_back =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send);
    KokkosFFT::Distributed::Impl::all2all(send, recv, comm);
    auto h_recv =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), recv);

    for (int p = 0; p < size; ++p) {
      for (std::size_t i = 0; i < count; ++i) {
        EXPECT_EQ(h_recv(p, i), (p * size + me) * 100.0 + i);
        EXPECT_EQ(h_back(p, i), h_send(p, i));
      }
    }
  }
  ::MPI_Comm_free(&half);
}

/// \brief Set or unset an environment variable, and restore its previous
/// value at the end of the scope
struct ScopedEnv {
  std::string m_name, m_old;
  bool m_had_old = false;

  ScopedEnv(const std::string& name, const char* value) : m_name(name) {
    if (const char* old = std::getenv(name.c_str()); old != nullptr) {
      m_old     = old;
      m_had_old = true;
    }
    if (value != nullptr) {
      ::setenv(name.c_str(), value, 1);
    } else {
      ::unsetenv(name.c_str());
    }
  }
  ~ScopedEnv() {
    if (m_had_old) {
      ::setenv(m_name.c_str(), m_old.c_str(), 1);
    } else {
      ::unsetenv(m_name.c_str());
    }
  }
};

/// \brief The default mode is device if it is compiled, host otherwise, and
/// KOKKOSFFT_SHMEM_COMM overrides it. The host mode has no signal word. A
/// mode that differs between ranks, or the device mode when it is not
/// compiled, throws on every rank.
void test_mode(int rank, int nprocs) {
  using KokkosFFT::Distributed::Impl::default_shmem_comm_mode;
#if defined(ENABLE_TPL_SHMEM_DEVICE_COMM)
  constexpr bool has_device = true;
#else
  constexpr bool has_device = false;
#endif

  {
    ScopedEnv env("KOKKOSFFT_SHMEM_COMM", nullptr);
    EXPECT_EQ(default_shmem_comm_mode(),
              has_device ? SHMEMCommMode::device : SHMEMCommMode::host);
  }
  {
    ScopedEnv env("KOKKOSFFT_SHMEM_COMM", "host");
    EXPECT_EQ(default_shmem_comm_mode(), SHMEMCommMode::host);
    CommType comm(MPI_COMM_WORLD);
    EXPECT_EQ(comm.mode(), SHMEMCommMode::host);
    EXPECT_EQ(comm.signal(), nullptr);
  }
  {
    ScopedEnv env("KOKKOSFFT_SHMEM_COMM", "device");
    EXPECT_EQ(default_shmem_comm_mode(), SHMEMCommMode::device);
  }
  {
    ScopedEnv env("KOKKOSFFT_SHMEM_COMM", "gpu");
    EXPECT_THROW(default_shmem_comm_mode(), std::runtime_error);
  }

  const auto exec = Kokkos::DefaultExecutionSpace();
  if (has_device) {
    CommType comm(MPI_COMM_WORLD, exec, SHMEMCommMode::device);
    EXPECT_EQ(comm.mode(), SHMEMCommMode::device);
    EXPECT_NE(comm.signal(), nullptr);
    EXPECT_EQ(comm.epoch(), 0u);
  } else {
    EXPECT_THROW(
        { CommType comm(MPI_COMM_WORLD, exec, SHMEMCommMode::device); },
        std::runtime_error);
  }

  if (nprocs > 1) {
    const auto mode = rank == 0 ? SHMEMCommMode::host : SHMEMCommMode::device;
    EXPECT_THROW({ CommType comm(MPI_COMM_WORLD, exec, mode); },
                 std::runtime_error);
  }
}

#if defined(ENABLE_TPL_SHMEM_DEVICE_COMM)
/// \brief The device-initiated all-to-all on a sub-communicator (strided
/// team): many exchanges in a row, more than the transposes of a plan, with
/// the buffers swapped, as consecutive transposes do. Rank r of the team
/// receives value s * size + r from rank s. The signal word counts the
/// chunks of every exchange (epoch).
void test_put_all2all(int rank) {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using ViewType = Kokkos::View<double**, Kokkos::LayoutRight, execution_space>;
  using BufferType =
      KokkosFFT::Distributed::Impl::CommBuffer<CommType, execution_space,
                                               double>;
  using KokkosFFT::Distributed::Impl::put_all2all;

  MPI_Comm half;
  ::MPI_Comm_split(MPI_COMM_WORLD, rank % 2, rank, &half);
  {
    const auto exec = execution_space();
    CommType comm(half, exec, SHMEMCommMode::device);
    const int size = comm.size(), me = comm.rank();
    const std::size_t count = 5;
    BufferType buffer0("buffer0", size * count, true),
        buffer1("buffer1", size * count, true);
    ViewType view0(buffer0.data(), size, count),
        view1(buffer1.data(), size, count);

    auto h_send = Kokkos::create_mirror_view(view0);
    for (int p = 0; p < size; ++p) {
      for (std::size_t i = 0; i < count; ++i) {
        h_send(p, i) = (me * size + p) * 100.0 + i;
      }
    }
    Kokkos::deep_copy(view0, h_send);

    // An even number of exchanges gives the data back; one more transposes it
    const int nexchanges = 20;
    for (int e = 0; e < nexchanges; ++e) {
      if (e % 2 == 0) {
        put_all2all(exec, view0, view1, comm);
      } else {
        put_all2all(exec, view1, view0, comm);
      }
    }
    auto h_back =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view0);
    put_all2all(exec, view0, view1, comm);
    auto h_recv =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view1);

    EXPECT_EQ(comm.epoch(), std::uint64_t(nexchanges + 1));
    std::uint64_t h_signal = 0;
    Kokkos::deep_copy(
        h_signal,
        Kokkos::View<std::uint64_t, Kokkos::CudaSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>(comm.signal()));
    EXPECT_EQ(h_signal, comm.epoch() * size);

    for (int p = 0; p < size; ++p) {
      for (std::size_t i = 0; i < count; ++i) {
        EXPECT_EQ(h_recv(p, i), (p * size + me) * 100.0 + i);
        EXPECT_EQ(h_back(p, i), h_send(p, i));
      }
    }

    // A wrapper in host mode, or plain Views, are rejected
    CommType host_comm(half, exec, SHMEMCommMode::host);
    EXPECT_THROW(put_all2all(exec, view0, view1, host_comm),
                 std::runtime_error);
    ViewType plain("plain", size, count);
    EXPECT_THROW(put_all2all(exec, plain, view1, comm), std::runtime_error);
  }
  ::MPI_Comm_free(&half);
}

/// \brief A move hands over the signal word and the epoch
void test_move_signal() {
  const auto exec = Kokkos::DefaultExecutionSpace();
  CommType comm0(MPI_COMM_WORLD, exec, SHMEMCommMode::device);
  auto* signal = comm0.signal();
  comm0.next_epoch();

  CommType comm1(std::move(comm0));
  EXPECT_EQ(comm1.signal(), signal);
  EXPECT_EQ(comm1.epoch(), 1u);
  EXPECT_EQ(comm0.signal(), nullptr);  // NOLINT(bugprone-use-after-move)

  CommType comm2(MPI_COMM_WORLD, exec, SHMEMCommMode::device);
  comm2 = std::move(comm1);
  EXPECT_EQ(comm2.signal(), signal);
  EXPECT_EQ(comm2.mode(), SHMEMCommMode::device);
  EXPECT_EQ(comm1.signal(), nullptr);  // NOLINT(bugprone-use-after-move)
}

/// \brief KOKKOSFFT_SHMEM_FUSED_PACK selects the fused pack ("1", the
/// default) or not ("0"), read at construction
void test_fused_pack_env() {
  using KokkosFFT::Distributed::Impl::default_shmem_fused_pack;
  {
    ScopedEnv env("KOKKOSFFT_SHMEM_FUSED_PACK", nullptr);
    EXPECT_TRUE(default_shmem_fused_pack());
  }
  {
    ScopedEnv env("KOKKOSFFT_SHMEM_FUSED_PACK", "0");
    EXPECT_FALSE(default_shmem_fused_pack());
    CommType comm(MPI_COMM_WORLD, Kokkos::DefaultExecutionSpace(),
                  SHMEMCommMode::device);
    EXPECT_FALSE(comm.fused_pack());
  }
  {
    ScopedEnv env("KOKKOSFFT_SHMEM_FUSED_PACK", "1");
    EXPECT_TRUE(default_shmem_fused_pack());
  }
  {
    ScopedEnv env("KOKKOSFFT_SHMEM_FUSED_PACK", "yes");
    EXPECT_THROW(default_shmem_fused_pack(), std::runtime_error);
  }
}

/// \brief The peer table of a symmetric buffer: the entry of this PE is the
/// local address, the others are the addresses nvshmem_ptr gives (0 if not
/// reachable by load/store), and m_has_peer tells whether one of them is
/// set. A table is computed once per buffer. A plain View has no peers.
void test_peer_table(int rank) {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using BufferType =
      KokkosFFT::Distributed::Impl::CommBuffer<CommType, execution_space,
                                               double>;
  MPI_Comm half;
  ::MPI_Comm_split(MPI_COMM_WORLD, rank % 2, rank, &half);
  {
    CommType comm(half, execution_space(), SHMEMCommMode::device);
    BufferType buffer("buffer", 16, true);

    const auto table = comm.peer_table(buffer.data());
    EXPECT_EQ(table.m_buffer, buffer.data());
    auto h_peers =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), table.m_peers);
    ASSERT_EQ(h_peers.extent(0), std::size_t(comm.size()));
    EXPECT_EQ(h_peers(comm.rank()),
              reinterpret_cast<std::uintptr_t>(buffer.data()));
    bool has_peer = false;
    for (int pe = 0; pe < comm.size(); ++pe) {
      EXPECT_EQ(h_peers(pe), reinterpret_cast<std::uintptr_t>(nvshmem_ptr(
                                 buffer.data(), comm.world_pe(pe))));
      if (pe != comm.rank() && h_peers(pe) != 0) has_peer = true;
    }
    EXPECT_EQ(table.m_has_peer, has_peer);

    // Kept: the same device View on the second call
    EXPECT_EQ(comm.peer_table(buffer.data()).m_peers.data(),
              table.m_peers.data());

    // Not on the symmetric heap: no address on any PE
    Kokkos::View<double*, execution_space> plain("plain", 16);
    const auto plain_table = comm.peer_table(plain.data());
    auto h_plain = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                       plain_table.m_peers);
    EXPECT_FALSE(plain_table.m_has_peer);
    for (int pe = 0; pe < comm.size(); ++pe) EXPECT_EQ(h_plain(pe), 0u);
  }
  ::MPI_Comm_free(&half);
}

/// \brief The three exchanges of a transpose on a strided team (the 2D
/// transpose of Test_TransBlock.cpp between an x-pencil and a y-pencil):
/// host mode (default_exchange), device mode without the fused pack (D1)
/// and with it (D2), give the same output, and back the input, over
/// several exchanges in a row with the same buffers. D2 falls back to D1
/// when no peer is reachable by load/store (one process).
template <typename T, typename LayoutType>
void test_exchange_variants(int rank) {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using View2DType      = Kokkos::View<T**, LayoutType, execution_space>;
  using View3DType      = Kokkos::View<T***, LayoutType, execution_space>;
  using extents_type    = std::array<std::size_t, 2>;
  using KokkosFFT::Distributed::Impl::exchange;
  using KokkosFFT::Impl::create_layout;

  MPI_Comm half;
  ::MPI_Comm_split(MPI_COMM_WORLD, rank % 2, rank, &half);
  int nprocs = 1;
  ::MPI_Comm_size(half, &nprocs);
  const std::size_t n = nprocs;

  const extents_type global_extents{11, 10}, map01{0, 1};
  const extents_type topology0{1, n}, topology1{n, 1};
  // Not structured bindings: they are captured by the lambda below
  const auto extents0 =
      std::get<0>(KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_extents, topology0, half));
  const auto extents1 =
      std::get<0>(KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_extents, topology1, half));
  const auto buffer_extents =
      KokkosFFT::Distributed::Impl::compute_buffer_extents<LayoutType>(
          global_extents, topology0, topology1);

  execution_space exec;
  View2DType in("in", create_layout<LayoutType>(extents0));
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345 + rank);
  Kokkos::fill_random(in, random_pool, 1.0);
  auto h_in = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), in);

  // Forward and backward transposes, several times in a row
  auto run = [&](SHMEMCommMode mode, const char* fused, bool& used_fused) {
    ScopedEnv env("KOKKOSFFT_SHMEM_FUSED_PACK", fused);
    CommType comm(half, exec, mode);
    auto buffers = make_comm_buffers<View3DType>(
        comm, create_layout<LayoutType>(buffer_extents));
    View2DType out("out", create_layout<LayoutType>(extents1)),
        back("back", create_layout<LayoutType>(extents0));
    used_fused = mode == SHMEMCommMode::device && comm.fused_pack() &&
                 comm.peer_table(buffers.m_recv.data()).m_has_peer;
    for (int i = 0; i < 5; ++i) {
      exchange(exec, comm, in, buffers.m_send, buffers.m_recv, out, map01, 0,
               map01, 1);
      exchange(exec, comm, out, buffers.m_send, buffers.m_recv, back, map01, 1,
               map01, 0);
    }
    exec.fence();
    return std::make_pair(
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out),
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), back));
  };

  bool host_fused = false, d1_fused = false, d2_fused = false;
  auto [h_out_host, h_back_host] = run(SHMEMCommMode::host, "1", host_fused);
  auto [h_out_d1, h_back_d1]     = run(SHMEMCommMode::device, "0", d1_fused);
  auto [h_out_d2, h_back_d2]     = run(SHMEMCommMode::device, "1", d2_fused);
  EXPECT_FALSE(host_fused);
  EXPECT_FALSE(d1_fused);
  if (nprocs == 1) EXPECT_FALSE(d2_fused);

  for (std::size_t i0 = 0; i0 < h_out_host.extent(0); ++i0) {
    for (std::size_t i1 = 0; i1 < h_out_host.extent(1); ++i1) {
      EXPECT_EQ(h_out_d1(i0, i1), h_out_host(i0, i1));
      EXPECT_EQ(h_out_d2(i0, i1), h_out_host(i0, i1));
    }
  }
  for (std::size_t i0 = 0; i0 < h_in.extent(0); ++i0) {
    for (std::size_t i1 = 0; i1 < h_in.extent(1); ++i1) {
      EXPECT_EQ(h_back_host(i0, i1), h_in(i0, i1));
      EXPECT_EQ(h_back_d1(i0, i1), h_in(i0, i1));
      EXPECT_EQ(h_back_d2(i0, i1), h_in(i0, i1));
    }
  }
  ::MPI_Comm_free(&half);
}

/// \brief An exchange whose input is staged in its receive buffer, as the
/// plans do when the input of a transpose was in the send buffer
/// (BlockExecutor::transpose swaps the buffers). The peers write into the
/// receive buffer, so the pack cannot be fused there: uses_fused_pack is
/// false, fused_put_all2all throws, and every variant gives the output of
/// the host mode with an input that is not in a buffer, several times in a
/// row. With a separate input, the fused pack is used if a peer is reachable.
template <typename T, typename LayoutType>
void test_exchange_recv_input(int rank) {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using View2DType      = Kokkos::View<T**, LayoutType, execution_space>;
  using View3DType      = Kokkos::View<T***, LayoutType, execution_space>;
  using extents_type    = std::array<std::size_t, 2>;
  using KokkosFFT::Distributed::Impl::exchange;
  using KokkosFFT::Distributed::Impl::fused_put_all2all;
  using KokkosFFT::Distributed::Impl::NVSHMEMPeerTable;
  using KokkosFFT::Distributed::Impl::uses_fused_pack;
  using KokkosFFT::Impl::create_layout;

  MPI_Comm half;
  ::MPI_Comm_split(MPI_COMM_WORLD, rank % 2, rank, &half);
  int nprocs = 1;
  ::MPI_Comm_size(half, &nprocs);
  const std::size_t n = nprocs;

  // Large enough for the kernels of the PEs to overlap in time
  const extents_type global_extents{301, 256}, map01{0, 1};
  const extents_type topology0{1, n}, topology1{n, 1};
  const auto extents0 =
      std::get<0>(KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_extents, topology0, half));
  const auto extents1 =
      std::get<0>(KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_extents, topology1, half));
  const auto buffer_extents =
      KokkosFFT::Distributed::Impl::compute_buffer_extents<LayoutType>(
          global_extents, topology0, topology1);

  execution_space exec;
  View2DType in("in", create_layout<LayoutType>(extents0));
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345 + rank);
  Kokkos::fill_random(in, random_pool, 1.0);

  // The reference: host mode, the input in its own View
  View2DType ref("ref", create_layout<LayoutType>(extents1));
  {
    CommType comm(half, exec, SHMEMCommMode::host);
    auto buffers = make_comm_buffers<View3DType>(
        comm, create_layout<LayoutType>(buffer_extents));
    exchange(exec, comm, in, buffers.m_send, buffers.m_recv, ref, map01, 0,
             map01, 1);
  }
  auto h_ref = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ref);

  auto run = [&](SHMEMCommMode mode, const char* fused) {
    ScopedEnv env("KOKKOSFFT_SHMEM_FUSED_PACK", fused);
    CommType comm(half, exec, mode);
    auto buffers = make_comm_buffers<View3DType>(
        comm, create_layout<LayoutType>(buffer_extents));
    View2DType staged(buffers.m_recv.data(),
                      create_layout<LayoutType>(extents0));
    View2DType out("out", create_layout<LayoutType>(extents1));

    NVSHMEMPeerTable peers;
    EXPECT_FALSE(uses_fused_pack(comm, staged, buffers.m_recv, peers));
    if (mode == SHMEMCommMode::device && comm.fused_pack()) {
      const bool has_peer = comm.peer_table(buffers.m_recv.data()).m_has_peer;
      EXPECT_EQ(uses_fused_pack(comm, in, buffers.m_recv, peers), has_peer);
      EXPECT_THROW(
          fused_put_all2all(exec, staged, buffers.m_send, buffers.m_recv, comm,
                            comm.peer_table(buffers.m_recv.data()), map01, 0),
          std::runtime_error);
    }

    for (int i = 0; i < 5; ++i) {
      // The exchange overwrites the receive buffer: stage the input again
      Kokkos::deep_copy(exec, staged, in);
      exchange(exec, comm, staged, buffers.m_send, buffers.m_recv, out, map01,
               0, map01, 1);
      exec.fence();
      auto h_out =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
      std::size_t mismatches = 0;
      for (std::size_t i0 = 0; i0 < h_ref.extent(0); ++i0) {
        for (std::size_t i1 = 0; i1 < h_ref.extent(1); ++i1) {
          if (h_out(i0, i1) != h_ref(i0, i1)) ++mismatches;
        }
      }
      EXPECT_EQ(mismatches, 0u) << "exchange " << i;
    }
  };

  run(SHMEMCommMode::host, "1");
  run(SHMEMCommMode::device, "0");
  run(SHMEMCommMode::device, "1");
  ::MPI_Comm_free(&half);
}
#endif

}  // namespace

TEST_F(TestNVSHMEMComm, mode) { test_mode(m_rank, m_nprocs); }

#if defined(ENABLE_TPL_SHMEM_DEVICE_COMM)
TEST_F(TestNVSHMEMComm, put_all2all) { test_put_all2all(m_rank); }

TEST_F(TestNVSHMEMComm, move_signal) { test_move_signal(); }

TEST_F(TestNVSHMEMComm, fused_pack_env) { test_fused_pack_env(); }

TEST_F(TestNVSHMEMComm, peer_table) { test_peer_table(m_rank); }

TYPED_TEST_SUITE(TestNVSHMEMExchange, exchange_types);

TYPED_TEST(TestNVSHMEMExchange, exchange_variants) {
  using value_type  = typename TestFixture::value_type;
  using layout_type = typename TestFixture::layout_type;
  test_exchange_variants<value_type, layout_type>(this->m_rank);
}

TYPED_TEST(TestNVSHMEMExchange, exchange_recv_input) {
  using value_type  = typename TestFixture::value_type;
  using layout_type = typename TestFixture::layout_type;
  test_exchange_recv_input<value_type, layout_type>(this->m_rank);
}
#endif

TEST_F(TestNVSHMEMComm, comm_buffer) { test_comm_buffer(m_rank, m_nprocs); }

TEST_F(TestNVSHMEMComm, all2all_team) { test_all2all_team(m_rank); }

TEST_F(TestNVSHMEMComm, setup) { test_setup(m_rank, m_nprocs); }

TEST_F(TestNVSHMEMComm, world) { test_world(); }

TEST_F(TestNVSHMEMComm, cartesian) { test_cartesian(m_nprocs); }

TEST_F(TestNVSHMEMComm, many_teams) { test_many_teams(m_rank); }

TEST_F(TestNVSHMEMComm, not_strided) { test_not_strided(m_rank, m_nprocs); }

TEST_F(TestNVSHMEMComm, move) { test_move(m_rank); }
