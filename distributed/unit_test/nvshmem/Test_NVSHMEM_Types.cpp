#include <mpi.h>
#include <type_traits>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "nvshmem/KokkosFFT_Distributed_NVSHMEM_Types.hpp"
#include "nvshmem/KokkosFFT_Distributed_NVSHMEM_Comm.hpp"

namespace {
using CommType = KokkosFFT::Distributed::Impl::ScopedNVSHMEMComm;

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

}  // namespace

TEST_F(TestNVSHMEMComm, setup) { test_setup(m_rank, m_nprocs); }

TEST_F(TestNVSHMEMComm, world) { test_world(); }

TEST_F(TestNVSHMEMComm, cartesian) { test_cartesian(m_nprocs); }

TEST_F(TestNVSHMEMComm, many_teams) { test_many_teams(m_rank); }

TEST_F(TestNVSHMEMComm, not_strided) { test_not_strided(m_rank, m_nprocs); }

TEST_F(TestNVSHMEMComm, move) { test_move(m_rank); }
