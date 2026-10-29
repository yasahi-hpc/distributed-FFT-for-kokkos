#include <vector>
#include <mpi.h>
#include <gtest/gtest.h>
#include "KokkosFFT_Distributed_MPI_Teams.hpp"

namespace {
using PETriplet = KokkosFFT::Distributed::Impl::PETriplet;

struct TestMPITeams : public ::testing::Test {
  int m_rank   = 0;
  int m_nprocs = 1;

  virtual void SetUp() {
    ::MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &m_nprocs);
  }
};

// The world communicator is the whole parent team, whatever the PE ids are,
// as long as they are strided
void test_world(int rank, int nprocs) {
  using KokkosFFT::Distributed::Impl::compute_pe_triplet;
  using KokkosFFT::Distributed::Impl::gather_pe_triplets;

  auto triplet = compute_pe_triplet(MPI_COMM_WORLD, rank);
  EXPECT_EQ(triplet, (PETriplet{0, 1, nprocs}));

  auto triplets = gather_pe_triplets(MPI_COMM_WORLD, triplet);
  EXPECT_EQ(triplets, std::vector<PETriplet>{triplet});

  // PE ids 1, 3, 5, ... (a single PE has the stride 1)
  auto odd_triplet = compute_pe_triplet(MPI_COMM_WORLD, 2 * rank + 1);
  EXPECT_EQ(odd_triplet, (PETriplet{1, nprocs > 1 ? 2 : 1, nprocs}));
}

// The row and column communicators of a cartesian communicator created
// without reordering, as in create_pencil_communicators. With the cartesian
// rank c0 * dims[1] + c1, a row (c1 fixed) is {c1, dims[1], dims[0]} and a
// column (c0 fixed) is {c0 * dims[1], 1, dims[1]}.
void test_cartesian(int rank, int nprocs) {
  using KokkosFFT::Distributed::Impl::compute_pe_triplet;
  using KokkosFFT::Distributed::Impl::gather_pe_triplets;

  int dims[2] = {0, 0}, periods[2] = {1, 1};
  ::MPI_Dims_create(nprocs, 2, dims);

  MPI_Comm cart_comm;
  ::MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

  int coords[2];
  ::MPI_Cart_coords(cart_comm, rank, 2, coords);

  MPI_Comm row_comm, col_comm;
  int remain_row[2] = {1, 0}, remain_col[2] = {0, 1};
  ::MPI_Cart_sub(cart_comm, remain_row, &row_comm);
  ::MPI_Cart_sub(cart_comm, remain_col, &col_comm);

  auto row = compute_pe_triplet(row_comm, rank);
  auto col = compute_pe_triplet(col_comm, rank);
  EXPECT_EQ(row, (PETriplet{coords[1], dims[0] > 1 ? dims[1] : 1, dims[0]}));
  EXPECT_EQ(col, (PETriplet{coords[0] * dims[1], 1, dims[1]}));

  // Every rank gets the same triplets, in the order of the first rank that
  // has each
  std::vector<PETriplet> ref_rows, ref_cols;
  for (int c1 = 0; c1 < dims[1]; ++c1) {
    ref_rows.push_back(PETriplet{c1, dims[0] > 1 ? dims[1] : 1, dims[0]});
  }
  for (int c0 = 0; c0 < dims[0]; ++c0) {
    ref_cols.push_back(PETriplet{c0 * dims[1], 1, dims[1]});
  }
  EXPECT_EQ(gather_pe_triplets(MPI_COMM_WORLD, row), ref_rows);
  EXPECT_EQ(gather_pe_triplets(MPI_COMM_WORLD, col), ref_cols);

  ::MPI_Comm_free(&row_comm);
  ::MPI_Comm_free(&col_comm);
  ::MPI_Comm_free(&cart_comm);
}

// Even and odd ranks
void test_strided_split(int rank, int nprocs) {
  using KokkosFFT::Distributed::Impl::compute_pe_triplet;
  using KokkosFFT::Distributed::Impl::gather_pe_triplets;

  MPI_Comm comm;
  ::MPI_Comm_split(MPI_COMM_WORLD, rank % 2, rank, &comm);

  auto size_of = [&](int color) { return (nprocs - color + 1) / 2; };
  auto ref     = [&](int color) {
    return PETriplet{color, size_of(color) > 1 ? 2 : 1, size_of(color)};
  };

  auto triplet = compute_pe_triplet(comm, rank);
  EXPECT_EQ(triplet, ref(rank % 2));

  std::vector<PETriplet> ref_triplets = {ref(0)};
  if (nprocs > 1) ref_triplets.push_back(ref(1));
  EXPECT_EQ(gather_pe_triplets(MPI_COMM_WORLD, triplet), ref_triplets);

  ::MPI_Comm_free(&comm);
}

// The PEs in rank order are decreasing
void test_reversed(int rank, int nprocs) {
  using KokkosFFT::Distributed::Impl::compute_pe_triplet;

  MPI_Comm comm;
  ::MPI_Comm_split(MPI_COMM_WORLD, 0, nprocs - rank, &comm);

  if (nprocs > 1) {
    EXPECT_THROW({ [[maybe_unused]] auto t = compute_pe_triplet(comm, rank); },
                 std::runtime_error);
  } else {
    EXPECT_EQ(compute_pe_triplet(comm, rank), (PETriplet{0, 1, 1}));
  }

  ::MPI_Comm_free(&comm);
}

// Ranks {0, 1, 3} and {2} of 4: the first one is not strided
void test_not_strided(int rank) {
  using KokkosFFT::Distributed::Impl::compute_pe_triplet;

  MPI_Comm comm;
  ::MPI_Comm_split(MPI_COMM_WORLD, rank == 2, rank, &comm);

  if (rank == 2) {
    EXPECT_EQ(compute_pe_triplet(comm, rank), (PETriplet{2, 1, 1}));
  } else {
    EXPECT_THROW({ [[maybe_unused]] auto t = compute_pe_triplet(comm, rank); },
                 std::runtime_error);
  }

  ::MPI_Comm_free(&comm);
}

}  // namespace

TEST_F(TestMPITeams, World) { test_world(m_rank, m_nprocs); }

TEST_F(TestMPITeams, Cartesian) { test_cartesian(m_rank, m_nprocs); }

TEST_F(TestMPITeams, StridedSplit) { test_strided_split(m_rank, m_nprocs); }

TEST_F(TestMPITeams, Reversed) { test_reversed(m_rank, m_nprocs); }

TEST_F(TestMPITeams, NotStrided) {
  if (m_nprocs != 4) {
    GTEST_SKIP() << "The number of MPI processes should be 4 for this test";
  }
  test_not_strided(m_rank);
}
