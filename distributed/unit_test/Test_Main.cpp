#include <mpi.h>
#include <gtest/gtest.h>

class MpiEnvironment : public ::testing::Environment {
 public:
  ~MpiEnvironment() override {}

  // Override this to define how to set up the environment.
  void SetUp() override {
    m_comm = MPI_COMM_WORLD;
    // Ensure all processes start tests together
    ::MPI_Barrier(m_comm);
  }

  // Override this to define how to tear down the environment.
  void TearDown() override {
    // Ensure all processes finish tests together
    ::MPI_Barrier(m_comm);
  }

  MPI_Comm m_comm;
};

int main(int argc, char *argv[]) {
  // Initialize MPI first
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if (provided != MPI_THREAD_MULTIPLE) {
    throw std::runtime_error("MPI_THREAD_MULTIPLE is needed");
  }

  // Initialize google test
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MpiEnvironment());

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  auto &test_listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (rank != 0)
    delete test_listeners.Release(test_listeners.default_result_printer());

  // run tests
  auto result = RUN_ALL_TESTS();

  // Finalize MPI before exiting
  MPI_Finalize();

  return result;
}
