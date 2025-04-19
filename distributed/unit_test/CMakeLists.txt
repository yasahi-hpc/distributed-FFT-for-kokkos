add_executable(unit-tests-mpi
    Test_Main.cpp
    Test_Reshape.cpp
)

find_package(MPI REQUIRED COMPONENTS CXX)

target_compile_features(unit-tests-mpi PUBLIC cxx_std_17)
target_compile_options(unit-tests-mpi PUBLIC -std=c++17)

target_link_libraries(unit-tests-mpi PUBLIC MPI::MPI_CXX GTest::gtest distributed)

# Enable GoogleTest
include(GoogleTest)
gtest_discover_tests(unit-tests-mpi PROPERTIES DISCOVERY_TIMEOUT 600 DISCOVERY_MODE PRE_TEST)
