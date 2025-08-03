set(SOURCES
    Test_Main.cpp
    Test_All2All.cpp
    Test_Mapping.cpp
    Test_Pack.cpp
    Test_Unpack.cpp
    Test_Extents.cpp
    Test_MPI_Helper.cpp
    Test_Common_Utils.cpp
    Test_Topologies.cpp
    Test_TransBlock.cpp
    Test_SharedPlan.cpp
    Test_Plan.cpp
    #Test_SlabBlockAnalyses.cpp
    Test_SlabPlan.cpp
    Test_PencilPlan.cpp
)

if(ENABLE_CUFFT_MP AND Kokkos_ENABLE_CUDA)
list(APPEND SOURCES Test_TplPlan.cpp)
endif()

add_executable(unit-tests-mpi ${SOURCES})
find_package(MPI REQUIRED COMPONENTS CXX)

target_compile_features(unit-tests-mpi PUBLIC cxx_std_17)
target_compile_options(unit-tests-mpi PUBLIC -std=c++17)

target_link_libraries(unit-tests-mpi PUBLIC MPI::MPI_CXX GTest::gtest GTest::gmock distributed)

# Enable GoogleTest
include(GoogleTest)
gtest_discover_tests(unit-tests-mpi PROPERTIES DISCOVERY_TIMEOUT 600 DISCOVERY_MODE PRE_TEST)
