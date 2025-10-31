if(ENABLE_CUFFT_MP AND Kokkos_ENABLE_CUDA)
    set(SOURCES Test_Main.cpp Test_Types.cpp Test_TplPlan.cpp)
elseif(ENABLE_ROCFFT_MPI AND Kokkos_ENABLE_HIP AND KokkosFFT_ENABLE_ROCFFT)
    set(SOURCES
        Test_Main.cpp
        Test_Types.cpp
        Test_All2All.cpp
        Test_Mapping.cpp
        Test_Pack.cpp
        Test_Unpack.cpp
        Test_Extents.cpp
        Test_MPI_Types.cpp
        Test_MPI_Helper.cpp
        Test_Common_Utils.cpp
        Test_Topologies.cpp
        Test_TransBlock.cpp
        Test_SharedPlan.cpp
        Test_Plan.cpp
        #Test_SlabBlockAnalyses.cpp
        Test_SlabPlan.cpp
        Test_PencilPlan.cpp
        Test_TplPlan.cpp
    )
else()
    set(SOURCES
        Test_Main.cpp
        Test_Types.cpp
        Test_All2All.cpp
        Test_Mapping.cpp
        Test_Pack.cpp
        Test_Unpack.cpp
        Test_Extents.cpp
        Test_MPI_Types.cpp
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
endif()

add_executable(unit-tests-mpi ${SOURCES})
find_package(MPI REQUIRED COMPONENTS CXX)

target_compile_features(unit-tests-mpi PUBLIC cxx_std_17)
target_compile_options(unit-tests-mpi PUBLIC -std=c++17)

target_link_libraries(unit-tests-mpi PUBLIC MPI::MPI_CXX GTest::gtest GTest::gmock distributed)

# Enable GoogleTest
include(GoogleTest)

# This is the core instruction for CTest.
# NAME: A descriptive name for the test.
# COMMAND: The exact command to run.
#  - ${MPIEXEC_EXECUTABLE}: Variable from find_package(MPI) for mpiexec/mpirun.
#  - -n 2: The flag to specify running with 2 processes.
#  - $<TARGET_FILE:unit-tests-mpi>: A CMake generator expression that correctly
#   resolves to the path of our compiled executable.
add_test(
  NAME UnitTests1MPI
  COMMAND ${MPIEXEC_EXECUTABLE} -n 1 $<TARGET_FILE:unit-tests-mpi>
)
add_test(
  NAME UnitTests2MPI
  COMMAND ${MPIEXEC_EXECUTABLE} -n 2 $<TARGET_FILE:unit-tests-mpi>
)
add_test(
  NAME UnitTests4MPI
  COMMAND ${MPIEXEC_EXECUTABLE} -n 4 $<TARGET_FILE:unit-tests-mpi>
)

set_property(TEST UnitTests1MPI PROPERTY TIMEOUT 600)
set_property(TEST UnitTests2MPI PROPERTY TIMEOUT 600)
set_property(TEST UnitTests4MPI PROPERTY TIMEOUT 1800)

if(ENABLE_NCCL AND Kokkos_ENABLE_CUDA)
  set(NCCL_SOURCES Test_Main.cpp nccl/Test_NCCL_Types.cpp)
  add_executable(unit-tests-nccl ${NCCL_SOURCES})

  target_compile_features(unit-tests-nccl PUBLIC cxx_std_17)
  target_link_libraries(unit-tests-nccl PUBLIC MPI::MPI_CXX GTest::gtest GTest::gmock distributed)

  add_test(
    NAME UnitTests1NCCL
    COMMAND ${MPIEXEC_EXECUTABLE} -n 1 $<TARGET_FILE:unit-tests-nccl>
  )
  add_test(
    NAME UnitTests2NCCL
    COMMAND ${MPIEXEC_EXECUTABLE} -n 2 $<TARGET_FILE:unit-tests-nccl>
  )
  add_test(
    NAME UnitTests4NCCL
    COMMAND ${MPIEXEC_EXECUTABLE} -n 4 $<TARGET_FILE:unit-tests-nccl>
  )

  set_property(TEST UnitTests1NCCL PROPERTY TIMEOUT 600)
  set_property(TEST UnitTests2NCCL PROPERTY TIMEOUT 600)
  set_property(TEST UnitTests4NCCL PROPERTY TIMEOUT 1800)
endif()
