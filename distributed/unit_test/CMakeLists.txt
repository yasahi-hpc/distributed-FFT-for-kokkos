find_package(MPI REQUIRED COMPONENTS CXX)

# Tests that do not use MPI: run with one process only
set(SERIAL_SOURCES
    Test_Main.cpp
    Test_Types.cpp
    Test_TplComm.cpp
    Test_Mapping.cpp
    Test_Pack.cpp
    Test_Unpack.cpp
    Test_Extents.cpp
    Test_MPI_Types.cpp
    Test_ContainerAnalyses.cpp
    Test_Topologies.cpp
    Test_BlockAnalysesHelpers.cpp
    Test_Schedule.cpp
    Test_PlanTraits.cpp
)

# Tests that use MPI: run with several numbers of processes
set(MPI_SOURCES
    Test_Main.cpp
    Test_All2All.cpp
    Test_MPI_Extents.cpp
    Test_MPI_Teams.cpp
    Test_TransBlock.cpp
    Test_BasePlan.cpp
    Test_BlockAnalyses.cpp
    Test_SharedPlan.cpp
    Test_Plan.cpp
    Test_SlabPlan.cpp
    Test_PencilPlan.cpp
)

if(ENABLE_CUFFT_MP AND Kokkos_ENABLE_CUDA)
  set(SERIAL_SOURCES Test_Main.cpp Test_Types.cpp)
  set(MPI_SOURCES Test_Main.cpp Test_TplPlan.cpp)
elseif(ENABLE_ROCFFT_MPI AND Kokkos_ENABLE_HIP AND KokkosFFT_ENABLE_ROCFFT)
  list(APPEND MPI_SOURCES Test_TplPlan.cpp)
endif()

add_executable(unit-tests ${SERIAL_SOURCES})
target_link_libraries(unit-tests PUBLIC MPI::MPI_CXX GTest::gtest GTest::gmock distributed)

# Stop if the serial tests are run with several processes
target_compile_definitions(unit-tests PRIVATE KOKKOSFFT_SERIAL_UNIT_TESTS)

add_executable(unit-tests-mpi ${MPI_SOURCES})
target_link_libraries(unit-tests-mpi PUBLIC MPI::MPI_CXX GTest::gtest GTest::gmock distributed)

# Add one CTest test named <target>-np<N> per number of processes N
function(add_mpi_unit_tests target)
  cmake_parse_arguments(ARG "" "TIMEOUT" "NPROCS" ${ARGN})
  foreach(np IN LISTS ARG_NPROCS)
    add_test(NAME ${target}-np${np} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS}
                                            $<TARGET_FILE:${target}> ${MPIEXEC_POSTFLAGS}
    )
    set_tests_properties(${target}-np${np} PROPERTIES TIMEOUT ${ARG_TIMEOUT} PROCESSORS ${np})
  endforeach()
endfunction()

add_mpi_unit_tests(unit-tests NPROCS 1 TIMEOUT 600)
add_mpi_unit_tests(unit-tests-mpi NPROCS 1 2 TIMEOUT 600)
add_mpi_unit_tests(unit-tests-mpi NPROCS 4 TIMEOUT 2400)

if(ENABLE_NCCL AND (Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP))
  set(NCCL_SOURCES Test_Main.cpp nccl/Test_NCCL_Types.cpp)
  add_executable(unit-tests-nccl ${NCCL_SOURCES})
  target_link_libraries(unit-tests-nccl PUBLIC MPI::MPI_CXX GTest::gtest GTest::gmock distributed)
  add_mpi_unit_tests(unit-tests-nccl NPROCS 1 TIMEOUT 600)
endif()

if(ENABLE_NVSHMEM AND Kokkos_ENABLE_CUDA)
  set(NVSHMEM_SOURCES Test_Main.cpp nvshmem/Test_NVSHMEM_Types.cpp)
  add_executable(unit-tests-nvshmem ${NVSHMEM_SOURCES})
  target_link_libraries(unit-tests-nvshmem PUBLIC MPI::MPI_CXX GTest::gtest GTest::gmock distributed)
  add_mpi_unit_tests(unit-tests-nvshmem NPROCS 1 2 4 TIMEOUT 600)
endif()

if(ENABLE_ONECCL AND Kokkos_ENABLE_SYCL)
  set(oneCCL_SOURCES Test_Main.cpp oneCCL/Test_oneCCL_Types.cpp)
  add_executable(unit-tests-oneccl ${oneCCL_SOURCES})
  target_link_libraries(unit-tests-oneccl PUBLIC MPI::MPI_CXX GTest::gtest GTest::gmock distributed)
  add_mpi_unit_tests(unit-tests-oneccl NPROCS 1 TIMEOUT 600)
endif()
