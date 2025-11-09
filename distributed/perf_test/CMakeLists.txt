set(MPI_BENCHMARK_NAME MPI_Perftest)

check_git_version()

add_executable(${MPI_BENCHMARK_NAME}
  BenchmarkMPIMain.cpp
  PerfTest_All2All.cpp
  PerfTest_FFT3D.cpp
)

target_link_libraries(${MPI_BENCHMARK_NAME} PUBLIC benchmark::benchmark distributed)

string(TIMESTAMP BENCHMARK_TIME "%Y-%m-%d_T%H-%M-%S" UTC)
set(MPI_BENCHMARK_ARGS --benchmark_counters_tabular=true --benchmark_format=json
    --benchmark_out=${MPI_BENCHMARK_NAME}_${BENCHMARK_TIME}.json)

add_test(
  NAME PerformanceTestMPI4
  COMMAND ${MPIEXEC_EXECUTABLE} -n 4 ${MPI_BENCHMARK_NAME} ${MPI_BENCHMARK_ARGS}
)

set(BENCHMARK_NAME Perftest)

add_executable(${BENCHMARK_NAME}
  BenchmarkMain.cpp
  PerfTest_Pack.cpp
  PerfTest_Unpack.cpp
  PerfTest_Transpose.cpp
)

target_link_libraries(${BENCHMARK_NAME} PUBLIC benchmark::benchmark distributed)

string(TIMESTAMP BENCHMARK_TIME "%Y-%m-%d_T%H-%M-%S" UTC)
set(BENCHMARK_ARGS --benchmark_counters_tabular=true --benchmark_format=json
    --benchmark_out=${BENCHMARK_NAME}_${BENCHMARK_TIME}.json)

add_test(NAME PerformanceTest COMMAND ${BENCHMARK_NAME} ${BENCHMARK_ARGS})
