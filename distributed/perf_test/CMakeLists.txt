set(BENCHMARK_NAME PerformanceTest_Benchmark)

check_git_version()

add_executable(${BENCHMARK_NAME}
  BenchmarkMain.cpp
  PerfTest_All2All.cpp
)

target_link_libraries(${BENCHMARK_NAME} PUBLIC benchmark::benchmark distributed)

string(TIMESTAMP BENCHMARK_TIME "%Y-%m-%d_T%H-%M-%S" UTC)
set(BENCHMARK_ARGS --benchmark_counters_tabular=true --benchmark_format=json
    --benchmark_out=${BENCHMARK_NAME}_${BENCHMARK_TIME}.json)

add_test(
  NAME PerformanceTestMPI4
  COMMAND ${MPIEXEC_EXECUTABLE} -n 4 ${BENCHMARK_NAME} ${BENCHMARK_ARGS}
)
