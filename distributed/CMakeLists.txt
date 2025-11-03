add_subdirectory(src)

if(ENABLE_TESTS)
  add_subdirectory(unit_test)
endif()

if(ENABLE_BENCHMARK)
  add_subdirectory(perf_test)
endif()
