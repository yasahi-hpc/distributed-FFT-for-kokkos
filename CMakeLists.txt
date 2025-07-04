cmake_minimum_required(VERSION 3.22)
project(distributed-fft VERSION 0.0.1 LANGUAGES CXX)

add_subdirectory(tpls/kokkos)
add_subdirectory(tpls/kokkos-fft)
add_subdirectory(tpls/googletest)

option(ENABLE_EXAMPLES "Build examples" OFF)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(distributed)
endif()

if(ENABLE_EXAMPLES)
    add_subdirectory(examples)
endif()
