cmake_minimum_required(VERSION 3.22)
project(distributed-fft VERSION 0.0.1 LANGUAGES CXX)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Version information
set(DISTRIBUTED_FFT_VERSION ${PROJECT_VERSION})
set(DISTRIBUTED_FFT_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(DISTRIBUTED_FFT_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(DISTRIBUTED_FFT_VERSION_PATCH ${PROJECT_VERSION_PATCH})

add_subdirectory(tpls/kokkos)
add_subdirectory(tpls/kokkos-fft)

option(ENABLE_TESTS "Build tests" OFF)
option(ENABLE_EXAMPLES "Build examples" OFF)
option(ENABLE_CUFFT_MP "Enable cufft-mp as a KokkosFFT backend on NVIDIA GPUs" OFF)

# googletest
if(ENABLE_TESTS)
  include(CTest)
  find_package(GTest CONFIG)
  if(NOT GTest_FOUND)
    add_subdirectory(tpls/googletest)
  endif()
endif()

add_subdirectory(distributed)

if(ENABLE_EXAMPLES)
    add_subdirectory(examples)
endif()
