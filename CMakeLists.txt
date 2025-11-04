cmake_minimum_required(VERSION 3.22)
project(distributed-fft VERSION 0.0.1 LANGUAGES CXX)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Version information
set(DISTRIBUTED_FFT_VERSION ${PROJECT_VERSION})
set(DISTRIBUTED_FFT_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(DISTRIBUTED_FFT_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(DISTRIBUTED_FFT_VERSION_PATCH ${PROJECT_VERSION_PATCH})

set(KOKKOS_REQUIRED_VERSION 4.5.0)
set(KOKKOSFFT_REQUIRED_VERSION 0.4.0)

option(ENABLE_TESTS "Build tests" OFF)
option(ENABLE_BENCHMARK "Build benchmarks" OFF)
option(ENABLE_EXAMPLES "Build examples" OFF)
option(ENABLE_NCCL "Enable NCCL as a communication backend on NVIDIA GPUs" OFF)
option(ENABLE_RCCL "Enable RCCL as a communication backend on AMD GPUs" OFF)
option(ENABLE_CUFFT_MP "Enable cufft-mp as a KokkosFFT backend on NVIDIA GPUs" OFF)
option(ENABLE_ROCFFT_MPI "Enable rocfft-mpi as a KokkosFFT backend on AMD GPUs" OFF)
option(ENABLE_INTERNAL_KOKKOS "Build internal Kokkos instead of relying on external one" OFF)
option(ENABLE_INTERNAL_KOKKOSFFT "Build internal Kokkos-fft instead of relying on external one" OFF)

# Check the device tpls usage
set(ENABLED_DEVICE_COMM_TPLS "")
set(KOKKOSFFT_DEVICE_COMM_TPL_LIST "NCCL" "RCCL")
foreach(DEVICE_COMM_TPL IN LISTS KOKKOSFFT_DEVICE_COMM_TPL_LIST)
  if(ENABLE_${DEVICE_COMM_TPL})
    list(APPEND ENABLED_DEVICE_COMM_TPLS ${DEVICE_COMM_TPL})
  endif()
endforeach()

# Get the number of enabled tpls
list(LENGTH ENABLED_DEVICE_COMM_TPLS COUNT)
if(COUNT GREATER 1)
  message(FATAL_ERROR "Error: Only one device tpls may be enabled. The following options are enabled: ${ENABLED_DEVICE_COMM_TPLS}")
endif()

# Set the global variables
if(ENABLED_DEVICE_COMM_TPLS)
  foreach(DEVICE_COMM_TPL IN LISTS ENABLED_DEVICE_COMM_TPLS)
    set(KOKKOSFFT_DISTRIBUTED_ENABLE_TPL_${DEVICE_COMM_TPL} TRUE)
  endforeach()
endif()

if (NOT ENABLE_INTERNAL_KOKKOS)
    # First check, Kokkos is added as subdirectory or not
    if(NOT TARGET Kokkos::kokkos)
        find_package(Kokkos ${KOKKOS_REQUIRED_VERSION} REQUIRED)
    endif()
else ()
    add_subdirectory(tpls/kokkos)
endif ()

if (NOT ENABLE_INTERNAL_KOKKOSFFT)
    # First check, Kokkos is added as subdirectory or not
    if(NOT TARGET KokkosFFT::fft)
        find_package(KokkosFFT ${KOKKOSFFT_REQUIRED_VERSION} REQUIRED)
    endif()
else ()
    add_subdirectory(tpls/kokkos-fft)
endif ()

# ==================================================================
# CMake Summary
# ==================================================================

message("")
message(STATUS "distributed-FFT version: ${DISTRIBUTED_FFT_VERSION_MAJOR}.${DISTRIBUTED_FFT_VERSION_MINOR}.${DISTRIBUTED_FFT_VERSION_PATCH}")

# googletest
if(ENABLE_TESTS)
  include(CTest)
  find_package(GTest CONFIG)
  if(NOT GTest_FOUND)
    add_subdirectory(tpls/googletest)
  endif()
endif()

# Google benchmark
if(ENABLE_BENCHMARK)
  option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library." OFF)
  find_package(benchmark CONFIG)
  if(NOT benchmark_FOUND)
    add_subdirectory(tpls/benchmark)
  endif()
  include(KokkosFFT_Git_Hash)
endif()

# Configure files to display configuration
# Configure the library
set(PACKAGE_NAME_CONFIG_FILES KokkosFFT_Distributed_config.hpp KokkosFFT_Distributed_Version_Info.hpp)

foreach(CONFIG_FILE ${PACKAGE_NAME_CONFIG_FILES})
  configure_file(cmake/${CONFIG_FILE}.in
                 ${CMAKE_CURRENT_BINARY_DIR}/distributed/src/${CONFIG_FILE}
                 @ONLY
  )
endforeach()

add_subdirectory(distributed)

if(ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()
