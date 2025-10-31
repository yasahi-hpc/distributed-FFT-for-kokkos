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
option(ENABLE_EXAMPLES "Build examples" OFF)
option(ENABLE_NCCL "Enable NCCL as a communication backend on NVIDIA GPUs" OFF)
option(ENABLE_CUFFT_MP "Enable cufft-mp as a KokkosFFT backend on NVIDIA GPUs" OFF)
option(ENABLE_ROCFFT_MPI "Enable rocfft-mpi as a KokkosFFT backend on AMD GPUs" OFF)
option(ENABLE_INTERNAL_KOKKOS "Build internal Kokkos instead of relying on external one" OFF)
option(ENABLE_INTERNAL_KOKKOSFFT "Build internal Kokkos-fft instead of relying on external one" OFF)


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

add_subdirectory(distributed)

if(ENABLE_EXAMPLES)
    add_subdirectory(examples)
endif()
