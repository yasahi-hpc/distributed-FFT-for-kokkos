add_library(distributed INTERFACE)

find_package(MPI REQUIRED COMPONENTS CXX)

target_link_libraries(distributed
    INTERFACE
    Kokkos::kokkos
    MPI::MPI_CXX
)

if(ENABLE_NCCL AND Kokkos_ENABLE_CUDA)
  find_package(NCCL REQUIRED)
  target_compile_definitions(distributed INTERFACE ENABLE_TPL_NCCL)
  target_link_libraries(distributed INTERFACE NCCL::NCCL)
endif()

if(ENABLE_RCCL AND Kokkos_ENABLE_HIP)
  find_package(RCCL REQUIRED)
  target_compile_definitions(distributed INTERFACE ENABLE_TPL_NCCL)
  target_link_libraries(distributed INTERFACE roc::rccl)
endif()

if(ENABLE_ONECCL AND Kokkos_ENABLE_SYCL)
  find_package(oneCCL REQUIRED)
  target_compile_definitions(distributed INTERFACE ENABLE_TPL_ONECCL)
  target_link_libraries(distributed INTERFACE oneCCL::oneCCL)
endif()

if(ENABLE_CUFFT_MP AND Kokkos_ENABLE_CUDA)
  find_package(cuFFTMp REQUIRED)
  target_include_directories(distributed INTERFACE ${cuFFTMp_INCLUDE_DIRS})
  # We add cuFFTMp to the link libraries before and after KokkosFFT to ensure that cuFFTMp will
  # appear before cuFFT on the linker command line
  target_link_libraries(distributed INTERFACE ${cuFFTMp_LIBRARIES})
  target_link_libraries(distributed INTERFACE KokkosFFT::fft)
  target_link_libraries(distributed INTERFACE ${cuFFTMp_LIBRARIES})
  target_compile_definitions(distributed INTERFACE ENABLE_TPL_CUFFT_MP PRIORITIZE_TPL_PLAN_IF_AVAILABLE)
else()
  target_link_libraries(distributed INTERFACE KokkosFFT::fft)
endif()

if(ENABLE_ROCFFT_MPI AND Kokkos_ENABLE_HIP AND KokkosFFT_ENABLE_ROCFFT)
  target_compile_definitions(distributed INTERFACE ENABLE_TPL_ROCFFT_MPI PRIORITIZE_TPL_PLAN_IF_AVAILABLE)
endif()

target_include_directories(distributed INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                           $<INSTALL_INTERFACE:${INSTALL_INCLUDEDIR}>
)
