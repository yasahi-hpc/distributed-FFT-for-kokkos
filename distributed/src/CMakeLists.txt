add_library(distributed INTERFACE)

find_package(MPI REQUIRED COMPONENTS CXX)

target_link_libraries(distributed
    INTERFACE
    Kokkos::kokkos
    MPI::MPI_CXX
)

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
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                           $<INSTALL_INTERFACE:${INSTALL_INCLUDEDIR}>
)
