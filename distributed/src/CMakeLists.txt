add_library(distributed INTERFACE)

find_package(MPI REQUIRED COMPONENTS CXX)

target_link_libraries(distributed
    INTERFACE
    Kokkos::kokkos
    KokkosFFT::fft
    MPI::MPI_CXX
)

if(ENABLE_CUFFT_MP AND Kokkos_ENABLE_CUDA)
  find_package(cuFFTMp REQUIRED)
  target_include_directories(distributed INTERFACE ${cuFFTMp_INCLUDE_DIRS})
  target_link_libraries(distributed INTERFACE ${cuFFTMp_LIBRARIES})
  target_compile_definitions(distributed INTERFACE ENABLE_TPL_CUFFT_MP PRIOTIZE_TPL_PLAN_IF_AVAILABLE)
endif()

target_include_directories(distributed INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                           $<INSTALL_INTERFACE:${INSTALL_INCLUDEDIR}>
)
