add_executable(derivative_X_slab_v2 main.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)

target_link_libraries(derivative_X_slab_v2 PUBLIC MPI::MPI_CXX Kokkos::kokkos KokkosFFT::fft distributed)
target_compile_features(derivative_X_slab_v2 PUBLIC cxx_std_17)
