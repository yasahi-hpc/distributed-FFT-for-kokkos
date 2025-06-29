add_executable(derivative_X_slab main.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)

target_link_libraries(derivative_X_slab PUBLIC MPI::MPI_CXX Kokkos::kokkos KokkosFFT::fft distributed)
target_compile_features(derivative_X_slab PUBLIC cxx_std_17)
