add_executable(derivative_Y_slab main.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)

target_link_libraries(derivative_Y_slab PUBLIC MPI::MPI_CXX Kokkos::kokkos KokkosFFT::fft distributed)
target_compile_features(derivative_Y_slab PUBLIC cxx_std_17)
