add_executable(fftn-Xslab main.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)

target_link_libraries(fftn-Xslab PUBLIC MPI::MPI_CXX Kokkos::kokkos KokkosFFT::fft)
target_compile_features(fftn-Xslab PUBLIC cxx_std_17)
