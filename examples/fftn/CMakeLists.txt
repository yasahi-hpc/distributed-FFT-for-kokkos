add_executable(pencil-3D main.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)

target_link_libraries(pencil-3D PUBLIC MPI::MPI_CXX Kokkos::kokkos KokkosFFT::fft)
target_compile_features(pencil-3D PUBLIC cxx_std_17)
