add_executable(derivative-2D main.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)

target_link_libraries(derivative-2D PUBLIC MPI::MPI_CXX Kokkos::kokkos KokkosFFT::fft)
target_compile_features(derivative-2D PUBLIC cxx_std_17)
