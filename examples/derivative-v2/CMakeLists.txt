add_executable(derivative-2D-v2 main.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)

target_link_libraries(derivative-2D-v2 PUBLIC MPI::MPI_CXX Kokkos::kokkos KokkosFFT::fft distributed)
target_compile_features(derivative-2D-v2 PUBLIC cxx_std_17)
