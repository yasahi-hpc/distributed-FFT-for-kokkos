add_executable(inplace_FFT main.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)

target_link_libraries(inplace_FFT PUBLIC MPI::MPI_CXX Kokkos::kokkos distributed)
target_compile_features(inplace_FFT PUBLIC cxx_std_17)
