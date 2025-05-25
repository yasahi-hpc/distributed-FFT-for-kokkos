add_executable(rfftn_Z_pencil main.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)

target_link_libraries(rfftn_Z_pencil PUBLIC MPI::MPI_CXX Kokkos::kokkos KokkosFFT::fft)
target_compile_features(rfftn_Z_pencil PUBLIC cxx_std_17)
