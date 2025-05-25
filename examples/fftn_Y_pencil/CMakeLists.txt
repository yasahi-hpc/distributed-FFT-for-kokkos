add_executable(fftn_Y_pencil main.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)

target_link_libraries(fftn_Y_pencil PUBLIC MPI::MPI_CXX Kokkos::kokkos KokkosFFT::fft)
target_compile_features(fftn_Y_pencil PUBLIC cxx_std_17)
