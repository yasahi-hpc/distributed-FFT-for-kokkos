add_executable(fftn_X_pencil main.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)

target_link_libraries(fftn_X_pencil PUBLIC MPI::MPI_CXX Kokkos::kokkos KokkosFFT::fft)
target_compile_features(fftn_X_pencil PUBLIC cxx_std_17)
