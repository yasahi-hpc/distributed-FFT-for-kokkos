add_executable(rfftn_X_pencil main.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)

target_link_libraries(rfftn_X_pencil PUBLIC MPI::MPI_CXX Kokkos::kokkos KokkosFFT::fft)
target_compile_features(rfftn_X_pencil PUBLIC cxx_std_17)
