add_executable(rfftn_Z_pencil_v1 main.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)

target_link_libraries(rfftn_Z_pencil_v1 PUBLIC MPI::MPI_CXX Kokkos::kokkos KokkosFFT::fft distributed)
target_compile_features(rfftn_Z_pencil_v1 PUBLIC cxx_std_17)
