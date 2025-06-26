add_executable(rfftn_Y_pencil_v2 main.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)

target_link_libraries(rfftn_Y_pencil_v2 PUBLIC MPI::MPI_CXX Kokkos::kokkos KokkosFFT::fft distributed)
target_compile_features(rfftn_Y_pencil_v2 PUBLIC cxx_std_17)
