add_executable(rfftn_X_pencil_v2 main.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)

target_link_libraries(rfftn_X_pencil_v2 PUBLIC MPI::MPI_CXX Kokkos::kokkos KokkosFFT::fft distributed)
target_compile_features(rfftn_X_pencil_v2 PUBLIC cxx_std_17)
