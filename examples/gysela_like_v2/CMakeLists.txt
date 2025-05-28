add_executable(gysela_like_v2 main.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)

target_link_libraries(gysela_like_v2 PUBLIC MPI::MPI_CXX Kokkos::kokkos KokkosFFT::fft distributed)
target_compile_features(gysela_like_v2 PUBLIC cxx_std_17)
