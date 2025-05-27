add_executable(gysela_like main.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)

target_link_libraries(gysela_like PUBLIC MPI::MPI_CXX Kokkos::kokkos KokkosFFT::fft)
target_compile_features(gysela_like PUBLIC cxx_std_17)
