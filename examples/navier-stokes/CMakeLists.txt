add_executable(navier-stokes main.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)

target_link_libraries(navier-stokes PUBLIC MPI::MPI_CXX Kokkos::kokkos KokkosFFT::fft distributed)
target_compile_features(navier-stokes PUBLIC cxx_std_17)
