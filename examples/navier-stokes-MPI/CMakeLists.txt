add_executable(navier-stokes-MPI main.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)

target_link_libraries(navier-stokes-MPI PUBLIC MPI::MPI_CXX Kokkos::kokkos distributed)
target_compile_features(navier-stokes-MPI PUBLIC cxx_std_17)
