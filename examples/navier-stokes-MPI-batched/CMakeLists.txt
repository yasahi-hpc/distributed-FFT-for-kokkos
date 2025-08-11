add_executable(navier-stokes-MPI-batched main.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)

target_link_libraries(navier-stokes-MPI-batched PUBLIC MPI::MPI_CXX Kokkos::kokkos distributed)
target_compile_features(navier-stokes-MPI-batched PUBLIC cxx_std_17)
