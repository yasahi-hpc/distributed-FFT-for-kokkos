#ifndef MPI_HELPER_HPP
#define MPI_HELPER_HPP

#include <type_traits>
#include <mpi.h>
#include <vector>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>

template <typename ValueType>
struct MPIDataType {};

template <>
struct MPIDataType<int> {
  static constexpr MPI_Datatype type() noexcept { return MPI_INT32_T; }
};

template <>
struct MPIDataType<std::size_t> {
  static constexpr MPI_Datatype type() noexcept { return MPI_UINT64_T; }
};

template <>
struct MPIDataType<float> {
  static constexpr MPI_Datatype type() noexcept { return MPI_FLOAT; }
};

template <>
struct MPIDataType<double> {
  static constexpr MPI_Datatype type() noexcept { return MPI_DOUBLE; }
};

template <>
struct MPIDataType<Kokkos::complex<float>> {
  static constexpr MPI_Datatype type() noexcept {
    return MPI_CXX_FLOAT_COMPLEX;
  }
};

template <>
struct MPIDataType<Kokkos::complex<double>> {
  static constexpr MPI_Datatype type() noexcept {
    return MPI_CXX_DOUBLE_COMPLEX;
  }
};

template <std::size_t DIM>
std::size_t get_size(const std::array<std::size_t, DIM> &topology) {
  return std::accumulate(topology.begin(), topology.end(), 1,
                         std::multiplies<std::size_t>());
}

/// \brief
///
/// E.g. Topology (1, 2, 4)
///      rank0: (0, 0)
///      rank1: (0, 1)
///      rank2: (0, 2)
///      rank3: (0, 3)
///      rank4: (1, 0)
///      rank5: (1, 1)
///      rank6: (1, 2)
///      rank7: (1, 3)
///
template <typename ViewType>
auto get_global_shape(const ViewType &v,
                      const std::array<std::size_t, ViewType::rank()> &topology,
                      MPI_Comm comm) {
  auto extents    = KokkosFFT::Impl::extract_extents(v);
  auto total_size = get_size(topology);

  std::vector<std::size_t> gathered_extents(ViewType::rank() * total_size);
  std::array<std::size_t, ViewType::rank()> global_extents = {};
  MPI_Datatype mpi_data_type = MPIDataType<std::size_t>::type();

  // Data are stored as
  // rank0: extents
  // rank1: extents
  // ...
  // rankn:
  MPI_Allgather(extents.data(), extents.size(), mpi_data_type,
                gathered_extents.data(), extents.size(), mpi_data_type, comm);

  std::size_t stride = total_size;
  for (std::size_t i = 0; i < topology.size(); i++) {
    if (topology.at(i) == 1) {
      global_extents.at(i) = extents.at(i);
    } else {
      // Maybe better to check that the shape is something like
      // n, n, n, n_remain
      std::size_t sum = 0;
      stride /= topology.at(i);
      for (std::size_t j = 0; j < topology.at(i); j++) {
        sum += gathered_extents.at(i + extents.size() * stride * j);
      }
      global_extents.at(i) = sum;
    }
  }
  return global_extents;
}

#endif
