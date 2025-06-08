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

template <std::size_t DIM = 1>
auto rank_to_coord(const std::array<std::size_t, DIM> &topology,
                   const std::size_t rank) {
  std::array<std::size_t, DIM> coord;
  std::size_t rank_tmp  = rank;
  int64_t topology_size = topology.size() - 1;

  for (int64_t i = topology_size; i >= 0; i--) {
    coord.at(i) = rank_tmp % topology.at(i);
    rank_tmp /= topology.at(i);
  }
  return coord;
}

// Data are stored as
// rank0: extents
// rank1: extents
// ...
// rankn:
template <std::size_t DIM = 1>
auto get_local_shape(const std::array<std::size_t, DIM> &extents,
                     const std::array<std::size_t, DIM> &topology,
                     MPI_Comm comm, bool equal_extents = false) {
  // Check that topology includes two or less non-one elements
  std::array<std::size_t, DIM> local_extents;
  std::copy(extents.begin(), extents.end(), local_extents.begin());
  auto total_size = get_size(topology);

  int rank, nprocs;
  ::MPI_Comm_rank(comm, &rank);
  ::MPI_Comm_size(comm, &nprocs);

  KOKKOSFFT_THROW_IF(total_size != nprocs,
                     "topology size must be identical to mpi size.");

  std::array<std::size_t, DIM> coords =
      rank_to_coord(topology, static_cast<std::size_t>(rank));

  for (std::size_t i = 0; i < extents.size(); i++) {
    if (topology.at(i) != 1) {
      std::size_t n        = extents.at(i);
      std::size_t t        = topology.at(i);
      std::size_t quotient = (n - 1) / t + 1;
      if (equal_extents) {
        local_extents.at(i) = quotient;
      } else {
        std::size_t remainder = n - quotient * (t - 1);
        local_extents.at(i) =
            (coords.at(i) == topology.at(i) - 1) ? remainder : quotient;
      }
    }
  }

  return local_extents;
}

#endif
