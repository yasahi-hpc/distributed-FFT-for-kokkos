#ifndef KOKKOSFFT_DISTRIBUTED_MPI_HELPER_HPP
#define KOKKOSFFT_DISTRIBUTED_MPI_HELPER_HPP

#include <type_traits>
#include <mpi.h>
#include <vector>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_MPI_Types.hpp"
#include "KokkosFFT_Distributed_Types.hpp"
#include "KokkosFFT_Distributed_Utils.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief
/// LayoutRight
/// E.g. Topology (1, 2, 4)
///      rank0: (0, 0)
///      rank1: (0, 1)
///      rank2: (0, 2)
///      rank3: (0, 3)
///      rank4: (1, 0)
///      rank5: (1, 1)
///      rank6: (1, 2)
///      rank7: (1, 3)
/// LayoutLeft
/// E.g. Topology (1, 2, 4)
///      rank0: (0, 0)
///      rank1: (1, 0)
///      rank2: (0, 1)
///      rank3: (1, 1)
///      rank4: (0, 2)
///      rank5: (1, 2)
///      rank6: (0, 3)
///      rank7: (1, 3)
///
template <typename ViewType, typename LayoutType = Kokkos::LayoutRight>
auto get_global_shape(
    const ViewType &v,
    const Topology<std::size_t, ViewType::rank(), LayoutType> &topology,
    MPI_Comm comm) {
  auto extents    = KokkosFFT::Impl::extract_extents(v);
  auto total_size = KokkosFFT::Impl::total_size(topology);

  std::vector<std::size_t> gathered_extents(ViewType::rank() * total_size);
  std::array<std::size_t, ViewType::rank()> global_extents = {};
  MPI_Datatype mpi_data_type = mpi_datatype_v<std::size_t>;

  // Data are stored as
  // rank0: extents
  // rank1: extents
  // ...
  // rankn:
  MPI_Allgather(extents.data(), extents.size(), mpi_data_type,
                gathered_extents.data(), extents.size(), mpi_data_type, comm);

  if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutRight>) {
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
  } else {
    std::size_t stride = 1;
    for (std::size_t i = 0; i < topology.size(); i++) {
      if (topology.at(i) == 1) {
        global_extents.at(i) = extents.at(i);
      } else {
        std::size_t sum = 0;
        for (std::size_t j = 0; j < topology.at(i); j++) {
          sum += gathered_extents.at(i + extents.size() * stride * j);
        }
        stride *= topology.at(i);
        global_extents.at(i) = sum;
      }
    }
  }
  return global_extents;
}

template <typename ViewType>
auto get_global_shape(const ViewType &v,
                      const std::array<std::size_t, ViewType::rank()> &topology,
                      MPI_Comm comm) {
  return get_global_shape(v, Topology<std::size_t, ViewType::rank()>(topology),
                          comm);
}

/// \brief Compute the local extents for the next block given the current rank
/// and layout (compile time version)
// Data are stored as
// rank0: extents
// rank1: extents
// ...
// rankn:
/// \tparam DIM Number of dimensions (default is 1)
/// \tparam LayoutType Layout type for the Input Topology (default is
/// Kokkos::LayoutRight)
/// \param[in] extents Extents of the current block
/// \param[in] topology Topology of the current block
/// \param[in] map Map of the current block
/// \param[in] rank Current rank
/// \return The local extents for the next block
template <std::size_t DIM = 1, typename LayoutType = Kokkos::LayoutRight>
auto compute_next_extents(
    const std::array<std::size_t, DIM> &extents,
    const Topology<std::size_t, DIM, LayoutType> &topology,
    const std::array<std::size_t, DIM> &map, std::size_t rank) {
  std::array<std::size_t, DIM> local_extents, next_extents;
  std::copy(extents.begin(), extents.end(), local_extents.begin());

  auto coords = rank_to_coord(topology, rank);
  for (std::size_t i = 0; i < extents.size(); i++) {
    if (topology.at(i) != 1) {
      std::size_t n = extents.at(i);
      std::size_t t = topology.at(i);

      std::size_t quotient  = n / t;
      std::size_t remainder = n % t;
      // Distribute the remainder acrocss the first few elements
      local_extents.at(i) =
          (coords.at(i) < remainder) ? quotient + 1 : quotient;
    }
  }

  for (std::size_t i = 0; i < extents.size(); i++) {
    std::size_t mapped_idx = map.at(i);
    next_extents.at(i)     = local_extents.at(mapped_idx);
  }

  return next_extents;
}

/// \brief Compute the local extents for the next block given the current rank
/// and layout (run time version)
// Data are stored as
// rank0: extents
// rank1: extents
// ...
// rankn:
/// \tparam DIM Number of dimensions (default is 1)
/// \param[in] extents Extents of the current block
/// \param[in] topology Topology of the current block
/// \param[in] map Map of the current block
/// \param[in] rank Current rank
/// \param[in] is_layout_right Layout type for the Input Topology (default is
/// true)
/// \return The local extents for the next block
template <std::size_t DIM = 1>
auto compute_next_extents(const std::array<std::size_t, DIM> &extents,
                          const std::array<std::size_t, DIM> &topology,
                          const std::array<std::size_t, DIM> &map,
                          std::size_t rank, bool is_layout_right = true) {
  if (is_layout_right) {
    return compute_next_extents(
        extents, Topology<std::size_t, DIM, Kokkos::LayoutRight>(topology), map,
        rank);
  } else {
    return compute_next_extents(
        extents, Topology<std::size_t, DIM, Kokkos::LayoutLeft>(topology), map,
        rank);
  }
}

template <typename ContainerType>
auto get_max(const ContainerType &values, MPI_Comm comm) {
  using value_type = KokkosFFT::Impl::base_container_value_type<ContainerType>;
  MPI_Datatype mpi_data_type = mpi_datatype_v<value_type>;
  value_type max_value       = 0;
  value_type lmax_value      = *std::max_element(values.begin(), values.end());

  MPI_Allreduce(&lmax_value, &max_value, 1, mpi_data_type, MPI_MAX, comm);
  return max_value;
}
}  // namespace Impl

template <std::size_t DIM = 1, typename LayoutType = Kokkos::LayoutRight>
auto rank_to_coord(const Topology<std::size_t, DIM, LayoutType> &topology,
                   const std::size_t rank) {
  std::array<std::size_t, DIM> coord;
  std::size_t rank_tmp  = rank;
  int64_t topology_size = topology.size();

  if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutRight>) {
    for (int64_t i = topology_size - 1; i >= 0; i--) {
      coord.at(i) = rank_tmp % topology.at(i);
      rank_tmp /= topology.at(i);
    }
  } else {
    for (int64_t i = 0; i < topology_size; i++) {
      coord.at(i) = rank_tmp % topology.at(i);
      rank_tmp /= topology.at(i);
    }
  }

  return coord;
}

template <std::size_t DIM = 1>
auto rank_to_coord(const std::array<std::size_t, DIM> &topology,
                   const std::size_t rank) {
  return rank_to_coord(Topology<std::size_t, DIM>(topology), rank);
}

template <std::size_t DIM = 1, typename LayoutType = Kokkos::LayoutRight>
auto compute_local_extents(
    const std::array<std::size_t, DIM> &extents,
    const Topology<std::size_t, DIM, LayoutType> &topology, MPI_Comm comm) {
  // Check that topology includes two or less non-one elements
  std::array<std::size_t, DIM> local_extents = {};
  std::array<std::size_t, DIM> local_starts  = {};
  std::copy(extents.begin(), extents.end(), local_extents.begin());
  auto total_size = KokkosFFT::Impl::total_size(topology);

  int rank, nprocs;
  ::MPI_Comm_rank(comm, &rank);
  ::MPI_Comm_size(comm, &nprocs);

  KOKKOSFFT_THROW_IF(total_size != static_cast<std::size_t>(nprocs),
                     "topology size must be identical to mpi size.");

  std::array<std::size_t, DIM> coords =
      rank_to_coord(topology, static_cast<std::size_t>(rank));

  for (std::size_t i = 0; i < extents.size(); i++) {
    if (topology.at(i) != 1) {
      std::size_t n = extents.at(i);
      std::size_t t = topology.at(i);

      std::size_t quotient  = n / t;
      std::size_t remainder = n % t;

      // Distribute the remainder acrocss the first few elements
      local_extents.at(i) =
          (coords.at(i) < remainder) ? quotient + 1 : quotient;
    }
  }

  std::vector<std::size_t> gathered_extents(DIM * total_size);
  MPI_Datatype mpi_data_type = Impl::mpi_datatype_v<std::size_t>;

  // Data are stored as
  // rank0: extents
  // rank1: extents
  // ...
  // rankn:
  MPI_Allgather(local_extents.data(), local_extents.size(), mpi_data_type,
                gathered_extents.data(), local_extents.size(), mpi_data_type,
                comm);

  std::size_t stride = total_size;
  for (std::size_t i = 0; i < topology.size(); i++) {
    if (topology.at(i) != 1) {
      // Maybe better to check that the shape is something like
      // n, n, n, n_remain
      std::size_t sum = 0;
      stride /= topology.at(i);
      for (std::size_t j = 0; j < coords.at(i); j++) {
        sum += gathered_extents.at(i + extents.size() * stride * j);
      }
      local_starts.at(i) = sum;
    }
  }

  return std::make_tuple(local_extents, local_starts);
}

template <std::size_t DIM = 1>
auto compute_local_extents(const std::array<std::size_t, DIM> &extents,
                           const std::array<std::size_t, DIM> &topology,
                           MPI_Comm comm) {
  return compute_local_extents(extents, Topology<std::size_t, DIM>(topology),
                               comm);
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
  auto total_size = KokkosFFT::Impl::total_size(topology);

  int rank, nprocs;
  ::MPI_Comm_rank(comm, &rank);
  ::MPI_Comm_size(comm, &nprocs);

  KOKKOSFFT_THROW_IF(static_cast<int>(total_size) != nprocs,
                     "topology size must be identical to mpi size.");

  std::array<std::size_t, DIM> coords =
      rank_to_coord(topology, static_cast<std::size_t>(rank));

  for (std::size_t i = 0; i < extents.size(); i++) {
    if (topology.at(i) != 1) {
      std::size_t n = extents.at(i);
      std::size_t t = topology.at(i);

      if (equal_extents) {
        // Distribute data with sufficient extent size
        local_extents.at(i) = (n - 1) / t + 1;
      } else {
        std::size_t quotient  = n / t;
        std::size_t remainder = n % t;

        // Distribute the remainder acrocss the first few elements
        local_extents.at(i) =
            (coords.at(i) < remainder) ? quotient + 1 : quotient;
      }
    }
  }

  return local_extents;
}

}  // namespace Distributed
}  // namespace KokkosFFT

#endif
