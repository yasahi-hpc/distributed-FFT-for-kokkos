#ifndef EXTENTS_HPP
#define EXTENTS_HPP

#include <algorithm>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "Mapping.hpp"

template <typename iType, std::size_t DIM = 1>
auto merge_topology(const std::array<iType, DIM> &in_topology,
                    const std::array<iType, DIM> &out_topology) {
  auto in_size  = get_size(in_topology);
  auto out_size = get_size(out_topology);

  KOKKOSFFT_THROW_IF(in_size != out_size,
                     "Input and output topologies must have the same size.");

  // Check if two topologies are two convertible pencils
  std::vector<iType> diff_indices = find_differences(in_topology, out_topology);
  KOKKOSFFT_THROW_IF(
      diff_indices.size() != 2,
      "Input and output topologies must differ exactly two positions.");

  std::array<iType, DIM> topology = {};
  for (std::size_t i = 0; i < in_topology.size(); i++) {
    topology.at(i) = std::max(in_topology.at(i), out_topology.at(i));
  }
  return topology;
}

template <typename iType, std::size_t DIM = 1>
auto diff_toplogy(const std::array<iType, DIM> &in_topology,
                  const std::array<iType, DIM> &out_topology) {
  std::vector<iType> diff_indices = find_differences(in_topology, out_topology);
  KOKKOSFFT_THROW_IF(
      diff_indices.size() != 1,
      "Input and output topologies must differ exactly one positions.");
  iType diff_idx = diff_indices.at(0);

  return std::max(in_topology.at(diff_idx), out_topology.at(diff_idx));
}

// Example
// Global View extents (n0, n1, n2, n3)
// in-topology = {1, p0, p1, 1} // X-pencil
// out-topology = {p0, 1, p1, 1} // Y-pencil
// Buffer View (p0, n0/p0, n1/p0, n2/p1, n3)
template <typename LayoutType, std::size_t DIM = 1>
auto get_buffer_extents(const std::array<std::size_t, DIM> &extents,
                        const std::array<std::size_t, DIM> &in_topology,
                        const std::array<std::size_t, DIM> &out_topology) {
  std::array<std::size_t, DIM + 1> buffer_extents;
  auto merged_topology = merge_topology(in_topology, out_topology);
  auto p0 = diff_toplogy(merged_topology, in_topology);  // return 1 or p0
  if (std::is_same_v<LayoutType, Kokkos::LayoutRight>) {
    buffer_extents.at(0) = p0;
    for (std::size_t i = 0; i < extents.size(); i++) {
      buffer_extents.at(i + 1) = extents.at(i) / merged_topology.at(i);
    }
  } else {
    for (std::size_t i = 0; i < extents.size(); i++) {
      buffer_extents.at(i) = extents.at(i) / merged_topology.at(i);
    }
    buffer_extents.back() = p0;
  }
  return buffer_extents;
}

// Example
// Global View extents (n0, n1, n2, n3)
// topology = {p0, 1, p1, 1} // Y-pencil
// map (0, 2, 3, 1)
template <std::size_t DIM = 1>
auto get_next_extents(const std::array<std::size_t, DIM> &extents,
                      const std::array<std::size_t, DIM> &topology,
                      const std::array<std::size_t, DIM> &map) {
  std::array<std::size_t, DIM> next_extents;

  for (std::size_t i = 0; i < extents.size(); i++) {
    std::size_t mapped_idx = map.at(i);
    next_extents.at(i)     = extents.at(mapped_idx) / topology.at(mapped_idx);
  }

  return next_extents;
}

#endif
