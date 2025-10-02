#ifndef KOKKOSFFT_DISTRIBUTED_TYPES_HPP
#define KOKKOSFFT_DISTRIBUTED_TYPES_HPP

#include <array>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>
#include <Kokkos_Core.hpp>

enum class TopologyType {
  Pencil,
  Slab,
  Shared,
  Invalid,
};

enum class BlockType {
  Transpose,
  FFT,
};

/// \brief BlockInfo is a structure that holds information about a block
/// of data in a distributed FFT operation.
/// It contains the input and output topologies, extents, buffer extents,
/// maps, axes, and the type of Transpose and FFT block.
/// For Transpose blocks, m_axes is unused.
/// For FFT blocks, m_in_topology, m_out_topology, m_buffer_extents, m_in_map,
/// m_in_axis , m_out_map and m_out_axis are unused.
///
/// \tparam DIM The dimensionality of the input and output data.
///
template <std::size_t DIM>
struct BlockInfo {
  using map_type            = std::array<std::size_t, DIM>;
  using extents_type        = std::array<std::size_t, DIM>;
  using axes_type           = std::vector<std::size_t>;
  using buffer_extents_type = std::array<std::size_t, DIM + 1>;

  //! The input topology
  extents_type m_in_topology = {};

  //! The output topology
  extents_type m_out_topology = {};

  //! The extents of Input View
  extents_type m_in_extents = {};

  //! The extents of Output View
  extents_type m_out_extents = {};

  //! The extents of the buffer used for Transpose
  buffer_extents_type m_buffer_extents = {};

  //! The permutation map for the input View
  map_type m_in_map = {};

  //! The permutation map for the output View
  map_type m_out_map = {};

  //! The axis along which the input View is split
  std::size_t m_in_axis = 0;

  //! The axis along which the output View is merged
  std::size_t m_out_axis = 0;

  //! The axes along which the FFT is performed
  axes_type m_axes;

  //! The axis along which the MPI all2all is performed
  std::size_t m_comm_axis = 0;

  //! The type of the block (Transpose, FFT, FFT2, TransposeAndFFT)
  BlockType m_block_type;

  //! The idx of the transpose or FFT block in the plan
  std::size_t m_block_idx = 0;

  bool operator==(const BlockInfo& other) const {
    return m_in_topology == other.m_in_topology &&
           m_out_topology == other.m_out_topology &&
           m_in_extents == other.m_in_extents &&
           m_out_extents == other.m_out_extents &&
           m_buffer_extents == other.m_buffer_extents &&
           m_in_map == other.m_in_map && m_out_map == other.m_out_map &&
           m_in_axis == other.m_in_axis && m_out_axis == other.m_out_axis &&
           m_axes == other.m_axes && m_comm_axis == other.m_comm_axis &&
           m_block_type == other.m_block_type &&
           m_block_idx == other.m_block_idx;
  }

  bool operator!=(const BlockInfo& other) const { return !(*this == other); }
};

template <typename T, std::size_t N, typename LayoutType = Kokkos::LayoutRight>
class Topology {
 private:
  std::array<T, N> m_data;

 public:
  // Default constructor
  constexpr Topology() = default;

  // Constructor from std::array
  constexpr Topology(const std::array<T, N>& arr) : m_data{arr} {}

  // Constructor from initializer list
  constexpr Topology(std::initializer_list<T> init) {
    if (init.size() != N) {
      throw std::length_error("Initializer list size must match array size");
    }
    std::copy(init.begin(), init.end(), m_data.begin());
  }

  // Copy constructor
  constexpr Topology(const Topology& other) = default;

  // Move constructor
  constexpr Topology(Topology&& other) noexcept = default;

  // Copy assignment
  constexpr Topology& operator=(const Topology& other) = default;

  // Move assignment
  constexpr Topology& operator=(Topology&& other) noexcept = default;

  // Comparison operators
  constexpr bool operator==(const Topology& other) const noexcept {
    return m_data == other.m_data;
  }

  constexpr bool operator!=(const Topology& other) const noexcept {
    return !(*this == other);
  }

  // Access element with bounds checking
  constexpr T& at(std::size_t pos) { return m_data.at(pos); }

  constexpr const T& at(std::size_t pos) const { return m_data.at(pos); }

  // Access element without bounds checking
  constexpr T& operator[](std::size_t pos) { return m_data[pos]; }

  constexpr const T& operator[](std::size_t pos) const { return m_data[pos]; }

  // Get the size of the array
  constexpr std::size_t size() const noexcept { return N; }

  // Get underlying data
  constexpr const std::array<T, N>& array() const noexcept { return m_data; }

  constexpr std::array<T, N>& array() noexcept { return m_data; }

  // Iterators
  constexpr auto begin() noexcept { return m_data.begin(); }
  constexpr auto end() noexcept { return m_data.end(); }
  constexpr auto begin() const noexcept { return m_data.begin(); }
  constexpr auto end() const noexcept { return m_data.end(); }
  constexpr auto cbegin() const noexcept { return m_data.cbegin(); }
  constexpr auto cend() const noexcept { return m_data.cend(); }
  constexpr auto rbegin() noexcept { return m_data.rbegin(); }
  constexpr auto rend() noexcept { return m_data.rend(); }
  constexpr auto rbegin() const noexcept { return m_data.rbegin(); }
  constexpr auto rend() const noexcept { return m_data.rend(); }
  constexpr auto crbegin() const noexcept { return m_data.crbegin(); }
  constexpr auto crend() const noexcept { return m_data.crend(); }
};

#endif
