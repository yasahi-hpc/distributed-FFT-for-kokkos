#ifndef KOKKOSFFT_DISTRIBUTED_TYPES_HPP
#define KOKKOSFFT_DISTRIBUTED_TYPES_HPP

#include <array>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>
#include <Kokkos_Core.hpp>

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

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

}  // namespace Impl

/// \brief A container class that wraps std::array to provide topology
/// information for distributed FFT operations with additional
/// std::array-compatible interface.
///
/// This class provides a complete std::array-compatible interface while adding
/// layout type information for Kokkos integration. It maintains the same
/// performance characteristics as std::array with compile-time size checking.
///
/// \tparam T The element type stored in the topology
/// \tparam N The number of elements in the topology (compile-time constant)
/// \tparam LayoutType The Kokkos layout type (default: Kokkos::LayoutRight)
template <typename T, std::size_t N, typename LayoutType = Kokkos::LayoutRight>
class Topology {
 public:
  // Type definitions (std::array compatibility)
  using value_type       = T;
  using size_type        = std::size_t;
  using difference_type  = std::ptrdiff_t;
  using reference        = T&;
  using const_reference  = const T&;
  using pointer          = T*;
  using const_pointer    = const T*;
  using iterator         = typename std::array<T, N>::iterator;
  using const_iterator   = typename std::array<T, N>::const_iterator;
  using reverse_iterator = typename std::array<T, N>::reverse_iterator;
  using const_reverse_iterator =
      typename std::array<T, N>::const_reverse_iterator;
  using layout_type = LayoutType;

 private:
  std::array<T, N> m_data;

 public:
  // Constructors and assignment operators

  /// \brief Default constructor - creates an uninitialized topology
  constexpr Topology() = default;

  /// \brief Constructor from std::array
  /// \param[in] arr The std::array to copy from
  constexpr Topology(const std::array<T, N>& arr) : m_data{arr} {}

  /// \brief Constructor from initializer list
  /// \param[in] init The initializer list containing exactly N elements
  /// \throws std::length_error if initializer list size != N
  constexpr Topology(std::initializer_list<T> init) {
    if (init.size() != N) {
      throw std::length_error("Initializer list size must match array size");
    }
    auto it = init.begin();
    for (std::size_t i = 0; i < N; ++i) {
      m_data[i] = *it++;
    }
  }

  /// \brief Copy constructor
  /// \param[in] other The topology to copy from
  constexpr Topology(const Topology& other) = default;

  /// \brief Move constructor
  /// \param[in, out] other The topology to move from
  constexpr Topology(Topology&& other) noexcept = default;

  /// \brief Copy assignment operator
  /// \param[in] other The topology to copy from
  /// \return Reference to this topology
  constexpr Topology& operator=(const Topology& other) = default;

  /// \brief Move assignment operator
  /// \param[in, out] other The topology to move from
  /// \return Reference to this topology
  constexpr Topology& operator=(Topology&& other) noexcept = default;

  // Element access

  /// \brief Access element with bounds checking
  /// \param[in] pos The position of the element to access
  /// \return Reference to the element at position pos
  /// \throws std::out_of_range if pos >= N
  constexpr reference at(size_type pos) { return m_data.at(pos); }

  /// \brief Access element with bounds checking (const version)
  /// \param[in] pos The position of the element to access
  /// \return Const reference to the element at position pos
  /// \throws std::out_of_range if pos >= N
  constexpr const_reference at(size_type pos) const { return m_data.at(pos); }

  /// \brief Access element without bounds checking
  /// \param[in] pos The position of the element to access
  /// \return Reference to the element at position pos
  constexpr reference operator[](size_type pos) { return m_data[pos]; }

  /// \brief Access element without bounds checking (const version)
  /// \param[in] pos The position of the element to access
  /// \return Const reference to the element at position pos
  constexpr const_reference operator[](size_type pos) const {
    return m_data[pos];
  }

  /// \brief Access the first element
  /// \return Reference to the first element
  /// \note Undefined behavior if empty() returns true
  constexpr reference front() { return m_data.front(); }

  /// \brief Access the first element (const version)
  /// \return Const reference to the first element
  /// \note Undefined behavior if empty() returns true
  constexpr const_reference front() const { return m_data.front(); }

  /// \brief Access the last element
  /// \return Reference to the last element
  /// \note Undefined behavior if empty() returns true
  constexpr reference back() { return m_data.back(); }

  /// \brief Access the last element (const version)
  /// \return Const reference to the last element
  /// \note Undefined behavior if empty() returns true
  constexpr const_reference back() const { return m_data.back(); }

  /// \brief Get direct access to the underlying array
  /// \return Pointer to the underlying element storage
  constexpr pointer data() noexcept { return m_data.data(); }

  /// \brief Get direct access to the underlying array (const version)
  /// \return Const pointer to the underlying element storage
  constexpr const_pointer data() const noexcept { return m_data.data(); }

  // Iterators

  /// \brief Get iterator to the beginning
  /// \return Iterator to the first element
  constexpr iterator begin() noexcept { return m_data.begin(); }

  /// \brief Get iterator to the end
  /// \return Iterator to one past the last element
  constexpr iterator end() noexcept { return m_data.end(); }

  /// \brief Get const iterator to the beginning
  /// \return Const iterator to the first element
  constexpr const_iterator begin() const noexcept { return m_data.begin(); }

  /// \brief Get const iterator to the end
  /// \return Const iterator to one past the last element
  constexpr const_iterator end() const noexcept { return m_data.end(); }

  /// \brief Get const iterator to the beginning
  /// \return Const iterator to the first element
  constexpr const_iterator cbegin() const noexcept { return m_data.cbegin(); }

  /// \brief Get const iterator to the end
  /// \return Const iterator to one past the last element
  constexpr const_iterator cend() const noexcept { return m_data.cend(); }

  /// \brief Get reverse iterator to the reverse beginning
  /// \return Reverse iterator to the last element
  constexpr reverse_iterator rbegin() noexcept { return m_data.rbegin(); }

  /// \brief Get reverse iterator to the reverse end
  /// \return Reverse iterator to one before the first element
  constexpr reverse_iterator rend() noexcept { return m_data.rend(); }

  /// \brief Get const reverse iterator to the reverse beginning
  /// \return Const reverse iterator to the last element
  constexpr const_reverse_iterator rbegin() const noexcept {
    return m_data.rbegin();
  }

  /// \brief Get const reverse iterator to the reverse end
  /// \return Const reverse iterator to one before the first element
  constexpr const_reverse_iterator rend() const noexcept {
    return m_data.rend();
  }

  /// \brief Get const reverse iterator to the reverse beginning
  /// \return Const reverse iterator to the last element
  constexpr const_reverse_iterator crbegin() const noexcept {
    return m_data.crbegin();
  }

  /// \brief Get const reverse iterator to the reverse end
  /// \return Const reverse iterator to one before the first element
  constexpr const_reverse_iterator crend() const noexcept {
    return m_data.crend();
  }

  // Capacity

  /// \brief Check if the container is empty
  /// \return true if N == 0, false otherwise
  constexpr bool empty() const noexcept { return N == 0; }

  /// \brief Get the number of elements
  /// \return The number of elements in the topology (always N)
  constexpr size_type size() const noexcept { return N; }

  /// \brief Get the maximum number of elements
  /// \return The maximum number of elements (always N)
  constexpr size_type max_size() const noexcept { return N; }

  // Operations

  /// \brief Fill the container with specified value
  /// \param[in] value The value to assign to all elements
  constexpr void fill(const T& value) { m_data.fill(value); }

  /// \brief Swap contents with another topology
  /// \param[in, out] other The topology to swap with
  constexpr void swap(Topology& other) noexcept(
      std::is_nothrow_swappable_v<T>) {
    m_data.swap(other.m_data);
  }

  // Comparison operators

  /// \brief Equality comparison operator
  /// \param[in] other The topology to compare with
  /// \return true if all elements are equal, false otherwise
  constexpr bool operator==(const Topology& other) const noexcept {
    return m_data == other.m_data;
  }

  /// \brief Inequality comparison operator
  /// \param[in] other The topology to compare with
  /// \return true if any element is not equal, false otherwise
  constexpr bool operator!=(const Topology& other) const noexcept {
    return !(*this == other);
  }

  /// \brief Less-than comparison operator
  /// \param[in] other The topology to compare with
  /// \return true if this is lexicographically less than other
  constexpr bool operator<(const Topology& other) const noexcept {
    return m_data < other.m_data;
  }

  /// \brief Less-than-or-equal comparison operator
  /// \param[in] other The topology to compare with
  /// \return true if this is lexicographically less than or equal to other
  constexpr bool operator<=(const Topology& other) const noexcept {
    return m_data <= other.m_data;
  }

  /// \brief Greater-than comparison operator
  /// \param[in] other The topology to compare with
  /// \return true if this is lexicographically greater than other
  constexpr bool operator>(const Topology& other) const noexcept {
    return m_data > other.m_data;
  }

  /// \brief Greater-than-or-equal comparison operator
  /// \param[in] other The topology to compare with
  /// \return true if this is lexicographically greater than or equal to other
  constexpr bool operator>=(const Topology& other) const noexcept {
    return m_data >= other.m_data;
  }

  // Additional methods for compatibility and convenience

  /// \brief Get the underlying std::array (const version)
  /// \return Const reference to the underlying std::array
  constexpr const std::array<T, N>& array() const noexcept { return m_data; }

  /// \brief Get the underlying std::array
  /// \return Reference to the underlying std::array
  constexpr std::array<T, N>& array() noexcept { return m_data; }
};

// Non-member functions for std::array compatibility

/// \brief Swap two topologies
/// \tparam T Element type
/// \tparam N Number of elements
/// \tparam LayoutType Layout type
/// \param[in, out] lhs First topology
/// \param[in, out] rhs Second topology
template <typename T, std::size_t N, typename LayoutType>
constexpr void swap(
    Topology<T, N, LayoutType>& lhs,
    Topology<T, N, LayoutType>& rhs) noexcept(noexcept(lhs.swap(rhs))) {
  lhs.swap(rhs);
}

/// \brief Get element by index (non-member function)
/// \tparam I Index (must be < N)
/// \tparam T Element type
/// \tparam N Number of elements
/// \tparam LayoutType Layout type
/// \param[in, out] topology The topology to access
/// \return Reference to the element at index I
template <std::size_t I, typename T, std::size_t N, typename LayoutType>
constexpr T& get(Topology<T, N, LayoutType>& topology) noexcept {
  static_assert(I < N, "Index out of bounds");
  return topology[I];
}

/// \brief Get element by index (const non-member function)
/// \tparam I Index (must be < N)
/// \tparam T Element type
/// \tparam N Number of elements
/// \tparam LayoutType Layout type
/// \param[in] topology The topology to access
/// \return Const reference to the element at index I
template <std::size_t I, typename T, std::size_t N, typename LayoutType>
constexpr const T& get(const Topology<T, N, LayoutType>& topology) noexcept {
  static_assert(I < N, "Index out of bounds");
  return topology[I];
}

/// \brief Get element by index (rvalue reference version)
/// \tparam I Index (must be < N)
/// \tparam T Element type
/// \tparam N Number of elements
/// \tparam LayoutType Layout type
/// \param[in, out] topology The topology to access
/// \return Rvalue reference to the element at index I
template <std::size_t I, typename T, std::size_t N, typename LayoutType>
constexpr T&& get(Topology<T, N, LayoutType>&& topology) noexcept {
  static_assert(I < N, "Index out of bounds");
  return std::move(topology[I]);
}

/// \brief Get element by index (const rvalue reference version)
/// \tparam I Index (must be < N)
/// \tparam T Element type
/// \tparam N Number of elements
/// \tparam LayoutType Layout type
/// \param[in] topology The topology to access
/// \return Const rvalue reference to the element at index I
template <std::size_t I, typename T, std::size_t N, typename LayoutType>
constexpr const T&& get(const Topology<T, N, LayoutType>&& topology) noexcept {
  static_assert(I < N, "Index out of bounds");
  return std::move(topology[I]);
}

}  // namespace Distributed
}  // namespace KokkosFFT

#endif
