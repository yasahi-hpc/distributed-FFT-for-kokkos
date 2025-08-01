#ifndef TYPES_HPP
#define TYPES_HPP

#include <array>
#include <vector>

enum class TopologyType {
  Pencil,
  Slab,
  Shared,
  Invalid,
};

enum class BlockType {
  Transpose,
  FFT,
  FFT2,
  TransposeAndFFT,
};

enum class OperationType {
  F,
  FT,
  FFT,
  FTF,
  FTFT,
  FTF2,
  FTF2T,
  TF,
  TFT,
  TFTT,
  TFTF,
  TFTFT,
  F2,
  F2T,
  F2TF,
  F2TFT,
  TF2,
  TF2T,
  TF2TF,
  TFTF2,
  TF2TFT,
  TFTF2T
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

  //! The MPI communicator for the block
  MPI_Comm m_comm = MPI_COMM_NULL;

  //! The type of the block (Transpose, FFT, FFT2, TransposeAndFFT)
  BlockType m_block_type;

  bool operator==(const BlockInfo& other) const {
    return m_in_topology == other.m_in_topology &&
           m_out_topology == other.m_out_topology &&
           m_in_extents == other.m_in_extents &&
           m_out_extents == other.m_out_extents &&
           m_buffer_extents == other.m_buffer_extents &&
           m_in_map == other.m_in_map && m_out_map == other.m_out_map &&
           m_in_axis == other.m_in_axis && m_out_axis == other.m_out_axis &&
           m_axes == other.m_axes && m_comm == other.m_comm &&
           m_block_type == other.m_block_type;
  }

  bool operator!=(const BlockInfo& other) const { return !(*this == other); }
};

#endif
