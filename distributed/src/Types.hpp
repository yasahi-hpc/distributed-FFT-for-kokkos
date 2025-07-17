#ifndef TYPES_HPP
#define TYPES_HPP

#include <array>

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

enum class OperationType { F, FT, FFT, FTF, FTFT, TF, TFT, TFTF, TFTFT };

template <std::size_t DIM>
struct BlockInfo {
  using map_type            = std::array<std::size_t, DIM>;
  using extents_type        = std::array<std::size_t, DIM>;
  using buffer_extents_type = std::array<std::size_t, DIM + 1>;

  extents_type m_in_topology;
  extents_type m_out_topology;

  extents_type m_in_extents;
  extents_type m_out_extents;

  buffer_extents_type m_buffer_extents;

  map_type m_in_map;
  map_type m_out_map;

  std::size_t m_in_axis  = 0;
  std::size_t m_out_axis = 0;

  BlockType m_block_type;
};

#endif
