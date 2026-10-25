#ifndef TEST_BLOCK_ANALYSES_UTILS_HPP
#define TEST_BLOCK_ANALYSES_UTILS_HPP

#include <algorithm>
#include <array>
#include <numeric>
#include "KokkosFFT_Distributed_BlockAnalyses.hpp"

/// \brief Check that a map is a permutation of (0, 1, ..., DIM-1)
template <std::size_t DIM>
bool is_permutation(const std::array<std::size_t, DIM>& map) {
  std::array<std::size_t, DIM> identity{};
  std::iota(identity.begin(), identity.end(), std::size_t(0));
  return std::is_permutation(map.begin(), map.end(), identity.begin());
}

/// \brief Make a reference FFT block.
/// The topologies, the buffer extents and the axes of a transpose are unused
/// for an FFT block and thus kept default initialized.
template <std::size_t DIM>
auto make_fft_block(const std::array<std::size_t, DIM>& in_extents,
                    const std::array<std::size_t, DIM>& out_extents,
                    const std::array<std::size_t, DIM>& map,
                    std::size_t fft_dim, std::size_t block_idx) {
  KokkosFFT::Distributed::Impl::BlockInfo<DIM> block;
  block.m_block_type  = KokkosFFT::Distributed::Impl::BlockType::FFT;
  block.m_in_extents  = in_extents;
  block.m_out_extents = out_extents;
  block.m_in_map      = map;
  block.m_out_map     = map;
  block.m_fft_dim     = fft_dim;
  block.m_block_idx   = block_idx;
  return block;
}

/// \brief Make a reference Transpose block.
/// The FFT dimension is unused for a transpose block and thus kept default
/// initialized.
template <std::size_t DIM>
auto make_transpose_block(
    const std::array<std::size_t, DIM>& in_topology,
    const std::array<std::size_t, DIM>& out_topology,
    const std::array<std::size_t, DIM>& in_extents,
    const std::array<std::size_t, DIM>& out_extents,
    const std::array<std::size_t, DIM + 1>& buffer_extents,
    const std::array<std::size_t, DIM>& in_map,
    const std::array<std::size_t, DIM>& out_map, std::size_t in_axis,
    std::size_t out_axis, std::size_t comm_axis, std::size_t block_idx) {
  KokkosFFT::Distributed::Impl::BlockInfo<DIM> block;
  block.m_block_type     = KokkosFFT::Distributed::Impl::BlockType::Transpose;
  block.m_in_topology    = in_topology;
  block.m_out_topology   = out_topology;
  block.m_in_extents     = in_extents;
  block.m_out_extents    = out_extents;
  block.m_buffer_extents = buffer_extents;
  block.m_in_map         = in_map;
  block.m_out_map        = out_map;
  block.m_in_axis        = in_axis;
  block.m_out_axis       = out_axis;
  block.m_comm_axis      = comm_axis;
  block.m_block_idx      = block_idx;
  return block;
}

#endif
