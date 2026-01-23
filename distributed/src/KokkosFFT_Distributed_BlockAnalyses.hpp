#ifndef KOKKOSFFT_DISTRIBUTED_BLOCK_ANALYSES_HPP
#define KOKKOSFFT_DISTRIBUTED_BLOCK_ANALYSES_HPP

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_Types.hpp"
#include "KokkosFFT_Distributed_MPI_Helper.hpp"
#include "KokkosFFT_Distributed_Mapping.hpp"
#include "KokkosFFT_Distributed_Topologies.hpp"
#include "KokkosFFT_Distributed_Extents.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief Count the number of blocks of a given type
/// \tparam BlockInfoType Type of BlockInfo
/// \param[in] block_infos Vector of BlockInfo for accumulated proposed blocks
/// \param[in] block_type Type of block to count
/// \return Number of blocks of the given type included in block_infos
template <typename BlockInfoType>
auto count_blocks(const std::vector<BlockInfoType>& block_infos,
                  BlockType block_type) {
  return std::count_if(block_infos.cbegin(), block_infos.cend(),
                       [block_type](BlockInfoType block_info) {
                         return block_info.m_block_type == block_type;
                       });
};

/// \brief Count the total number of FFT dimensions in the given blocks
/// \tparam BlockInfoType Type of BlockInfo
/// \param[in] block_infos Vector of BlockInfo for accumulated proposed blocks
/// \return Total number of FFT dimensions in the given blocks
template <typename BlockInfoType>
auto count_fft_dims(const std::vector<BlockInfoType>& block_infos)
    -> std::size_t {
  std::size_t total_fft_dims = 0;
  for (const auto& block_info : block_infos) {
    if (block_info.m_block_type == BlockType::FFT) {
      total_fft_dims += block_info.m_fft_dim;
    }
  }
  return total_fft_dims;
};

/// \brief Propose an FFT block
/// \tparam BlockInfoType Type of BlockInfo
/// \tparam DIM Number of dimensions
/// \param[in] map Mapping of local transpose to perform FFT on contiguous data
/// \param[in] in_extents Input extents
/// \param[in] out_extents Output extents
/// \param[in] fft_dim Number of FFT dimensions
/// \param[in] block_idx Index of the current block
/// \return A tuple containing the proposed block and the maximum buffer size
template <typename BlockInfoType, std::size_t DIM>
auto propose_fft_block(const std::array<std::size_t, DIM>& map,
                       const std::array<std::size_t, DIM>& in_extents,
                       const std::array<std::size_t, DIM>& out_extents,
                       std::size_t fft_dim, std::size_t block_idx) {
  BlockInfoType block;
  block.m_block_type  = BlockType::FFT;
  block.m_fft_dim     = fft_dim;
  block.m_in_extents  = in_extents;
  block.m_out_extents = out_extents;
  block.m_in_map      = map;
  block.m_out_map     = block.m_in_map;
  block.m_block_idx   = block_idx;

  std::size_t max_buffer_size =
      KokkosFFT::Impl::total_size(block.m_out_extents) * 2;
  return std::make_tuple(block, max_buffer_size);
};

/// \brief Propose a transpose block
/// \tparam BlockInfoType Type of BlockInfo
/// \tparam LayoutType Layout type for the Input/Output Views
/// \tparam DIM Number of dimensions
/// \param[in] map Mapping of local transpose to perform FFT on contiguous data
/// \param[in] in_topology Input topology
/// \param[in] out_topology Output topology
/// \param[in] in_extents Input extents
/// \param[in] global_extents Global input extents
/// \param[in] axes Axes of the transformation
/// \param[in] rank MPI rank
/// \param[in] is_layout_right Whether the layout is right
/// \param[in] is_last Whether this is the last block
/// \param[in] in_axis Input axis of the transpose
/// \param[in] out_axis Output axis of the transpose
/// \param[in] comm_axis Axis of communication
/// \param[in] block_idx Index of the current block
/// \param[in] size_factor Size factor of the block
/// \return A tuple containing the proposed block and the maximum buffer size
template <typename BlockInfoType, typename LayoutType, typename iType,
          std::size_t DIM>
auto propose_transpose_block(const std::array<std::size_t, DIM>& map,
                             const std::array<std::size_t, DIM>& in_topology,
                             const std::array<std::size_t, DIM>& out_topology,
                             const std::array<std::size_t, DIM>& in_extents,
                             const std::array<std::size_t, DIM>& global_extents,
                             std::vector<iType>& axes, std::size_t rank,
                             bool is_layout_right, bool is_last,
                             std::size_t in_axis, std::size_t out_axis,
                             std::size_t comm_axis, std::size_t block_idx,
                             std::size_t size_factor) {
  BlockInfoType block;
  block.m_block_type   = BlockType::Transpose;
  block.m_in_map       = map;
  block.m_in_axis      = in_axis;
  block.m_out_axis     = out_axis;
  block.m_comm_axis    = comm_axis;
  block.m_in_topology  = in_topology;
  block.m_out_topology = out_topology;
  block.m_in_extents   = in_extents;
  block.m_block_idx    = block_idx;

  if (is_last) {
    block.m_out_map = KokkosFFT::Impl::index_sequence<std::size_t, DIM, 0>();
  } else {
    block.m_out_map = axes.empty()
                          ? get_dst_map<LayoutType>(block.m_in_map, out_axis)
                          : get_dst_map<LayoutType>(block.m_in_map, axes);
  }

  block.m_out_extents = compute_next_extents(
      global_extents, out_topology, block.m_out_map, rank, is_layout_right);
  block.m_buffer_extents = compute_buffer_extents<LayoutType>(
      global_extents, in_topology, out_topology);

  std::vector<std::size_t> all_max_buffer_sizes = {
      KokkosFFT::Impl::total_size(block.m_in_extents),
      KokkosFFT::Impl::total_size(block.m_buffer_extents),
      KokkosFFT::Impl::total_size(block.m_out_extents)};
  std::size_t max_buffer_size = *std::max_element(all_max_buffer_sizes.begin(),
                                                  all_max_buffer_sizes.end());
  return std::make_tuple(block, max_buffer_size * size_factor);
};

/// \brief Propose a block (or blocks) for the Slab distributed FFT plan for the
/// current block
/// \tparam BlockInfoType Type of BlockInfo
/// \tparam ValueType Value type for the FFT
/// \tparam LayoutType Layout type for the Input/Output Views
/// \tparam FFT_DIM Number of FFT dimensions
/// \tparam TopologyType Topology type to represent pencil topologies
/// \tparam iType Index type for axes
/// \tparam DIM The dimension of the Input/Output Views
///
/// \param[in] in_extents Input extents
/// \param[in] gin_extents Global input extents
/// \param[in] gout_extents Global output extents
/// \param[in] block_infos Vector of BlockInfo for accumulated proposed blocks
/// \param[in] all_topologies Vector of all pencil topologies
/// \param[in] all_axes Vector of all axes for FFTs
/// \param[in] map Mapping of local transpose to prepare first FFT
/// \param[in] rank MPI rank
/// \param[in] block_idx Index of the current block
template <typename BlockInfoType, typename ValueType, typename LayoutType,
          std::size_t FFT_DIM, typename TopologyType, typename iType,
          std::size_t DIM>
auto propose_block(const std::array<std::size_t, DIM>& in_extents,
                   const std::array<std::size_t, DIM>& gin_extents,
                   const std::array<std::size_t, DIM>& gout_extents,
                   const std::vector<BlockInfoType>& block_infos,
                   const std::vector<TopologyType>& all_topologies,
                   const std::vector<std::vector<iType>>& all_axes,
                   [[maybe_unused]] const std::array<std::size_t, DIM>& map,
                   [[maybe_unused]] std::size_t rank, std::size_t block_idx) {
  auto compute_transpose_block =
      [&](const std::vector<BlockInfoType>& block_infos,
          std::size_t size_factor) {
        bool is_last          = block_idx == (all_topologies.size() - 1);
        auto nb_trans_blocks  = count_blocks(block_infos, BlockType::Transpose);
        auto current_topology = all_topologies.at(nb_trans_blocks);
        auto next_topology    = all_topologies.at(nb_trans_blocks + 1);
        auto [in_axis, out_axis] =
            slab_in_out_axes(current_topology, next_topology);

        std::vector<iType> next_axes =
            is_last ? std::vector<iType>{} : all_axes.at(block_idx + 1);
        if (block_infos.empty()) {
          auto src_map = KokkosFFT::Impl::index_sequence<std::size_t, DIM, 0>();
          // If the transpose is the first block, the fft block must follow it.
          return propose_transpose_block<BlockInfoType, LayoutType>(
              src_map, current_topology, next_topology, in_extents, gin_extents,
              next_axes, rank, true, false, in_axis, out_axis, 0, 0,
              size_factor);
        } else {
          auto in_map     = block_infos.back().m_out_map;
          auto in_extents = block_infos.back().m_out_extents;

          // There must be FFT block before this, so the data must be in complex
          return propose_transpose_block<BlockInfoType, LayoutType>(
              in_map, current_topology, next_topology, in_extents, gout_extents,
              next_axes, rank, true, is_last, in_axis, out_axis, 0,
              nb_trans_blocks, 2);
        }
      };

  using BlockVectorType = std::vector<std::tuple<BlockInfoType, std::size_t>>;
  BlockVectorType block_vector;
  auto axes    = all_axes.at(block_idx);
  auto fft_dim = axes.size();

  if (fft_dim == 0) {
    // axes is empty -> Transpose block
    const std::size_t size_factor =
        KokkosFFT::Impl::is_real_v<ValueType> ? 1 : 2;

    auto block_tuple = compute_transpose_block(block_infos, size_factor);
    block_vector.push_back(block_tuple);
    return block_vector;
  } else {
    bool is_last_fft_block = (count_fft_dims(block_infos) + fft_dim) == FFT_DIM;
    if (block_infos.empty()) {
      auto topology       = all_topologies.at(block_idx);
      auto fft_in_extents = compute_mapped_extents(in_extents, map);
      auto fft_out_extents =
          compute_next_extents(gout_extents, topology, map, rank);
      auto fft_block_tuple = propose_fft_block<BlockInfoType>(
          map, fft_in_extents, fft_out_extents, fft_dim, 0);
      block_vector.push_back(fft_block_tuple);
      if (!is_last_fft_block) {
        auto new_block_infos = block_infos;
        new_block_infos.push_back(std::get<0>(fft_block_tuple));
        auto trans_block_tuple = compute_transpose_block(new_block_infos, 2);
        block_vector.push_back(trans_block_tuple);
      }
      return block_vector;
    } else {
      auto nb_fft_blocks  = count_blocks(block_infos, BlockType::FFT);
      auto in_map         = block_infos.back().m_out_map;
      auto fft_in_extents = block_infos.back().m_out_extents;

      if (nb_fft_blocks == 0) {
        // First FFT block
        auto topology = block_infos.back().m_out_topology;
        auto fft_out_extents =
            compute_next_extents(gout_extents, topology, in_map, rank);
        auto fft_block_tuple = propose_fft_block<BlockInfoType>(
            in_map, fft_in_extents, fft_out_extents, fft_dim, 0);
        block_vector.push_back(fft_block_tuple);
      } else {
        auto fft_block_tuple = propose_fft_block<BlockInfoType>(
            in_map, fft_in_extents, fft_in_extents, fft_dim, nb_fft_blocks);
        block_vector.push_back(fft_block_tuple);
      }

      if (!is_last_fft_block) {
        auto new_block_infos = block_infos;
        new_block_infos.push_back(std::get<0>(block_vector.back()));
        auto trans_block_tuple = compute_transpose_block(new_block_infos, 2);
        block_vector.push_back(trans_block_tuple);
      }
      return block_vector;
    }
  }
}

/// \brief Propose a block (or blocks) for the Pencil distributed FFT plan for
/// the current block
/// \tparam BlockInfoType Type of BlockInfo
/// \tparam ValueType Value type for the FFT
/// \tparam LayoutType Layout type for the Input/Output Views
/// \tparam FFT_DIM Number of FFT dimensions
/// \tparam TopologyType Topology type to represent pencil topologies
/// \tparam iType Index type for axes
/// \tparam DIM The dimension of the Input/Output Views
///
/// \param[in] in_extents Input extents
/// \param[in] gin_extents Global input extents
/// \param[in] gout_extents Global output extents
/// \param[in] block_infos Vector of BlockInfo for accumulated proposed blocks
/// \param[in] all_topologies Vector of all pencil topologies
/// \param[in] all_trans_axes Vector of all transpose axes
/// \param[in] all_layouts Vector of all layouts
/// \param[in] all_axes Vector of all axes for FFTs
/// \param[in] map Mapping of local transpose to prepare first FFT
/// \param[in] rank MPI rank
/// \param[in] block_idx Index of the current block
template <typename BlockInfoType, typename ValueType, typename LayoutType,
          std::size_t FFT_DIM, typename TopologyType, typename iType,
          std::size_t DIM>
auto propose_block(const std::array<std::size_t, DIM>& in_extents,
                   const std::array<std::size_t, DIM>& gin_extents,
                   const std::array<std::size_t, DIM>& gout_extents,
                   const std::vector<BlockInfoType>& block_infos,
                   const std::vector<TopologyType>& all_topologies,
                   const std::vector<std::size_t>& all_trans_axes,
                   const std::vector<std::size_t>& all_layouts,
                   const std::vector<std::vector<iType>>& all_axes,
                   [[maybe_unused]] const std::array<std::size_t, DIM>& map,
                   [[maybe_unused]] std::size_t rank, std::size_t block_idx) {
  auto compute_transpose_block =
      [&](const std::vector<BlockInfoType>& block_infos,
          std::size_t size_factor) {
        bool is_last          = block_idx == (all_topologies.size() - 1);
        auto nb_trans_blocks  = count_blocks(block_infos, BlockType::Transpose);
        auto current_topology = all_topologies.at(nb_trans_blocks);
        auto next_topology    = all_topologies.at(nb_trans_blocks + 1);
        auto [in_axis, out_axis] =
            pencil_in_out_axes(current_topology, next_topology);

        std::vector<iType> next_axes =
            is_last ? std::vector<iType>{} : all_axes.at(block_idx + 1);
        auto comm_axis       = all_trans_axes.at(nb_trans_blocks);
        bool is_layout_right = all_layouts.at(nb_trans_blocks + 1);
        if (block_infos.empty()) {
          auto src_map = KokkosFFT::Impl::index_sequence<std::size_t, DIM, 0>();
          // If the transpose is the first block, the fft block must follow it.
          return propose_transpose_block<BlockInfoType, LayoutType>(
              src_map, current_topology, next_topology, in_extents, gin_extents,
              next_axes, rank, is_layout_right, false, in_axis, out_axis,
              comm_axis, 0, size_factor);
        } else {
          auto in_map     = block_infos.back().m_out_map;
          auto in_extents = block_infos.back().m_out_extents;

          // There must be FFT block before this, so the data must be in complex
          return propose_transpose_block<BlockInfoType, LayoutType>(
              in_map, current_topology, next_topology, in_extents, gout_extents,
              next_axes, rank, is_layout_right, is_last, in_axis, out_axis,
              comm_axis, nb_trans_blocks, 2);
        }
      };

  using BlockVectorType = std::vector<std::tuple<BlockInfoType, std::size_t>>;
  BlockVectorType block_vector;
  auto axes    = all_axes.at(block_idx);
  auto fft_dim = axes.size();

  if (fft_dim == 0) {
    // axes is empty -> Transpose block
    const std::size_t size_factor =
        KokkosFFT::Impl::is_real_v<ValueType> ? 1 : 2;

    auto block_tuple = compute_transpose_block(block_infos, size_factor);
    block_vector.push_back(block_tuple);
    return block_vector;
  } else {
    [[maybe_unused]] bool is_layout_right = all_layouts.at(block_idx);

    bool is_last_fft_block = (count_fft_dims(block_infos) + fft_dim) == FFT_DIM;
    if (block_infos.empty()) {
      auto topology        = all_topologies.at(block_idx);
      auto fft_in_extents  = compute_mapped_extents(in_extents, map);
      auto fft_out_extents = compute_next_extents(gout_extents, topology, map,
                                                  rank, is_layout_right);
      auto fft_block_tuple = propose_fft_block<BlockInfoType>(
          map, fft_in_extents, fft_out_extents, fft_dim, 0);
      block_vector.push_back(fft_block_tuple);
      if (!is_last_fft_block) {
        auto new_block_infos = block_infos;
        new_block_infos.push_back(std::get<0>(fft_block_tuple));
        auto trans_block_tuple = compute_transpose_block(new_block_infos, 2);
        block_vector.push_back(trans_block_tuple);
      }
      return block_vector;
    } else {
      auto nb_fft_blocks  = count_blocks(block_infos, BlockType::FFT);
      auto in_map         = block_infos.back().m_out_map;
      auto fft_in_extents = block_infos.back().m_out_extents;

      if (nb_fft_blocks == 0) {
        // First FFT block
        auto topology        = block_infos.back().m_out_topology;
        auto fft_out_extents = compute_next_extents(
            gout_extents, topology, in_map, rank, is_layout_right);
        auto fft_block_tuple = propose_fft_block<BlockInfoType>(
            in_map, fft_in_extents, fft_out_extents, fft_dim, 0);
        block_vector.push_back(fft_block_tuple);
      } else {
        auto fft_block_tuple = propose_fft_block<BlockInfoType>(
            in_map, fft_in_extents, fft_in_extents, fft_dim, nb_fft_blocks);
        block_vector.push_back(fft_block_tuple);
      }

      if (!is_last_fft_block) {
        auto new_block_infos = block_infos;
        new_block_infos.push_back(std::get<0>(block_vector.back()));
        auto trans_block_tuple = compute_transpose_block(new_block_infos, 2);
        block_vector.push_back(trans_block_tuple);
      }
      return block_vector;
    }
  }
}

/// \brief SlabBlockAnalysesInternal struct
/// Based on the list of slab topologies and FFT axes, propose the
/// sequence of blocks (Transpose and FFT) needed to perform the distributed
/// FFT.
/// \tparam ValueType Value type for the FFT
/// \tparam Layout Layout type for the Input/Output Views
/// \tparam iType Integer type for axes
/// \tparam DIM Dimension of the Input/Output Views
/// \tparam FFT_DIM Number of FFT dimensions
template <typename ValueType, typename Layout, typename iType, std::size_t DIM,
          std::size_t FFT_DIM>
struct SlabBlockAnalysesInternal {
  using BlockInfoType = BlockInfo<DIM>;
  using extents_type  = std::array<std::size_t, DIM>;
  std::vector<BlockInfoType> m_block_infos;
  std::size_t m_max_buffer_size;

  /// \brief Constructor for SlabBlockAnalysesInternal
  /// \param[in] in_extents Input extents
  /// \param[in] gin_extents Global input extents
  /// \param[in] gout_extents Global output extents
  /// \param[in] in_topology Input topology
  /// \param[in] out_topology Output topology
  /// \param[in] axes Axes of the FFT
  /// \param[in] rank MPI rank
  SlabBlockAnalysesInternal(const extents_type& in_extents,
                            const extents_type& gin_extents,
                            const extents_type& gout_extents,
                            const extents_type& in_topology,
                            const extents_type& out_topology,
                            const std::array<iType, FFT_DIM>& axes,
                            MPI_Comm comm) {
    auto [map, map_inv] = KokkosFFT::Impl::get_map_axes<Layout, DIM>(axes);

    // Get all relevant topologies
    auto all_topologies =
        get_all_slab_topologies(in_topology, out_topology, axes);

    auto all_axes = decompose_axes(all_topologies, axes);
    int rank;
    MPI_Comm_rank(comm, &rank);

    std::vector<std::size_t> all_max_buffer_sizes;
    for (std::size_t itopo = 0; itopo < all_topologies.size(); ++itopo) {
      auto blocks = propose_block<BlockInfoType, ValueType, Layout, FFT_DIM>(
          in_extents, gin_extents, gout_extents, m_block_infos, all_topologies,
          all_axes, map, rank, itopo);
      for (auto const& block : blocks) {
        auto [b, max_buffer_size] = block;
        m_block_infos.push_back(b);
        all_max_buffer_sizes.push_back(max_buffer_size);
      }
    }
    m_max_buffer_size = compute_global_max(all_max_buffer_sizes, comm);
  }
};

/// \brief PencilBlockAnalysesInternal struct
/// Based on the list of pencil topologies and FFT axes, propose the
/// sequence of blocks (Transpose and FFT) needed to perform the distributed
/// FFT.
/// \tparam ValueType Value type for the FFT
/// \tparam Layout Layout type for the Input/Output Views
/// \tparam iType Integer type for axes
/// \tparam DIM Dimension of the Input/Output Views
/// \tparam FFT_DIM Number of FFT dimensions
/// \tparam InLayoutType Layout type for the Input Topology
/// \tparam OutLayoutType Layout type for the Output Topology
template <typename ValueType, typename Layout, typename iType, std::size_t DIM,
          std::size_t FFT_DIM, typename InLayoutType, typename OutLayoutType>
struct PencilBlockAnalysesInternal {
  using BlockInfoType     = BlockInfo<DIM>;
  using extents_type      = std::array<std::size_t, DIM>;
  using in_topology_type  = Topology<std::size_t, DIM, InLayoutType>;
  using out_topology_type = Topology<std::size_t, DIM, OutLayoutType>;
  std::vector<BlockInfoType> m_block_infos;
  std::size_t m_max_buffer_size;

  /// \brief Constructor for PencilBlockAnalysesInternal
  /// \param[in] in_extents Input extents
  /// \param[in] gin_extents Global input extents
  /// \param[in] gout_extents Global output extents
  /// \param[in] in_topology Input topology
  /// \param[in] out_topology Output topology
  /// \param[in] axes Axes of the FFT
  /// \param[in] comm MPI communicator
  PencilBlockAnalysesInternal(const std::array<std::size_t, DIM>& in_extents,
                              const std::array<std::size_t, DIM>& gin_extents,
                              const std::array<std::size_t, DIM>& gout_extents,
                              const in_topology_type& in_topology,
                              const out_topology_type& out_topology,
                              const std::array<iType, FFT_DIM>& axes,
                              MPI_Comm comm) {
    auto [map, map_inv] = KokkosFFT::Impl::get_map_axes<Layout, DIM>(axes);

    // Get all relevant topologies
    auto [all_topologies, all_trans_axes, all_layouts] =
        get_all_pencil_topologies(in_topology, out_topology, axes);
    auto all_axes = decompose_axes(all_topologies, axes);

    int rank;
    MPI_Comm_rank(comm, &rank);

    std::vector<std::size_t> all_max_buffer_sizes;
    for (std::size_t itopo = 0; itopo < all_topologies.size(); ++itopo) {
      auto blocks = propose_block<BlockInfoType, ValueType, Layout, FFT_DIM>(
          in_extents, gin_extents, gout_extents, m_block_infos, all_topologies,
          all_trans_axes, all_layouts, all_axes, map, rank, itopo);
      for (auto const& block : blocks) {
        auto [b, max_buffer_size] = block;
        m_block_infos.push_back(b);
        all_max_buffer_sizes.push_back(max_buffer_size);
      }
    }
    m_max_buffer_size = compute_global_max(all_max_buffer_sizes, comm);
  }
};

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
