#ifndef PENCIL_BLOCK_ANALYSES_HPP
#define PENCIL_BLOCK_ANALYSES_HPP

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "MPI_Helper.hpp"
#include "Utils.hpp"
#include "Types.hpp"
#include "Topologies.hpp"
#include "Extents.hpp"

template <typename ValueType, typename Layout, typename iType, std::size_t DIM,
          std::size_t FFT_DIM>
struct PencilBlockAnalysesInternal;

/// \brief Get all pencil block info for a given input and output topology
/// 1. FFT // E.g. {1, P0, P1}
/// 2. FFT + Transpose // E.g. {1, P0, P1} FFT (ax=0) -> {P0, 1, P1}
/// 3. Transpose + FFT // E.g. {P0, 1, P1} -> {1, P0, P1} FFT (ax=0)
/// 4. Transpose + FFT + Transpose // E.g. {P0, 1, P1} -> {1, P0, P1} FFT (ax=0)
/// -> {P0, 1, P1}
template <typename ValueType, typename Layout, typename iType, std::size_t DIM>
struct PencilBlockAnalysesInternal<ValueType, Layout, iType, DIM, 1> {
  using BlockInfoType = BlockInfo<DIM>;
  using extents_type  = std::array<std::size_t, DIM>;
  std::vector<BlockInfoType> m_block_infos;
  std::size_t m_max_buffer_size;
  OperationType m_op_type;

  PencilBlockAnalysesInternal(const std::array<std::size_t, DIM>& in_extents,
                              const std::array<std::size_t, DIM>& out_extents,
                              const std::array<std::size_t, DIM>& gin_extents,
                              const std::array<std::size_t, DIM>& gout_extents,
                              const std::array<std::size_t, DIM>& in_topology,
                              const std::array<std::size_t, DIM>& out_topology,
                              const std::array<iType, 1>& axes, MPI_Comm comm,
                              const bool is_same_order) {
    auto src_map = KokkosFFT::Impl::index_sequence<std::size_t, DIM, 0>();
    auto [map, map_inv] = get_map_axes<Layout, iType, DIM, 1>(axes);

    // Get all relevant topologies
    auto [all_topologies, all_trans_axes] = get_all_pencil_topologies(
        in_topology, out_topology, axes, is_same_order);

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    const std::size_t size_factor =
        KokkosFFT::Impl::is_real_v<ValueType> ? 1 : 2;

    std::vector<std::size_t> all_max_buffer_sizes;

    std::size_t nb_topologies = all_topologies.size();
    // std::cout << "Number of topologies: " << nb_topologies << std::endl;
    if (nb_topologies == 1) {
      // FFT batched
      // E.g. {1, Px, Py} + FFT ax=0
      m_op_type = OperationType::F;
      BlockInfoType block;
      block.m_in_map     = map;
      block.m_out_map    = map;
      block.m_in_extents = get_mapped_extents(in_extents, map);
      block.m_out_extents =
          get_next_extents(gout_extents, in_topology, map, comm);
      block.m_block_type = BlockType::FFT;
      block.m_axes = get_contiguous_axes<Layout, iType, DIM>(to_vector(axes));
      m_block_infos.push_back(block);

      // Data is always complex
      all_max_buffer_sizes.push_back(get_size(out_extents) * 2);
      m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
    } else if (nb_topologies == 2) {
      auto last_axis = axes.back();
      auto first_dim = in_topology.at(last_axis);
      if (first_dim != 1) {
        m_op_type = OperationType::TF;
        // T + FFT
        // E.g. {Px, 1, Py} -> {1, Px, Py} + FFT (ax=0)
        BlockInfoType block0;
        auto [in_axis0, out_axis0] = get_pencil(in_topology, out_topology);
        block0.m_in_map            = src_map;
        block0.m_out_map   = get_dst_map<Layout, DIM>(src_map, out_axis0);
        block0.m_in_axis   = in_axis0;
        block0.m_out_axis  = out_axis0;
        block0.m_comm_axis = all_trans_axes.at(0);

        block0.m_in_topology  = in_topology;
        block0.m_out_topology = out_topology;
        block0.m_in_extents   = in_extents;
        block0.m_out_extents =
            get_next_extents(gin_extents, out_topology, block0.m_out_map, comm);
        block0.m_buffer_extents =
            get_buffer_extents<Layout>(gin_extents, in_topology, out_topology);
        block0.m_block_type = BlockType::Transpose;

        m_block_infos.push_back(block0);

        all_max_buffer_sizes.push_back(get_size(block0.m_in_extents) *
                                       size_factor);
        all_max_buffer_sizes.push_back(get_size(block0.m_buffer_extents) *
                                       size_factor);
        all_max_buffer_sizes.push_back(get_size(block0.m_out_extents) *
                                       size_factor);

        BlockInfoType block1;
        block1.m_in_extents = block0.m_out_extents;
        block1.m_out_extents =
            get_mapped_extents(out_extents, block0.m_out_map);
        block1.m_block_type = BlockType::FFT;
        block1.m_axes =
            get_contiguous_axes<Layout, iType, DIM>(to_vector(axes));
        block1.m_in_map  = block0.m_out_map;
        block1.m_out_map = block0.m_out_map;
        m_block_infos.push_back(block1);

        all_max_buffer_sizes.push_back(get_size(block1.m_out_extents) * 2);
        m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
      } else {
        m_op_type = OperationType::FT;
        // FFT + T
        // E.g. {1, Px, Py} + FFT (ax=0) -> {Px, 1, Py}
        BlockInfoType block0;
        block0.m_in_map     = map;
        block0.m_out_map    = map;
        block0.m_in_extents = get_mapped_extents(in_extents, map);
        block0.m_out_extents =
            get_next_extents(gout_extents, in_topology, map, comm);
        block0.m_block_type = BlockType::FFT;
        block0.m_axes =
            get_contiguous_axes<Layout, iType, DIM>(to_vector(axes));
        m_block_infos.push_back(block0);
        all_max_buffer_sizes.push_back(get_size(block0.m_out_extents) * 2);

        BlockInfoType block1;
        auto [in_axis1, out_axis1] = get_pencil(in_topology, out_topology);
        block1.m_in_map            = map;
        block1.m_out_map           = src_map;
        block1.m_in_axis           = in_axis1;
        block1.m_out_axis          = out_axis1;
        block1.m_comm_axis         = all_trans_axes.at(0);

        block1.m_in_topology  = in_topology;
        block1.m_out_topology = out_topology;
        block1.m_in_extents   = block0.m_out_extents;
        block1.m_out_extents  = out_extents;
        block1.m_buffer_extents =
            get_buffer_extents<Layout>(gout_extents, in_topology, out_topology);
        block1.m_block_type = BlockType::Transpose;
        m_block_infos.push_back(block1);

        // Data is always complex
        all_max_buffer_sizes.push_back(get_size(block1.m_in_extents) * 2);
        all_max_buffer_sizes.push_back(get_size(block1.m_buffer_extents) * 2);
        all_max_buffer_sizes.push_back(get_size(block1.m_out_extents) * 2);
        m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
      }
    } else if (nb_topologies == 3) {
      m_op_type = OperationType::TFT;
      // T + FFT + T
      // E.g. {1,Px,Py} -> {Px,1,Py} + FFT (ax=1) -> {1,Px,Py}

      auto mid_topology = all_topologies.at(1);

      BlockInfoType block0;
      auto [in_axis0, out_axis0] = get_pencil(in_topology, mid_topology);
      block0.m_in_map            = src_map;
      block0.m_out_map           = get_dst_map<Layout, DIM>(src_map, out_axis0);
      block0.m_in_axis           = in_axis0;
      block0.m_out_axis          = out_axis0;
      block0.m_comm_axis         = all_trans_axes.at(0);

      // Do we really need this?
      block0.m_in_topology  = in_topology;
      block0.m_out_topology = mid_topology;
      block0.m_in_extents   = in_extents;

      bool is_layout_right = block0.m_comm_axis == 0;
      block0.m_out_extents = get_next_extents(
          gin_extents, mid_topology, block0.m_out_map, comm, is_layout_right);
      block0.m_buffer_extents =
          get_buffer_extents<Layout>(gin_extents, in_topology, mid_topology);

      block0.m_block_type = BlockType::Transpose;
      m_block_infos.push_back(block0);

      all_max_buffer_sizes.push_back(get_size(block0.m_in_extents) *
                                     size_factor);
      all_max_buffer_sizes.push_back(get_size(block0.m_buffer_extents) *
                                     size_factor);
      all_max_buffer_sizes.push_back(get_size(block0.m_out_extents) *
                                     size_factor);

      BlockInfoType block1;
      block1.m_in_extents  = block0.m_out_extents;
      block1.m_out_extents = get_next_extents(
          gout_extents, mid_topology, block0.m_out_map, comm, is_layout_right);
      block1.m_block_type = BlockType::FFT;
      block1.m_axes =  // to_vector(get_mapped_axes(axes, block0.m_out_map));
          get_contiguous_axes<Layout, iType, DIM>(to_vector(axes));
      m_block_infos.push_back(block1);

      all_max_buffer_sizes.push_back(get_size(block1.m_out_extents) * 2);

      BlockInfoType block2;
      auto [in_axis2, out_axis2] = get_pencil(mid_topology, out_topology);
      block2.m_in_map            = block0.m_out_map;
      block2.m_out_map           = src_map;
      block2.m_in_axis           = in_axis2;
      block2.m_out_axis          = out_axis2;
      block2.m_comm_axis         = all_trans_axes.at(1);
      block2.m_in_topology       = mid_topology;
      block2.m_out_topology      = out_topology;
      block2.m_in_extents        = block1.m_out_extents;
      block2.m_out_extents       = out_extents;
      block2.m_buffer_extents =
          get_buffer_extents<Layout>(gout_extents, mid_topology, out_topology);
      block2.m_block_type = BlockType::Transpose;
      m_block_infos.push_back(block2);

      all_max_buffer_sizes.push_back(get_size(block2.m_in_extents) * 2);
      all_max_buffer_sizes.push_back(get_size(block2.m_buffer_extents) * 2);
      all_max_buffer_sizes.push_back(get_size(block2.m_out_extents) * 2);
      m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
    } else if (nb_topologies == 4) {
      m_op_type          = OperationType::TFTT;
      auto mid_topology0 = all_topologies.at(1),
           mid_topology1 = all_topologies.at(2);
      // E.g. {1, Px, Py} -> {Py, Px, 1} + FFT (ax=2) -> {1, Px, Py}
      // -> {Px, 1, Py}
      BlockInfoType block0;
      auto [in_axis0, out_axis0] = get_pencil(in_topology, mid_topology0);
      block0.m_in_map            = src_map;
      block0.m_out_map           = get_dst_map<Layout, DIM>(src_map, out_axis0);
      block0.m_in_axis           = in_axis0;
      block0.m_out_axis          = out_axis0;
      block0.m_comm_axis         = all_trans_axes.at(0);

      // Do we really need this?
      block0.m_in_topology  = in_topology;
      block0.m_out_topology = mid_topology0;
      block0.m_in_extents   = in_extents;
      block0.m_out_extents =
          get_next_extents(gin_extents, mid_topology0, block0.m_out_map, comm);
      block0.m_buffer_extents =
          get_buffer_extents<Layout>(gin_extents, in_topology, mid_topology0);

      block0.m_block_type = BlockType::Transpose;
      m_block_infos.push_back(block0);

      all_max_buffer_sizes.push_back(get_size(block0.m_in_extents) *
                                     size_factor);
      all_max_buffer_sizes.push_back(get_size(block0.m_buffer_extents) *
                                     size_factor);
      all_max_buffer_sizes.push_back(get_size(block0.m_out_extents) *
                                     size_factor);

      BlockInfoType block1;
      block1.m_in_extents = block0.m_out_extents;
      block1.m_out_extents =
          get_next_extents(gout_extents, mid_topology0, block0.m_out_map, comm);
      block1.m_block_type = BlockType::FFT;
      block1.m_axes =  // to_vector(get_mapped_axes(axes, block0.m_out_map));
          get_contiguous_axes<Layout, iType, DIM>(to_vector(axes));
      m_block_infos.push_back(block1);

      all_max_buffer_sizes.push_back(get_size(block1.m_out_extents) * 2);

      BlockInfoType block2;
      auto [in_axis2, out_axis2] = get_pencil(mid_topology0, mid_topology1);
      block2.m_in_map            = block0.m_out_map;
      block2.m_out_map           = block2.m_in_map;
      block2.m_in_axis           = in_axis2;
      block2.m_out_axis          = out_axis2;
      block2.m_comm_axis         = all_trans_axes.at(1);
      block2.m_in_topology       = mid_topology0;
      block2.m_out_topology      = mid_topology1;
      block2.m_in_extents =
          get_next_extents(gout_extents, mid_topology1, block2.m_out_map, comm);
      block2.m_out_extents    = out_extents;
      block2.m_buffer_extents = get_buffer_extents<Layout>(
          gout_extents, mid_topology0, mid_topology1);
      block2.m_block_type = BlockType::Transpose;
      m_block_infos.push_back(block2);

      all_max_buffer_sizes.push_back(get_size(block2.m_in_extents) * 2);
      all_max_buffer_sizes.push_back(get_size(block2.m_buffer_extents) * 2);
      all_max_buffer_sizes.push_back(get_size(block2.m_out_extents) * 2);

      BlockInfoType block3;
      auto [in_axis3, out_axis3] = get_pencil(mid_topology1, out_topology);
      block3.m_in_map            = block2.m_out_map;
      block3.m_out_map           = src_map;
      block3.m_in_axis           = in_axis3;
      block3.m_out_axis          = out_axis3;
      block3.m_comm_axis         = all_trans_axes.at(2);
      block3.m_in_topology       = mid_topology1;
      block3.m_out_topology      = out_topology;
      block3.m_in_extents        = block2.m_out_extents;
      block3.m_out_extents       = out_extents;
      block3.m_buffer_extents =
          get_buffer_extents<Layout>(gout_extents, mid_topology1, out_topology);
      block3.m_block_type = BlockType::Transpose;
      m_block_infos.push_back(block3);

      all_max_buffer_sizes.push_back(get_size(block3.m_in_extents) * 2);
      all_max_buffer_sizes.push_back(get_size(block3.m_buffer_extents) * 2);
      all_max_buffer_sizes.push_back(get_size(block3.m_out_extents) * 2);

      m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
    }
  }
};

/*
/// \brief Get all pencil block info for a given input and output topology
/// 2D case
/// 1. FFT + T + FFT // E.g. {1, P0, P1} + FFT (ax=0) -> {P0, 1, P1} + FFT
(ax=1)
/// 2. FFT + T + FFT + T // E.g. {1, P0, P1} + FFT (ax=0) -> {P0, 1, P1} + FFT
(ax=1) -> {1, P0, P1}
/// 3. T + FFT + T + FFT // E.g. {1, P0, P1} -> {P0, 1, P1} + FFT (ax=1) -> {1,
P0, P1} + FFT (ax=0)
/// 4. T + FFT + T + FFT + T // E.g. {1, P0, P1} -> {P0, 1, P1} + FFT (ax=1) ->
{1, P0, P1} + FFT (ax=0) -> {P0, P1, 1} template <typename Layout, typename
iType, std::size_t DIM> struct PencilBlockAnalysesInternal<Layout, iType, DIM,
2> { using BlockInfoType = BlockInfo<DIM>; std::vector<BlockInfoType>
m_block_infos;

  PencilBlockAnalysesInternal(const std::array<std::size_t, DIM>& in_extents,
    const std::array<std::size_t, DIM>& out_extents,
    const std::array<std::size_t, DIM>& gin_extents,
    const std::array<std::size_t, DIM>& gout_extents,
    const std::array<std::size_t, DIM>& in_topology,
    const std::array<std::size_t, DIM>& out_topology,
    const std::array<iType, 1>& axes,
    MPI_Comm comm0,
    MPI_Comm comm1) {

    auto src_map = KokkosFFT::Impl::index_sequence<std::size_t, DIM, 0>();
    // Get all relevant topologies
    auto all_topologies =
      get_shuffled_topologies(in_topology, out_topology, axes);

    std::size_t nb_topologies = all_topologies.size();
    if (nb_topologies == 1) {
      // E.g. {1, 1, P} + FFT ax=0
      BlockInfoType block;
      block.m_in_topology = in_topology;
      block.m_out_topology = out_topology;
      block.m_in_extents = in_extents;
      block.m_out_extents = out_extents;
      block.m_buffer_extents = {}; // unused
      block.m_block_type = BlockType::FFT;
      m_block_infos.push_back(block);
    } else if (nb_topologies == 2) {
      auto last_axis = axes.back();
      auto first_dim = in_topology.at(last_axis);
      if (first_dim != 1) {
        // E.g. {P, 1, 1} -> {1, 1, P} + FFT (ax=0)
        BlockInfoType block0;
        block0.m_in_topology = in_topology;
        block0.m_out_topology = out_topology;
        block0.m_in_extents = in_extents;
        block0.m_out_extents = get_next_extents(gin_extents, out_topology,
src_map, comm); block0.m_buffer_extents = get_buffer_extents<Layout>(
            gin_extents, in_topology, out_topology);
        block0.m_block_type = BlockType::Transpose;
        m_block_infos.push_back(block0);

        BlockInfoType block1;
        block1.m_in_topology = out_topology;
        block1.m_out_topology = out_topology;
        block1.m_in_extents = block0.m_out_extents;
        block1.m_out_extents = out_extents;
        block1.m_buffer_extents = {}; // unused
        block1.m_block_type = BlockType::FFT;
        m_block_infos.push_back(block1);
      } else {
        // E.g. {1, 1, P} + FFT (ax=0) -> {P, 1, 1}
        BlockInfoType block0;
        block0.m_in_topology = in_topology;
        block0.m_out_topology = in_topology;
        block0.m_in_extents = in_extents;
        block0.m_out_extents = get_next_extents(gout_extents, in_topology,
src_map, comm); block0.m_buffer_extents = {}; // unused block0.m_block_type =
BlockType::FFT; m_block_infos.push_back(block0);

        BlockInfoType block1;
        block1.m_in_topology = in_topology;
        block1.m_out_topology = out_topology;
        block1.m_in_extents = block0.m_out_extents;
        block1.m_out_extents = out_extents;
        block1.m_buffer_extents = get_buffer_extents<Layout>(
            gout_extents, in_topology, out_topology);
        block1.m_block_type = BlockType::Transpose;
        m_block_infos.push_back(block1);
      }
    } else if (nb_topologies == 3) {
      auto mid_topology = all_topologies.at(1);
      // E.g. {P, 1, 1} -> {1, 1, P} + FFT (ax=0) -> {P, 1, 1}
      BlockInfoType block0;
      block0.m_in_topology = in_topology;
      block0.m_out_topology = mid_topology;
      block0.m_in_extents = in_extents;
      block0.m_out_extents = get_next_extents(gin_extents, mid_topology,
src_map, comm); block0.m_buffer_extents = get_buffer_extents<Layout>(
          gin_extents, in_topology, mid_topology);
      block0.m_block_type = BlockType::Transpose;
      m_block_infos.push_back(block0);

      BlockInfoType block1;
      block1.m_in_topology = mid_topology;
      block1.m_out_topology = mid_topology;
      block1.m_in_extents = block0.m_out_extents;
      block1.m_out_extents = get_next_extents(gout_extents, mid_topology,
src_map, comm); block1.m_buffer_extents = {}; // unused block1.m_block_type =
BlockType::FFT; m_block_infos.push_back(block1);

      BlockInfoType block2;
      block2.m_in_topology = mid_topology;
      block2.m_out_topology = out_topology;
      block2.m_in_extents = block1.m_out_extents;
      block2.m_out_extents = out_extents;
      block2.m_buffer_extents = get_buffer_extents<Layout>(
          gout_extents, mid_topology, out_topology);
      block2.m_block_type = BlockType::Transpose;
      m_block_infos.push_back(block2);
    }
  }
};
*/

/*

/// \brief Get all slab block info for a given input and output topology
/// Batched case
/// 1. FFT // E.g. {1, 1, P}
/// 2. FFT + Transpose // E.g. {P, 1, 1} FFT (ax=2) -> {1, 1, P}
/// 3. Transpose + FFT // E.g. {P, 1, 1} -> {1, 1, P} FFT (ax=0)
/// 4. Transpose + FFT + Transpose // E.g. {1, 1, P} + FFT (ax=1) -> {P, 1, 1}
FFT (ax=2) -> {1, 1, P} template <typename Layout, typename iType, std::size_t
DIM> struct SlabBlockAnalysesInternal<Layout, iType, DIM, 2> { using
BlockInfoType = BlockInfo<DIM>; std::vector<BlockInfoType> m_block_infos;

  SlabBlockAnalysesInternal(const std::array<std::size_t, DIM>& in_extents,
    const std::array<std::size_t, DIM>& out_extents,
    const std::array<std::size_t, DIM>& gin_extents,
    const std::array<std::size_t, DIM>& gout_extents,
    const std::array<std::size_t, DIM>& in_topology,
    const std::array<std::size_t, DIM>& out_topology,
    const std::array<iType, 2>& axes) {

    auto src_map = KokkosFFT::Impl::index_sequence<std::size_t, 2, 0>();
    // Get all relevant topologies
    auto all_topologies =
      get_all_slab_topologies(in_topology, out_topology, axes);

    std::size_t nb_topologies = all_topologies.size();
    if (nb_topologies == 1) {
      // E.g. {1, 1, P} + FFT ax=0
      BlockInfoType block;
      block.m_in_topology = in_topology;
      block.m_out_topology = out_topology;
      block.m_in_extents = in_extents;
      block.m_out_extents = out_extents;
      block.m_buffer_extents = {}; // unused
      block.m_block_type = BlockType::FFT;
      m_block_infos.push_back(block);
    } else if (nb_topologies == 2) {
      auto last_axis = axes.back();
      auto first_dim = in_topology.at(last_axis);
      if (first_dim != 1) {
        // E.g. {P, 1, 1} -> {1, 1, P} + FFT (ax=0)
        BlockInfoType block0;
        block0.m_in_topology = in_topology;
        block0.m_out_topology = out_topology;
        block0.m_in_extents = in_extents;
        block0.m_out_extents = get_next_extents(gin_extents, out_topology,
src_map); block0.m_buffer_extents = get_buffer_extents<Layout>( gin_extents,
in_topology, out_topology); block0.m_block_type = BlockType::Transpose;
        m_block_infos.push_back(block0);

        BlockInfoType block1;
        block1.m_in_topology = out_topology;
        block1.m_out_topology = out_topology;
        block1.m_in_extents = block0.m_out_extents;
        block1.m_out_extents = out_extents;
        block1.m_buffer_extents = {}; // unused
        block1.m_block_type = BlockType::FFT;
        m_block_infos.push_back(block1);
      } else {
        // E.g. {1, 1, P} + FFT (ax=0) -> {P, 1, 1}
        BlockInfoType block0;
        block0.m_in_topology = in_topology;
        block0.m_out_topology = in_topology;
        block0.m_in_extents = in_extents;
        block0.m_out_extents = get_next_extents(gout_extents, in_topology,
src_map); block0.m_buffer_extents = {}; // unused block0.m_block_type =
BlockType::FFT; m_block_infos.push_back(block0);

        BlockInfoType block1;
        block1.m_in_topology = in_topology;
        block1.m_out_topology = out_topology;
        block1.m_in_extents = block0.m_out_extents;
        block1.m_out_extents = out_extents;
        block1.m_buffer_extents = get_buffer_extents<Layout>(
            gin_extents, in_topology, out_topology);
        block1.m_block_type = BlockType::Transpose;
        m_block_infos.push_back(block1);
      }
    } else if (nb_topologies == 3) {
      auto mid_topology = all_topologies.at(1);
      // E.g. {P, 1, 1} -> {1, 1, P} + FFT (ax=0) -> {P, 1, 1}
      BlockInfoType block0;
      block0.m_in_topology = in_topology;
      block0.m_out_topology = mid_topology;
      block0.m_in_extents = in_extents;
      block0.m_out_extents = get_next_extents(gin_extents, out_topology,
src_map); block0.m_buffer_extents = get_buffer_extents<Layout>( gin_extents,
in_topology, out_topology); block0.m_block_type = BlockType::Transpose;
      m_block_infos.push_back(block0);

      BlockInfoType block1;
      block1.m_in_topology = mid_topology;
      block1.m_out_topology = mid_topology;
      block1.m_in_extents = block0.m_out_extents;
      block1.m_out_extents = get_next_extents(gout_extents, out_topology,
src_map); block1.m_buffer_extents = {}; // unused block1.m_block_type =
BlockType::FFT; m_block_infos.push_back(block1);

      BlockInfoType block2;
      block2.m_in_topology = mid_topology;
      block2.m_out_topology = out_topology;
      block2.m_in_extents = block1.m_out_extents;
      block2.m_out_extents = out_extents;
      block2.m_buffer_extents = get_buffer_extents<Layout>(
          gout_extents, mid_topology, out_topology);
      block2.m_block_type = BlockType::Transpose;
      m_block_infos.push_back(block2);
    }
  }
};
*/

/*
/// \brief Get all slab block info for a given input and output topology
/// 1. FFT
/// 2. FFT + Transpose
/// 3. Transpose
///
/// \tparam InViewType
/// \tparam OutViewType
/// \tparam iType The index type used for the topology.
/// \tparam DIM The dimensionality of the topology.
/// \tparam FFT_DIM The dimensionality of the FFT axes.
///
template <typename InViewType, typename OutViewType, typename iType, std::size_t
FFT_DIM> struct SlabBlockAnalyses { static constexpr std::size_t DIM =
InViewType::rank(); using BlockInfoType = BlockInfo<DIM>;

  SlabBlockAnalyses(const InViewType& in, const OutViewType& out,
    const std::array<std::size_t, DIM>& in_topology,
    const std::array<std::size_t, DIM>& out_topology,
    const std::array<iType, FFT_DIM>& axes) {

  }
};
*/

#endif
