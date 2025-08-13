#ifndef SLAB_BLOCK_ANALYSES_HPP
#define SLAB_BLOCK_ANALYSES_HPP

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "MPI_Helper.hpp"
#include "Utils.hpp"
#include "Types.hpp"
#include "Topologies.hpp"
#include "Extents.hpp"

template <typename ValueType, typename Layout, typename iType, std::size_t DIM,
          std::size_t FFT_DIM>
struct SlabBlockAnalysesInternal;

/// \brief Get all slab block info for a given input and output topology
/// 1. FFT
/// E.g. {1, 1, P}
/// 2. FFT + Transpose
/// E.g. {P, 1, 1} FFT (ax=2) -> {1, 1, P}
/// 3. Transpose + FFT
/// E.g. {P, 1, 1} -> {1, 1, P} FFT (ax=0)
/// 4. Transpose + FFT + Transpose
/// E.g. {P, 1, 1} -> {1, 1, P} FFT (ax=0) -> {P, 1, 1}
template <typename ValueType, typename Layout, typename iType, std::size_t DIM>
struct SlabBlockAnalysesInternal<ValueType, Layout, iType, DIM, 1> {
  using BlockInfoType = BlockInfo<DIM>;
  using extents_type  = std::array<std::size_t, DIM>;
  std::vector<BlockInfoType> m_block_infos;
  std::size_t m_max_buffer_size;
  OperationType m_op_type;

  SlabBlockAnalysesInternal(
      const extents_type& in_extents, const extents_type& out_extents,
      const extents_type& gin_extents, const extents_type& gout_extents,
      const extents_type& in_topology, const extents_type& out_topology,
      const std::array<iType, 1>& axes, MPI_Comm comm = MPI_COMM_WORLD) {
    auto src_map = KokkosFFT::Impl::index_sequence<std::size_t, DIM, 0>();
    auto [map, map_inv] = get_map_axes<Layout, iType, DIM, 1>(axes);
    // Get all relevant topologies
    auto all_topologies =
        get_all_slab_topologies(in_topology, out_topology, axes);

    const std::size_t size_factor =
        KokkosFFT::Impl::is_real_v<ValueType> ? 1 : 2;

    std::vector<std::size_t> all_max_buffer_sizes;
    std::size_t nb_topologies = all_topologies.size();
    if (nb_topologies == 1) {
      // E.g. {1, 1, P} + FFT ax=0
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
        // E.g. {P, 1, 1} -> {1, 1, P} + FFT (ax=0)
        BlockInfoType block0;
        auto [in_axis0, out_axis0] = get_slab(in_topology, out_topology);
        block0.m_in_map            = src_map;
        block0.m_out_map  = get_dst_map<Layout, DIM>(src_map, out_axis0);
        block0.m_in_axis  = in_axis0;
        block0.m_out_axis = out_axis0;

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
        // E.g. {1, 1, P} + FFT (ax=0) -> {P, 1, 1}
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
        auto [in_axis1, out_axis1] = get_slab(in_topology, out_topology);
        block1.m_in_map            = map;
        block1.m_out_map           = src_map;
        block1.m_in_axis           = in_axis1;
        block1.m_out_axis          = out_axis1;

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
      m_op_type         = OperationType::TFT;
      auto mid_topology = all_topologies.at(1);
      // E.g. {P, 1, 1} -> {1, 1, P} + FFT (ax=0) -> {P, 1, 1}
      BlockInfoType block0;
      auto [in_axis0, out_axis0] = get_slab(in_topology, mid_topology);
      block0.m_in_map            = src_map;
      block0.m_out_map           = get_dst_map<Layout, DIM>(src_map, out_axis0);
      block0.m_in_axis           = in_axis0;
      block0.m_out_axis          = out_axis0;

      // Do we really need this?
      block0.m_in_topology  = in_topology;
      block0.m_out_topology = mid_topology;
      block0.m_in_extents   = in_extents;
      block0.m_out_extents =
          get_next_extents(gin_extents, mid_topology, block0.m_out_map, comm);
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
      block1.m_in_extents = block0.m_out_extents;
      block1.m_out_extents =
          get_next_extents(gout_extents, mid_topology, block0.m_out_map, comm);
      block1.m_block_type = BlockType::FFT;
      block1.m_axes =  // to_vector(get_mapped_axes(axes, block0.m_out_map));
          get_contiguous_axes<Layout, iType, DIM>(to_vector(axes));
      m_block_infos.push_back(block1);

      all_max_buffer_sizes.push_back(get_size(block1.m_out_extents) * 2);

      BlockInfoType block2;
      auto [in_axis2, out_axis2] = get_slab(mid_topology, out_topology);
      block2.m_in_map            = block0.m_out_map;
      block2.m_out_map           = src_map;
      block2.m_in_axis           = in_axis2;
      block2.m_out_axis          = out_axis2;
      block2.m_in_topology       = mid_topology;
      block2.m_out_topology      = out_topology;
      block2.m_in_extents        = block1.m_out_extents;
      block2.m_out_extents       = out_extents;
      block2.m_buffer_extents =
          get_buffer_extents<Layout>(gout_extents, mid_topology, out_topology);
      block2.m_block_type = BlockType::Transpose;
      block2.m_block_idx  = 1;
      m_block_infos.push_back(block2);

      all_max_buffer_sizes.push_back(get_size(block2.m_in_extents) * 2);
      all_max_buffer_sizes.push_back(get_size(block2.m_buffer_extents) * 2);
      all_max_buffer_sizes.push_back(get_size(block2.m_out_extents) * 2);
      m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
    }
  }
};

/// \brief Get all slab block info for a given input and output topology
/// 2D case
/// 1. FFT2 {ax=0,1}
/// 2. FFT2 {ax=0,1} + T
/// 3. T + FFT2 {ax=1,2}
/// E.g. {1,1,P} -> {P,1,1} + FFT2 {ax=1,2}
/// 3. FFT + T + FFT
/// E.g. {1, P} + FFT (ax=0) -> {P, 1} + FFT (ax=1)
/// 4. FFT + T + FFT + T
/// E.g. {1, P} + FFT (ax=0) -> {P, 1} + FFT (ax=1) -> {1, P}
/// 5. T + FFT + T + FFT
/// E.g. {1, P} -> {P, 1} + FFT (ax=1) -> {1, P} + FFT (ax=0)
/// 6. T + FFT + T + FFT + T
/// E.g. {1, P} -> {P, 1} + FFT (ax=1) -> {1, P} + FFT (ax=0) -> {P, 1}
template <typename ValueType, typename Layout, typename iType, std::size_t DIM>
struct SlabBlockAnalysesInternal<ValueType, Layout, iType, DIM, 2> {
  using BlockInfoType = BlockInfo<DIM>;
  using extents_type  = std::array<std::size_t, DIM>;
  std::vector<BlockInfoType> m_block_infos;
  std::size_t m_max_buffer_size;
  OperationType m_op_type;

  SlabBlockAnalysesInternal(
      const extents_type& in_extents, const extents_type& out_extents,
      const extents_type& gin_extents, const extents_type& gout_extents,
      const extents_type& in_topology, const extents_type& out_topology,
      const std::array<iType, 2>& axes, MPI_Comm comm = MPI_COMM_WORLD) {
    auto src_map = KokkosFFT::Impl::index_sequence<std::size_t, DIM, 0>();
    auto [map, map_inv] = get_map_axes<Layout, iType, DIM, 2>(axes);
    // Get all relevant topologies
    auto all_topologies =
        get_all_slab_topologies(in_topology, out_topology, axes);
    auto all_axes = decompose_axes(all_topologies, axes);

    const std::size_t size_factor =
        KokkosFFT::Impl::is_real_v<ValueType> ? 1 : 2;
    std::vector<std::size_t> all_max_buffer_sizes;

    std::size_t nb_topologies = all_topologies.size();
    if (nb_topologies == 1) {
      // 1. FFT2 with axes = {ax=0,1}
      // E.g. {1, 1, P} + FFT2 {ax=0,1}

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
      // 1. FTF
      // FFT2 with axes = {ax=0,1} for 2D View
      // E.g. {1,p} + FFT {ax=0} -> {P,1} + FFT {ax=1}
      // 2. FT
      // FFT2 with axes = {ax=0,1} for 3D View
      // E.g {1,1,P} + FFT2 {ax=0,1} -> {P,1,1}
      // 3. TF
      // FFT2 with axes = {ax=1,2} for 3D View
      // E.g {1,1,P} -> {P,1,1} + FFT2 {ax=1,2}
      auto axes0 = all_axes.at(0), axes1 = all_axes.at(1);
      // For 3D+ Views
      if (axes0.size() == 0) {
        // TF
        // E.g. {1,1,P} -> {P,1,1} + FFT2 {ax=1,2}
        m_op_type = OperationType::TF;

        BlockInfoType block0;
        auto [in_axis0, out_axis0] = get_slab(in_topology, out_topology);
        block0.m_in_map            = src_map;
        block0.m_out_map  = get_dst_map<Layout, iType, DIM>(src_map, axes1);
        block0.m_in_axis  = in_axis0;
        block0.m_out_axis = out_axis0;

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
        // FT or FTF
        if (axes1.size() == 0) {
          // FFT2 {ax=0,1} + T
          m_op_type = OperationType::FT;
        } else {
          // FFT + T + FFT
          m_op_type = OperationType::FTF;
        }

        BlockInfoType block0;
        block0.m_in_map     = map;
        block0.m_out_map    = map;
        block0.m_in_extents = get_mapped_extents(in_extents, map);
        block0.m_out_extents =
            get_next_extents(gout_extents, in_topology, map, comm);
        block0.m_block_type = BlockType::FFT;
        block0.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes0);
        m_block_infos.push_back(block0);
        all_max_buffer_sizes.push_back(get_size(block0.m_out_extents) * 2);

        BlockInfoType block1;
        auto [in_axis1, out_axis1] = get_slab(in_topology, out_topology);
        block1.m_in_map            = block0.m_out_map;
        block1.m_out_map           = axes1.size() == 0 ? src_map
                                                       : get_dst_map<Layout, DIM>(
                                                   block1.m_in_map, out_axis1);
        block1.m_in_axis           = in_axis1;
        block1.m_out_axis          = out_axis1;

        block1.m_in_topology  = in_topology;
        block1.m_out_topology = out_topology;
        block1.m_in_extents   = block0.m_out_extents;
        block1.m_out_extents  = get_next_extents(gout_extents, out_topology,
                                                 block1.m_out_map, comm);
        block1.m_buffer_extents =
            get_buffer_extents<Layout>(gout_extents, in_topology, out_topology);
        block1.m_block_type = BlockType::Transpose;
        m_block_infos.push_back(block1);

        // Data is always complex
        all_max_buffer_sizes.push_back(get_size(block1.m_in_extents) * 2);
        all_max_buffer_sizes.push_back(get_size(block1.m_buffer_extents) * 2);
        all_max_buffer_sizes.push_back(get_size(block1.m_out_extents) * 2);

        if (axes1.size() != 0) {
          BlockInfoType block2;
          block2.m_in_extents  = block1.m_out_extents;
          block2.m_out_extents = block1.m_out_extents;
          block2.m_block_type  = BlockType::FFT;
          block2.m_axes        = get_contiguous_axes<Layout, iType, DIM>(axes1);
          block2.m_in_map      = block1.m_out_map;
          block2.m_out_map     = block1.m_out_map;
          block2.m_block_idx   = 1;
          m_block_infos.push_back(block2);

          all_max_buffer_sizes.push_back(get_size(block2.m_out_extents) * 2);
        }

        m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
      }
    } else if (nb_topologies == 3) {
      /// 0. Transpose + FFT + Transpose
      /// E.g. {1, 1, P} -> {1, P, 1} + FFT2 {ax=0, 2} + {1, 1, P}
      /// 1. Transpose + FFT + Transpose + FFT
      /// E.g. {1, P} -> {1, P} with ax = {0,1}
      /// {1, P} -> {P, 1} + FFT (ax=1) -> {1, P} + FFT (ax=0)
      /// 2. FFT + Transpose + FFT + Transpose
      /// E.g. {1, P} -> {1, P} with ax = {1,0}
      /// {1, P} + FFT (ax=0) -> {P, 1} + FFT (ax=1) -> {1, P}

      auto mid_topology = all_topologies.at(1);

      auto axes0 = all_axes.at(0), axes1 = all_axes.at(1),
           axes2 = all_axes.at(2);

      if (axes0.size() == 0) {
        // TFT or TFTF
        if (axes2.size() == 0) {
          // 0. Transpose + FFT + Transpose
          m_op_type = OperationType::TFT;
        } else {
          // 1. Transpose + FFT + Transpose + FFT
          m_op_type = OperationType::TFTF;
        }

        BlockInfoType block0;
        auto [in_axis0, out_axis0] = get_slab(in_topology, mid_topology);
        block0.m_in_map            = src_map;
        block0.m_out_map  = get_dst_map<Layout, iType, DIM>(src_map, axes1);
        block0.m_in_axis  = in_axis0;
        block0.m_out_axis = out_axis0;

        block0.m_in_topology  = in_topology;
        block0.m_out_topology = mid_topology;
        block0.m_in_extents   = in_extents;
        block0.m_out_extents =
            get_next_extents(gin_extents, mid_topology, block0.m_out_map, comm);
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
        block1.m_out_extents = get_next_extents(gout_extents, mid_topology,
                                                block0.m_out_map, comm);
        block1.m_block_type  = BlockType::FFT;
        block1.m_axes        = get_contiguous_axes<Layout, iType, DIM>(axes1);
        m_block_infos.push_back(block1);

        all_max_buffer_sizes.push_back(get_size(block1.m_out_extents) * 2);

        BlockInfoType block2;
        auto [in_axis2, out_axis2] = get_slab(mid_topology, out_topology);
        block2.m_in_map            = block0.m_out_map;
        block2.m_out_map           = axes2.size() == 0 ? src_map
                                                       : get_dst_map<Layout, DIM>(
                                                   block0.m_out_map, out_axis2);
        block2.m_in_axis           = in_axis2;
        block2.m_out_axis          = out_axis2;

        block2.m_in_topology    = mid_topology;
        block2.m_out_topology   = out_topology;
        block2.m_in_extents     = block1.m_out_extents;
        block2.m_out_extents    = get_next_extents(gout_extents, out_topology,
                                                   block2.m_out_map, comm);
        block2.m_buffer_extents = get_buffer_extents<Layout>(
            gout_extents, mid_topology, out_topology);
        block2.m_block_type = BlockType::Transpose;
        block2.m_block_idx  = 1;
        m_block_infos.push_back(block2);

        all_max_buffer_sizes.push_back(get_size(block2.m_in_extents) * 2);
        all_max_buffer_sizes.push_back(get_size(block2.m_buffer_extents) * 2);
        all_max_buffer_sizes.push_back(get_size(block2.m_out_extents) * 2);

        if (axes2.size() != 0) {
          BlockInfoType block3;
          block3.m_in_extents  = block2.m_out_extents;
          block3.m_out_extents = block2.m_out_extents;
          block3.m_block_type  = BlockType::FFT;
          block3.m_axes        = get_contiguous_axes<Layout, iType, DIM>(axes2);
          block3.m_in_map      = block2.m_out_map;
          block3.m_out_map     = block2.m_out_map;
          block3.m_block_idx   = 1;
          m_block_infos.push_back(block3);

          all_max_buffer_sizes.push_back(get_size(block3.m_out_extents) * 2);
        }
        m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
      } else {
        // FTFT
        m_op_type = OperationType::FTFT;

        BlockInfoType block0;
        block0.m_in_map     = map;
        block0.m_out_map    = map;
        block0.m_in_extents = get_mapped_extents(in_extents, map);
        block0.m_out_extents =
            get_next_extents(gout_extents, in_topology, map, comm);
        block0.m_block_type = BlockType::FFT;
        block0.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes0);
        m_block_infos.push_back(block0);

        // Data is always complex
        all_max_buffer_sizes.push_back(get_size(block0.m_out_extents) * 2);

        BlockInfoType block1;
        auto [in_axis1, out_axis1] = get_slab(in_topology, mid_topology);
        block1.m_in_map            = map;
        block1.m_out_map  = get_dst_map<Layout, iType, DIM>(map, axes1);
        block1.m_in_axis  = in_axis1;
        block1.m_out_axis = out_axis1;

        block1.m_in_topology  = in_topology;
        block1.m_out_topology = mid_topology;
        block1.m_in_extents   = block0.m_out_extents;
        block1.m_out_extents  = get_next_extents(
            gout_extents, block1.m_out_topology, block1.m_out_map, comm);
        block1.m_buffer_extents = get_buffer_extents<Layout>(
            gout_extents, in_topology, block1.m_out_topology);
        block1.m_block_type = BlockType::Transpose;
        m_block_infos.push_back(block1);

        all_max_buffer_sizes.push_back(get_size(block1.m_in_extents) * 2);
        all_max_buffer_sizes.push_back(get_size(block1.m_buffer_extents) * 2);
        all_max_buffer_sizes.push_back(get_size(block1.m_out_extents) * 2);

        BlockInfoType block2;
        block2.m_in_extents  = block1.m_out_extents;
        block2.m_out_extents = get_next_extents(
            gout_extents, block1.m_out_topology, block1.m_out_map, comm);
        block2.m_block_type = BlockType::FFT;
        block2.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes1);
        block2.m_block_idx  = 1;
        m_block_infos.push_back(block2);

        all_max_buffer_sizes.push_back(get_size(block2.m_out_extents) * 2);

        BlockInfoType block3;
        auto [in_axis3, out_axis3] = get_slab(mid_topology, out_topology);
        block3.m_in_map            = block1.m_out_map;
        block3.m_out_map           = src_map;
        block3.m_in_axis           = in_axis3;
        block3.m_out_axis          = out_axis3;

        block3.m_in_topology    = mid_topology;
        block3.m_out_topology   = out_topology;
        block3.m_in_extents     = block2.m_out_extents;
        block3.m_out_extents    = out_extents;
        block3.m_buffer_extents = get_buffer_extents<Layout>(
            gout_extents, block3.m_in_topology, block3.m_out_topology);
        block3.m_block_type = BlockType::Transpose;
        block3.m_block_idx  = 1;
        m_block_infos.push_back(block3);

        all_max_buffer_sizes.push_back(get_size(block3.m_in_extents) * 2);
        all_max_buffer_sizes.push_back(get_size(block3.m_buffer_extents) * 2);
        all_max_buffer_sizes.push_back(get_size(block3.m_out_extents) * 2);

        m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
      }
    } else if (nb_topologies == 4) {
      // 1. Transpose + FFT + Transpose + FFT + Transpose
      // E.g. {1, P} -> {P, 1} + FFT (ax=1)
      // -> {1, P} + FFT (ax=0) -> {P, 1}
      m_op_type          = OperationType::TFTFT;
      auto mid_topology0 = all_topologies.at(1),
           mid_topology1 = all_topologies.at(2);
      auto axes0 = all_axes.at(0), axes1 = all_axes.at(1),
           axes2 = all_axes.at(2);

      BlockInfoType block0;
      auto [in_axis0, out_axis0] = get_slab(in_topology, mid_topology0);
      block0.m_in_map            = src_map;
      block0.m_out_map  = get_dst_map<Layout, iType, DIM>(src_map, axes1);
      block0.m_in_axis  = in_axis0;
      block0.m_out_axis = out_axis0;

      block0.m_in_topology  = in_topology;
      block0.m_out_topology = out_topology;
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
      block1.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes1);
      m_block_infos.push_back(block1);

      all_max_buffer_sizes.push_back(get_size(block1.m_out_extents) * 2);

      BlockInfoType block2;
      auto [in_axis2, out_axis2] = get_slab(mid_topology0, mid_topology1);
      block2.m_in_map            = block0.m_out_map;
      block2.m_out_map =
          get_dst_map<Layout, iType, DIM>(block2.m_in_map, axes2);
      block2.m_in_axis  = in_axis2;
      block2.m_out_axis = out_axis2;

      block2.m_in_topology  = mid_topology0;
      block2.m_out_topology = mid_topology1;
      block2.m_in_extents   = block1.m_out_extents;
      block2.m_out_extents =
          get_next_extents(gout_extents, mid_topology1, block2.m_out_map, comm);
      block2.m_buffer_extents = get_buffer_extents<Layout>(
          gout_extents, mid_topology0, mid_topology1);
      block2.m_block_type = BlockType::Transpose;
      block2.m_block_idx  = 1;
      m_block_infos.push_back(block2);

      all_max_buffer_sizes.push_back(get_size(block2.m_in_extents) * 2);
      all_max_buffer_sizes.push_back(get_size(block2.m_buffer_extents) * 2);
      all_max_buffer_sizes.push_back(get_size(block2.m_out_extents) * 2);

      BlockInfoType block3;
      block3.m_in_extents  = block2.m_out_extents;
      block3.m_out_extents = block2.m_out_extents;
      block3.m_block_type  = BlockType::FFT;
      block3.m_axes        = get_contiguous_axes<Layout, iType, DIM>(axes2);
      block3.m_in_map      = block2.m_out_map;
      block3.m_out_map     = block2.m_out_map;
      block3.m_block_idx   = 1;
      m_block_infos.push_back(block3);

      all_max_buffer_sizes.push_back(get_size(block3.m_out_extents) * 2);

      BlockInfoType block4;
      auto [in_axis4, out_axis4] = get_slab(mid_topology1, out_topology);
      block4.m_in_map            = block2.m_out_map;
      block4.m_out_map           = src_map;
      block4.m_in_axis           = in_axis4;
      block4.m_out_axis          = out_axis4;

      block4.m_in_topology  = mid_topology1;
      block4.m_out_topology = out_topology;
      block4.m_in_extents   = block3.m_out_extents;
      block4.m_out_extents  = out_extents;
      block4.m_buffer_extents =
          get_buffer_extents<Layout>(gout_extents, mid_topology1, out_topology);
      block4.m_block_type = BlockType::Transpose;
      block4.m_block_idx  = 2;
      m_block_infos.push_back(block4);

      all_max_buffer_sizes.push_back(get_size(block4.m_in_extents) * 2);
      all_max_buffer_sizes.push_back(get_size(block4.m_buffer_extents) * 2);
      all_max_buffer_sizes.push_back(get_size(block4.m_out_extents) * 2);

      m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
    }
  }
};

/// \brief Get all slab block info for a given input and output topology
/// 3D case
/// 1. FFT + T + FFT2
/// E.g. {1, 1, P} + FFT {ax=1} -> {1, P, 1} + FFT2 {ax=0,2}
/// 2. FFT2 + T + FFT
/// E.g. {P, 1, 1} + FFT2 {ax=1,2} -> {1, 1, P} + FFT {ax=0}
/// 3. FFT + T + FFT2 + T
/// E.g. {1, 1, P} + FFT {ax=1} -> {1, P, 1} + FFT2 {ax=0,2} -> {1, 1, P}
/// 4. FFT2 + T + FFT + T
/// E.g. {P, 1, 1} + FFT2 {ax=1,2} -> {1, 1, P} + FFT {ax=0} -> {P, 1, 1}
/// 5. T + FFT2 + T + FFT
/// E.g. {1, 1, P} -> {P, 1, 1} + FFT2 {ax=1,2} -> {1, P, 1} + FFT {ax=0}
/// 6. T + FFT2 + T + FFT
/// E.g. {1, 1, P} -> {P, 1, 1} + FFT2 {ax=1,2} -> {1, 1, P} + FFT {ax=0}
/// 7. T + FFT2 + T + FFT + T
/// E.g. {1, 1, P} -> {1, P, 1} + FFT2 {ax=0,2}
/// -> {P, 1, 1} + FFT {ax=1} -> {1, P, 1}
template <typename ValueType, typename Layout, typename iType, std::size_t DIM>
struct SlabBlockAnalysesInternal<ValueType, Layout, iType, DIM, 3> {
  using BlockInfoType = BlockInfo<DIM>;
  using extents_type  = std::array<std::size_t, DIM>;
  std::vector<BlockInfoType> m_block_infos;
  std::size_t m_max_buffer_size;
  OperationType m_op_type;

  SlabBlockAnalysesInternal(const std::array<std::size_t, DIM>& in_extents,
                            const std::array<std::size_t, DIM>& out_extents,
                            const std::array<std::size_t, DIM>& gin_extents,
                            const std::array<std::size_t, DIM>& gout_extents,
                            const std::array<std::size_t, DIM>& in_topology,
                            const std::array<std::size_t, DIM>& out_topology,
                            const std::array<iType, 3>& axes,
                            MPI_Comm comm = MPI_COMM_WORLD) {
    auto src_map = KokkosFFT::Impl::index_sequence<std::size_t, DIM, 0>();
    auto [map, map_inv] = get_map_axes<Layout, iType, DIM, 3>(axes);

    // Get all relevant topologies
    auto all_topologies =
        get_all_slab_topologies(in_topology, out_topology, axes);
    auto all_axes = decompose_axes(all_topologies, axes);

    const std::size_t size_factor =
        KokkosFFT::Impl::is_real_v<ValueType> ? 1 : 2;
    std::vector<std::size_t> all_max_buffer_sizes;

    std::size_t nb_topologies = all_topologies.size();
    if (nb_topologies == 1) {
      // 1. FFT3 with axes = {0,1,2}
      // E.g. {1, 1, 1, P} + FFT {ax=0,1,2}

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
      auto axes0 = all_axes.at(0), axes1 = all_axes.at(1);

      // 0. FFT3 + Transpose with axes = {0,2,1}
      // E.g. {1, 1, 1, P} + FFT {ax=0,2,1} -> {1, 1, P, 1}
      // or
      // 1. Transpose + FFT3 with axes = {0,1,2}
      // E.g. {1, 1, P, 1} + FFT {ax=0,1,2} -> {1, 1, 1, P}
      // or
      // 2. FFT + Transpose + FFT2 with axes = {0,2,1}
      // E.g. {1, 1, P} + FFT {ax=1} -> {1, P, 1} + FFT2 {ax=0,2}
      // or
      // 3. FFT2 + Transpose + FFT with axes = {1,0,2}
      // E.g. {P, 1, 1} + FFT2 {ax=1,2} -> {1, 1, P} + FFT {ax=0}

      if (axes0.size() == 0) {
        // T + FFT3
        m_op_type = OperationType::TF;
        BlockInfoType block0;
        auto [in_axis0, out_axis0] = get_slab(in_topology, out_topology);
        block0.m_in_map            = src_map;
        block0.m_out_map  = get_dst_map<Layout, iType, DIM>(src_map, axes1);
        block0.m_in_axis  = in_axis0;
        block0.m_out_axis = out_axis0;

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
        if (axes1.size() == 0) {
          // FFT3 {ax=0,1,2} + T
          m_op_type = OperationType::FT;
        } else {
          // FFT2 {ax=0,1} + T + FFT {ax=2}
          m_op_type = OperationType::FTF;
        }

        BlockInfoType block0;
        block0.m_in_map     = map;
        block0.m_out_map    = map;
        block0.m_in_extents = get_mapped_extents(in_extents, map);
        block0.m_out_extents =
            get_next_extents(gout_extents, in_topology, map, comm);
        block0.m_block_type = BlockType::FFT;
        block0.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes0);
        m_block_infos.push_back(block0);

        // Data is always complex
        all_max_buffer_sizes.push_back(get_size(block0.m_out_extents) * 2);

        BlockInfoType block1;
        auto [in_axis1, out_axis1] = get_slab(in_topology, out_topology);
        block1.m_in_map            = map;
        block1.m_out_map =
            axes1.size() == 0 ? src_map
                              : get_dst_map<Layout, iType, DIM>(src_map, axes1);
        block1.m_in_axis  = in_axis1;
        block1.m_out_axis = out_axis1;

        block1.m_in_topology  = in_topology;
        block1.m_out_topology = out_topology;
        block1.m_in_extents   = block0.m_out_extents;
        block1.m_out_extents  = get_next_extents(gout_extents, out_topology,
                                                 block1.m_out_map, comm);
        block1.m_buffer_extents =
            get_buffer_extents<Layout>(gout_extents, in_topology, out_topology);
        block1.m_block_type = BlockType::Transpose;
        m_block_infos.push_back(block1);

        all_max_buffer_sizes.push_back(get_size(block1.m_in_extents) * 2);
        all_max_buffer_sizes.push_back(get_size(block1.m_buffer_extents) * 2);
        all_max_buffer_sizes.push_back(get_size(block1.m_out_extents) * 2);

        if (axes1.size() != 0) {
          BlockInfoType block2;
          block2.m_in_extents  = block1.m_out_extents;
          block2.m_out_extents = get_next_extents(gout_extents, out_topology,
                                                  block1.m_out_map, comm);
          block2.m_block_type  = BlockType::FFT;
          block2.m_axes        = get_contiguous_axes<Layout, iType, DIM>(axes1);
          m_block_infos.push_back(block2);

          all_max_buffer_sizes.push_back(get_size(block2.m_out_extents) * 2);
        }
      }
      m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
    } else if (nb_topologies == 3) {
      /// 0. Transpose + FFT + Transpose
      /// E.g. {1, 1, 1, P} -> {P, 1, 1, 1} + FFT3 {ax=1,2,3} + {1, 1, 1, P}
      /// 1. Transpose + FFT + Transpose + FFT
      /// E.g. {1, 1, P} -> {P, 1, 1} + FFT2 {ax=1,2} -> {1, 1, P} + FFT {ax=0}
      /// 2. FFT + Transpose + FFT + Transpose
      /// E.g. {1, 1, P} -> {1, 1, P} with ax = {1, 2, 0}
      /// {1, 1, P} + FFT {ax=0} -> {P, 1, 1} + FFT2 {ax=1,2} -> {1, 1, P}
      auto mid_topology = all_topologies.at(1);

      auto axes0 = all_axes.at(0), axes1 = all_axes.at(1),
           axes2 = all_axes.at(2);
      if (axes0.size() == 0) {
        if (axes2.size() == 0) {
          // 0. Transpose + FFT + Transpose
          m_op_type = OperationType::TFT;
        } else {
          // 1. Transpose + FFT + Transpose + FFT
          m_op_type = OperationType::TFTF;
        }
        BlockInfoType block0;
        auto [in_axis0, out_axis0] = get_slab(in_topology, mid_topology);
        block0.m_in_map            = src_map;
        block0.m_out_map      = get_dst_map<Layout, iType, DIM>(src_map, axes1);
        block0.m_in_axis      = in_axis0;
        block0.m_out_axis     = out_axis0;
        block0.m_in_topology  = in_topology;
        block0.m_out_topology = out_topology;
        block0.m_in_extents   = in_extents;
        block0.m_out_extents =
            get_next_extents(gin_extents, mid_topology, block0.m_out_map, comm);
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
        block1.m_out_extents = get_next_extents(gout_extents, mid_topology,
                                                block0.m_out_map, comm);
        block1.m_block_type  = BlockType::FFT;
        block1.m_axes        = get_contiguous_axes<Layout, iType, DIM>(axes1);
        m_block_infos.push_back(block1);
        all_max_buffer_sizes.push_back(get_size(block1.m_out_extents) * 2);
        BlockInfoType block2;
        auto [in_axis2, out_axis2] = get_slab(mid_topology, out_topology);
        block2.m_in_map            = block0.m_out_map;
        block2.m_out_map           = get_dst_map<Layout, DIM>(
            block0.m_out_map, out_axis2);  // This needs to be fixed
        block2.m_in_axis        = in_axis2;
        block2.m_out_axis       = out_axis2;
        block2.m_in_topology    = mid_topology;
        block2.m_out_topology   = out_topology;
        block2.m_in_extents     = block1.m_out_extents;
        block2.m_out_extents    = get_next_extents(gout_extents, out_topology,
                                                   block2.m_out_map, comm);
        block2.m_buffer_extents = get_buffer_extents<Layout>(
            gout_extents, mid_topology, out_topology);
        block2.m_block_type = BlockType::Transpose;
        m_block_infos.push_back(block2);
        all_max_buffer_sizes.push_back(get_size(block2.m_in_extents) * 2);
        all_max_buffer_sizes.push_back(get_size(block2.m_buffer_extents) * 2);
        all_max_buffer_sizes.push_back(get_size(block2.m_out_extents) * 2);
        if (axes2.size() != 0) {
          BlockInfoType block3;
          block3.m_in_extents  = block2.m_out_extents;
          block3.m_out_extents = block2.m_out_extents;
          block3.m_block_type  = BlockType::FFT;
          block3.m_axes        = get_contiguous_axes<Layout, iType, DIM>(axes2);
          block3.m_in_map      = block2.m_out_map;
          block3.m_out_map     = block2.m_out_map;
          m_block_infos.push_back(block3);
          all_max_buffer_sizes.push_back(get_size(block3.m_out_extents) * 2);
        }
        m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
      } else {
        // 2. FFT + Transpose + FFT + Transpose
        /// E.g. {1, 1, P} -> {1, 1, P} with ax = {1, 2, 0}
        /// {1, 1, P} + FFT {ax=0} -> {P, 1, 1} + FFT2 {ax=1,2} -> {1, 1, P}
        m_op_type = OperationType::FTFT;

        BlockInfoType block0;
        block0.m_in_map     = map;
        block0.m_out_map    = map;
        block0.m_in_extents = get_mapped_extents(in_extents, map);
        block0.m_out_extents =
            get_next_extents(gout_extents, in_topology, map, comm);
        block0.m_block_type = BlockType::FFT;
        block0.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes0);
        m_block_infos.push_back(block0);

        // Data is always complex
        all_max_buffer_sizes.push_back(get_size(block0.m_out_extents) * 2);

        BlockInfoType block1;
        auto [in_axis1, out_axis1] = get_slab(in_topology, mid_topology);
        block1.m_in_map            = map;
        block1.m_out_map  = get_dst_map<Layout, iType, DIM>(map, axes1);
        block1.m_in_axis  = in_axis1;
        block1.m_out_axis = out_axis1;

        block1.m_in_topology  = in_topology;
        block1.m_out_topology = mid_topology;
        block1.m_in_extents   = block0.m_out_extents;
        block1.m_out_extents  = get_next_extents(
            gout_extents, block1.m_out_topology, block1.m_out_map, comm);
        block1.m_buffer_extents = get_buffer_extents<Layout>(
            gout_extents, in_topology, block1.m_out_topology);
        block1.m_block_type = BlockType::Transpose;
        m_block_infos.push_back(block1);

        all_max_buffer_sizes.push_back(get_size(block1.m_in_extents) * 2);
        all_max_buffer_sizes.push_back(get_size(block1.m_buffer_extents) * 2);
        all_max_buffer_sizes.push_back(get_size(block1.m_out_extents) * 2);

        BlockInfoType block2;
        block2.m_in_extents  = block1.m_out_extents;
        block2.m_out_extents = get_next_extents(
            gout_extents, block1.m_out_topology, block1.m_out_map, comm);
        block2.m_block_type = BlockType::FFT;
        block2.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes1);
        m_block_infos.push_back(block2);

        all_max_buffer_sizes.push_back(get_size(block2.m_out_extents) * 2);

        BlockInfoType block3;
        auto [in_axis3, out_axis3] = get_slab(mid_topology, out_topology);
        block3.m_in_map            = block1.m_out_map;
        block3.m_out_map           = src_map;
        block3.m_in_axis           = in_axis3;
        block3.m_out_axis          = out_axis3;

        block3.m_in_topology    = mid_topology;
        block3.m_out_topology   = out_topology;
        block3.m_in_extents     = block2.m_out_extents;
        block3.m_out_extents    = out_extents;
        block3.m_buffer_extents = get_buffer_extents<Layout>(
            gout_extents, block3.m_in_topology, block3.m_out_topology);
        block3.m_block_type = BlockType::Transpose;
        m_block_infos.push_back(block3);

        all_max_buffer_sizes.push_back(get_size(block3.m_in_extents) * 2);
        all_max_buffer_sizes.push_back(get_size(block3.m_buffer_extents) * 2);
        all_max_buffer_sizes.push_back(get_size(block3.m_out_extents) * 2);

        m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
      }
    } else if (nb_topologies == 4) {
      auto mid_topology0 = all_topologies.at(1),
           mid_topology1 = all_topologies.at(2);
      // 1. Transpose + FFT + Transpose + FFT + Transpose
      // E.g. {1, 1, P} -> {1, P, 1} + FFT2 ax = {0, 2}
      // -> {P, 1, 1} + FFT ax = {1} -> {1, P, 1}
      auto axes0 = all_axes.at(0), axes1 = all_axes.at(1),
           axes2 = all_axes.at(2);
      m_op_type  = OperationType::TFTFT;

      BlockInfoType block0;
      auto [in_axis0, out_axis0] = get_slab(in_topology, mid_topology0);
      block0.m_in_map            = src_map;
      block0.m_out_map  = get_dst_map<Layout, iType, DIM>(src_map, axes1);
      block0.m_in_axis  = in_axis0;
      block0.m_out_axis = out_axis0;

      block0.m_in_topology  = in_topology;
      block0.m_out_topology = out_topology;
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
      block1.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes1);
      m_block_infos.push_back(block1);

      all_max_buffer_sizes.push_back(get_size(block1.m_out_extents) * 2);

      BlockInfoType block2;
      auto [in_axis2, out_axis2] = get_slab(mid_topology0, mid_topology1);
      block2.m_in_map            = block0.m_out_map;
      block2.m_out_map =
          get_dst_map<Layout, iType, DIM>(block2.m_in_map, axes2);
      block2.m_in_axis  = in_axis2;
      block2.m_out_axis = out_axis2;

      block2.m_in_topology  = mid_topology0;
      block2.m_out_topology = mid_topology1;
      block2.m_in_extents   = block1.m_out_extents;
      block2.m_out_extents =
          get_next_extents(gout_extents, mid_topology1, block2.m_out_map, comm);
      block2.m_buffer_extents = get_buffer_extents<Layout>(
          gout_extents, mid_topology0, mid_topology1);
      block2.m_block_type = BlockType::Transpose;
      m_block_infos.push_back(block2);

      all_max_buffer_sizes.push_back(get_size(block2.m_in_extents) * 2);
      all_max_buffer_sizes.push_back(get_size(block2.m_buffer_extents) * 2);
      all_max_buffer_sizes.push_back(get_size(block2.m_out_extents) * 2);

      BlockInfoType block3;
      block3.m_in_extents  = block2.m_out_extents;
      block3.m_out_extents = block2.m_out_extents;
      block3.m_block_type  = BlockType::FFT;
      block3.m_axes        = get_contiguous_axes<Layout, iType, DIM>(axes2);
      block3.m_in_map      = block2.m_out_map;
      block3.m_out_map     = block2.m_out_map;
      m_block_infos.push_back(block3);

      all_max_buffer_sizes.push_back(get_size(block3.m_out_extents) * 2);

      BlockInfoType block4;
      auto [in_axis4, out_axis4] = get_slab(mid_topology1, out_topology);
      block4.m_in_map            = block2.m_out_map;
      block4.m_out_map           = src_map;
      block4.m_in_axis           = in_axis4;
      block4.m_out_axis          = out_axis4;

      block4.m_in_topology  = mid_topology1;
      block4.m_out_topology = out_topology;
      block4.m_in_extents   = block3.m_out_extents;
      block4.m_out_extents  = out_extents;
      block4.m_buffer_extents =
          get_buffer_extents<Layout>(gout_extents, mid_topology1, out_topology);
      block4.m_block_type = BlockType::Transpose;
      m_block_infos.push_back(block4);

      all_max_buffer_sizes.push_back(get_size(block4.m_in_extents) * 2);
      all_max_buffer_sizes.push_back(get_size(block4.m_buffer_extents) * 2);
      all_max_buffer_sizes.push_back(get_size(block4.m_out_extents) * 2);

      m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
    }
  }
};

#endif
