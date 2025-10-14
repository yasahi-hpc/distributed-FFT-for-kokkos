#ifndef KOKKOSFFT_DISTRIBUTED_PENCIL_BLOCK_ANALYSES_HPP
#define KOKKOSFFT_DISTRIBUTED_PENCIL_BLOCK_ANALYSES_HPP

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_MPI_Helper.hpp"
#include "KokkosFFT_Distributed_Utils.hpp"
#include "KokkosFFT_Distributed_Types.hpp"
#include "KokkosFFT_Distributed_Topologies.hpp"
#include "KokkosFFT_Distributed_Extents.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

template <typename ValueType, typename Layout, typename iType, std::size_t DIM,
          std::size_t FFT_DIM, typename InLayoutType, typename OutLayoutType>
struct PencilBlockAnalysesInternal;

/// \brief Get all pencil block info for a given input and output topology
/// 1. FFT // E.g. {1, P0, P1}
/// 2. FFT + Transpose // E.g. {1, P0, P1} FFT (ax=0) -> {P0, 1, P1}
/// 3. Transpose + FFT // E.g. {P0, 1, P1} -> {1, P0, P1} FFT (ax=0)
/// 4. Transpose + FFT + Transpose
/// E.g. {P0, 1, P1} -> {1, P0, P1} FFT (ax=0) -> {P0, 1, P1}
template <typename ValueType, typename Layout, typename iType, std::size_t DIM,
          typename InLayoutType, typename OutLayoutType>
struct PencilBlockAnalysesInternal<ValueType, Layout, iType, DIM, 1,
                                   InLayoutType, OutLayoutType> {
  using BlockInfoType     = BlockInfo<DIM>;
  using extents_type      = std::array<std::size_t, DIM>;
  using in_topology_type  = Topology<std::size_t, DIM, InLayoutType>;
  using out_topology_type = Topology<std::size_t, DIM, OutLayoutType>;
  std::vector<BlockInfoType> m_block_infos;
  std::size_t m_max_buffer_size;

  PencilBlockAnalysesInternal(const std::array<std::size_t, DIM>& in_extents,
                              const std::array<std::size_t, DIM>& out_extents,
                              const std::array<std::size_t, DIM>& gin_extents,
                              const std::array<std::size_t, DIM>& gout_extents,
                              const in_topology_type& in_topology,
                              const out_topology_type& out_topology,
                              const std::array<iType, 1>& axes, MPI_Comm comm) {
    auto src_map = KokkosFFT::Impl::index_sequence<std::size_t, DIM, 0>();
    auto [map, map_inv] = get_map_axes<Layout, iType, DIM, 1>(axes);

    // Get all relevant topologies
    auto [all_topologies, all_trans_axes, all_layouts] =
        get_all_pencil_topologies(in_topology, out_topology, axes);

    const std::size_t size_factor =
        KokkosFFT::Impl::is_real_v<ValueType> ? 1 : 2;

    std::vector<std::size_t> all_max_buffer_sizes;

    std::size_t nb_topologies = all_topologies.size();
    if (nb_topologies == 1) {
      // FFT batched
      // E.g. {1, Px, Py} + FFT ax=0
      BlockInfoType block;
      block.m_in_map      = map;
      block.m_out_map     = map;
      block.m_in_extents  = get_mapped_extents(in_extents, map);
      block.m_out_extents = get_mapped_extents(out_extents, map);
      block.m_block_type  = BlockType::FFT;
      block.m_axes = get_contiguous_axes<Layout, iType, DIM>(to_vector(axes));
      m_block_infos.push_back(block);

      // Data is always complex
      all_max_buffer_sizes.push_back(KokkosFFT::Impl::total_size(out_extents) *
                                     2);
      m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
    } else if (nb_topologies == 2) {
      auto last_axis = axes.back();
      auto first_dim = in_topology.at(last_axis);
      if (first_dim != 1) {
        // T + FFT
        // E.g. {Px, 1, Py} -> {1, Px, Py} + FFT (ax=0)
        BlockInfoType block0;
        auto [in_axis0, out_axis0] =
            get_pencil(in_topology.array(), out_topology.array());
        block0.m_in_map    = src_map;
        block0.m_out_map   = get_dst_map<Layout>(src_map, out_axis0);
        block0.m_in_axis   = in_axis0;
        block0.m_out_axis  = out_axis0;
        block0.m_comm_axis = all_trans_axes.at(0);

        block0.m_in_topology  = in_topology.array();
        block0.m_out_topology = out_topology.array();
        block0.m_in_extents   = in_extents;
        block0.m_out_extents =
            get_next_extents(gin_extents, out_topology, block0.m_out_map, comm);
        block0.m_buffer_extents = get_buffer_extents<Layout>(
            gin_extents, in_topology.array(), out_topology.array());
        block0.m_block_type = BlockType::Transpose;

        m_block_infos.push_back(block0);

        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_in_extents) * size_factor);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_buffer_extents) * size_factor);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_out_extents) * size_factor);

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

        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block1.m_out_extents) * 2);
        m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
      } else {
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
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_out_extents) * 2);

        BlockInfoType block1;
        auto [in_axis1, out_axis1] =
            get_pencil(in_topology.array(), out_topology.array());
        block1.m_in_map    = map;
        block1.m_out_map   = src_map;
        block1.m_in_axis   = in_axis1;
        block1.m_out_axis  = out_axis1;
        block1.m_comm_axis = all_trans_axes.at(0);

        block1.m_in_extents     = block0.m_out_extents;
        block1.m_out_extents    = out_extents;
        block1.m_buffer_extents = get_buffer_extents<Layout>(
            gout_extents, in_topology.array(), out_topology.array());
        block1.m_block_type = BlockType::Transpose;
        m_block_infos.push_back(block1);

        // Data is always complex
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block1.m_in_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block1.m_buffer_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block1.m_out_extents) * 2);
        m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
      }
    } else if (nb_topologies == 3) {
      // T + FFT + T
      // E.g. {1,Px,Py} -> {Px,1,Py} + FFT (ax=1) -> {1,Px,Py}

      auto mid_topo = all_topologies.at(1);

      BlockInfoType block0;
      auto [in_axis0, out_axis0] = get_pencil(in_topology.array(), mid_topo);
      block0.m_in_map            = src_map;
      block0.m_out_map           = get_dst_map<Layout>(src_map, out_axis0);
      block0.m_in_axis           = in_axis0;
      block0.m_out_axis          = out_axis0;
      block0.m_comm_axis         = all_trans_axes.at(0);
      block0.m_in_extents        = in_extents;
      bool is_layout_right0      = all_layouts.at(1);
      block0.m_out_extents       = get_next_extents(
          gin_extents, mid_topo, block0.m_out_map, comm, is_layout_right0);
      block0.m_buffer_extents = get_buffer_extents<Layout>(
          gin_extents, in_topology.array(), mid_topo);

      block0.m_block_type = BlockType::Transpose;
      m_block_infos.push_back(block0);

      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block0.m_in_extents) * size_factor);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block0.m_buffer_extents) * size_factor);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block0.m_out_extents) * size_factor);

      BlockInfoType block1;
      block1.m_in_extents  = block0.m_out_extents;
      block1.m_out_extents = get_next_extents(
          gout_extents, mid_topo, block0.m_out_map, comm, is_layout_right0);

      block1.m_block_type = BlockType::FFT;
      block1.m_axes = get_contiguous_axes<Layout, iType, DIM>(to_vector(axes));
      m_block_infos.push_back(block1);

      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block1.m_out_extents) * 2);

      BlockInfoType block2;
      auto [in_axis2, out_axis2] = get_pencil(mid_topo, out_topology.array());
      block2.m_in_map            = block0.m_out_map;
      block2.m_out_map           = src_map;
      block2.m_in_axis           = in_axis2;
      block2.m_out_axis          = out_axis2;
      block2.m_comm_axis         = all_trans_axes.at(1);
      block2.m_in_topology       = mid_topo;
      block2.m_out_topology      = out_topology.array();
      block2.m_in_extents        = block1.m_out_extents;
      block2.m_out_extents       = out_extents;
      block2.m_buffer_extents    = get_buffer_extents<Layout>(
          gout_extents, mid_topo, out_topology.array());
      block2.m_block_type = BlockType::Transpose;
      block2.m_block_idx  = 1;
      m_block_infos.push_back(block2);

      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block2.m_in_extents) * 2);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block2.m_buffer_extents) * 2);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block2.m_out_extents) * 2);
      m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
    } else if (nb_topologies == 4) {
      auto mid_topo0 = all_topologies.at(1), mid_topo1 = all_topologies.at(2);
      // E.g. {1, Px, Py} -> {Py, Px, 1} + FFT (ax=2) -> {1, Px, Py}
      // -> {Px, 1, Py}
      BlockInfoType block0;
      auto [in_axis0, out_axis0] = get_pencil(in_topology.array(), mid_topo0);
      block0.m_in_map            = src_map;
      block0.m_out_map           = get_dst_map<Layout>(src_map, out_axis0);
      block0.m_in_axis           = in_axis0;
      block0.m_out_axis          = out_axis0;
      block0.m_comm_axis         = all_trans_axes.at(0);
      block0.m_in_extents        = in_extents;
      bool is_layout_right0      = all_layouts.at(1);
      block0.m_out_extents       = get_next_extents(
          gin_extents, mid_topo0, block0.m_out_map, comm, is_layout_right0);
      block0.m_buffer_extents = get_buffer_extents<Layout>(
          gin_extents, in_topology.array(), mid_topo0);

      block0.m_block_type = BlockType::Transpose;
      m_block_infos.push_back(block0);

      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block0.m_in_extents) * size_factor);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block0.m_buffer_extents) * size_factor);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block0.m_out_extents) * size_factor);

      BlockInfoType block1;
      block1.m_in_extents  = block0.m_out_extents;
      block1.m_out_extents = get_next_extents(
          gout_extents, mid_topo0, block0.m_out_map, comm, is_layout_right0);

      block1.m_block_type = BlockType::FFT;
      block1.m_axes = get_contiguous_axes<Layout, iType, DIM>(to_vector(axes));
      m_block_infos.push_back(block1);

      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block1.m_out_extents) * 2);

      BlockInfoType block2;
      auto [in_axis2, out_axis2] = get_pencil(mid_topo0, mid_topo1);
      block2.m_in_map            = block0.m_out_map;
      block2.m_out_map           = block2.m_in_map;
      block2.m_in_axis           = in_axis2;
      block2.m_out_axis          = out_axis2;
      block2.m_comm_axis         = all_trans_axes.at(1);
      block2.m_in_extents        = block1.m_out_extents;
      bool is_layout_right2      = all_layouts.at(2);
      block2.m_out_extents       = get_next_extents(
          gout_extents, mid_topo1, block2.m_out_map, comm, is_layout_right2);
      block2.m_buffer_extents =
          get_buffer_extents<Layout>(gout_extents, mid_topo0, mid_topo1);
      block2.m_block_type = BlockType::Transpose;
      block2.m_block_idx  = 1;
      m_block_infos.push_back(block2);

      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block2.m_in_extents) * 2);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block2.m_buffer_extents) * 2);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block2.m_out_extents) * 2);

      BlockInfoType block3;
      auto [in_axis3, out_axis3] = get_pencil(mid_topo1, out_topology.array());
      block3.m_in_map            = block2.m_out_map;
      block3.m_out_map           = src_map;
      block3.m_in_axis           = in_axis3;
      block3.m_out_axis          = out_axis3;
      block3.m_comm_axis         = all_trans_axes.at(2);
      block3.m_in_extents        = block2.m_out_extents;
      block3.m_out_extents       = out_extents;
      block3.m_buffer_extents    = get_buffer_extents<Layout>(
          gout_extents, mid_topo1, out_topology.array());
      block3.m_block_type = BlockType::Transpose;
      block3.m_block_idx  = 2;
      m_block_infos.push_back(block3);

      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block3.m_in_extents) * 2);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block3.m_buffer_extents) * 2);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block3.m_out_extents) * 2);

      m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
    }
  }
};

/// \brief Get all pencil block info for a given input and output topology
/// 2D case
/// 1. FFT + T + FFT // E.g. {1, P0, P1} + FFT (ax=0) ->
/// {P0, 1, P1} + FFT (ax=1)
/// 2. FFT + T + FFT + T // E.g. {1, P0, P1} + FFT (ax=0)
/// -> {P0, 1, P1} + FFT (ax=1) -> {1, P0, P1}
/// 3. T + FFT + T + FFT // E.g. {1, P0, P1} -> {P0, 1, P1} + FFT (ax=1)
/// -> {1, P0, P1} + FFT (ax=0)
/// 4. T + FFT + T + FFT + T // E.g. {1, P0, P1} -> {P0, 1, P1} + FFT (ax=1)
/// -> {1, P0, P1} + FFT (ax=0) -> {P0, P1, 1}
template <typename ValueType, typename Layout, typename iType, std::size_t DIM,
          typename InLayoutType, typename OutLayoutType>
struct PencilBlockAnalysesInternal<ValueType, Layout, iType, DIM, 2,
                                   InLayoutType, OutLayoutType> {
  using BlockInfoType     = BlockInfo<DIM>;
  using extents_type      = std::array<std::size_t, DIM>;
  using in_topology_type  = Topology<std::size_t, DIM, InLayoutType>;
  using out_topology_type = Topology<std::size_t, DIM, OutLayoutType>;
  std::vector<BlockInfoType> m_block_infos;
  std::size_t m_max_buffer_size;

  PencilBlockAnalysesInternal(const std::array<std::size_t, DIM>& in_extents,
                              const std::array<std::size_t, DIM>& out_extents,
                              const std::array<std::size_t, DIM>& gin_extents,
                              const std::array<std::size_t, DIM>& gout_extents,
                              const in_topology_type& in_topology,
                              const out_topology_type& out_topology,
                              const std::array<iType, 2>& axes, MPI_Comm comm) {
    auto src_map = KokkosFFT::Impl::index_sequence<std::size_t, DIM, 0>();
    auto [map, map_inv] = get_map_axes<Layout, iType, DIM, 2>(axes);

    // Get all relevant topologies
    auto [all_topologies, all_trans_axes, all_layouts] =
        get_all_pencil_topologies(in_topology, out_topology, axes);
    auto all_axes = decompose_axes(all_topologies, axes);

    const std::size_t size_factor =
        KokkosFFT::Impl::is_real_v<ValueType> ? 1 : 2;

    std::vector<std::size_t> all_max_buffer_sizes;

    std::size_t nb_topologies = all_topologies.size();
    if (nb_topologies == 1) {
      // FFT batched
      // E.g. {1, 1, Px, Py} + FFT2 {ax=0, 1}
      BlockInfoType block;
      block.m_in_map      = map;
      block.m_out_map     = map;
      block.m_in_extents  = get_mapped_extents(in_extents, map);
      block.m_out_extents = get_mapped_extents(out_extents, map);
      block.m_block_type  = BlockType::FFT;
      block.m_axes = get_contiguous_axes<Layout, iType, DIM>(to_vector(axes));
      m_block_infos.push_back(block);

      // Data is always complex
      all_max_buffer_sizes.push_back(KokkosFFT::Impl::total_size(out_extents) *
                                     2);
      m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
    } else if (nb_topologies == 2) {
      // 0. T + FFT
      // 1. FFT + T
      // 2. FFT + T + FFT
      auto axes0 = all_axes.at(0), axes1 = all_axes.at(1);
      if (axes0.size() == 0) {
        // T + FFT
        // E.g. {Px, 1, Py} -> {1, Px, Py} + FFT (ax=0)
        BlockInfoType block0;
        auto [in_axis0, out_axis0] =
            get_pencil(in_topology.array(), out_topology.array());
        block0.m_in_map    = src_map;
        block0.m_out_map   = get_dst_map<Layout>(src_map, axes1);
        block0.m_in_axis   = in_axis0;
        block0.m_out_axis  = out_axis0;
        block0.m_comm_axis = all_trans_axes.at(0);

        block0.m_in_topology  = in_topology.array();
        block0.m_out_topology = out_topology.array();
        block0.m_in_extents   = in_extents;
        block0.m_out_extents =
            get_next_extents(gin_extents, out_topology, block0.m_out_map, comm);
        block0.m_buffer_extents = get_buffer_extents<Layout>(
            gin_extents, in_topology.array(), out_topology.array());
        block0.m_block_type = BlockType::Transpose;

        m_block_infos.push_back(block0);

        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_in_extents) * size_factor);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_buffer_extents) * size_factor);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_out_extents) * size_factor);

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

        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block1.m_out_extents) * 2);
        m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
      } else {
        // FFT + T
        // FFT + T + FFT
        // E.g. {1, Px, Py} + FFT (ax=0) -> {Px, 1, Py}
        BlockInfoType block0;
        block0.m_in_map     = map;
        block0.m_out_map    = map;
        block0.m_in_extents = get_mapped_extents(in_extents, map);
        block0.m_out_extents =
            get_next_extents(gout_extents, in_topology, map, comm);
        block0.m_block_type = BlockType::FFT;
        block0.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes0);
        m_block_infos.push_back(block0);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_out_extents) * 2);

        if (axes1.size() == 0) {
          BlockInfoType block1;
          auto [in_axis1, out_axis1] =
              get_pencil(in_topology.array(), out_topology.array());
          block1.m_in_map    = map;
          block1.m_out_map   = src_map;
          block1.m_in_axis   = in_axis1;
          block1.m_out_axis  = out_axis1;
          block1.m_comm_axis = all_trans_axes.at(0);

          block1.m_in_extents     = block0.m_out_extents;
          block1.m_out_extents    = out_extents;
          block1.m_buffer_extents = get_buffer_extents<Layout>(
              gout_extents, in_topology.array(), out_topology.array());
          block1.m_block_type = BlockType::Transpose;
          m_block_infos.push_back(block1);

          // Data is always complex
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block1.m_in_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block1.m_buffer_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block1.m_out_extents) * 2);

        } else {
          // FFT + T + FFT
          BlockInfoType block1;
          auto [in_axis1, out_axis1] =
              get_pencil(in_topology.array(), out_topology.array());
          block1.m_in_map    = block0.m_out_map;
          block1.m_out_map   = get_dst_map<Layout>(block1.m_in_map, out_axis1);
          block1.m_in_axis   = in_axis1;
          block1.m_out_axis  = out_axis1;
          block1.m_comm_axis = all_trans_axes.at(0);

          block1.m_in_extents     = block0.m_out_extents;
          block1.m_out_extents    = get_next_extents(gout_extents, out_topology,
                                                     block1.m_out_map, comm);
          block1.m_buffer_extents = get_buffer_extents<Layout>(
              gout_extents, in_topology.array(), out_topology.array());
          block1.m_block_type = BlockType::Transpose;
          m_block_infos.push_back(block1);

          // Data is always complex
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block1.m_in_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block1.m_buffer_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block1.m_out_extents) * 2);

          BlockInfoType block2;
          block2.m_in_map      = block1.m_out_map;
          block2.m_out_map     = block1.m_out_map;
          block2.m_in_extents  = block1.m_out_extents;
          block2.m_out_extents = block2.m_in_extents;
          block2.m_block_type  = BlockType::FFT;
          block2.m_block_idx   = 1;
          block2.m_axes        = get_contiguous_axes<Layout, iType, DIM>(axes1);
          m_block_infos.push_back(block2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block2.m_out_extents) * 2);
        }

        m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
      }
    } else if (nb_topologies == 3) {
      // 0. FFT + T + T
      // 1. FFT + T + FFT + T
      // 2. FFT + T + T + FFT
      // 3. T + FFT + T
      // 4. T + FFT + T + FFT

      // T + FFT + T
      // E.g. {1,Px,Py} -> {Px,1,Py} + FFT (ax=1) -> {1,Px,Py}

      auto mid_topo = all_topologies.at(1);
      auto axes0 = all_axes.at(0), axes1 = all_axes.at(1),
           axes2 = all_axes.at(2);

      if (axes0.size() == 0) {
        BlockInfoType block0;
        auto [in_axis0, out_axis0] = get_pencil(in_topology.array(), mid_topo);
        block0.m_in_map            = src_map;
        block0.m_out_map           = get_dst_map<Layout>(src_map, axes1);
        block0.m_in_axis           = in_axis0;
        block0.m_out_axis          = out_axis0;
        block0.m_comm_axis         = all_trans_axes.at(0);
        block0.m_in_extents        = in_extents;
        bool is_layout_right0      = all_layouts.at(1);
        block0.m_out_extents       = get_next_extents(
            gin_extents, mid_topo, block0.m_out_map, comm, is_layout_right0);
        block0.m_buffer_extents = get_buffer_extents<Layout>(
            gin_extents, in_topology.array(), mid_topo);
        block0.m_block_type = BlockType::Transpose;
        m_block_infos.push_back(block0);

        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_in_extents) * size_factor);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_buffer_extents) * size_factor);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_out_extents) * size_factor);

        BlockInfoType block1;
        block1.m_in_map      = block0.m_out_map;
        block1.m_out_map     = block0.m_out_map;
        block1.m_in_extents  = block0.m_out_extents;
        block1.m_out_extents = get_next_extents(
            gout_extents, mid_topo, block1.m_out_map, comm, is_layout_right0);
        block1.m_block_type = BlockType::FFT;
        block1.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes1);
        m_block_infos.push_back(block1);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block1.m_out_extents) * 2);

        if (axes2.size() == 0) {
          // TFT
          BlockInfoType block2;
          auto [in_axis2, out_axis2] =
              get_pencil(mid_topo, out_topology.array());
          block2.m_in_map         = block1.m_out_map;
          block2.m_out_map        = src_map;
          block2.m_in_axis        = in_axis2;
          block2.m_out_axis       = out_axis2;
          block2.m_comm_axis      = all_trans_axes.at(1);
          block2.m_in_topology    = mid_topo;
          block2.m_out_topology   = out_topology.array();
          block2.m_in_extents     = block1.m_out_extents;
          block2.m_out_extents    = out_extents;
          block2.m_buffer_extents = get_buffer_extents<Layout>(
              gout_extents, mid_topo, out_topology.array());
          block2.m_block_type = BlockType::Transpose;
          block2.m_block_idx  = 1;
          m_block_infos.push_back(block2);

          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block2.m_in_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block2.m_buffer_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block2.m_out_extents) * 2);
        } else {
          BlockInfoType block2;
          auto [in_axis2, out_axis2] =
              get_pencil(mid_topo, out_topology.array());
          block2.m_in_map    = block1.m_out_map;
          block2.m_out_map   = get_dst_map<Layout>(block2.m_in_map, out_axis2);
          block2.m_in_axis   = in_axis2;
          block2.m_out_axis  = out_axis2;
          block2.m_comm_axis = all_trans_axes.at(1);
          block2.m_in_topology    = mid_topo;
          block2.m_out_topology   = out_topology.array();
          block2.m_in_extents     = block1.m_out_extents;
          block2.m_out_extents    = get_next_extents(gout_extents, out_topology,
                                                     block2.m_out_map, comm);
          block2.m_buffer_extents = get_buffer_extents<Layout>(
              gout_extents, mid_topo, out_topology.array());
          block2.m_block_type = BlockType::Transpose;
          block2.m_block_idx  = 1;
          m_block_infos.push_back(block2);

          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block2.m_in_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block2.m_buffer_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block2.m_out_extents) * 2);

          BlockInfoType block3;
          block3.m_in_map      = block2.m_out_map;
          block3.m_out_map     = block2.m_out_map;
          block3.m_in_extents  = block2.m_out_extents;
          block3.m_out_extents = block3.m_in_extents;
          block3.m_block_type  = BlockType::FFT;
          block3.m_block_idx   = 1;
          block3.m_axes        = get_contiguous_axes<Layout, iType, DIM>(axes2);
          m_block_infos.push_back(block3);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block3.m_out_extents) * 2);
        }  // if (axes2.size() == 0)
      } else {
        // 0. FFT + T + T
        // 1. FFT + T + FFT + T
        // 2. FFT + T + T + FFT
        BlockInfoType block0;
        block0.m_in_map     = map;
        block0.m_out_map    = map;
        block0.m_in_extents = get_mapped_extents(in_extents, map);
        block0.m_out_extents =
            get_next_extents(gout_extents, in_topology, map, comm);
        block0.m_block_type = BlockType::FFT;
        block0.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes0);
        m_block_infos.push_back(block0);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_out_extents) * 2);

        BlockInfoType block1;
        auto [in_axis1, out_axis1] = get_pencil(in_topology.array(), mid_topo);
        block1.m_in_map            = map;
        block1.m_out_map   = get_dst_map<Layout>(block0.m_out_map, out_axis1);
        block1.m_in_axis   = in_axis1;
        block1.m_out_axis  = out_axis1;
        block1.m_comm_axis = all_trans_axes.at(0);
        bool is_layout_right1 = all_layouts.at(1);

        block1.m_in_extents  = block0.m_out_extents;
        block1.m_out_extents = get_next_extents(
            gout_extents, mid_topo, block1.m_out_map, comm, is_layout_right1);

        block1.m_buffer_extents = get_buffer_extents<Layout>(
            gout_extents, in_topology.array(), mid_topo);
        block1.m_block_type = BlockType::Transpose;
        m_block_infos.push_back(block1);

        // Data is always complex
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block1.m_in_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block1.m_buffer_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block1.m_out_extents) * 2);

        if (axes1.size() == 0) {
          if (axes2.size() == 0) {
            // FTT
            BlockInfoType block2;
            auto [in_axis2, out_axis2] =
                get_pencil(mid_topo, out_topology.array());
            block2.m_in_map         = block1.m_out_map;
            block2.m_out_map        = src_map;
            block2.m_in_axis        = in_axis2;
            block2.m_out_axis       = out_axis2;
            block2.m_comm_axis      = all_trans_axes.at(1);
            block2.m_in_topology    = mid_topo;
            block2.m_out_topology   = out_topology.array();
            block2.m_in_extents     = block1.m_out_extents;
            block2.m_out_extents    = out_extents;
            block2.m_buffer_extents = get_buffer_extents<Layout>(
                gout_extents, mid_topo, out_topology.array());
            block2.m_block_type = BlockType::Transpose;
            block2.m_block_idx  = 1;
            m_block_infos.push_back(block2);

            all_max_buffer_sizes.push_back(
                KokkosFFT::Impl::total_size(block2.m_in_extents) * 2);
            all_max_buffer_sizes.push_back(
                KokkosFFT::Impl::total_size(block2.m_buffer_extents) * 2);
            all_max_buffer_sizes.push_back(
                KokkosFFT::Impl::total_size(block2.m_out_extents) * 2);
          } else {
            // FTTF
            BlockInfoType block2;
            auto [in_axis2, out_axis2] =
                get_pencil(mid_topo, out_topology.array());
            block2.m_in_map   = block1.m_out_map;
            block2.m_out_map  = get_dst_map<Layout>(block2.m_in_map, out_axis2);
            block2.m_in_axis  = in_axis2;
            block2.m_out_axis = out_axis2;
            block2.m_comm_axis    = all_trans_axes.at(1);
            block2.m_in_topology  = mid_topo;
            block2.m_out_topology = out_topology.array();
            block2.m_in_extents   = block1.m_out_extents;
            block2.m_out_extents  = get_next_extents(gout_extents, out_topology,
                                                     block2.m_out_map, comm);
            block2.m_buffer_extents = get_buffer_extents<Layout>(
                gout_extents, mid_topo, out_topology.array());
            block2.m_block_type = BlockType::Transpose;
            block2.m_block_idx  = 1;
            m_block_infos.push_back(block2);

            all_max_buffer_sizes.push_back(
                KokkosFFT::Impl::total_size(block2.m_in_extents) * 2);
            all_max_buffer_sizes.push_back(
                KokkosFFT::Impl::total_size(block2.m_buffer_extents) * 2);
            all_max_buffer_sizes.push_back(
                KokkosFFT::Impl::total_size(block2.m_out_extents) * 2);

            BlockInfoType block3;
            block3.m_in_map      = block2.m_out_map;
            block3.m_out_map     = block2.m_out_map;
            block3.m_in_extents  = block2.m_out_extents;
            block3.m_out_extents = block3.m_in_extents;
            block3.m_block_type  = BlockType::FFT;
            block3.m_block_idx   = 1;
            block3.m_axes = get_contiguous_axes<Layout, iType, DIM>(axes2);
            m_block_infos.push_back(block3);
            all_max_buffer_sizes.push_back(
                KokkosFFT::Impl::total_size(block3.m_out_extents) * 2);
          }  // if (axes2.size() == 0)
        } else {
          // FTFT
          BlockInfoType block2;
          block2.m_in_extents  = block1.m_out_extents;
          block2.m_out_extents = get_next_extents(
              gout_extents, mid_topo, block1.m_out_map, comm, is_layout_right1);
          block2.m_block_type = BlockType::FFT;
          block2.m_block_idx  = 1;
          block2.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes1);
          m_block_infos.push_back(block2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block2.m_out_extents) * 2);

          BlockInfoType block3;
          auto [in_axis3, out_axis3] =
              get_pencil(mid_topo, out_topology.array());
          block3.m_in_map      = block1.m_out_map;
          block3.m_out_map     = src_map;
          block3.m_in_axis     = in_axis3;
          block3.m_out_axis    = out_axis3;
          block3.m_comm_axis   = all_trans_axes.at(1);
          block3.m_in_extents  = block2.m_out_extents;
          block3.m_out_extents = out_extents;

          block3.m_buffer_extents = get_buffer_extents<Layout>(
              gout_extents, mid_topo, out_topology.array());
          block3.m_block_type = BlockType::Transpose;
          block3.m_block_idx  = 1;
          m_block_infos.push_back(block3);

          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block3.m_in_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block3.m_buffer_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block3.m_out_extents) * 2);
        }
      }  // if (axes0.size() == 0)
      m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
    } else if (nb_topologies == 4) {
      // 0. FFT + T + FFT + T + T
      // 1. FFT + T + T + FFT + T
      // 2. T + FFT + T + FFT + T
      // 3. T + FFT + T + T + FFT
      // 4. T + FFT + T + T

      auto mid_topo0 = all_topologies.at(1), mid_topo1 = all_topologies.at(2);

      auto axes0 = all_axes.at(0), axes1 = all_axes.at(1),
           axes2 = all_axes.at(2), axes3 = all_axes.at(3);

      if (axes0.size() == 0) {
        BlockInfoType block0;
        auto [in_axis0, out_axis0] = get_pencil(in_topology.array(), mid_topo0);
        block0.m_in_map            = src_map;
        block0.m_out_map           = get_dst_map<Layout>(src_map, axes1);
        block0.m_in_axis           = in_axis0;
        block0.m_out_axis          = out_axis0;
        block0.m_comm_axis         = all_trans_axes.at(0);
        block0.m_in_extents        = in_extents;

        bool is_layout_right0 = all_layouts.at(1);
        block0.m_out_extents  = get_next_extents(
            gin_extents, mid_topo0, block0.m_out_map, comm, is_layout_right0);
        block0.m_buffer_extents = get_buffer_extents<Layout>(
            gin_extents, in_topology.array(), mid_topo0);

        block0.m_block_type = BlockType::Transpose;
        m_block_infos.push_back(block0);

        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_in_extents) * size_factor);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_buffer_extents) * size_factor);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_out_extents) * size_factor);

        BlockInfoType block1;
        block1.m_in_extents  = block0.m_out_extents;
        block1.m_out_extents = get_next_extents(
            gout_extents, mid_topo0, block0.m_out_map, comm, is_layout_right0);

        block1.m_block_type = BlockType::FFT;
        block1.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes1);
        m_block_infos.push_back(block1);

        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block1.m_out_extents) * 2);

        BlockInfoType block2;
        auto [in_axis2, out_axis2] = get_pencil(mid_topo0, mid_topo1);
        block2.m_in_map            = block0.m_out_map;
        block2.m_out_map      = get_dst_map<Layout>(block2.m_in_map, out_axis2);
        block2.m_in_axis      = in_axis2;
        block2.m_out_axis     = out_axis2;
        block2.m_comm_axis    = all_trans_axes.at(1);
        bool is_layout_right2 = all_layouts.at(2);

        block2.m_in_extents  = block1.m_out_extents;
        block2.m_out_extents = get_next_extents(
            gout_extents, mid_topo1, block2.m_out_map, comm, is_layout_right2);
        block2.m_buffer_extents =
            get_buffer_extents<Layout>(gout_extents, mid_topo0, mid_topo1);
        block2.m_block_type = BlockType::Transpose;
        block2.m_block_idx  = 1;
        m_block_infos.push_back(block2);

        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block2.m_in_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block2.m_buffer_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block2.m_out_extents) * 2);

        if (axes2.size() == 0) {
          if (axes3.size() == 0) {
            // TFTT
            BlockInfoType block3;
            auto [in_axis3, out_axis3] =
                get_pencil(mid_topo1, out_topology.array());
            block3.m_in_map    = block2.m_out_map;
            block3.m_out_map   = src_map;
            block3.m_in_axis   = in_axis3;
            block3.m_out_axis  = out_axis3;
            block3.m_comm_axis = all_trans_axes.at(2);

            block3.m_in_extents  = block2.m_out_extents;
            block3.m_out_extents = out_extents;

            block3.m_buffer_extents = get_buffer_extents<Layout>(
                gout_extents, mid_topo1, out_topology.array());
            block3.m_block_type = BlockType::Transpose;
            block3.m_block_idx  = 2;
            m_block_infos.push_back(block3);

            all_max_buffer_sizes.push_back(
                KokkosFFT::Impl::total_size(block3.m_in_extents) * 2);
            all_max_buffer_sizes.push_back(
                KokkosFFT::Impl::total_size(block3.m_buffer_extents) * 2);
            all_max_buffer_sizes.push_back(
                KokkosFFT::Impl::total_size(block3.m_out_extents) * 2);
          } else {
            // TFTTF
            BlockInfoType block3;
            auto [in_axis3, out_axis3] =
                get_pencil(mid_topo1, out_topology.array());
            block3.m_in_map   = block2.m_out_map;
            block3.m_out_map  = get_dst_map<Layout>(block3.m_in_map, out_axis3);
            block3.m_in_axis  = in_axis3;
            block3.m_out_axis = out_axis3;
            block3.m_comm_axis   = all_trans_axes.at(2);
            block3.m_in_extents  = block2.m_out_extents;
            block3.m_out_extents = get_next_extents(gout_extents, out_topology,
                                                    block3.m_out_map, comm);

            block3.m_buffer_extents = get_buffer_extents<Layout>(
                gout_extents, mid_topo1, out_topology.array());
            block3.m_block_type = BlockType::Transpose;
            block3.m_block_idx  = 2;
            m_block_infos.push_back(block3);

            all_max_buffer_sizes.push_back(
                KokkosFFT::Impl::total_size(block3.m_in_extents) * 2);
            all_max_buffer_sizes.push_back(
                KokkosFFT::Impl::total_size(block3.m_buffer_extents) * 2);
            all_max_buffer_sizes.push_back(
                KokkosFFT::Impl::total_size(block3.m_out_extents) * 2);

            BlockInfoType block4;
            block4.m_in_map      = block3.m_out_map;
            block4.m_out_map     = block4.m_in_map;
            block4.m_in_extents  = block3.m_out_extents;
            block4.m_out_extents = block4.m_in_extents;
            block4.m_block_type  = BlockType::FFT;
            block4.m_block_idx   = 1;
            block4.m_axes = get_contiguous_axes<Layout, iType, DIM>(axes3);
            m_block_infos.push_back(block4);

            all_max_buffer_sizes.push_back(
                KokkosFFT::Impl::total_size(block4.m_out_extents) * 2);
          }
        } else {
          // T + FFT + T + FFT + T
          BlockInfoType block3;
          block3.m_in_extents = block2.m_out_extents;
          block3.m_out_extents =
              get_next_extents(gout_extents, mid_topo1, block2.m_out_map, comm,
                               is_layout_right2);
          block3.m_block_type = BlockType::FFT;
          block3.m_block_idx  = 1;
          block3.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes2);
          m_block_infos.push_back(block3);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block3.m_out_extents) * 2);

          BlockInfoType block4;
          auto [in_axis4, out_axis4] =
              get_pencil(mid_topo1, out_topology.array());
          block4.m_in_map         = block2.m_out_map;
          block4.m_out_map        = src_map;
          block4.m_in_axis        = in_axis4;
          block4.m_out_axis       = out_axis4;
          block4.m_comm_axis      = all_trans_axes.at(2);
          block4.m_in_extents     = block3.m_out_extents;
          block4.m_out_extents    = out_extents;
          block4.m_buffer_extents = get_buffer_extents<Layout>(
              gout_extents, mid_topo1, out_topology.array());
          block4.m_block_type = BlockType::Transpose;
          block4.m_block_idx  = 2;
          m_block_infos.push_back(block4);

          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block4.m_in_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block4.m_buffer_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block4.m_out_extents) * 2);
        }  // if (axes2.size() == 0)
      } else {
        BlockInfoType block0;
        block0.m_in_map     = map;
        block0.m_out_map    = map;
        block0.m_in_extents = get_mapped_extents(in_extents, map);
        block0.m_out_extents =
            get_next_extents(gout_extents, in_topology, map, comm);
        block0.m_block_type = BlockType::FFT;
        block0.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes0);
        m_block_infos.push_back(block0);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_out_extents) * 2);

        BlockInfoType block1;
        auto [in_axis1, out_axis1] = get_pencil(in_topology.array(), mid_topo0);
        block1.m_in_map            = map;
        block1.m_out_map   = get_dst_map<Layout>(block0.m_out_map, out_axis1);
        block1.m_in_axis   = in_axis1;
        block1.m_out_axis  = out_axis1;
        block1.m_comm_axis = all_trans_axes.at(0);
        bool is_layout_right1 = all_layouts.at(1);

        block1.m_in_extents  = block0.m_out_extents;
        block1.m_out_extents = get_next_extents(
            gout_extents, mid_topo0, block1.m_out_map, comm, is_layout_right1);

        block1.m_buffer_extents = get_buffer_extents<Layout>(
            gout_extents, in_topology.array(), mid_topo0);
        block1.m_block_type = BlockType::Transpose;
        m_block_infos.push_back(block1);

        // Data is always complex
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block1.m_in_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block1.m_buffer_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block1.m_out_extents) * 2);
        if (axes1.size() == 0) {
          // FTTFT
          BlockInfoType block2;
          auto [in_axis2, out_axis2] = get_pencil(mid_topo0, mid_topo1);
          block2.m_in_map            = block0.m_out_map;
          block2.m_out_map   = get_dst_map<Layout>(block2.m_in_map, out_axis2);
          block2.m_in_axis   = in_axis2;
          block2.m_out_axis  = out_axis2;
          block2.m_comm_axis = all_trans_axes.at(1);
          bool is_layout_right2 = all_layouts.at(2);

          block2.m_in_extents = block1.m_out_extents;
          block2.m_out_extents =
              get_next_extents(gout_extents, mid_topo1, block2.m_out_map, comm,
                               is_layout_right2);
          block2.m_buffer_extents =
              get_buffer_extents<Layout>(gout_extents, mid_topo0, mid_topo1);
          block2.m_block_type = BlockType::Transpose;
          block2.m_block_idx  = 1;
          m_block_infos.push_back(block2);

          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block2.m_in_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block2.m_buffer_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block2.m_out_extents) * 2);

          BlockInfoType block3;
          block3.m_in_map     = block2.m_out_map;
          block3.m_out_map    = block2.m_out_map;
          block3.m_in_extents = block2.m_out_extents;
          block3.m_out_extents =
              get_next_extents(gout_extents, mid_topo1, block3.m_out_map, comm,
                               is_layout_right2);
          block3.m_block_type = BlockType::FFT;
          block3.m_block_idx  = 1;
          block3.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes2);
          m_block_infos.push_back(block3);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block3.m_out_extents) * 2);
        } else {
          // FTFTT
          BlockInfoType block2;
          block2.m_in_extents = block1.m_out_extents;
          block2.m_out_extents =
              get_next_extents(gout_extents, mid_topo0, block1.m_out_map, comm,
                               is_layout_right1);
          block2.m_block_type = BlockType::FFT;
          block2.m_block_idx  = 1;
          block2.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes1);
          m_block_infos.push_back(block2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block2.m_out_extents) * 2);

          BlockInfoType block3;
          auto [in_axis3, out_axis3] = get_pencil(mid_topo0, mid_topo1);
          block3.m_in_map            = block1.m_out_map;
          block3.m_out_map   = get_dst_map<Layout>(block3.m_in_map, out_axis3);
          block3.m_in_axis   = in_axis3;
          block3.m_out_axis  = out_axis3;
          block3.m_comm_axis = all_trans_axes.at(1);
          bool is_layout_right3 = all_layouts.at(2);

          block3.m_in_extents = block2.m_out_extents;
          block3.m_out_extents =
              get_next_extents(gout_extents, mid_topo1, block3.m_out_map, comm,
                               is_layout_right3);

          block3.m_buffer_extents =
              get_buffer_extents<Layout>(gout_extents, mid_topo0, mid_topo1);
          block3.m_block_type = BlockType::Transpose;
          block3.m_block_idx  = 1;
          m_block_infos.push_back(block3);

          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block3.m_in_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block3.m_buffer_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block3.m_out_extents) * 2);
        }  // if (axes1.size() == 0)
        BlockInfoType block4;
        auto [in_axis4, out_axis4] =
            get_pencil(mid_topo1, out_topology.array());
        block4.m_in_map         = m_block_infos.back().m_out_map;
        block4.m_out_map        = src_map;
        block4.m_in_axis        = in_axis4;
        block4.m_out_axis       = out_axis4;
        block4.m_comm_axis      = all_trans_axes.at(2);
        block4.m_in_extents     = m_block_infos.back().m_out_extents;
        block4.m_out_extents    = out_extents;
        block4.m_buffer_extents = get_buffer_extents<Layout>(
            gout_extents, mid_topo1, out_topology.array());
        block4.m_block_type = BlockType::Transpose;
        block4.m_block_idx  = 2;
        m_block_infos.push_back(block4);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block4.m_in_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block4.m_buffer_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block4.m_out_extents) * 2);
      }  // if (axes0.size() == 0)

      m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
    } else if (nb_topologies == 5) {
      auto mid_topo0 = all_topologies.at(1), mid_topo1 = all_topologies.at(2),
           mid_topo2 = all_topologies.at(3);

      auto axes0 = all_axes.at(0), axes1 = all_axes.at(1),
           axes2 = all_axes.at(2), axes3 = all_axes.at(3),
           axes4 = all_axes.at(4);

      // 0. T + FFT + T + FFT + T + T
      BlockInfoType block0;
      auto [in_axis0, out_axis0] = get_pencil(in_topology.array(), mid_topo0);
      block0.m_in_map            = src_map;
      block0.m_out_map           = get_dst_map<Layout>(src_map, out_axis0);
      block0.m_in_axis           = in_axis0;
      block0.m_out_axis          = out_axis0;
      block0.m_comm_axis         = all_trans_axes.at(0);
      block0.m_in_extents        = in_extents;
      bool is_layout_right0      = all_layouts.at(1);
      block0.m_out_extents       = get_next_extents(
          gin_extents, mid_topo0, block0.m_out_map, comm, is_layout_right0);
      block0.m_buffer_extents = get_buffer_extents<Layout>(
          gin_extents, in_topology.array(), mid_topo0);
      block0.m_block_type = BlockType::Transpose;
      m_block_infos.push_back(block0);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block0.m_in_extents) * size_factor);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block0.m_buffer_extents) * size_factor);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block0.m_out_extents) * size_factor);

      BlockInfoType block1;
      block1.m_in_extents  = block0.m_out_extents;
      block1.m_out_extents = get_next_extents(
          gout_extents, mid_topo0, block0.m_out_map, comm, is_layout_right0);
      block1.m_block_type = BlockType::FFT;
      block1.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes1);
      m_block_infos.push_back(block1);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block1.m_out_extents) * 2);

      BlockInfoType block2;
      auto [in_axis2, out_axis2] = get_pencil(mid_topo0, mid_topo1);
      block2.m_in_map            = block0.m_out_map;
      block2.m_out_map      = get_dst_map<Layout>(block2.m_in_map, out_axis2);
      block2.m_in_axis      = in_axis2;
      block2.m_out_axis     = out_axis2;
      block2.m_comm_axis    = all_trans_axes.at(1);
      bool is_layout_right2 = all_layouts.at(2);
      block2.m_in_extents   = block1.m_out_extents;
      block2.m_out_extents  = get_next_extents(
          gout_extents, mid_topo1, block2.m_out_map, comm, is_layout_right2);
      block2.m_buffer_extents =
          get_buffer_extents<Layout>(gout_extents, mid_topo0, mid_topo1);
      block2.m_block_type = BlockType::Transpose;
      block2.m_block_idx  = 1;
      m_block_infos.push_back(block2);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block2.m_in_extents) * 2);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block2.m_buffer_extents) * 2);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block2.m_out_extents) * 2);

      BlockInfoType block3;
      block3.m_in_map      = block2.m_out_map;
      block3.m_out_map     = block2.m_out_map;
      block3.m_in_extents  = block2.m_out_extents;
      block3.m_out_extents = get_next_extents(
          gout_extents, mid_topo1, block3.m_out_map, comm, is_layout_right2);
      block3.m_block_type = BlockType::FFT;
      block3.m_block_idx  = 1;
      block3.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes2);
      m_block_infos.push_back(block3);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block3.m_out_extents) * 2);

      BlockInfoType block4;
      auto [in_axis4, out_axis4] = get_pencil(mid_topo1, mid_topo2);
      block4.m_in_map            = block2.m_out_map;
      block4.m_out_map      = get_dst_map<Layout>(block4.m_in_map, out_axis4);
      block4.m_in_axis      = in_axis4;
      block4.m_out_axis     = out_axis4;
      block4.m_comm_axis    = all_trans_axes.at(2);
      bool is_layout_right4 = all_layouts.at(3);
      block4.m_in_extents   = block3.m_out_extents;
      block4.m_out_extents  = get_next_extents(
          gout_extents, mid_topo2, block4.m_out_map, comm, is_layout_right4);
      block4.m_buffer_extents =
          get_buffer_extents<Layout>(gout_extents, mid_topo1, mid_topo2);
      block4.m_block_type = BlockType::Transpose;
      block4.m_block_idx  = 2;
      m_block_infos.push_back(block4);

      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block4.m_in_extents) * 2);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block4.m_buffer_extents) * 2);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block4.m_out_extents) * 2);

      BlockInfoType block5;
      auto [in_axis5, out_axis5] = get_pencil(mid_topo2, out_topology.array());
      block5.m_in_map            = block4.m_out_map;
      block5.m_out_map           = src_map;
      block5.m_in_axis           = in_axis5;
      block5.m_out_axis          = out_axis5;
      block5.m_comm_axis         = all_trans_axes.at(3);
      block5.m_in_extents        = block4.m_out_extents;
      block5.m_out_extents       = out_extents;
      block5.m_buffer_extents    = get_buffer_extents<Layout>(
          gout_extents, mid_topo2, out_topology.array());
      block5.m_block_type = BlockType::Transpose;
      block5.m_block_idx  = 3;
      m_block_infos.push_back(block5);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block5.m_in_extents) * 2);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block5.m_buffer_extents) * 2);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block5.m_out_extents) * 2);

      m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
    }  // if (nb_topologies == 1)
  }
};

/// \brief Get all pencil block info for a given input and output topology
/// 3D case
/// 1. FFT + T + FFT // E.g. {1, P0, P1} + FFT (ax=0) ->
/// {P0, 1, P1} + FFT (ax=1)
/// 2. FFT + T + FFT + T // E.g. {1, P0, P1} + FFT (ax=0)
/// -> {P0, 1, P1} + FFT (ax=1) -> {1, P0, P1}
/// 3. T + FFT + T + FFT // E.g. {1, P0, P1} -> {P0, 1, P1} + FFT (ax=1)
/// -> {1, P0, P1} + FFT (ax=0)
/// 4. T + FFT + T + FFT + T // E.g. {1, P0, P1} -> {P0, 1, P1} + FFT (ax=1)
/// -> {1, P0, P1} + FFT (ax=0) -> {P0, P1, 1}
template <typename ValueType, typename Layout, typename iType, std::size_t DIM,
          typename InLayoutType, typename OutLayoutType>
struct PencilBlockAnalysesInternal<ValueType, Layout, iType, DIM, 3,
                                   InLayoutType, OutLayoutType> {
  using BlockInfoType     = BlockInfo<DIM>;
  using extents_type      = std::array<std::size_t, DIM>;
  using in_topology_type  = Topology<std::size_t, DIM, InLayoutType>;
  using out_topology_type = Topology<std::size_t, DIM, OutLayoutType>;
  std::vector<BlockInfoType> m_block_infos;
  std::size_t m_max_buffer_size;

  PencilBlockAnalysesInternal(const std::array<std::size_t, DIM>& in_extents,
                              const std::array<std::size_t, DIM>& out_extents,
                              const std::array<std::size_t, DIM>& gin_extents,
                              const std::array<std::size_t, DIM>& gout_extents,
                              const in_topology_type& in_topology,
                              const out_topology_type& out_topology,
                              const std::array<iType, 3>& axes, MPI_Comm comm) {
    auto src_map = KokkosFFT::Impl::index_sequence<std::size_t, DIM, 0>();
    auto [map, map_inv] = get_map_axes<Layout, iType, DIM, 3>(axes);

    // Get all relevant topologies
    auto [all_topologies, all_trans_axes, all_layouts] =
        get_all_pencil_topologies(in_topology, out_topology, axes);
    auto all_axes = decompose_axes(all_topologies, axes);

    const std::size_t size_factor =
        KokkosFFT::Impl::is_real_v<ValueType> ? 1 : 2;

    std::vector<std::size_t> all_max_buffer_sizes;

    std::size_t nb_topologies = all_topologies.size();
    if (nb_topologies == 1) {
      // 0. FFT batched
      // E.g. {1, 1, 1, Px, Py} + FFT3 {ax=0, 1, 2}
      BlockInfoType block;
      block.m_in_map      = map;
      block.m_out_map     = map;
      block.m_in_extents  = get_mapped_extents(in_extents, map);
      block.m_out_extents = get_mapped_extents(out_extents, map);
      block.m_block_type  = BlockType::FFT;
      block.m_axes = get_contiguous_axes<Layout, iType, DIM>(to_vector(axes));
      m_block_infos.push_back(block);

      // Data is always complex
      all_max_buffer_sizes.push_back(KokkosFFT::Impl::total_size(out_extents) *
                                     2);
      m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
    } else if (nb_topologies == 2) {
      // 0. T + FFT
      // 1. FFT + T
      // 2. FFT + T + FFT

      auto axes0 = all_axes.at(0), axes1 = all_axes.at(1);

      if (axes0.size() == 0) {
        // T + FFT
        // E.g. {Px, 1, Py} -> {1, Px, Py} + FFT (ax=0)
        BlockInfoType block0;
        auto [in_axis0, out_axis0] =
            get_pencil(in_topology.array(), out_topology.array());
        block0.m_in_map    = src_map;
        block0.m_out_map   = get_dst_map<Layout>(src_map, axes1);
        block0.m_in_axis   = in_axis0;
        block0.m_out_axis  = out_axis0;
        block0.m_comm_axis = all_trans_axes.at(0);

        block0.m_in_topology  = in_topology.array();
        block0.m_out_topology = out_topology.array();
        block0.m_in_extents   = in_extents;
        block0.m_out_extents =
            get_next_extents(gin_extents, out_topology, block0.m_out_map, comm);
        block0.m_buffer_extents = get_buffer_extents<Layout>(
            gin_extents, in_topology.array(), out_topology.array());
        block0.m_block_type = BlockType::Transpose;

        m_block_infos.push_back(block0);

        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_in_extents) * size_factor);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_buffer_extents) * size_factor);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_out_extents) * size_factor);

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

        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block1.m_out_extents) * 2);
        m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
      } else {
        // FFT + T
        // E.g. {1, Px, Py} + FFT (ax=0) -> {Px, 1, Py}
        BlockInfoType block0;
        block0.m_in_map     = map;
        block0.m_out_map    = map;
        block0.m_in_extents = get_mapped_extents(in_extents, map);
        block0.m_out_extents =
            get_next_extents(gout_extents, in_topology, map, comm);
        block0.m_block_type = BlockType::FFT;
        block0.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes0);
        m_block_infos.push_back(block0);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_out_extents) * 2);

        if (axes1.size() == 0) {
          BlockInfoType block1;
          auto [in_axis1, out_axis1] =
              get_pencil(in_topology.array(), out_topology.array());
          block1.m_in_map    = map;
          block1.m_out_map   = src_map;
          block1.m_in_axis   = in_axis1;
          block1.m_out_axis  = out_axis1;
          block1.m_comm_axis = all_trans_axes.at(0);

          block1.m_in_extents     = block0.m_out_extents;
          block1.m_out_extents    = out_extents;
          block1.m_buffer_extents = get_buffer_extents<Layout>(
              gout_extents, in_topology.array(), out_topology.array());
          block1.m_block_type = BlockType::Transpose;
          m_block_infos.push_back(block1);

          // Data is always complex
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block1.m_in_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block1.m_buffer_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block1.m_out_extents) * 2);

        } else {
          // FFT + T + FFT
          BlockInfoType block1;
          auto [in_axis1, out_axis1] =
              get_pencil(in_topology.array(), out_topology.array());
          block1.m_in_map    = block0.m_out_map;
          block1.m_out_map   = get_dst_map<Layout>(block1.m_in_map, out_axis1);
          block1.m_in_axis   = in_axis1;
          block1.m_out_axis  = out_axis1;
          block1.m_comm_axis = all_trans_axes.at(0);

          block1.m_in_extents     = block0.m_out_extents;
          block1.m_out_extents    = get_next_extents(gout_extents, out_topology,
                                                     block1.m_out_map, comm);
          block1.m_buffer_extents = get_buffer_extents<Layout>(
              gout_extents, in_topology.array(), out_topology.array());
          block1.m_block_type = BlockType::Transpose;
          m_block_infos.push_back(block1);

          // Data is always complex
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block1.m_in_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block1.m_buffer_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block1.m_out_extents) * 2);

          BlockInfoType block2;
          block2.m_in_map      = block1.m_out_map;
          block2.m_out_map     = block1.m_out_map;
          block2.m_in_extents  = block1.m_out_extents;
          block2.m_out_extents = block2.m_in_extents;
          block2.m_block_type  = BlockType::FFT;
          block2.m_block_idx   = 1;
          block2.m_axes        = get_contiguous_axes<Layout, iType, DIM>(axes1);
          m_block_infos.push_back(block2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block2.m_out_extents) * 2);
        }
        m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
      }
    } else if (nb_topologies == 3) {
      // 0. FFT + T + FFT + T + FFT
      // 1. FFT + T + FFT + T
      // e.g. (n0, n1, n2/px, n3/py) -> (n0/px, n1, n2, n3/py) -> (n0, n1,
      // n2/px, n3/py)
      // 2. T + F + T + F
      // e.g. (n0, n1, n2/px, n3/py) -> (n0/px, n1, n2, n3/py) -> (n0, n1,
      // n2/px, n3/py)

      auto mid_topo = all_topologies.at(1);

      auto axes0 = all_axes.at(0), axes1 = all_axes.at(1),
           axes2 = all_axes.at(2);

      if (axes0.size() == 0) {
        BlockInfoType block0;
        auto [in_axis0, out_axis0] = get_pencil(in_topology.array(), mid_topo);
        block0.m_in_map            = src_map;
        block0.m_out_map           = get_dst_map<Layout>(src_map, axes1);
        block0.m_in_axis           = in_axis0;
        block0.m_out_axis          = out_axis0;
        block0.m_comm_axis         = all_trans_axes.at(0);
        block0.m_in_extents        = in_extents;

        bool is_layout_right0 = all_layouts.at(1);
        block0.m_out_extents  = get_next_extents(
            gin_extents, mid_topo, block0.m_out_map, comm, is_layout_right0);
        block0.m_buffer_extents = get_buffer_extents<Layout>(
            gin_extents, in_topology.array(), mid_topo);

        block0.m_block_type = BlockType::Transpose;
        m_block_infos.push_back(block0);

        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_in_extents) * size_factor);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_buffer_extents) * size_factor);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_out_extents) * size_factor);

        BlockInfoType block1;
        block1.m_in_extents  = block0.m_out_extents;
        block1.m_out_extents = get_next_extents(
            gout_extents, mid_topo, block0.m_out_map, comm, is_layout_right0);

        block1.m_block_type = BlockType::FFT;
        block1.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes1);
        m_block_infos.push_back(block1);

        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block1.m_out_extents) * 2);

        BlockInfoType block2;
        auto [in_axis2, out_axis2] = get_pencil(mid_topo, out_topology.array());
        block2.m_in_map            = block0.m_out_map;
        block2.m_out_map   = get_dst_map<Layout>(block2.m_in_map, axes2);
        block2.m_in_axis   = in_axis2;
        block2.m_out_axis  = out_axis2;
        block2.m_comm_axis = all_trans_axes.at(1);

        block2.m_in_extents     = block1.m_out_extents;
        block2.m_out_extents    = get_next_extents(gout_extents, out_topology,
                                                   block2.m_out_map, comm);
        block2.m_buffer_extents = get_buffer_extents<Layout>(
            gout_extents, mid_topo, out_topology.array());
        block2.m_block_type = BlockType::Transpose;
        block2.m_block_idx  = 1;
        m_block_infos.push_back(block2);

        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block2.m_in_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block2.m_buffer_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block2.m_out_extents) * 2);

        BlockInfoType block3;
        block3.m_in_map      = block2.m_out_map;
        block3.m_out_map     = block2.m_out_map;
        block3.m_in_extents  = block2.m_out_extents;
        block3.m_out_extents = block3.m_in_extents;
        block3.m_block_type  = BlockType::FFT;
        block3.m_block_idx   = 1;
        block3.m_axes        = get_contiguous_axes<Layout, iType, DIM>(axes2);
        m_block_infos.push_back(block3);

        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block3.m_out_extents) * 2);
        m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
      } else {
        BlockInfoType block0;
        block0.m_in_map     = map;
        block0.m_out_map    = map;
        block0.m_in_extents = get_mapped_extents(in_extents, map);
        block0.m_out_extents =
            get_next_extents(gout_extents, in_topology, map, comm);
        block0.m_block_type = BlockType::FFT;
        block0.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes0);
        m_block_infos.push_back(block0);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_out_extents) * 2);

        BlockInfoType block1;
        auto [in_axis1, out_axis1] = get_pencil(in_topology.array(), mid_topo);
        block1.m_in_map            = map;
        block1.m_out_map      = get_dst_map<Layout>(block1.m_in_map, axes1);
        block1.m_in_axis      = in_axis1;
        block1.m_out_axis     = out_axis1;
        block1.m_comm_axis    = all_trans_axes.at(0);
        bool is_layout_right1 = all_layouts.at(1);

        block1.m_in_extents  = block0.m_out_extents;
        block1.m_out_extents = get_next_extents(
            gout_extents, mid_topo, block1.m_out_map, comm, is_layout_right1);

        block1.m_buffer_extents = get_buffer_extents<Layout>(
            gout_extents, in_topology.array(), mid_topo);
        block1.m_block_type = BlockType::Transpose;
        m_block_infos.push_back(block1);

        // Data is always complex
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block1.m_in_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block1.m_buffer_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block1.m_out_extents) * 2);

        BlockInfoType block2;
        block2.m_in_extents  = block1.m_out_extents;
        block2.m_out_extents = get_next_extents(
            gout_extents, mid_topo, block1.m_out_map, comm, is_layout_right1);
        block2.m_block_type = BlockType::FFT;
        block2.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes1);
        block2.m_block_idx  = 1;
        m_block_infos.push_back(block2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block2.m_out_extents) * 2);

        if (axes2.size() == 0) {
          BlockInfoType block3;
          auto [in_axis3, out_axis3] =
              get_pencil(mid_topo, out_topology.array());
          block3.m_in_map      = block1.m_out_map;
          block3.m_out_map     = src_map;
          block3.m_in_axis     = in_axis3;
          block3.m_out_axis    = out_axis3;
          block3.m_comm_axis   = all_trans_axes.at(1);
          block3.m_in_extents  = block2.m_out_extents;
          block3.m_out_extents = out_extents;

          block3.m_buffer_extents = get_buffer_extents<Layout>(
              gout_extents, mid_topo, out_topology.array());
          block3.m_block_type = BlockType::Transpose;
          block3.m_block_idx  = 1;
          m_block_infos.push_back(block3);

          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block3.m_in_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block3.m_buffer_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block3.m_out_extents) * 2);
        } else {
          BlockInfoType block3;
          auto [in_axis3, out_axis3] =
              get_pencil(mid_topo, out_topology.array());
          block3.m_in_map    = block1.m_out_map;
          block3.m_out_map   = get_dst_map<Layout>(block3.m_in_map, out_axis3);
          block3.m_in_axis   = in_axis3;
          block3.m_out_axis  = out_axis3;
          block3.m_comm_axis = all_trans_axes.at(1);
          bool is_layout_right3 = all_layouts.at(2);

          block3.m_in_extents = block2.m_out_extents;
          block3.m_out_extents =
              get_next_extents(gout_extents, out_topology.array(),
                               block3.m_out_map, comm, is_layout_right3);

          block3.m_buffer_extents = get_buffer_extents<Layout>(
              gout_extents, mid_topo, out_topology.array());
          block3.m_block_type = BlockType::Transpose;
          block3.m_block_idx  = 1;
          m_block_infos.push_back(block3);

          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block3.m_in_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block3.m_buffer_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block3.m_out_extents) * 2);

          BlockInfoType block4;
          block4.m_in_map      = block3.m_out_map;
          block4.m_out_map     = block3.m_out_map;
          block4.m_in_extents  = block3.m_out_extents;
          block4.m_out_extents = block4.m_in_extents;
          block4.m_block_type  = BlockType::FFT;
          block4.m_block_idx   = 2;
          block4.m_axes        = get_contiguous_axes<Layout, iType, DIM>(axes2);
          m_block_infos.push_back(block4);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block4.m_out_extents) * 2);
        }

        m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
      }
    } else if (nb_topologies == 4) {
      // 0. FFT + T + FFT + T + FFT + T
      // 1. FFT + T + FFT + T + T + FFT
      // 2. T + FFT + T + FFT + T + FFT
      // 3. T + FFT + T + FFT + T

      auto mid_topo0 = all_topologies.at(1), mid_topo1 = all_topologies.at(2);

      auto axes0 = all_axes.at(0), axes1 = all_axes.at(1),
           axes2 = all_axes.at(2), axes3 = all_axes.at(3);

      if (axes0.size() == 0) {
        BlockInfoType block0;
        auto [in_axis0, out_axis0] = get_pencil(in_topology.array(), mid_topo0);
        block0.m_in_map            = src_map;
        block0.m_out_map    = get_dst_map<Layout>(block0.m_in_map, axes1);
        block0.m_in_axis    = in_axis0;
        block0.m_out_axis   = out_axis0;
        block0.m_comm_axis  = all_trans_axes.at(0);
        block0.m_in_extents = in_extents;

        bool is_layout_right0 = all_layouts.at(1);
        block0.m_out_extents  = get_next_extents(
            gin_extents, mid_topo0, block0.m_out_map, comm, is_layout_right0);
        block0.m_buffer_extents = get_buffer_extents<Layout>(
            gin_extents, in_topology.array(), mid_topo0);

        block0.m_block_type = BlockType::Transpose;
        m_block_infos.push_back(block0);

        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_in_extents) * size_factor);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_buffer_extents) * size_factor);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_out_extents) * size_factor);

        BlockInfoType block1;
        block1.m_in_extents  = block0.m_out_extents;
        block1.m_out_extents = get_next_extents(
            gout_extents, mid_topo0, block0.m_out_map, comm, is_layout_right0);

        block1.m_block_type = BlockType::FFT;
        block1.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes1);
        m_block_infos.push_back(block1);

        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block1.m_out_extents) * 2);

        BlockInfoType block2;
        auto [in_axis2, out_axis2] = get_pencil(mid_topo0, mid_topo1);
        block2.m_in_map            = block0.m_out_map;
        block2.m_out_map      = get_dst_map<Layout>(block2.m_in_map, axes2);
        block2.m_in_axis      = in_axis2;
        block2.m_out_axis     = out_axis2;
        block2.m_comm_axis    = all_trans_axes.at(1);
        bool is_layout_right2 = all_layouts.at(2);

        block2.m_in_extents  = block1.m_out_extents;
        block2.m_out_extents = get_next_extents(
            gout_extents, mid_topo1, block2.m_out_map, comm, is_layout_right2);
        block2.m_buffer_extents =
            get_buffer_extents<Layout>(gout_extents, mid_topo0, mid_topo1);
        block2.m_block_type = BlockType::Transpose;
        block2.m_block_idx  = 1;
        m_block_infos.push_back(block2);

        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block2.m_in_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block2.m_buffer_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block2.m_out_extents) * 2);

        BlockInfoType block3;
        block3.m_in_extents  = block2.m_out_extents;
        block3.m_out_extents = get_next_extents(
            gout_extents, mid_topo1, block2.m_out_map, comm, is_layout_right2);
        block3.m_block_type = BlockType::FFT;
        block3.m_block_idx  = 1;
        block3.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes2);
        m_block_infos.push_back(block3);

        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block3.m_out_extents) * 2);

        if (axes3.size() == 0) {
          BlockInfoType block4;
          auto [in_axis4, out_axis4] =
              get_pencil(mid_topo1, out_topology.array());
          block4.m_in_map         = block2.m_out_map;
          block4.m_out_map        = src_map;
          block4.m_in_axis        = in_axis4;
          block4.m_out_axis       = out_axis4;
          block4.m_comm_axis      = all_trans_axes.at(2);
          block4.m_in_extents     = block3.m_out_extents;
          block4.m_out_extents    = out_extents;
          block4.m_buffer_extents = get_buffer_extents<Layout>(
              gout_extents, mid_topo1, out_topology.array());
          block4.m_block_type = BlockType::Transpose;
          block4.m_block_idx  = 2;
          m_block_infos.push_back(block4);

          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block4.m_in_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block4.m_buffer_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block4.m_out_extents) * 2);
        } else {
          BlockInfoType block4;
          auto [in_axis4, out_axis4] =
              get_pencil(mid_topo1, out_topology.array());
          block4.m_in_map    = block2.m_out_map;
          block4.m_out_map   = get_dst_map<Layout>(block4.m_in_map, out_axis4);
          block4.m_in_axis   = in_axis4;
          block4.m_out_axis  = out_axis4;
          block4.m_comm_axis = all_trans_axes.at(2);
          bool is_layout_right4 = all_layouts.at(3);

          block4.m_in_extents = block3.m_out_extents;
          block4.m_out_extents =
              get_next_extents(gout_extents, out_topology.array(),
                               block4.m_out_map, comm, is_layout_right4);
          block4.m_buffer_extents = get_buffer_extents<Layout>(
              gout_extents, mid_topo1, out_topology.array());
          block4.m_block_type = BlockType::Transpose;
          block4.m_block_idx  = 2;
          m_block_infos.push_back(block4);

          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block4.m_in_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block4.m_buffer_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block4.m_out_extents) * 2);

          BlockInfoType block5;
          block5.m_in_map      = block4.m_out_map;
          block5.m_out_map     = block4.m_out_map;
          block5.m_in_extents  = block4.m_out_extents;
          block5.m_out_extents = block5.m_in_extents;
          block5.m_block_type  = BlockType::FFT;
          block5.m_block_idx   = 2;
          block5.m_axes        = get_contiguous_axes<Layout, iType, DIM>(axes3);
          m_block_infos.push_back(block5);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block5.m_out_extents) * 2);
        }

        m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
      } else {
        BlockInfoType block0;
        block0.m_in_map     = map;
        block0.m_out_map    = map;
        block0.m_in_extents = get_mapped_extents(in_extents, map);
        block0.m_out_extents =
            get_next_extents(gout_extents, in_topology, map, comm);
        block0.m_block_type = BlockType::FFT;
        block0.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes0);
        m_block_infos.push_back(block0);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_out_extents) * 2);

        BlockInfoType block1;
        auto [in_axis1, out_axis1] = get_pencil(in_topology.array(), mid_topo0);
        block1.m_in_map            = map;
        block1.m_out_map   = get_dst_map<Layout>(block0.m_out_map, out_axis1);
        block1.m_in_axis   = in_axis1;
        block1.m_out_axis  = out_axis1;
        block1.m_comm_axis = all_trans_axes.at(0);
        bool is_layout_right1 = all_layouts.at(1);

        block1.m_in_extents  = block0.m_out_extents;
        block1.m_out_extents = get_next_extents(
            gout_extents, mid_topo0, block1.m_out_map, comm, is_layout_right1);

        block1.m_buffer_extents = get_buffer_extents<Layout>(
            gout_extents, in_topology.array(), mid_topo0);
        block1.m_block_type = BlockType::Transpose;
        m_block_infos.push_back(block1);

        // Data is always complex
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block1.m_in_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block1.m_buffer_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block1.m_out_extents) * 2);

        BlockInfoType block2;
        block2.m_in_extents  = block1.m_out_extents;
        block2.m_out_extents = get_next_extents(
            gout_extents, mid_topo0, block1.m_out_map, comm, is_layout_right1);
        block2.m_block_type = BlockType::FFT;
        block2.m_block_idx  = 1;
        block2.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes1);
        m_block_infos.push_back(block2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block2.m_out_extents) * 2);

        BlockInfoType block3;
        auto [in_axis3, out_axis3] = get_pencil(mid_topo0, mid_topo1);
        block3.m_in_map            = block1.m_out_map;
        block3.m_out_map      = get_dst_map<Layout>(block3.m_in_map, out_axis3);
        block3.m_in_axis      = in_axis3;
        block3.m_out_axis     = out_axis3;
        block3.m_comm_axis    = all_trans_axes.at(1);
        bool is_layout_right3 = all_layouts.at(2);

        block3.m_in_extents  = block2.m_out_extents;
        block3.m_out_extents = get_next_extents(
            gout_extents, mid_topo1, block3.m_out_map, comm, is_layout_right3);

        block3.m_buffer_extents =
            get_buffer_extents<Layout>(gout_extents, mid_topo0, mid_topo1);
        block3.m_block_type = BlockType::Transpose;
        block3.m_block_idx  = 1;
        m_block_infos.push_back(block3);

        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block3.m_in_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block3.m_buffer_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block3.m_out_extents) * 2);

        if (axes3.size() == 0) {
          // 0. FFT + T + FFT + T + FFT + T
          BlockInfoType block4;
          block4.m_in_extents = block3.m_out_extents;
          block4.m_out_extents =
              get_next_extents(gout_extents, mid_topo1, block3.m_out_map, comm,
                               is_layout_right3);
          block4.m_block_type = BlockType::FFT;
          block4.m_block_idx  = 2;
          block4.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes2);
          m_block_infos.push_back(block4);

          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block4.m_out_extents) * 2);

          BlockInfoType block5;
          auto [in_axis5, out_axis5] =
              get_pencil(mid_topo1, out_topology.array());
          block5.m_in_map    = block3.m_out_map;
          block5.m_out_map   = src_map;
          block5.m_in_axis   = in_axis5;
          block5.m_out_axis  = out_axis5;
          block5.m_comm_axis = all_trans_axes.at(2);

          block5.m_in_extents  = block4.m_out_extents;
          block5.m_out_extents = out_extents;

          block5.m_buffer_extents = get_buffer_extents<Layout>(
              gout_extents, mid_topo1, out_topology.array());
          block5.m_block_type = BlockType::Transpose;
          block5.m_block_idx  = 2;
          m_block_infos.push_back(block5);

          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block5.m_in_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block5.m_buffer_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block5.m_out_extents) * 2);

        } else {
          // 1. FFT + T + FFT + T + T + FFT
          BlockInfoType block4;
          auto [in_axis4, out_axis4] =
              get_pencil(mid_topo1, out_topology.array());
          block4.m_in_map    = block3.m_out_map;
          block4.m_out_map   = get_dst_map<Layout>(block4.m_in_map, out_axis4);
          block4.m_in_axis   = in_axis4;
          block4.m_out_axis  = out_axis4;
          block4.m_comm_axis = all_trans_axes.at(2);
          bool is_layout_right4 = all_layouts.at(3);

          block4.m_in_extents = block3.m_out_extents;
          block4.m_out_extents =
              get_next_extents(gout_extents, out_topology.array(),
                               block4.m_out_map, comm, is_layout_right4);

          block4.m_buffer_extents = get_buffer_extents<Layout>(
              gout_extents, mid_topo1, out_topology.array());
          block4.m_block_type = BlockType::Transpose;
          block4.m_block_idx  = 2;
          m_block_infos.push_back(block4);

          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block4.m_in_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block4.m_buffer_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block4.m_out_extents) * 2);

          BlockInfoType block5;
          block5.m_in_map      = block4.m_out_map;
          block5.m_out_map     = block4.m_out_map;
          block5.m_in_extents  = block4.m_out_extents;
          block5.m_out_extents = block5.m_in_extents;
          block5.m_block_type  = BlockType::FFT;
          block5.m_block_idx   = 2;
          block5.m_axes        = get_contiguous_axes<Layout, iType, DIM>(axes3);
          m_block_infos.push_back(block5);

          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block5.m_out_extents) * 2);
        }

        m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
      }
    } else if (nb_topologies == 5) {
      // 0. FFT + T + FFT + T + FFT + T + T
      // 1. T + FFT + T + FFT + T + FFT + T
      // 2. T + FFT + T + FFT + T + T + FFT
      // 3. T + FFT + T + FFT + T + T

      auto mid_topo0 = all_topologies.at(1), mid_topo1 = all_topologies.at(2),
           mid_topo2 = all_topologies.at(3);

      auto axes0 = all_axes.at(0), axes1 = all_axes.at(1),
           axes2 = all_axes.at(2), axes3 = all_axes.at(3),
           axes4 = all_axes.at(4);

      if (axes0.size() == 0) {
        BlockInfoType block0;
        auto [in_axis0, out_axis0] = get_pencil(in_topology.array(), mid_topo0);
        block0.m_in_map            = src_map;
        block0.m_out_map    = get_dst_map<Layout>(block0.m_in_map, axes1);
        block0.m_in_axis    = in_axis0;
        block0.m_out_axis   = out_axis0;
        block0.m_comm_axis  = all_trans_axes.at(0);
        block0.m_in_extents = in_extents;

        // bool is_layout_right0 = block0.m_comm_axis == 0;
        bool is_layout_right0 = all_layouts.at(1);
        block0.m_out_extents  = get_next_extents(
            gin_extents, mid_topo0, block0.m_out_map, comm, is_layout_right0);
        block0.m_buffer_extents = get_buffer_extents<Layout>(
            gin_extents, in_topology.array(), mid_topo0);

        block0.m_block_type = BlockType::Transpose;
        m_block_infos.push_back(block0);

        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_in_extents) * size_factor);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_buffer_extents) * size_factor);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_out_extents) * size_factor);

        BlockInfoType block1;
        block1.m_in_extents  = block0.m_out_extents;
        block1.m_out_extents = get_next_extents(
            gout_extents, mid_topo0, block0.m_out_map, comm, is_layout_right0);

        block1.m_block_type = BlockType::FFT;
        block1.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes1);
        m_block_infos.push_back(block1);

        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block1.m_out_extents) * 2);

        BlockInfoType block2;
        auto [in_axis2, out_axis2] = get_pencil(mid_topo0, mid_topo1);
        block2.m_in_map            = block0.m_out_map;
        block2.m_out_map      = get_dst_map<Layout>(block2.m_in_map, axes2);
        block2.m_in_axis      = in_axis2;
        block2.m_out_axis     = out_axis2;
        block2.m_comm_axis    = all_trans_axes.at(1);
        bool is_layout_right2 = all_layouts.at(2);
        block2.m_in_extents   = block1.m_out_extents;
        block2.m_out_extents  = get_next_extents(
            gout_extents, mid_topo1, block2.m_out_map, comm, is_layout_right2);
        block2.m_buffer_extents =
            get_buffer_extents<Layout>(gout_extents, mid_topo0, mid_topo1);
        block2.m_block_type = BlockType::Transpose;
        block2.m_block_idx  = 1;
        m_block_infos.push_back(block2);

        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block2.m_in_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block2.m_buffer_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block2.m_out_extents) * 2);

        BlockInfoType block3;
        block3.m_in_extents  = block2.m_out_extents;
        block3.m_out_extents = get_next_extents(
            gout_extents, mid_topo1, block2.m_out_map, comm, is_layout_right2);
        block3.m_block_type = BlockType::FFT;
        block3.m_block_idx  = 1;
        block3.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes2);
        m_block_infos.push_back(block3);

        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block3.m_out_extents) * 2);

        BlockInfoType block4;
        auto [in_axis4, out_axis4] = get_pencil(mid_topo1, mid_topo2);
        block4.m_in_map            = block2.m_out_map;
        block4.m_out_map      = get_dst_map<Layout>(block4.m_in_map, out_axis4);
        block4.m_in_axis      = in_axis4;
        block4.m_out_axis     = out_axis4;
        block4.m_comm_axis    = all_trans_axes.at(2);
        bool is_layout_right4 = all_layouts.at(3);
        block4.m_in_extents   = block3.m_out_extents;
        block4.m_out_extents  = get_next_extents(
            gout_extents, mid_topo2, block4.m_out_map, comm, is_layout_right4);
        block4.m_buffer_extents =
            get_buffer_extents<Layout>(gout_extents, mid_topo1, mid_topo2);
        block4.m_block_type = BlockType::Transpose;
        block4.m_block_idx  = 2;
        m_block_infos.push_back(block4);

        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block4.m_in_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block4.m_buffer_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block4.m_out_extents) * 2);

        if (axes3.size() != 0) {
          // 1. T + FFT + T + FFT + T + FFT + T
          BlockInfoType block5;
          block5.m_in_extents = block4.m_out_extents;
          block5.m_out_extents =
              get_next_extents(gout_extents, mid_topo2, block4.m_out_map, comm,
                               is_layout_right4);
          block5.m_block_type = BlockType::FFT;
          block5.m_block_idx  = 2;
          block5.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes3);
          m_block_infos.push_back(block5);

          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block5.m_out_extents) * 2);

          BlockInfoType block6;
          auto [in_axis6, out_axis6] =
              get_pencil(mid_topo2, out_topology.array());
          block6.m_in_map    = block4.m_out_map;
          block6.m_out_map   = src_map;
          block6.m_in_axis   = in_axis6;
          block6.m_out_axis  = out_axis6;
          block6.m_comm_axis = all_trans_axes.at(3);

          block6.m_in_extents  = block5.m_out_extents;
          block6.m_out_extents = out_extents;

          block6.m_buffer_extents = get_buffer_extents<Layout>(
              gout_extents, mid_topo2, out_topology.array());
          block6.m_block_type = BlockType::Transpose;
          block6.m_block_idx  = 3;
          m_block_infos.push_back(block6);

          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block6.m_in_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block6.m_buffer_extents) * 2);
          all_max_buffer_sizes.push_back(
              KokkosFFT::Impl::total_size(block6.m_out_extents) * 2);

        } else {
          if (axes4.size() == 0) {
            // 2. T + FFT + T + FFT + T + T
            BlockInfoType block5;
            auto [in_axis5, out_axis5] =
                get_pencil(mid_topo2, out_topology.array());
            block5.m_in_map      = block4.m_out_map;
            block5.m_out_map     = src_map;
            block5.m_in_axis     = in_axis5;
            block5.m_out_axis    = out_axis5;
            block5.m_comm_axis   = all_trans_axes.at(3);
            block5.m_in_extents  = block4.m_out_extents;
            block5.m_out_extents = out_extents;

            block5.m_buffer_extents = get_buffer_extents<Layout>(
                gout_extents, mid_topo2, out_topology.array());
            block5.m_block_type = BlockType::Transpose;
            block5.m_block_idx  = 3;
            m_block_infos.push_back(block5);

            all_max_buffer_sizes.push_back(
                KokkosFFT::Impl::total_size(block5.m_in_extents) * 2);
            all_max_buffer_sizes.push_back(
                KokkosFFT::Impl::total_size(block5.m_buffer_extents) * 2);
            all_max_buffer_sizes.push_back(
                KokkosFFT::Impl::total_size(block5.m_out_extents) * 2);
          } else {
            // 2. T + FFT + T + FFT + T + T + FFT
            BlockInfoType block5;
            auto [in_axis5, out_axis5] =
                get_pencil(mid_topo2, out_topology.array());
            block5.m_in_map   = block4.m_out_map;
            block5.m_out_map  = get_dst_map<Layout>(block5.m_in_map, out_axis5);
            block5.m_in_axis  = in_axis5;
            block5.m_out_axis = out_axis5;
            block5.m_comm_axis    = all_trans_axes.at(3);
            bool is_layout_right5 = all_layouts.at(4);
            block5.m_in_extents   = block4.m_out_extents;
            block5.m_out_extents =
                get_next_extents(gout_extents, out_topology.array(),
                                 block5.m_out_map, comm, is_layout_right5);

            block5.m_buffer_extents = get_buffer_extents<Layout>(
                gout_extents, mid_topo2, out_topology.array());
            block5.m_block_type = BlockType::Transpose;
            block5.m_block_idx  = 3;
            m_block_infos.push_back(block5);

            all_max_buffer_sizes.push_back(
                KokkosFFT::Impl::total_size(block5.m_in_extents) * 2);
            all_max_buffer_sizes.push_back(
                KokkosFFT::Impl::total_size(block5.m_buffer_extents) * 2);
            all_max_buffer_sizes.push_back(
                KokkosFFT::Impl::total_size(block5.m_out_extents) * 2);

            BlockInfoType block6;
            block6.m_in_map      = block5.m_out_map;
            block6.m_out_map     = block5.m_out_map;
            block6.m_in_extents  = block5.m_out_extents;
            block6.m_out_extents = block6.m_in_extents;
            block6.m_block_type  = BlockType::FFT;
            block6.m_block_idx   = 2;
            block6.m_axes = get_contiguous_axes<Layout, iType, DIM>(axes4);
            m_block_infos.push_back(block6);

            all_max_buffer_sizes.push_back(
                KokkosFFT::Impl::total_size(block6.m_out_extents) * 2);
          }
        }

        m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
      } else {
        // 0. FFT + T + FFT + T + FFT + T + T
        BlockInfoType block0;
        block0.m_in_map     = map;
        block0.m_out_map    = map;
        block0.m_in_extents = get_mapped_extents(in_extents, map);
        block0.m_out_extents =
            get_next_extents(gout_extents, in_topology, map, comm);
        block0.m_block_type = BlockType::FFT;
        block0.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes0);
        m_block_infos.push_back(block0);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block0.m_out_extents) * 2);

        BlockInfoType block1;
        auto [in_axis1, out_axis1] = get_pencil(in_topology.array(), mid_topo0);
        block1.m_in_map            = map;
        block1.m_out_map   = get_dst_map<Layout>(block0.m_out_map, out_axis1);
        block1.m_in_axis   = in_axis1;
        block1.m_out_axis  = out_axis1;
        block1.m_comm_axis = all_trans_axes.at(0);
        bool is_layout_right1 = all_layouts.at(1);

        block1.m_in_extents  = block0.m_out_extents;
        block1.m_out_extents = get_next_extents(
            gout_extents, mid_topo0, block1.m_out_map, comm, is_layout_right1);

        block1.m_buffer_extents = get_buffer_extents<Layout>(
            gout_extents, in_topology.array(), mid_topo0);
        block1.m_block_type = BlockType::Transpose;
        m_block_infos.push_back(block1);

        // Data is always complex
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block1.m_in_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block1.m_buffer_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block1.m_out_extents) * 2);

        BlockInfoType block2;
        block2.m_in_extents  = block1.m_out_extents;
        block2.m_out_extents = get_next_extents(
            gout_extents, mid_topo0, block1.m_out_map, comm, is_layout_right1);
        block2.m_block_type = BlockType::FFT;
        block2.m_block_idx  = 1;
        block2.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes1);
        m_block_infos.push_back(block2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block2.m_out_extents) * 2);

        BlockInfoType block3;
        auto [in_axis3, out_axis3] = get_pencil(mid_topo0, mid_topo1);
        block3.m_in_map            = block1.m_out_map;
        block3.m_out_map      = get_dst_map<Layout>(block3.m_in_map, out_axis3);
        block3.m_in_axis      = in_axis3;
        block3.m_out_axis     = out_axis3;
        block3.m_comm_axis    = all_trans_axes.at(1);
        bool is_layout_right3 = all_layouts.at(2);

        block3.m_in_extents  = block2.m_out_extents;
        block3.m_out_extents = get_next_extents(
            gout_extents, mid_topo1, block3.m_out_map, comm, is_layout_right3);

        block3.m_buffer_extents =
            get_buffer_extents<Layout>(gout_extents, mid_topo0, mid_topo1);
        block3.m_block_type = BlockType::Transpose;
        block3.m_block_idx  = 1;
        m_block_infos.push_back(block3);

        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block3.m_in_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block3.m_buffer_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block3.m_out_extents) * 2);

        BlockInfoType block4;
        block4.m_in_extents  = block3.m_out_extents;
        block4.m_out_extents = get_next_extents(
            gout_extents, mid_topo1, block3.m_out_map, comm, is_layout_right3);
        block4.m_block_type = BlockType::FFT;
        block4.m_block_idx  = 2;
        block4.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes2);
        m_block_infos.push_back(block4);

        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block4.m_out_extents) * 2);

        BlockInfoType block5;
        auto [in_axis5, out_axis5] = get_pencil(mid_topo1, mid_topo2);
        block5.m_in_map            = block3.m_out_map;
        block5.m_out_map      = get_dst_map<Layout>(block5.m_in_map, out_axis5);
        block5.m_in_axis      = in_axis5;
        block5.m_out_axis     = out_axis5;
        block5.m_comm_axis    = all_trans_axes.at(2);
        bool is_layout_right5 = all_layouts.at(3);

        block5.m_in_extents  = block4.m_out_extents;
        block5.m_out_extents = get_next_extents(
            gout_extents, mid_topo2, block5.m_out_map, comm, is_layout_right5);

        block5.m_buffer_extents =
            get_buffer_extents<Layout>(gout_extents, mid_topo1, mid_topo2);
        block5.m_block_type = BlockType::Transpose;
        block5.m_block_idx  = 2;
        m_block_infos.push_back(block5);

        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block5.m_in_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block5.m_buffer_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block5.m_out_extents) * 2);

        BlockInfoType block6;
        auto [in_axis6, out_axis6] =
            get_pencil(mid_topo2, out_topology.array());
        block6.m_in_map    = block5.m_out_map;
        block6.m_out_map   = src_map;
        block6.m_in_axis   = in_axis6;
        block6.m_out_axis  = out_axis6;
        block6.m_comm_axis = all_trans_axes.at(3);

        block6.m_in_extents  = block5.m_out_extents;
        block6.m_out_extents = out_extents;

        block6.m_buffer_extents = get_buffer_extents<Layout>(
            gout_extents, mid_topo2, out_topology.array());
        block6.m_block_type = BlockType::Transpose;
        block6.m_block_idx  = 3;
        m_block_infos.push_back(block6);

        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block6.m_in_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block6.m_buffer_extents) * 2);
        all_max_buffer_sizes.push_back(
            KokkosFFT::Impl::total_size(block6.m_out_extents) * 2);

        m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
      }
    } else if (nb_topologies == 6) {
      // 0. T + FFT + T + FFT + T + FFT + T + T
      auto mid_topo0 = all_topologies.at(1), mid_topo1 = all_topologies.at(2),
           mid_topo2 = all_topologies.at(3), mid_topo3 = all_topologies.at(4);

      auto axes0 = all_axes.at(0), axes1 = all_axes.at(1),
           axes2 = all_axes.at(2), axes3 = all_axes.at(3),
           axes4 = all_axes.at(4), axes5 = all_axes.at(5);

      BlockInfoType block0;
      auto [in_axis0, out_axis0] = get_pencil(in_topology.array(), mid_topo0);
      block0.m_in_map            = src_map;
      block0.m_out_map           = get_dst_map<Layout>(src_map, out_axis0);
      block0.m_in_axis           = in_axis0;
      block0.m_out_axis          = out_axis0;
      block0.m_comm_axis         = all_trans_axes.at(0);
      block0.m_in_extents        = in_extents;

      bool is_layout_right0 = all_layouts.at(1);
      block0.m_out_extents  = get_next_extents(
          gin_extents, mid_topo0, block0.m_out_map, comm, is_layout_right0);
      block0.m_buffer_extents = get_buffer_extents<Layout>(
          gin_extents, in_topology.array(), mid_topo0);

      block0.m_block_type = BlockType::Transpose;
      m_block_infos.push_back(block0);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block0.m_in_extents) * size_factor);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block0.m_buffer_extents) * size_factor);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block0.m_out_extents) * size_factor);

      BlockInfoType block1;
      block1.m_in_extents  = block0.m_out_extents;
      block1.m_out_extents = get_next_extents(
          gout_extents, mid_topo0, block0.m_out_map, comm, is_layout_right0);
      block1.m_block_type = BlockType::FFT;
      block1.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes1);
      m_block_infos.push_back(block1);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block1.m_out_extents) * 2);

      BlockInfoType block2;
      auto [in_axis2, out_axis2] = get_pencil(mid_topo0, mid_topo1);
      block2.m_in_map            = block0.m_out_map;
      block2.m_out_map      = get_dst_map<Layout>(block2.m_in_map, out_axis2);
      block2.m_in_axis      = in_axis2;
      block2.m_out_axis     = out_axis2;
      block2.m_comm_axis    = all_trans_axes.at(1);
      bool is_layout_right2 = all_layouts.at(2);
      block2.m_in_extents   = block1.m_out_extents;
      block2.m_out_extents  = get_next_extents(
          gout_extents, mid_topo1, block2.m_out_map, comm, is_layout_right2);
      block2.m_buffer_extents =
          get_buffer_extents<Layout>(gout_extents, mid_topo0, mid_topo1);
      block2.m_block_type = BlockType::Transpose;
      block2.m_block_idx  = 1;
      m_block_infos.push_back(block2);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block2.m_in_extents) * 2);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block2.m_buffer_extents) * 2);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block2.m_out_extents) * 2);

      BlockInfoType block3;
      block3.m_in_extents  = block2.m_out_extents;
      block3.m_out_extents = get_next_extents(
          gout_extents, mid_topo1, block2.m_out_map, comm, is_layout_right2);
      block3.m_block_type = BlockType::FFT;
      block3.m_block_idx  = 1;
      block3.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes2);
      m_block_infos.push_back(block3);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block3.m_out_extents) * 2);

      BlockInfoType block4;
      auto [in_axis4, out_axis4] = get_pencil(mid_topo1, mid_topo2);
      block4.m_in_map            = block2.m_out_map;
      block4.m_out_map      = get_dst_map<Layout>(block4.m_in_map, out_axis4);
      block4.m_in_axis      = in_axis4;
      block4.m_out_axis     = out_axis4;
      block4.m_comm_axis    = all_trans_axes.at(2);
      bool is_layout_right4 = all_layouts.at(3);
      block4.m_in_extents   = block3.m_out_extents;
      block4.m_out_extents  = get_next_extents(
          gout_extents, mid_topo2, block4.m_out_map, comm, is_layout_right4);
      block4.m_buffer_extents =
          get_buffer_extents<Layout>(gout_extents, mid_topo1, mid_topo2);
      block4.m_block_type = BlockType::Transpose;
      block4.m_block_idx  = 2;
      m_block_infos.push_back(block4);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block4.m_in_extents) * 2);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block4.m_buffer_extents) * 2);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block4.m_out_extents) * 2);

      BlockInfoType block5;
      block5.m_in_extents  = block4.m_out_extents;
      block5.m_out_extents = get_next_extents(
          gout_extents, mid_topo2, block4.m_out_map, comm, is_layout_right4);
      block5.m_block_type = BlockType::FFT;
      block5.m_block_idx  = 2;
      block5.m_axes       = get_contiguous_axes<Layout, iType, DIM>(axes3);
      m_block_infos.push_back(block5);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block5.m_out_extents) * 2);

      BlockInfoType block6;
      auto [in_axis6, out_axis6] = get_pencil(mid_topo2, mid_topo3);
      block6.m_in_map            = block4.m_out_map;
      block6.m_out_map      = get_dst_map<Layout>(block6.m_in_map, out_axis6);
      block6.m_in_axis      = in_axis6;
      block6.m_out_axis     = out_axis6;
      block6.m_comm_axis    = all_trans_axes.at(3);
      bool is_layout_right6 = all_layouts.at(4);
      block6.m_in_extents   = block5.m_out_extents;
      block6.m_out_extents  = get_next_extents(
          gout_extents, mid_topo3, block6.m_out_map, comm, is_layout_right6);
      block6.m_buffer_extents =
          get_buffer_extents<Layout>(gout_extents, mid_topo2, mid_topo3);
      block6.m_block_type = BlockType::Transpose;
      block6.m_block_idx  = 3;
      m_block_infos.push_back(block6);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block6.m_in_extents) * 2);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block6.m_buffer_extents) * 2);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block6.m_out_extents) * 2);

      BlockInfoType block7;
      auto [in_axis7, out_axis7] = get_pencil(mid_topo3, out_topology.array());
      block7.m_in_map            = block6.m_out_map;
      block7.m_out_map           = src_map;
      block7.m_in_axis           = in_axis7;
      block7.m_out_axis          = out_axis7;
      block7.m_comm_axis         = all_trans_axes.at(4);
      block7.m_in_extents        = block6.m_out_extents;
      block7.m_out_extents       = out_extents;
      block7.m_buffer_extents    = get_buffer_extents<Layout>(
          gout_extents, mid_topo3, out_topology.array());
      block7.m_block_type = BlockType::Transpose;
      block7.m_block_idx  = 4;
      m_block_infos.push_back(block7);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block7.m_in_extents) * 2);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block7.m_buffer_extents) * 2);
      all_max_buffer_sizes.push_back(
          KokkosFFT::Impl::total_size(block7.m_out_extents) * 2);

      m_max_buffer_size = get_max(all_max_buffer_sizes, comm);
    }
  }
};

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
