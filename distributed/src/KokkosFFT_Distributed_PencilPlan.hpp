#ifndef KOKKOSFFT_DISTRIBUTED_PENCILPLAN_HPP
#define KOKKOSFFT_DISTRIBUTED_PENCILPLAN_HPP

#include <vector>
#include <memory>
#include <iostream>
#include <sstream>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_TransBlock.hpp"
#include "KokkosFFT_Distributed_MPI_Extents.hpp"
#include "KokkosFFT_Distributed_Helper.hpp"
#include "KokkosFFT_Distributed_Extents.hpp"
#include "KokkosFFT_Distributed_Topologies.hpp"
#include "KokkosFFT_Distributed_BlockAnalyses.hpp"
#include "KokkosFFT_Distributed_InternalPlan.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief Create pencil communicators (cartesian, row, column)
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam TopologyType Type of topology container
/// \tparam CommType Type of communicator wrapper
/// \param[in] exec_space Kokkos execution space
/// \param[in] topology The topology of the distributed FFT
/// \param[in] comm MPI communicator
/// \param[out] cart_comm The cartesian communicator
/// \param[out] cart_comms The row and column communicators
/// \param[out] cart_tpl_comms The communicator wrappers for row and column
template <typename ExecutionSpace, typename TopologyType, typename CommType>
void create_pencil_communicators(
    const ExecutionSpace& exec_space, const TopologyType& topology,
    const MPI_Comm& comm, MPI_Comm& cart_comm,
    std::vector<MPI_Comm>& cart_comms,
    std::vector<std::unique_ptr<CommType>>& cart_tpl_comms) {
  // Create a cartesian communicator
  std::vector<int> dims;
  for (auto& dim : topology) {
    if (dim > 1) {
      dims.push_back(static_cast<int>(dim));
    }
  }

  int periods[2] = {1, 1};  // Periodic in all directions
  ::MPI_Cart_create(comm, 2, dims.data(), periods, 0, &cart_comm);

  // split into row‐ and col‐ communicators
  ::MPI_Comm row_comm, col_comm;

  int remain_dims[2];

  // keep Y‐axis for row_comm (all procs with same px)
  remain_dims[0] = 1;
  remain_dims[1] = 0;

  ::MPI_Cart_sub(cart_comm, remain_dims, &row_comm);

  // keep X‐axis for col_comm (all procs with same py)
  remain_dims[0] = 0;
  remain_dims[1] = 1;
  ::MPI_Cart_sub(cart_comm, remain_dims, &col_comm);

  cart_comms = {row_comm, col_comm};

  // Ensure vector is empty before adding
  cart_tpl_comms.clear();
  cart_tpl_comms.emplace_back(std::make_unique<CommType>(row_comm, exec_space));
  cart_tpl_comms.emplace_back(std::make_unique<CommType>(col_comm, exec_space));
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM, typename InLayoutType, typename OutLayoutType>
struct PencilInternalPlan;

/// \brief Internal plan for 1D FFT with pencil decomposition
/// 1. 1D FFT
/// 1. Transpose + 1D FFT
/// 1. Transpose + 1D FFT + Transpose
///
/// \tparam ExecutionSpace
/// \tparam InViewType
/// \tparam OutViewType

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          typename InLayoutType, typename OutLayoutType>
struct PencilInternalPlan<ExecutionSpace, InViewType, OutViewType, 1,
                          InLayoutType, OutLayoutType> {
  using execSpace      = ExecutionSpace;
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;
  using float_type   = KokkosFFT::Impl::base_floating_point_type<in_value_type>;
  using complex_type = Kokkos::complex<float_type>;
  using LayoutType   = typename InViewType::array_layout;

  static constexpr std::size_t DIM = InViewType::rank();

  using axes_type     = KokkosFFT::axis_type<1>;
  using topology_type = KokkosFFT::shape_type<DIM>;
  using in_topology_type =
      Topology<std::size_t, InViewType::rank(), InLayoutType>;
  using out_topology_type =
      Topology<std::size_t, OutViewType::rank(), OutLayoutType>;

  using PencilBlockAnalysesType =
      PencilBlockAnalysesInternal<in_value_type, LayoutType, std::size_t, DIM,
                                  1, InLayoutType, OutLayoutType>;

  using AllocationViewType =
      Kokkos::View<complex_type*, LayoutType, ExecutionSpace>;

  // Type for transpose Block
  using CommType       = KokkosFFT::Distributed::Impl::TplComm<ExecutionSpace>;
  using TransBlockType = TransBlock<ExecutionSpace, DIM>;

  using FFTStageType = FFTStage<ExecutionSpace, InViewType, OutViewType>;

  execSpace m_exec_space;
  axes_type m_axes;
  in_topology_type m_in_topology;
  out_topology_type m_out_topology;
  MPI_Comm m_comm;

  // Cartesian Communicators
  MPI_Comm m_cart_comm;
  std::vector<MPI_Comm> m_cart_comms;
  std::vector<std::unique_ptr<CommType>> m_cart_tpl_comms;

  // Analyse topology
  std::unique_ptr<PencilBlockAnalysesType> m_block_analyses;

  // Buffer view types
  InViewType m_in_T;
  OutViewType m_out_T, m_out_T2;

  // Buffer Allocations
  AllocationViewType m_send_buffer_allocation, m_recv_buffer_allocation;
  std::vector<AllocationViewType> m_fft_buffer_allocations;

  // Internal transpose blocks
  std::vector<std::unique_ptr<TransBlockType>> m_trans_blocks;

  // Internal FFT stage
  std::unique_ptr<FFTStageType> m_fft_stage;

 public:
  explicit PencilInternalPlan(const ExecutionSpace& exec_space,
                              const InViewType& in, const OutViewType& out,
                              const axes_type& axes,
                              const in_topology_type& in_topology,
                              const out_topology_type& out_topology,
                              const MPI_Comm& comm)
      : m_exec_space(exec_space),
        m_axes(axes),
        m_in_topology(in_topology),
        m_out_topology(out_topology),
        m_comm(comm) {
    create_pencil_communicators(m_exec_space, m_in_topology, m_comm,
                                m_cart_comm, m_cart_comms, m_cart_tpl_comms);

    // First get global shape to define buffer and next shape
    auto in_extents   = KokkosFFT::Impl::extract_extents(in);
    auto gin_extents  = compute_global_extents(in, m_in_topology, m_comm);
    auto gout_extents = compute_global_extents(out, m_out_topology, m_comm);

    auto non_negative_axes =
        KokkosFFT::Impl::convert_base_int_type<std::size_t>(
            KokkosFFT::Impl::convert_negative_axes(axes, DIM));

    m_block_analyses = std::make_unique<PencilBlockAnalysesType>(
        in_extents, gin_extents, gout_extents, in_topology, out_topology,
        non_negative_axes, m_comm);

    KOKKOSFFT_THROW_IF(!(m_block_analyses->m_block_infos.size() >= 1 &&
                         m_block_analyses->m_block_infos.size() <= 4),
                       "Number blocks must be in [1, 4]");

    // Allocate buffer views
    std::size_t complex_alloc_size =
        (m_block_analyses->m_max_buffer_size + 1) / 2;
    m_send_buffer_allocation =
        AllocationViewType("send_buffer_allocation", complex_alloc_size);
    m_recv_buffer_allocation =
        AllocationViewType("recv_buffer_allocation", complex_alloc_size);

    std::size_t nb_blocks = m_block_analyses->m_block_infos.size();
    for (std::size_t block_idx = 0; block_idx < nb_blocks; ++block_idx) {
      set_block_impl(block_idx);
    }

    KOKKOSFFT_THROW_IF(m_in_T.size() == 0 || m_out_T.size() == 0,
                       "Internal views are not set");

    // Using aligned data
    m_fft_buffer_allocations =
        allocate_fft_work_areas<complex_type, AllocationViewType>(*m_fft_stage);
  }

  ~PencilInternalPlan() {
    // Free sub communicators
    for (auto& comm : m_cart_comms) {
      ::MPI_Comm_free(&comm);
    }

    // Then free cart communicator
    ::MPI_Comm_free(&m_cart_comm);
  }

  void forward(const InViewType& in, const OutViewType& out) const {
    std::size_t nb_blocks = m_block_analyses->m_block_infos.size();
    for (std::size_t block_idx = 0; block_idx < nb_blocks; ++block_idx) {
      forward_impl(in, out, block_idx);
    }
  }

  void backward(const OutViewType& out, const InViewType& in) const {
    int64_t nb_blocks = m_block_analyses->m_block_infos.size();
    for (int64_t block_idx = nb_blocks - 1; block_idx >= 0; --block_idx) {
      backward_impl(out, in, block_idx);
    }
  }

 private:
  void set_block_impl(const std::size_t block_idx) {
    auto block      = m_block_analyses->m_block_infos.at(block_idx);
    auto block_type = block.m_block_type;

    if (block_type == BlockType::FFT) {
      m_in_T = InViewType(
          reinterpret_cast<in_value_type*>(m_send_buffer_allocation.data()),
          KokkosFFT::Impl::create_layout<LayoutType>(block.m_in_extents));
      m_out_T = OutViewType(
          m_recv_buffer_allocation.data(),
          KokkosFFT::Impl::create_layout<LayoutType>(block.m_out_extents));

      m_fft_stage =
          std::make_unique<FFTStageType>(m_exec_space, m_in_T, m_out_T, 1);

      // Local transposes are needed only for the first and the last blocks
      if (block_idx == 0) m_fft_stage->set_in_map(block.m_in_map);
      if (block_idx == m_block_analyses->m_block_infos.size() - 1) {
        m_fft_stage->set_out_map(block.m_out_map);
      }
    } else {
      if (block.m_block_idx == 2) {
        m_out_T2 = OutViewType(
            m_send_buffer_allocation.data(),
            KokkosFFT::Impl::create_layout<LayoutType>(block.m_in_extents));
      }
      m_trans_blocks.push_back(std::make_unique<TransBlockType>(
          m_exec_space, block.m_buffer_extents, block.m_in_map, block.m_in_axis,
          block.m_out_map, block.m_out_axis));
    }
  }

  template <typename InType, typename OutType>
  void forward_impl(const InType& in, const OutType& out,
                    const std::size_t block_idx) const {
    auto block      = m_block_analyses->m_block_infos.at(block_idx);
    auto block_type = block.m_block_type;
    if (block_idx == 0) {
      if (block_type == BlockType::FFT) {
        OutViewType out_view =
            block_idx == m_block_analyses->m_block_infos.size() - 1 ? out
                                                                    : m_out_T;
        m_fft_stage->forward(in, out_view, m_in_T, m_out_T);
      } else if (block_type == BlockType::Transpose) {
        (*m_trans_blocks.at(block.m_block_idx))(
            *m_cart_tpl_comms.at(block.m_comm_axis), in, m_in_T,
            m_send_buffer_allocation, m_recv_buffer_allocation,
            KokkosFFT::Direction::forward);
      }
    } else {
      if (block_type == BlockType::FFT) {
        OutViewType out_view =
            block_idx == m_block_analyses->m_block_infos.size() - 1 ? out
                                                                    : m_out_T;
        m_fft_stage->forward(m_in_T, out_view, m_in_T, m_out_T);
      } else if (block_type == BlockType::Transpose) {
        OutViewType out_view = m_out_T, out_view2 = m_out_T2;
        if (block_idx == m_block_analyses->m_block_infos.size() - 1) {
          if (block.m_block_idx == 0 || block.m_block_idx == 1) {
            out_view  = m_out_T;
            out_view2 = out;
          } else {
            out_view  = m_out_T2;
            out_view2 = out;
          }
        }

        AllocationViewType send_buffer = m_send_buffer_allocation,
                           recv_buffer = m_recv_buffer_allocation;

        if (KokkosFFT::Impl::are_aliasing(out_view.data(),
                                          send_buffer.data())) {
          send_buffer = m_recv_buffer_allocation;
          recv_buffer = m_send_buffer_allocation;
        }

        (*m_trans_blocks.at(block.m_block_idx))(
            *m_cart_tpl_comms.at(block.m_comm_axis), out_view, out_view2,
            send_buffer, recv_buffer, KokkosFFT::Direction::forward);
      }
    }
  }

  template <typename InType, typename OutType>
  void backward_impl(const OutType& out, const InType& in,
                     const std::size_t block_idx) const {
    auto block      = m_block_analyses->m_block_infos.at(block_idx);
    auto block_type = block.m_block_type;
    if (block_idx == 0) {
      if (block_type == BlockType::FFT) {
        OutViewType out_view =
            block_idx == m_block_analyses->m_block_infos.size() - 1 ? out
                                                                    : m_out_T;
        m_fft_stage->backward(out_view, in, m_in_T, m_out_T);
      } else if (block_type == BlockType::Transpose) {
        (*m_trans_blocks.at(block.m_block_idx))(
            *m_cart_tpl_comms.at(block.m_comm_axis), m_in_T, in,
            m_recv_buffer_allocation, m_send_buffer_allocation,
            KokkosFFT::Direction::backward);
      }
    } else {
      if (block_type == BlockType::FFT) {
        OutViewType out_view =
            block_idx == m_block_analyses->m_block_infos.size() - 1 ? out
                                                                    : m_out_T;
        m_fft_stage->backward(out_view, m_in_T, m_in_T, m_out_T);
      } else if (block_type == BlockType::Transpose) {
        OutViewType out_view = m_out_T, out_view2 = m_out_T2;
        if (block_idx == m_block_analyses->m_block_infos.size() - 1) {
          if (block.m_block_idx == 0 || block.m_block_idx == 1) {
            out_view  = m_out_T;
            out_view2 = out;
          } else {
            out_view  = m_out_T2;
            out_view2 = out;
          }
        }

        AllocationViewType send_buffer = m_send_buffer_allocation,
                           recv_buffer = m_recv_buffer_allocation;

        if (KokkosFFT::Impl::are_aliasing(out_view.data(),
                                          recv_buffer.data())) {
          send_buffer = m_recv_buffer_allocation;
          recv_buffer = m_send_buffer_allocation;
        }

        (*m_trans_blocks.at(block.m_block_idx))(
            *m_cart_tpl_comms.at(block.m_comm_axis), out_view2, out_view,
            send_buffer, recv_buffer, KokkosFFT::Direction::backward);
      }
    }
  }
};

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          typename InLayoutType, typename OutLayoutType>
struct PencilInternalPlan<ExecutionSpace, InViewType, OutViewType, 2,
                          InLayoutType, OutLayoutType> {
  using execSpace      = ExecutionSpace;
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;
  using float_type   = KokkosFFT::Impl::base_floating_point_type<in_value_type>;
  using complex_type = Kokkos::complex<float_type>;
  using LayoutType   = typename InViewType::array_layout;

  static constexpr std::size_t DIM = InViewType::rank();

  using axes_type     = KokkosFFT::axis_type<2>;
  using topology_type = KokkosFFT::shape_type<DIM>;
  using in_topology_type =
      Topology<std::size_t, InViewType::rank(), InLayoutType>;
  using out_topology_type =
      Topology<std::size_t, OutViewType::rank(), OutLayoutType>;

  using PencilBlockAnalysesType =
      PencilBlockAnalysesInternal<in_value_type, LayoutType, std::size_t, DIM,
                                  2, InLayoutType, OutLayoutType>;

  using AllocationViewType =
      Kokkos::View<complex_type*, LayoutType, ExecutionSpace>;

  // Type for transpose Block
  using CommType       = KokkosFFT::Distributed::Impl::TplComm<ExecutionSpace>;
  using TransBlockType = TransBlock<ExecutionSpace, DIM>;

  // First FFT to perform can be Real to Complex
  using FFTStageType0 = FFTStage<ExecutionSpace, InViewType, OutViewType>;

  // Following FFTs to perform must be Complex to Complex
  using FFTStageType1 = FFTStage<ExecutionSpace, OutViewType, OutViewType>;

  execSpace m_exec_space;
  axes_type m_axes;
  in_topology_type m_in_topology;
  out_topology_type m_out_topology;
  MPI_Comm m_comm;

  // Cartesian Communicators
  MPI_Comm m_cart_comm;
  std::vector<MPI_Comm> m_cart_comms;
  std::vector<std::unique_ptr<CommType>> m_cart_tpl_comms;

  // Analyse topology
  std::unique_ptr<PencilBlockAnalysesType> m_block_analyses;

  // Buffer view types
  InViewType m_in_T;
  OutViewType m_out_T, m_fft_view;

  // Buffer Allocations
  using ptr_pair_type = std::pair<complex_type*, complex_type*>;
  AllocationViewType m_send_buffer_allocation, m_recv_buffer_allocation;
  std::vector<AllocationViewType> m_fft_buffer_allocations;

  std::vector<ptr_pair_type> m_in_out_ptr;

  // Internal transpose blocks
  std::vector<std::unique_ptr<TransBlockType>> m_trans_blocks;

  // Internal FFT stages
  std::unique_ptr<FFTStageType0> m_fft_stage0;
  std::vector<std::unique_ptr<FFTStageType1>> m_fft_stages;

 public:
  explicit PencilInternalPlan(const ExecutionSpace& exec_space,
                              const InViewType& in, const OutViewType& out,
                              const axes_type& axes,
                              const in_topology_type& in_topology,
                              const out_topology_type& out_topology,
                              const MPI_Comm& comm)
      : m_exec_space(exec_space),
        m_axes(axes),
        m_in_topology(in_topology),
        m_out_topology(out_topology),
        m_comm(comm) {
    create_pencil_communicators(m_exec_space, m_in_topology, m_comm,
                                m_cart_comm, m_cart_comms, m_cart_tpl_comms);

    // First get global shape to define buffer and next shape
    auto in_extents   = KokkosFFT::Impl::extract_extents(in);
    auto gin_extents  = compute_global_extents(in, m_in_topology, m_comm);
    auto gout_extents = compute_global_extents(out, m_out_topology, m_comm);

    auto non_negative_axes =
        KokkosFFT::Impl::convert_base_int_type<std::size_t>(
            KokkosFFT::Impl::convert_negative_axes(axes, DIM));

    m_block_analyses = std::make_unique<PencilBlockAnalysesType>(
        in_extents, gin_extents, gout_extents, in_topology, out_topology,
        non_negative_axes, m_comm);

    KOKKOSFFT_THROW_IF(!(m_block_analyses->m_block_infos.size() >= 1 &&
                         m_block_analyses->m_block_infos.size() <= 6),
                       "Number blocks must be in [1, 6]");

    // Allocate buffer views
    std::size_t complex_alloc_size =
        (m_block_analyses->m_max_buffer_size + 1) / 2;
    m_send_buffer_allocation =
        AllocationViewType("send_buffer_allocation", complex_alloc_size);
    m_recv_buffer_allocation =
        AllocationViewType("recv_buffer_allocation", complex_alloc_size);

    std::size_t nb_blocks = m_block_analyses->m_block_infos.size();
    for (std::size_t block_idx = 0; block_idx < nb_blocks; ++block_idx) {
      set_block_impl(block_idx);
    }

    KOKKOSFFT_THROW_IF(m_in_T.size() == 0 || m_out_T.size() == 0,
                       "Internal views are not set");

    KOKKOSFFT_THROW_IF(m_in_out_ptr.size() != nb_blocks,
                       "m_in_out_ptr must have the size of nb_blocks");

    // Using aligned data
    m_fft_buffer_allocations =
        allocate_fft_work_areas<complex_type, AllocationViewType>(*m_fft_stage0,
                                                                  m_fft_stages);
  }

  ~PencilInternalPlan() {
    // Free sub communicators
    for (auto& comm : m_cart_comms) {
      ::MPI_Comm_free(&comm);
    }

    // Then free cart communicator
    ::MPI_Comm_free(&m_cart_comm);
  }

  void forward(const InViewType& in, const OutViewType& out) const {
    std::size_t nb_blocks = m_block_analyses->m_block_infos.size();
    for (std::size_t block_idx = 0; block_idx < nb_blocks; ++block_idx) {
      forward_impl(in, out, block_idx);
    }
  }

  void backward(const OutViewType& out, const InViewType& in) const {
    int64_t nb_blocks = m_block_analyses->m_block_infos.size();
    for (int64_t block_idx = nb_blocks - 1; block_idx >= 0; --block_idx) {
      backward_impl(out, in, block_idx);
    }
  }

 private:
  /// \brief Whether the output of the last FFT stage is locally transposed
  bool has_out_map() const {
    return m_fft_stages.empty() ? m_fft_stage0->has_out_map()
                                : m_fft_stages.back()->has_out_map();
  }

  void set_block_impl(const std::size_t block_idx) {
    auto block      = m_block_analyses->m_block_infos.at(block_idx);
    auto block_type = block.m_block_type;

    if (block_type == BlockType::FFT) {
      // Local transposes are needed only for the first and the last blocks
      auto set_maps = [&](auto& stage) {
        if (block_idx == 0) stage.set_in_map(block.m_in_map);
        if (block_idx == m_block_analyses->m_block_infos.size() - 1) {
          stage.set_out_map(block.m_out_map);
        }
      };

      if (block.m_block_idx == 0) {
        m_in_T = InViewType(
            reinterpret_cast<in_value_type*>(m_send_buffer_allocation.data()),
            KokkosFFT::Impl::create_layout<LayoutType>(block.m_in_extents));
        m_out_T = OutViewType(
            m_recv_buffer_allocation.data(),
            KokkosFFT::Impl::create_layout<LayoutType>(block.m_out_extents));

        m_fft_stage0 = std::make_unique<FFTStageType0>(
            m_exec_space, m_in_T, m_out_T, block.m_fft_dim);
        set_maps(*m_fft_stage0);
        m_in_out_ptr.push_back(ptr_pair_type{nullptr, m_out_T.data()});
      } else {
        auto* last_ptr = m_in_out_ptr.back().second;

        m_fft_view = OutViewType(
            last_ptr,
            KokkosFFT::Impl::create_layout<LayoutType>(block.m_out_extents));
        m_fft_stages.push_back(std::make_unique<FFTStageType1>(
            m_exec_space, m_fft_view, m_fft_view, 1));
        set_maps(*m_fft_stages.back());
        m_in_out_ptr.push_back(
            ptr_pair_type{m_fft_view.data(), m_fft_view.data()});
      }
    } else {
      m_trans_blocks.push_back(std::make_unique<TransBlockType>(
          m_exec_space, block.m_buffer_extents, block.m_in_map, block.m_in_axis,
          block.m_out_map, block.m_out_axis));

      if (m_in_out_ptr.size() == 0) {
        m_in_out_ptr.push_back(
            ptr_pair_type{nullptr, m_send_buffer_allocation.data()});
      } else {
        auto* last_out = m_in_out_ptr.back().second;
        auto* next_out = KokkosFFT::Impl::are_aliasing(
                             last_out, m_send_buffer_allocation.data())
                             ? m_recv_buffer_allocation.data()
                             : m_send_buffer_allocation.data();
        m_in_out_ptr.push_back(ptr_pair_type{last_out, next_out});
      }
    }
  }

  template <typename InType, typename OutType>
  void forward_impl(const InType& in, const OutType& out,
                    const int64_t block_idx) const {
    auto block      = m_block_analyses->m_block_infos.at(block_idx);
    auto block_type = block.m_block_type;

    int64_t last_block_idx = m_block_analyses->m_block_infos.size() - 1;
    if (block_idx == 0) {
      if (block_type == BlockType::FFT) {
        OutViewType out_view = m_out_T;
        if (block_idx == last_block_idx) {
          out_view = out;
        }
        m_fft_stage0->forward(in, out_view, m_in_T, m_out_T);
      } else if (block_type == BlockType::Transpose) {
        (*m_trans_blocks.at(block.m_block_idx))(
            *m_cart_tpl_comms.at(block.m_comm_axis), in, m_in_T,
            m_send_buffer_allocation, m_recv_buffer_allocation,
            KokkosFFT::Direction::forward);
      }
    } else {
      if (block_type == BlockType::FFT) {
        if (block.m_block_idx == 0) {
          OutViewType out_view = m_out_T;
          if (block_idx == last_block_idx) {
            out_view = out;
          }
          m_fft_stage0->forward(m_in_T, out_view, m_in_T, m_out_T);
        } else {
          auto* current_in = m_in_out_ptr.at(block_idx).first;
          OutViewType cin_view(
              current_in,
              KokkosFFT::Impl::create_layout<LayoutType>(block.m_in_extents));

          OutViewType cout_view = cin_view;
          if (block_idx == last_block_idx) {
            if (!has_out_map()) {
              cin_view = out;
            }
            cout_view = out;
          }
          m_fft_stages.at(0)->forward(cin_view, cout_view, m_fft_view,
                                      m_fft_view);
        }
      } else if (block_type == BlockType::Transpose) {
        auto *current_in  = m_in_out_ptr.at(block_idx).first,
             *current_out = m_in_out_ptr.at(block_idx).second;
        OutViewType out_view(
            current_in,
            KokkosFFT::Impl::create_layout<LayoutType>(block.m_in_extents));
        OutViewType out_view2(
            current_out,
            KokkosFFT::Impl::create_layout<LayoutType>(block.m_out_extents));

        if ((block_idx == last_block_idx) ||
            ((block_idx == (last_block_idx - 1)) && !has_out_map() &&
             m_block_analyses->m_block_infos.back().m_block_type ==
                 BlockType::FFT)) {
          out_view2 = out;
        }

        AllocationViewType send_buffer = m_send_buffer_allocation,
                           recv_buffer = m_recv_buffer_allocation;

        if (KokkosFFT::Impl::are_aliasing(out_view.data(),
                                          send_buffer.data())) {
          send_buffer = m_recv_buffer_allocation;
          recv_buffer = m_send_buffer_allocation;
        }

        (*m_trans_blocks.at(block.m_block_idx))(
            *m_cart_tpl_comms.at(block.m_comm_axis), out_view, out_view2,
            send_buffer, recv_buffer, KokkosFFT::Direction::forward);
      }
    }
  }

  template <typename InType, typename OutType>
  void backward_impl(const OutType& out, const InType& in,
                     const int64_t block_idx) const {
    auto block      = m_block_analyses->m_block_infos.at(block_idx);
    auto block_type = block.m_block_type;

    int64_t last_block_idx = m_block_analyses->m_block_infos.size() - 1;
    if (block_idx == 0) {
      if (block_type == BlockType::FFT) {
        OutViewType out_view = block_idx == last_block_idx ? out : m_out_T;
        m_fft_stage0->backward(out_view, in, m_in_T, m_out_T);
      } else if (block_type == BlockType::Transpose) {
        (*m_trans_blocks.at(block.m_block_idx))(
            *m_cart_tpl_comms.at(block.m_comm_axis), m_in_T, in,
            m_recv_buffer_allocation, m_send_buffer_allocation,
            KokkosFFT::Direction::backward);
      }
    } else {
      if (block_type == BlockType::FFT) {
        if (block.m_block_idx == 0) {
          OutViewType out_view = block_idx == last_block_idx ? out : m_out_T;
          m_fft_stage0->backward(out_view, m_in_T, m_in_T, m_out_T);
        } else {
          auto* current_out = m_in_out_ptr.at(block_idx).second;
          OutViewType cin_view(
              current_out,
              KokkosFFT::Impl::create_layout<LayoutType>(block.m_in_extents));
          OutViewType cout_view = cin_view;
          if (block_idx == last_block_idx) {
            cout_view = out;
            if (!has_out_map()) {
              cin_view = out;
            }
          }
          m_fft_stages.at(0)->backward(cout_view, cin_view, m_fft_view,
                                       m_fft_view);
        }
      } else if (block_type == BlockType::Transpose) {
        auto *current_in  = m_in_out_ptr.at(block_idx).first,
             *current_out = m_in_out_ptr.at(block_idx).second;

        OutViewType out_view(
            current_in,
            KokkosFFT::Impl::create_layout<LayoutType>(block.m_in_extents));
        OutViewType out_view2(
            current_out,
            KokkosFFT::Impl::create_layout<LayoutType>(block.m_out_extents));

        if ((block_idx == last_block_idx) ||
            ((block_idx == (last_block_idx - 1)) && !has_out_map() &&
             m_block_analyses->m_block_infos.back().m_block_type ==
                 BlockType::FFT)) {
          out_view2 = out;
        }

        AllocationViewType send_buffer = m_send_buffer_allocation,
                           recv_buffer = m_recv_buffer_allocation;

        if (KokkosFFT::Impl::are_aliasing(out_view.data(),
                                          recv_buffer.data())) {
          send_buffer = m_recv_buffer_allocation;
          recv_buffer = m_send_buffer_allocation;
        }

        (*m_trans_blocks.at(block.m_block_idx))(
            *m_cart_tpl_comms.at(block.m_comm_axis), out_view2, out_view,
            send_buffer, recv_buffer, KokkosFFT::Direction::backward);
      }
    }
  }
};

/// \brief Internal plan for 3D FFT with slab decomposition
/// 1. 1D FFT + Transpose + 1D FFT + Transpose + 1D FFT (+ Transpose)
/// 2. 2D FFT + Transpose + 1D FFT (+ Transpose)
/// 3. Transpose + 3D FFT (+ Transpose)
/// 4. Transpose + 2D FFT + Transpose + 1D FFT (+ Transpose)
/// 5. Transpose + 1D FFT + Transpose + 1D FFT + Transpose + 1D FFT (+
/// Transpose)
///
/// \tparam ExecutionSpace
/// \tparam InViewType
/// \tparam OutViewType
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          typename InLayoutType, typename OutLayoutType>
struct PencilInternalPlan<ExecutionSpace, InViewType, OutViewType, 3,
                          InLayoutType, OutLayoutType> {
  using execSpace      = ExecutionSpace;
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;
  using float_type   = KokkosFFT::Impl::base_floating_point_type<in_value_type>;
  using complex_type = Kokkos::complex<float_type>;
  using LayoutType   = typename InViewType::array_layout;

  static constexpr std::size_t DIM = InViewType::rank();

  using axes_type     = KokkosFFT::axis_type<3>;
  using topology_type = KokkosFFT::shape_type<DIM>;
  using in_topology_type =
      Topology<std::size_t, InViewType::rank(), InLayoutType>;
  using out_topology_type =
      Topology<std::size_t, OutViewType::rank(), OutLayoutType>;

  using PencilBlockAnalysesType =
      PencilBlockAnalysesInternal<in_value_type, LayoutType, std::size_t, DIM,
                                  3, InLayoutType, OutLayoutType>;

  using AllocationViewType =
      Kokkos::View<complex_type*, LayoutType, ExecutionSpace>;

  // Type for transpose Block
  using CommType       = KokkosFFT::Distributed::Impl::TplComm<ExecutionSpace>;
  using TransBlockType = TransBlock<ExecutionSpace, DIM>;

  // First FFT to perform can be Real to Complex
  using FFTStageType0 = FFTStage<ExecutionSpace, InViewType, OutViewType>;

  // Following FFTs to perform must be Complex to Complex
  using FFTStageType1 = FFTStage<ExecutionSpace, OutViewType, OutViewType>;

  execSpace m_exec_space;
  axes_type m_axes;
  in_topology_type m_in_topology;
  out_topology_type m_out_topology;
  MPI_Comm m_comm;

  // Cartesian Communicators
  MPI_Comm m_cart_comm;
  std::vector<MPI_Comm> m_cart_comms;
  std::vector<std::unique_ptr<CommType>> m_cart_tpl_comms;

  // Analyse topology
  std::unique_ptr<PencilBlockAnalysesType> m_block_analyses;

  // Buffer view types
  InViewType m_in_T;
  OutViewType m_out_T, m_fft_view0, m_fft_view1;

  // Buffer Allocations
  using ptr_pair_type = std::pair<complex_type*, complex_type*>;
  AllocationViewType m_send_buffer_allocation, m_recv_buffer_allocation;
  std::vector<AllocationViewType> m_fft_buffer_allocations;

  std::vector<ptr_pair_type> m_in_out_ptr;

  // Internal transpose blocks
  std::vector<std::unique_ptr<TransBlockType>> m_trans_blocks;

  // Internal FFT stages
  std::unique_ptr<FFTStageType0> m_fft_stage0;
  std::vector<std::unique_ptr<FFTStageType1>> m_fft_stages;

 public:
  explicit PencilInternalPlan(const ExecutionSpace& exec_space,
                              const InViewType& in, const OutViewType& out,
                              const axes_type& axes,
                              const in_topology_type& in_topology,
                              const out_topology_type& out_topology,
                              const MPI_Comm& comm)
      : m_exec_space(exec_space),
        m_axes(axes),
        m_in_topology(in_topology),
        m_out_topology(out_topology),
        m_comm(comm) {
    create_pencil_communicators(m_exec_space, m_in_topology, m_comm,
                                m_cart_comm, m_cart_comms, m_cart_tpl_comms);

    // First get global shape to define buffer and next shape
    auto in_extents   = KokkosFFT::Impl::extract_extents(in);
    auto gin_extents  = compute_global_extents(in, m_in_topology, m_comm);
    auto gout_extents = compute_global_extents(out, m_out_topology, m_comm);

    auto non_negative_axes =
        KokkosFFT::Impl::convert_base_int_type<std::size_t>(
            KokkosFFT::Impl::convert_negative_axes(axes, DIM));

    m_block_analyses = std::make_unique<PencilBlockAnalysesType>(
        in_extents, gin_extents, gout_extents, in_topology, out_topology,
        non_negative_axes, m_comm);

    KOKKOSFFT_THROW_IF(!(m_block_analyses->m_block_infos.size() >= 1 &&
                         m_block_analyses->m_block_infos.size() <= 8),
                       "Number blocks must be in [1, 8]");

    // Allocate buffer views
    std::size_t complex_alloc_size =
        (m_block_analyses->m_max_buffer_size + 1) / 2;
    m_send_buffer_allocation =
        AllocationViewType("send_buffer_allocation", complex_alloc_size);
    m_recv_buffer_allocation =
        AllocationViewType("recv_buffer_allocation", complex_alloc_size);

    std::size_t nb_blocks = m_block_analyses->m_block_infos.size();
    for (std::size_t block_idx = 0; block_idx < nb_blocks; ++block_idx) {
      set_block_impl(block_idx);
    }

    KOKKOSFFT_THROW_IF(m_in_T.size() == 0 || m_out_T.size() == 0,
                       "Internal views are not set");

    KOKKOSFFT_THROW_IF(m_in_out_ptr.size() != nb_blocks,
                       "m_in_out_ptr must have the size of nb_blocks");

    m_fft_buffer_allocations =
        allocate_fft_work_areas<complex_type, AllocationViewType>(*m_fft_stage0,
                                                                  m_fft_stages);
  }

  ~PencilInternalPlan() {
    // Free sub communicators
    for (auto& comm : m_cart_comms) {
      ::MPI_Comm_free(&comm);
    }

    // Then free cart communicator
    ::MPI_Comm_free(&m_cart_comm);
  }

  void forward(const InViewType& in, const OutViewType& out) const {
    std::size_t nb_blocks = m_block_analyses->m_block_infos.size();
    for (std::size_t block_idx = 0; block_idx < nb_blocks; ++block_idx) {
      forward_impl(in, out, block_idx);
    }
  }

  void backward(const OutViewType& out, const InViewType& in) const {
    int64_t nb_blocks = m_block_analyses->m_block_infos.size();
    for (int64_t block_idx = nb_blocks - 1; block_idx >= 0; --block_idx) {
      backward_impl(out, in, block_idx);
    }
  }

 private:
  /// \brief Whether the output of the last FFT stage is locally transposed
  bool has_out_map() const {
    return m_fft_stages.empty() ? m_fft_stage0->has_out_map()
                                : m_fft_stages.back()->has_out_map();
  }

  void set_block_impl(const std::size_t block_idx) {
    auto block      = m_block_analyses->m_block_infos.at(block_idx);
    auto block_type = block.m_block_type;

    if (block_type == BlockType::FFT) {
      // Local transposes are needed only for the first and the last blocks
      auto set_maps = [&](auto& stage) {
        if (block_idx == 0) stage.set_in_map(block.m_in_map);
        if (block_idx == m_block_analyses->m_block_infos.size() - 1) {
          stage.set_out_map(block.m_out_map);
        }
      };

      if (block.m_block_idx == 0) {
        m_in_T = InViewType(
            reinterpret_cast<in_value_type*>(m_send_buffer_allocation.data()),
            KokkosFFT::Impl::create_layout<LayoutType>(block.m_in_extents));
        m_out_T = OutViewType(
            m_recv_buffer_allocation.data(),
            KokkosFFT::Impl::create_layout<LayoutType>(block.m_out_extents));

        m_fft_stage0 = std::make_unique<FFTStageType0>(
            m_exec_space, m_in_T, m_out_T, block.m_fft_dim);
        set_maps(*m_fft_stage0);
        m_in_out_ptr.push_back(ptr_pair_type{nullptr, m_out_T.data()});
      } else if (block.m_block_idx == 1) {
        auto* last_ptr = m_in_out_ptr.back().second;

        m_fft_view0 = OutViewType(
            last_ptr,
            KokkosFFT::Impl::create_layout<LayoutType>(block.m_out_extents));

        m_fft_stages.push_back(std::make_unique<FFTStageType1>(
            m_exec_space, m_fft_view0, m_fft_view0, block.m_fft_dim));
        set_maps(*m_fft_stages.back());
        m_in_out_ptr.push_back(
            ptr_pair_type{m_fft_view0.data(), m_fft_view0.data()});
      } else {
        auto* last_ptr = m_in_out_ptr.back().second;

        m_fft_view1 = OutViewType(
            last_ptr,
            KokkosFFT::Impl::create_layout<LayoutType>(block.m_out_extents));
        m_fft_stages.push_back(std::make_unique<FFTStageType1>(
            m_exec_space, m_fft_view1, m_fft_view1, 1));
        set_maps(*m_fft_stages.back());
        m_in_out_ptr.push_back(
            ptr_pair_type{m_fft_view1.data(), m_fft_view1.data()});
      }
    } else {
      m_trans_blocks.push_back(std::make_unique<TransBlockType>(
          m_exec_space, block.m_buffer_extents, block.m_in_map, block.m_in_axis,
          block.m_out_map, block.m_out_axis));

      if (m_in_out_ptr.size() == 0) {
        m_in_out_ptr.push_back(
            ptr_pair_type{nullptr, m_send_buffer_allocation.data()});
      } else {
        auto* last_out = m_in_out_ptr.back().second;
        auto* next_out = KokkosFFT::Impl::are_aliasing(
                             last_out, m_send_buffer_allocation.data())
                             ? m_recv_buffer_allocation.data()
                             : m_send_buffer_allocation.data();
        m_in_out_ptr.push_back(ptr_pair_type{last_out, next_out});
      }
    }
  }

  template <typename InType, typename OutType>
  void forward_impl(const InType& in, const OutType& out,
                    const int64_t block_idx) const {
    auto block      = m_block_analyses->m_block_infos.at(block_idx);
    auto block_type = block.m_block_type;

    int64_t last_block_idx = m_block_analyses->m_block_infos.size() - 1;
    if (block_idx == 0) {
      if (block_type == BlockType::FFT) {
        OutViewType out_view = m_out_T;
        if (block_idx == last_block_idx) {
          out_view = out;
        }
        m_fft_stage0->forward(in, out_view, m_in_T, m_out_T);
      } else if (block_type == BlockType::Transpose) {
        (*m_trans_blocks.at(block.m_block_idx))(
            *m_cart_tpl_comms.at(block.m_comm_axis), in, m_in_T,
            m_send_buffer_allocation, m_recv_buffer_allocation,
            KokkosFFT::Direction::forward);
      }
    } else {
      if (block_type == BlockType::FFT) {
        if (block.m_block_idx == 0) {
          OutViewType out_view = m_out_T;
          if (block_idx == last_block_idx) {
            out_view = out;
          }
          m_fft_stage0->forward(m_in_T, out_view, m_in_T, m_out_T);
        } else {
          auto* current_in = m_in_out_ptr.at(block_idx).first;
          OutViewType cin_view(
              current_in,
              KokkosFFT::Impl::create_layout<LayoutType>(block.m_in_extents));

          OutViewType cout_view = cin_view;
          if (block_idx == last_block_idx) {
            if (!has_out_map()) {
              cin_view = out;
            }
            cout_view = out;
          }
          if (block.m_block_idx == 1) {
            m_fft_stages.at(0)->forward(cin_view, cout_view, m_fft_view0,
                                        m_fft_view0);
          } else {
            m_fft_stages.at(1)->forward(cin_view, cout_view, m_fft_view1,
                                        m_fft_view1);
          }
        }
      } else if (block_type == BlockType::Transpose) {
        auto *current_in  = m_in_out_ptr.at(block_idx).first,
             *current_out = m_in_out_ptr.at(block_idx).second;
        OutViewType out_view(
            current_in,
            KokkosFFT::Impl::create_layout<LayoutType>(block.m_in_extents));
        OutViewType out_view2(
            current_out,
            KokkosFFT::Impl::create_layout<LayoutType>(block.m_out_extents));

        if ((block_idx == last_block_idx) ||
            ((block_idx == (last_block_idx - 1)) && !has_out_map() &&
             m_block_analyses->m_block_infos.back().m_block_type ==
                 BlockType::FFT)) {
          out_view2 = out;
        }

        AllocationViewType send_buffer = m_send_buffer_allocation,
                           recv_buffer = m_recv_buffer_allocation;

        if (KokkosFFT::Impl::are_aliasing(out_view.data(),
                                          send_buffer.data())) {
          send_buffer = m_recv_buffer_allocation;
          recv_buffer = m_send_buffer_allocation;
        }

        (*m_trans_blocks.at(block.m_block_idx))(
            *m_cart_tpl_comms.at(block.m_comm_axis), out_view, out_view2,
            send_buffer, recv_buffer, KokkosFFT::Direction::forward);
      }
    }
  }

  template <typename InType, typename OutType>
  void backward_impl(const OutType& out, const InType& in,
                     const int64_t block_idx) const {
    auto block      = m_block_analyses->m_block_infos.at(block_idx);
    auto block_type = block.m_block_type;

    int64_t last_block_idx = m_block_analyses->m_block_infos.size() - 1;
    if (block_idx == 0) {
      if (block_type == BlockType::FFT) {
        OutViewType out_view = block_idx == last_block_idx ? out : m_out_T;
        m_fft_stage0->backward(out_view, in, m_in_T, m_out_T);
      } else if (block_type == BlockType::Transpose) {
        (*m_trans_blocks.at(block.m_block_idx))(
            *m_cart_tpl_comms.at(block.m_comm_axis), m_in_T, in,
            m_recv_buffer_allocation, m_send_buffer_allocation,
            KokkosFFT::Direction::backward);
      }
    } else {
      if (block_type == BlockType::FFT) {
        if (block.m_block_idx == 0) {
          OutViewType out_view = block_idx == last_block_idx ? out : m_out_T;
          m_fft_stage0->backward(out_view, m_in_T, m_in_T, m_out_T);
        } else {
          auto* current_out = m_in_out_ptr.at(block_idx).second;
          OutViewType cin_view(
              current_out,
              KokkosFFT::Impl::create_layout<LayoutType>(block.m_in_extents));
          OutViewType cout_view = cin_view;
          if (block_idx == last_block_idx) {
            cout_view = out;
            if (!has_out_map()) {
              cin_view = out;
            }
          }
          if (block.m_block_idx == 1) {
            m_fft_stages.at(0)->backward(cout_view, cin_view, m_fft_view0,
                                         m_fft_view0);
          } else {
            m_fft_stages.at(1)->backward(cout_view, cin_view, m_fft_view1,
                                         m_fft_view1);
          }
        }
      } else if (block_type == BlockType::Transpose) {
        auto *current_in  = m_in_out_ptr.at(block_idx).first,
             *current_out = m_in_out_ptr.at(block_idx).second;

        OutViewType out_view(
            current_in,
            KokkosFFT::Impl::create_layout<LayoutType>(block.m_in_extents));
        OutViewType out_view2(
            current_out,
            KokkosFFT::Impl::create_layout<LayoutType>(block.m_out_extents));

        if ((block_idx == last_block_idx) ||
            ((block_idx == (last_block_idx - 1)) && !has_out_map() &&
             m_block_analyses->m_block_infos.back().m_block_type ==
                 BlockType::FFT)) {
          out_view2 = out;
        }

        AllocationViewType send_buffer = m_send_buffer_allocation,
                           recv_buffer = m_recv_buffer_allocation;

        if (KokkosFFT::Impl::are_aliasing(out_view.data(),
                                          recv_buffer.data())) {
          send_buffer = m_recv_buffer_allocation;
          recv_buffer = m_send_buffer_allocation;
        }

        (*m_trans_blocks.at(block.m_block_idx))(
            *m_cart_tpl_comms.at(block.m_comm_axis), out_view2, out_view,
            send_buffer, recv_buffer, KokkosFFT::Direction::backward);
      }
    }
  }
};

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1, typename InLayoutType = Kokkos::LayoutRight,
          typename OutLayoutType = Kokkos::LayoutRight>
class PencilPlan : public InternalPlan<ExecutionSpace, InViewType, OutViewType,
                                       DIM, InLayoutType, OutLayoutType> {
  using InternalPlanType =
      PencilInternalPlan<ExecutionSpace, InViewType, OutViewType, DIM,
                         InLayoutType, OutLayoutType>;
  using extents_type = std::array<std::size_t, InViewType::rank()>;
  using in_topology_type =
      Topology<std::size_t, InViewType::rank(), InLayoutType>;
  using out_topology_type =
      Topology<std::size_t, OutViewType::rank(), OutLayoutType>;
  using axes_type = KokkosFFT::axis_type<DIM>;

  //! The real value type for normalization
  using normalization_float_type = double;

  //! Execution space
  ExecutionSpace m_exec_space;

  //! Internal plan
  InternalPlanType m_internal_plan;

  ///@{
  //! extents of in/out views
  extents_type m_in_extents, m_out_extents;
  ///@}

  //! aliases to parents methods
  using InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM, InLayoutType,
                     OutLayoutType>::good;
  using InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM, InLayoutType,
                     OutLayoutType>::get_norm;
  using InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM, InLayoutType,
                     OutLayoutType>::get_fft_extents;

 public:
  explicit PencilPlan(
      const ExecutionSpace& exec_space, const InViewType& in,
      const OutViewType& out, const axes_type& axes,
      const extents_type& in_topology, const extents_type& out_topology,
      const MPI_Comm& comm,
      KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward)
      : PencilPlan(exec_space, in, out, axes,
                   Topology<std::size_t, InViewType::rank()>(in_topology),
                   Topology<std::size_t, OutViewType::rank()>(out_topology),
                   comm, norm) {}

  explicit PencilPlan(
      const ExecutionSpace& exec_space, const InViewType& in,
      const OutViewType& out, const axes_type& axes,
      const in_topology_type& in_topology,
      const out_topology_type& out_topology, const MPI_Comm& comm,
      KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward)
      : InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM, InLayoutType,
                     OutLayoutType>(exec_space, in, out, axes, in_topology,
                                    out_topology, comm, norm),
        m_exec_space(exec_space),
        m_internal_plan(exec_space, in, out, axes, in_topology, out_topology,
                        comm),
        m_in_extents(KokkosFFT::Impl::extract_extents(in)),
        m_out_extents(KokkosFFT::Impl::extract_extents(out)) {
    auto in_size  = KokkosFFT::Impl::total_size(in_topology);
    auto out_size = KokkosFFT::Impl::total_size(out_topology);

    KOKKOSFFT_THROW_IF(in_size != out_size,
                       "Input and output topologies must have the same size.");

    bool is_pencil = are_specified_topologies(TopologyType::Pencil, in_topology,
                                              out_topology);
    KOKKOSFFT_THROW_IF(
        !is_pencil, "Input and output topologies must be pencil topologies.");
  }

  PencilPlan(const PencilPlan&) = default;
  void forward(const InViewType& in, const OutViewType& out) const override {
    good(in, out);
    m_internal_plan.forward(in, out);
    KokkosFFT::Impl::normalize<normalization_float_type>(
        m_exec_space, out, KokkosFFT::Direction::forward, get_norm(),
        get_fft_extents());
  }

  void backward(const OutViewType& out, const InViewType& in) const override {
    good(in, out);
    m_internal_plan.backward(out, in);
    KokkosFFT::Impl::normalize<normalization_float_type>(
        m_exec_space, in, KokkosFFT::Direction::backward, get_norm(),
        get_fft_extents());
  }

  /// \brief Get the name of the plan implementation
  /// \return Name of the plan implementation
  std::string name() const override { return std::string("PencilPlan"); }
};

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
