#ifndef KOKKOSFFT_DISTRIBUTED_SLABPLAN_HPP
#define KOKKOSFFT_DISTRIBUTED_SLABPLAN_HPP

#include <vector>
#include <memory>
#include <iostream>
#include <sstream>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_TransBlock.hpp"
#include "KokkosFFT_Distributed_Mapping.hpp"
#include "KokkosFFT_Distributed_MPI_Helper.hpp"
#include "KokkosFFT_Distributed_Helper.hpp"
#include "KokkosFFT_Distributed_Extents.hpp"
#include "KokkosFFT_Distributed_Topologies.hpp"
#include "KokkosFFT_Distributed_SlabBlockAnalyses.hpp"
#include "KokkosFFT_Distributed_InternalPlan.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM>
struct SlabInternalPlan;

/// \brief Internal plan for 1D FFT with slab decomposition
/// 1. 1D FFT
/// 1. Transpose + 1D FFT
/// 1. Transpose + 1D FFT + Transpose
///
/// \tparam ExecutionSpace
/// \tparam InViewType
/// \tparam OutViewType

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
struct SlabInternalPlan<ExecutionSpace, InViewType, OutViewType, 1> {
  using execSpace      = ExecutionSpace;
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;
  using float_type   = KokkosFFT::Impl::base_floating_point_type<in_value_type>;
  using complex_type = Kokkos::complex<float_type>;
  using LayoutType   = typename InViewType::array_layout;

  static constexpr std::size_t DIM = InViewType::rank();

  using axes_type     = KokkosFFT::axis_type<1>;
  using topology_type = KokkosFFT::shape_type<DIM>;
  using int_map_type  = std::array<int, DIM>;

  using SlabBlockAnalysesType =
      SlabBlockAnalysesInternal<in_value_type, LayoutType, std::size_t, DIM, 1>;

  using AllocationViewType =
      Kokkos::View<complex_type*, LayoutType, ExecutionSpace>;

  // Type for transpose Block
  using CommType       = KokkosFFT::Distributed::Impl::TplComm<ExecutionSpace>;
  using TransBlockType = TransBlock<ExecutionSpace, DIM>;

  using FFTForwardPlanType =
      KokkosFFT::DynPlan<ExecutionSpace, InViewType, OutViewType>;
  using FFTBackwardPlanType =
      KokkosFFT::DynPlan<ExecutionSpace, OutViewType, InViewType>;

  execSpace m_exec_space;
  axes_type m_axes;
  topology_type m_in_topology;
  topology_type m_out_topology;
  CommType m_comm;

  // Analyse topology
  std::unique_ptr<SlabBlockAnalysesType> m_block_analyses;

  // Buffer view types
  InViewType m_in_T;
  OutViewType m_out_T;

  // Buffer Allocations
  AllocationViewType m_send_buffer_allocation, m_recv_buffer_allocation,
      m_fft_buffer_allocation;

  // Internal transpose blocks
  std::vector<std::unique_ptr<TransBlockType>> m_trans_blocks;

  // Internal FFT plans
  std::unique_ptr<FFTForwardPlanType> m_forward_plan;
  std::unique_ptr<FFTBackwardPlanType> m_backward_plan;

  // Mappings for local transpose
  int_map_type m_map_forward_in = {}, m_map_forward_out = {},
               m_map_backward_in = {}, m_map_backward_out = {};

 public:
  explicit SlabInternalPlan(const ExecutionSpace& exec_space,
                            const InViewType& in, const OutViewType& out,
                            const axes_type& axes,
                            const topology_type& in_topology,
                            const topology_type& out_topology,
                            const MPI_Comm& comm)
      : m_exec_space(exec_space),
        m_axes(axes),
        m_in_topology(in_topology),
        m_out_topology(out_topology),
        m_comm(comm, exec_space) {
    KOKKOSFFT_THROW_IF(
        !are_valid_extents(in, out, axes, in_topology, out_topology, comm),
        "Extents are not valid");

    // First get global shape to define buffer and next shape
    auto in_extents  = KokkosFFT::Impl::extract_extents(in);
    auto out_extents = KokkosFFT::Impl::extract_extents(out);

    auto gin_extents  = get_global_shape(in, m_in_topology, comm);
    auto gout_extents = get_global_shape(out, m_out_topology, comm);

    auto non_negative_axes =
        KokkosFFT::Impl::convert_base_int_type<std::size_t>(
            KokkosFFT::Impl::convert_negative_axes(axes, DIM));

    m_block_analyses = std::make_unique<SlabBlockAnalysesType>(
        in_extents, out_extents, gin_extents, gout_extents, in_topology,
        out_topology, non_negative_axes, comm);

    KOKKOSFFT_THROW_IF(!(m_block_analyses->m_block_infos.size() >= 1 &&
                         m_block_analyses->m_block_infos.size() <= 3),
                       "Number blocks must be in [1, 3]");

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
    auto workspace_size =
        KokkosFFT::compute_required_workspace_size<complex_type>(
            *m_forward_plan, *m_backward_plan);
    m_fft_buffer_allocation =
        AllocationViewType("fft_buffer_allocation", workspace_size);
    m_forward_plan->set_work_area(m_fft_buffer_allocation);
    m_backward_plan->set_work_area(m_fft_buffer_allocation);
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
  template <typename InType, typename OutType>
  void forward_fft(const InType& in, const OutType& out) const {
    if (m_map_forward_in == int_map_type{} &&
        m_map_forward_out == int_map_type{}) {
      KokkosFFT::execute(*m_forward_plan, in, out,
                         KokkosFFT::Normalization::none);
    } else if (m_map_forward_in == int_map_type{} &&
               m_map_forward_out != int_map_type{}) {
      KokkosFFT::execute(*m_forward_plan, in, m_out_T,
                         KokkosFFT::Normalization::none);
      KokkosFFT::Impl::transpose(m_exec_space, m_out_T, out, m_map_forward_out,
                                 true);
    } else if (m_map_forward_in != int_map_type{} &&
               m_map_forward_out == int_map_type{}) {
      KokkosFFT::Impl::transpose(m_exec_space, in, m_in_T, m_map_forward_in,
                                 true);
      KokkosFFT::execute(*m_forward_plan, m_in_T, out,
                         KokkosFFT::Normalization::none);
    } else if (m_map_forward_in != int_map_type{} &&
               m_map_forward_out != int_map_type{}) {
      KokkosFFT::Impl::transpose(m_exec_space, in, m_in_T, m_map_forward_in,
                                 true);
      KokkosFFT::execute(*m_forward_plan, m_in_T, m_out_T,
                         KokkosFFT::Normalization::none);
      KokkosFFT::Impl::transpose(m_exec_space, m_out_T, out, m_map_forward_out,
                                 true);
    }
  }

  template <typename InType, typename OutType>
  void backward_fft(const OutType& out, const InType& in) const {
    if (m_map_backward_in == int_map_type{} &&
        m_map_backward_out == int_map_type{}) {
      KokkosFFT::execute(*m_backward_plan, out, in,
                         KokkosFFT::Normalization::none);
    } else if (m_map_backward_in == int_map_type{} &&
               m_map_backward_out != int_map_type{}) {
      KokkosFFT::execute(*m_backward_plan, out, m_in_T,
                         KokkosFFT::Normalization::none);
      KokkosFFT::Impl::transpose(m_exec_space, m_in_T, in, m_map_backward_out,
                                 true);
    } else if (m_map_backward_in != int_map_type{} &&
               m_map_backward_out == int_map_type{}) {
      KokkosFFT::Impl::transpose(m_exec_space, out, m_out_T, m_map_backward_in,
                                 true);
      KokkosFFT::execute(*m_backward_plan, m_out_T, in,
                         KokkosFFT::Normalization::none);
    } else if (m_map_backward_in != int_map_type{} &&
               m_map_backward_out != int_map_type{}) {
      KokkosFFT::Impl::transpose(m_exec_space, out, m_out_T, m_map_backward_in,
                                 true);
      KokkosFFT::execute(*m_backward_plan, m_out_T, m_in_T,
                         KokkosFFT::Normalization::none);
      KokkosFFT::Impl::transpose(m_exec_space, m_in_T, in, m_map_backward_out,
                                 true);
    }
  }

  void set_block_impl(const std::size_t block_idx) {
    auto src_map    = KokkosFFT::Impl::index_sequence<std::size_t, DIM, 0>();
    auto block      = m_block_analyses->m_block_infos.at(block_idx);
    auto block_type = block.m_block_type;

    if (block_type == BlockType::FFT) {
      m_in_T = InViewType(
          reinterpret_cast<in_value_type*>(m_send_buffer_allocation.data()),
          KokkosFFT::Impl::create_layout<LayoutType>(block.m_in_extents));
      m_out_T = OutViewType(
          m_recv_buffer_allocation.data(),
          KokkosFFT::Impl::create_layout<LayoutType>(block.m_out_extents));

      m_forward_plan = std::make_unique<FFTForwardPlanType>(
          m_exec_space, m_in_T, m_out_T, KokkosFFT::Direction::forward, 1);
      m_backward_plan = std::make_unique<FFTBackwardPlanType>(
          m_exec_space, m_out_T, m_in_T, KokkosFFT::Direction::backward, 1);
      if (block_idx == 0) {
        // In this case, input data needed to be transposed locally
        if (block.m_in_map != src_map) {
          for (std::size_t i = 0; i < DIM; ++i) {
            m_map_forward_in.at(i) = block.m_in_map.at(i);
            m_map_backward_out.at(i) =
                KokkosFFT::Impl::get_index(block.m_in_map, i);
          }
        }
      }

      if (block_idx == m_block_analyses->m_block_infos.size() - 1) {
        if (block.m_out_map != src_map) {
          for (std::size_t i = 0; i < DIM; ++i) {
            m_map_forward_out.at(i) =
                KokkosFFT::Impl::get_index(block.m_out_map, i);
            m_map_backward_in.at(i) = block.m_out_map.at(i);
          }
        }
      }
    } else {
      m_trans_blocks.push_back(std::make_unique<TransBlockType>(
          m_exec_space, block.m_buffer_extents, block.m_in_map, block.m_in_axis,
          block.m_out_map, block.m_out_axis));
    }
  }

  template <typename InType, typename OutType>
  void forward_impl(const InType& in, const OutType& out,
                    const std::size_t block_idx) const {
    auto block           = m_block_analyses->m_block_infos.at(block_idx);
    auto block_type      = block.m_block_type;
    InViewType in_view   = m_in_T;
    OutViewType out_view = m_out_T, out_view_T = m_out_T;

    if (block_idx == m_block_analyses->m_block_infos.size() - 1) {
      out_view   = block_type == BlockType::FFT ? out : m_out_T;
      out_view_T = out;
    }

    if (block_idx == 0) {
      if (block_type == BlockType::FFT) {
        forward_fft(in, out_view);
      } else if (block_type == BlockType::Transpose) {
        (*m_trans_blocks.at(block.m_block_idx))(
            m_comm, in, in_view, m_send_buffer_allocation,
            m_recv_buffer_allocation, KokkosFFT::Direction::forward);
      }
    } else {
      if (block_type == BlockType::FFT) {
        forward_fft(in_view, out_view);
      } else if (block_type == BlockType::Transpose) {
        (*m_trans_blocks.at(block.m_block_idx))(
            m_comm, out_view, out_view_T, m_send_buffer_allocation,
            m_recv_buffer_allocation, KokkosFFT::Direction::forward);
      }
    }
  }

  template <typename InType, typename OutType>
  void backward_impl(const OutType& out, const InType& in,
                     const std::size_t block_idx) const {
    auto block         = m_block_analyses->m_block_infos.at(block_idx);
    auto block_type    = block.m_block_type;
    InViewType in_view = m_in_T, in_view_T = m_in_T;
    OutViewType out_view = m_out_T;

    if (block_idx == 0) {
      in_view   = block_type == BlockType::FFT ? in : m_in_T;
      in_view_T = in;
    }

    if (block_idx == m_block_analyses->m_block_infos.size() - 1) {
      if (block_type == BlockType::FFT) {
        backward_fft(out, in_view);
      } else if (block_type == BlockType::Transpose) {
        (*m_trans_blocks.at(block.m_block_idx))(
            m_comm, out, out_view, m_recv_buffer_allocation,
            m_send_buffer_allocation, KokkosFFT::Direction::backward);
      }
    } else {
      if (block_type == BlockType::FFT) {
        backward_fft(out_view, in_view);
      } else if (block_type == BlockType::Transpose) {
        (*m_trans_blocks.at(block.m_block_idx))(
            m_comm, in_view, in_view_T, m_recv_buffer_allocation,
            m_send_buffer_allocation, KokkosFFT::Direction::backward);
      }
    }
  }
};

/// \brief Internal plan for 2D FFT with slab decomposition
/// 1. 1D FFT + Transpose + 1D FFT (+ Transpose)
/// 2. Transpose + 2D FFT (+ Transpose)
/// 3. Transpose + 1D FFT + Transpose + 1D FFT (+ Transpose)
///
/// \tparam ExecutionSpace
/// \tparam InViewType
/// \tparam OutViewType

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
struct SlabInternalPlan<ExecutionSpace, InViewType, OutViewType, 2> {
  using execSpace      = ExecutionSpace;
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;
  using float_type   = KokkosFFT::Impl::base_floating_point_type<in_value_type>;
  using complex_type = Kokkos::complex<float_type>;
  using LayoutType   = typename InViewType::array_layout;

  static constexpr std::size_t DIM = InViewType::rank();

  using axes_type     = KokkosFFT::axis_type<2>;
  using topology_type = KokkosFFT::shape_type<DIM>;
  using int_map_type  = std::array<int, DIM>;

  using SlabBlockAnalysesType =
      SlabBlockAnalysesInternal<in_value_type, LayoutType, std::size_t, DIM, 2>;

  using AllocationViewType =
      Kokkos::View<complex_type*, LayoutType, ExecutionSpace>;

  // Type for transpose Block
  using CommType       = KokkosFFT::Distributed::Impl::TplComm<ExecutionSpace>;
  using TransBlockType = TransBlock<ExecutionSpace, DIM>;

  // First FFT to perform can be Real to Complex
  using FFTForwardPlanType0 =
      KokkosFFT::DynPlan<ExecutionSpace, InViewType, OutViewType>;
  using FFTBackwardPlanType0 =
      KokkosFFT::DynPlan<ExecutionSpace, OutViewType, InViewType>;

  // Second FFT to perform must be Complex to Complex
  using FFTForwardPlanType1 =
      KokkosFFT::DynPlan<ExecutionSpace, OutViewType, OutViewType>;
  using FFTBackwardPlanType1 =
      KokkosFFT::DynPlan<ExecutionSpace, OutViewType, OutViewType>;

  execSpace m_exec_space;
  axes_type m_axes;
  topology_type m_in_topology;
  topology_type m_out_topology;
  CommType m_comm;

  // Analyse topology
  std::unique_ptr<SlabBlockAnalysesType> m_block_analyses;
  std::array<std::size_t, 1> m_fft_dims = {};

  // Buffer view types
  InViewType m_in_T;
  OutViewType m_out_T, m_out_T2;

  // Buffer Allocations
  AllocationViewType m_send_buffer_allocation, m_recv_buffer_allocation,
      m_fft_buffer_allocation;

#if defined(KOKKOS_ENABLE_SYCL)
  AllocationViewType m_fft_buffer_allocation1;
#endif

  // Internal transpose blocks
  std::vector<std::unique_ptr<TransBlockType>> m_trans_blocks;

  // Internal FFT plans
  std::unique_ptr<FFTForwardPlanType0> m_fft_plan0;
  std::unique_ptr<FFTForwardPlanType1> m_fft_plan1;
  std::unique_ptr<FFTBackwardPlanType0> m_ifft_plan0;
  std::unique_ptr<FFTBackwardPlanType1> m_ifft_plan1;

  // Mappings for local transpose
  int_map_type m_map_forward_in = {}, m_map_forward_out = {},
               m_map_backward_in = {}, m_map_backward_out = {};

 public:
  explicit SlabInternalPlan(const ExecutionSpace& exec_space,
                            const InViewType& in, const OutViewType& out,
                            const axes_type& axes,
                            const topology_type& in_topology,
                            const topology_type& out_topology,
                            const MPI_Comm& comm)
      : m_exec_space(exec_space),
        m_axes(axes),
        m_in_topology(in_topology),
        m_out_topology(out_topology),
        m_comm(comm, exec_space) {
    KOKKOSFFT_THROW_IF(
        !are_valid_extents(in, out, axes, in_topology, out_topology, comm),
        "Extents are not valid");

    // First get global shape to define buffer and next shape
    auto in_extents  = KokkosFFT::Impl::extract_extents(in);
    auto out_extents = KokkosFFT::Impl::extract_extents(out);

    auto gin_extents  = get_global_shape(in, m_in_topology, comm);
    auto gout_extents = get_global_shape(out, m_out_topology, comm);

    auto non_negative_axes =
        KokkosFFT::Impl::convert_base_int_type<std::size_t>(
            KokkosFFT::Impl::convert_negative_axes(axes, DIM));

    m_block_analyses = std::make_unique<SlabBlockAnalysesType>(
        in_extents, out_extents, gin_extents, gout_extents, in_topology,
        out_topology, non_negative_axes, comm);

    KOKKOSFFT_THROW_IF(m_block_analyses->m_block_infos.size() > 5,
                       "Maximum five blocks are expected");

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
    std::size_t workspace_size = 0;
#if defined(KOKKOS_ENABLE_SYCL)
    workspace_size = KokkosFFT::compute_required_workspace_size<complex_type>(
        *m_fft_plan0, *m_ifft_plan0);
    m_fft_buffer_allocation =
        AllocationViewType("fft_buffer_allocation", workspace_size);
    m_fft_plan0->set_work_area(m_fft_buffer_allocation);
    m_ifft_plan0->set_work_area(m_fft_buffer_allocation);
    if (m_fft_plan1 != nullptr) {
      workspace_size = KokkosFFT::compute_required_workspace_size<complex_type>(
          *m_fft_plan1, *m_ifft_plan1);
      m_fft_buffer_allocation1 =
          AllocationViewType("fft_buffer_allocation1", workspace_size);
      m_fft_plan1->set_work_area(m_fft_buffer_allocation1);
      m_ifft_plan1->set_work_area(m_fft_buffer_allocation1);
    }
#else
    if (m_fft_plan1 != nullptr) {
      workspace_size = KokkosFFT::compute_required_workspace_size<complex_type>(
          *m_fft_plan0, *m_fft_plan1, *m_ifft_plan0, *m_ifft_plan1);
      m_fft_buffer_allocation =
          AllocationViewType("fft_buffer_allocation", workspace_size);
      m_fft_plan0->set_work_area(m_fft_buffer_allocation);
      m_fft_plan1->set_work_area(m_fft_buffer_allocation);
      m_ifft_plan0->set_work_area(m_fft_buffer_allocation);
      m_ifft_plan1->set_work_area(m_fft_buffer_allocation);
    } else {
      workspace_size = KokkosFFT::compute_required_workspace_size<complex_type>(
          *m_fft_plan0, *m_ifft_plan0);
      m_fft_buffer_allocation =
          AllocationViewType("fft_buffer_allocation", workspace_size);
      m_fft_plan0->set_work_area(m_fft_buffer_allocation);
      m_ifft_plan0->set_work_area(m_fft_buffer_allocation);
    }
#endif
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
  template <std::size_t STEP, typename InType, typename OutType>
  void forward_fft(const InType& in, const OutType& out) const {
    if constexpr (STEP == 0) {
      if (m_fft_dims.at(STEP) == 2) {
        if (m_map_forward_in == int_map_type{} &&
            m_map_forward_out == int_map_type{}) {
          KokkosFFT::execute(*m_fft_plan0, in, out,
                             KokkosFFT::Normalization::none);
        } else if (m_map_forward_in == int_map_type{} &&
                   m_map_forward_out != int_map_type{}) {
          KokkosFFT::execute(*m_fft_plan0, m_in_T, m_out_T,
                             KokkosFFT::Normalization::none);
          KokkosFFT::Impl::transpose(m_exec_space, m_out_T, out,
                                     m_map_forward_out, true);
        } else if (m_map_forward_in != int_map_type{} &&
                   m_map_forward_out == int_map_type{}) {
          KokkosFFT::Impl::transpose(m_exec_space, in, m_in_T, m_map_forward_in,
                                     true);
          KokkosFFT::execute(*m_fft_plan0, m_in_T, out,
                             KokkosFFT::Normalization::none);
        } else {
          KokkosFFT::Impl::transpose(m_exec_space, in, m_in_T, m_map_forward_in,
                                     true);
          KokkosFFT::execute(*m_fft_plan0, m_in_T, m_out_T,
                             KokkosFFT::Normalization::none);
          KokkosFFT::Impl::transpose(m_exec_space, m_out_T, out,
                                     m_map_forward_out, true);
        }
      } else {
        if (m_map_forward_in == int_map_type{}) {
          KokkosFFT::execute(*m_fft_plan0, in, out,
                             KokkosFFT::Normalization::none);
        } else {
          KokkosFFT::Impl::transpose(m_exec_space, in, m_in_T, m_map_forward_in,
                                     true);
          KokkosFFT::execute(*m_fft_plan0, m_in_T, out,
                             KokkosFFT::Normalization::none);
        }
      }
    } else if constexpr (STEP == 1) {
      if (m_map_forward_out == int_map_type{}) {
        KokkosFFT::execute(*m_fft_plan1, in, out,
                           KokkosFFT::Normalization::none);
      } else {
        KokkosFFT::execute(*m_fft_plan1, in, m_out_T2,
                           KokkosFFT::Normalization::none);
        KokkosFFT::Impl::transpose(m_exec_space, m_out_T2, out,
                                   m_map_forward_out, true);
      }
    }
  }

  template <std::size_t STEP, typename InType, typename OutType>
  void backward_fft(const OutType& out, const InType& in) const {
    if constexpr (STEP == 0) {
      if (m_fft_dims.at(STEP) == 2) {
        if (m_map_backward_in == int_map_type{} &&
            m_map_backward_out == int_map_type{}) {
          KokkosFFT::execute(*m_ifft_plan0, out, in,
                             KokkosFFT::Normalization::none);
        } else if (m_map_backward_in == int_map_type{} &&
                   m_map_backward_out != int_map_type{}) {
          KokkosFFT::execute(*m_ifft_plan0, out, m_in_T,
                             KokkosFFT::Normalization::none);
          KokkosFFT::Impl::transpose(m_exec_space, m_in_T, in,
                                     m_map_backward_out, true);
        } else if (m_map_backward_in != int_map_type{} &&
                   m_map_backward_out == int_map_type{}) {
          KokkosFFT::Impl::transpose(m_exec_space, out, m_out_T,
                                     m_map_backward_in, true);
          KokkosFFT::execute(*m_ifft_plan0, m_out_T, in,
                             KokkosFFT::Normalization::none);
        } else if (m_map_backward_in != int_map_type{} &&
                   m_map_backward_out != int_map_type{}) {
          KokkosFFT::Impl::transpose(m_exec_space, out, m_out_T,
                                     m_map_backward_in, true);
          KokkosFFT::execute(*m_ifft_plan0, m_out_T, m_in_T,
                             KokkosFFT::Normalization::none);
          KokkosFFT::Impl::transpose(m_exec_space, m_in_T, in,
                                     m_map_backward_out, true);
        }
      } else {
        if (m_map_backward_out == int_map_type{}) {
          KokkosFFT::execute(*m_ifft_plan0, out, in,
                             KokkosFFT::Normalization::none);
        } else {
          KokkosFFT::execute(*m_ifft_plan0, out, m_in_T,
                             KokkosFFT::Normalization::none);
          KokkosFFT::Impl::transpose(m_exec_space, m_in_T, in,
                                     m_map_backward_out, true);
        }
      }
    } else if constexpr (STEP == 1) {
      if (m_map_backward_in == int_map_type{}) {
        KokkosFFT::execute(*m_ifft_plan1, out, in,
                           KokkosFFT::Normalization::none);
      } else {
        KokkosFFT::Impl::transpose(m_exec_space, out, m_out_T2,
                                   m_map_backward_in, true);
        KokkosFFT::execute(*m_ifft_plan1, m_out_T2, in,
                           KokkosFFT::Normalization::none);
      }
    }
  }

  void set_block_impl(const std::size_t block_idx) {
    auto src_map    = KokkosFFT::Impl::index_sequence<std::size_t, DIM, 0>();
    auto block      = m_block_analyses->m_block_infos.at(block_idx);
    auto block_type = block.m_block_type;

    if (block_type == BlockType::FFT) {
      if (block.m_block_idx == 0) {
        m_fft_dims.at(0) = block.m_fft_dim;
        m_in_T           = InViewType(
            reinterpret_cast<in_value_type*>(m_send_buffer_allocation.data()),
            KokkosFFT::Impl::create_layout<LayoutType>(block.m_in_extents));
        m_out_T = OutViewType(
            m_recv_buffer_allocation.data(),
            KokkosFFT::Impl::create_layout<LayoutType>(block.m_out_extents));

        m_fft_plan0 = std::make_unique<FFTForwardPlanType0>(
            m_exec_space, m_in_T, m_out_T, KokkosFFT::Direction::forward,
            block.m_fft_dim);
        m_ifft_plan0 = std::make_unique<FFTBackwardPlanType0>(
            m_exec_space, m_out_T, m_in_T, KokkosFFT::Direction::backward,
            block.m_fft_dim);
      } else {
        m_out_T2 = OutViewType(
            m_send_buffer_allocation.data(),
            KokkosFFT::Impl::create_layout<LayoutType>(block.m_out_extents));
        m_fft_plan1 = std::make_unique<FFTForwardPlanType1>(
            m_exec_space, m_out_T2, m_out_T2, KokkosFFT::Direction::forward, 1);
        m_ifft_plan1 = std::make_unique<FFTBackwardPlanType1>(
            m_exec_space, m_out_T2, m_out_T2, KokkosFFT::Direction::backward,
            1);
      }

      if (block_idx == 0) {
        // In this case, input data needed to be transposed locally
        if (block.m_in_map != src_map) {
          for (std::size_t i = 0; i < DIM; ++i) {
            m_map_forward_in.at(i) = block.m_in_map.at(i);
            m_map_backward_out.at(i) =
                KokkosFFT::Impl::get_index(block.m_in_map, i);
          }
        }
      }

      if (block_idx == m_block_analyses->m_block_infos.size() - 1) {
        if (block.m_out_map != src_map) {
          for (std::size_t i = 0; i < DIM; ++i) {
            m_map_forward_out.at(i) =
                KokkosFFT::Impl::get_index(block.m_out_map, i);
            m_map_backward_in.at(i) = block.m_out_map.at(i);
          }
        }
      }
    } else {
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
        forward_fft<0>(in, out_view);
      } else if (block_type == BlockType::Transpose) {
        (*m_trans_blocks.at(block.m_block_idx))(
            m_comm, in, m_in_T, m_send_buffer_allocation,
            m_recv_buffer_allocation, KokkosFFT::Direction::forward);
      }
    } else {
      if (block_type == BlockType::FFT) {
        if (block.m_block_idx == 0) {
          OutViewType out_view =
              block_idx == m_block_analyses->m_block_infos.size() - 1 ? out
                                                                      : m_out_T;
          forward_fft<0>(m_in_T, out_view);
        } else {
          OutViewType cin_view = m_out_T2, cout_view = m_out_T2;
          if (block_idx == m_block_analyses->m_block_infos.size() - 1) {
            cout_view = out;
            cin_view  = m_map_forward_out == int_map_type{} ? out : m_out_T2;
          }
          forward_fft<1>(cin_view, cout_view);
        }
      } else if (block_type == BlockType::Transpose) {
        OutViewType out_view = m_out_T;
        OutViewType out_view2 =
            m_map_forward_out == int_map_type{} ? out : m_out_T2;

        if (m_block_analyses->m_block_infos.back().m_block_type ==
            BlockType::Transpose) {
          out_view2 = m_out_T2;
        }

        if (block_idx == m_block_analyses->m_block_infos.size() - 1) {
          out_view  = m_fft_dims.at(0) == 1 ? m_out_T2 : m_out_T;
          out_view2 = out;
        }

        AllocationViewType send_buffer = m_send_buffer_allocation,
                           recv_buffer = m_recv_buffer_allocation;

        if (KokkosFFT::Impl::are_aliasing(out_view.data(),
                                          send_buffer.data())) {
          send_buffer = m_recv_buffer_allocation;
          recv_buffer = m_send_buffer_allocation;
        }

        (*m_trans_blocks.at(block.m_block_idx))(m_comm, out_view, out_view2,
                                                send_buffer, recv_buffer,
                                                KokkosFFT::Direction::forward);
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
        backward_fft<0>(out_view, in);
      } else if (block_type == BlockType::Transpose) {
        (*m_trans_blocks.at(block.m_block_idx))(
            m_comm, m_in_T, in, m_recv_buffer_allocation,
            m_send_buffer_allocation, KokkosFFT::Direction::backward);
      }
    } else {
      if (block_type == BlockType::FFT) {
        if (block.m_block_idx == 0) {
          OutViewType out_view =
              block_idx == m_block_analyses->m_block_infos.size() - 1 ? out
                                                                      : m_out_T;
          backward_fft<0>(out_view, m_in_T);
        } else {
          OutViewType cin_view = m_out_T2, cout_view = m_out_T2;
          if (block_idx == m_block_analyses->m_block_infos.size() - 1) {
            cout_view = out;
            cin_view  = m_map_backward_in == int_map_type{} ? out : m_out_T2;
          }
          backward_fft<1>(cout_view, cin_view);
        }
      } else if (block_type == BlockType::Transpose) {
        OutViewType out_view = m_out_T;
        OutViewType out_view2 =
            m_map_backward_in == int_map_type{} ? out : m_out_T2;

        if (m_block_analyses->m_block_infos.back().m_block_type ==
            BlockType::Transpose) {
          out_view2 = m_out_T2;
        }

        if (block_idx == m_block_analyses->m_block_infos.size() - 1) {
          out_view  = m_fft_dims.at(0) == 1 ? m_out_T2 : m_out_T;
          out_view2 = out;
        }

        AllocationViewType send_buffer = m_send_buffer_allocation,
                           recv_buffer = m_recv_buffer_allocation;

        if (KokkosFFT::Impl::are_aliasing(out_view.data(),
                                          recv_buffer.data())) {
          send_buffer = m_recv_buffer_allocation;
          recv_buffer = m_send_buffer_allocation;
        }

        (*m_trans_blocks.at(block.m_block_idx))(m_comm, out_view2, out_view,
                                                send_buffer, recv_buffer,
                                                KokkosFFT::Direction::backward);
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
template <typename ExecutionSpace, typename InViewType, typename OutViewType>
struct SlabInternalPlan<ExecutionSpace, InViewType, OutViewType, 3> {
  using execSpace      = ExecutionSpace;
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;
  using float_type   = KokkosFFT::Impl::base_floating_point_type<in_value_type>;
  using complex_type = Kokkos::complex<float_type>;
  using LayoutType   = typename InViewType::array_layout;

  static constexpr std::size_t DIM = InViewType::rank();

  using axes_type     = KokkosFFT::axis_type<3>;
  using topology_type = KokkosFFT::shape_type<DIM>;
  using int_map_type  = std::array<int, DIM>;

  using SlabBlockAnalysesType =
      SlabBlockAnalysesInternal<in_value_type, LayoutType, std::size_t, DIM, 3>;

  using AllocationViewType =
      Kokkos::View<complex_type*, LayoutType, ExecutionSpace>;

  // Type for transpose Block
  using CommType       = KokkosFFT::Distributed::Impl::TplComm<ExecutionSpace>;
  using TransBlockType = TransBlock<ExecutionSpace, DIM>;

  // First FFT to perform can be Real to Complex
  using FFTForwardPlanType0 =
      KokkosFFT::DynPlan<ExecutionSpace, InViewType, OutViewType>;
  using FFTBackwardPlanType0 =
      KokkosFFT::DynPlan<ExecutionSpace, OutViewType, InViewType>;

  // Second FFT to perform must be Complex to Complex
  using FFTForwardPlanType1 =
      KokkosFFT::DynPlan<ExecutionSpace, OutViewType, OutViewType>;
  using FFTBackwardPlanType1 =
      KokkosFFT::DynPlan<ExecutionSpace, OutViewType, OutViewType>;

  execSpace m_exec_space;
  axes_type m_axes;
  topology_type m_in_topology;
  topology_type m_out_topology;
  CommType m_comm;

  // Analyse topology
  std::unique_ptr<SlabBlockAnalysesType> m_block_analyses;
  std::array<std::size_t, 1> m_fft_dims = {};

  // Buffer view types
  InViewType m_in_T;
  OutViewType m_out_T, m_out_T2;

  // Buffer Allocations
  AllocationViewType m_send_buffer_allocation, m_recv_buffer_allocation,
      m_fft_buffer_allocation;

#if defined(KOKKOS_ENABLE_SYCL)
  AllocationViewType m_fft_buffer_allocation1;
#endif

  // Internal transpose blocks
  std::vector<std::unique_ptr<TransBlockType>> m_trans_blocks;

  // Internal FFT plans
  std::unique_ptr<FFTForwardPlanType0> m_fft_plan0;
  std::unique_ptr<FFTForwardPlanType1> m_fft_plan1;
  std::unique_ptr<FFTBackwardPlanType0> m_ifft_plan0;
  std::unique_ptr<FFTBackwardPlanType1> m_ifft_plan1;

  int_map_type m_map_forward_in = {}, m_map_forward_out = {},
               m_map_backward_in = {}, m_map_backward_out = {};

 public:
  explicit SlabInternalPlan(const ExecutionSpace& exec_space,
                            const InViewType& in, const OutViewType& out,
                            const axes_type& axes,
                            const topology_type& in_topology,
                            const topology_type& out_topology,
                            const MPI_Comm& comm)
      : m_exec_space(exec_space),
        m_axes(axes),
        m_in_topology(in_topology),
        m_out_topology(out_topology),
        m_comm(comm, exec_space) {
    KOKKOSFFT_THROW_IF(
        !are_valid_extents(in, out, axes, in_topology, out_topology, comm),
        "Extents are not valid");

    // First get global shape to define buffer and next shape
    auto in_extents  = KokkosFFT::Impl::extract_extents(in);
    auto out_extents = KokkosFFT::Impl::extract_extents(out);

    auto gin_extents  = get_global_shape(in, m_in_topology, comm);
    auto gout_extents = get_global_shape(out, m_out_topology, comm);

    auto non_negative_axes =
        KokkosFFT::Impl::convert_base_int_type<std::size_t>(
            KokkosFFT::Impl::convert_negative_axes(axes, DIM));

    m_block_analyses = std::make_unique<SlabBlockAnalysesType>(
        in_extents, out_extents, gin_extents, gout_extents, in_topology,
        out_topology, non_negative_axes, comm);

    KOKKOSFFT_THROW_IF(!(m_block_analyses->m_block_infos.size() >= 1 &&
                         m_block_analyses->m_block_infos.size() <= 5),
                       "Number blocks must be in [1, 5]");

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

    std::size_t workspace_size = 0;
#if defined(KOKKOS_ENABLE_SYCL)
    workspace_size = KokkosFFT::compute_required_workspace_size<complex_type>(
        *m_fft_plan0, *m_ifft_plan0);
    m_fft_buffer_allocation =
        AllocationViewType("fft_buffer_allocation", workspace_size);
    m_fft_plan0->set_work_area(m_fft_buffer_allocation);
    m_ifft_plan0->set_work_area(m_fft_buffer_allocation);
    if (m_fft_plan1 != nullptr) {
      workspace_size = KokkosFFT::compute_required_workspace_size<complex_type>(
          *m_fft_plan1, *m_ifft_plan1);
      m_fft_buffer_allocation1 =
          AllocationViewType("fft_buffer_allocation1", workspace_size);
      m_fft_plan1->set_work_area(m_fft_buffer_allocation1);
      m_ifft_plan1->set_work_area(m_fft_buffer_allocation1);
    }
#else
    if (m_fft_plan1 != nullptr) {
      workspace_size = KokkosFFT::compute_required_workspace_size<complex_type>(
          *m_fft_plan0, *m_fft_plan1, *m_ifft_plan0, *m_ifft_plan1);
      m_fft_buffer_allocation =
          AllocationViewType("fft_buffer_allocation", workspace_size);
      m_fft_plan0->set_work_area(m_fft_buffer_allocation);
      m_fft_plan1->set_work_area(m_fft_buffer_allocation);
      m_ifft_plan0->set_work_area(m_fft_buffer_allocation);
      m_ifft_plan1->set_work_area(m_fft_buffer_allocation);
    } else {
      workspace_size = KokkosFFT::compute_required_workspace_size<complex_type>(
          *m_fft_plan0, *m_ifft_plan0);
      m_fft_buffer_allocation =
          AllocationViewType("fft_buffer_allocation", workspace_size);
      m_fft_plan0->set_work_area(m_fft_buffer_allocation);
      m_ifft_plan0->set_work_area(m_fft_buffer_allocation);
    }
#endif
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
  template <std::size_t STEP, typename InType, typename OutType>
  void forward_fft(const InType& in, const OutType& out) const {
    if constexpr (STEP == 0) {
      if (m_fft_dims.at(STEP) == 3) {
        if (m_map_forward_in == int_map_type{} &&
            m_map_forward_out == int_map_type{}) {
          KokkosFFT::execute(*m_fft_plan0, in, out,
                             KokkosFFT::Normalization::none);
        } else if (m_map_forward_in == int_map_type{} &&
                   m_map_forward_out != int_map_type{}) {
          KokkosFFT::execute(*m_fft_plan0, m_in_T, m_out_T,
                             KokkosFFT::Normalization::none);
          KokkosFFT::Impl::transpose(m_exec_space, m_out_T, out,
                                     m_map_forward_out, true);
        } else if (m_map_forward_in != int_map_type{} &&
                   m_map_forward_out == int_map_type{}) {
          KokkosFFT::Impl::transpose(m_exec_space, in, m_in_T, m_map_forward_in,
                                     true);
          KokkosFFT::execute(*m_fft_plan0, m_in_T, out,
                             KokkosFFT::Normalization::none);
        } else {
          KokkosFFT::Impl::transpose(m_exec_space, in, m_in_T, m_map_forward_in,
                                     true);
          KokkosFFT::execute(*m_fft_plan0, m_in_T, m_out_T,
                             KokkosFFT::Normalization::none);
          KokkosFFT::Impl::transpose(m_exec_space, m_out_T, out,
                                     m_map_forward_out, true);
        }
      } else {
        if (m_map_forward_in == int_map_type{}) {
          KokkosFFT::execute(*m_fft_plan0, in, out,
                             KokkosFFT::Normalization::none);
        } else {
          KokkosFFT::Impl::transpose(m_exec_space, in, m_in_T, m_map_forward_in,
                                     true);
          KokkosFFT::execute(*m_fft_plan0, m_in_T, out,
                             KokkosFFT::Normalization::none);
        }
      }
    } else if constexpr (STEP == 1) {
      if (m_map_forward_out == int_map_type{}) {
        KokkosFFT::execute(*m_fft_plan1, in, out,
                           KokkosFFT::Normalization::none);
      } else {
        KokkosFFT::execute(*m_fft_plan1, in, m_out_T2,
                           KokkosFFT::Normalization::none);
        KokkosFFT::Impl::transpose(m_exec_space, m_out_T2, out,
                                   m_map_forward_out, true);
      }
    }
  }

  template <std::size_t STEP, typename InType, typename OutType>
  void backward_fft(const OutType& out, const InType& in) const {
    if constexpr (STEP == 0) {
      if (m_fft_dims.at(STEP) == 3) {
        if (m_map_backward_in == int_map_type{} &&
            m_map_backward_out == int_map_type{}) {
          KokkosFFT::execute(*m_ifft_plan0, out, in,
                             KokkosFFT::Normalization::none);
        } else if (m_map_backward_in == int_map_type{} &&
                   m_map_backward_out != int_map_type{}) {
          KokkosFFT::execute(*m_ifft_plan0, out, m_in_T,
                             KokkosFFT::Normalization::none);
          KokkosFFT::Impl::transpose(m_exec_space, m_in_T, in,
                                     m_map_backward_out, true);
        } else if (m_map_backward_in != int_map_type{} &&
                   m_map_backward_out == int_map_type{}) {
          KokkosFFT::Impl::transpose(m_exec_space, out, m_out_T,
                                     m_map_backward_in, true);
          KokkosFFT::execute(*m_ifft_plan0, m_out_T, in,
                             KokkosFFT::Normalization::none);
        } else if (m_map_backward_in != int_map_type{} &&
                   m_map_backward_out != int_map_type{}) {
          KokkosFFT::Impl::transpose(m_exec_space, out, m_out_T,
                                     m_map_backward_in, true);
          KokkosFFT::execute(*m_ifft_plan0, m_out_T, m_in_T,
                             KokkosFFT::Normalization::none);
          KokkosFFT::Impl::transpose(m_exec_space, m_in_T, in,
                                     m_map_backward_out, true);
        }
      } else {
        if (m_map_backward_out == int_map_type{}) {
          KokkosFFT::execute(*m_ifft_plan0, out, in,
                             KokkosFFT::Normalization::none);
        } else {
          KokkosFFT::execute(*m_ifft_plan0, out, m_in_T,
                             KokkosFFT::Normalization::none);
          KokkosFFT::Impl::transpose(m_exec_space, m_in_T, in,
                                     m_map_backward_out, true);
        }
      }
    } else if constexpr (STEP == 1) {
      if (m_map_backward_in == int_map_type{}) {
        KokkosFFT::execute(*m_ifft_plan1, out, in,
                           KokkosFFT::Normalization::none);
      } else {
        KokkosFFT::Impl::transpose(m_exec_space, out, m_out_T2,
                                   m_map_backward_in, true);
        KokkosFFT::execute(*m_ifft_plan1, m_out_T2, in,
                           KokkosFFT::Normalization::none);
      }
    }
  }

  void set_block_impl(const std::size_t block_idx) {
    auto src_map    = KokkosFFT::Impl::index_sequence<std::size_t, DIM, 0>();
    auto block      = m_block_analyses->m_block_infos.at(block_idx);
    auto block_type = block.m_block_type;

    if (block_type == BlockType::FFT) {
      if (block.m_block_idx == 0) {
        m_fft_dims.at(0) = block.m_fft_dim;
        m_in_T           = InViewType(
            reinterpret_cast<in_value_type*>(m_send_buffer_allocation.data()),
            KokkosFFT::Impl::create_layout<LayoutType>(block.m_in_extents));
        m_out_T = OutViewType(
            m_recv_buffer_allocation.data(),
            KokkosFFT::Impl::create_layout<LayoutType>(block.m_out_extents));

        m_fft_plan0 = std::make_unique<FFTForwardPlanType0>(
            m_exec_space, m_in_T, m_out_T, KokkosFFT::Direction::forward,
            block.m_fft_dim);
        m_ifft_plan0 = std::make_unique<FFTBackwardPlanType0>(
            m_exec_space, m_out_T, m_in_T, KokkosFFT::Direction::backward,
            block.m_fft_dim);
      } else {
        m_out_T2 = OutViewType(
            m_send_buffer_allocation.data(),
            KokkosFFT::Impl::create_layout<LayoutType>(block.m_out_extents));
        m_fft_plan1 = std::make_unique<FFTForwardPlanType1>(
            m_exec_space, m_out_T2, m_out_T2, KokkosFFT::Direction::forward,
            block.m_fft_dim);
        m_ifft_plan1 = std::make_unique<FFTBackwardPlanType1>(
            m_exec_space, m_out_T2, m_out_T2, KokkosFFT::Direction::backward,
            block.m_fft_dim);
      }

      if (block_idx == 0) {
        // In this case, input data needed to be transposed locally
        if (block.m_in_map != src_map) {
          for (std::size_t i = 0; i < DIM; ++i) {
            m_map_forward_in.at(i) = block.m_in_map.at(i);
            m_map_backward_out.at(i) =
                KokkosFFT::Impl::get_index(block.m_in_map, i);
          }
        }
      }

      if (block_idx == m_block_analyses->m_block_infos.size() - 1) {
        if (block.m_out_map != src_map) {
          for (std::size_t i = 0; i < DIM; ++i) {
            m_map_forward_out.at(i) =
                KokkosFFT::Impl::get_index(block.m_out_map, i);
            m_map_backward_in.at(i) = block.m_out_map.at(i);
          }
        }
      }
    } else {
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
        forward_fft<0>(in, out_view);
      } else if (block_type == BlockType::Transpose) {
        (*m_trans_blocks.at(block.m_block_idx))(
            m_comm, in, m_in_T, m_send_buffer_allocation,
            m_recv_buffer_allocation, KokkosFFT::Direction::forward);
      }
    } else {
      if (block_type == BlockType::FFT) {
        if (block.m_block_idx == 0) {
          OutViewType out_view =
              block_idx == m_block_analyses->m_block_infos.size() - 1 ? out
                                                                      : m_out_T;
          forward_fft<0>(m_in_T, out_view);
        } else {
          OutViewType cin_view = m_out_T2, cout_view = m_out_T2;
          if (block_idx == m_block_analyses->m_block_infos.size() - 1) {
            cout_view = out;
            cin_view  = m_map_forward_out == int_map_type{} ? out : m_out_T2;
          }
          forward_fft<1>(cin_view, cout_view);
        }
      } else if (block_type == BlockType::Transpose) {
        OutViewType out_view = m_out_T;
        OutViewType out_view2 =
            m_map_forward_out == int_map_type{} ? out : m_out_T2;

        if (m_block_analyses->m_block_infos.back().m_block_type ==
            BlockType::Transpose) {
          out_view2 = m_out_T2;
        }

        if (block_idx == m_block_analyses->m_block_infos.size() - 1) {
          out_view  = m_fft_dims.at(0) == 3 ? m_out_T : m_out_T2;
          out_view2 = out;
        }

        AllocationViewType send_buffer = m_send_buffer_allocation,
                           recv_buffer = m_recv_buffer_allocation;

        if (KokkosFFT::Impl::are_aliasing(out_view.data(),
                                          send_buffer.data())) {
          send_buffer = m_recv_buffer_allocation;
          recv_buffer = m_send_buffer_allocation;
        }

        (*m_trans_blocks.at(block.m_block_idx))(m_comm, out_view, out_view2,
                                                send_buffer, recv_buffer,
                                                KokkosFFT::Direction::forward);
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
        backward_fft<0>(out_view, in);
      } else if (block_type == BlockType::Transpose) {
        (*m_trans_blocks.at(block.m_block_idx))(
            m_comm, m_in_T, in, m_recv_buffer_allocation,
            m_send_buffer_allocation, KokkosFFT::Direction::backward);
      }
    } else {
      if (block_type == BlockType::FFT) {
        if (block.m_block_idx == 0) {
          OutViewType out_view =
              block_idx == m_block_analyses->m_block_infos.size() - 1 ? out
                                                                      : m_out_T;
          backward_fft<0>(out_view, m_in_T);
        } else {
          OutViewType cin_view = m_out_T2, cout_view = m_out_T2;
          if (block_idx == m_block_analyses->m_block_infos.size() - 1) {
            cout_view = out;
            cin_view  = m_map_backward_in == int_map_type{} ? out : m_out_T2;
          }
          backward_fft<1>(cout_view, cin_view);
        }
      } else if (block_type == BlockType::Transpose) {
        OutViewType out_view = m_out_T;
        OutViewType out_view2 =
            m_map_backward_in == int_map_type{} ? out : m_out_T2;

        if (m_block_analyses->m_block_infos.back().m_block_type ==
            BlockType::Transpose) {
          out_view2 = m_out_T2;
        }

        if (block_idx == m_block_analyses->m_block_infos.size() - 1) {
          out_view  = m_fft_dims.at(0) == 3 ? m_out_T : m_out_T2;
          out_view2 = out;
        }

        AllocationViewType send_buffer = m_send_buffer_allocation,
                           recv_buffer = m_recv_buffer_allocation;

        if (KokkosFFT::Impl::are_aliasing(out_view.data(),
                                          recv_buffer.data())) {
          send_buffer = m_recv_buffer_allocation;
          recv_buffer = m_send_buffer_allocation;
        }

        (*m_trans_blocks.at(block.m_block_idx))(m_comm, out_view2, out_view,
                                                send_buffer, recv_buffer,
                                                KokkosFFT::Direction::backward);
      }
    }
  }
};

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1, typename InLayoutType = Kokkos::LayoutRight,
          typename OutLayoutType = Kokkos::LayoutRight>
class SlabPlan : public InternalPlan<ExecutionSpace, InViewType, OutViewType,
                                     DIM, InLayoutType, OutLayoutType> {
  using InternalPlanType =
      SlabInternalPlan<ExecutionSpace, InViewType, OutViewType, DIM>;
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
  explicit SlabPlan(
      const ExecutionSpace& exec_space, const InViewType& in,
      const OutViewType& out, const axes_type& axes,
      const extents_type& in_topology, const extents_type& out_topology,
      const MPI_Comm& comm,
      KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward)
      : SlabPlan(exec_space, in, out, axes,
                 Topology<std::size_t, InViewType::rank()>(in_topology),
                 Topology<std::size_t, OutViewType::rank()>(out_topology), comm,
                 norm) {}

  explicit SlabPlan(
      const ExecutionSpace& exec_space, const InViewType& in,
      const OutViewType& out, const axes_type& axes,
      const in_topology_type& in_topology,
      const out_topology_type& out_topology, const MPI_Comm& comm,
      KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward)
      : InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM, InLayoutType,
                     OutLayoutType>(exec_space, in, out, axes, in_topology,
                                    out_topology, comm, norm),
        m_exec_space(exec_space),
        m_internal_plan(exec_space, in, out, axes, in_topology.array(),
                        out_topology.array(), comm),
        m_in_extents(KokkosFFT::Impl::extract_extents(in)),
        m_out_extents(KokkosFFT::Impl::extract_extents(out)) {
    auto in_size  = KokkosFFT::Impl::total_size(in_topology);
    auto out_size = KokkosFFT::Impl::total_size(out_topology);

    KOKKOSFFT_THROW_IF(in_size != out_size,
                       "Input and output topologies must have the same size.");

    bool is_slab = are_slab_topologies(in_topology, out_topology);
    KOKKOSFFT_THROW_IF(!is_slab,
                       "Input and output topologies must be slab topologies.");
  }

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

  std::string label() const override { return std::string("SlabPlan"); }
};

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
