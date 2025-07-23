#ifndef SLABPLAN_HPP
#define SLABPLAN_HPP

#include <vector>
#include <memory>
#include <iostream>
#include <sstream>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "Block.hpp"
#include "Mapping.hpp"
#include "MPI_Helper.hpp"
#include "Helper.hpp"
#include "Extents.hpp"
#include "Topologies.hpp"
#include "SlabBlockAnalyses.hpp"
#include "InternalPlan.hpp"

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

  // Defining buffer type
  using in_buffer_data_type =
      KokkosFFT::Impl::add_pointer_n_t<in_value_type, DIM + 1>;
  using InBufferType =
      Kokkos::View<in_buffer_data_type, typename InViewType::array_layout,
                   typename InViewType::execution_space>;
  using fft_data_type = KokkosFFT::Impl::add_pointer_n_t<complex_type, DIM>;
  using FFTViewType =
      Kokkos::View<fft_data_type, typename InViewType::array_layout,
                   typename InViewType::execution_space>;
  using fft_buffer_data_type =
      KokkosFFT::Impl::add_pointer_n_t<complex_type, DIM + 1>;
  using FFTBufferType =
      Kokkos::View<fft_buffer_data_type, typename InViewType::array_layout,
                   typename InViewType::execution_space>;
  using out_buffer_data_type =
      KokkosFFT::Impl::add_pointer_n_t<out_value_type, DIM + 1>;
  using OutBufferType =
      Kokkos::View<out_buffer_data_type, typename OutViewType::array_layout,
                   typename OutViewType::execution_space>;

  using AllocationViewType =
      Kokkos::View<complex_type*, LayoutType, ExecutionSpace>;

  using ForwardBlockType0 =
      Block<ExecutionSpace, InViewType, InViewType, InBufferType, 1>;
  using ForwardBlockType1 =
      Block<ExecutionSpace, FFTViewType, FFTViewType, FFTBufferType, 1>;
  using BackwardBlockType0 =
      Block<ExecutionSpace, InViewType, InViewType, InBufferType, 1>;
  using BackwardBlockType1 =
      Block<ExecutionSpace, FFTViewType, FFTViewType, FFTBufferType, 1>;

  using FFTForwardPlanType =
      KokkosFFT::Plan<ExecutionSpace, InViewType, FFTViewType, 1>;
  using FFTBackwardPlanType =
      KokkosFFT::Plan<ExecutionSpace, FFTViewType, InViewType, 1>;

  execSpace m_exec_space;
  axes_type m_axes;
  topology_type m_in_topology;
  topology_type m_out_topology;
  MPI_Comm m_comm;
  KokkosFFT::Normalization m_normalization;

  // Analyse topology
  std::unique_ptr<SlabBlockAnalysesType> m_block_analyses;
  OperationType m_op_type;

  // Buffer view types
  InViewType m_in_T;
  OutViewType m_out_T;
  FFTViewType m_fft;

  // Buffer Allocations
  AllocationViewType m_send_buffer_allocation, m_recv_buffer_allocation;
  InBufferType m_send_buffer, m_recv_buffer;
  OutBufferType m_csend_buffer, m_crecv_buffer;

  std::unique_ptr<ForwardBlockType0> m_forward_block0;
  std::unique_ptr<ForwardBlockType1> m_forward_block1;
  std::unique_ptr<BackwardBlockType0> m_backward_block0;
  std::unique_ptr<BackwardBlockType1> m_backward_block1;

  // Internal FFT plans
  std::unique_ptr<FFTForwardPlanType> m_forward_plan;
  std::unique_ptr<FFTBackwardPlanType> m_backward_plan;

  int_map_type m_map_forward = {}, m_map_backward = {};

 public:
  explicit SlabInternalPlan(const ExecutionSpace& exec_space,
                            const InViewType& in, const OutViewType& out,
                            const axes_type& axes,
                            const topology_type& in_topology,
                            const topology_type& out_topology,
                            const MPI_Comm& comm,
                            KokkosFFT::Normalization normalization =
                                KokkosFFT::Normalization::backward)
      : m_exec_space(exec_space),
        m_axes(axes),
        m_in_topology(in_topology),
        m_out_topology(out_topology),
        m_comm(comm),
        m_normalization(normalization) {
    KOKKOSFFT_THROW_IF(
        !are_valid_extents(in, out, axes, in_topology, out_topology, comm),
        "Extents are not valid");

    // First get global shape to define buffer and next shape
    auto in_extents  = KokkosFFT::Impl::extract_extents(in);
    auto out_extents = KokkosFFT::Impl::extract_extents(out);

    auto gin_extents  = get_global_shape(in, m_in_topology, m_comm);
    auto gout_extents = get_global_shape(out, m_out_topology, m_comm);

    auto non_negative_axes =
        convert_negative_axes<std::size_t, int, 1, DIM>(axes);

    m_block_analyses = std::make_unique<SlabBlockAnalysesType>(
        in_extents, out_extents, gin_extents, gout_extents, in_topology,
        out_topology, non_negative_axes, m_comm);
    m_op_type = m_block_analyses->m_op_type;

    KOKKOSFFT_THROW_IF(m_block_analyses->m_block_infos.size() > 3,
                       "Maximum three blocks are expected");

    // Allocate buffer views
    m_send_buffer_allocation = AllocationViewType(
        "send_buffer_allocation", m_block_analyses->m_max_buffer_size);
    m_recv_buffer_allocation = AllocationViewType(
        "recv_buffer_allocation", m_block_analyses->m_max_buffer_size);

    switch (m_op_type) {
      case OperationType::F: {
        auto block0 = m_block_analyses->m_block_infos.at(0);
        // This means that FFT can be performed on the first dim
        // without transpose
        m_fft = FFTViewType("fft", KokkosFFT::Impl::create_layout<LayoutType>(
                                       block0.m_out_extents));
        m_forward_plan = std::make_unique<FFTForwardPlanType>(
            m_exec_space, in, m_fft, KokkosFFT::Direction::forward, m_axes);
        m_backward_plan = std::make_unique<FFTBackwardPlanType>(
            m_exec_space, m_fft, in, KokkosFFT::Direction::backward, m_axes);
        break;
      }
      case OperationType::FT: {
        auto block0 = m_block_analyses->m_block_infos.at(0);
        // This means that FFT can be performed on the first dim
        // without transpose
        m_fft = FFTViewType("fft", KokkosFFT::Impl::create_layout<LayoutType>(
                                       block0.m_out_extents));
        m_forward_plan = std::make_unique<FFTForwardPlanType>(
            m_exec_space, in, m_fft, KokkosFFT::Direction::forward, m_axes);
        m_backward_plan = std::make_unique<FFTBackwardPlanType>(
            m_exec_space, m_fft, in, KokkosFFT::Direction::backward, m_axes);

        auto block1 = m_block_analyses->m_block_infos.at(1);
        m_out_T     = OutViewType(
            m_send_buffer_allocation.data(),
            KokkosFFT::Impl::create_layout<LayoutType>(block1.m_out_extents));
        m_csend_buffer =
            OutBufferType(m_send_buffer_allocation.data(),
                          KokkosFFT::Impl::create_layout<LayoutType>(
                              block1.m_buffer_extents));
        m_crecv_buffer =
            OutBufferType(m_recv_buffer_allocation.data(),
                          KokkosFFT::Impl::create_layout<LayoutType>(
                              block1.m_buffer_extents));
        m_forward_block1 = std::make_unique<ForwardBlockType1>(
            m_exec_space, m_fft, m_out_T, m_csend_buffer, m_crecv_buffer,
            block1.m_in_map, block1.m_in_axis, block1.m_out_map,
            block1.m_out_axis, m_comm);
        m_backward_block1 = std::make_unique<BackwardBlockType1>(
            m_exec_space, m_out_T, m_fft, m_csend_buffer, m_crecv_buffer,
            block1.m_out_map, block1.m_out_axis, block1.m_in_map,
            block1.m_in_axis, m_comm);
        break;
      }
      case OperationType::TF: {
        auto block0 = m_block_analyses->m_block_infos.at(0);
        m_in_T      = InViewType(
            reinterpret_cast<in_value_type*>(m_send_buffer_allocation.data()),
            KokkosFFT::Impl::create_layout<LayoutType>(block0.m_out_extents));
        m_send_buffer = InBufferType(
            reinterpret_cast<in_value_type*>(m_send_buffer_allocation.data()),
            KokkosFFT::Impl::create_layout<LayoutType>(
                block0.m_buffer_extents));
        m_recv_buffer = InBufferType(
            reinterpret_cast<in_value_type*>(m_recv_buffer_allocation.data()),
            KokkosFFT::Impl::create_layout<LayoutType>(
                block0.m_buffer_extents));

        m_forward_block0 = std::make_unique<ForwardBlockType0>(
            m_exec_space, in, m_in_T, m_send_buffer, m_recv_buffer,
            block0.m_in_map, block0.m_in_axis, block0.m_out_map,
            block0.m_out_axis, m_comm);

        m_backward_block0 = std::make_unique<BackwardBlockType0>(
            m_exec_space, m_in_T, in, m_send_buffer, m_recv_buffer,
            block0.m_out_map, block0.m_out_axis, block0.m_in_map,
            block0.m_in_axis, m_comm);

        auto block1 = m_block_analyses->m_block_infos.at(1);

        // This means that FFT can be performed on the first dim
        // without transpose
        m_fft = FFTViewType("fft", KokkosFFT::Impl::create_layout<LayoutType>(
                                       block1.m_out_extents));
        m_forward_plan = std::make_unique<FFTForwardPlanType>(
            m_exec_space, m_in_T, m_fft, KokkosFFT::Direction::forward,
            to_array<int, std::size_t, 1>(block1.m_axes));
        m_backward_plan = std::make_unique<FFTBackwardPlanType>(
            m_exec_space, m_fft, m_in_T, KokkosFFT::Direction::backward,
            to_array<int, std::size_t, 1>(block1.m_axes));
        if (block0.m_out_map != block0.m_in_map) {
          for (std::size_t i = 0; i < DIM; ++i) {
            m_map_forward.at(i) =
                KokkosFFT::Impl::get_index(block0.m_out_map, i);
            m_map_backward.at(i) = block0.m_out_map.at(i);
          }
        }
        break;
      }
      case OperationType::TFT: {
        auto block0 = m_block_analyses->m_block_infos.at(0);
        m_in_T      = InViewType(
            reinterpret_cast<in_value_type*>(m_send_buffer_allocation.data()),
            KokkosFFT::Impl::create_layout<LayoutType>(block0.m_out_extents));
        m_send_buffer = InBufferType(
            reinterpret_cast<in_value_type*>(m_send_buffer_allocation.data()),
            KokkosFFT::Impl::create_layout<LayoutType>(
                block0.m_buffer_extents));
        m_recv_buffer = InBufferType(
            reinterpret_cast<in_value_type*>(m_recv_buffer_allocation.data()),
            KokkosFFT::Impl::create_layout<LayoutType>(
                block0.m_buffer_extents));
        m_forward_block0 = std::make_unique<ForwardBlockType0>(
            m_exec_space, in, m_in_T, m_send_buffer, m_recv_buffer,
            block0.m_in_map, block0.m_in_axis, block0.m_out_map,
            block0.m_out_axis, m_comm);
        m_backward_block0 = std::make_unique<BackwardBlockType0>(
            m_exec_space, m_in_T, in, m_send_buffer, m_recv_buffer,
            block0.m_out_map, block0.m_out_axis, block0.m_in_map,
            block0.m_in_axis, m_comm);

        auto block1 = m_block_analyses->m_block_infos.at(1);
        // This means that FFT can be performed on the first dim
        // without transpose
        m_fft = FFTViewType("fft", KokkosFFT::Impl::create_layout<LayoutType>(
                                       block1.m_out_extents));

        m_forward_plan = std::make_unique<FFTForwardPlanType>(
            m_exec_space, m_in_T, m_fft, KokkosFFT::Direction::forward,
            to_array<int, std::size_t, 1>(block1.m_axes));
        m_backward_plan = std::make_unique<FFTBackwardPlanType>(
            m_exec_space, m_fft, m_in_T, KokkosFFT::Direction::backward,
            to_array<int, std::size_t, 1>(block1.m_axes));

        auto block2 = m_block_analyses->m_block_infos.at(2);
        m_out_T     = OutViewType(
            m_send_buffer_allocation.data(),
            KokkosFFT::Impl::create_layout<LayoutType>(block2.m_out_extents));
        m_csend_buffer =
            OutBufferType(m_send_buffer_allocation.data(),
                          KokkosFFT::Impl::create_layout<LayoutType>(
                              block2.m_buffer_extents));
        m_crecv_buffer =
            OutBufferType(m_recv_buffer_allocation.data(),
                          KokkosFFT::Impl::create_layout<LayoutType>(
                              block2.m_buffer_extents));
        m_forward_block1 = std::make_unique<ForwardBlockType1>(
            m_exec_space, m_fft, m_out_T, m_csend_buffer, m_crecv_buffer,
            block2.m_in_map, block2.m_in_axis, block2.m_out_map,
            block2.m_out_axis, m_comm);
        m_backward_block1 = std::make_unique<BackwardBlockType1>(
            m_exec_space, m_out_T, m_fft, m_csend_buffer, m_crecv_buffer,
            block2.m_out_map, block2.m_out_axis, block2.m_in_map,
            block2.m_in_axis, m_comm);
        break;
      }
      default:  // No Operation
        break;
    };
  }

  void forward(const InViewType& in, const OutViewType& out) const {
    switch (m_op_type) {
      case OperationType::F: {
        KokkosFFT::execute(*m_forward_plan, in, out);
        break;
      }
      case OperationType::FT: {
        KokkosFFT::execute(*m_forward_plan, in, m_fft);
        (*m_forward_block1)(m_fft, out);
        break;
      }
      case OperationType::TF: {
        (*m_forward_block0)(in, m_in_T);
        if (m_map_forward == int_map_type{}) {
          KokkosFFT::execute(*m_forward_plan, m_in_T, out);
        } else {
          OutViewType out_T(reinterpret_cast<out_value_type*>(
                                m_recv_buffer_allocation.data()),
                            m_fft.layout());
          KokkosFFT::execute(*m_forward_plan, m_in_T, out_T);
          safe_transpose(m_exec_space, out_T, out, m_map_forward);
        }

        break;
      }
      case OperationType::TFT: {
        (*m_forward_block0)(in, m_in_T);
        KokkosFFT::execute(*m_forward_plan, m_in_T, m_fft);
        (*m_forward_block1)(m_fft, out);
        break;
      }
      default: break;
    };
  }

  void backward(const OutViewType& out, const InViewType& in) const {
    switch (m_op_type) {
      case OperationType::F: {
        KokkosFFT::execute(*m_backward_plan, out, in);
        break;
      }
      case OperationType::FT: {
        (*m_backward_block1)(out, m_fft);
        KokkosFFT::execute(*m_backward_plan, m_fft, in);
        break;
      }
      case OperationType::TF: {
        InViewType in_T(
            reinterpret_cast<in_value_type*>(m_recv_buffer_allocation.data()),
            m_in_T.layout());

        if (m_map_backward == int_map_type{}) {
          // No transpose needed
          KokkosFFT::execute(*m_backward_plan, out, in_T);
        } else {
          OutViewType out_T(reinterpret_cast<out_value_type*>(
                                m_send_buffer_allocation.data()),
                            m_fft.layout());
          safe_transpose(m_exec_space, out, out_T, m_map_backward);
          KokkosFFT::execute(*m_backward_plan, out_T, in_T);
        }
        (*m_backward_block0)(in_T, in);
        break;
      }
      case OperationType::TFT: {
        InViewType in_T(
            reinterpret_cast<in_value_type*>(m_recv_buffer_allocation.data()),
            m_in_T.layout());
        (*m_backward_block1)(out, m_fft);
        KokkosFFT::execute(*m_backward_plan, m_fft, in_T);
        (*m_backward_block0)(in_T, in);
        break;
      }
      default: break;
    };
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

  // Defining buffer type
  using in_buffer_data_type =
      KokkosFFT::Impl::add_pointer_n_t<in_value_type, DIM + 1>;
  using InBufferType =
      Kokkos::View<in_buffer_data_type, typename InViewType::array_layout,
                   typename InViewType::execution_space>;
  using fft_data_type = KokkosFFT::Impl::add_pointer_n_t<complex_type, DIM>;
  using FFTViewType =
      Kokkos::View<fft_data_type, typename InViewType::array_layout,
                   typename InViewType::execution_space>;
  using fft_buffer_data_type =
      KokkosFFT::Impl::add_pointer_n_t<complex_type, DIM + 1>;
  using FFTBufferType =
      Kokkos::View<fft_buffer_data_type, typename InViewType::array_layout,
                   typename InViewType::execution_space>;
  using out_buffer_data_type =
      KokkosFFT::Impl::add_pointer_n_t<out_value_type, DIM + 1>;
  using OutBufferType =
      Kokkos::View<out_buffer_data_type, typename OutViewType::array_layout,
                   typename OutViewType::execution_space>;

  using AllocationViewType =
      Kokkos::View<complex_type*, LayoutType, ExecutionSpace>;

  using ForwardBlockType0 =
      Block<ExecutionSpace, InViewType, InViewType, InBufferType, 1>;
  using ForwardBlockType1 =
      Block<ExecutionSpace, FFTViewType, FFTViewType, FFTBufferType, 1>;
  using BackwardBlockType0 =
      Block<ExecutionSpace, InViewType, InViewType, InBufferType, 1>;
  using BackwardBlockType1 =
      Block<ExecutionSpace, FFTViewType, FFTViewType, FFTBufferType, 1>;

  using FFTForwardPlanType =
      KokkosFFT::Plan<ExecutionSpace, InViewType, FFTViewType, 1>;
  using FFT2ForwardPlanType =
      KokkosFFT::Plan<ExecutionSpace, InViewType, FFTViewType, 2>;
  using FFTBackwardPlanType =
      KokkosFFT::Plan<ExecutionSpace, FFTViewType, InViewType, 1>;
  using FFT2BackwardPlanType =
      KokkosFFT::Plan<ExecutionSpace, FFTViewType, InViewType, 2>;

  execSpace m_exec_space;
  axes_type m_axes;
  topology_type m_in_topology;
  topology_type m_out_topology;
  MPI_Comm m_comm;
  KokkosFFT::Normalization m_normalization;

  // Analyse topology
  std::unique_ptr<SlabBlockAnalysesType> m_block_analyses;
  OperationType m_op_type;

  // Buffer view types
  InViewType m_in_T;
  OutViewType m_out_T;
  FFTViewType m_fft;

  // Buffer Allocations
  AllocationViewType m_send_buffer_allocation, m_recv_buffer_allocation;
  InBufferType m_send_buffer, m_recv_buffer;
  OutBufferType m_csend_buffer, m_crecv_buffer;

  std::unique_ptr<ForwardBlockType0> m_forward_block0;
  std::unique_ptr<ForwardBlockType1> m_forward_block1;
  std::unique_ptr<BackwardBlockType0> m_backward_block0;
  std::unique_ptr<BackwardBlockType1> m_backward_block1;

  // Internal FFT plans
  std::unique_ptr<FFTForwardPlanType> m_forward_plan;
  std::unique_ptr<FFT2ForwardPlanType> m_forward_plan2;
  std::unique_ptr<FFTBackwardPlanType> m_backward_plan;
  std::unique_ptr<FFT2BackwardPlanType> m_backward_plan2;

  int_map_type m_map_forward = {}, m_map_backward = {};

 public:
  explicit SlabInternalPlan(const ExecutionSpace& exec_space,
                            const InViewType& in, const OutViewType& out,
                            const axes_type& axes,
                            const topology_type& in_topology,
                            const topology_type& out_topology,
                            const MPI_Comm& comm,
                            KokkosFFT::Normalization normalization =
                                KokkosFFT::Normalization::backward)
      : m_exec_space(exec_space),
        m_axes(axes),
        m_in_topology(in_topology),
        m_out_topology(out_topology),
        m_comm(comm),
        m_normalization(normalization) {
    KOKKOSFFT_THROW_IF(
        !are_valid_extents(in, out, axes, in_topology, out_topology, comm),
        "Extents are not valid");

    // First get global shape to define buffer and next shape
    auto in_extents  = KokkosFFT::Impl::extract_extents(in);
    auto out_extents = KokkosFFT::Impl::extract_extents(out);

    auto gin_extents  = get_global_shape(in, m_in_topology, m_comm);
    auto gout_extents = get_global_shape(out, m_out_topology, m_comm);

    auto non_negative_axes =
        convert_negative_axes<std::size_t, int, 2, DIM>(axes);

    m_block_analyses = std::make_unique<SlabBlockAnalysesType>(
        in_extents, out_extents, gin_extents, gout_extents, in_topology,
        out_topology, non_negative_axes, m_comm);
    m_op_type = m_block_analyses->m_op_type;

    KOKKOSFFT_THROW_IF(m_block_analyses->m_block_infos.size() > 5,
                       "Maximum five blocks are expected");

    // Allocate buffer views
    m_send_buffer_allocation = AllocationViewType(
        "send_buffer_allocation", m_block_analyses->m_max_buffer_size);
    m_recv_buffer_allocation = AllocationViewType(
        "recv_buffer_allocation", m_block_analyses->m_max_buffer_size);

    switch (m_op_type) {
      case OperationType::F: {
        auto block0 = m_block_analyses->m_block_infos.at(0);
        // This means that FFT2 can be performed on the first dims
        // without transpose
        m_fft = FFTViewType("fft", KokkosFFT::Impl::create_layout<LayoutType>(
                                       block0.m_out_extents));

        m_forward_plan2 = std::make_unique<FFT2ForwardPlanType>(
            m_exec_space, in, m_fft, KokkosFFT::Direction::forward, m_axes);
        m_backward_plan2 = std::make_unique<FFT2BackwardPlanType>(
            m_exec_space, m_fft, in, KokkosFFT::Direction::backward, m_axes);
        break;
      }
      case OperationType::FT: {
        auto block0 = m_block_analyses->m_block_infos.at(0);
        // This means that FFT2 can be performed on the first dims
        // without transpose
        m_fft = FFTViewType("fft", KokkosFFT::Impl::create_layout<LayoutType>(
                                       block0.m_out_extents));
        m_forward_plan2 = std::make_unique<FFT2ForwardPlanType>(
            m_exec_space, in, m_fft, KokkosFFT::Direction::forward, m_axes);
        m_backward_plan2 = std::make_unique<FFT2BackwardPlanType>(
            m_exec_space, m_fft, in, KokkosFFT::Direction::backward, m_axes);

        auto block1 = m_block_analyses->m_block_infos.at(1);
        m_out_T     = OutViewType(
            m_send_buffer_allocation.data(),
            KokkosFFT::Impl::create_layout<LayoutType>(block1.m_out_extents));
        m_csend_buffer =
            OutBufferType(m_send_buffer_allocation.data(),
                          KokkosFFT::Impl::create_layout<LayoutType>(
                              block1.m_buffer_extents));
        m_crecv_buffer =
            OutBufferType(m_recv_buffer_allocation.data(),
                          KokkosFFT::Impl::create_layout<LayoutType>(
                              block1.m_buffer_extents));
        m_forward_block1 = std::make_unique<ForwardBlockType1>(
            m_exec_space, m_fft, m_out_T, m_csend_buffer, m_crecv_buffer,
            block1.m_in_map, block1.m_in_axis, block1.m_out_map,
            block1.m_out_axis, m_comm);
        m_backward_block1 = std::make_unique<BackwardBlockType1>(
            m_exec_space, m_out_T, m_fft, m_csend_buffer, m_crecv_buffer,
            block1.m_out_map, block1.m_out_axis, block1.m_in_map,
            block1.m_in_axis, m_comm);
        break;
      }
      case OperationType::TF: {
        std::cout << "OperationType::TF" << std::endl;
        auto block0 = m_block_analyses->m_block_infos.at(0);
        m_in_T      = InViewType(
            reinterpret_cast<in_value_type*>(m_send_buffer_allocation.data()),
            KokkosFFT::Impl::create_layout<LayoutType>(block0.m_out_extents));
        m_send_buffer = InBufferType(
            reinterpret_cast<in_value_type*>(m_send_buffer_allocation.data()),
            KokkosFFT::Impl::create_layout<LayoutType>(
                block0.m_buffer_extents));
        m_recv_buffer = InBufferType(
            reinterpret_cast<in_value_type*>(m_recv_buffer_allocation.data()),
            KokkosFFT::Impl::create_layout<LayoutType>(
                block0.m_buffer_extents));

        m_forward_block0 = std::make_unique<ForwardBlockType0>(
            m_exec_space, in, m_in_T, m_send_buffer, m_recv_buffer,
            block0.m_in_map, block0.m_in_axis, block0.m_out_map,
            block0.m_out_axis, m_comm);

        m_backward_block0 = std::make_unique<BackwardBlockType0>(
            m_exec_space, m_in_T, in, m_send_buffer, m_recv_buffer,
            block0.m_out_map, block0.m_out_axis, block0.m_in_map,
            block0.m_in_axis, m_comm);

        auto block1 = m_block_analyses->m_block_infos.at(1);

        // This means that FFT can be performed on the first dim
        // without transpose
        m_fft = FFTViewType("fft", KokkosFFT::Impl::create_layout<LayoutType>(
                                       block1.m_out_extents));
        m_forward_plan2 = std::make_unique<FFT2ForwardPlanType>(
            m_exec_space, m_in_T, m_fft, KokkosFFT::Direction::forward,
            to_array<int, std::size_t, 2>(block1.m_axes));
        m_backward_plan2 = std::make_unique<FFT2BackwardPlanType>(
            m_exec_space, m_fft, m_in_T, KokkosFFT::Direction::backward,
            to_array<int, std::size_t, 2>(block1.m_axes));
        if (block0.m_out_map != block0.m_in_map) {
          for (std::size_t i = 0; i < DIM; ++i) {
            m_map_forward.at(i) =
                KokkosFFT::Impl::get_index(block0.m_out_map, i);
            m_map_backward.at(i) = block0.m_out_map.at(i);
          }
        }
        break;
      }

      /*
      case OperationType::TFT: {
        auto block0 = m_block_analyses->m_block_infos.at(0);
        m_in_T      = InViewType(
            reinterpret_cast<in_value_type*>(m_send_buffer_allocation.data()),
            KokkosFFT::Impl::create_layout<LayoutType>(block0.m_out_extents));
        m_send_buffer = InBufferType(
            reinterpret_cast<in_value_type*>(m_send_buffer_allocation.data()),
            KokkosFFT::Impl::create_layout<LayoutType>(
                block0.m_buffer_extents));
        m_recv_buffer = InBufferType(
            reinterpret_cast<in_value_type*>(m_recv_buffer_allocation.data()),
            KokkosFFT::Impl::create_layout<LayoutType>(
                block0.m_buffer_extents));
        m_forward_block0 = std::make_unique<ForwardBlockType0>(
            m_exec_space, in, m_in_T, m_send_buffer, m_recv_buffer,
            block0.m_in_map, block0.m_in_axis, block0.m_out_map,
            block0.m_out_axis, m_comm);
        m_backward_block0 = std::make_unique<BackwardBlockType0>(
            m_exec_space, m_in_T, in, m_send_buffer, m_recv_buffer,
            block0.m_out_map, block0.m_out_axis, block0.m_in_map,
            block0.m_in_axis, m_comm);

        auto block1 = m_block_analyses->m_block_infos.at(1);
        // This means that FFT can be performed on the first dim
        // without transpose
        m_fft = FFTViewType("fft", KokkosFFT::Impl::create_layout<LayoutType>(
                                       block1.m_out_extents));

        m_forward_plan = std::make_unique<FFTForwardPlanType>(
            m_exec_space, m_in_T, m_fft, KokkosFFT::Direction::forward,
            to_array<int, std::size_t, 1>(block1.m_axes));
        m_backward_plan = std::make_unique<FFTBackwardPlanType>(
            m_exec_space, m_fft, m_in_T, KokkosFFT::Direction::backward,
            to_array<int, std::size_t, 1>(block1.m_axes));

        auto block2 = m_block_analyses->m_block_infos.at(2);
        m_out_T     = OutViewType(
            m_send_buffer_allocation.data(),
            KokkosFFT::Impl::create_layout<LayoutType>(block2.m_out_extents));
        m_csend_buffer =
            OutBufferType(m_send_buffer_allocation.data(),
                          KokkosFFT::Impl::create_layout<LayoutType>(
                              block2.m_buffer_extents));
        m_crecv_buffer =
            OutBufferType(m_recv_buffer_allocation.data(),
                          KokkosFFT::Impl::create_layout<LayoutType>(
                              block2.m_buffer_extents));
        m_forward_block1 = std::make_unique<ForwardBlockType1>(
            m_exec_space, m_fft, m_out_T, m_csend_buffer, m_crecv_buffer,
            block2.m_in_map, block2.m_in_axis, block2.m_out_map,
            block2.m_out_axis, m_comm);
        m_backward_block1 = std::make_unique<BackwardBlockType1>(
            m_exec_space, m_out_T, m_fft, m_csend_buffer, m_crecv_buffer,
            block2.m_out_map, block2.m_out_axis, block2.m_in_map,
            block2.m_in_axis, m_comm);
        break;
      }
      */
      default:  // No Operation
        break;
    };
  }

  void forward(const InViewType& in, const OutViewType& out) const {
    switch (m_op_type) {
      case OperationType::F: {
        KokkosFFT::execute(*m_forward_plan2, in, out);
        break;
      }
      case OperationType::FT: {
        KokkosFFT::execute(*m_forward_plan2, in, m_fft);
        (*m_forward_block1)(m_fft, out);
        break;
      }
      default: break;
    };
  }

  void backward(const OutViewType& out, const InViewType& in) const {
    switch (m_op_type) {
      case OperationType::F: {
        KokkosFFT::execute(*m_backward_plan2, out, in);
        break;
      }
      case OperationType::FT: {
        (*m_backward_block1)(out, m_fft);
        KokkosFFT::execute(*m_backward_plan2, m_fft, in);
        break;
      }
      default: break;
    };
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

  // Defining buffer type
  using in_buffer_data_type =
      KokkosFFT::Impl::add_pointer_n_t<in_value_type, DIM + 1>;
  using InBufferType =
      Kokkos::View<in_buffer_data_type, typename InViewType::array_layout,
                   typename InViewType::execution_space>;
  using fft_data_type = KokkosFFT::Impl::add_pointer_n_t<complex_type, DIM>;
  using FFTViewType =
      Kokkos::View<fft_data_type, typename InViewType::array_layout,
                   typename InViewType::execution_space>;
  using fft_buffer_data_type =
      KokkosFFT::Impl::add_pointer_n_t<complex_type, DIM + 1>;
  using FFTBufferType =
      Kokkos::View<fft_buffer_data_type, typename InViewType::array_layout,
                   typename InViewType::execution_space>;
  using out_buffer_data_type =
      KokkosFFT::Impl::add_pointer_n_t<out_value_type, DIM + 1>;
  using OutBufferType =
      Kokkos::View<out_buffer_data_type, typename OutViewType::array_layout,
                   typename OutViewType::execution_space>;

  using AllocationViewType =
      Kokkos::View<complex_type*, LayoutType, ExecutionSpace>;

  using ForwardBlockType0 =
      Block<ExecutionSpace, InViewType, InViewType, InBufferType, 1>;
  using ForwardBlockType1 =
      Block<ExecutionSpace, FFTViewType, FFTViewType, FFTBufferType, 1>;
  using BackwardBlockType0 =
      Block<ExecutionSpace, InViewType, InViewType, InBufferType, 1>;
  using BackwardBlockType1 =
      Block<ExecutionSpace, FFTViewType, FFTViewType, FFTBufferType, 1>;

  // First FFT to perform can be Real to Complex
  using FFTForwardPlanType0 =
      KokkosFFT::Plan<ExecutionSpace, InViewType, FFTViewType, 1>;
  using FFT2ForwardPlanType0 =
      KokkosFFT::Plan<ExecutionSpace, InViewType, FFTViewType, 2>;
  using FFTBackwardPlanType0 =
      KokkosFFT::Plan<ExecutionSpace, FFTViewType, InViewType, 1>;
  using FFT2BackwardPlanType0 =
      KokkosFFT::Plan<ExecutionSpace, FFTViewType, InViewType, 2>;

  // Second FFT to perform must be Complex to Complex
  using FFTForwardPlanType1 =
      KokkosFFT::Plan<ExecutionSpace, FFTViewType, FFTViewType, 1>;
  using FFT2ForwardPlanType1 =
      KokkosFFT::Plan<ExecutionSpace, FFTViewType, FFTViewType, 2>;
  using FFTBackwardPlanType1 =
      KokkosFFT::Plan<ExecutionSpace, FFTViewType, FFTViewType, 1>;
  using FFT2BackwardPlanType1 =
      KokkosFFT::Plan<ExecutionSpace, FFTViewType, FFTViewType, 2>;

  execSpace m_exec_space;
  axes_type m_axes;
  topology_type m_in_topology;
  topology_type m_out_topology;
  MPI_Comm m_comm;
  KokkosFFT::Normalization m_normalization;

  // Analyse topology
  std::unique_ptr<SlabBlockAnalysesType> m_block_analyses;
  OperationType m_op_type;

  // Buffer view types
  InViewType m_in_T;
  OutViewType m_out_T;
  FFTViewType m_fft;

  // Buffer Allocations
  AllocationViewType m_send_buffer_allocation, m_recv_buffer_allocation;
  InBufferType m_send_buffer, m_recv_buffer;
  OutBufferType m_csend_buffer, m_crecv_buffer;

  std::unique_ptr<ForwardBlockType0> m_forward_block0;
  std::unique_ptr<ForwardBlockType1> m_forward_block1, m_forward_block2;
  std::unique_ptr<BackwardBlockType0> m_backward_block0;
  std::unique_ptr<BackwardBlockType1> m_backward_block1, m_backward_block2;

  // Internal FFT plans
  std::unique_ptr<FFTForwardPlanType0> m_fft_plan0;
  std::unique_ptr<FFTForwardPlanType1> m_fft_plan1;
  std::unique_ptr<FFT2ForwardPlanType0> m_fft2_plan0;
  std::unique_ptr<FFT2ForwardPlanType1> m_fft2_plan1;
  std::unique_ptr<FFTBackwardPlanType0> m_ifft_plan0;
  std::unique_ptr<FFTBackwardPlanType1> m_ifft_plan1;
  std::unique_ptr<FFT2BackwardPlanType0> m_ifft2_plan0;
  std::unique_ptr<FFT2BackwardPlanType1> m_ifft2_plan1;

  int_map_type m_map_forward_in = {}, m_map_forward_out = {},
               m_map_backward_in = {}, m_map_backward_out = {};

 public:
  explicit SlabInternalPlan(const ExecutionSpace& exec_space,
                            const InViewType& in, const OutViewType& out,
                            const axes_type& axes,
                            const topology_type& in_topology,
                            const topology_type& out_topology,
                            const MPI_Comm& comm,
                            KokkosFFT::Normalization normalization =
                                KokkosFFT::Normalization::backward)
      : m_exec_space(exec_space),
        m_axes(axes),
        m_in_topology(in_topology),
        m_out_topology(out_topology),
        m_comm(comm),
        m_normalization(normalization) {
    KOKKOSFFT_THROW_IF(
        !are_valid_extents(in, out, axes, in_topology, out_topology, comm),
        "Extents are not valid");

    auto src_map = KokkosFFT::Impl::index_sequence<std::size_t, DIM, 0>();

    // First get global shape to define buffer and next shape
    auto in_extents  = KokkosFFT::Impl::extract_extents(in);
    auto out_extents = KokkosFFT::Impl::extract_extents(out);

    auto gin_extents  = get_global_shape(in, m_in_topology, m_comm);
    auto gout_extents = get_global_shape(out, m_out_topology, m_comm);

    auto non_negative_axes =
        convert_negative_axes<std::size_t, int, 3, DIM>(axes);

    m_block_analyses = std::make_unique<SlabBlockAnalysesType>(
        in_extents, out_extents, gin_extents, gout_extents, in_topology,
        out_topology, non_negative_axes, m_comm);
    m_op_type = m_block_analyses->m_op_type;

    KOKKOSFFT_THROW_IF(!(m_block_analyses->m_block_infos.size() >= 3 &&
                         m_block_analyses->m_block_infos.size() <= 5),
                       "Number blocks must be in [3, 5]");

    // Allocate buffer views
    m_send_buffer_allocation = AllocationViewType(
        "send_buffer_allocation", m_block_analyses->m_max_buffer_size);
    m_recv_buffer_allocation = AllocationViewType(
        "recv_buffer_allocation", m_block_analyses->m_max_buffer_size);

    switch (m_op_type) {
      case OperationType::FTF: {
        auto block0 = m_block_analyses->m_block_infos.at(0);

        m_in_T = InViewType(
            reinterpret_cast<in_value_type*>(m_recv_buffer_allocation.data()),
            KokkosFFT::Impl::create_layout<LayoutType>(block0.m_in_extents));
        m_fft = FFTViewType("fft", KokkosFFT::Impl::create_layout<LayoutType>(
                                       block0.m_out_extents));

        if (block0.m_axes.size() == 1) {
          m_op_type = OperationType::FTF2;
          // Then FFT1 + Transpose + FFT2
          m_fft_plan0 = std::make_unique<FFTForwardPlanType0>(
              m_exec_space, m_in_T, m_fft, KokkosFFT::Direction::forward,
              to_array<int, std::size_t, 1>(block0.m_axes));
          m_ifft_plan0 = std::make_unique<FFTBackwardPlanType0>(
              m_exec_space, m_fft, m_in_T, KokkosFFT::Direction::backward,
              to_array<int, std::size_t, 1>(block0.m_axes));
        } else {
          m_op_type = OperationType::F2TF;
          // Then FFT2 + Transpose + FFT1
          m_fft2_plan0 = std::make_unique<FFT2ForwardPlanType0>(
              m_exec_space, m_in_T, m_fft, KokkosFFT::Direction::forward,
              to_array<int, std::size_t, 2>(block0.m_axes));
          m_ifft2_plan0 = std::make_unique<FFT2BackwardPlanType0>(
              m_exec_space, m_fft, m_in_T, KokkosFFT::Direction::backward,
              to_array<int, std::size_t, 2>(block0.m_axes));
        }

        auto block1 = m_block_analyses->m_block_infos.at(1);
        m_out_T     = OutViewType(
            m_send_buffer_allocation.data(),
            KokkosFFT::Impl::create_layout<LayoutType>(block1.m_out_extents));
        m_csend_buffer =
            OutBufferType(m_send_buffer_allocation.data(),
                          KokkosFFT::Impl::create_layout<LayoutType>(
                              block1.m_buffer_extents));
        m_crecv_buffer =
            OutBufferType(m_recv_buffer_allocation.data(),
                          KokkosFFT::Impl::create_layout<LayoutType>(
                              block1.m_buffer_extents));
        m_forward_block1 = std::make_unique<ForwardBlockType1>(
            m_exec_space, m_fft, m_out_T, m_csend_buffer, m_crecv_buffer,
            block1.m_in_map, block1.m_in_axis, block1.m_out_map,
            block1.m_out_axis, m_comm);
        m_backward_block1 = std::make_unique<BackwardBlockType1>(
            m_exec_space, m_out_T, m_fft, m_csend_buffer, m_crecv_buffer,
            block1.m_out_map, block1.m_out_axis, block1.m_in_map,
            block1.m_in_axis, m_comm);

        auto block2 = m_block_analyses->m_block_infos.at(2);
        if (block2.m_axes.size() == 1) {
          // FFT along the final axis
          m_fft_plan1 = std::make_unique<FFTForwardPlanType1>(
              m_exec_space, m_out_T, m_out_T, KokkosFFT::Direction::forward,
              to_array<int, std::size_t, 1>(block2.m_axes));
          m_ifft_plan1 = std::make_unique<FFTBackwardPlanType1>(
              m_exec_space, m_out_T, m_out_T, KokkosFFT::Direction::backward,
              to_array<int, std::size_t, 1>(block2.m_axes));
        } else {
          // FFT2 along the final axeis
          m_fft2_plan1 = std::make_unique<FFT2ForwardPlanType1>(
              m_exec_space, m_out_T, m_out_T, KokkosFFT::Direction::forward,
              to_array<int, std::size_t, 2>(block2.m_axes));
          m_ifft2_plan1 = std::make_unique<FFT2BackwardPlanType1>(
              m_exec_space, m_out_T, m_out_T, KokkosFFT::Direction::backward,
              to_array<int, std::size_t, 2>(block2.m_axes));
        }

        // In this case, output data needed to be transposed locally
        if (block1.m_out_map != src_map) {
          for (std::size_t i = 0; i < DIM; ++i) {
            m_map_forward_out.at(i) =
                KokkosFFT::Impl::get_index(block1.m_out_map, i);
            m_map_backward_in.at(i) = block1.m_out_map.at(i);
          }
        }

        // In this case, input data needed to be transposed locally
        if (block0.m_in_map != src_map) {
          for (std::size_t i = 0; i < DIM; ++i) {
            m_map_forward_in.at(i) = block0.m_in_map.at(i);
            m_map_backward_out.at(i) =
                KokkosFFT::Impl::get_index(block0.m_in_map, i);
          }
        }
        break;
      }
      case OperationType::TFTF: {
        auto block0 = m_block_analyses->m_block_infos.at(0);

        m_in_T = InViewType(
            reinterpret_cast<in_value_type*>(m_send_buffer_allocation.data()),
            KokkosFFT::Impl::create_layout<LayoutType>(block0.m_out_extents));
        m_send_buffer = InBufferType(
            reinterpret_cast<in_value_type*>(m_send_buffer_allocation.data()),
            KokkosFFT::Impl::create_layout<LayoutType>(
                block0.m_buffer_extents));
        m_recv_buffer = InBufferType(
            reinterpret_cast<in_value_type*>(m_recv_buffer_allocation.data()),
            KokkosFFT::Impl::create_layout<LayoutType>(
                block0.m_buffer_extents));

        m_forward_block0 = std::make_unique<ForwardBlockType0>(
            m_exec_space, in, m_in_T, m_send_buffer, m_recv_buffer,
            block0.m_in_map, block0.m_in_axis, block0.m_out_map,
            block0.m_out_axis, m_comm);

        m_backward_block0 = std::make_unique<BackwardBlockType0>(
            m_exec_space, m_in_T, in, m_send_buffer, m_recv_buffer,
            block0.m_out_map, block0.m_out_axis, block0.m_in_map,
            block0.m_in_axis, m_comm);

        auto block1 = m_block_analyses->m_block_infos.at(1);

        m_fft = FFTViewType("fft", KokkosFFT::Impl::create_layout<LayoutType>(
                                       block1.m_out_extents));

        // Not sure block1.m_axes.size() == 1 is satisfied
        if (block1.m_axes.size() == 1) {
          m_op_type = OperationType::TFTF2;
          // Then FFT1 + Transpose + FFT2
          m_fft_plan0 = std::make_unique<FFTForwardPlanType0>(
              m_exec_space, m_in_T, m_fft, KokkosFFT::Direction::forward,
              to_array<int, std::size_t, 1>(block1.m_axes));
          m_ifft_plan0 = std::make_unique<FFTBackwardPlanType0>(
              m_exec_space, m_fft, m_in_T, KokkosFFT::Direction::backward,
              to_array<int, std::size_t, 1>(block1.m_axes));
        } else {
          m_op_type = OperationType::TF2TF;
          // Then FFT2 + Transpose + FFT1
          m_fft2_plan0 = std::make_unique<FFT2ForwardPlanType0>(
              m_exec_space, m_in_T, m_fft, KokkosFFT::Direction::forward,
              to_array<int, std::size_t, 2>(block1.m_axes));
          m_ifft2_plan0 = std::make_unique<FFT2BackwardPlanType0>(
              m_exec_space, m_fft, m_in_T, KokkosFFT::Direction::backward,
              to_array<int, std::size_t, 2>(block1.m_axes));
        }

        auto block2 = m_block_analyses->m_block_infos.at(2);
        m_out_T     = OutViewType(
            m_send_buffer_allocation.data(),
            KokkosFFT::Impl::create_layout<LayoutType>(block2.m_out_extents));
        m_csend_buffer =
            OutBufferType(m_send_buffer_allocation.data(),
                          KokkosFFT::Impl::create_layout<LayoutType>(
                              block2.m_buffer_extents));
        m_crecv_buffer =
            OutBufferType(m_recv_buffer_allocation.data(),
                          KokkosFFT::Impl::create_layout<LayoutType>(
                              block2.m_buffer_extents));
        m_forward_block1 = std::make_unique<ForwardBlockType1>(
            m_exec_space, m_fft, m_out_T, m_csend_buffer, m_crecv_buffer,
            block2.m_in_map, block2.m_in_axis, block2.m_out_map,
            block2.m_out_axis, m_comm);
        m_backward_block1 = std::make_unique<BackwardBlockType1>(
            m_exec_space, m_out_T, m_fft, m_csend_buffer, m_crecv_buffer,
            block2.m_out_map, block2.m_out_axis, block2.m_in_map,
            block2.m_in_axis, m_comm);

        auto block3 = m_block_analyses->m_block_infos.at(3);

        if (block3.m_axes.size() == 1) {
          // FFT along the final axis
          m_fft_plan1 = std::make_unique<FFTForwardPlanType1>(
              m_exec_space, m_out_T, m_out_T, KokkosFFT::Direction::forward,
              to_array<int, std::size_t, 1>(block3.m_axes));
          m_ifft_plan1 = std::make_unique<FFTBackwardPlanType1>(
              m_exec_space, m_out_T, m_out_T, KokkosFFT::Direction::backward,
              to_array<int, std::size_t, 1>(block3.m_axes));
        } else {
          // FFT2 along the final axes
          m_fft2_plan1 = std::make_unique<FFT2ForwardPlanType1>(
              m_exec_space, m_out_T, m_out_T, KokkosFFT::Direction::forward,
              to_array<int, std::size_t, 2>(block3.m_axes));
          m_ifft2_plan1 = std::make_unique<FFT2BackwardPlanType1>(
              m_exec_space, m_out_T, m_out_T, KokkosFFT::Direction::backward,
              to_array<int, std::size_t, 2>(block3.m_axes));
        }

        // In this case, output data needed to be transposed locally
        if (block3.m_out_map != src_map) {
          for (std::size_t i = 0; i < DIM; ++i) {
            m_map_forward_out.at(i) =
                KokkosFFT::Impl::get_index(block3.m_out_map, i);
            m_map_backward_in.at(i) = block3.m_out_map.at(i);
          }
        }

        // In this case, input data needed to be transposed locally
        // if (block0.m_in_map != src_map) {
        //  for (std::size_t i = 0; i < DIM; ++i) {
        //    m_map_forward_in.at(i) = block0.m_in_map.at(i);
        //    m_map_backward_out.at(i) =
        //        KokkosFFT::Impl::get_index(block0.m_in_map, i);
        //  }
        //}
        break;
      }
      case OperationType::FTFT: {
        auto block0 = m_block_analyses->m_block_infos.at(0);

        m_in_T = InViewType(
            reinterpret_cast<in_value_type*>(m_recv_buffer_allocation.data()),
            KokkosFFT::Impl::create_layout<LayoutType>(block0.m_in_extents));
        m_fft = FFTViewType("fft", KokkosFFT::Impl::create_layout<LayoutType>(
                                       block0.m_out_extents));

        if (block0.m_axes.size() == 1) {
          m_op_type = OperationType::FTF2T;
          // Then FFT1 + Transpose + FFT2 + Transpose
          m_fft_plan0 = std::make_unique<FFTForwardPlanType0>(
              m_exec_space, m_in_T, m_fft, KokkosFFT::Direction::forward,
              to_array<int, std::size_t, 1>(block0.m_axes));
          m_ifft_plan0 = std::make_unique<FFTBackwardPlanType0>(
              m_exec_space, m_fft, m_in_T, KokkosFFT::Direction::backward,
              to_array<int, std::size_t, 1>(block0.m_axes));
        } else {
          m_op_type = OperationType::F2TFT;
          // Then FFT2 + Transpose + FFT1 + Transpose
          m_fft2_plan0 = std::make_unique<FFT2ForwardPlanType0>(
              m_exec_space, m_in_T, m_fft, KokkosFFT::Direction::forward,
              to_array<int, std::size_t, 2>(block0.m_axes));
          m_ifft2_plan0 = std::make_unique<FFT2BackwardPlanType0>(
              m_exec_space, m_fft, m_in_T, KokkosFFT::Direction::backward,
              to_array<int, std::size_t, 2>(block0.m_axes));
        }

        auto block1 = m_block_analyses->m_block_infos.at(1);
        m_out_T     = OutViewType(
            m_send_buffer_allocation.data(),
            KokkosFFT::Impl::create_layout<LayoutType>(block1.m_out_extents));
        m_csend_buffer =
            OutBufferType(m_send_buffer_allocation.data(),
                          KokkosFFT::Impl::create_layout<LayoutType>(
                              block1.m_buffer_extents));
        m_crecv_buffer =
            OutBufferType(m_recv_buffer_allocation.data(),
                          KokkosFFT::Impl::create_layout<LayoutType>(
                              block1.m_buffer_extents));
        m_forward_block1 = std::make_unique<ForwardBlockType1>(
            m_exec_space, m_fft, m_out_T, m_csend_buffer, m_crecv_buffer,
            block1.m_in_map, block1.m_in_axis, block1.m_out_map,
            block1.m_out_axis, m_comm);
        m_backward_block1 = std::make_unique<BackwardBlockType1>(
            m_exec_space, m_out_T, m_fft, m_csend_buffer, m_crecv_buffer,
            block1.m_out_map, block1.m_out_axis, block1.m_in_map,
            block1.m_in_axis, m_comm);

        auto block2 = m_block_analyses->m_block_infos.at(2);
        OutViewType out_T(m_recv_buffer_allocation.data(), m_out_T.layout());

        // Cannot make this in-place
        if (block2.m_axes.size() == 1) {
          // FFT along the final axis
          m_fft_plan1 = std::make_unique<FFTForwardPlanType1>(
              m_exec_space, m_out_T, out_T, KokkosFFT::Direction::forward,
              to_array<int, std::size_t, 1>(block2.m_axes));
          m_ifft_plan1 = std::make_unique<FFTBackwardPlanType1>(
              m_exec_space, m_out_T, out_T, KokkosFFT::Direction::backward,
              to_array<int, std::size_t, 1>(block2.m_axes));
        } else {
          // FFT2 along the final axeis
          m_fft2_plan1 = std::make_unique<FFT2ForwardPlanType1>(
              m_exec_space, m_out_T, out_T, KokkosFFT::Direction::forward,
              to_array<int, std::size_t, 2>(block2.m_axes));
          m_ifft2_plan1 = std::make_unique<FFT2BackwardPlanType1>(
              m_exec_space, m_out_T, out_T, KokkosFFT::Direction::backward,
              to_array<int, std::size_t, 2>(block2.m_axes));
        }

        auto block3 = m_block_analyses->m_block_infos.at(3);
        m_csend_buffer =
            OutBufferType(m_send_buffer_allocation.data(),
                          KokkosFFT::Impl::create_layout<LayoutType>(
                              block3.m_buffer_extents));
        m_crecv_buffer =
            OutBufferType(m_recv_buffer_allocation.data(),
                          KokkosFFT::Impl::create_layout<LayoutType>(
                              block3.m_buffer_extents));
        m_forward_block2 = std::make_unique<ForwardBlockType1>(
            m_exec_space, out_T, out, m_csend_buffer, m_crecv_buffer,
            block3.m_in_map, block3.m_in_axis, block3.m_out_map,
            block3.m_out_axis, m_comm);
        m_backward_block2 = std::make_unique<BackwardBlockType1>(
            m_exec_space, out, out_T, m_csend_buffer, m_crecv_buffer,
            block3.m_out_map, block3.m_out_axis, block3.m_in_map,
            block3.m_in_axis, m_comm);

        // In this case, output data needed to be transposed locally
        if (block3.m_out_map != src_map) {
          for (std::size_t i = 0; i < DIM; ++i) {
            m_map_forward_out.at(i) =
                KokkosFFT::Impl::get_index(block3.m_out_map, i);
            m_map_backward_in.at(i) = block3.m_out_map.at(i);
          }
        }

        // In this case, input data needed to be transposed locally
        if (block0.m_in_map != src_map) {
          for (std::size_t i = 0; i < DIM; ++i) {
            m_map_forward_in.at(i) = block0.m_in_map.at(i);
            m_map_backward_out.at(i) =
                KokkosFFT::Impl::get_index(block0.m_in_map, i);
          }
        }
        break;
      }
      default:  // No Operation
        break;
    };
  }

  void forward(const InViewType& in, const OutViewType& out) const {
    switch (m_op_type) {
      case OperationType::FTF2: {
        if (m_map_forward_in == int_map_type{}) {
          KokkosFFT::execute(*m_fft_plan0, in, m_fft);
        } else {
          InViewType in_T(
              reinterpret_cast<in_value_type*>(m_recv_buffer_allocation.data()),
              m_in_T.layout());
          safe_transpose(m_exec_space, in, in_T, m_map_forward_in);
          KokkosFFT::execute(*m_fft_plan0, in_T, m_fft);
        }

        if (m_map_forward_out == int_map_type{}) {
          (*m_forward_block1)(m_fft, out);
          KokkosFFT::execute(*m_fft2_plan1, out, out);
        } else {
          (*m_forward_block1)(m_fft, m_out_T);
          KokkosFFT::execute(*m_fft2_plan1, m_out_T, m_out_T);
          safe_transpose(m_exec_space, m_out_T, out, m_map_forward_out);
        }
        break;
      }
      case OperationType::F2TF: {
        if (m_map_forward_in == int_map_type{}) {
          KokkosFFT::execute(*m_fft2_plan0, in, m_fft);
        } else {
          InViewType in_T(
              reinterpret_cast<in_value_type*>(m_recv_buffer_allocation.data()),
              m_in_T.layout());
          safe_transpose(m_exec_space, in, in_T, m_map_forward_in);
          KokkosFFT::execute(*m_fft2_plan0, in_T, m_fft);
        }

        if (m_map_forward_out == int_map_type{}) {
          (*m_forward_block1)(m_fft, out);
          KokkosFFT::execute(*m_fft_plan1, out, out);
        } else {
          (*m_forward_block1)(m_fft, m_out_T);
          KokkosFFT::execute(*m_fft_plan1, m_out_T, m_out_T);
          safe_transpose(m_exec_space, m_out_T, out, m_map_forward_out);
        }
        break;
      }
      case OperationType::TFTF2: {
        (*m_forward_block0)(in, m_in_T);
        KokkosFFT::execute(*m_fft_plan0, m_in_T, m_fft);
        if (m_map_forward_out == int_map_type{}) {
          (*m_forward_block1)(m_fft, out);
          KokkosFFT::execute(*m_fft2_plan1, out, out);
        } else {
          (*m_forward_block1)(m_fft, m_out_T);
          KokkosFFT::execute(*m_fft2_plan1, m_out_T, m_out_T);
          safe_transpose(m_exec_space, m_out_T, out, m_map_forward_out);
        }
        break;
      }
      case OperationType::TF2TF: {
        (*m_forward_block0)(in, m_in_T);
        KokkosFFT::execute(*m_fft2_plan0, m_in_T, m_fft);
        if (m_map_forward_out == int_map_type{}) {
          (*m_forward_block1)(m_fft, out);
          KokkosFFT::execute(*m_fft_plan1, out, out);
        } else {
          (*m_forward_block1)(m_fft, m_out_T);
          KokkosFFT::execute(*m_fft_plan1, m_out_T, m_out_T);
          safe_transpose(m_exec_space, m_out_T, out, m_map_forward_out);
        }
        break;
      }
      case OperationType::FTF2T: {
        if (m_map_forward_in == int_map_type{}) {
          KokkosFFT::execute(*m_fft_plan0, in, m_fft);
        } else {
          InViewType in_T(
              reinterpret_cast<in_value_type*>(m_recv_buffer_allocation.data()),
              m_in_T.layout());
          safe_transpose(m_exec_space, in, in_T, m_map_forward_in);
          KokkosFFT::execute(*m_fft_plan0, in_T, m_fft);
        }

        (*m_forward_block1)(m_fft, m_out_T);

        OutViewType out_T(
            reinterpret_cast<out_value_type*>(m_recv_buffer_allocation.data()),
            m_out_T.layout());
        KokkosFFT::execute(*m_fft2_plan1, m_out_T, out_T);

        (*m_forward_block2)(out_T, out);
        break;
      }
      case OperationType::F2TFT: {
        if (m_map_forward_in == int_map_type{}) {
          KokkosFFT::execute(*m_fft2_plan0, in, m_fft);
        } else {
          InViewType in_T(
              reinterpret_cast<in_value_type*>(m_recv_buffer_allocation.data()),
              m_in_T.layout());
          safe_transpose(m_exec_space, in, in_T, m_map_forward_in);
          KokkosFFT::execute(*m_fft2_plan0, in_T, m_fft);
        }

        (*m_forward_block1)(m_fft, m_out_T);

        OutViewType out_T(
            reinterpret_cast<out_value_type*>(m_recv_buffer_allocation.data()),
            m_out_T.layout());
        KokkosFFT::execute(*m_fft_plan1, m_out_T, out_T);

        (*m_forward_block2)(out_T, out);
        break;
      }
      default: break;
    };
  }

  void backward(const OutViewType& out, const InViewType& in) const {
    switch (m_op_type) {
      case OperationType::FTF2: {
        OutViewType out_T(
            reinterpret_cast<out_value_type*>(m_recv_buffer_allocation.data()),
            m_out_T.layout());
        if (m_map_backward_in == int_map_type{}) {
          KokkosFFT::execute(*m_ifft2_plan1, out, out);
          (*m_backward_block1)(out, m_fft);
        } else {
          safe_transpose(m_exec_space, out, out_T, m_map_backward_in);
          KokkosFFT::execute(*m_ifft2_plan1, out_T, out_T);
          (*m_backward_block1)(out_T, m_fft);
        }

        if (m_map_backward_out == int_map_type{}) {
          KokkosFFT::execute(*m_ifft_plan0, m_fft, in);
        } else {
          InViewType in_T(
              reinterpret_cast<in_value_type*>(m_recv_buffer_allocation.data()),
              m_in_T.layout());
          KokkosFFT::execute(*m_ifft_plan0, m_fft, in_T);
          safe_transpose(m_exec_space, in_T, in, m_map_backward_out);
        }
        break;
      }
      case OperationType::F2TF: {
        OutViewType out_T(
            reinterpret_cast<out_value_type*>(m_recv_buffer_allocation.data()),
            m_out_T.layout());
        if (m_map_backward_in == int_map_type{}) {
          KokkosFFT::execute(*m_ifft_plan1, out, out);
          (*m_backward_block1)(out, m_fft);
        } else {
          safe_transpose(m_exec_space, out, out_T, m_map_backward_in);
          KokkosFFT::execute(*m_ifft_plan1, out_T, out_T);
          (*m_backward_block1)(out_T, m_fft);
        }

        if (m_map_backward_out == int_map_type{}) {
          KokkosFFT::execute(*m_ifft2_plan0, m_fft, in);
        } else {
          InViewType in_T(
              reinterpret_cast<in_value_type*>(m_recv_buffer_allocation.data()),
              m_in_T.layout());
          KokkosFFT::execute(*m_ifft2_plan0, m_fft, in_T);
          safe_transpose(m_exec_space, in_T, in, m_map_backward_out);
        }
        break;
      }
      case OperationType::TFTF2: {
        InViewType in_T(
            reinterpret_cast<in_value_type*>(m_recv_buffer_allocation.data()),
            m_in_T.layout());
        OutViewType out_T(
            reinterpret_cast<out_value_type*>(m_recv_buffer_allocation.data()),
            m_out_T.layout());
        if (m_map_backward_in == int_map_type{}) {
          KokkosFFT::execute(*m_ifft2_plan1, out, out);
          (*m_backward_block1)(out, m_fft);
        } else {
          safe_transpose(m_exec_space, out, out_T, m_map_backward_in);
          KokkosFFT::execute(*m_ifft2_plan1, out_T, out_T);
          (*m_backward_block1)(out_T, m_fft);
        }
        KokkosFFT::execute(*m_ifft_plan0, m_fft, in_T);
        (*m_backward_block0)(in_T, in);
        break;
      }
      case OperationType::TF2TF: {
        InViewType in_T(
            reinterpret_cast<in_value_type*>(m_recv_buffer_allocation.data()),
            m_in_T.layout());
        OutViewType out_T(
            reinterpret_cast<out_value_type*>(m_recv_buffer_allocation.data()),
            m_out_T.layout());
        if (m_map_backward_in == int_map_type{}) {
          KokkosFFT::execute(*m_ifft_plan1, out, out);
          (*m_backward_block1)(out, m_fft);
        } else {
          safe_transpose(m_exec_space, out, out_T, m_map_backward_in);
          KokkosFFT::execute(*m_ifft_plan1, out_T, out_T);
          (*m_backward_block1)(out_T, m_fft);
        }
        KokkosFFT::execute(*m_ifft2_plan0, m_fft, in_T);
        (*m_backward_block0)(in_T, in);
        break;
      }
      case OperationType::FTF2T: {
        OutViewType out_T(
            reinterpret_cast<out_value_type*>(m_recv_buffer_allocation.data()),
            m_out_T.layout());

        (*m_backward_block2)(out, m_out_T);
        KokkosFFT::execute(*m_ifft2_plan1, m_out_T, out_T);

        (*m_backward_block1)(out_T, m_fft);

        if (m_map_backward_out == int_map_type{}) {
          KokkosFFT::execute(*m_ifft_plan0, m_fft, in);
        } else {
          InViewType in_T(
              reinterpret_cast<in_value_type*>(m_recv_buffer_allocation.data()),
              m_in_T.layout());
          KokkosFFT::execute(*m_ifft_plan0, m_fft, in_T);
          safe_transpose(m_exec_space, in_T, in, m_map_backward_out);
        }
        break;
      }
      case OperationType::F2TFT: {
        OutViewType out_T(
            reinterpret_cast<out_value_type*>(m_recv_buffer_allocation.data()),
            m_out_T.layout());

        (*m_backward_block2)(out, m_out_T);
        KokkosFFT::execute(*m_ifft_plan1, m_out_T, out_T);

        (*m_backward_block1)(out_T, m_fft);

        if (m_map_backward_out == int_map_type{}) {
          KokkosFFT::execute(*m_ifft2_plan0, m_fft, in);
        } else {
          InViewType in_T(
              reinterpret_cast<in_value_type*>(m_recv_buffer_allocation.data()),
              m_in_T.layout());
          KokkosFFT::execute(*m_ifft2_plan0, m_fft, in_T);
          safe_transpose(m_exec_space, in_T, in, m_map_backward_out);
        }
        break;
      }

      default: break;
    };
  }
};

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1>
class SlabPlan
    : public InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM> {
  using InternalPlanType =
      SlabInternalPlan<ExecutionSpace, InViewType, OutViewType, DIM>;
  using extents_type = std::array<std::size_t, InViewType::rank()>;
  using axes_type    = KokkosFFT::axis_type<DIM>;

  InternalPlanType m_internal_plan;
  extents_type m_in_extents, m_out_extents;

  using InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM>::good;

 public:
  explicit SlabPlan(
      const ExecutionSpace& exec_space, const InViewType& in,
      const OutViewType& out, const axes_type& axes,
      const extents_type& in_topology, const extents_type& out_topology,
      const MPI_Comm& comm,
      KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward)
      : InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM>(
            exec_space, in, out, axes, in_topology, out_topology, comm, norm),
        m_internal_plan(exec_space, in, out, axes, in_topology, out_topology,
                        comm, norm),
        m_in_extents(KokkosFFT::Impl::extract_extents(in)),
        m_out_extents(KokkosFFT::Impl::extract_extents(out)) {
    auto in_size  = get_size(in_topology);
    auto out_size = get_size(out_topology);

    KOKKOSFFT_THROW_IF(in_size != out_size,
                       "Input and output topologies must have the same size.");

    bool is_slab =
        is_slab_topology(in_topology) && is_slab_topology(out_topology);
    KOKKOSFFT_THROW_IF(!is_slab,
                       "Input and output topologies must be slab topologies.");
  }

  void forward(const InViewType& in, const OutViewType& out) const override {
    good(in, out);
    m_internal_plan.forward(in, out);
  }

  void backward(const OutViewType& out, const InViewType& in) const override {
    good(in, out);
    m_internal_plan.backward(out, in);
  }
};

#endif
