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
  // using SlabBlockAnalysesType = SlabBlockAnalysesInternal<in_value_type,
  // LayoutType, int, DIM, 1>;

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
        std::cout << "OperationType::F" << std::endl;
        auto block0 = m_block_analyses->m_block_infos.at(0);
        // This means that FFT can be performed on the first dim
        // without transpose
        m_fft = FFTViewType("fft", KokkosFFT::Impl::create_layout<LayoutType>(
                                       block0.m_out_extents));
        m_forward_plan = std::make_unique<FFTForwardPlanType>(
            m_exec_space, in, m_fft, KokkosFFT::Direction::forward, m_axes);
        break;
      }
      case OperationType::FT: {
        std::cout << "OperationType::FT" << std::endl;
        auto block0 = m_block_analyses->m_block_infos.at(0);
        // This means that FFT can be performed on the first dim
        // without transpose
        std::cout << "block0.m_out_extents: " << block0.m_out_extents[0] << ", "
                  << block0.m_out_extents[1] << std::endl;
        m_fft = FFTViewType("fft", KokkosFFT::Impl::create_layout<LayoutType>(
                                       block0.m_out_extents));
        m_forward_plan = std::make_unique<FFTForwardPlanType>(
            m_exec_space, in, m_fft, KokkosFFT::Direction::forward, m_axes);
        m_backward_plan = std::make_unique<FFTBackwardPlanType>(
            m_exec_space, m_fft, in, KokkosFFT::Direction::backward, m_axes);

        auto block1 = m_block_analyses->m_block_infos.at(1);
        std::cout << "block1.m_out_extents: " << block1.m_out_extents[0] << ", "
                  << block1.m_out_extents[1] << std::endl;
        m_out_T = OutViewType(
            m_send_buffer_allocation.data(),
            KokkosFFT::Impl::create_layout<LayoutType>(block1.m_out_extents));
        std::cout << "block1.m_buffer_extents: " << block1.m_buffer_extents[0]
                  << ", " << block1.m_buffer_extents[1] << ", "
                  << block1.m_buffer_extents[2] << std::endl;
        m_csend_buffer =
            OutBufferType(m_send_buffer_allocation.data(),
                          KokkosFFT::Impl::create_layout<LayoutType>(
                              block1.m_buffer_extents));
        m_crecv_buffer =
            OutBufferType(m_recv_buffer_allocation.data(),
                          KokkosFFT::Impl::create_layout<LayoutType>(
                              block1.m_buffer_extents));
        std::cout << "block1.m_in_map, m_in_axis: " << block1.m_in_map[0]
                  << ", " << block1.m_in_map[1] << ", " << block1.m_in_axis
                  << std::endl;
        std::cout << "block1.m_out_map, m_out_axis: " << block1.m_out_map[0]
                  << ", " << block1.m_out_map[1] << ", " << block1.m_out_axis
                  << std::endl;
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
        std::cout << "block0.m_out_extents: " << block0.m_out_extents[0] << ", "
                  << block0.m_out_extents[1] << std::endl;
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
        std::cout << "block0.m_buffer_extents: " << block0.m_buffer_extents[0]
                  << ", " << block0.m_buffer_extents[1] << ", "
                  << block0.m_buffer_extents[2] << std::endl;
        m_forward_block0 = std::make_unique<ForwardBlockType0>(
            m_exec_space, in, m_in_T, m_send_buffer, m_recv_buffer,
            block0.m_in_map, block0.m_in_axis, block0.m_out_map,
            block0.m_out_axis, m_comm);

        std::cout << "block0.in_map: " << block0.m_in_map[0] << ", "
                  << block0.m_in_map[1] << std::endl;
        std::cout << "block0.in_axis: " << block0.m_in_axis << std::endl;
        std::cout << "block0.out_map: " << block0.m_out_map[0] << ", "
                  << block0.m_out_map[1] << std::endl;
        std::cout << "block0.out_axis: " << block0.m_out_axis << std::endl;
        m_backward_block0 = std::make_unique<BackwardBlockType0>(
            m_exec_space, m_in_T, in, m_send_buffer, m_recv_buffer,
            block0.m_out_map, block0.m_out_axis, block0.m_in_map,
            block0.m_in_axis, m_comm);

        auto block1 = m_block_analyses->m_block_infos.at(1);
        std::cout << "block1.m_out_extents: " << block1.m_out_extents[0] << ", "
                  << block1.m_out_extents[1] << std::endl;
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
        // if (block1.m_out_extents != out_extents) {
        if (block0.m_out_map != block0.m_in_map) {
          std::cout << "block0.m_out_map != block0.m_in_map" << std::endl;
          for (std::size_t i = 0; i < DIM; ++i) {
            m_map_forward.at(i) =
                KokkosFFT::Impl::get_index(block0.m_out_map, i);
            m_map_backward.at(i) = block0.m_out_map.at(i);
          }
        }
        break;
      }
      case OperationType::TFT: {
        std::cout << "OperationType::TFT" << std::endl;
        auto block0 = m_block_analyses->m_block_infos.at(0);
        m_in_T      = InViewType(
            reinterpret_cast<in_value_type*>(m_send_buffer_allocation.data()),
            KokkosFFT::Impl::create_layout<LayoutType>(block0.m_out_extents));
        std::cout << "block0.m_out_extents: " << block0.m_out_extents[0] << ", "
                  << block0.m_out_extents[1] << std::endl;
        std::cout << "block0.m_buffer_extents: " << block0.m_buffer_extents[0]
                  << ", " << block0.m_buffer_extents[1] << ", "
                  << block0.m_buffer_extents[2] << std::endl;
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

        std::cout << "block0.in_map: " << block0.m_in_map[0] << ", "
                  << block0.m_in_map[1] << std::endl;
        std::cout << "block0.in_axis: " << block0.m_in_axis << std::endl;
        std::cout << "block0.out_map: " << block0.m_out_map[0] << ", "
                  << block0.m_out_map[1] << std::endl;
        std::cout << "block0.out_axis: " << block0.m_out_axis << std::endl;

        auto block1 = m_block_analyses->m_block_infos.at(1);
        // This means that FFT can be performed on the first dim
        // without transpose
        m_fft = FFTViewType("fft", KokkosFFT::Impl::create_layout<LayoutType>(
                                       block1.m_out_extents));

        std::cout << "block1.m_out_extents: " << block1.m_out_extents[0] << ", "
                  << block1.m_out_extents[1] << std::endl;
        std::cout << "block1.m_axes: " << block1.m_axes[0] << std::endl;
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
        std::cout << "block2.m_out_extents: " << block2.m_out_extents[0] << ", "
                  << block2.m_out_extents[1] << std::endl;
        std::cout << "block2.m_buffer_extents: " << block2.m_buffer_extents[0]
                  << ", " << block2.m_buffer_extents[1] << ", "
                  << block2.m_buffer_extents[2] << std::endl;
        std::cout << "block2.in_map: " << block2.m_in_map[0] << ", "
                  << block2.m_in_map[1] << std::endl;
        std::cout << "block2.in_axis: " << block2.m_in_axis << std::endl;
        std::cout << "block2.out_map: " << block2.m_out_map[0] << ", "
                  << block2.m_out_map[1] << std::endl;
        std::cout << "block2.out_axis: " << block2.m_out_axis << std::endl;
        break;
      }
      default:  // No Operation
        break;
    };
  }

  void forward(const InViewType& in, const OutViewType& out) {
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

  void backward(const OutViewType& out, const InViewType& in) {
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

        // KokkosFFT::execute(*m_backward_plan, out, in_T);
        //(*m_backward_block0)(in_T, in);

        if (m_map_backward == int_map_type{}) {
          // No transpose needed
          KokkosFFT::execute(*m_backward_plan, out, in_T);
        } else {
          OutViewType out_T(reinterpret_cast<out_value_type*>(
                                m_send_buffer_allocation.data()),
                            m_fft.layout());
          safe_transpose(m_exec_space, out, out_T, m_map_backward);

          std::cout << "out_T.extent(0) = " << out_T.extent(0) << std::endl;
          std::cout << "out_T.extent(1) = " << out_T.extent(1) << std::endl;
          std::cout << "in_T.extent(0) = " << in_T.extent(0) << std::endl;
          std::cout << "in_T.extent(1) = " << in_T.extent(1) << std::endl;
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

/*
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
  using execSpace     = ExecutionSpace;
  using in_value_type = typename InViewType::non_const_value_type;
  using float_type   = KokkosFFT::Impl::base_floating_point_type<in_value_type>;
  using complex_type = Kokkos::complex<float_type>;
};

/// \brief Internal plan for 3D FFT with slab decomposition
/// 1. 1D FFT + Transpose + 1D FFT + Transpose + 1D FFT (+ Transpose)
/// 2. 2D FFT + Transpose + 1D FFT (+ Transpose)
/// 3. Transpose + 3D FFT (+ Transpose)
/// 4. Transpose + 2D FFT + Transpose + 1D FFT (+ Transpose)
/// 5. Transpose + 1D FFT + Transpose + 1D FFT + Transpose + 1D FFT (+
Transpose)
///
/// \tparam ExecutionSpace
/// \tparam InViewType
/// \tparam OutViewType

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
struct SlabInternalPlan<ExecutionSpace, InViewType, OutViewType, 3> {
  using execSpace     = ExecutionSpace;
  using in_value_type = typename InViewType::non_const_value_type;
  using float_type   = KokkosFFT::Impl::base_floating_point_type<in_value_type>;
  using complex_type = Kokkos::complex<float_type>;
};
*/

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1>
class SlabPlan {
  static_assert(DIM >= 1 && DIM <= 3,
                "SlabPlan: the Rank of FFT axes must be between 1 and 3");
  using InternalPlanType =
      SlabInternalPlan<ExecutionSpace, InViewType, OutViewType, DIM>;
  using extents_type = std::array<std::size_t, InViewType::rank()>;
  using axes_type    = KokkosFFT::axis_type<DIM>;

  InternalPlanType m_internal_plan;
  extents_type m_in_extents, m_out_extents;

 public:
  explicit SlabPlan(
      const ExecutionSpace& exec_space, const InViewType& in,
      const OutViewType& out, const axes_type& axes,
      const extents_type& in_topology, const extents_type& out_topology,
      const MPI_Comm& comm,
      KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward)
      : m_internal_plan(exec_space, in, out, axes, in_topology, out_topology,
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

  void forward(const InViewType& in, const OutViewType& out) {
    good(in, out);
    m_internal_plan.forward(in, out);
  }

  void backward(const OutViewType& out, const InViewType& in) {
    good(in, out);
    m_internal_plan.backward(out, in);
  }

 private:
  void good(const InViewType& in, const OutViewType& out) const {
    auto in_extents  = KokkosFFT::Impl::extract_extents(in);
    auto out_extents = KokkosFFT::Impl::extract_extents(out);

    KOKKOSFFT_THROW_IF(in_extents != m_in_extents,
                       "extents of input View for plan and "
                       "execution are not identical.");

    KOKKOSFFT_THROW_IF(out_extents != m_out_extents,
                       "extents of output View for plan and "
                       "execution are not identical.");
  }
};

#endif
