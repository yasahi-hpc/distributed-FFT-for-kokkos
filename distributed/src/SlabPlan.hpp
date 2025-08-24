#ifndef SLABPLAN_HPP
#define SLABPLAN_HPP

#include <vector>
#include <memory>
#include <iostream>
#include <sstream>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "TransBlock.hpp"
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

  using AllocationViewType =
      Kokkos::View<complex_type*, LayoutType, ExecutionSpace>;

  // Type for transpose Block
  using TransBlockType = TransBlock<ExecutionSpace, DIM>;

  using FFTForwardPlanType =
      KokkosFFT::Plan<ExecutionSpace, InViewType, OutViewType, 1>;
  using FFTBackwardPlanType =
      KokkosFFT::Plan<ExecutionSpace, OutViewType, InViewType, 1>;

  execSpace m_exec_space;
  axes_type m_axes;
  topology_type m_in_topology;
  topology_type m_out_topology;
  MPI_Comm m_comm;
  KokkosFFT::Normalization m_normalization;

  // Analyse topology
  std::unique_ptr<SlabBlockAnalysesType> m_block_analyses;

  // Buffer view types
  InViewType m_in_T;
  OutViewType m_out_T;

  // Buffer Allocations
  AllocationViewType m_send_buffer_allocation, m_recv_buffer_allocation;

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
      KokkosFFT::execute(*m_forward_plan, in, out, m_normalization);
    } else if (m_map_forward_in == int_map_type{} &&
               m_map_forward_out != int_map_type{}) {
      KokkosFFT::execute(*m_forward_plan, in, m_out_T, m_normalization);
      safe_transpose(m_exec_space, m_out_T, out, m_map_forward_out);
    } else if (m_map_forward_in != int_map_type{} &&
               m_map_forward_out == int_map_type{}) {
      safe_transpose(m_exec_space, in, m_in_T, m_map_forward_in);
      KokkosFFT::execute(*m_forward_plan, m_in_T, out, m_normalization);
    } else if (m_map_forward_in != int_map_type{} &&
               m_map_forward_out != int_map_type{}) {
      safe_transpose(m_exec_space, in, m_in_T, m_map_forward_in);
      KokkosFFT::execute(*m_forward_plan, m_in_T, m_out_T, m_normalization);
      safe_transpose(m_exec_space, m_out_T, out, m_map_forward_out);
    }
  }

  template <typename InType, typename OutType>
  void backward_fft(const OutType& out, const InType& in) const {
    if (m_map_backward_in == int_map_type{} &&
        m_map_backward_out == int_map_type{}) {
      KokkosFFT::execute(*m_backward_plan, out, in, m_normalization);
    } else if (m_map_backward_in == int_map_type{} &&
               m_map_backward_out != int_map_type{}) {
      KokkosFFT::execute(*m_backward_plan, out, m_in_T, m_normalization);
      safe_transpose(m_exec_space, m_in_T, in, m_map_backward_out);
    } else if (m_map_backward_in != int_map_type{} &&
               m_map_backward_out == int_map_type{}) {
      safe_transpose(m_exec_space, out, m_out_T, m_map_backward_in);
      KokkosFFT::execute(*m_backward_plan, m_out_T, in, m_normalization);
    } else if (m_map_backward_in != int_map_type{} &&
               m_map_backward_out != int_map_type{}) {
      safe_transpose(m_exec_space, out, m_out_T, m_map_backward_in);
      KokkosFFT::execute(*m_backward_plan, m_out_T, m_in_T, m_normalization);
      safe_transpose(m_exec_space, m_in_T, in, m_map_backward_out);
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
          m_exec_space, m_in_T, m_out_T, KokkosFFT::Direction::forward,
          to_array<int, std::size_t, 1>(block.m_axes));
      m_backward_plan = std::make_unique<FFTBackwardPlanType>(
          m_exec_space, m_out_T, m_in_T, KokkosFFT::Direction::backward,
          to_array<int, std::size_t, 1>(block.m_axes));
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
          block.m_out_map, block.m_out_axis, m_comm));
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
            in, in_view, m_send_buffer_allocation, m_recv_buffer_allocation,
            KokkosFFT::Direction::forward);
      }
    } else {
      if (block_type == BlockType::FFT) {
        forward_fft(in_view, out_view);
      } else if (block_type == BlockType::Transpose) {
        (*m_trans_blocks.at(block.m_block_idx))(
            out_view, out_view_T, m_send_buffer_allocation,
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
            out, out_view, m_recv_buffer_allocation, m_send_buffer_allocation,
            KokkosFFT::Direction::backward);
      }
    } else {
      if (block_type == BlockType::FFT) {
        backward_fft(out_view, in_view);
      } else if (block_type == BlockType::Transpose) {
        (*m_trans_blocks.at(block.m_block_idx))(
            in_view, in_view_T, m_recv_buffer_allocation,
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
  using TransBlockType = TransBlock<ExecutionSpace, DIM>;

  // First FFT to perform can be Real to Complex
  using FFTForwardPlanType0 =
      KokkosFFT::Plan<ExecutionSpace, InViewType, OutViewType, 1>;
  using FFT2ForwardPlanType0 =
      KokkosFFT::Plan<ExecutionSpace, InViewType, OutViewType, 2>;
  using FFTBackwardPlanType0 =
      KokkosFFT::Plan<ExecutionSpace, OutViewType, InViewType, 1>;
  using FFT2BackwardPlanType0 =
      KokkosFFT::Plan<ExecutionSpace, OutViewType, InViewType, 2>;

  // Second FFT to perform must be Complex to Complex
  using FFTForwardPlanType1 =
      KokkosFFT::Plan<ExecutionSpace, OutViewType, OutViewType, 1>;
  using FFTBackwardPlanType1 =
      KokkosFFT::Plan<ExecutionSpace, OutViewType, OutViewType, 1>;

  execSpace m_exec_space;
  axes_type m_axes;
  topology_type m_in_topology;
  topology_type m_out_topology;
  MPI_Comm m_comm;
  KokkosFFT::Normalization m_normalization;

  // Analyse topology
  std::unique_ptr<SlabBlockAnalysesType> m_block_analyses;
  std::size_t first_FFT_dim = 1;

  // Buffer view types
  InViewType m_in_T;
  OutViewType m_out_T, m_out_T2;

  // Buffer Allocations
  AllocationViewType m_send_buffer_allocation, m_recv_buffer_allocation;

  // Internal transpose blocks
  std::vector<std::unique_ptr<TransBlockType>> m_trans_blocks;

  // Internal FFT plans
  std::unique_ptr<FFTForwardPlanType0> m_fft_plan0;
  std::unique_ptr<FFTForwardPlanType1> m_fft_plan1;
  std::unique_ptr<FFT2ForwardPlanType0> m_fft2_plan0;
  std::unique_ptr<FFTBackwardPlanType0> m_ifft_plan0;
  std::unique_ptr<FFTBackwardPlanType1> m_ifft_plan1;
  std::unique_ptr<FFT2BackwardPlanType0> m_ifft2_plan0;

  // Mappings for local transpose
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
      if (first_FFT_dim == 1) {
        if (m_map_forward_in == int_map_type{}) {
          KokkosFFT::execute(*m_fft_plan0, in, out, m_normalization);
        } else {
          safe_transpose(m_exec_space, in, m_in_T, m_map_forward_in);
          KokkosFFT::execute(*m_fft_plan0, m_in_T, out, m_normalization);
        }
      } else if (first_FFT_dim == 2) {
        if (m_map_forward_in == int_map_type{} &&
            m_map_forward_out == int_map_type{}) {
          KokkosFFT::execute(*m_fft2_plan0, in, out, m_normalization);
        } else if (m_map_forward_in == int_map_type{} &&
                   m_map_forward_out != int_map_type{}) {
          KokkosFFT::execute(*m_fft2_plan0, m_in_T, m_out_T, m_normalization);
          safe_transpose(m_exec_space, m_out_T, out, m_map_forward_out);
        } else if (m_map_forward_in != int_map_type{} &&
                   m_map_forward_out == int_map_type{}) {
          safe_transpose(m_exec_space, in, m_in_T, m_map_forward_in);
          KokkosFFT::execute(*m_fft2_plan0, m_in_T, out, m_normalization);
        } else {
          safe_transpose(m_exec_space, in, m_in_T, m_map_forward_in);
          KokkosFFT::execute(*m_fft2_plan0, m_in_T, m_out_T, m_normalization);
          safe_transpose(m_exec_space, m_out_T, out, m_map_forward_out);
        }
      }
    } else if constexpr (STEP == 1) {
      if (first_FFT_dim == 1) {
        if (m_map_forward_out == int_map_type{}) {
          KokkosFFT::execute(*m_fft_plan1, in, out, m_normalization);
        } else {
          KokkosFFT::execute(*m_fft_plan1, in, m_out_T2, m_normalization);
          safe_transpose(m_exec_space, m_out_T2, out, m_map_forward_out);
        }
      }
    }
  }

  template <std::size_t STEP, typename InType, typename OutType>
  void backward_fft(const OutType& out, const InType& in) const {
    if constexpr (STEP == 0) {
      if (first_FFT_dim == 1) {
        if (m_map_backward_out == int_map_type{}) {
          KokkosFFT::execute(*m_ifft_plan0, out, in, m_normalization);
        } else {
          KokkosFFT::execute(*m_ifft_plan0, out, m_in_T, m_normalization);
          safe_transpose(m_exec_space, m_in_T, in, m_map_backward_out);
        }
      } else if (first_FFT_dim == 2) {
        if (m_map_backward_in == int_map_type{} &&
            m_map_backward_out == int_map_type{}) {
          KokkosFFT::execute(*m_ifft2_plan0, out, in, m_normalization);
        } else if (m_map_backward_in == int_map_type{} &&
                   m_map_backward_out != int_map_type{}) {
          KokkosFFT::execute(*m_ifft2_plan0, out, m_in_T, m_normalization);
          safe_transpose(m_exec_space, m_in_T, in, m_map_backward_out);
        } else if (m_map_backward_in != int_map_type{} &&
                   m_map_backward_out == int_map_type{}) {
          safe_transpose(m_exec_space, out, m_out_T, m_map_backward_in);
          KokkosFFT::execute(*m_ifft2_plan0, m_out_T, in, m_normalization);
        } else if (m_map_backward_in != int_map_type{} &&
                   m_map_backward_out != int_map_type{}) {
          safe_transpose(m_exec_space, out, m_out_T, m_map_backward_in);
          KokkosFFT::execute(*m_ifft2_plan0, m_out_T, m_in_T, m_normalization);
          safe_transpose(m_exec_space, m_in_T, in, m_map_backward_out);
        }
      }
    } else if constexpr (STEP == 1) {
      if (m_map_backward_in == int_map_type{}) {
        KokkosFFT::execute(*m_ifft_plan1, out, in, m_normalization);
      } else {
        safe_transpose(m_exec_space, out, m_out_T2, m_map_backward_in);
        KokkosFFT::execute(*m_ifft_plan1, m_out_T2, in, m_normalization);
      }
    }
  }

  void set_block_impl(const std::size_t block_idx) {
    auto src_map    = KokkosFFT::Impl::index_sequence<std::size_t, DIM, 0>();
    auto block      = m_block_analyses->m_block_infos.at(block_idx);
    auto block_type = block.m_block_type;

    if (block_type == BlockType::FFT) {
      if (block.m_block_idx == 0) {
        first_FFT_dim = block.m_axes.size();
        m_in_T        = InViewType(
            reinterpret_cast<in_value_type*>(m_send_buffer_allocation.data()),
            KokkosFFT::Impl::create_layout<LayoutType>(block.m_in_extents));
        m_out_T = OutViewType(
            m_recv_buffer_allocation.data(),
            KokkosFFT::Impl::create_layout<LayoutType>(block.m_out_extents));

        if (block.m_axes.size() == 1) {
          m_fft_plan0 = std::make_unique<FFTForwardPlanType0>(
              m_exec_space, m_in_T, m_out_T, KokkosFFT::Direction::forward,
              to_array<int, std::size_t, 1>(block.m_axes));
          m_ifft_plan0 = std::make_unique<FFTBackwardPlanType0>(
              m_exec_space, m_out_T, m_in_T, KokkosFFT::Direction::backward,
              to_array<int, std::size_t, 1>(block.m_axes));
        } else {
          m_fft2_plan0 = std::make_unique<FFT2ForwardPlanType0>(
              m_exec_space, m_in_T, m_out_T, KokkosFFT::Direction::forward,
              to_array<int, std::size_t, 2>(block.m_axes));
          m_ifft2_plan0 = std::make_unique<FFT2BackwardPlanType0>(
              m_exec_space, m_out_T, m_in_T, KokkosFFT::Direction::backward,
              to_array<int, std::size_t, 2>(block.m_axes));
        }
      } else {
        m_out_T2 = OutViewType(
            m_send_buffer_allocation.data(),
            KokkosFFT::Impl::create_layout<LayoutType>(block.m_out_extents));
        m_fft_plan1 = std::make_unique<FFTForwardPlanType1>(
            m_exec_space, m_out_T2, m_out_T2, KokkosFFT::Direction::forward,
            to_array<int, std::size_t, 1>(block.m_axes));
        m_ifft_plan1 = std::make_unique<FFTBackwardPlanType1>(
            m_exec_space, m_out_T2, m_out_T2, KokkosFFT::Direction::backward,
            to_array<int, std::size_t, 1>(block.m_axes));
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
          block.m_out_map, block.m_out_axis, m_comm));
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
            in, m_in_T, m_send_buffer_allocation, m_recv_buffer_allocation,
            KokkosFFT::Direction::forward);
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
          out_view  = first_FFT_dim == 1 ? m_out_T2 : m_out_T;
          out_view2 = out;
        }

        AllocationViewType send_buffer = m_send_buffer_allocation,
                           recv_buffer = m_recv_buffer_allocation;

        if (KokkosFFT::Impl::are_aliasing(out_view.data(),
                                          send_buffer.data())) {
          send_buffer = m_recv_buffer_allocation;
          recv_buffer = m_send_buffer_allocation;
        }

        (*m_trans_blocks.at(block.m_block_idx))(out_view, out_view2,
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
            m_in_T, in, m_recv_buffer_allocation, m_send_buffer_allocation,
            KokkosFFT::Direction::backward);
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
          out_view  = first_FFT_dim == 1 ? m_out_T2 : m_out_T;
          out_view2 = out;
        }

        AllocationViewType send_buffer = m_send_buffer_allocation,
                           recv_buffer = m_recv_buffer_allocation;

        if (KokkosFFT::Impl::are_aliasing(out_view.data(),
                                          recv_buffer.data())) {
          send_buffer = m_recv_buffer_allocation;
          recv_buffer = m_send_buffer_allocation;
        }

        (*m_trans_blocks.at(block.m_block_idx))(out_view2, out_view,
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
  using TransBlockType = TransBlock<ExecutionSpace, DIM>;

  // First FFT to perform can be Real to Complex
  using FFTForwardPlanType0 =
      KokkosFFT::Plan<ExecutionSpace, InViewType, OutViewType, 1>;
  using FFT2ForwardPlanType0 =
      KokkosFFT::Plan<ExecutionSpace, InViewType, OutViewType, 2>;
  using FFT3ForwardPlanType0 =
      KokkosFFT::Plan<ExecutionSpace, InViewType, OutViewType, 3>;
  using FFTBackwardPlanType0 =
      KokkosFFT::Plan<ExecutionSpace, OutViewType, InViewType, 1>;
  using FFT2BackwardPlanType0 =
      KokkosFFT::Plan<ExecutionSpace, OutViewType, InViewType, 2>;
  using FFT3BackwardPlanType0 =
      KokkosFFT::Plan<ExecutionSpace, OutViewType, InViewType, 3>;

  // Second FFT to perform must be Complex to Complex
  using FFTForwardPlanType1 =
      KokkosFFT::Plan<ExecutionSpace, OutViewType, OutViewType, 1>;
  using FFT2ForwardPlanType1 =
      KokkosFFT::Plan<ExecutionSpace, OutViewType, OutViewType, 2>;
  using FFTBackwardPlanType1 =
      KokkosFFT::Plan<ExecutionSpace, OutViewType, OutViewType, 1>;
  using FFT2BackwardPlanType1 =
      KokkosFFT::Plan<ExecutionSpace, OutViewType, OutViewType, 2>;

  execSpace m_exec_space;
  axes_type m_axes;
  topology_type m_in_topology;
  topology_type m_out_topology;
  MPI_Comm m_comm;
  KokkosFFT::Normalization m_normalization;

  // Analyse topology
  std::unique_ptr<SlabBlockAnalysesType> m_block_analyses;
  std::size_t first_FFT_dim = 1;

  // Buffer view types
  InViewType m_in_T;
  OutViewType m_out_T, m_out_T2;

  // Buffer Allocations
  AllocationViewType m_send_buffer_allocation, m_recv_buffer_allocation;

  // Internal transpose blocks
  std::vector<std::unique_ptr<TransBlockType>> m_trans_blocks;

  // Internal FFT plans
  std::unique_ptr<FFTForwardPlanType0> m_fft_plan0;
  std::unique_ptr<FFTForwardPlanType1> m_fft_plan1;
  std::unique_ptr<FFT2ForwardPlanType0> m_fft2_plan0;
  std::unique_ptr<FFT2ForwardPlanType1> m_fft2_plan1;
  std::unique_ptr<FFT3ForwardPlanType0> m_fft3_plan0;
  std::unique_ptr<FFTBackwardPlanType0> m_ifft_plan0;
  std::unique_ptr<FFTBackwardPlanType1> m_ifft_plan1;
  std::unique_ptr<FFT2BackwardPlanType0> m_ifft2_plan0;
  std::unique_ptr<FFT2BackwardPlanType1> m_ifft2_plan1;
  std::unique_ptr<FFT3BackwardPlanType0> m_ifft3_plan0;

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
      if (first_FFT_dim == 1) {
        if (m_map_forward_in == int_map_type{}) {
          KokkosFFT::execute(*m_fft_plan0, in, out, m_normalization);
        } else {
          safe_transpose(m_exec_space, in, m_in_T, m_map_forward_in);
          KokkosFFT::execute(*m_fft_plan0, m_in_T, out, m_normalization);
        }
      } else if (first_FFT_dim == 2) {
        if (m_map_forward_in == int_map_type{}) {
          KokkosFFT::execute(*m_fft2_plan0, in, out, m_normalization);
        } else {
          safe_transpose(m_exec_space, in, m_in_T, m_map_forward_in);
          KokkosFFT::execute(*m_fft2_plan0, m_in_T, out, m_normalization);
        }
      } else if (first_FFT_dim == 3) {
        if (m_map_forward_in == int_map_type{} &&
            m_map_forward_out == int_map_type{}) {
          KokkosFFT::execute(*m_fft3_plan0, in, out, m_normalization);
        } else if (m_map_forward_in == int_map_type{} &&
                   m_map_forward_out != int_map_type{}) {
          KokkosFFT::execute(*m_fft3_plan0, m_in_T, m_out_T, m_normalization);
          safe_transpose(m_exec_space, m_out_T, out, m_map_forward_out);
        } else if (m_map_forward_in != int_map_type{} &&
                   m_map_forward_out == int_map_type{}) {
          safe_transpose(m_exec_space, in, m_in_T, m_map_forward_in);
          KokkosFFT::execute(*m_fft3_plan0, m_in_T, out, m_normalization);
        } else {
          safe_transpose(m_exec_space, in, m_in_T, m_map_forward_in);
          KokkosFFT::execute(*m_fft3_plan0, m_in_T, m_out_T, m_normalization);
          safe_transpose(m_exec_space, m_out_T, out, m_map_forward_out);
        }
      }
    } else if constexpr (STEP == 1) {
      if (first_FFT_dim == 1) {
        if (m_map_forward_out == int_map_type{}) {
          KokkosFFT::execute(*m_fft2_plan1, in, out, m_normalization);
        } else {
          KokkosFFT::execute(*m_fft2_plan1, in, m_out_T2, m_normalization);
          safe_transpose(m_exec_space, m_out_T2, out, m_map_forward_out);
        }
      } else if (first_FFT_dim == 2) {
        if (m_map_forward_out == int_map_type{}) {
          KokkosFFT::execute(*m_fft_plan1, in, out, m_normalization);
        } else {
          KokkosFFT::execute(*m_fft_plan1, in, m_out_T2, m_normalization);
          safe_transpose(m_exec_space, m_out_T2, out, m_map_forward_out);
        }
      }
    }
  }

  template <std::size_t STEP, typename InType, typename OutType>
  void backward_fft(const OutType& out, const InType& in) const {
    if constexpr (STEP == 0) {
      if (first_FFT_dim == 1) {
        if (m_map_backward_out == int_map_type{}) {
          KokkosFFT::execute(*m_ifft_plan0, out, in, m_normalization);
        } else {
          KokkosFFT::execute(*m_ifft_plan0, out, m_in_T, m_normalization);
          safe_transpose(m_exec_space, m_in_T, in, m_map_backward_out);
        }
      } else if (first_FFT_dim == 2) {
        if (m_map_backward_out == int_map_type{}) {
          KokkosFFT::execute(*m_ifft2_plan0, out, in, m_normalization);
        } else {
          KokkosFFT::execute(*m_ifft2_plan0, out, m_in_T, m_normalization);
          safe_transpose(m_exec_space, m_in_T, in, m_map_backward_out);
        }
      } else if (first_FFT_dim == 3) {
        if (m_map_backward_in == int_map_type{} &&
            m_map_backward_out == int_map_type{}) {
          KokkosFFT::execute(*m_ifft3_plan0, out, in, m_normalization);
        } else if (m_map_backward_in == int_map_type{} &&
                   m_map_backward_out != int_map_type{}) {
          KokkosFFT::execute(*m_ifft3_plan0, out, m_in_T, m_normalization);
          safe_transpose(m_exec_space, m_in_T, in, m_map_backward_out);
        } else if (m_map_backward_in != int_map_type{} &&
                   m_map_backward_out == int_map_type{}) {
          safe_transpose(m_exec_space, out, m_out_T, m_map_backward_in);
          KokkosFFT::execute(*m_ifft3_plan0, m_out_T, in, m_normalization);
        } else if (m_map_backward_in != int_map_type{} &&
                   m_map_backward_out != int_map_type{}) {
          safe_transpose(m_exec_space, out, m_out_T, m_map_backward_in);
          KokkosFFT::execute(*m_ifft3_plan0, m_out_T, m_in_T, m_normalization);
          safe_transpose(m_exec_space, m_in_T, in, m_map_backward_out);
        }
      }
    } else if constexpr (STEP == 1) {
      if (first_FFT_dim == 1) {
        if (m_map_backward_in == int_map_type{}) {
          KokkosFFT::execute(*m_ifft2_plan1, out, in, m_normalization);
        } else {
          safe_transpose(m_exec_space, out, m_out_T2, m_map_backward_in);
          KokkosFFT::execute(*m_ifft2_plan1, m_out_T2, in, m_normalization);
        }
      } else {
        if (m_map_backward_in == int_map_type{}) {
          KokkosFFT::execute(*m_ifft_plan1, out, in, m_normalization);
        } else {
          safe_transpose(m_exec_space, out, m_out_T2, m_map_backward_in);
          KokkosFFT::execute(*m_ifft_plan1, m_out_T2, in, m_normalization);
        }
      }
    }
  }

  void set_block_impl(const std::size_t block_idx) {
    auto src_map    = KokkosFFT::Impl::index_sequence<std::size_t, DIM, 0>();
    auto block      = m_block_analyses->m_block_infos.at(block_idx);
    auto block_type = block.m_block_type;

    if (block_type == BlockType::FFT) {
      if (block.m_block_idx == 0) {
        first_FFT_dim = block.m_axes.size();
        m_in_T        = InViewType(
            reinterpret_cast<in_value_type*>(m_send_buffer_allocation.data()),
            KokkosFFT::Impl::create_layout<LayoutType>(block.m_in_extents));
        m_out_T = OutViewType(
            m_recv_buffer_allocation.data(),
            KokkosFFT::Impl::create_layout<LayoutType>(block.m_out_extents));

        if (block.m_axes.size() == 1) {
          m_fft_plan0 = std::make_unique<FFTForwardPlanType0>(
              m_exec_space, m_in_T, m_out_T, KokkosFFT::Direction::forward,
              to_array<int, std::size_t, 1>(block.m_axes));
          m_ifft_plan0 = std::make_unique<FFTBackwardPlanType0>(
              m_exec_space, m_out_T, m_in_T, KokkosFFT::Direction::backward,
              to_array<int, std::size_t, 1>(block.m_axes));
        } else if (block.m_axes.size() == 2) {
          m_fft2_plan0 = std::make_unique<FFT2ForwardPlanType0>(
              m_exec_space, m_in_T, m_out_T, KokkosFFT::Direction::forward,
              to_array<int, std::size_t, 2>(block.m_axes));
          m_ifft2_plan0 = std::make_unique<FFT2BackwardPlanType0>(
              m_exec_space, m_out_T, m_in_T, KokkosFFT::Direction::backward,
              to_array<int, std::size_t, 2>(block.m_axes));
        } else {
          m_fft3_plan0 = std::make_unique<FFT3ForwardPlanType0>(
              m_exec_space, m_in_T, m_out_T, KokkosFFT::Direction::forward,
              to_array<int, std::size_t, 3>(block.m_axes));
          m_ifft3_plan0 = std::make_unique<FFT3BackwardPlanType0>(
              m_exec_space, m_out_T, m_in_T, KokkosFFT::Direction::backward,
              to_array<int, std::size_t, 3>(block.m_axes));
        }
      } else {
        m_out_T2 = OutViewType(
            m_send_buffer_allocation.data(),
            KokkosFFT::Impl::create_layout<LayoutType>(block.m_out_extents));
        if (block.m_axes.size() == 1) {
          m_fft_plan1 = std::make_unique<FFTForwardPlanType1>(
              m_exec_space, m_out_T2, m_out_T2, KokkosFFT::Direction::forward,
              to_array<int, std::size_t, 1>(block.m_axes));
          m_ifft_plan1 = std::make_unique<FFTBackwardPlanType1>(
              m_exec_space, m_out_T2, m_out_T2, KokkosFFT::Direction::backward,
              to_array<int, std::size_t, 1>(block.m_axes));
        } else {
          m_fft2_plan1 = std::make_unique<FFT2ForwardPlanType1>(
              m_exec_space, m_out_T2, m_out_T2, KokkosFFT::Direction::forward,
              to_array<int, std::size_t, 2>(block.m_axes));
          m_ifft2_plan1 = std::make_unique<FFT2BackwardPlanType1>(
              m_exec_space, m_out_T2, m_out_T2, KokkosFFT::Direction::backward,
              to_array<int, std::size_t, 2>(block.m_axes));
        }
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
          block.m_out_map, block.m_out_axis, m_comm));
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
            in, m_in_T, m_send_buffer_allocation, m_recv_buffer_allocation,
            KokkosFFT::Direction::forward);
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
          out_view  = first_FFT_dim == 3 ? m_out_T : m_out_T2;
          out_view2 = out;
        }

        AllocationViewType send_buffer = m_send_buffer_allocation,
                           recv_buffer = m_recv_buffer_allocation;

        if (KokkosFFT::Impl::are_aliasing(out_view.data(),
                                          send_buffer.data())) {
          send_buffer = m_recv_buffer_allocation;
          recv_buffer = m_send_buffer_allocation;
        }

        (*m_trans_blocks.at(block.m_block_idx))(out_view, out_view2,
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
            m_in_T, in, m_recv_buffer_allocation, m_send_buffer_allocation,
            KokkosFFT::Direction::backward);
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
          out_view  = first_FFT_dim == 3 ? m_out_T : m_out_T2;
          out_view2 = out;
        }

        AllocationViewType send_buffer = m_send_buffer_allocation,
                           recv_buffer = m_recv_buffer_allocation;

        if (KokkosFFT::Impl::are_aliasing(out_view.data(),
                                          recv_buffer.data())) {
          send_buffer = m_recv_buffer_allocation;
          recv_buffer = m_send_buffer_allocation;
        }

        (*m_trans_blocks.at(block.m_block_idx))(out_view2, out_view,
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

  InternalPlanType m_internal_plan;
  extents_type m_in_extents, m_out_extents;

  using InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM, InLayoutType,
                     OutLayoutType>::good;

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
        m_internal_plan(exec_space, in, out, axes, in_topology.array(),
                        out_topology.array(), comm, norm),
        m_in_extents(KokkosFFT::Impl::extract_extents(in)),
        m_out_extents(KokkosFFT::Impl::extract_extents(out)) {
    auto in_size  = get_size(in_topology);
    auto out_size = get_size(out_topology);

    KOKKOSFFT_THROW_IF(in_size != out_size,
                       "Input and output topologies must have the same size.");

    bool is_slab = is_slab_topology(in_topology.array()) &&
                   is_slab_topology(out_topology.array());
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

  std::string label() const override { return std::string("SlabPlan"); }
};

#endif
