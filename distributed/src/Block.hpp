#ifndef BLOCK_HPP
#define BLOCK_HPP

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "PackUnpack.hpp"
#include "All2All.hpp"

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          typename BufferType, std::size_t DIM = 1>
class Block {
  using execSpace    = ExecutionSpace;
  using LayoutType   = typename InViewType::array_layout;
  using extents_type = KokkosFFT::shape_type<InViewType::rank()>;
  using axes_type    = KokkosFFT::axis_type<DIM>;

  ExecutionSpace m_exec;
  InViewType m_in;
  OutViewType m_out;
  BufferType m_send_buffer, m_recv_buffer;
  extents_type m_src_map, m_dst_map;
  std::size_t m_src_axis, m_dst_axis;
  MPI_Comm m_comm;

 public:
  explicit Block(const ExecutionSpace& exec_space, const InViewType& in,
                 const OutViewType& out, const BufferType& send_buffer,
                 const BufferType& recv_buffer, const extents_type& src_map,
                 std::size_t src_axis, const extents_type& dst_map,
                 std::size_t dst_axis, MPI_Comm comm = MPI_COMM_WORLD)
      : m_exec(exec_space),
        m_in(in),
        m_out(out),
        m_send_buffer(send_buffer),
        m_recv_buffer(recv_buffer),
        m_src_map(src_map),
        m_dst_map(dst_map),
        m_src_axis(src_axis),
        m_dst_axis(dst_axis),
        m_comm(comm) {}

  void operator()(const InViewType& in, const OutViewType& out) const {
    pack(m_exec, in, m_send_buffer, m_src_map, m_src_axis);
    m_exec.fence();
    All2All<ExecutionSpace, BufferType>(m_send_buffer, m_recv_buffer, m_comm,
                                        m_exec)(m_send_buffer, m_recv_buffer);
    unpack(m_exec, m_recv_buffer, out, m_dst_map, m_dst_axis);
  }
};

template <typename ExecutionSpace, typename InViewType, typename FFTInViewType,
          typename OutViewType, typename BufferType, std::size_t DIM = 1>
class FFTForwardBlock {
  using execSpace    = ExecutionSpace;
  using LayoutType   = typename OutViewType::array_layout;
  using extents_type = KokkosFFT::shape_type<InViewType::rank()>;
  using axes_type    = KokkosFFT::axis_type<DIM>;
  using FFTPlanType =
      KokkosFFT::Plan<ExecutionSpace, FFTInViewType, OutViewType, DIM>;
  using BlockType =
      Block<ExecutionSpace, InViewType, FFTInViewType, BufferType, DIM>;

  InViewType m_in;
  FFTInViewType m_fft_in;
  OutViewType m_out;
  ExecutionSpace m_exec;
  FFTPlanType m_plan;
  BlockType m_block;

 public:
  explicit FFTForwardBlock(
      const ExecutionSpace& exec_space, const InViewType& in,
      const FFTInViewType& fft_in, const OutViewType& out,
      const BufferType& send_buffer, const BufferType& recv_buffer,
      const extents_type& src_map, const std::size_t src_axis,
      const extents_type& dst_map, const std::size_t dst_axis,
      MPI_Comm comm = MPI_COMM_WORLD)
      : m_fft_in(fft_in),
        m_out(out),
        m_plan(exec_space, fft_in, out, KokkosFFT::Direction::forward,
               KokkosFFT::Impl::get_index(dst_map, dst_axis)),
        m_block(exec_space, in, fft_in, send_buffer, recv_buffer, src_map,
                src_axis, dst_map, dst_axis, comm) {}

  void operator()(const InViewType& in, const OutViewType& out) const {
    m_block(in, m_fft_in);
    KokkosFFT::execute(m_plan, m_fft_in, out);
  }
};

template <typename ExecutionSpace, typename InViewType, typename FFTOutViewType,
          typename OutViewType, typename BufferType, std::size_t DIM = 1>
class FFTBackwardBlock {
  using execSpace    = ExecutionSpace;
  using LayoutType   = typename OutViewType::array_layout;
  using extents_type = KokkosFFT::shape_type<InViewType::rank()>;
  using axes_type    = KokkosFFT::axis_type<DIM>;
  using FFTPlanType =
      KokkosFFT::Plan<ExecutionSpace, InViewType, FFTOutViewType, DIM>;
  using BlockType =
      Block<ExecutionSpace, FFTOutViewType, OutViewType, BufferType, DIM>;

  InViewType m_in;
  FFTOutViewType m_fft_out;
  OutViewType m_out;
  ExecutionSpace m_exec;
  FFTPlanType m_plan;
  BlockType m_block;

 public:
  explicit FFTBackwardBlock(
      const ExecutionSpace& exec_space, const InViewType& in,
      const FFTOutViewType& fft_out, const OutViewType& out,
      const BufferType& send_buffer, const BufferType& recv_buffer,
      const extents_type& src_map, const std::size_t src_axis,
      const extents_type& dst_map, const std::size_t dst_axis,
      MPI_Comm comm = MPI_COMM_WORLD)
      : m_in(in),
        m_fft_out(fft_out),
        m_plan(exec_space, in, fft_out, KokkosFFT::Direction::backward,
               KokkosFFT::Impl::get_index(src_map, src_axis)),
        m_block(exec_space, fft_out, out, send_buffer, recv_buffer, src_map,
                src_axis, dst_map, src_axis, comm) {}

  void operator()(const InViewType& in, const OutViewType& out) const {
    KokkosFFT::execute(m_plan, in, m_fft_out);
    m_block(m_fft_out, out);
  }
};

#endif
