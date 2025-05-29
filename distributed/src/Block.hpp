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

  InViewType m_in;
  OutViewType m_out;
  BufferType m_send_buffer, m_recv_buffer;
  MPI_Comm m_comm;
  ExecutionSpace m_exec;
  extents_type m_src_map, m_dst_map;
  std::size_t m_src_axis, m_dst_axis;

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
        m_src_axis(src_axis),
        m_dst_map(dst_map),
        m_dst_axis(dst_axis),
        m_comm(comm) {}

  void operator()() const {
    pack(m_exec, m_in, m_send_buffer, m_src_map, m_src_axis);
    m_exec.fence();
    All2All<ExecutionSpace, BufferType>(m_send_buffer, m_recv_buffer, m_comm,
                                        m_exec)();
    unpack(m_exec, m_recv_buffer, m_out, m_dst_map, m_dst_axis);
  }
};

/*
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
typename BufferType, std::size_t DIM=1> class Block { using execSpace =
ExecutionSpace; using LayoutType = typename InViewType::array_layout; using
out_value_type = typename OutViewType::non_const_value_type;

  using InplaceInViewType = typename
KokkosFFT::Impl::ConvertedViewType<InViewType, out_value_type>::type;

  using PlanType = KokkosFFT::Plan<ExecutionSpace, InViewType,
InplaceInViewType, DIM>;

  using extents_type = KokkosFFT::shape_type<InViewType::rank()>;
  using axes_type    = KokkosFFT::axis_type<DIM>;

  InViewType m_in;
  OutViewType m_out;
  BufferType m_send_buffer, m_recv_buffer;
  MPI_Comm m_comm;
  ExecutionSpace m_exec;

  public:
  explicit Block(const ExecutionSpace& exec_space, const InViewType& in,
                 const OutViewType& out, KokkosFFT::Direction direction,
                 const extents_type& fft_extents, const axes_type& fft_axes,
                 const extents_type& src_map, int src_axis, const extents_type&
dst_map, int dst_axis, MPI_Comm comm = MPI_COMM_WORLD) {}

  void operator()(const InViewType& in, const OutViewType& out,
KokkosFFT::Normalization norm) const {
    //InplaceInViewType in_inplace(reinterpret_cast<out_value_type
*>(in.data()), create_layout<LayoutType>(m_fft_extents));
    //execute(m_plan, in, in_inplace, norm);
    //pack(m_exec, in_inplace, m_send_buffer, m_src_map, m_src_axis);
    //m_exec.fence();
    //All2All<ExecutionSpace, BufferType>(m_send_buffer, m_recv_buffer, m_comm,
m_exec)();
    //unpack(m_exec, m_recv_buffer, out, m_dst_map, m_dst_axis);
  }
};
*/

#endif
