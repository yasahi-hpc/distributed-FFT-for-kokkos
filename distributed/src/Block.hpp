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

  void operator()() const {
    m_block();
    KokkosFFT::execute(m_plan, m_fft_in, m_out);
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

  void operator()() const {
    KokkosFFT::execute(m_plan, m_in, m_fft_out);
    m_block();
  }
};

/*
template <typename ExecutionSpace, typename InViewType, typename FFTInViewType,
          typename OutViewType, typename BufferType, std::size_t DIM = 1>
class FFTBlock {
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
  KokkosFFT::Direction m_direction;
  BlockType m_block;

 public:
  explicit FFTBlock(const ExecutionSpace& exec_space, const InViewType& in,
                    const FFTInViewType& fft_in, const OutViewType& out,
                    const BufferType& send_buffer, const BufferType&
recv_buffer, const extents_type& src_map, const std::size_t src_axis, const
extents_type& dst_map, const std::size_t dst_axis, KokkosFFT::Direction
direction, MPI_Comm comm = MPI_COMM_WORLD) : m_in(in), m_fft_out(fft_),
        m_plan(exec_space, fft_in, out, direction,
               KokkosFFT::Impl::get_index(dst_map, dst_axis)),
        m_direction(direction),
        m_block(exec_space, in, fft_in, send_buffer, recv_buffer, src_map,
                src_axis, dst_map, src_axis, comm) {}

  void operator()() const {
    m_block();
    KokkosFFT::execute(m_plan, m_fft_out, m_out);
  }
};
*/

/*
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          typename BufferType, std::size_t DIM = 1>
class FFTBlock {
  using execSpace    = ExecutionSpace;
  using LayoutType   = typename OutViewType::array_layout;
  using extents_type = KokkosFFT::shape_type<InViewType::rank()>;
  using axes_type    = KokkosFFT::axis_type<DIM>;
  using out_value_type = typename OutViewType::non_const_value_type;
  using buffer_value_type = KokkosFFT::Impl::add_pointer_n_t<out_value_type,
OutViewType::rank()+1>;

  using BufferType = Kokkos::View<buffer_value_type, LayoutType,
ExecutionSpace>; using BlockType  = Block<ExecutionSpace, InViewType,
OutViewType, BufferType, DIM>;

  InViewType m_in;
  OutViewType m_out;
  BufferType m_send_buffer, m_recv_buffer;
  MPI_Comm m_comm;
  ExecutionSpace m_exec;
  extents_type m_in_topology, m_out_topology;
  BlockType m_block;

 public:
  explicit FFTBlock(const ExecutionSpace& exec_space, const InViewType& in,
                    const OutViewType& out, const extents_type& fft_extents,
                    const extents_type& in_topology, const extents_type&
out_topology, MPI_Comm comm = MPI_COMM_WORLD) : m_exec(exec_space), m_in(in),
        m_out(out),
        m_in_topology(in_topology),
        m_out_topology(out_topology),
        m_send_buffer("send_buffer",
KokkosFFT::Impl::create_layout<LayoutType>(get_buffer_extents(out))),
        m_send_buffer("recv_buffer",
KokkosFFT::Impl::create_layout<LayoutType>(get_buffer_extents(out))),
        m_block(exec_space, in, out, m_send_buffer, m_recv_buffer,
                get_src_map(m_in_topology, m_in_topology),
get_src_axis(m_in_topology, m_in_topology), get_dst_map(m_in_topology,
m_in_topology), get_dst_axis(m_in_topology, m_in_topology), comm); {

  }

  void operator()() const {

  }
};
*/

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
