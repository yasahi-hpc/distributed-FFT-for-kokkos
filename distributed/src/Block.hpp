#ifndef BLOCK_HPP
#define BLOCK_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
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
  extents_type m_in_extents, m_out_extents;

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
        m_comm(comm),
        m_in_extents(KokkosFFT::Impl::extract_extents(in)),
        m_out_extents(KokkosFFT::Impl::extract_extents(out)) {
    auto send_buffer_size = m_send_buffer.size();
    auto recv_buffer_size = m_recv_buffer.size();
    KOKKOSFFT_THROW_IF(send_buffer_size != recv_buffer_size,
                       "Send and receive buffers must be the same size.");

    auto in_size  = m_in.size();
    auto out_size = m_out.size();
    KOKKOSFFT_THROW_IF(
        in_size > send_buffer_size || out_size > send_buffer_size,
        "Input and output views must be smaller than the send and receive "
        "buffers.");
  }

  void operator()(const InViewType& in, const OutViewType& out) const {
    good(in, out);

    pack(m_exec, in, m_send_buffer, m_src_map, m_src_axis);
    m_exec.fence();
    All2All<ExecutionSpace, BufferType>(m_send_buffer, m_recv_buffer, m_comm,
                                        m_exec)(m_send_buffer, m_recv_buffer);
    unpack(m_exec, m_recv_buffer, out, m_dst_map, m_dst_axis);
  }

 private:
  void good(const InViewType& in, const OutViewType& out) const {
    auto in_extents  = KokkosFFT::Impl::extract_extents(in);
    auto out_extents = KokkosFFT::Impl::extract_extents(out);

    auto mismatched_extents = [&](extents_type extents,
                                  extents_type plan_extents) -> std::string {
      std::string message;
      message += "View (";
      message += std::to_string(extents.at(0));
      for (std::size_t r = 1; r < extents.size(); r++) {
        message += ",";
        message += std::to_string(extents.at(r));
      }
      message += "), ";
      message += "Plan (";
      message += std::to_string(plan_extents.at(0));
      for (std::size_t r = 1; r < plan_extents.size(); r++) {
        message += ",";
        message += std::to_string(plan_extents.at(r));
      }
      message += ")";
      return message;
    };

    KOKKOSFFT_THROW_IF(in_extents != m_in_extents,
                       "extents of input View for plan and "
                       "execution are not identical: " +
                           mismatched_extents(in_extents, m_in_extents));

    KOKKOSFFT_THROW_IF(out_extents != m_out_extents,
                       "extents of output View for plan and "
                       "execution are not identical: " +
                           mismatched_extents(out_extents, m_out_extents));

    // Check in and send_buffer are not aliasing
    KOKKOSFFT_THROW_IF(
        KokkosFFT::Impl::are_aliasing(in.data(), m_send_buffer.data()),
        "input: " + in.label() + " and send_buffer: " + m_send_buffer.label() +
            " must not be aliasing");

    // Check out and recv_buffer are not aliasing
    KOKKOSFFT_THROW_IF(
        KokkosFFT::Impl::are_aliasing(out.data(), m_recv_buffer.data()),
        "output: " + out.label() + " and recv_buffer: " +
            m_recv_buffer.label() + " must not be aliasing");
  }
};

template <typename ExecutionSpace, std::size_t DIM>
class TransBlock {
  using execSpace           = ExecutionSpace;
  using buffer_extents_type = KokkosFFT::shape_type<DIM + 1>;
  using extents_type        = KokkosFFT::shape_type<DIM>;

  static constexpr std::size_t Rank = DIM;

  ExecutionSpace m_exec;
  buffer_extents_type m_buffer_extents;
  extents_type m_src_map, m_dst_map;
  std::size_t m_src_axis, m_dst_axis;
  MPI_Comm m_comm;

 public:
  explicit TransBlock(const ExecutionSpace& exec_space,
                      const buffer_extents_type& buffer_extents,
                      const extents_type& src_map, std::size_t src_axis,
                      const extents_type& dst_map, std::size_t dst_axis,
                      MPI_Comm comm = MPI_COMM_WORLD)
      : m_exec(exec_space),
        m_buffer_extents(buffer_extents),
        m_src_map(src_map),
        m_dst_map(dst_map),
        m_src_axis(src_axis),
        m_dst_axis(dst_axis),
        m_comm(comm) {}

  template <typename ViewType, typename BufferType>
  void operator()(const ViewType& in, const ViewType& out,
                  const BufferType& send, const BufferType& recv,
                  KokkosFFT::Direction direction) const {
    using value_type = typename ViewType::non_const_value_type;
    using LayoutType = typename ViewType::array_layout;
    using buffer_data_type =
        KokkosFFT::Impl::add_pointer_n_t<value_type, DIM + 1>;
    using buffer_view_type = Kokkos::View<buffer_data_type, LayoutType,
                                          typename ViewType::execution_space>;

    Kokkos::Profiling::ScopedRegion region("TransBlock");
    // Making unmanaged views from meta data
    buffer_view_type send_buffer(
        reinterpret_cast<value_type*>(send.data()),
        KokkosFFT::Impl::create_layout<LayoutType>(m_buffer_extents)),
        recv_buffer(
            reinterpret_cast<value_type*>(recv.data()),
            KokkosFFT::Impl::create_layout<LayoutType>(m_buffer_extents));

    good(in, out, send_buffer, recv_buffer);

    bool is_forward = direction == KokkosFFT::Direction::forward;
    auto src_map    = is_forward ? m_src_map : m_dst_map;
    auto dst_map    = is_forward ? m_dst_map : m_src_map;
    auto src_axis   = is_forward ? m_src_axis : m_dst_axis;
    auto dst_axis   = is_forward ? m_dst_axis : m_src_axis;

    pack(m_exec, in, send_buffer, src_map, src_axis);
    m_exec.fence();
    All2All(send_buffer, recv_buffer, m_comm, m_exec)(send_buffer, recv_buffer);
    unpack(m_exec, recv_buffer, out, dst_map, dst_axis);
  }

 private:
  template <typename ViewType, typename BufferType>
  void good(const ViewType& in, const ViewType& out, const BufferType& send,
            const BufferType& recv) const {
    // static_assert(BufferType::rank() == ViewType::rank() + 1);

    using view_value_type   = typename ViewType::non_const_value_type;
    using buffer_value_type = typename BufferType::non_const_value_type;

    static_assert(std::is_same_v<buffer_value_type, view_value_type>);

    auto buffer_size = send.size();
    // auto recv_buffer_size = recv.size();
    // KOKKOSFFT_THROW_IF(send_buffer_size != recv_buffer_size,
    //                    "Send and receive buffers must be the same size.");

    auto in_size  = in.size();
    auto out_size = out.size();
    KOKKOSFFT_THROW_IF(
        in_size > buffer_size || out_size > buffer_size,
        "Input and output views must be smaller than the send and receive "
        "buffers.");

    // Check in and send_buffer are not aliasing
    KOKKOSFFT_THROW_IF(KokkosFFT::Impl::are_aliasing(in.data(), send.data()),
                       "input: " + in.label() + " and send_buffer: " +
                           send.label() + " must not be aliasing");

    // Check out and recv_buffer are not aliasing
    KOKKOSFFT_THROW_IF(KokkosFFT::Impl::are_aliasing(out.data(), recv.data()),
                       "output: " + out.label() + " and recv_buffer: " +
                           recv.label() + " must not be aliasing");
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
                src_axis, dst_map, dst_axis, comm) {}

  void operator()(const InViewType& in, const OutViewType& out) const {
    KokkosFFT::execute(m_plan, in, m_fft_out);
    m_block(m_fft_out, out);
  }
};

#endif
