#ifndef KOKKOSFFT_DISTRIBUTED_BLOCK_HPP
#define KOKKOSFFT_DISTRIBUTED_BLOCK_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_PackUnpack.hpp"
#include "KokkosFFT_Distributed_All2All.hpp"

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

#endif
