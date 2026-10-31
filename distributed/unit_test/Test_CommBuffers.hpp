#ifndef TEST_COMM_BUFFERS_HPP
#define TEST_COMM_BUFFERS_HPP

#include <cstddef>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_All2All.hpp"
#include "KokkosFFT_Distributed_CommBuffer.hpp"

/// \brief Send and receive Views on the buffers that the transposes of a plan
/// allocate for the communicator wrapper CommType (CommBuffer). The all2all
/// of the wrappers is tested on them, as a plan calls it. Not copyable, not
/// movable.
/// \tparam ViewType Type of the send and receive Views
/// \tparam CommType Type of the communicator wrapper
template <typename ViewType, typename CommType>
struct CommBuffers {
  using value_type = typename ViewType::non_const_value_type;
  using BufferType = KokkosFFT::Distributed::Impl::CommBuffer<
      CommType, typename ViewType::execution_space, value_type>;

  BufferType m_send_buffer, m_recv_buffer;
  ViewType m_send, m_recv;

  explicit CommBuffers(const typename ViewType::array_layout& layout)
      : m_send_buffer("send_buffer", size(layout), true),
        m_recv_buffer("recv_buffer", size(layout), true),
        m_send(m_send_buffer.data(), layout),
        m_recv(m_recv_buffer.data(), layout) {}

  static std::size_t size(const typename ViewType::array_layout& layout) {
    std::size_t n = 1;
    for (std::size_t r = 0; r < ViewType::rank(); ++r) n *= layout.dimension[r];
    return n;
  }
};

/// \brief Make the send and receive Views on CommBuffers for a communicator
/// wrapper
/// \tparam ViewType Type of the send and receive Views
/// \tparam CommType Type of the communicator wrapper
/// \param[in] comm The communicator wrapper (selects the allocation)
/// \param[in] layout The layout of the send and receive Views
template <typename ViewType, typename CommType>
CommBuffers<ViewType, CommType> make_comm_buffers(
    const CommType& /*comm*/, const typename ViewType::array_layout& layout) {
  return CommBuffers<ViewType, CommType>(layout);
}

/// \brief Make the send and receive Views on CommBuffers for a communicator
/// wrapper, with the layout of view
/// \tparam CommType Type of the communicator wrapper
/// \tparam ViewType Type of the send and receive Views
/// \param[in] comm The communicator wrapper (selects the allocation)
/// \param[in] view A View with the extents of the buffers
template <typename CommType, typename ViewType>
CommBuffers<ViewType, CommType> make_comm_buffers(const CommType& comm,
                                                  const ViewType& view) {
  return make_comm_buffers<ViewType>(comm, view.layout());
}

#endif
