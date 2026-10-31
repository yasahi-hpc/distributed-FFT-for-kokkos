#ifndef KOKKOSFFT_DISTRIBUTED_COMM_BUFFER_HPP
#define KOKKOSFFT_DISTRIBUTED_COMM_BUFFER_HPP

#include <cstddef>
#include <string>
#include <Kokkos_Core.hpp>

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief A send or receive buffer of the transposes. The communicator
/// wrapper type chooses how it is allocated. By default, it is a managed
/// Kokkos View. A backend that needs another allocation (e.g. on a symmetric
/// heap) specializes this class for its wrapper type.
///
/// It is neither copyable nor movable, so that the code using it also works
/// with allocations whose address is shared with other ranks.
///
/// \tparam CommType Type of the communicator wrapper (TplComm)
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam T Value type of the buffer
template <typename CommType, typename ExecutionSpace, typename T>
class CommBuffer {
  using AllocationViewType = Kokkos::View<T*, ExecutionSpace>;

  AllocationViewType m_allocation;

 public:
  /// \brief Allocate the buffer
  /// \param[in] label Label of the allocation
  /// \param[in] n Number of elements
  /// \param[in] is_exchanged Whether the buffer is exchanged between ranks
  /// (a send or receive buffer of a transpose), not only local scratch. The
  /// default allocation does not depend on it.
  CommBuffer(const std::string& label, std::size_t n,
             [[maybe_unused]] bool is_exchanged)
      : m_allocation(label, n) {}

  CommBuffer()                             = delete;
  CommBuffer(const CommBuffer&)            = delete;
  CommBuffer& operator=(const CommBuffer&) = delete;
  CommBuffer(CommBuffer&&)                 = delete;
  CommBuffer& operator=(CommBuffer&&)      = delete;
  ~CommBuffer()                            = default;

  /// \brief The pointer to the allocation
  T* data() const { return m_allocation.data(); }

  /// \brief The number of elements
  std::size_t size() const { return m_allocation.size(); }
};

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
