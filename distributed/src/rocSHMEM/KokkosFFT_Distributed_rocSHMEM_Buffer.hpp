#ifndef KOKKOSFFT_DISTRIBUTED_ROCSHMEM_BUFFER_HPP
#define KOKKOSFFT_DISTRIBUTED_ROCSHMEM_BUFFER_HPP

#include <cstddef>
#include <optional>
#include <string>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Asserts.hpp"
#include "KokkosFFT_Distributed_CommBuffer.hpp"
#include "KokkosFFT_Distributed_rocSHMEM_Types.hpp"
#include "KokkosFFT_Distributed_rocSHMEM_Init.hpp"
#include "KokkosFFT_Distributed_rocSHMEM_Comm.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief A send or receive buffer of the rocSHMEM transposes
///
/// A buffer exchanged between ranks is allocated on the rocSHMEM symmetric
/// heap: its size is reduced (maximum) over the SHMEM communicator, so that
/// every PE allocates the same size, and rocshmem_malloc and rocshmem_free
/// are collective over it. The buffers must therefore be allocated and freed
/// in the same order on every rank, before Kokkos::finalize. A buffer that
/// only holds local scratch is a managed Kokkos View, as by default.
///
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam T Value type of the buffer
template <typename ExecutionSpace, typename T>
class CommBuffer<ScopedrocSHMEMComm, ExecutionSpace, T> {
  using AllocationViewType = Kokkos::View<T*, ExecutionSpace>;

  //! The allocation of a local buffer
  std::optional<AllocationViewType> m_view;

  //! The symmetric allocation of an exchanged buffer
  T* m_symmetric     = nullptr;
  std::size_t m_size = 0;

 public:
  /// \brief Allocate the buffer
  /// \param[in] label Label of a local allocation
  /// \param[in] n Number of elements on this rank
  /// \param[in] is_exchanged Whether the buffer is exchanged between ranks.
  /// If so, the allocation is collective over the SHMEM communicator.
  /// \throws std::runtime_error if rocSHMEM is not set up, or if the
  /// symmetric heap is too small
  CommBuffer(const std::string& label, std::size_t n, bool is_exchanged) {
    if (!is_exchanged) {
      m_view.emplace(label, n);
      m_size = n;
      return;
    }

    MPI_Comm world = rocshmem_comm();
    KOKKOSFFT_THROW_IF(world == MPI_COMM_NULL,
                       "CommBuffer: rocSHMEM must be set up (by a "
                       "ScopedrocSHMEMComm) before allocating a symmetric "
                       "buffer");

    unsigned long long local = n, global = 0;
    ::MPI_Allreduce(&local, &global, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, world);
    m_size = static_cast<std::size_t>(global);
    if (m_size == 0) return;

    m_symmetric =
        static_cast<T*>(rocshmem::rocshmem_malloc(m_size * sizeof(T)));
    KOKKOSFFT_THROW_IF(m_symmetric == nullptr,
                       "CommBuffer: rocshmem_malloc of " +
                           std::to_string(m_size * sizeof(T)) +
                           " bytes failed. Increase the symmetric heap with "
                           "ROCSHMEM_HEAP_SIZE");
  }

  CommBuffer()                             = delete;
  CommBuffer(const CommBuffer&)            = delete;
  CommBuffer& operator=(const CommBuffer&) = delete;
  CommBuffer(CommBuffer&&)                 = delete;
  CommBuffer& operator=(CommBuffer&&)      = delete;

  ~CommBuffer() {
    if (m_symmetric) rocshmem::rocshmem_free(m_symmetric);
  }

  /// \brief The pointer to the allocation
  T* data() const { return m_view ? m_view->data() : m_symmetric; }

  /// \brief The number of elements. For an exchanged buffer, the maximum
  /// over the ranks, which may be more than requested on this rank.
  std::size_t size() const { return m_size; }

  /// \brief Whether the buffer is on the symmetric heap
  bool is_symmetric() const { return m_symmetric != nullptr; }
};

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
