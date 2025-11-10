#ifndef KOKKOSFFT_DISTRIBUTED_ONECCL_COMM_HPP
#define KOKKOSFFT_DISTRIBUTED_ONECCL_COMM_HPP

#include <cstdint>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_MPI_Comm.hpp"
#include "KokkosFFT_Distributed_oneCCL_Types.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

struct ScopedoneCCLComm {
  Kokkos::SYCL m_exec_space;
  ccl::vector_class<ccl::communicator> m_comms;
  ccl::vector_class<ccl::stream> m_streams;
  int m_rank = 0;
  int m_size = 1;

 public:
  explicit ScopedoneCCLComm(MPI_Comm comm, const Kokkos::SYCL &exec_space)
      : m_exec_space(exec_space) {
    ::MPI_Comm_rank(comm, &m_rank);
    ::MPI_Comm_size(comm, &m_size);

    /* create kvs */
    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (m_rank == 0) {
      kvs       = ccl::create_main_kvs();
      main_addr = kvs->get_address();
      MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, comm);
    } else {
      MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, comm);
      kvs = ccl::create_kvs(main_addr);
    }

    /* create communicator */
    sycl::queue q = exec_space.sycl_queue();
    auto dev      = ccl::create_device(q.get_device());
    auto ctx      = ccl::create_context(q.get_context());
    m_comms.emplace_back(
        ccl::create_communicator(m_size, m_rank, dev, ctx, kvs));
    m_streams.emplace_back(ccl::create_stream(q));
  }
  explicit ScopedoneCCLComm(MPI_Comm comm)
      : ScopedoneCCLComm(comm, Kokkos::SYCL{}) {}

  ScopedoneCCLComm() = delete;

  // Delete copy semantics
  ScopedoneCCLComm(const ScopedoneCCLComm &)            = delete;
  ScopedoneCCLComm &operator=(const ScopedoneCCLComm &) = delete;

  // Allow move semantics
  ScopedoneCCLComm(ScopedoneCCLComm &&other) noexcept
      : m_exec_space(other.m_exec_space),
        m_comms(std::move(other.m_comms)),
        m_streams(std::move(other.m_streams)),
        m_rank(other.m_rank),
        m_size(other.m_size) {}

  // Move assignment operator
  ScopedoneCCLComm &operator=(ScopedoneCCLComm &&other) noexcept {
    if (this != &other) {
      Kokkos::kokkos_swap(m_exec_space, other.m_exec_space);
      std::swap(m_comms, other.m_comms);
      std::swap(m_streams, other.m_streams);
      std::swap(m_rank, other.m_rank);
      std::swap(m_size, other.m_size);
    }
    return *this;
  }

  ~ScopedoneCCLComm() = default;

  const auto &comms() const { return m_comms; }
  auto stream() const { return m_streams[0]; }
  auto exec_space() const { return m_exec_space; }
  int rank() const { return m_rank; }
  int size() const { return m_size; }
};

template <typename ExecutionSpace>
using TplComm =
    std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::SYCL>,
                       ScopedoneCCLComm, ScopedMPIComm<ExecutionSpace>>;
}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
