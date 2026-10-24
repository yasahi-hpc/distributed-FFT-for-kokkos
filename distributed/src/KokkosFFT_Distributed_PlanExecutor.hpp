#ifndef KOKKOSFFT_DISTRIBUTED_PLAN_EXECUTOR_HPP
#define KOKKOSFFT_DISTRIBUTED_PLAN_EXECUTOR_HPP

#include <memory>
#include <string>
#include <vector>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Asserts.hpp"
#include "KokkosFFT_Distributed_All2All.hpp"
#include "KokkosFFT_Distributed_MPI_Extents.hpp"
#include "KokkosFFT_Distributed_BlockAnalyses.hpp"
#include "KokkosFFT_Distributed_BlockExecutor.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief Create pencil communicators (cartesian, row, column)
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam TopologyType Type of topology container
/// \tparam CommType Type of communicator wrapper
/// \param[in] exec_space Kokkos execution space
/// \param[in] topology The topology of the distributed FFT
/// \param[in] comm MPI communicator
/// \param[out] cart_comm The cartesian communicator
/// \param[out] cart_comms The row and column communicators
/// \param[out] cart_tpl_comms The communicator wrappers for row and column
template <typename ExecutionSpace, typename TopologyType, typename CommType>
void create_pencil_communicators(
    const ExecutionSpace& exec_space, const TopologyType& topology,
    const MPI_Comm& comm, MPI_Comm& cart_comm,
    std::vector<MPI_Comm>& cart_comms,
    std::vector<std::unique_ptr<CommType>>& cart_tpl_comms) {
  // Create a cartesian communicator
  std::vector<int> dims;
  for (auto& dim : topology) {
    if (dim > 1) {
      dims.push_back(static_cast<int>(dim));
    }
  }

  int periods[2] = {1, 1};  // Periodic in all directions
  ::MPI_Cart_create(comm, 2, dims.data(), periods, 0, &cart_comm);

  // split into row‐ and col‐ communicators
  ::MPI_Comm row_comm, col_comm;

  int remain_dims[2];

  // keep Y‐axis for row_comm (all procs with same px)
  remain_dims[0] = 1;
  remain_dims[1] = 0;

  ::MPI_Cart_sub(cart_comm, remain_dims, &row_comm);

  // keep X‐axis for col_comm (all procs with same py)
  remain_dims[0] = 0;
  remain_dims[1] = 1;
  ::MPI_Cart_sub(cart_comm, remain_dims, &col_comm);

  cart_comms = {row_comm, col_comm};

  // Ensure vector is empty before adding
  cart_tpl_comms.clear();
  cart_tpl_comms.emplace_back(std::make_unique<CommType>(row_comm, exec_space));
  cart_tpl_comms.emplace_back(std::make_unique<CommType>(col_comm, exec_space));
}

/// \brief Communicators of a slab decomposition: a single one for every
/// transpose
/// \tparam ExecutionSpace Kokkos execution space type
template <typename ExecutionSpace>
class SlabComms {
 public:
  using CommType = TplComm<ExecutionSpace>;

  //! The transposes of a slab plan communicate
  static constexpr bool has_communication = true;

 private:
  CommType m_comm;

 public:
  /// \param[in] exec_space Kokkos execution space
  /// \param[in] topology The input topology (unused for slabs)
  /// \param[in] comm MPI communicator
  template <typename TopologyType>
  SlabComms(const ExecutionSpace& exec_space, const TopologyType& /*topology*/,
            const MPI_Comm& comm)
      : m_comm(comm, exec_space) {}

  /// \brief The communicator of a transpose along a comm axis
  const CommType& operator()(std::size_t /*comm_axis*/) const { return m_comm; }
};

/// \brief Communicators of a pencil decomposition: the row and column
/// communicators of a cartesian communicator built from the input topology,
/// one per comm axis. They are freed when the object is destroyed.
/// \tparam ExecutionSpace Kokkos execution space type
template <typename ExecutionSpace>
class PencilComms {
 public:
  using CommType = TplComm<ExecutionSpace>;

  //! The transposes of a pencil plan communicate
  static constexpr bool has_communication = true;

 private:
  MPI_Comm m_cart_comm;
  std::vector<MPI_Comm> m_cart_comms;
  std::vector<std::unique_ptr<CommType>> m_cart_tpl_comms;

 public:
  /// \param[in] exec_space Kokkos execution space
  /// \param[in] topology The input topology
  /// \param[in] comm MPI communicator
  template <typename TopologyType>
  PencilComms(const ExecutionSpace& exec_space, const TopologyType& topology,
              const MPI_Comm& comm) {
    create_pencil_communicators(exec_space, topology, comm, m_cart_comm,
                                m_cart_comms, m_cart_tpl_comms);
  }

  PencilComms(const PencilComms&)            = delete;
  PencilComms& operator=(const PencilComms&) = delete;

  ~PencilComms() {
    // Free sub communicators
    for (auto& comm : m_cart_comms) {
      ::MPI_Comm_free(&comm);
    }

    // Then free cart communicator
    ::MPI_Comm_free(&m_cart_comm);
  }

  /// \brief The communicator of a transpose along a comm axis
  const CommType& operator()(std::size_t comm_axis) const {
    return *m_cart_tpl_comms.at(comm_axis);
  }
};

/// \brief Communicators of a plan without communication. Every FFT axis is
/// local, so the plan has no transpose block and never exchanges data between
/// ranks. It holds no MPI communicator, and BlockExecutor does not compile the
/// transpose path for it (has_communication is false).
/// \tparam ExecutionSpace Kokkos execution space type
template <typename ExecutionSpace>
class SharedComms {
 public:
  //! No transpose block may be executed with these communicators
  static constexpr bool has_communication = false;

  /// \param[in] exec_space Kokkos execution space (unused)
  /// \param[in] topology The input topology (unused)
  /// \param[in] comm MPI communicator (unused)
  template <typename TopologyType>
  SharedComms(const ExecutionSpace& /*exec_space*/,
              const TopologyType& /*topology*/, const MPI_Comm& /*comm*/) {}
};

/// \brief Executor of a distributed FFT plan: analyses the topologies into
/// blocks and runs them with a BlockExecutor. Slab and pencil
/// decompositions differ only in the block analyses, the communicators and
/// the topology types; SlabPlanExecutor, PencilPlanExecutor and
/// SharedPlanExecutor are aliases of this class.
///
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam InViewType Input Kokkos view type
/// \tparam OutViewType Output Kokkos view type
/// \tparam FFT_DIM Number of FFT dimensions
/// \tparam BlockAnalysesType Type of the block analyses (slab or pencil)
/// \tparam CommsType Communicators of the transposes (SlabComms, PencilComms
/// or SharedComms)
/// \tparam InTopologyType Type of the input topology
/// \tparam OutTopologyType Type of the output topology
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t FFT_DIM, typename BlockAnalysesType, typename CommsType,
          typename InTopologyType, typename OutTopologyType>
class BasePlanExecutor {
  static constexpr std::size_t DIM = InViewType::rank();

  using axes_type = KokkosFFT::axis_type<FFT_DIM>;

  // Type for the block executor
  using BlockExecutorType =
      BlockExecutor<ExecutionSpace, InViewType, OutViewType>;

  ExecutionSpace m_exec_space;

  // Communicators of the transposes
  CommsType m_comms;

  // Analyse topology
  std::unique_ptr<BlockAnalysesType> m_block_analyses;

  // Executes the blocks following the schedule
  std::unique_ptr<BlockExecutorType> m_executor;

 public:
  explicit BasePlanExecutor(const ExecutionSpace& exec_space,
                            const InViewType& in, const OutViewType& out,
                            const axes_type& axes,
                            const InTopologyType& in_topology,
                            const OutTopologyType& out_topology,
                            const MPI_Comm& comm)
      : m_exec_space(exec_space), m_comms(exec_space, in_topology, comm) {
    // First get global shape to define buffer and next shape
    auto in_extents   = KokkosFFT::Impl::extract_extents(in);
    auto gin_extents  = compute_global_extents(in, in_topology, comm);
    auto gout_extents = compute_global_extents(out, out_topology, comm);

    auto non_negative_axes =
        KokkosFFT::Impl::convert_base_int_type<std::size_t>(
            KokkosFFT::Impl::convert_negative_axes(axes, DIM));

    m_block_analyses = std::make_unique<BlockAnalysesType>(
        in_extents, gin_extents, gout_extents, in_topology, out_topology,
        non_negative_axes, comm);

    // Every FFT axis must be transformed exactly once. The number of blocks
    // is not limited: build_schedule validates the sequence of blocks.
    const std::size_t nb_fft_dims =
        count_fft_dims(m_block_analyses->m_block_infos);
    KOKKOSFFT_THROW_IF(nb_fft_dims != FFT_DIM,
                       "The FFT blocks must transform " +
                           std::to_string(FFT_DIM) + " axes, but transform " +
                           std::to_string(nb_fft_dims));

    m_executor = std::make_unique<BlockExecutorType>(
        m_exec_space, in, out, m_block_analyses->m_block_infos,
        m_block_analyses->m_max_buffer_size);
  }

  void forward(const InViewType& in, const OutViewType& out) const {
    m_executor->forward(m_comms, in, out);
  }

  void backward(const OutViewType& out, const InViewType& in) const {
    m_executor->backward(m_comms, out, in);
  }
};

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
