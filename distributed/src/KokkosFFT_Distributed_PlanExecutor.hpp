#ifndef KOKKOSFFT_DISTRIBUTED_PLAN_EXECUTOR_HPP
#define KOKKOSFFT_DISTRIBUTED_PLAN_EXECUTOR_HPP

#include <memory>
#include <string>
#include <vector>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Asserts.hpp"
#include "KokkosFFT_Distributed_MPI_Extents.hpp"
#include "KokkosFFT_Distributed_BlockAnalyses.hpp"
#include "KokkosFFT_Distributed_BlockExecutor.hpp"
#include "KokkosFFT_Distributed_PlanComms.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

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
