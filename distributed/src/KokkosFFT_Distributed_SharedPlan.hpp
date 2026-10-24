#ifndef KOKKOSFFT_DISTRIBUTED_SHARED_PLAN_HPP
#define KOKKOSFFT_DISTRIBUTED_SHARED_PLAN_HPP

#include <string>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_Types.hpp"
#include "KokkosFFT_Distributed_BlockAnalyses.hpp"
#include "KokkosFFT_Distributed_BasePlan.hpp"
#include "KokkosFFT_Distributed_PlanExecutor.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief Internal plan without communication: every FFT axis is local
///
/// The plan is a single FFT block without any transpose. The topologies may be
/// distributed along the other (batch) axes, but no data is exchanged between
/// the ranks during forward and backward. The FFT block runs DynPlans on the
/// FFT axes made contiguous by local transposes (FFTStage).
///
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam InViewType Input Kokkos view type
/// \tparam OutViewType Output Kokkos view type
/// \tparam FFT_DIM Number of FFT dimensions
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t FFT_DIM>
using SharedInternalPlan = DistributedInternalPlan<
    ExecutionSpace, InViewType, OutViewType, FFT_DIM,
    SharedBlockAnalysesInternal<typename InViewType::non_const_value_type,
                                typename InViewType::array_layout, std::size_t,
                                InViewType::rank(), FFT_DIM>,
    SharedComms<ExecutionSpace>, KokkosFFT::shape_type<InViewType::rank()>,
    KokkosFFT::shape_type<InViewType::rank()>>;

/// \brief Distributed FFT plan for topologies where every FFT axis is local.
/// It does not involve any communication between the ranks: the FFT is
/// performed locally on each rank.
///
/// The input and output topologies must be identical and must not be
/// distributed along any FFT axis. They may be distributed along the other
/// (batch) axes.
///
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam InViewType Input Kokkos view type
/// \tparam OutViewType Output Kokkos view type
/// \tparam DIM Number of FFT dimensions
/// \tparam InLayoutType Layout of the input topology
/// \tparam OutLayoutType Layout of the output topology
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1, typename InLayoutType = Kokkos::LayoutRight,
          typename OutLayoutType = Kokkos::LayoutRight>
class SharedPlan : public InternalPlan<ExecutionSpace, InViewType, OutViewType,
                                       DIM, InLayoutType, OutLayoutType> {
  using InternalPlanType =
      SharedInternalPlan<ExecutionSpace, InViewType, OutViewType, DIM>;
  using extents_type = std::array<std::size_t, InViewType::rank()>;
  using in_topology_type =
      Topology<std::size_t, InViewType::rank(), InLayoutType>;
  using out_topology_type =
      Topology<std::size_t, OutViewType::rank(), OutLayoutType>;
  using axes_type = KokkosFFT::axis_type<DIM>;

  //! The real value type for normalization
  using normalization_float_type = double;

  //! Execution space
  ExecutionSpace m_exec_space;

  //! Internal plan
  InternalPlanType m_internal_plan;

  //! aliases to parents methods
  using InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM, InLayoutType,
                     OutLayoutType>::good;
  using InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM, InLayoutType,
                     OutLayoutType>::get_norm;
  using InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM, InLayoutType,
                     OutLayoutType>::get_fft_extents;

 public:
  explicit SharedPlan(
      const ExecutionSpace& exec_space, const InViewType& in,
      const OutViewType& out, const axes_type& axes,
      const extents_type& in_topology, const extents_type& out_topology,
      const MPI_Comm& comm,
      KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward)
      : SharedPlan(exec_space, in, out, axes,
                   Topology<std::size_t, InViewType::rank()>(in_topology),
                   Topology<std::size_t, OutViewType::rank()>(out_topology),
                   comm, norm) {}

  /// \brief Constructor
  /// \throws std::runtime_error if the input and output topologies differ or
  /// if they are distributed along an FFT axis
  explicit SharedPlan(
      const ExecutionSpace& exec_space, const InViewType& in,
      const OutViewType& out, const axes_type& axes,
      const in_topology_type& in_topology,
      const out_topology_type& out_topology, const MPI_Comm& comm,
      KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward)
      : InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM, InLayoutType,
                     OutLayoutType>(exec_space, in, out, axes, in_topology,
                                    out_topology, comm, norm),
        m_exec_space(exec_space),
        m_internal_plan(exec_space, in, out, axes, in_topology.array(),
                        out_topology.array(), comm) {}

  void forward(const InViewType& in, const OutViewType& out) const override {
    good(in, out);
    m_internal_plan.forward(in, out);
    KokkosFFT::Impl::normalize<normalization_float_type>(
        m_exec_space, out, KokkosFFT::Direction::forward, get_norm(),
        get_fft_extents());
  }

  void backward(const OutViewType& out, const InViewType& in) const override {
    good(in, out);
    m_internal_plan.backward(out, in);
    KokkosFFT::Impl::normalize<normalization_float_type>(
        m_exec_space, in, KokkosFFT::Direction::backward, get_norm(),
        get_fft_extents());
  }

  /// \brief Get the name of the plan implementation
  /// \return Name of the plan implementation
  std::string name() const override { return std::string("SharedPlan"); }
};

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
