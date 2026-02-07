#ifndef KOKKOSFFT_DISTRIBUTED_SHARED_PLAN_HPP
#define KOKKOSFFT_DISTRIBUTED_SHARED_PLAN_HPP

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_InternalPlan.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1, typename InLayoutType = Kokkos::LayoutRight,
          typename OutLayoutType = Kokkos::LayoutRight>
class SharedPlan : public InternalPlan<ExecutionSpace, InViewType, OutViewType,
                                       DIM, InLayoutType, OutLayoutType> {
  using axes_type    = KokkosFFT::axis_type<DIM>;
  using extents_type = KokkosFFT::shape_type<InViewType::rank()>;
  using in_topology_type =
      Topology<std::size_t, InViewType::rank(), InLayoutType>;
  using out_topology_type =
      Topology<std::size_t, OutViewType::rank(), OutLayoutType>;

  using FFTForwardPlanType =
      KokkosFFT::Plan<ExecutionSpace, InViewType, OutViewType, DIM>;
  using FFTBackwardPlanType =
      KokkosFFT::Plan<ExecutionSpace, OutViewType, InViewType, DIM>;
  FFTForwardPlanType m_forward_plan;
  FFTBackwardPlanType m_backward_plan;

  using InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM, InLayoutType,
                     OutLayoutType>::good;
  using InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM, InLayoutType,
                     OutLayoutType>::get_norm;

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

  explicit SharedPlan(
      const ExecutionSpace& exec_space, const InViewType& in,
      const OutViewType& out, const axes_type& axes,
      const in_topology_type& in_topology,
      const out_topology_type& out_topology, const MPI_Comm& comm,
      KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward)
      : InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM, InLayoutType,
                     OutLayoutType>(exec_space, in, out, axes, in_topology,
                                    out_topology, comm, norm),
        m_forward_plan(exec_space, in, out, KokkosFFT::Direction::forward,
                       axes),
        m_backward_plan(exec_space, out, in, KokkosFFT::Direction::backward,
                        axes) {
    KOKKOSFFT_THROW_IF(in_topology != out_topology,
                       "in_topology must be identical to out_topology.");
    for (auto axis : axes) {
      // We have to check that all the extents of in_topology/out_topology
      // at axes are equal to 1
      KOKKOSFFT_THROW_IF(in_topology.at(axis) != 1,
                         "in/out_topology must not have at non-one element.");
    }
  }

  void forward(const InViewType& in, const OutViewType& out) const override {
    good(in, out);
    KokkosFFT::execute(m_forward_plan, in, out, get_norm());
  }

  void backward(const OutViewType& out, const InViewType& in) const override {
    good(in, out);
    KokkosFFT::execute(m_backward_plan, out, in, get_norm());
  }

  /// \brief Get the name of the plan implementation
  /// \return Name of the plan implementation
  std::string name() const override { return std::string("SharedPlan"); }
};

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
