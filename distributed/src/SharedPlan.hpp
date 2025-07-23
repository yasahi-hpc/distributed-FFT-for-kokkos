#ifndef SHARED_PLAN_HPP
#define SHARED_PLAN_HPP

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "InternalPlan.hpp"

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1>
class SharedPlan
    : public InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM> {
  using axes_type     = KokkosFFT::axis_type<DIM>;
  using topology_type = KokkosFFT::shape_type<InViewType::rank()>;

  using FFTForwardPlanType =
      KokkosFFT::Plan<ExecutionSpace, InViewType, OutViewType, DIM>;
  using FFTBackwardPlanType =
      KokkosFFT::Plan<ExecutionSpace, OutViewType, InViewType, DIM>;
  FFTForwardPlanType m_forward_plan;
  FFTBackwardPlanType m_backward_plan;
  KokkosFFT::Normalization m_norm;

  using InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM>::good;

 public:
  explicit SharedPlan(
      const ExecutionSpace& exec_space, const InViewType& in,
      const OutViewType& out, const axes_type& axes,
      const topology_type& in_topology, const topology_type& out_topology,
      const MPI_Comm& comm,
      KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward)
      : InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM>(
            exec_space, in, out, axes, in_topology, out_topology, comm, norm),
        m_forward_plan(exec_space, in, out, KokkosFFT::Direction::forward,
                       axes),
        m_backward_plan(exec_space, out, in, KokkosFFT::Direction::backward,
                        axes),
        m_norm(norm) {
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
    KokkosFFT::execute(m_forward_plan, in, out, m_norm);
  }

  void backward(const OutViewType& out, const InViewType& in) const override {
    good(in, out);
    KokkosFFT::execute(m_backward_plan, out, in, m_norm);
  }
};

#endif
