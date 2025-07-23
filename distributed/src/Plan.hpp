#ifndef PLAN_HPP
#define PLAN_HPP

#include <vector>
#include <memory>
#include <iostream>
#include <sstream>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "InternalPlan.hpp"
#include "SharedPlan.hpp"
#include "SlabPlan.hpp"

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1>
static std::unique_ptr<
    InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM>>
internal_plan_factory(
    const ExecutionSpace& exec_space, const InViewType& in,
    const OutViewType& out, const KokkosFFT::axis_type<DIM>& axes,
    const KokkosFFT::shape_type<InViewType::rank()>& in_topology,
    const KokkosFFT::shape_type<OutViewType::rank()>& out_topology,
    const MPI_Comm& comm,
    KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward) {
  bool is_shared =
      is_shared_topology(in_topology) && is_shared_topology(out_topology);
  bool is_slab =
      is_slab_topology(in_topology) && is_slab_topology(out_topology);
  bool is_pencil =
      is_pencil_topology(in_topology) && is_pencil_topology(out_topology);

  if (is_shared) {
    return std::make_unique<
        SharedPlan<ExecutionSpace, InViewType, OutViewType, DIM>>(
        exec_space, in, out, axes, in_topology, out_topology, comm, norm);
  } else if (is_slab) {
    return std::make_unique<
        SlabPlan<ExecutionSpace, InViewType, OutViewType, DIM>>(
        exec_space, in, out, axes, in_topology, out_topology, comm, norm);
  } else if (is_pencil) {
    // Pencil plans can be implemented similarly to slab plans
    // return std::make_unique<PencilPlan<ExecutionSpace, InViewType,
    // OutViewType, DIM>>(exec_space, in, out, axes, in_topology, out_topology,
    // comm);
    throw std::runtime_error("Pencil plans are not yet implemented.");
  } else {
    // Default case for unsupported topologies
    throw std::runtime_error("Unsupported topology for FFT plan.");
  }
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1>
class Plan {
  static_assert(DIM >= 1 && DIM <= 3,
                "Plan: the Rank of FFT axes must be between 1 and 3");
  using InternalPlanType =
      InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM>;
  using axes_type     = KokkosFFT::axis_type<DIM>;
  using topology_type = KokkosFFT::shape_type<InViewType::rank()>;

  std::unique_ptr<InternalPlanType> m_internal_plan;

 public:
  explicit Plan(
      const ExecutionSpace& exec_space, const InViewType& in,
      const OutViewType& out, const axes_type& axes,
      const topology_type& in_topology, const topology_type& out_topology,
      const MPI_Comm& comm,
      KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward) {
    m_internal_plan = internal_plan_factory(
        exec_space, in, out, axes, in_topology, out_topology, comm, norm);
  }

  void forward_impl(const InViewType& in, const OutViewType& out) const {
    m_internal_plan->forward(in, out);
  }

  void backward_impl(const OutViewType& out, const InViewType& in) const {
    m_internal_plan->backward(out, in);
  }
};

template <typename PlanType, typename InViewType, typename OutViewType>
void execute(const PlanType& plan, const InViewType& in, const OutViewType& out,
             KokkosFFT::Direction direction) {
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;
  if (direction == KokkosFFT::Direction::forward) {
    if constexpr (KokkosFFT::Impl::is_complex_v<out_value_type>) {
      plan.forward_impl(in, out);
    } else {
      KOKKOSFFT_THROW_IF(true,
                         "Forward FFT operation requires complex output type.");
    }
  } else if (direction == KokkosFFT::Direction::backward) {
    if constexpr (KokkosFFT::Impl::is_complex_v<in_value_type>) {
      plan.backward_impl(in, out);
    } else {
      KOKKOSFFT_THROW_IF(true,
                         "Backward FFT operation requires complex input type.");
    }
  } else {
    KOKKOSFFT_THROW_IF(true, "Invalid FFT direction specified.");
  }
}
#endif
