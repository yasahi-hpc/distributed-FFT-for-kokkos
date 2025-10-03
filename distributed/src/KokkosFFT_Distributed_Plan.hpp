#ifndef KOKKOSFFT_DISTRIBUTED_PLAN_HPP
#define KOKKOSFFT_DISTRIBUTED_PLAN_HPP

#include <vector>
#include <memory>
#include <iostream>
#include <sstream>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_InternalPlan.hpp"
#include "KokkosFFT_Distributed_SharedPlan.hpp"
#include "KokkosFFT_Distributed_SlabPlan.hpp"
#include "KokkosFFT_Distributed_PencilPlan.hpp"

#if defined(PRIORITIZE_TPL_PLAN_IF_AVAILABLE)
#include "KokkosFFT_Distributed_TplPlan.hpp"
#endif

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1, typename InLayoutType = Kokkos::LayoutRight,
          typename OutLayoutType = Kokkos::LayoutRight>
static std::unique_ptr<InternalPlan<ExecutionSpace, InViewType, OutViewType,
                                    DIM, InLayoutType, OutLayoutType>>
internal_plan_factory(
    const ExecutionSpace& exec_space, const InViewType& in,
    const OutViewType& out, const KokkosFFT::axis_type<DIM>& axes,
    const Topology<std::size_t, InViewType::rank(), InLayoutType>& in_topology,
    const Topology<std::size_t, OutViewType::rank(), OutLayoutType>&
        out_topology,
    const MPI_Comm& comm,
    KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward) {
#if defined(PRIORITIZE_TPL_PLAN_IF_AVAILABLE)
  if constexpr ((InViewType::rank() == 2 && DIM == 2) ||
                (InViewType::rank() == 3 && DIM == 3)) {
    if (is_tpl_available(exec_space, in, out, axes, in_topology,
                         out_topology)) {
      return std::make_unique<
          TplPlan<ExecutionSpace, InViewType, OutViewType, DIM>>(
          exec_space, in, out, axes, in_topology, out_topology, comm, norm);
    }
  }
#endif

  auto in_topo = in_topology.array(), out_topo = out_topology.array();

  bool is_shared = is_shared_topology(in_topo) && is_shared_topology(out_topo);
  bool is_slab   = is_slab_topology(in_topo) && is_slab_topology(out_topo);
  bool is_pencil = is_pencil_topology(in_topo) && is_pencil_topology(out_topo);

  if (is_shared) {
    return std::make_unique<
        SharedPlan<ExecutionSpace, InViewType, OutViewType, DIM>>(
        exec_space, in, out, axes, in_topology, out_topology, comm, norm);
  } else if (is_slab) {
    return std::make_unique<
        SlabPlan<ExecutionSpace, InViewType, OutViewType, DIM>>(
        exec_space, in, out, axes, in_topology, out_topology, comm, norm);
  } else if (is_pencil) {
    if constexpr (InViewType::rank() >= 3) {
      return std::make_unique<
          PencilPlan<ExecutionSpace, InViewType, OutViewType, DIM, InLayoutType,
                     OutLayoutType>>(exec_space, in, out, axes, in_topology,
                                     out_topology, comm, norm);
    } else {
      throw std::runtime_error(
          "Pencil plan supports 3D or higher dimensional Views");
    }
  } else {
    // Default case for unsupported topologies
    throw std::runtime_error("Unsupported topology for FFT plan.");
  }
}

}  // namespace Impl

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1, typename InLayoutType = Kokkos::LayoutRight,
          typename OutLayoutType = Kokkos::LayoutRight>
class Plan {
  static_assert(DIM >= 1 && DIM <= 3,
                "Plan: the Rank of FFT axes must be between 1 and 3");
  using InternalPlanType =
      InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM>;
  using axes_type = KokkosFFT::axis_type<DIM>;
  using in_topology_type =
      Topology<std::size_t, InViewType::rank(), InLayoutType>;
  using out_topology_type =
      Topology<std::size_t, OutViewType::rank(), OutLayoutType>;
  using extents_type = std::array<std::size_t, InViewType::rank()>;

  std::unique_ptr<InternalPlanType> m_internal_plan;

 public:
  explicit Plan(
      const ExecutionSpace& exec_space, const InViewType& in,
      const OutViewType& out, const axes_type& axes,
      const extents_type& in_topology, const extents_type& out_topology,
      const MPI_Comm& comm,
      KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward)
      : Plan(exec_space, in, out, axes,
             Topology<std::size_t, InViewType::rank()>(in_topology),
             Topology<std::size_t, OutViewType::rank()>(out_topology), comm,
             norm) {}
  explicit Plan(
      const ExecutionSpace& exec_space, const InViewType& in,
      const OutViewType& out, const axes_type& axes,
      const in_topology_type& in_topology,
      const out_topology_type& out_topology, const MPI_Comm& comm,
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

}  // namespace Distributed
}  // namespace KokkosFFT

#endif
