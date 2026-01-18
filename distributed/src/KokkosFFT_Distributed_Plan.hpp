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

/// \brief Factory function to create a distributed FFT plan based on the given
/// topologies.
/// \tparam ExecutionSpace Kokkos execution space
/// \tparam InViewType Input Kokkos view type
/// \tparam OutViewType Output Kokkos view type
/// \tparam DIM Rank of FFT (default: 1)
/// \tparam InLayoutType Layout of input topology (default: LayoutRight)
/// \tparam OutLayoutType Layout of output topology (default: LayoutRight)
/// \param[in] exec_space Kokkos execution space
/// \param[in] in Input Kokkos view
/// \param[in] out Output Kokkos view
/// \param[in] axes FFT axes
/// \param[in] in_topology Input topology
/// \param[in] out_topology Output topology
/// \param[in] comm MPI communicator
/// \param[in] norm Normalization type (default: backward)
/// \return A unique pointer to the created distributed FFT plan
/// \throws std::runtime_error if the input or output topology sizes do not
/// match
/// \throws std::runtime_error if the topology is unsupported or invalid
/// \throws std::runtime_error if Pencil plan is requested for Views with rank
/// less than 3
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
  int nprocs;
  MPI_Comm_size(comm, &nprocs);

  auto in_total_size  = KokkosFFT::Impl::total_size(in_topology);
  auto out_total_size = KokkosFFT::Impl::total_size(out_topology);

  KOKKOSFFT_THROW_IF(in_total_size != out_total_size,
                     "in_topology and out_topology must have the same size.");
  KOKKOSFFT_THROW_IF(static_cast<int>(in_total_size) != nprocs,
                     "topology size must be identical to mpi size.");

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
  auto topology_type = get_common_topology_type(in_topology, out_topology);
  if (topology_type == TopologyType::Shared) {
    return std::make_unique<
        SharedPlan<ExecutionSpace, InViewType, OutViewType, DIM>>(
        exec_space, in, out, axes, in_topology, out_topology, comm, norm);
  } else if (topology_type == TopologyType::Slab) {
    return std::make_unique<
        SlabPlan<ExecutionSpace, InViewType, OutViewType, DIM>>(
        exec_space, in, out, axes, in_topology, out_topology, comm, norm);
  } else if (topology_type == TopologyType::Pencil) {
    if constexpr (InViewType::rank() >= 3) {
      return std::make_unique<
          PencilPlan<ExecutionSpace, InViewType, OutViewType, DIM, InLayoutType,
                     OutLayoutType>>(exec_space, in, out, axes, in_topology,
                                     out_topology, comm, norm);
    } else {
      throw std::runtime_error(
          "Pencil plan supports 3D or higher dimensional Views");
    }
  } else if (topology_type == TopologyType::Empty) {
    throw std::runtime_error("Empty topology is not supported for FFT plan.");
  } else if (topology_type == TopologyType::Brick) {
    throw std::runtime_error("Brick topology is not supported for FFT plan.");
  } else {
    // Default case for unsupported topologies
    throw std::runtime_error("Unsupported topology for FFT plan.");
  }
}

}  // namespace Impl

/// \brief Distributed FFT Plan class
/// \tparam ExecutionSpace Kokkos execution space
/// \tparam InViewType Input Kokkos view type
/// \tparam OutViewType Output Kokkos view type
/// \tparam DIM Rank of FFT axes (default: 1)
/// \tparam InLayoutType Layout of input topology (default: LayoutRight)
/// \tparam OutLayoutType Layout of output topology (default: LayoutRight)
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1, typename InLayoutType = Kokkos::LayoutRight,
          typename OutLayoutType = Kokkos::LayoutRight>
class Plan {
  using InternalPlanType =
      Impl::InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM>;
  using axes_type = KokkosFFT::axis_type<DIM>;
  using in_topology_type =
      Topology<std::size_t, InViewType::rank(), InLayoutType>;
  using out_topology_type =
      Topology<std::size_t, OutViewType::rank(), OutLayoutType>;
  using extents_type   = std::array<std::size_t, InViewType::rank()>;
  using out_value_type = typename OutViewType::non_const_value_type;
  static_assert(DIM >= 1 && DIM <= 3,
                "Plan: the Rank of FFT axes must be between 1 and 3");
  static_assert(KokkosFFT::Impl::is_complex_v<out_value_type>,
                "Plan: the output type must be complex, while the input type "
                "can be either real or complex");

  std::unique_ptr<InternalPlanType> m_internal_plan;

 public:
  /// \brief Distributed FFT Plan constructor
  /// \param[in] exec_space Kokkos execution space
  /// \param[in] in Input Kokkos view
  /// \param[in] out Output Kokkos view
  /// \param[in] axes FFT axes
  /// \param[in] in_topology Input topology in std::array
  /// \param[in] out_topology Output topology in std::array
  /// \param[in] comm MPI communicator
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

  /// \brief Distributed FFT Plan constructor
  /// \param[in] exec_space Kokkos execution space
  /// \param[in] in Input Kokkos view
  /// \param[in] out Output Kokkos view
  /// \param[in] axes FFT axes
  /// \param[in] in_topology Input topology
  /// \param[in] out_topology Output topology
  /// \param[in] comm MPI communicator
  explicit Plan(
      const ExecutionSpace& exec_space, const InViewType& in,
      const OutViewType& out, const axes_type& axes,
      const in_topology_type& in_topology,
      const out_topology_type& out_topology, const MPI_Comm& comm,
      KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward) {
    m_internal_plan = Impl::internal_plan_factory(
        exec_space, in, out, axes, in_topology, out_topology, comm, norm);
  }

  /// \brief Distributed FFT forward operation
  /// \param[in] in Input Kokkos view
  /// \param[out] out Output Kokkos view
  void forward_impl(const InViewType& in, const OutViewType& out) const {
    m_internal_plan->forward(in, out);
  }

  /// \brief Distributed FFT backward operation
  /// \param[in] out Output Kokkos view
  /// \param[out] in Input Kokkos view
  void backward_impl(const OutViewType& out, const InViewType& in) const {
    m_internal_plan->backward(out, in);
  }
};

/// \brief Distributed FFT execute function
/// \tparam PlanType Distributed FFT plan
/// \tparam InViewType Input Kokkos view type
/// \tparam OutViewType Output Kokkos view type
///
/// \param[in] plan Distributed FFT plan
/// \param[in] in Input Kokkos view
/// \param[out] out Output Kokkos view
/// \param[in] direction FFT direction
/// \throw std::runtime_error if the FFT direction is invalid
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
