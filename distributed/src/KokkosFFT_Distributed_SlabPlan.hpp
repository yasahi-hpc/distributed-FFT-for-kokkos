#ifndef KOKKOSFFT_DISTRIBUTED_SLABPLAN_HPP
#define KOKKOSFFT_DISTRIBUTED_SLABPLAN_HPP

#include <vector>
#include <memory>
#include <iostream>
#include <sstream>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_TransBlock.hpp"
#include "KokkosFFT_Distributed_MPI_Extents.hpp"
#include "KokkosFFT_Distributed_Helper.hpp"
#include "KokkosFFT_Distributed_Extents.hpp"
#include "KokkosFFT_Distributed_Topologies.hpp"
#include "KokkosFFT_Distributed_BlockAnalyses.hpp"
#include "KokkosFFT_Distributed_BasePlan.hpp"
#include "KokkosFFT_Distributed_PlanExecutor.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief Internal plan with slab decomposition
///
/// FFT_DIM = 1:
/// 1. 1D FFT
/// 2. Transpose + 1D FFT
/// 3. Transpose + 1D FFT + Transpose
///
/// FFT_DIM = 2:
/// 1. 1D FFT + Transpose + 1D FFT (+ Transpose)
/// 2. Transpose + 2D FFT (+ Transpose)
/// 3. Transpose + 1D FFT + Transpose + 1D FFT (+ Transpose)
///
/// FFT_DIM = 3:
/// 1. 1D FFT + Transpose + 1D FFT + Transpose + 1D FFT (+ Transpose)
/// 2. 2D FFT + Transpose + 1D FFT (+ Transpose)
/// 3. Transpose + 3D FFT (+ Transpose)
/// 4. Transpose + 2D FFT + Transpose + 1D FFT (+ Transpose)
/// 5. Transpose + 1D FFT + Transpose + 1D FFT + Transpose + 1D FFT (+
///    Transpose)
///
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam InViewType Input Kokkos view type
/// \tparam OutViewType Output Kokkos view type
/// \tparam FFT_DIM Number of FFT dimensions
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t FFT_DIM>
using SlabPlanExecutor = BasePlanExecutor<
    ExecutionSpace, InViewType, OutViewType, FFT_DIM,
    SlabBlockAnalysesInternal<typename InViewType::non_const_value_type,
                              typename InViewType::array_layout, std::size_t,
                              InViewType::rank(), FFT_DIM>,
    SlabComms<ExecutionSpace>, KokkosFFT::shape_type<InViewType::rank()>,
    KokkosFFT::shape_type<InViewType::rank()>>;

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1, typename InLayoutType = Kokkos::LayoutRight,
          typename OutLayoutType = Kokkos::LayoutRight>
class SlabPlan : public BasePlan<ExecutionSpace, InViewType, OutViewType, DIM,
                                 InLayoutType, OutLayoutType> {
  using PlanExecutorType =
      SlabPlanExecutor<ExecutionSpace, InViewType, OutViewType, DIM>;
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
  PlanExecutorType m_plan_executor;

  ///@{
  //! extents of in/out views
  extents_type m_in_extents, m_out_extents;
  ///@}

  //! aliases to parents methods
  using BasePlan<ExecutionSpace, InViewType, OutViewType, DIM, InLayoutType,
                 OutLayoutType>::good;
  using BasePlan<ExecutionSpace, InViewType, OutViewType, DIM, InLayoutType,
                 OutLayoutType>::get_norm;
  using BasePlan<ExecutionSpace, InViewType, OutViewType, DIM, InLayoutType,
                 OutLayoutType>::get_fft_extents;

 public:
  explicit SlabPlan(
      const ExecutionSpace& exec_space, const InViewType& in,
      const OutViewType& out, const axes_type& axes,
      const extents_type& in_topology, const extents_type& out_topology,
      const MPI_Comm& comm,
      KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward)
      : SlabPlan(exec_space, in, out, axes,
                 Topology<std::size_t, InViewType::rank()>(in_topology),
                 Topology<std::size_t, OutViewType::rank()>(out_topology), comm,
                 norm) {}

  explicit SlabPlan(
      const ExecutionSpace& exec_space, const InViewType& in,
      const OutViewType& out, const axes_type& axes,
      const in_topology_type& in_topology,
      const out_topology_type& out_topology, const MPI_Comm& comm,
      KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward)
      : BasePlan<ExecutionSpace, InViewType, OutViewType, DIM, InLayoutType,
                 OutLayoutType>(exec_space, in, out, axes, in_topology,
                                out_topology, comm, norm),
        m_exec_space(exec_space),
        m_plan_executor(exec_space, in, out, axes, in_topology.array(),
                        out_topology.array(), comm),
        m_in_extents(KokkosFFT::Impl::extract_extents(in)),
        m_out_extents(KokkosFFT::Impl::extract_extents(out)) {
    auto in_size  = KokkosFFT::Impl::total_size(in_topology);
    auto out_size = KokkosFFT::Impl::total_size(out_topology);

    KOKKOSFFT_THROW_IF(in_size != out_size,
                       "Input and output topologies must have the same size.");

    bool is_slab =
        are_specified_topologies(TopologyType::Slab, in_topology, out_topology);
    KOKKOSFFT_THROW_IF(!is_slab,
                       "Input and output topologies must be slab topologies.");
  }

  void forward(const InViewType& in, const OutViewType& out) const override {
    good(in, out);
    m_plan_executor.forward(in, out);
    KokkosFFT::Impl::normalize<normalization_float_type>(
        m_exec_space, out, KokkosFFT::Direction::forward, get_norm(),
        get_fft_extents());
  }

  void backward(const OutViewType& out, const InViewType& in) const override {
    good(in, out);
    m_plan_executor.backward(out, in);
    KokkosFFT::Impl::normalize<normalization_float_type>(
        m_exec_space, in, KokkosFFT::Direction::backward, get_norm(),
        get_fft_extents());
  }

  /// \brief Get the name of the plan implementation
  /// \return Name of the plan implementation
  std::string name() const override { return std::string("SlabPlan"); }
};

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
