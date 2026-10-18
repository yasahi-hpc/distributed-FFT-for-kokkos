#ifndef KOKKOSFFT_DISTRIBUTED_PENCILPLAN_HPP
#define KOKKOSFFT_DISTRIBUTED_PENCILPLAN_HPP

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
#include "KokkosFFT_Distributed_InternalPlan.hpp"
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

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM, typename InLayoutType, typename OutLayoutType>
struct PencilInternalPlan;

/// \brief Internal plan for 1D FFT with pencil decomposition
/// 1. 1D FFT
/// 1. Transpose + 1D FFT
/// 1. Transpose + 1D FFT + Transpose
///
/// \tparam ExecutionSpace
/// \tparam InViewType
/// \tparam OutViewType

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          typename InLayoutType, typename OutLayoutType>
struct PencilInternalPlan<ExecutionSpace, InViewType, OutViewType, 1,
                          InLayoutType, OutLayoutType> {
  using execSpace      = ExecutionSpace;
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;
  using float_type   = KokkosFFT::Impl::base_floating_point_type<in_value_type>;
  using complex_type = Kokkos::complex<float_type>;
  using LayoutType   = typename InViewType::array_layout;

  static constexpr std::size_t DIM = InViewType::rank();

  using axes_type     = KokkosFFT::axis_type<1>;
  using topology_type = KokkosFFT::shape_type<DIM>;
  using in_topology_type =
      Topology<std::size_t, InViewType::rank(), InLayoutType>;
  using out_topology_type =
      Topology<std::size_t, OutViewType::rank(), OutLayoutType>;

  using PencilBlockAnalysesType =
      PencilBlockAnalysesInternal<in_value_type, LayoutType, std::size_t, DIM,
                                  1, InLayoutType, OutLayoutType>;

  // Type for communicators
  using CommType = KokkosFFT::Distributed::Impl::TplComm<ExecutionSpace>;

  // Type for the block executor
  using BlockExecutorType =
      BlockExecutor<ExecutionSpace, InViewType, OutViewType>;

  execSpace m_exec_space;
  axes_type m_axes;
  in_topology_type m_in_topology;
  out_topology_type m_out_topology;
  MPI_Comm m_comm;

  // Cartesian Communicators
  MPI_Comm m_cart_comm;
  std::vector<MPI_Comm> m_cart_comms;
  std::vector<std::unique_ptr<CommType>> m_cart_tpl_comms;

  // Analyse topology
  std::unique_ptr<PencilBlockAnalysesType> m_block_analyses;

  // Executes the blocks following the schedule
  std::unique_ptr<BlockExecutorType> m_executor;

 public:
  explicit PencilInternalPlan(const ExecutionSpace& exec_space,
                              const InViewType& in, const OutViewType& out,
                              const axes_type& axes,
                              const in_topology_type& in_topology,
                              const out_topology_type& out_topology,
                              const MPI_Comm& comm)
      : m_exec_space(exec_space),
        m_axes(axes),
        m_in_topology(in_topology),
        m_out_topology(out_topology),
        m_comm(comm) {
    create_pencil_communicators(m_exec_space, m_in_topology, m_comm,
                                m_cart_comm, m_cart_comms, m_cart_tpl_comms);

    // First get global shape to define buffer and next shape
    auto in_extents   = KokkosFFT::Impl::extract_extents(in);
    auto gin_extents  = compute_global_extents(in, m_in_topology, m_comm);
    auto gout_extents = compute_global_extents(out, m_out_topology, m_comm);

    auto non_negative_axes =
        KokkosFFT::Impl::convert_base_int_type<std::size_t>(
            KokkosFFT::Impl::convert_negative_axes(axes, DIM));

    m_block_analyses = std::make_unique<PencilBlockAnalysesType>(
        in_extents, gin_extents, gout_extents, in_topology, out_topology,
        non_negative_axes, m_comm);

    KOKKOSFFT_THROW_IF(!(m_block_analyses->m_block_infos.size() >= 1 &&
                         m_block_analyses->m_block_infos.size() <= 4),
                       "Number blocks must be in [1, 4]");

    m_executor = std::make_unique<BlockExecutorType>(
        m_exec_space, m_block_analyses->m_block_infos,
        m_block_analyses->m_max_buffer_size);
  }

  ~PencilInternalPlan() {
    // Free sub communicators
    for (auto& comm : m_cart_comms) {
      ::MPI_Comm_free(&comm);
    }

    // Then free cart communicator
    ::MPI_Comm_free(&m_cart_comm);
  }

  void forward(const InViewType& in, const OutViewType& out) const {
    m_executor->forward(comms(), in, out);
  }

  void backward(const OutViewType& out, const InViewType& in) const {
    m_executor->backward(comms(), out, in);
  }

 private:
  /// \brief The communicator of a transpose along a comm axis
  auto comms() const {
    return [this](std::size_t comm_axis) -> const CommType& {
      return *m_cart_tpl_comms.at(comm_axis);
    };
  }
};

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          typename InLayoutType, typename OutLayoutType>
struct PencilInternalPlan<ExecutionSpace, InViewType, OutViewType, 2,
                          InLayoutType, OutLayoutType> {
  using execSpace      = ExecutionSpace;
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;
  using float_type   = KokkosFFT::Impl::base_floating_point_type<in_value_type>;
  using complex_type = Kokkos::complex<float_type>;
  using LayoutType   = typename InViewType::array_layout;

  static constexpr std::size_t DIM = InViewType::rank();

  using axes_type     = KokkosFFT::axis_type<2>;
  using topology_type = KokkosFFT::shape_type<DIM>;
  using in_topology_type =
      Topology<std::size_t, InViewType::rank(), InLayoutType>;
  using out_topology_type =
      Topology<std::size_t, OutViewType::rank(), OutLayoutType>;

  using PencilBlockAnalysesType =
      PencilBlockAnalysesInternal<in_value_type, LayoutType, std::size_t, DIM,
                                  2, InLayoutType, OutLayoutType>;

  // Type for communicators
  using CommType = KokkosFFT::Distributed::Impl::TplComm<ExecutionSpace>;

  // Type for the block executor
  using BlockExecutorType =
      BlockExecutor<ExecutionSpace, InViewType, OutViewType>;

  execSpace m_exec_space;
  axes_type m_axes;
  in_topology_type m_in_topology;
  out_topology_type m_out_topology;
  MPI_Comm m_comm;

  // Cartesian Communicators
  MPI_Comm m_cart_comm;
  std::vector<MPI_Comm> m_cart_comms;
  std::vector<std::unique_ptr<CommType>> m_cart_tpl_comms;

  // Analyse topology
  std::unique_ptr<PencilBlockAnalysesType> m_block_analyses;

  // Executes the blocks following the schedule
  std::unique_ptr<BlockExecutorType> m_executor;

 public:
  explicit PencilInternalPlan(const ExecutionSpace& exec_space,
                              const InViewType& in, const OutViewType& out,
                              const axes_type& axes,
                              const in_topology_type& in_topology,
                              const out_topology_type& out_topology,
                              const MPI_Comm& comm)
      : m_exec_space(exec_space),
        m_axes(axes),
        m_in_topology(in_topology),
        m_out_topology(out_topology),
        m_comm(comm) {
    create_pencil_communicators(m_exec_space, m_in_topology, m_comm,
                                m_cart_comm, m_cart_comms, m_cart_tpl_comms);

    // First get global shape to define buffer and next shape
    auto in_extents   = KokkosFFT::Impl::extract_extents(in);
    auto gin_extents  = compute_global_extents(in, m_in_topology, m_comm);
    auto gout_extents = compute_global_extents(out, m_out_topology, m_comm);

    auto non_negative_axes =
        KokkosFFT::Impl::convert_base_int_type<std::size_t>(
            KokkosFFT::Impl::convert_negative_axes(axes, DIM));

    m_block_analyses = std::make_unique<PencilBlockAnalysesType>(
        in_extents, gin_extents, gout_extents, in_topology, out_topology,
        non_negative_axes, m_comm);

    KOKKOSFFT_THROW_IF(!(m_block_analyses->m_block_infos.size() >= 1 &&
                         m_block_analyses->m_block_infos.size() <= 6),
                       "Number blocks must be in [1, 6]");

    m_executor = std::make_unique<BlockExecutorType>(
        m_exec_space, m_block_analyses->m_block_infos,
        m_block_analyses->m_max_buffer_size);
  }

  ~PencilInternalPlan() {
    // Free sub communicators
    for (auto& comm : m_cart_comms) {
      ::MPI_Comm_free(&comm);
    }

    // Then free cart communicator
    ::MPI_Comm_free(&m_cart_comm);
  }

  void forward(const InViewType& in, const OutViewType& out) const {
    m_executor->forward(comms(), in, out);
  }

  void backward(const OutViewType& out, const InViewType& in) const {
    m_executor->backward(comms(), out, in);
  }

 private:
  /// \brief The communicator of a transpose along a comm axis
  auto comms() const {
    return [this](std::size_t comm_axis) -> const CommType& {
      return *m_cart_tpl_comms.at(comm_axis);
    };
  }
};

/// \brief Internal plan for 3D FFT with slab decomposition
/// 1. 1D FFT + Transpose + 1D FFT + Transpose + 1D FFT (+ Transpose)
/// 2. 2D FFT + Transpose + 1D FFT (+ Transpose)
/// 3. Transpose + 3D FFT (+ Transpose)
/// 4. Transpose + 2D FFT + Transpose + 1D FFT (+ Transpose)
/// 5. Transpose + 1D FFT + Transpose + 1D FFT + Transpose + 1D FFT (+
/// Transpose)
///
/// \tparam ExecutionSpace
/// \tparam InViewType
/// \tparam OutViewType
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          typename InLayoutType, typename OutLayoutType>
struct PencilInternalPlan<ExecutionSpace, InViewType, OutViewType, 3,
                          InLayoutType, OutLayoutType> {
  using execSpace      = ExecutionSpace;
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;
  using float_type   = KokkosFFT::Impl::base_floating_point_type<in_value_type>;
  using complex_type = Kokkos::complex<float_type>;
  using LayoutType   = typename InViewType::array_layout;

  static constexpr std::size_t DIM = InViewType::rank();

  using axes_type     = KokkosFFT::axis_type<3>;
  using topology_type = KokkosFFT::shape_type<DIM>;
  using in_topology_type =
      Topology<std::size_t, InViewType::rank(), InLayoutType>;
  using out_topology_type =
      Topology<std::size_t, OutViewType::rank(), OutLayoutType>;

  using PencilBlockAnalysesType =
      PencilBlockAnalysesInternal<in_value_type, LayoutType, std::size_t, DIM,
                                  3, InLayoutType, OutLayoutType>;

  // Type for communicators
  using CommType = KokkosFFT::Distributed::Impl::TplComm<ExecutionSpace>;

  // Type for the block executor
  using BlockExecutorType =
      BlockExecutor<ExecutionSpace, InViewType, OutViewType>;

  execSpace m_exec_space;
  axes_type m_axes;
  in_topology_type m_in_topology;
  out_topology_type m_out_topology;
  MPI_Comm m_comm;

  // Cartesian Communicators
  MPI_Comm m_cart_comm;
  std::vector<MPI_Comm> m_cart_comms;
  std::vector<std::unique_ptr<CommType>> m_cart_tpl_comms;

  // Analyse topology
  std::unique_ptr<PencilBlockAnalysesType> m_block_analyses;

  // Executes the blocks following the schedule
  std::unique_ptr<BlockExecutorType> m_executor;

 public:
  explicit PencilInternalPlan(const ExecutionSpace& exec_space,
                              const InViewType& in, const OutViewType& out,
                              const axes_type& axes,
                              const in_topology_type& in_topology,
                              const out_topology_type& out_topology,
                              const MPI_Comm& comm)
      : m_exec_space(exec_space),
        m_axes(axes),
        m_in_topology(in_topology),
        m_out_topology(out_topology),
        m_comm(comm) {
    create_pencil_communicators(m_exec_space, m_in_topology, m_comm,
                                m_cart_comm, m_cart_comms, m_cart_tpl_comms);

    // First get global shape to define buffer and next shape
    auto in_extents   = KokkosFFT::Impl::extract_extents(in);
    auto gin_extents  = compute_global_extents(in, m_in_topology, m_comm);
    auto gout_extents = compute_global_extents(out, m_out_topology, m_comm);

    auto non_negative_axes =
        KokkosFFT::Impl::convert_base_int_type<std::size_t>(
            KokkosFFT::Impl::convert_negative_axes(axes, DIM));

    m_block_analyses = std::make_unique<PencilBlockAnalysesType>(
        in_extents, gin_extents, gout_extents, in_topology, out_topology,
        non_negative_axes, m_comm);

    KOKKOSFFT_THROW_IF(!(m_block_analyses->m_block_infos.size() >= 1 &&
                         m_block_analyses->m_block_infos.size() <= 8),
                       "Number blocks must be in [1, 8]");

    m_executor = std::make_unique<BlockExecutorType>(
        m_exec_space, m_block_analyses->m_block_infos,
        m_block_analyses->m_max_buffer_size);
  }

  ~PencilInternalPlan() {
    // Free sub communicators
    for (auto& comm : m_cart_comms) {
      ::MPI_Comm_free(&comm);
    }

    // Then free cart communicator
    ::MPI_Comm_free(&m_cart_comm);
  }

  void forward(const InViewType& in, const OutViewType& out) const {
    m_executor->forward(comms(), in, out);
  }

  void backward(const OutViewType& out, const InViewType& in) const {
    m_executor->backward(comms(), out, in);
  }

 private:
  /// \brief The communicator of a transpose along a comm axis
  auto comms() const {
    return [this](std::size_t comm_axis) -> const CommType& {
      return *m_cart_tpl_comms.at(comm_axis);
    };
  }
};

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1, typename InLayoutType = Kokkos::LayoutRight,
          typename OutLayoutType = Kokkos::LayoutRight>
class PencilPlan : public InternalPlan<ExecutionSpace, InViewType, OutViewType,
                                       DIM, InLayoutType, OutLayoutType> {
  using InternalPlanType =
      PencilInternalPlan<ExecutionSpace, InViewType, OutViewType, DIM,
                         InLayoutType, OutLayoutType>;
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

  ///@{
  //! extents of in/out views
  extents_type m_in_extents, m_out_extents;
  ///@}

  //! aliases to parents methods
  using InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM, InLayoutType,
                     OutLayoutType>::good;
  using InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM, InLayoutType,
                     OutLayoutType>::get_norm;
  using InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM, InLayoutType,
                     OutLayoutType>::get_fft_extents;

 public:
  explicit PencilPlan(
      const ExecutionSpace& exec_space, const InViewType& in,
      const OutViewType& out, const axes_type& axes,
      const extents_type& in_topology, const extents_type& out_topology,
      const MPI_Comm& comm,
      KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward)
      : PencilPlan(exec_space, in, out, axes,
                   Topology<std::size_t, InViewType::rank()>(in_topology),
                   Topology<std::size_t, OutViewType::rank()>(out_topology),
                   comm, norm) {}

  explicit PencilPlan(
      const ExecutionSpace& exec_space, const InViewType& in,
      const OutViewType& out, const axes_type& axes,
      const in_topology_type& in_topology,
      const out_topology_type& out_topology, const MPI_Comm& comm,
      KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward)
      : InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM, InLayoutType,
                     OutLayoutType>(exec_space, in, out, axes, in_topology,
                                    out_topology, comm, norm),
        m_exec_space(exec_space),
        m_internal_plan(exec_space, in, out, axes, in_topology, out_topology,
                        comm),
        m_in_extents(KokkosFFT::Impl::extract_extents(in)),
        m_out_extents(KokkosFFT::Impl::extract_extents(out)) {
    auto in_size  = KokkosFFT::Impl::total_size(in_topology);
    auto out_size = KokkosFFT::Impl::total_size(out_topology);

    KOKKOSFFT_THROW_IF(in_size != out_size,
                       "Input and output topologies must have the same size.");

    bool is_pencil = are_specified_topologies(TopologyType::Pencil, in_topology,
                                              out_topology);
    KOKKOSFFT_THROW_IF(
        !is_pencil, "Input and output topologies must be pencil topologies.");
  }

  PencilPlan(const PencilPlan&) = default;
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
  std::string name() const override { return std::string("PencilPlan"); }
};

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
