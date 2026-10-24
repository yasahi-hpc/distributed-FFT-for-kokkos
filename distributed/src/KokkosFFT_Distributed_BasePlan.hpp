#ifndef KOKKOSFFT_DISTRIBUTED_BASE_PLAN_HPP
#define KOKKOSFFT_DISTRIBUTED_BASE_PLAN_HPP

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_Types.hpp"
#include "KokkosFFT_Distributed_MPI_Extents.hpp"
#include "KokkosFFT_Distributed_Extents.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief A local FFT stage of a distributed FFT plan
/// It owns the forward and backward FFT plans of one FFT block together with
/// the local transposes applied before and after the FFT. The local
/// transpose of the input (in-map) is only needed if the FFT block is the
/// first block of the plan, and the one of the output (out-map) only if the
/// FFT block is the last block of the plan.
///
/// forward:  in -> [in-map: in -> in_T] -> FFT -> [out-map: out_T -> out]
/// backward: out -> [out-map: out -> out_T] -> iFFT -> [in-map: in_T -> in]
///
/// \tparam ExecutionSpace Kokkos execution space
/// \tparam InViewType Input Kokkos view type of the forward FFT
/// \tparam OutViewType Output Kokkos view type of the forward FFT
template <typename ExecutionSpace, typename InViewType, typename OutViewType>
class FFTStage {
  static constexpr std::size_t DIM = InViewType::rank();
  using map_type                   = std::array<std::size_t, DIM>;
  using int_map_type               = std::array<int, DIM>;
  using ForwardPlanType =
      KokkosFFT::DynPlan<ExecutionSpace, InViewType, OutViewType>;
  using BackwardPlanType =
      KokkosFFT::DynPlan<ExecutionSpace, OutViewType, InViewType>;

  ExecutionSpace m_exec_space;

  // Internal FFT plans
  ForwardPlanType m_forward_plan;
  BackwardPlanType m_backward_plan;

  // Mappings for local transpose. An empty map means no local transpose.
  int_map_type m_map_forward_in = {}, m_map_forward_out = {},
               m_map_backward_in = {}, m_map_backward_out = {};

 public:
  /// \brief FFTStage constructor
  /// \param[in] exec_space Kokkos execution space
  /// \param[in] in Input view of the forward FFT
  /// \param[in] out Output view of the forward FFT
  /// \param[in] fft_dim Dimensionality of the FFT
  explicit FFTStage(const ExecutionSpace& exec_space, const InViewType& in,
                    const OutViewType& out, std::size_t fft_dim)
      : m_exec_space(exec_space),
        m_forward_plan(exec_space, in, out, KokkosFFT::Direction::forward,
                       fft_dim),
        m_backward_plan(exec_space, out, in, KokkosFFT::Direction::backward,
                        fft_dim) {}

  /// \brief Set the local transpose of the input of the forward FFT.
  /// No op if the map is the identity.
  /// \param[in] map Mapping of the input of the FFT block
  void set_in_map(const map_type& map) {
    if (map == KokkosFFT::Impl::index_sequence<std::size_t, DIM, 0>()) return;
    for (std::size_t i = 0; i < DIM; ++i) {
      m_map_forward_in.at(i)   = map.at(i);
      m_map_backward_out.at(i) = KokkosFFT::Impl::get_index(map, i);
    }
  }

  /// \brief Set the local transpose of the output of the forward FFT.
  /// No op if the map is the identity.
  /// \param[in] map Mapping of the output of the FFT block
  void set_out_map(const map_type& map) {
    if (map == KokkosFFT::Impl::index_sequence<std::size_t, DIM, 0>()) return;
    for (std::size_t i = 0; i < DIM; ++i) {
      m_map_forward_out.at(i) = KokkosFFT::Impl::get_index(map, i);
      m_map_backward_in.at(i) = map.at(i);
    }
  }

  /// \brief Whether the output of the forward FFT is locally transposed
  bool has_out_map() const { return m_map_forward_out != int_map_type{}; }

  /// \brief Forward FFT with the local transposes
  /// \param[in] in Input view
  /// \param[out] out Output view
  /// \param[in] in_T Buffer view for the local transpose of the input
  /// \param[in] out_T Buffer view for the local transpose of the output
  void forward(const InViewType& in, const OutViewType& out,
               const InViewType& in_T, const OutViewType& out_T) const {
    const bool transpose_in  = m_map_forward_in != int_map_type{};
    const bool transpose_out = m_map_forward_out != int_map_type{};

    if (transpose_in) {
      KokkosFFT::Impl::transpose(m_exec_space, in, in_T, m_map_forward_in,
                                 true);
    }
    KokkosFFT::execute(m_forward_plan, transpose_in ? in_T : in,
                       transpose_out ? out_T : out,
                       KokkosFFT::Normalization::none);
    if (transpose_out) {
      KokkosFFT::Impl::transpose(m_exec_space, out_T, out, m_map_forward_out,
                                 true);
    }
  }

  /// \brief Backward FFT with the local transposes
  /// \param[in] out Input view of the backward FFT
  /// \param[out] in Output view of the backward FFT
  /// \param[in] in_T Buffer view for the local transpose of the output
  /// \param[in] out_T Buffer view for the local transpose of the input
  void backward(const OutViewType& out, const InViewType& in,
                const InViewType& in_T, const OutViewType& out_T) const {
    const bool transpose_in  = m_map_backward_in != int_map_type{};
    const bool transpose_out = m_map_backward_out != int_map_type{};

    if (transpose_in) {
      KokkosFFT::Impl::transpose(m_exec_space, out, out_T, m_map_backward_in,
                                 true);
    }
    KokkosFFT::execute(m_backward_plan, transpose_in ? out_T : out,
                       transpose_out ? in_T : in,
                       KokkosFFT::Normalization::none);
    if (transpose_out) {
      KokkosFFT::Impl::transpose(m_exec_space, in_T, in, m_map_backward_out,
                                 true);
    }
  }

  /// \brief Workspace size required by the forward and backward FFTs
  /// \tparam ScalarType The scalar type used to allocate the workspace
  /// \return The workspace size in ScalarType
  template <typename ScalarType>
  std::size_t workspace_size() const {
    return KokkosFFT::compute_required_workspace_size<ScalarType>(
        m_forward_plan, m_backward_plan);
  }

  /// \brief Set the workspace of the forward and backward FFTs
  /// \param[in] work Workspace view
  template <typename WorkViewType>
  void set_work_area(const WorkViewType& work) {
    m_forward_plan.set_work_area(work);
    m_backward_plan.set_work_area(work);
  }
};

/// \brief Allocate the FFT workspace(s) and set them to the FFT stages.
/// With SYCL, each stage has its own workspace. Otherwise, a single
/// workspace large enough for all the stages is shared.
///
/// \tparam ScalarType The scalar type used to allocate the workspace
/// \tparam AllocationViewType The type of the workspace views
/// \tparam FirstStageType The type of the first FFT stage
/// \tparam StageType The type of the other FFT stages
///
/// \param[in,out] first_stage The first FFT stage
/// \param[in,out] stages The other FFT stages
/// \return The allocated workspace(s)
template <typename ScalarType, typename AllocationViewType,
          typename FirstStageType, typename StageType = FirstStageType>
std::vector<AllocationViewType> allocate_fft_work_areas(
    FirstStageType& first_stage,
    const std::vector<std::unique_ptr<StageType>>& stages = {}) {
  std::vector<AllocationViewType> allocations;
#if defined(KOKKOS_ENABLE_SYCL)
  allocations.emplace_back("fft_buffer_allocation",
                           first_stage.template workspace_size<ScalarType>());
  first_stage.set_work_area(allocations.back());
  for (std::size_t i = 0; i < stages.size(); ++i) {
    allocations.emplace_back(
        "fft_buffer_allocation" + std::to_string(i + 1),
        stages.at(i)->template workspace_size<ScalarType>());
    stages.at(i)->set_work_area(allocations.back());
  }
#else
  std::size_t workspace_size =
      first_stage.template workspace_size<ScalarType>();
  for (const auto& stage : stages) {
    workspace_size =
        std::max(workspace_size, stage->template workspace_size<ScalarType>());
  }
  allocations.emplace_back("fft_buffer_allocation", workspace_size);
  first_stage.set_work_area(allocations.back());
  for (const auto& stage : stages) {
    stage->set_work_area(allocations.back());
  }
#endif
  return allocations;
}

/// \brief Base class of the distributed FFT plans
/// \tparam ExecutionSpace Kokkos execution space
/// \tparam InViewType Input Kokkos view type
/// \tparam OutViewType Output Kokkos view type
/// \tparam DIM Rank of FFT (default: 1)
/// \tparam InLayoutType Layout of input topology (default: LayoutRight)
/// \tparam OutLayoutType Layout of output topology (default: LayoutRight)
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1, typename InLayoutType = Kokkos::LayoutRight,
          typename OutLayoutType = Kokkos::LayoutRight>
class BasePlan {
  using axes_type        = KokkosFFT::axis_type<DIM>;
  using extents_type     = KokkosFFT::shape_type<InViewType::rank()>;
  using fft_extents_type = KokkosFFT::shape_type<DIM>;
  using in_topology_type =
      KokkosFFT::Distributed::Topology<std::size_t, InViewType::rank(),
                                       InLayoutType>;
  using out_topology_type =
      KokkosFFT::Distributed::Topology<std::size_t, InViewType::rank(),
                                       OutLayoutType>;

  const extents_type m_in_extents, m_out_extents;

 protected:
  //! Normalization
  KokkosFFT::Normalization m_norm;

  //! FFT size
  fft_extents_type m_fft_extents;

 public:
  /// \brief BasePlan constructor
  /// \param[in] exec_space Kokkos execution space
  /// \param[in] in Input Kokkos view
  /// \param[in] out Output Kokkos view
  /// \param[in] axes FFT axes
  /// \param[in] in_topology Input topology in std::array
  /// \param[in] out_topology Output topology in std::array
  /// \param[in] comm MPI communicator
  explicit BasePlan(const ExecutionSpace& exec_space, const InViewType& in,
                    const OutViewType& out, const axes_type& axes,
                    const extents_type& in_topology,
                    const extents_type& out_topology, const MPI_Comm& comm,
                    KokkosFFT::Normalization norm)
      : BasePlan(exec_space, in, out, axes, in_topology, out_topology, comm,
                 norm) {}

  /// \brief BasePlan constructor
  /// \param[in] exec_space Kokkos execution space
  /// \param[in] in Input Kokkos view
  /// \param[in] out Output Kokkos view
  /// \param[in] axes FFT axes
  /// \param[in] in_topology Input topology
  /// \param[in] out_topology Output topology
  /// \param[in] comm MPI communicator
  explicit BasePlan(const ExecutionSpace& /*exec_space*/, const InViewType& in,
                    const OutViewType& out, const axes_type& axes,
                    const in_topology_type& in_topology,
                    const out_topology_type& out_topology, const MPI_Comm& comm,
                    KokkosFFT::Normalization norm)
      : m_in_extents(KokkosFFT::Impl::extract_extents(in)),
        m_out_extents(KokkosFFT::Impl::extract_extents(out)),
        m_norm(norm) {
    auto gin_extents  = compute_global_extents(in, in_topology, comm);
    auto gout_extents = compute_global_extents(out, out_topology, comm);
    auto non_negative_axes =
        KokkosFFT::Impl::convert_base_int_type<std::size_t>(
            KokkosFFT::Impl::convert_negative_axes(axes, InViewType::rank()));

    m_fft_extents =
        compute_fft_extents(gin_extents, gout_extents, non_negative_axes);
  }

  virtual ~BasePlan() = default;

  /// \brief Forward FFT operation
  /// \param[in] Input view
  /// \param[out] Output view
  virtual void forward(const InViewType& in, const OutViewType& out) const = 0;

  /// \brief Backward FFT operation
  /// \param[out] Output view
  /// \param[in] Input view
  virtual void backward(const OutViewType& out, const InViewType& in) const = 0;

  /// \brief Get the name of the plan implementation
  /// \return Name of the plan implementation
  virtual std::string name() const = 0;

 protected:
  KokkosFFT::Normalization get_norm() const { return m_norm; }
  auto get_fft_extents() const { return m_fft_extents; }

  void good(const InViewType& in, const OutViewType& out) const {
    auto in_extents  = KokkosFFT::Impl::extract_extents(in);
    auto out_extents = KokkosFFT::Impl::extract_extents(out);

    auto mismatched_extents = [&](extents_type extents,
                                  extents_type plan_extents) -> std::string {
      std::string message;
      message += "View (";
      message += std::to_string(extents.at(0));
      for (std::size_t r = 1; r < extents.size(); r++) {
        message += ",";
        message += std::to_string(extents.at(r));
      }
      message += "), ";
      message += "Plan (";
      message += std::to_string(plan_extents.at(0));
      for (std::size_t r = 1; r < plan_extents.size(); r++) {
        message += ",";
        message += std::to_string(plan_extents.at(r));
      }
      message += ")";
      return message;
    };

    KOKKOSFFT_THROW_IF(in_extents != m_in_extents,
                       "extents of input View for plan and "
                       "execution are not identical: " +
                           mismatched_extents(in_extents, m_in_extents));

    KOKKOSFFT_THROW_IF(out_extents != m_out_extents,
                       "extents of output View for plan and "
                       "execution are not identical: " +
                           mismatched_extents(out_extents, m_out_extents));
  }
};

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
