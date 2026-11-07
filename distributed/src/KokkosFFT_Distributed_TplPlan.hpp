#ifndef KOKKOSFFT_DISTRIBUTED_TPL_PLAN_HPP
#define KOKKOSFFT_DISTRIBUTED_TPL_PLAN_HPP

#include <memory>
#include <type_traits>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_Extents.hpp"
#include "KokkosFFT_Distributed_BasePlan.hpp"
#include "KokkosFFT_Distributed_TplAxes.hpp"

#if defined(ENABLE_TPL_CUFFT_MP)
#include "cuFFTMp/KokkosFFT_Distributed_CufftMp_Types.hpp"
#include "cuFFTMp/KokkosFFT_Distributed_CufftMp_Plan.hpp"
#include "cuFFTMp/KokkosFFT_Distributed_CufftMp_Transform.hpp"
#endif

#if defined(ENABLE_TPL_ROCFFT_MPI)
#include "rocFFT/KokkosFFT_Distributed_RocfftMPI_Types.hpp"
#include "rocFFT/KokkosFFT_Distributed_RocfftMPI_Plan.hpp"
#include "rocFFT/KokkosFFT_Distributed_RocfftMPI_Transform.hpp"
#endif

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief Whether a vendor plan type applies the normalization itself (in the
/// transform, or fused into a copy of its execution), so that TplPlan must
/// not normalize again: PlanType::scales_in_transform, false if absent
template <typename PlanType, typename = void>
struct tpl_scales_in_transform : std::false_type {};

template <typename PlanType>
struct tpl_scales_in_transform<
    PlanType, std::void_t<decltype(PlanType::scales_in_transform)>>
    : std::bool_constant<PlanType::scales_in_transform> {};

template <typename PlanType>
inline constexpr bool tpl_scales_in_transform_v =
    tpl_scales_in_transform<PlanType>::value;

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1, typename InLayoutType = Kokkos::LayoutRight,
          typename OutLayoutType = Kokkos::LayoutRight>
class TplPlan : public BasePlan<ExecutionSpace, InViewType, OutViewType, DIM,
                                InLayoutType, OutLayoutType> {
 private:
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;
  using axes_type      = KokkosFFT::axis_type<DIM>;
  using extents_type   = KokkosFFT::shape_type<InViewType::rank()>;
  using in_topology_type =
      Topology<std::size_t, InViewType::rank(), InLayoutType>;
  using out_topology_type =
      Topology<std::size_t, OutViewType::rank(), OutLayoutType>;
  using map_type = KokkosFFT::axis_type<InViewType::rank()>;

  //! The base class, with the layouts of the topologies
  using base_type = BasePlan<ExecutionSpace, InViewType, OutViewType, DIM,
                             InLayoutType, OutLayoutType>;

  //! The type of fft plan
  using fft_plan_type =
      typename InternalTplPlanType<ExecutionSpace, in_value_type,
                                   out_value_type>::type;

  //! The real value type for normalization
  using normalization_float_type = double;

  //! LayoutRight Views on the memory of the in/out Views, with the extents
  //! reversed (reversed_view): the vendor plans run on them for LayoutLeft
  //! Views when that needs no transpose (is_tpl_reversed)
  using execution_space_type = typename InViewType::execution_space;
  using InReversedViewType   = Kokkos::View<
      KokkosFFT::Impl::add_pointer_n_t<in_value_type, InViewType::rank()>,
      Kokkos::LayoutRight, execution_space_type,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using OutReversedViewType = Kokkos::View<
      KokkosFFT::Impl::add_pointer_n_t<out_value_type, OutViewType::rank()>,
      Kokkos::LayoutRight, execution_space_type,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  static constexpr bool m_is_layout_left =
      std::is_same_v<typename InViewType::array_layout, Kokkos::LayoutLeft>;

 private:
  //! Execution space
  ExecutionSpace m_exec_space;

  //! Dynamically allocatable fft plan.
  std::unique_ptr<fft_plan_type> m_plan;

  //! maps for forward and backward transpose
  map_type m_in_map, m_out_map;

  ///@{
  //! extents of in/out views for plan
  extents_type m_in_mapped_extents, m_out_mapped_extents;
  ///@}

  //! Whether the plan runs on the reversed LayoutRight Views of LayoutLeft
  //! Views (no copy), rather than on the Views as they are
  bool m_is_reversed = false;

  using base_type::get_fft_extents;
  using base_type::get_norm;
  using base_type::good;

 public:
  explicit TplPlan(
      const ExecutionSpace& exec_space, const InViewType& in,
      const OutViewType& out, const axes_type& axes,
      const extents_type& in_topology, const extents_type& out_topology,
      const MPI_Comm& comm,
      KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward)
      : TplPlan(exec_space, in, out, axes, in_topology_type(in_topology),
                out_topology_type(out_topology), comm, norm) {}

  explicit TplPlan(
      const ExecutionSpace& exec_space, const InViewType& in,
      const OutViewType& out, const axes_type& axes,
      const in_topology_type& in_topology,
      const out_topology_type& out_topology, const MPI_Comm& comm,
      KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward)
      : base_type(exec_space, in, out, axes, in_topology, out_topology, comm,
                  norm),
        m_exec_space(exec_space) {
#if defined(ENABLE_TPL_ROCFFT_MPI)
    // The rocFFT plan is created out-of-place (rocfft_placement_notinplace)
    // and, without transpose, executed directly on the input and output
    // Views, which must then not share their data. plan_factory uses the
    // internal plans for in-place transforms instead (is_tpl_available).
    KOKKOSFFT_THROW_IF(this->is_inplace(),
                       "TplPlan: in-place transforms are not supported with "
                       "rocFFT");
#endif

    // Calling setup function
    using float_type = KokkosFFT::Impl::base_floating_point_type<in_value_type>;
    setup<ExecutionSpace, float_type>(comm);

    // A LayoutLeft View is the memory of a LayoutRight View with the extents
    // reversed: the plan is made for that View (with the axes and the
    // topologies reversed) when it needs no transpose there. Otherwise, it is
    // made for the Views as they are, and the execution goes through a copy
    // that changes the layout (and transposes if the axes need it).
    if constexpr (m_is_layout_left) {
      m_is_reversed = is_tpl_reversed<InViewType>(axes);
    }
    if (m_is_reversed) {
      if constexpr (m_is_layout_left) {
        init(reversed_view<InReversedViewType>(in),
             reversed_view<OutReversedViewType>(out),
             reversed_tpl_axes(axes, InViewType::rank()),
             reversed_topology(in_topology), reversed_topology(out_topology),
             comm);
      }
    } else {
      init(in, out, axes, in_topology, out_topology, comm);
    }
  }

  ~TplPlan() noexcept = default;

  TplPlan()                          = delete;
  TplPlan(const TplPlan&)            = delete;
  TplPlan& operator=(const TplPlan&) = delete;
  TplPlan& operator=(TplPlan&&)      = delete;
  TplPlan(TplPlan&&)                 = delete;

  void forward(const InViewType& in, const OutViewType& out) const override {
    good(in, out);
    if constexpr (m_is_layout_left) {
      if (m_is_reversed) {
        execute_impl(*m_plan, reversed_view<InReversedViewType>(in),
                     reversed_view<OutReversedViewType>(out),
                     m_in_mapped_extents, m_out_mapped_extents, m_in_map,
                     m_out_map, KokkosFFT::Direction::forward);
        return;
      }
    }
    execute_impl(*m_plan, in, out, m_in_mapped_extents, m_out_mapped_extents,
                 m_in_map, m_out_map, KokkosFFT::Direction::forward);
  }

  void backward(const OutViewType& out, const InViewType& in) const override {
    good(in, out);
    if constexpr (m_is_layout_left) {
      if (m_is_reversed) {
        execute_impl(*m_plan, reversed_view<OutReversedViewType>(out),
                     reversed_view<InReversedViewType>(in),
                     m_out_mapped_extents, m_in_mapped_extents, m_in_map,
                     m_out_map, KokkosFFT::Direction::backward);
        return;
      }
    }
    execute_impl(*m_plan, out, in, m_out_mapped_extents, m_in_mapped_extents,
                 m_in_map, m_out_map, KokkosFFT::Direction::backward);
  }

  /// \brief Whether the plan runs on the reversed LayoutRight Views of
  /// LayoutLeft Views, without a copy (for the tests)
  bool is_reversed() const noexcept { return m_is_reversed; }

  /// \brief Get the name of the plan implementation
  /// \return Name of the plan implementation
  std::string name() const override { return m_plan->name(); }

 private:
  /// \brief A LayoutRight View on the memory of view, with the extents
  /// reversed: the same data as the LayoutLeft View view
  /// \tparam ReversedViewType The LayoutRight unmanaged View type
  /// \tparam ViewType The View type
  /// \param[in] view The View
  template <typename ReversedViewType, typename ViewType>
  static ReversedViewType reversed_view(const ViewType& view) {
    return ReversedViewType(
        view.data(),
        KokkosFFT::Impl::create_layout<Kokkos::LayoutRight>(
            reversed_extents(KokkosFFT::Impl::extract_extents(view))));
  }

  /// \brief Make the vendor plan for in and out (the Views, or their reversed
  /// Views), the axes and the topologies in the same space
  /// \param[in] in Input View
  /// \param[in] out Output View
  /// \param[in] axes Axes of the FFT
  /// \param[in] in_topology Topology of the input
  /// \param[in] out_topology Topology of the output
  /// \param[in] comm MPI communicator
  template <typename InView, typename OutView, typename InTopologyType,
            typename OutTopologyType>
  void init(const InView& in, const OutView& out, const axes_type& axes,
            const InTopologyType& in_topology,
            const OutTopologyType& out_topology, const MPI_Comm& comm) {
    // The vendor plan is made with the axes in canonical order, which gives
    // the same result: no transpose when only their order differs. The base
    // class keeps the given axes (checks, normalization). The map is in the
    // LayoutRight convention: the vendor plans describe the data in the
    // LayoutRight order of its indices; a LayoutLeft View that is not
    // reversed is converted by the copy in exec_plan.
    const auto tpl_axes =
        canonical_tpl_axes<in_value_type>(axes, InViewType::rank());
    std::tie(m_in_map, m_out_map) =
        KokkosFFT::Impl::get_map_axes<Kokkos::LayoutRight, InViewType::rank()>(
            tpl_axes);

    auto gin_extents        = compute_global_extents(in, in_topology, comm);
    auto gout_extents       = compute_global_extents(out, out_topology, comm);
    auto gin_padded_extents = gin_extents;

#if defined(ENABLE_TPL_CUFFT_MP)
    if (KokkosFFT::Impl::is_real_v<in_value_type>) {
      auto non_negative_axes =
          KokkosFFT::Impl::convert_base_int_type<std::size_t>(tpl_axes);
      gin_padded_extents =
          KokkosFFT::Impl::compute_padded_extents<in_value_type>(
              gout_extents, non_negative_axes);
    }
#endif

    auto [in_extents, in_starts] =
        compute_local_extents_and_starts(gin_padded_extents, in_topology, comm);
    auto [out_extents, out_starts] =
        compute_local_extents_and_starts(gout_extents, out_topology, comm);

    m_in_mapped_extents =
        KokkosFFT::Impl::compute_mapped_extents(in_extents, m_in_map);
    m_out_mapped_extents =
        KokkosFFT::Impl::compute_mapped_extents(out_extents, m_in_map);

    // The topologies keep their layout (which rank owns which block)
    create_plan(m_exec_space, m_plan, in, out, tpl_axes, m_in_map, in_topology,
                out_topology, comm);
  }

  template <typename PlanType, typename InView, typename OutView>
  void execute_impl(const PlanType& plan, const InView& in, const OutView& out,
                    const extents_type& in_extents,
                    const extents_type& out_extents, const map_type& in_map,
                    const map_type& out_map,
                    KokkosFFT::Direction direction) const {
    if constexpr (tpl_scales_in_transform_v<PlanType>) {
      // The plan normalizes its output (cuFFTMp: in its copy-out)
      auto [coef, to_normalize] =
          KokkosFFT::Impl::get_coefficients<normalization_float_type>(
              direction, get_norm(), get_fft_extents());
      exec_plan(m_exec_space, plan, in, out, in_extents, out_extents, in_map,
                out_map, direction,
                to_normalize ? coef : normalization_float_type(1));
    } else {
      exec_plan(m_exec_space, plan, in, out, in_extents, out_extents, in_map,
                out_map, direction);
      KokkosFFT::Impl::normalize<normalization_float_type>(
          m_exec_space, out, direction, get_norm(), get_fft_extents());
    }
  }
};

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
