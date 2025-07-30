#ifndef TPL_PLAN_HPP
#define TPL_PLAN_HPP

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "InternalPlan.hpp"

#if defined(ENABLE_TPL_CUFFT_MP)
#include "CufftMp_Types.hpp"
#include "CufftMp_Plan.hpp"
#include "CufftMp_Transform.hpp"
#endif

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1>
class TplPlan
    : public InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM> {
 private:
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;
  using axes_type      = KokkosFFT::axis_type<DIM>;
  using extents_type   = KokkosFFT::shape_type<InViewType::rank()>;
  using map_type       = KokkosFFT::axis_type<InViewType::rank()>;

  //! The type of fft plan
  using fft_plan_type =
      typename InternalTplPlanType<ExecutionSpace, in_value_type,
                                   out_value_type>::type;

 private:
  //! Execution space
  ExecutionSpace m_exec_space;

  //! Dynamically allocatable fft plan.
  std::unique_ptr<fft_plan_type> m_plan;

  //! maps for forward and backward transpose
  map_type m_in_map, m_out_map;

  ///@{
  //! extents of in/out views
  extents_type m_in_extents, m_out_extents;
  ///@}

  ///@{
  //! extents of in/out desc views
  extents_type m_in_desc_extents, m_out_desc_extents;
  ///@}

  std::string m_label;

  std::size_t m_fft_size;

  using InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM>::good;
  using InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM>::get_norm;

 public:
  explicit TplPlan(
      const ExecutionSpace& exec_space, const InViewType& in,
      const OutViewType& out, const axes_type& axes,
      const extents_type& in_topology, const extents_type& out_topology,
      const MPI_Comm& comm,
      KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward)
      : InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM>(
            exec_space, in, out, axes, in_topology, out_topology, comm, norm),
        m_exec_space(exec_space) {
    auto non_negative_axes =
        convert_negative_axes<std::size_t, int, DIM, DIM>(axes);

    m_in_extents  = KokkosFFT::Impl::extract_extents(in);
    m_out_extents = KokkosFFT::Impl::extract_extents(out);

    std::tie(m_in_map, m_out_map) = KokkosFFT::Impl::get_map_axes(in, axes);

    m_out_desc_extents = get_mapped_extents(m_out_extents, m_in_map);
    if constexpr (KokkosFFT::Impl::is_real_v<in_value_type>) {
      m_in_desc_extents = get_padded_extents(m_out_desc_extents);
    } else {
      m_in_desc_extents = get_mapped_extents(m_in_extents, m_in_map);
    }

    // Only support 2D or 3D FFTs
    m_fft_size = create_plan(m_exec_space, m_plan, in, out, axes, in_topology,
                             out_topology, comm);
  }

  ~TplPlan() noexcept = default;

  TplPlan()                          = delete;
  TplPlan(const TplPlan&)            = delete;
  TplPlan& operator=(const TplPlan&) = delete;
  TplPlan& operator=(TplPlan&&)      = delete;
  TplPlan(TplPlan&&)                 = delete;

  void forward(const InViewType& in, const OutViewType& out) const override {
    good(in, out);
    execute_impl(*m_plan, in, out, m_in_desc_extents, m_out_desc_extents,
                 m_in_map, m_out_map, KokkosFFT::Direction::forward);
  }

  void backward(const OutViewType& out, const InViewType& in) const override {
    good(in, out);
    execute_impl(*m_plan, out, in, m_out_desc_extents, m_in_desc_extents,
                 m_out_map, m_in_map, KokkosFFT::Direction::backward);
  }

  std::string get_label() const { return m_label; }

 private:
  template <typename PlanType, typename InView, typename OutView>
  void execute_impl(const PlanType& plan, const InView& in, const OutView& out,
                    const extents_type& in_extents,
                    const extents_type& out_extents, const map_type& in_map,
                    const map_type& out_map,
                    KokkosFFT::Direction direction) const {
    exec_plan(m_exec_space, plan, in, out, in_extents, out_extents, in_map,
              out_map, direction);
    KokkosFFT::Impl::normalize(m_exec_space, out, direction, get_norm(),
                               m_fft_size);
  }
};

#endif
