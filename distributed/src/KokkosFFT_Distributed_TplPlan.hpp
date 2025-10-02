#ifndef KOKKOSFFT_DISTRIBUTED_TPL_PLAN_HPP
#define KOKKOSFFT_DISTRIBUTED_TPL_PLAN_HPP

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_InternalPlan.hpp"

#if defined(ENABLE_TPL_CUFFT_MP)
#include "KokkosFFT_Distributed_CufftMp_Types.hpp"
#include "KokkosFFT_Distributed_CufftMp_Plan.hpp"
#include "KokkosFFT_Distributed_CufftMp_Transform.hpp"
#endif

#if defined(ENABLE_TPL_ROCFFT_MPI)
#include "KokkosFFT_Distributed_RocfftMPI_Types.hpp"
#include "KokkosFFT_Distributed_RocfftMPI_Plan.hpp"
#include "KokkosFFT_Distributed_RocfftMPI_Transform.hpp"
#endif

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1, typename InLayoutType = Kokkos::LayoutRight,
          typename OutLayoutType = Kokkos::LayoutRight>
class TplPlan : public InternalPlan<ExecutionSpace, InViewType, OutViewType,
                                    DIM, InLayoutType, OutLayoutType> {
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
  //! extents of in/out views for plan
  extents_type m_in_mapped_extents, m_out_mapped_extents;
  ///@}

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
      : TplPlan(exec_space, in, out, axes,
                Topology<std::size_t, InViewType::rank()>(in_topology),
                Topology<std::size_t, OutViewType::rank()>(out_topology), comm,
                norm) {}

  explicit TplPlan(
      const ExecutionSpace& exec_space, const InViewType& in,
      const OutViewType& out, const axes_type& axes,
      const in_topology_type& in_topology,
      const out_topology_type& out_topology, const MPI_Comm& comm,
      KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward)
      : InternalPlan<ExecutionSpace, InViewType, OutViewType, DIM>(
            exec_space, in, out, axes, in_topology, out_topology, comm, norm),
        m_exec_space(exec_space),
        m_in_extents(KokkosFFT::Impl::extract_extents(in)),
        m_out_extents(KokkosFFT::Impl::extract_extents(out)) {
    // We need to construct the mapping based on the LayoutRight view
    // The LayoutLeft view is converted to LayoutRight view via deep_copy
    // or safe_transpose in exec_plan
    using in_data_type =
        KokkosFFT::Impl::add_pointer_n_t<in_value_type, InViewType::rank()>;
    using execution_space_type = typename InViewType::execution_space;
    using InRightViewType =
        Kokkos::View<in_data_type, Kokkos::LayoutRight, execution_space_type>;
    InRightViewType in_right(
        in.data(),
        KokkosFFT::Impl::create_layout<Kokkos::LayoutRight>(m_in_extents));
    std::tie(m_in_map, m_out_map) =
        KokkosFFT::Impl::get_map_axes(in_right, axes);

    auto non_negative_axes =
        convert_negative_axes<std::size_t, int, DIM, DIM>(axes);

    auto gin_extents        = get_global_shape(in, in_topology, comm);
    auto gout_extents       = get_global_shape(out, out_topology, comm);
    auto gin_padded_extents = gin_extents;
    if (KokkosFFT::Impl::is_real_v<in_value_type>) {
      gin_padded_extents = get_padded_extents(gout_extents, non_negative_axes);
    }

    auto [in_extents, in_starts] =
        get_local_extents(gin_padded_extents, in_topology, comm);
    auto [out_extents, out_starts] =
        get_local_extents(gout_extents, out_topology, comm);

    m_in_mapped_extents  = get_mapped_extents(in_extents, m_in_map);
    m_out_mapped_extents = get_mapped_extents(out_extents, m_in_map);

    // Calling setup function
    using float_type = KokkosFFT::Impl::base_floating_point_type<in_value_type>;
    KokkosFFT::Impl::setup<ExecutionSpace, float_type>();

    // Only support 2D or 3D FFTs
    m_fft_size = create_plan(m_exec_space, m_plan, in, out, axes, m_in_map,
                             in_topology.array(), out_topology.array(), comm);
  }

  ~TplPlan() noexcept = default;

  TplPlan()                          = delete;
  TplPlan(const TplPlan&)            = delete;
  TplPlan& operator=(const TplPlan&) = delete;
  TplPlan& operator=(TplPlan&&)      = delete;
  TplPlan(TplPlan&&)                 = delete;

  void forward(const InViewType& in, const OutViewType& out) const override {
    good(in, out);
    execute_impl(*m_plan, in, out, m_in_mapped_extents, m_out_mapped_extents,
                 m_in_map, m_out_map, KokkosFFT::Direction::forward);
  }

  void backward(const OutViewType& out, const InViewType& in) const override {
    good(in, out);
    execute_impl(*m_plan, out, in, m_out_mapped_extents, m_in_mapped_extents,
                 m_in_map, m_out_map, KokkosFFT::Direction::backward);
  }

  std::string label() const override { return m_plan->label(); }

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
