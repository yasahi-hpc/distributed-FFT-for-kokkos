#ifndef KOKKOSFFT_DISTRIBUTED_INTERNAL_PLAN_HPP
#define KOKKOSFFT_DISTRIBUTED_INTERNAL_PLAN_HPP

#include <string>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_Types.hpp"
#include "KokkosFFT_Distributed_MPI_Helper.hpp"
#include "KokkosFFT_Distributed_Utils.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1, typename InLayoutType = Kokkos::LayoutRight,
          typename OutLayoutType = Kokkos::LayoutRight>
class InternalPlan {
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
  explicit InternalPlan(const ExecutionSpace& exec_space, const InViewType& in,
                        const OutViewType& out, const axes_type& axes,
                        const extents_type& in_topology,
                        const extents_type& out_topology, const MPI_Comm& comm,
                        KokkosFFT::Normalization norm)
      : InternalPlan(exec_space, in, out, axes, in_topology, out_topology, comm,
                     norm) {}

  explicit InternalPlan(const ExecutionSpace& /*exec_space*/,
                        const InViewType& in, const OutViewType& out,
                        const axes_type& axes,
                        const in_topology_type& in_topology,
                        const out_topology_type& out_topology,
                        const MPI_Comm& comm, KokkosFFT::Normalization norm)
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

  virtual ~InternalPlan() = default;

  /// \brief Forward FFT operation
  /// \param[in] Input view
  /// \param[out] Output view
  virtual void forward(const InViewType& in, const OutViewType& out) const = 0;

  /// \brief Backward FFT operation
  /// \param[out] Output view
  /// \param[in] Input view
  virtual void backward(const OutViewType& out, const InViewType& in) const = 0;

  virtual std::string label() const = 0;

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
