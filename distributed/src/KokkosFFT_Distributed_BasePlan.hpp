#ifndef KOKKOSFFT_DISTRIBUTED_BASE_PLAN_HPP
#define KOKKOSFFT_DISTRIBUTED_BASE_PLAN_HPP

#include <array>
#include <string>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_Types.hpp"
#include "KokkosFFT_Distributed_MPI_Extents.hpp"
#include "KokkosFFT_Distributed_Extents.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief Base class of the distributed FFT plans
///
/// In-place transforms: a plan is in-place when its input and output Views
/// point to the same data (in.data() == out.data()), as in KokkosFFT. Each
/// View keeps the extents of the out-of-place transform and is read or
/// written densely in them. The input and output Views must either start at
/// the same address or not overlap at all; a partial overlap throws. A plan
/// must be executed in the placement it was created for.
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

  //! Whether the input and output Views point to the same data
  const bool m_is_inplace;

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
      : BasePlan(exec_space, in, out, axes, in_topology_type(in_topology),
                 out_topology_type(out_topology), comm, norm) {}

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
        m_is_inplace(check_placement(in, out)),
        m_norm(norm) {
    auto gin_extents  = compute_global_extents(in, in_topology, comm);
    auto gout_extents = compute_global_extents(out, out_topology, comm);
    auto non_negative_axes =
        KokkosFFT::Impl::convert_base_int_type<std::size_t>(
            KokkosFFT::Impl::convert_negative_axes(axes, InViewType::rank()));

    m_fft_extents =
        compute_fft_extents(gin_extents, gout_extents, non_negative_axes);
  }

  // Not copyable, not movable: the derived plans are built in place and
  // held through std::unique_ptr<BasePlan> in Plan.
  BasePlan()                           = delete;
  BasePlan(const BasePlan&)            = delete;
  BasePlan& operator=(const BasePlan&) = delete;
  BasePlan(BasePlan&&)                 = delete;
  BasePlan& operator=(BasePlan&&)      = delete;
  virtual ~BasePlan()                  = default;

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

  /// \brief Whether the plan is in-place: its input and output Views point
  /// to the same data
  bool is_inplace() const { return m_is_inplace; }

 protected:
  KokkosFFT::Normalization get_norm() const { return m_norm; }
  auto get_fft_extents() const { return m_fft_extents; }

  /// \brief Whether the input and output Views point to the same data. Views
  /// without data (empty) never do.
  /// \throws std::runtime_error if the Views overlap partially, i.e. they
  /// overlap but do not start at the same address
  static bool check_placement(const InViewType& in, const OutViewType& out) {
    const bool is_aliasing =
        in.data() != nullptr &&
        KokkosFFT::Impl::are_aliasing(in.data(), out.data());
    KOKKOSFFT_THROW_IF(!is_aliasing && are_overlapping(in, out),
                       "the input and output Views overlap partially: they "
                       "must either point to the same data (in-place) or not "
                       "overlap (out-of-place)");
    return is_aliasing;
  }

  void good(const InViewType& in, const OutViewType& out) const {
    const bool is_inplace = check_placement(in, out);
    KOKKOSFFT_THROW_IF(
        is_inplace != m_is_inplace,
        m_is_inplace ? "the plan is in-place: the input and output Views "
                       "must point to the same data"
                     : "the plan is out-of-place: the input and output Views "
                       "must not point to the same data");

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
