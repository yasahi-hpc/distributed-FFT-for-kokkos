#ifndef INTERNAL_PLAN_HPP
#define INTERNAL_PLAN_HPP

#include <string>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1>
class InternalPlan {
  using axes_type     = KokkosFFT::axis_type<DIM>;
  using topology_type = KokkosFFT::shape_type<InViewType::rank()>;
  using extents_type  = KokkosFFT::shape_type<InViewType::rank()>;

  const extents_type m_in_extents, m_out_extents;

 protected:
  //! Normalization
  KokkosFFT::Normalization m_norm;

 public:
  explicit InternalPlan(const ExecutionSpace& /*exec_space*/,
                        const InViewType& in, const OutViewType& out,
                        const axes_type& /*axes*/,
                        const topology_type& /*in_topology*/,
                        const topology_type& /*out_topology*/,
                        const MPI_Comm& /*comm*/, KokkosFFT::Normalization norm,
                        const bool /*is_same_order*/)
      : m_in_extents(KokkosFFT::Impl::extract_extents(in)),
        m_out_extents(KokkosFFT::Impl::extract_extents(out)),
        m_norm(norm) {}

  virtual ~InternalPlan() = default;

  /// \brief Forward FFT operation
  /// \param[in] Input view
  /// \param[out] Output view
  virtual void forward(const InViewType& in, const OutViewType& out) const = 0;

  /// \brief Backward FFT operation
  /// \param[out] Output view
  /// \param[in] Input view
  virtual void backward(const OutViewType& out, const InViewType& in) const = 0;

  virtual std::string get_label() const = 0;

 protected:
  KokkosFFT::Normalization get_norm() const { return m_norm; }

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

#endif
