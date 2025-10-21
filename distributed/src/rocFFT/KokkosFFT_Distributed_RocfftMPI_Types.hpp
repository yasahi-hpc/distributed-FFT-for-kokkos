#ifndef KOKKOSFFT_DISTRIBUTED_ROCFFT_MPI_TYPES_HPP
#define KOKKOSFFT_DISTRIBUTED_ROCFFT_MPI_TYPES_HPP

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief A class that wraps rocfft_field for RAII
struct ScopedRocfftField {
 private:
  rocfft_field m_field = nullptr;

 public:
  /// \param[in] field_lower (start0, start1, ..., start_batch)
  /// \param[in] field_upper (end0, end1, ..., end_batch)
  /// \param[in] brick_stride (stride0, stride1, ..., stride_batch)
  /// \param[in] batch_size
  /// \param[in] deviceID
  ScopedRocfftField(const std::vector<std::size_t> &field_lower,
                    const std::vector<std::size_t> &field_upper,
                    const std::vector<std::size_t> &brick_stride,
                    int deviceID) {
    rocfft_status status = rocfft_field_create(&m_field);
    KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                       "rocfft_field_create failed");

    std::vector<std::size_t> lower   = field_lower;
    std::vector<std::size_t> upper   = field_upper;
    std::vector<std::size_t> strides = brick_stride;
    lower.push_back(0);
    upper.push_back(batch_size);
    strides.push_back(0);

    // Strides would be compued by
    // (1, extents[n-1], strides[1] * extents[n-2], ...)
    rocfft_brick brick = nullptr;
    status             = rocfft_brick_create(&brick, lower.data(), upper.data(),
                                             strides.data(), lower.size(), deviceID);

    KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                       "rocfft_brick_create failed");
    status = rocfft_field_add_brick(m_field, brick);
    KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                       "rocfft_field_add_brick failed");
    status = rocfft_brick_destroy(brick);
    brick  = nullptr;
    KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                       "rocfft_brick_destroy failed");
  }
  ~ScopedRocfftField() noexcept {
    rocfft_status status = rocfft_field_destroy(m_field);
    if (status != rocfft_status_success)
      Kokkos::abort("rocfft_field_destroy failed");
  }
  ScopedRocfftField(const ScopedRocfftField &)            = delete;
  ScopedRocfftField &operator=(const ScopedRocfftField &) = delete;
  ScopedRocfftField &operator=(ScopedRocfftField &&)      = delete;
  ScopedRocfftField(ScopedRocfftField &&)                 = delete;

  rocfft_field field() const noexcept { return m_field; }
};

/// \brief A class that wraps rocFFT plan with MPI
template <typename ExecutionSpace, typename T1, typename T2>
struct ScopedRocfftMPIPlan {
 private:
  using floating_point_type    = KokkosFFT::Impl::base_floating_point_type<T1>;
  rocfft_precision m_precision = std::is_same_v<floating_point_type, float>
                                     ? rocfft_precision_single
                                     : rocfft_precision_double;
  rocfft_plan m_plan;
  rocfft_plan_description m_description;
  MPI_Comm m_comm;

 public:
  ScopedRocfftMPIPlan(const std::vector<std::size_t> &length,
                      const std::vector<std::size_t> &lower_input,
                      const std::vector<std::size_t> &upper_input,
                      const std::vector<std::size_t> &lower_output,
                      const std::vector<std::size_t> &upper_output,
                      const std::vector<std::size_t> &strides_input,
                      const std::vector<std::size_t> &strides_output,
                      KokkosFFT::Direction direction, const MPI_Comm &comm)
      : m_comm(comm) {
    rocfft_status status = rocfft_plan_description_create(&m_description);
    KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                       "rocfft_plan_description_create failed");

    status = rocfft_plan_description_set_comm(m_description, rocfft_comm_mpi,
                                              &m_comm);
    KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                       "rocfft_plan_description_set_comm failed");

    // Do not set stride information via the descriptor, they are to be defined
    // during field creation below
    constexpr auto transform_type =
        KokkosFFT::Impl::transform_type<ExecutionSpace, T1, T2>::type();
    auto [in_array_type, out_array_type, fft_direction] =
        get_in_out_array_type(transform_type, direction);
    status = rocfft_plan_description_set_data_layout(
        m_description, in_array_type, out_array_type, nullptr, nullptr, 0,
        nullptr, 0, 0, nullptr, 0);
    KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                       "rocfft_plan_description_set_data_layout failed");

    ScopedRocfftField scoped_infield(lower_input, upper_input, strides_input, 1,
                                     0);
    ScopedRocfftField scoped_outfield(lower_output, upper_output,
                                      strides_output, 1, 0);
    status = rocfft_plan_description_add_infield(m_description,
                                                 scoped_infield.field());
    KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                       "rocfft_plan_description_add_infield failed");

    status = rocfft_plan_description_add_outfield(m_description,
                                                  scoped_outfield.field());
    KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                       "rocfft_plan_description_add_outfield failed");

    // inplace or Out-of-place transform
    const rocfft_result_placement place = rocfft_placement_notinplace;

    // Create a plan
    status = rocfft_plan_create(&m_plan, place, fft_direction, m_precision,
                                length.size(),   // Dimension
                                length.data(),   // Lengths
                                1,               // Number of transforms
                                m_description);  // Description
    KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                       "rocfft_plan_create failed");
  }

  ~ScopedRocfftMPIPlan() noexcept {
    rocfft_status status = rocfft_plan_description_destroy(m_description);
    if (status != rocfft_status_success)
      Kokkos::abort("rocfft_plan_description_destroy failed");

    status = rocfft_plan_destroy(m_plan);
    if (status != rocfft_status_success)
      Kokkos::abort("rocfft_plan_destroy failed");
  }

  ScopedRocfftMPIPlan()                                       = delete;
  ScopedRocfftMPIPlan(const ScopedRocfftMPIPlan &)            = delete;
  ScopedRocfftMPIPlan &operator=(const ScopedRocfftMPIPlan &) = delete;
  ScopedRocfftMPIPlan &operator=(ScopedRocfftMPIPlan &&)      = delete;
  ScopedRocfftMPIPlan(ScopedRocfftMPIPlan &&)                 = delete;

  rocfft_plan plan() const noexcept { return m_plan; }
  void commit(const Kokkos::HIP & /*exec_space*/) const {}

  // Helper to get input and output array type and direction from transform type
  auto get_in_out_array_type(KokkosFFT::Impl::FFTWTransformType type,
                             KokkosFFT::Direction direction) {
    rocfft_array_type in_array_type, out_array_type;
    rocfft_transform_type fft_direction;

    if (type == KokkosFFT::Impl::FFTWTransformType::C2C ||
        type == KokkosFFT::Impl::FFTWTransformType::Z2Z) {
      in_array_type  = rocfft_array_type_complex_interleaved;
      out_array_type = rocfft_array_type_complex_interleaved;
      fft_direction  = direction == KokkosFFT::Direction::forward
                           ? rocfft_transform_type_complex_forward
                           : rocfft_transform_type_complex_inverse;
    } else if (type == KokkosFFT::Impl::FFTWTransformType::R2C ||
               type == KokkosFFT::Impl::FFTWTransformType::D2Z) {
      in_array_type  = rocfft_array_type_real;
      out_array_type = rocfft_array_type_hermitian_interleaved;
      fft_direction  = rocfft_transform_type_real_forward;
    } else if (type == KokkosFFT::Impl::FFTWTransformType::C2R ||
               type == KokkosFFT::Impl::FFTWTransformType::Z2D) {
      in_array_type  = rocfft_array_type_hermitian_interleaved;
      out_array_type = rocfft_array_type_real;
      fft_direction  = rocfft_transform_type_real_inverse;
    }

    return std::tuple<rocfft_array_type, rocfft_array_type,
                      rocfft_transform_type>(
        {in_array_type, out_array_type, fft_direction});
  }
};

template <typename ExecutionSpace, typename T1, typename T2>
struct ScopedRocfftMPIBidirectionalPlan {
 private:
  using ScopedRocfftMPIForwardPlanType =
      ScopedRocfftMPIPlan<ExecutionSpace, T1, T2>;
  using ScopedRocfftMPIBackwardPlanType =
      ScopedRocfftMPIPlan<ExecutionSpace, T2, T1>;
  ScopedRocfftMPIForwardPlanType m_plan_forward;
  ScopedRocfftMPIBackwardPlanType m_plan_backward;

 public:
  ScopedRocfftMPIBidirectionalPlan(
      const std::vector<std::size_t> &length,
      const std::vector<std::size_t> &lower_input,
      const std::vector<std::size_t> &upper_input,
      const std::vector<std::size_t> &lower_output,
      const std::vector<std::size_t> &upper_output,
      const std::vector<std::size_t> &strides_input,
      const std::vector<std::size_t> &strides_output, const MPI_Comm &comm)
      : m_plan_forward(length, lower_input, upper_input, lower_output,
                       upper_output, strides_input, strides_output,
                       KokkosFFT::Direction::forward, comm),
        m_plan_backward(length, lower_output, upper_output, lower_input,
                        upper_input, strides_output, strides_input,
                        KokkosFFT::Direction::backward, comm) {}

  ScopedRocfftMPIBidirectionalPlan() = delete;
  ScopedRocfftMPIBidirectionalPlan(const ScopedRocfftMPIBidirectionalPlan &) =
      delete;
  ScopedRocfftMPIBidirectionalPlan &operator=(
      const ScopedRocfftMPIBidirectionalPlan &) = delete;
  ScopedRocfftMPIBidirectionalPlan &operator=(
      ScopedRocfftMPIBidirectionalPlan &&) = delete;
  ScopedRocfftMPIBidirectionalPlan(ScopedRocfftMPIBidirectionalPlan &&) =
      delete;

  rocfft_plan plan(KokkosFFT::Direction direction) const noexcept {
    return direction == KokkosFFT::Direction::forward ? m_plan_forward.plan()
                                                      : m_plan_backward.plan();
  }

  void commit(const Kokkos::HIP &exec_space) const {
    m_plan_forward.commit(exec_space);
    m_plan_backward.commit(exec_space);
  }

  std::string label() const { return std::string("rocFFTMPIPlan"); }
};

template <typename ExecutionSpace, typename T1, typename T2>
struct InternalTplPlanType {
  using type = ScopedRocfftMPIBidirectionalPlan<ExecutionSpace, T1, T2>;
};

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
