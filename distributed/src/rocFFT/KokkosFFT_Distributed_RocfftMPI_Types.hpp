#ifndef KOKKOSFFT_DISTRIBUTED_ROCFFT_MPI_TYPES_HPP
#define KOKKOSFFT_DISTRIBUTED_ROCFFT_MPI_TYPES_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>
#include <mpi.h>
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
  /// The brick has the FFT dimensions, fastest first, then the batch
  /// dimension (rocFFT: dim + 1 entries).
  /// \param[in] lower (start0, start1, ..., start_batch)
  /// \param[in] upper (end0, end1, ..., end_batch)
  /// \param[in] strides (stride0, stride1, ..., stride_batch), in elements
  /// \param[in] deviceID HIP device on which the brick is
  ScopedRocfftField(const std::vector<std::size_t> &lower,
                    const std::vector<std::size_t> &upper,
                    const std::vector<std::size_t> &strides, int deviceID) {
    KOKKOSFFT_CHECK_ROCFFT_CALL(rocfft_field_create(&m_field));

    rocfft_brick brick = nullptr;
    KOKKOSFFT_CHECK_ROCFFT_CALL(rocfft_brick_create(
        &brick, const_cast<std::size_t *>(lower.data()),
        const_cast<std::size_t *>(upper.data()),
        const_cast<std::size_t *>(strides.data()), lower.size(), deviceID));
    KOKKOSFFT_CHECK_ROCFFT_CALL(rocfft_field_add_brick(m_field, brick));
    KOKKOSFFT_CHECK_ROCFFT_CALL(rocfft_brick_destroy(brick));
    brick = nullptr;
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

  //! The plan description keeps the address of this member
  //! (rocfft_plan_description_set_comm), so this class must not be copied or
  //! moved. It is held through std::unique_ptr in TplPlan.
  MPI_Comm m_comm;

  //! Size in bytes of the work buffer of the plan on this device
  std::size_t m_workspace_size = 0;

 public:
  ScopedRocfftMPIPlan(const std::vector<std::size_t> &length,
                      const std::vector<std::size_t> &lower_input,
                      const std::vector<std::size_t> &upper_input,
                      const std::vector<std::size_t> &lower_output,
                      const std::vector<std::size_t> &upper_output,
                      const std::vector<std::size_t> &strides_input,
                      const std::vector<std::size_t> &strides_output,
                      std::size_t number_of_transforms,
                      KokkosFFT::Direction direction, double scale_factor,
                      int device, const MPI_Comm &comm)
      : m_comm(comm) {
    KOKKOSFFT_CHECK_ROCFFT_CALL(rocfft_plan_description_create(&m_description));
    KOKKOSFFT_CHECK_ROCFFT_CALL(rocfft_plan_description_set_comm(
        m_description, rocfft_comm_mpi, &m_comm));

    // Do not set stride information via the descriptor, they are to be defined
    // during field creation below
    auto [in_array_type, out_array_type, fft_direction] =
        KokkosFFT::Impl::get_in_out_array_type<
            KokkosFFT::Impl::transform_kind_v<T1, T2>>(direction);
    KOKKOSFFT_CHECK_ROCFFT_CALL(rocfft_plan_description_set_data_layout(
        m_description, in_array_type, out_array_type, nullptr, nullptr, 0,
        nullptr, 0, 0, nullptr, 0));

    // The normalization, applied by rocFFT to the result (R8): no pass of
    // its own over the output
    if (scale_factor != 1.0) {
      KOKKOSFFT_CHECK_ROCFFT_CALL(rocfft_plan_description_set_scale_factor(
          m_description, scale_factor));
    }

    // The bricks are on the device of this rank (the one of the execution
    // space): with every GPU visible to every rank, device 0 would make
    // rocFFT read the data of the other ranks on the wrong GPU
    ScopedRocfftField scoped_infield(lower_input, upper_input, strides_input,
                                     device);
    ScopedRocfftField scoped_outfield(lower_output, upper_output,
                                      strides_output, device);
    KOKKOSFFT_CHECK_ROCFFT_CALL(rocfft_plan_description_add_infield(
        m_description, scoped_infield.field()));
    KOKKOSFFT_CHECK_ROCFFT_CALL(rocfft_plan_description_add_outfield(
        m_description, scoped_outfield.field()));

    // inplace or Out-of-place transform
    const rocfft_result_placement place = rocfft_placement_notinplace;

    // Create a plan. rocfft_plan_create can fail on one rank only (e.g.
    // rocFFT built without ROCFFT_MPI_ENABLE, or a brick it rejects), so the
    // result is checked on every rank before the plan is used: otherwise the
    // other ranks would hang in the next collective call.
    const rocfft_status status =
        rocfft_plan_create(&m_plan, place, fft_direction, m_precision,
                           length.size(),         // Dimension
                           length.data(),         // Lengths
                           number_of_transforms,  // The global batch
                           m_description);        // Description
    int is_created = status == rocfft_status_success, all_created = 0;
    ::MPI_Allreduce(&is_created, &all_created, 1, MPI_INT, MPI_LAND, m_comm);
    if (!all_created) {
      if (is_created) rocfft_plan_destroy(m_plan);
      rocfft_plan_description_destroy(m_description);
      KOKKOSFFT_THROW_IF(
          true,
          std::string("ScopedRocfftMPIPlan: rocfft_plan_create failed ") +
              (is_created ? "on another rank"
                          : "on this rank with status " +
                                std::to_string(static_cast<int>(status))) +
              ". rocFFT MPI transforms need rocFFT built with "
              "ROCFFT_MPI_ENABLE and a GPU-aware MPI.");
    }

    // The work buffer is provided by the caller (execution info), shared by
    // the plans of both directions: without one, rocfft_execute allocates
    // and frees a work buffer at every call
    KOKKOSFFT_CHECK_ROCFFT_CALL(
        rocfft_plan_get_work_buffer_size(m_plan, &m_workspace_size));
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

  /// \brief Size in bytes of the work buffer of the plan on this device
  std::size_t workspace_size() const noexcept { return m_workspace_size; }
};

/// \brief The forward and backward rocFFT MPI plans of a TplPlan, with one
/// execution info shared by both: the stream of the execution space, and one
/// work buffer of the larger of the sizes the two plans need, allocated once
/// (the plans run one after the other on the same stream).
///
/// When the axes need a transpose (the plans are made for the data in the
/// mapped order), two data buffers hold the data in that order: A the input
/// of the forward transform (the output of the backward one), B the output of
/// the forward transform (the input of the backward one). The user's Views
/// are only read (input) and written (output) by the transposes, so the
/// input is preserved. In-place without a transpose, the plans stay
/// out-of-place: the input is copied into A (the staging buffer, large
/// enough for the input of either direction) and the plan runs from A into
/// the output; so does the backward (C2R) transform of an R2C plan, which
/// may overwrite its input. Otherwise, both buffers are empty.
///
/// With strided bricks (T4b), the bricks describe the Views with their own
/// strides in the order of the plan, so that a permutation or a change of
/// layout needs no transpose: no buffer B, and A only as the staging buffer
/// of the backward (C2R) transform of an R2C plan.
///
/// The normalization is done by rocFFT (the scale factor of each plan, fixed
/// at its creation), so TplPlan does not normalize (scales_in_transform).
template <typename ExecutionSpace, typename T1, typename T2>
struct ScopedRocfftMPIBidirectionalPlan {
  //! rocFFT applies the normalization
  //! (rocfft_plan_description_set_scale_factor)
  static constexpr bool scales_in_transform = true;

 private:
  using ScopedRocfftMPIForwardPlanType =
      ScopedRocfftMPIPlan<ExecutionSpace, T1, T2>;
  using ScopedRocfftMPIBackwardPlanType =
      ScopedRocfftMPIPlan<ExecutionSpace, T2, T1>;
  using ByteViewType = Kokkos::View<char *, ExecutionSpace>;
  ScopedRocfftMPIForwardPlanType m_plan_forward;
  ScopedRocfftMPIBackwardPlanType m_plan_backward;
  ByteViewType m_work_buffer;
  KokkosFFT::Impl::ScopedRocfftExecutionInfo m_execution_info;
  ByteViewType m_buffer_a, m_buffer_b;

  //! Whether the bricks describe the Views with their own strides
  bool m_is_strided = false;

 public:
  /// \param[in] in_buffer_size Size in bytes of buffer A (the forward input in
  /// the mapped order, or the staging buffer of an in-place plan), 0 if not
  /// needed
  /// \param[in] out_buffer_size Size in bytes of buffer B (the forward output
  /// in the mapped order), 0 without transpose
  /// \param[in] number_of_transforms The global batch (1 without a batch
  /// dimension)
  /// \param[in] is_strided Whether the bricks describe the Views with their
  /// own strides (no transpose)
  /// \param[in] forward_scale Scale factor of the forward plan
  /// \param[in] backward_scale Scale factor of the backward plan
  /// \param[in] device HIP device of this rank, where its bricks are
  ScopedRocfftMPIBidirectionalPlan(
      const std::vector<std::size_t> &length,
      const std::vector<std::size_t> &lower_input,
      const std::vector<std::size_t> &upper_input,
      const std::vector<std::size_t> &lower_output,
      const std::vector<std::size_t> &upper_output,
      const std::vector<std::size_t> &strides_input,
      const std::vector<std::size_t> &strides_output,
      std::size_t in_buffer_size, std::size_t out_buffer_size,
      std::size_t number_of_transforms, bool is_strided, double forward_scale,
      double backward_scale, int device, const MPI_Comm &comm)
      : m_plan_forward(length, lower_input, upper_input, lower_output,
                       upper_output, strides_input, strides_output,
                       number_of_transforms, KokkosFFT::Direction::forward,
                       forward_scale, device, comm),
        m_plan_backward(length, lower_output, upper_output, lower_input,
                        upper_input, strides_output, strides_input,
                        number_of_transforms, KokkosFFT::Direction::backward,
                        backward_scale, device, comm),
        m_work_buffer("KokkosFFT::Distributed::rocFFTMPI_work_buffer",
                      std::max(m_plan_forward.workspace_size(),
                               m_plan_backward.workspace_size())),
        m_buffer_a("KokkosFFT::Distributed::rocFFTMPI_buffer_a",
                   in_buffer_size),
        m_buffer_b("KokkosFFT::Distributed::rocFFTMPI_buffer_b",
                   out_buffer_size),
        m_is_strided(is_strided) {
    m_execution_info.set_work_area(m_work_buffer.data(), m_work_buffer.size());
  }

  /// \brief A View of the given type on the buffer holding the input
  /// (is_input) or the output of the transform in this direction, in the
  /// mapped order
  /// \tparam ViewType Kokkos View type (LayoutRight) to wrap the buffer
  /// \param[in] direction Direction of the transform
  /// \param[in] is_input Whether the input or the output buffer
  /// \param[in] extents Extents of the View (mapped)
  template <typename ViewType, std::size_t DIM>
  auto buffer_data(KokkosFFT::Direction direction, bool is_input,
                   const std::array<std::size_t, DIM> &extents) const {
    using value_type      = typename ViewType::non_const_value_type;
    using layout_type     = typename ViewType::array_layout;
    const bool is_forward = direction == KokkosFFT::Direction::forward;
    const auto &buffer    = is_forward == is_input ? m_buffer_a : m_buffer_b;
    return ViewType(reinterpret_cast<value_type *>(buffer.data()),
                    KokkosFFT::Impl::create_layout<layout_type>(extents));
  }

  /// \brief A View of the given type on the staging buffer (A) of an
  /// in-place or R2C plan, for the input of either direction
  /// \tparam ViewType Kokkos View type (LayoutRight) to wrap the buffer
  /// \param[in] extents Extents of the View
  /// \throws std::runtime_error if the staging buffer is too small
  template <typename ViewType, std::size_t DIM>
  auto staging_data(const std::array<std::size_t, DIM> &extents) const {
    using value_type  = typename ViewType::non_const_value_type;
    using layout_type = typename ViewType::array_layout;
    KOKKOSFFT_THROW_IF(
        m_buffer_a.size() <
            KokkosFFT::Impl::total_size(extents) * sizeof(value_type),
        "ScopedRocfftMPIBidirectionalPlan: the staging buffer is too small "
        "(the plan was made out-of-place, or without a staging buffer)");
    return ViewType(reinterpret_cast<value_type *>(m_buffer_a.data()),
                    KokkosFFT::Impl::create_layout<layout_type>(extents));
  }

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

  /// \brief Run the plans on the stream of exec_space
  void commit(const Kokkos::HIP &exec_space) {
    m_execution_info.commit(exec_space);
  }

  /// \brief Whether the bricks describe the Views with their own strides
  /// (T4b): the plans run on the Views, without a transpose
  bool is_strided() const noexcept { return m_is_strided; }

  /// \brief Size in bytes of buffer B, which only the transposing path has
  /// (for the tests)
  std::size_t transpose_buffer_size() const noexcept {
    return m_buffer_b.size();
  }

  /// \brief The execution info of both plans (stream and work buffer)
  rocfft_execution_info execution_info() const noexcept {
    return m_execution_info.execution_info();
  }

  /// \brief Size in bytes of the work buffer shared by the plans
  std::size_t workspace_size() const noexcept { return m_work_buffer.size(); }

  /// \brief Get the name of the plan implementation
  /// \return Name of the plan implementation
  std::string name() const { return std::string("rocFFTMPIPlan"); }
};

template <typename ExecutionSpace, typename T1, typename T2>
struct InternalTplPlanType {
  using type = ScopedRocfftMPIBidirectionalPlan<ExecutionSpace, T1, T2>;
};

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
