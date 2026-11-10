#ifndef KOKKOSFFT_DISTRIBUTED_CUFFT_MP_TYPES_HPP
#define KOKKOSFFT_DISTRIBUTED_CUFFT_MP_TYPES_HPP

#include <algorithm>
#include <string>
#include <mpi.h>
#include <cufftMp.h>
#include <nvshmem.h>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

struct cuFFTHandleWrapper {
  cufftHandle m_handle = 0;
  cufftHandle plan() const noexcept { return m_handle; }
};

/// \brief RAII wrapper for cufftMp plans
/// This class handles both forward and backward plans
/// For complex-to-complex transforms, only the forward plan is created
///
/// The data buffer and the scratch of the plans are on the NVSHMEM symmetric
/// heap, allocated by commit with the same size on every PE. The plans do not
/// allocate their own scratch (cufftSetAutoAllocation(plan, 0)): one scratch,
/// of the largest size the plans need, is shared by them (cufftSetWorkArea),
/// as they run one after the other on the same stream.
///
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam T1 Input data type
/// \tparam T2 Output data type
template <typename ExecutionSpace, typename T1, typename T2>
struct ScopedCufftMpPlan {
  //! The normalization is applied by exec_plan, in the copy out of the
  //! NVSHMEM buffer (cuFFTMp has no scale factor): TplPlan does not normalize
  static constexpr bool scales_in_transform = true;

  using buffer_data_type =
      typename KokkosFFT::Impl::fft_data_type<ExecutionSpace, T2>::type;
  //@{
  //! cufftHandle for forward and backward plans
  cuFFTHandleWrapper m_plan_f, m_plan_b;
  ///@}

  //! GPU memory (NVSHMEM): the data, transformed in place
  buffer_data_type *m_buffer = nullptr;
  std::size_t m_buffer_size  = 0;

  //! GPU memory (NVSHMEM): the scratch shared by the plans
  void *m_workarea             = nullptr;
  std::size_t m_workspace_size = 0;

 private:
  //! The communicator of the plans. cuFFTMp keeps the pointer it is given
  //! (comm_handle), which must stay valid until cufftDestroy: a member, as
  //! the plan is neither copied nor moved.
  MPI_Comm m_comm;

  //@{
  //! Whether cufftCreate succeeded for the forward and backward handles
  bool m_is_created_f = false, m_is_created_b = false;
  ///@}

  /// \brief The first cuFFT call of the plan creation that failed on this
  /// rank. The creation goes on after a failure so that the ranks make the
  /// same collective calls; check_creation then fails on every rank.
  struct CreationStatus {
    cufftResult m_result = CUFFT_SUCCESS;
    const char *m_call   = "";

    /// \brief Record the result of a call
    /// \return Whether the call succeeded
    bool record(cufftResult result, const char *call) {
      if (m_result == CUFFT_SUCCESS && result != CUFFT_SUCCESS) {
        m_result = result;
        m_call   = call;
      }
      return result == CUFFT_SUCCESS;
    }
  };

#define KOKKOSFFT_DISTRIBUTED_RECORD_CUFFT_CALL(status, call) \
  (status).record((call), #call)

  /// \brief Create the forward (and, for real transforms, backward) handles
  /// without automatic allocation of the scratch. Called first by the
  /// constructors: the work already submitted is completed before the
  /// collective planning of cuFFTMp and the NVSHMEM allocations (commit), as
  /// a transform of another plan may still be communicating on the device.
  void create_handles(CreationStatus &status) {
    Kokkos::fence("ScopedCufftMpPlan: before the plan creation");
    m_is_created_f = KOKKOSFFT_DISTRIBUTED_RECORD_CUFFT_CALL(
        status, cufftCreate(&m_plan_f.m_handle));
    KOKKOSFFT_DISTRIBUTED_RECORD_CUFFT_CALL(
        status, cufftSetAutoAllocation(m_plan_f.m_handle, 0));
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      m_is_created_b = KOKKOSFFT_DISTRIBUTED_RECORD_CUFFT_CALL(
          status, cufftCreate(&m_plan_b.m_handle));
      KOKKOSFFT_DISTRIBUTED_RECORD_CUFFT_CALL(
          status, cufftSetAutoAllocation(m_plan_b.m_handle, 0));
    }
  }

  /// \brief Check on every rank that the plans were created. A cuFFTMp call
  /// can fail on one rank only; throwing there alone would leave the other
  /// ranks waiting in the next collective call. The cost is one MPI_Allreduce
  /// of an int per plan, small next to the collective planning of cuFFTMp.
  /// \param[in] status The result of the creation on this rank
  /// \throws std::runtime_error on every rank if the creation failed on any
  void check_creation(const CreationStatus &status) {
    int is_created  = status.m_result == CUFFT_SUCCESS;
    int all_created = 0;
    ::MPI_Allreduce(&is_created, &all_created, 1, MPI_INT, MPI_LAND, m_comm);
    if (all_created) return;

    // The destructor does not run when the constructor throws
    if (m_is_created_f) cufftDestroy(m_plan_f.m_handle);
    if (m_is_created_b) cufftDestroy(m_plan_b.m_handle);

    int rank = 0;
    ::MPI_Comm_rank(m_comm, &rank);
    std::string message =
        "ScopedCufftMpPlan: the creation of the cuFFTMp plans failed ";
    if (is_created) {
      message += "on another rank";
    } else {
      message += "on rank " + std::to_string(rank) + ": " + status.m_call +
                 " failed with error code " +
                 std::to_string(static_cast<int>(status.m_result)) + " (" +
                 std::string(
                     KokkosFFT::Impl::cufft_result_to_string(status.m_result)) +
                 ")";
    }
    KOKKOSFFT_THROW_IF(true, message);
  }

 public:
  /// \brief Constructor for 2D FFT plans
  /// \param[in] nx Global size in X dimension
  /// \param[in] ny Global size in Y dimension
  /// \param[in] comm MPI communicator
  /// \param[in] is_xslab Whether the topology is x-slab
  /// \throws std::runtime_error on every rank if the creation failed on any
  ScopedCufftMpPlan(int nx, int ny, MPI_Comm comm, bool is_xslab)
      : m_comm(comm) {
    CreationStatus status;
    create_handles(status);
    cufftXtSubFormat subformat_forward =
        is_xslab ? CUFFT_XT_FORMAT_INPLACE : CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
    cufftXtSubFormat subformat_inverse =
        is_xslab ? CUFFT_XT_FORMAT_INPLACE_SHUFFLED : CUFFT_XT_FORMAT_INPLACE;
    std::size_t workspace = 0;
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      auto r2c_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T1, T2>::type();
      auto c2r_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T2, T1>::type();
      KOKKOSFFT_DISTRIBUTED_RECORD_CUFFT_CALL(
          status, cufftXtSetSubformatDefault(
                      m_plan_f.m_handle, subformat_forward, subformat_inverse));
      KOKKOSFFT_DISTRIBUTED_RECORD_CUFFT_CALL(
          status, cufftXtSetSubformatDefault(
                      m_plan_b.m_handle, subformat_forward, subformat_inverse));
      KOKKOSFFT_DISTRIBUTED_RECORD_CUFFT_CALL(
          status, cufftMpMakePlan2d(m_plan_f.m_handle, nx, ny, r2c_type,
                                    &m_comm, CUFFT_COMM_MPI, &workspace));
      m_workspace_size = std::max(m_workspace_size, workspace);
      KOKKOSFFT_DISTRIBUTED_RECORD_CUFFT_CALL(
          status, cufftMpMakePlan2d(m_plan_b.m_handle, nx, ny, c2r_type,
                                    &m_comm, CUFFT_COMM_MPI, &workspace));
      m_workspace_size = std::max(m_workspace_size, workspace);
    } else {
      auto c2c_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T1, T2>::type();
      KOKKOSFFT_DISTRIBUTED_RECORD_CUFFT_CALL(
          status, cufftXtSetSubformatDefault(
                      m_plan_f.m_handle, subformat_forward, subformat_inverse));
      KOKKOSFFT_DISTRIBUTED_RECORD_CUFFT_CALL(
          status, cufftMpMakePlan2d(m_plan_f.m_handle, nx, ny, c2c_type,
                                    &m_comm, CUFFT_COMM_MPI, &workspace));
      m_workspace_size = std::max(m_workspace_size, workspace);
    }
    check_creation(status);
  }

  /// \brief Constructor for 3D FFT plans
  /// \param[in] nx Global size in X dimension
  /// \param[in] ny Global size in Y dimension
  /// \param[in] nz Global size in Z dimension
  /// \param[in] comm MPI communicator
  /// \param[in] is_xslab Whether the topology is x-slab
  /// \throws std::runtime_error on every rank if the creation failed on any
  ScopedCufftMpPlan(int nx, int ny, int nz, MPI_Comm comm, bool is_xslab)
      : m_comm(comm) {
    CreationStatus status;
    create_handles(status);
    cufftXtSubFormat subformat_forward =
        is_xslab ? CUFFT_XT_FORMAT_INPLACE : CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
    cufftXtSubFormat subformat_inverse =
        is_xslab ? CUFFT_XT_FORMAT_INPLACE_SHUFFLED : CUFFT_XT_FORMAT_INPLACE;
    std::size_t workspace = 0;
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      auto r2c_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T1, T2>::type();
      auto c2r_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T2, T1>::type();
      KOKKOSFFT_DISTRIBUTED_RECORD_CUFFT_CALL(
          status, cufftXtSetSubformatDefault(
                      m_plan_f.m_handle, subformat_forward, subformat_inverse));
      KOKKOSFFT_DISTRIBUTED_RECORD_CUFFT_CALL(
          status, cufftXtSetSubformatDefault(
                      m_plan_b.m_handle, subformat_forward, subformat_inverse));
      KOKKOSFFT_DISTRIBUTED_RECORD_CUFFT_CALL(
          status, cufftMpMakePlan3d(m_plan_f.m_handle, nx, ny, nz, r2c_type,
                                    &m_comm, CUFFT_COMM_MPI, &workspace));
      m_workspace_size = std::max(m_workspace_size, workspace);
      KOKKOSFFT_DISTRIBUTED_RECORD_CUFFT_CALL(
          status, cufftMpMakePlan3d(m_plan_b.m_handle, nx, ny, nz, c2r_type,
                                    &m_comm, CUFFT_COMM_MPI, &workspace));
      m_workspace_size = std::max(m_workspace_size, workspace);
    } else {
      auto c2c_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T1, T2>::type();
      KOKKOSFFT_DISTRIBUTED_RECORD_CUFFT_CALL(
          status, cufftXtSetSubformatDefault(
                      m_plan_f.m_handle, subformat_forward, subformat_inverse));
      KOKKOSFFT_DISTRIBUTED_RECORD_CUFFT_CALL(
          status, cufftMpMakePlan3d(m_plan_f.m_handle, nx, ny, nz, c2c_type,
                                    &m_comm, CUFFT_COMM_MPI, &workspace));
      m_workspace_size = std::max(m_workspace_size, workspace);
    }
    check_creation(status);
  }

  /// \brief General constructor for FFT plans
  /// \param[in] fft_extents Global FFT sizes in each dimension
  /// \param[in] lower_input Lower bounds of input data distribution
  /// \param[in] upper_input Upper bounds of input data distribution
  /// \param[in] lower_output Lower bounds of output data distribution
  /// \param[in] upper_output Upper bounds of output data distribution
  /// \param[in] strides_input Strides of input data
  /// \param[in] strides_output Strides of output data
  /// \param[in] comm MPI communicator
  /// \throws std::runtime_error on every rank if the creation failed on any
  ScopedCufftMpPlan(std::vector<int> &fft_extents,
                    const std::vector<long long int> &lower_input,
                    const std::vector<long long int> &upper_input,
                    const std::vector<long long int> &lower_output,
                    const std::vector<long long int> &upper_output,
                    const std::vector<long long int> &strides_input,
                    const std::vector<long long int> &strides_output,
                    MPI_Comm comm)
      : m_comm(comm) {
    CreationStatus status;
    create_handles(status);
    int rank              = fft_extents.size();
    std::size_t workspace = 0;
    KOKKOSFFT_DISTRIBUTED_RECORD_CUFFT_CALL(
        status, cufftXtSetSubformatDefault(m_plan_f.m_handle,
                                           CUFFT_XT_FORMAT_DISTRIBUTED_INPUT,
                                           CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT));
    auto forward_type =
        KokkosFFT::Impl::transform_type<ExecutionSpace, T1, T2>::type();
    KOKKOSFFT_DISTRIBUTED_RECORD_CUFFT_CALL(
        status,
        cufftMpMakePlanDecomposition(
            m_plan_f.m_handle, rank, fft_extents.data(), lower_input.data(),
            upper_input.data(), strides_input.data(), lower_output.data(),
            upper_output.data(), strides_output.data(), forward_type, &m_comm,
            CUFFT_COMM_MPI, &workspace));
    m_workspace_size = std::max(m_workspace_size, workspace);
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      auto c2r_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T2, T1>::type();
      KOKKOSFFT_DISTRIBUTED_RECORD_CUFFT_CALL(
          status, cufftXtSetSubformatDefault(
                      m_plan_b.m_handle, CUFFT_XT_FORMAT_DISTRIBUTED_INPUT,
                      CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT));
      KOKKOSFFT_DISTRIBUTED_RECORD_CUFFT_CALL(
          status,
          cufftMpMakePlanDecomposition(
              m_plan_b.m_handle, rank, fft_extents.data(), lower_input.data(),
              upper_input.data(), strides_input.data(), lower_output.data(),
              upper_output.data(), strides_output.data(), c2r_type, &m_comm,
              CUFFT_COMM_MPI, &workspace));
      m_workspace_size = std::max(m_workspace_size, workspace);
    }
    check_creation(status);
  }

#undef KOKKOSFFT_DISTRIBUTED_RECORD_CUFFT_CALL
  ScopedCufftMpPlan()                                     = delete;
  ScopedCufftMpPlan(const ScopedCufftMpPlan &)            = delete;
  ScopedCufftMpPlan &operator=(const ScopedCufftMpPlan &) = delete;
  ScopedCufftMpPlan &operator=(ScopedCufftMpPlan &&)      = delete;
  ScopedCufftMpPlan(ScopedCufftMpPlan &&)                 = delete;

  /// \brief Destructor to free cufftMp resources. The work already submitted
  /// (a transform of this plan may still be running) is completed first. The
  /// plans are destroyed before the memory they use; nvshmem_free is
  /// collective, as the plans are destroyed on every PE.
  ~ScopedCufftMpPlan() noexcept {
    Kokkos::Profiling::ScopedRegion region("cleanup_plan[TPL_cuFFTMp]");
    Kokkos::fence("ScopedCufftMpPlan: before the plan destruction");
    bool is_destroyed = cufftDestroy(m_plan_f.m_handle) == CUFFT_SUCCESS;
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      is_destroyed &= cufftDestroy(m_plan_b.m_handle) == CUFFT_SUCCESS;
    }
    nvshmem_free(m_workarea);
    nvshmem_free(m_buffer);
    if (!is_destroyed) Kokkos::abort("cufftDestroy failed");
  }

  /// \brief Get the underlying cufftHandle
  /// \param[in] direction Direction of the FFT (forward/backward)
  /// \return cufftHandle for the specified direction
  auto plan([[maybe_unused]] KokkosFFT::Direction direction) const noexcept {
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      return direction == KokkosFFT::Direction::forward ? m_plan_f : m_plan_b;
    } else {
      return m_plan_f;
    }
  }

  /// \brief Get a Kokkos View wrapper around the GPU buffer
  /// \tparam ViewType Kokkos View type to wrap the buffer
  /// \tparam DIM Number of dimensions of the View
  ///
  /// \param[in] extents Extents of the View
  /// \return Kokkos View wrapping the GPU buffer
  template <typename ViewType, std::size_t DIM>
  auto buffer_data(const std::array<std::size_t, DIM> &extents) const {
    using value_type  = typename ViewType::non_const_value_type;
    using layout_type = typename ViewType::array_layout;
    return ViewType(reinterpret_cast<value_type *>(m_buffer),
                    KokkosFFT::Impl::create_layout<layout_type>(extents));
  }

  /// \brief Commit the plan to a specific execution space, and allocate the
  /// data buffer and the shared scratch on the NVSHMEM symmetric heap.
  /// Collective over comm: nvshmem_malloc needs the same size on every PE, so
  /// both sizes are the maximum over comm.
  /// \param[in] exec_space Kokkos execution space
  /// \param[in] buffer_size Size of the data buffer in bytes on this rank (the
  /// larger of the input and the output, see buffer_size_in_bytes)
  /// \param[in] comm MPI communicator of the plan
  /// \throws std::runtime_error if an allocation fails
  void commit(const Kokkos::Cuda &exec_space, std::size_t buffer_size,
              MPI_Comm comm) {
    KOKKOSFFT_CHECK_CUFFT_CALL(
        cufftSetStream(m_plan_f.m_handle, exec_space.cuda_stream()));
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      KOKKOSFFT_CHECK_CUFFT_CALL(
          cufftSetStream(m_plan_b.m_handle, exec_space.cuda_stream()));
    }

    // The same sizes on every PE
    unsigned long long sizes[2]     = {buffer_size, m_workspace_size};
    unsigned long long max_sizes[2] = {0, 0};
    ::MPI_Allreduce(sizes, max_sizes, 2, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm);
    m_buffer_size    = static_cast<std::size_t>(max_sizes[0]);
    m_workspace_size = static_cast<std::size_t>(max_sizes[1]);

    // The data buffer, used for both directions
    m_buffer = static_cast<buffer_data_type *>(nvshmem_malloc(m_buffer_size));
    KOKKOSFFT_THROW_IF(m_buffer_size > 0 && m_buffer == nullptr,
                       "ScopedCufftMpPlan: nvshmem_malloc of the data buffer "
                       "failed (NVSHMEM_SYMMETRIC_SIZE)");

    // The scratch, shared by the plans
    if (m_workspace_size > 0) {
      m_workarea = nvshmem_malloc(m_workspace_size);
      KOKKOSFFT_THROW_IF(m_workarea == nullptr,
                         "ScopedCufftMpPlan: nvshmem_malloc of the scratch "
                         "failed (NVSHMEM_SYMMETRIC_SIZE)");
    }
    KOKKOSFFT_CHECK_CUFFT_CALL(cufftSetWorkArea(m_plan_f.m_handle, m_workarea));
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      KOKKOSFFT_CHECK_CUFFT_CALL(
          cufftSetWorkArea(m_plan_b.m_handle, m_workarea));
    }
  }

  /// \brief Size in bytes of the data buffer (the same on every PE)
  std::size_t buffer_size() const noexcept { return m_buffer_size; }

  /// \brief Size in bytes of the scratch shared by the plans (the same on
  /// every PE)
  std::size_t workspace_size() const noexcept { return m_workspace_size; }

  /// \brief Get the name of the plan implementation
  /// \return Name of the plan implementation
  std::string name() const { return std::string("cufftMpPlan"); }
};

template <typename ExecutionSpace, typename T1, typename T2>
struct InternalTplPlanType {
  using type = ScopedCufftMpPlan<ExecutionSpace, T1, T2>;
};

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
