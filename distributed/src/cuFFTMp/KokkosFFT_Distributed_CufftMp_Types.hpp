#ifndef KOKKOSFFT_DISTRIBUTED_CUFFT_MP_TYPES_HPP
#define KOKKOSFFT_DISTRIBUTED_CUFFT_MP_TYPES_HPP

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
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam T1 Input data type
/// \tparam T2 Output data type
template <typename ExecutionSpace, typename T1, typename T2>
struct ScopedCufftMpPlan {
  using buffer_data_type =
      typename KokkosFFT::Impl::fft_data_type<ExecutionSpace, T2>::type *;
  //@{
  //! cufftHandle for forward and backward plans
  // cufftHandle m_plan_f = 0, m_plan_b = 0;
  cuFFTHandleWrapper m_plan_f, m_plan_b;
  ///@}

  //! GPU memory
  buffer_data_type *m_buffer = nullptr;

 public:
  /// \brief Constructor for 2D FFT plans
  /// \param[in] nx Global size in X dimension
  /// \param[in] ny Global size in Y dimension
  /// \param[in] comm MPI communicator
  /// \param[in] is_xslab Whether the topology is x-slab
  ScopedCufftMpPlan(int nx, int ny, MPI_Comm comm, bool is_xslab) {
    KOKKOSFFT_CHECK_CUFFT_CALL(cufftCreate(&m_plan_f.m_handle));
    cufftXtSubFormat subformat_forward =
        is_xslab ? CUFFT_XT_FORMAT_INPLACE : CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
    cufftXtSubFormat subformat_inverse =
        is_xslab ? CUFFT_XT_FORMAT_INPLACE_SHUFFLED : CUFFT_XT_FORMAT_INPLACE;
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      KOKKOSFFT_CHECK_CUFFT_CALL(cufftCreate(&m_plan_b.m_handle));
      std::size_t workspace = 0;
      auto r2c_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T1, T2>::type();
      auto c2r_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T2, T1>::type();
      KOKKOSFFT_CHECK_CUFFT_CALL(cufftXtSetSubformatDefault(
          m_plan_f.m_handle, subformat_forward, subformat_inverse));
      KOKKOSFFT_CHECK_CUFFT_CALL(cufftXtSetSubformatDefault(
          m_plan_b.m_handle, subformat_forward, subformat_inverse));

      KOKKOSFFT_CHECK_CUFFT_CALL(cufftMpMakePlan2d(m_plan_f.m_handle, nx, ny,
                                                   r2c_type, &comm,
                                                   CUFFT_COMM_MPI, &workspace));
      KOKKOSFFT_CHECK_CUFFT_CALL(cufftMpMakePlan2d(m_plan_b.m_handle, nx, ny,
                                                   c2r_type, &comm,
                                                   CUFFT_COMM_MPI, &workspace));
    } else {
      std::size_t workspace = 0;
      auto c2c_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T1, T2>::type();
      KOKKOSFFT_CHECK_CUFFT_CALL(cufftXtSetSubformatDefault(
          m_plan_f.m_handle, subformat_forward, subformat_inverse));
      KOKKOSFFT_CHECK_CUFFT_CALL(cufftMpMakePlan2d(m_plan_f.m_handle, nx, ny,
                                                   c2c_type, &comm,
                                                   CUFFT_COMM_MPI, &workspace));
    }
  }

  /// \brief Constructor for 3D FFT plans
  /// \param[in] nx Global size in X dimension
  /// \param[in] ny Global size in Y dimension
  /// \param[in] nz Global size in Z dimension
  /// \param[in] comm MPI communicator
  /// \param[in] is_xslab Whether the topology is x-slab
  ScopedCufftMpPlan(int nx, int ny, int nz, MPI_Comm comm, bool is_xslab) {
    KOKKOSFFT_CHECK_CUFFT_CALL(cufftCreate(&m_plan_f.m_handle));
    cufftXtSubFormat subformat_forward =
        is_xslab ? CUFFT_XT_FORMAT_INPLACE : CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
    cufftXtSubFormat subformat_inverse =
        is_xslab ? CUFFT_XT_FORMAT_INPLACE_SHUFFLED : CUFFT_XT_FORMAT_INPLACE;
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      KOKKOSFFT_CHECK_CUFFT_CALL(cufftCreate(&m_plan_b.m_handle));
      std::size_t workspace = 0;
      auto r2c_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T1, T2>::type();
      auto c2r_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T2, T1>::type();
      KOKKOSFFT_CHECK_CUFFT_CALL(cufftXtSetSubformatDefault(
          m_plan_f.m_handle, subformat_forward, subformat_inverse));
      KOKKOSFFT_CHECK_CUFFT_CALL(cufftXtSetSubformatDefault(
          m_plan_b.m_handle, subformat_forward, subformat_inverse));
      KOKKOSFFT_CHECK_CUFFT_CALL(cufftMpMakePlan3d(m_plan_f.m_handle, nx, ny,
                                                   nz, r2c_type, &comm,
                                                   CUFFT_COMM_MPI, &workspace));
      KOKKOSFFT_CHECK_CUFFT_CALL(cufftMpMakePlan3d(m_plan_b.m_handle, nx, ny,
                                                   nz, c2r_type, &comm,
                                                   CUFFT_COMM_MPI, &workspace));
    } else {
      std::size_t workspace = 0;
      auto c2c_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T1, T2>::type();
      KOKKOSFFT_CHECK_CUFFT_CALL(cufftXtSetSubformatDefault(
          m_plan_f.m_handle, subformat_forward, subformat_inverse));
      KOKKOSFFT_CHECK_CUFFT_CALL(cufftMpMakePlan3d(m_plan_f.m_handle, nx, ny,
                                                   nz, c2c_type, &comm,
                                                   CUFFT_COMM_MPI, &workspace));
    }
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
  ScopedCufftMpPlan(std::vector<int> &fft_extents,
                    const std::vector<long long int> &lower_input,
                    const std::vector<long long int> &upper_input,
                    const std::vector<long long int> &lower_output,
                    const std::vector<long long int> &upper_output,
                    const std::vector<long long int> &strides_input,
                    const std::vector<long long int> &strides_output,
                    MPI_Comm comm) {
    KOKKOSFFT_CHECK_CUFFT_CALL(cufftCreate(&m_plan_f.m_handle));
    int rank = fft_extents.size();
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      KOKKOSFFT_CHECK_CUFFT_CALL(cufftCreate(&m_plan_b.m_handle));
      auto r2c_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T1, T2>::type();
      auto c2r_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T2, T1>::type();
      std::size_t workspace = 0;
      KOKKOSFFT_CHECK_CUFFT_CALL(cufftXtSetSubformatDefault(
          m_plan_f.m_handle, CUFFT_XT_FORMAT_DISTRIBUTED_INPUT,
          CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT));
      KOKKOSFFT_CHECK_CUFFT_CALL(cufftXtSetSubformatDefault(
          m_plan_b.m_handle, CUFFT_XT_FORMAT_DISTRIBUTED_INPUT,
          CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT));

      KOKKOSFFT_CHECK_CUFFT_CALL(cufftMpMakePlanDecomposition(
          m_plan_f.m_handle, rank, fft_extents.data(), lower_input.data(),
          upper_input.data(), strides_input.data(), lower_output.data(),
          upper_output.data(), strides_output.data(), r2c_type, &comm,
          CUFFT_COMM_MPI, &workspace));

      KOKKOSFFT_CHECK_CUFFT_CALL(cufftMpMakePlanDecomposition(
          m_plan_b.m_handle, rank, fft_extents.data(), lower_input.data(),
          upper_input.data(), strides_input.data(), lower_output.data(),
          upper_output.data(), strides_output.data(), c2r_type, &comm,
          CUFFT_COMM_MPI, &workspace));
    } else {
      auto c2c_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T1, T2>::type();
      std::size_t workspace = 0;
      KOKKOSFFT_CHECK_CUFFT_CALL(cufftXtSetSubformatDefault(
          m_plan_f.m_handle, CUFFT_XT_FORMAT_DISTRIBUTED_INPUT,
          CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT));
      KOKKOSFFT_CHECK_CUFFT_CALL(cufftMpMakePlanDecomposition(
          m_plan_f.m_handle, rank, fft_extents.data(), lower_input.data(),
          upper_input.data(), strides_input.data(), lower_output.data(),
          upper_output.data(), strides_output.data(), c2c_type, &comm,
          CUFFT_COMM_MPI, &workspace));
    }
  }

  ScopedCufftMpPlan()                                     = delete;
  ScopedCufftMpPlan(const ScopedCufftMpPlan &)            = delete;
  ScopedCufftMpPlan &operator=(const ScopedCufftMpPlan &) = delete;
  ScopedCufftMpPlan &operator=(ScopedCufftMpPlan &&)      = delete;
  ScopedCufftMpPlan(ScopedCufftMpPlan &&)                 = delete;

  /// \brief Destructor to free cufftMp resources
  ~ScopedCufftMpPlan() noexcept {
    Kokkos::Profiling::ScopedRegion region("cleanup_plan[TPL_cuFFTMp]");
    nvshmem_free(m_buffer);
    cufftResult cufft_rt = cufftDestroy(m_plan_f.m_handle);
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      cufft_rt = cufftDestroy(m_plan_b.m_handle);
    }
    if (cufft_rt != CUFFT_SUCCESS) Kokkos::abort("cufftDestroy failed");
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

  /// \brief Commit the plan to a specific execution space
  /// \param[in] exec_space Kokkos execution space
  /// \param[in] size Size of the buffer
  void commit(const Kokkos::Cuda &exec_space, std::size_t size) {
    KOKKOSFFT_CHECK_CUFFT_CALL(
        cufftSetStream(m_plan_f.m_handle, exec_space.cuda_stream()));
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      KOKKOSFFT_CHECK_CUFFT_CALL(
          cufftSetStream(m_plan_b.m_handle, exec_space.cuda_stream()));
    }

    // Allocate GPU memory by nvhsmem
    m_buffer = static_cast<buffer_data_type *>(nvshmem_malloc(
        sizeof(buffer_data_type) * size));  // Allocate for both plans
  }

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
