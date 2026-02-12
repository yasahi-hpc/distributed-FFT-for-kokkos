#ifndef KOKKOSFFT_DISTRIBUTED_CUFFT_MP_TYPES_HPP
#define KOKKOSFFT_DISTRIBUTED_CUFFT_MP_TYPES_HPP

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include <cufftMp.h>

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief RAII wrapper for cufftMp plans
/// This class handles both forward and backward plans
/// For complex-to-complex transforms, only the forward plan is created
///
/// \tparam ExecutionSpace Kokkos execution space type
/// \tparam T1 Input data type
/// \tparam T2 Output data type
template <typename ExecutionSpace, typename T1, typename T2>
struct ScopedCufftMpPlan {
  //@{
  //! cufftHandle for forward and backward plans
  cufftHandle m_plan_f = 0, m_plan_b = 0;
  ///@}

  //! Descriptor for the plan
  cudaLibXtDesc *m_desc;

 public:
  /// \brief Constructor for 2D FFT plans
  /// \param[in] nx Global size in X dimension
  /// \param[in] ny Global size in Y dimension
  /// \param[in] comm MPI communicator
  /// \param[in] is_xslab Whether the topology is x-slab
  ScopedCufftMpPlan(int nx, int ny, MPI_Comm comm, bool is_xslab) {
    KOKKOSFFT_CHECK_CUFFT_CALL(cufftCreate(&m_plan_f));
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      KOKKOSFFT_CHECK_CUFFT_CALL(cufftCreate(&m_plan_b));
      std::size_t workspace = 0;
      auto r2c_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T1, T2>::type();
      auto c2r_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T2, T1>::type();

      KOKKOSFFT_CHECK_CUFFT_CALL(cufftMpMakePlan2d(
          m_plan_f, nx, ny, r2c_type, &comm, CUFFT_COMM_MPI, &workspace));
      KOKKOSFFT_CHECK_CUFFT_CALL(cufftMpMakePlan2d(
          m_plan_b, nx, ny, c2r_type, &comm, CUFFT_COMM_MPI, &workspace));
    } else {
      std::size_t workspace = 0;
      auto c2c_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T1, T2>::type();
      KOKKOSFFT_CHECK_CUFFT_CALL(cufftMpMakePlan2d(
          m_plan_f, nx, ny, c2c_type, &comm, CUFFT_COMM_MPI, &workspace));
    }

    cufftXtSubFormat subformat_forward =
        is_xslab ? CUFFT_XT_FORMAT_INPLACE : CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
    KOKKOSFFT_CHECK_CUFFT_CALL(
        cufftXtMalloc(m_plan_f, &m_desc, subformat_forward));
  }

  /// \brief Constructor for 3D FFT plans
  /// \param[in] nx Global size in X dimension
  /// \param[in] ny Global size in Y dimension
  /// \param[in] nz Global size in Z dimension
  /// \param[in] comm MPI communicator
  /// \param[in] is_xslab Whether the topology is x-slab
  ScopedCufftMpPlan(int nx, int ny, int nz, MPI_Comm comm, bool is_xslab) {
    KOKKOSFFT_CHECK_CUFFT_CALL(cufftCreate(&m_plan_f));
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      KOKKOSFFT_CHECK_CUFFT_CALL(cufftCreate(&m_plan_b));
      std::size_t workspace = 0;
      auto r2c_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T1, T2>::type();
      auto c2r_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T2, T1>::type();
      KOKKOSFFT_CHECK_CUFFT_CALL(cufftMpMakePlan3d(
          m_plan_f, nx, ny, nz, r2c_type, &comm, CUFFT_COMM_MPI, &workspace));
      KOKKOSFFT_CHECK_CUFFT_CALL(cufftMpMakePlan3d(
          m_plan_b, nx, ny, nz, c2r_type, &comm, CUFFT_COMM_MPI, &workspace));
    } else {
      std::size_t workspace = 0;
      auto c2c_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T1, T2>::type();
      KOKKOSFFT_CHECK_CUFFT_CALL(cufftMpMakePlan3d(
          m_plan_f, nx, ny, nz, c2c_type, &comm, CUFFT_COMM_MPI, &workspace));
    }

    cufftXtSubFormat subformat_forward =
        is_xslab ? CUFFT_XT_FORMAT_INPLACE : CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
    KOKKOSFFT_CHECK_CUFFT_CALL(
        cufftXtMalloc(m_plan_f, &m_desc, subformat_forward));
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
    KOKKOSFFT_CHECK_CUFFT_CALL(cufftCreate(&m_plan_f));
    int rank = fft_extents.size();
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      KOKKOSFFT_CHECK_CUFFT_CALL(cufftCreate(&m_plan_b));
      auto r2c_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T1, T2>::type();
      auto c2r_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T2, T1>::type();
      std::size_t workspace = 0;

      KOKKOSFFT_CHECK_CUFFT_CALL(cufftMpMakePlanDecomposition(
          m_plan_f, rank, fft_extents.data(), lower_input.data(),
          upper_input.data(), strides_input.data(), lower_output.data(),
          upper_output.data(), strides_output.data(), r2c_type, &comm,
          CUFFT_COMM_MPI, &workspace));

      KOKKOSFFT_CHECK_CUFFT_CALL(cufftMpMakePlanDecomposition(
          m_plan_b, rank, fft_extents.data(), lower_input.data(),
          upper_input.data(), strides_input.data(), lower_output.data(),
          upper_output.data(), strides_output.data(), c2r_type, &comm,
          CUFFT_COMM_MPI, &workspace));
    } else {
      auto c2c_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T1, T2>::type();
      std::size_t workspace = 0;
      KOKKOSFFT_CHECK_CUFFT_CALL(cufftMpMakePlanDecomposition(
          m_plan_f, rank, fft_extents.data(), lower_input.data(),
          upper_input.data(), strides_input.data(), lower_output.data(),
          upper_output.data(), strides_output.data(), c2c_type, &comm,
          CUFFT_COMM_MPI, &workspace));
    }
    KOKKOSFFT_CHECK_CUFFT_CALL(
        cufftXtMalloc(m_plan_f, &m_desc, CUFFT_XT_FORMAT_DISTRIBUTED_INPUT));
  }

  ScopedCufftMpPlan()                                     = delete;
  ScopedCufftMpPlan(const ScopedCufftMpPlan &)            = delete;
  ScopedCufftMpPlan &operator=(const ScopedCufftMpPlan &) = delete;
  ScopedCufftMpPlan &operator=(ScopedCufftMpPlan &&)      = delete;
  ScopedCufftMpPlan(ScopedCufftMpPlan &&)                 = delete;

  /// \brief Destructor to free cufftMp resources
  ~ScopedCufftMpPlan() noexcept {
    Kokkos::Profiling::ScopedRegion region("cleanup_plan[TPL_cuFFTMp]");
    cufftResult cufft_rt = cufftXtFree(m_desc);
    if (cufft_rt != CUFFT_SUCCESS) Kokkos::abort("cufftXtFree failed");
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      cufft_rt = cufftDestroy(m_plan_b);
    }
    cufft_rt = cufftDestroy(m_plan_f);
    if (cufft_rt != CUFFT_SUCCESS) Kokkos::abort("cufftDestroy failed");
  }

  /// \brief Get the underlying cufftHandle
  /// \param[in] direction Direction of the FFT (forward/backward)
  /// \return cufftHandle for the specified direction
  cufftHandle plan(
      [[maybe_unused]] KokkosFFT::Direction direction) const noexcept {
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      return direction == KokkosFFT::Direction::forward ? m_plan_f : m_plan_b;
    } else {
      return m_plan_f;
    }
  }

  /// \brief Get the underlying cudaLibXtDesc
  /// \return Pointer to cudaLibXtDesc
  cudaLibXtDesc *desc() const noexcept { return m_desc; }

  /// \brief Commit the plan to a specific execution space
  /// \param[in] exec_space Kokkos execution space
  void commit(const Kokkos::Cuda &exec_space) const {
    KOKKOSFFT_CHECK_CUFFT_CALL(
        cufftSetStream(m_plan_f, exec_space.cuda_stream()));
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      KOKKOSFFT_CHECK_CUFFT_CALL(
          cufftSetStream(m_plan_b, exec_space.cuda_stream()));
    }
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
