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
    cufftResult cufft_rt = cufftCreate(&m_plan_f);
    KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftCreate failed");
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      cufft_rt = cufftCreate(&m_plan_b);
      KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftCreate failed");
      cufft_rt = cufftMpAttachComm(m_plan_f, CUFFT_COMM_MPI, &comm);
      cufft_rt = cufftMpAttachComm(m_plan_b, CUFFT_COMM_MPI, &comm);
      KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftMpAttachComm failed");

      std::size_t workspace;
      auto r2c_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T1, T2>::type();
      auto c2r_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T2, T1>::type();
      cufft_rt = cufftMakePlan2d(m_plan_f, nx, ny, r2c_type, &workspace);
      cufft_rt = cufftMakePlan2d(m_plan_b, nx, ny, c2r_type, &workspace);
      KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftMakePlan2d failed");
    } else {
      cufft_rt = cufftMpAttachComm(m_plan_f, CUFFT_COMM_MPI, &comm);
      KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftMpAttachComm failed");
      std::size_t workspace;
      auto c2c_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T1, T2>::type();
      cufft_rt = cufftMakePlan2d(m_plan_f, nx, ny, c2c_type, &workspace);
      KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftMakePlan2d failed");
    }

    cufftXtSubFormat subformat_forward =
        is_xslab ? CUFFT_XT_FORMAT_INPLACE : CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
    cufft_rt = cufftXtMalloc(m_plan_f, &m_desc, subformat_forward);
    KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftXtMalloc failed");
  }

  /// \brief Constructor for 3D FFT plans
  /// \param[in] nx Global size in X dimension
  /// \param[in] ny Global size in Y dimension
  /// \param[in] nz Global size in Z dimension
  /// \param[in] comm MPI communicator
  /// \param[in] is_xslab Whether the topology is x-slab
  ScopedCufftMpPlan(int nx, int ny, int nz, MPI_Comm comm, bool is_xslab) {
    cufftResult cufft_rt = cufftCreate(&m_plan_f);
    KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftCreate failed");
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      cufft_rt = cufftCreate(&m_plan_b);
      KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftCreate failed");
      cufft_rt = cufftMpAttachComm(m_plan_f, CUFFT_COMM_MPI, &comm);
      cufft_rt = cufftMpAttachComm(m_plan_b, CUFFT_COMM_MPI, &comm);
      KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftMpAttachComm failed");

      std::size_t workspace;
      auto r2c_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T1, T2>::type();
      auto c2r_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T2, T1>::type();
      cufft_rt = cufftMakePlan3d(m_plan_f, nx, ny, nz, r2c_type, &workspace);
      cufft_rt = cufftMakePlan3d(m_plan_b, nx, ny, nz, c2r_type, &workspace);
      KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftMakePlan3d failed");
    } else {
      cufft_rt = cufftMpAttachComm(m_plan_f, CUFFT_COMM_MPI, &comm);
      KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftMpAttachComm failed");
      std::size_t workspace;
      auto c2c_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T1, T2>::type();
      cufft_rt = cufftMakePlan3d(m_plan_f, nx, ny, nz, c2c_type, &workspace);
      KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftMakePlan3d failed");
    }

    cufftXtSubFormat subformat_forward =
        is_xslab ? CUFFT_XT_FORMAT_INPLACE : CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
    cufft_rt = cufftXtMalloc(m_plan_f, &m_desc, subformat_forward);
    KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftXtMalloc failed");
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
  ScopedCufftMpPlan(const std::vector<int> &fft_extents,
                    const std::vector<long long int> &lower_input,
                    const std::vector<long long int> &upper_input,
                    const std::vector<long long int> &lower_output,
                    const std::vector<long long int> &upper_output,
                    const std::vector<long long int> &strides_input,
                    const std::vector<long long int> &strides_output,
                    MPI_Comm comm) {
    int rank             = fft_extents.size();
    cufftResult cufft_rt = cufftCreate(&m_plan_f);
    KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftCreate failed");
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      cufft_rt = cufftCreate(&m_plan_b);
      KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftCreate failed");
      cufft_rt = cufftMpAttachComm(m_plan_f, CUFFT_COMM_MPI, &comm);
      cufft_rt = cufftMpAttachComm(m_plan_b, CUFFT_COMM_MPI, &comm);
      KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftMpAttachComm failed");

      cufft_rt = cufftXtSetDistribution(
          m_plan_f, rank, lower_input.data(), upper_input.data(),
          lower_output.data(), upper_output.data(), strides_input.data(),
          strides_output.data());
      cufft_rt = cufftXtSetDistribution(
          m_plan_b, rank, lower_input.data(), upper_input.data(),
          lower_output.data(), upper_output.data(), strides_input.data(),
          strides_output.data());
      KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS,
                         "cufftXtSetDistribution failed");

      auto r2c_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T1, T2>::type();
      auto c2r_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T2, T1>::type();
      std::size_t workspace;
      if (rank == 2) {
        cufft_rt = cufftMakePlan2d(m_plan_f, fft_extents[0], fft_extents[1],
                                   r2c_type, &workspace);
        cufft_rt = cufftMakePlan2d(m_plan_b, fft_extents[0], fft_extents[1],
                                   c2r_type, &workspace);
        KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftMakePlan2d failed");
      } else if (rank == 3) {
        cufft_rt = cufftMakePlan3d(m_plan_f, fft_extents[0], fft_extents[1],
                                   fft_extents[2], r2c_type, &workspace);
        cufft_rt = cufftMakePlan3d(m_plan_b, fft_extents[0], fft_extents[1],
                                   fft_extents[2], c2r_type, &workspace);
        KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftMakePlan3d failed");
      } else {
        KOKKOSFFT_THROW_IF(true, "Unsupported rank for cufftMpPlan");
      }
    } else {
      cufft_rt = cufftMpAttachComm(m_plan_f, CUFFT_COMM_MPI, &comm);
      KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftMpAttachComm failed");

      cufft_rt = cufftXtSetDistribution(
          m_plan_f, rank, lower_input.data(), upper_input.data(),
          lower_output.data(), upper_output.data(), strides_input.data(),
          strides_output.data());
      KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS,
                         "cufftXtSetDistribution failed");

      auto c2c_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T1, T2>::type();
      std::size_t workspace;

      if (rank == 2) {
        cufft_rt = cufftMakePlan2d(m_plan_f, fft_extents[0], fft_extents[1],
                                   c2c_type, &workspace);
        KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftMakePlan2d failed");
      } else if (rank == 3) {
        cufft_rt = cufftMakePlan3d(m_plan_f, fft_extents[0], fft_extents[1],
                                   fft_extents[2], c2c_type, &workspace);
        KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftMakePlan3d failed");
      } else {
        KOKKOSFFT_THROW_IF(true, "Unsupported rank for cufftMpPlan");
      }
    }

    cufft_rt =
        cufftXtMalloc(m_plan_f, &m_desc, CUFFT_XT_FORMAT_DISTRIBUTED_INPUT);
    KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftXtMalloc failed");
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
      cufft_rt = cufftDestroy(m_plan_f);
      cufft_rt = cufftDestroy(m_plan_b);
    } else {
      cufft_rt = cufftDestroy(m_plan_f);
    }
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
    cufftResult cufft_rt;
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      cufft_rt = cufftSetStream(m_plan_f, exec_space.cuda_stream());
      cufft_rt = cufftSetStream(m_plan_b, exec_space.cuda_stream());
    } else {
      cufft_rt = cufftSetStream(m_plan_f, exec_space.cuda_stream());
    }
    KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftSetStream failed");
  }

  /// \brief Get the label for profiling
  /// \return Label string
  std::string label() const { return std::string("cufftMpPlan"); }
};

template <typename ExecutionSpace, typename T1, typename T2>
struct InternalTplPlanType {
  using type = ScopedCufftMpPlan<ExecutionSpace, T1, T2>;
};

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
