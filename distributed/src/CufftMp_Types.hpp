#ifndef CUFFT_MP_TYPES_HPP
#define CUFFT_MP_TYPES_HPP

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include <cufftMp.h>

template <typename ExecutionSpace, typename T1, typename T2>
struct ScopedCufftMpSlabPlan {
 private:
  cufftHandle m_plan_r2c, m_plan_c2r;
  cudaLibXtDesc *m_desc;
  MPI_Comm m_comm;

 public:
  ScopedCufftMpSlabPlan(int nx, int ny, const MPI_Comm& comm, bool is_xslab) : m_comm(comm) {
    cufftResult cufft_rt = cufftCreate(&m_plan_r2c);
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      // R2C + C2R
      cufft_rt = cufftCreate(&m_plan_c2r);
      cufft_rt = cufftMpAttachComm(m_plan_r2c, CUFFT_COMM_MPI, &m_comm);
      cufft_rt = cufftMpAttachComm(m_plan_c2r, CUFFT_COMM_MPI, &m_comm);

      std::size_t workspace;

      auto r2c_type = KokkosFFT::Impl::transform_type<ExecutionSpace, T1,
                                              T2>::type();
      auto c2r_type = KokkosFFT::Impl::transform_type<ExecutionSpace, T2,
                                              T1>::type();
      cufft_rt = cufftMakePlan2d(m_plan_r2c, nx, ny, r2c_type, &workspace);
      cufft_rt = cufftMakePlan2d(m_plan_c2r, nx, ny, c2r_type, &workspace);
      KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftMakePlan2d failed");
    } else {
      // C2C
      cufft_rt = cufftMpAttachComm(m_plan_r2c, CUFFT_COMM_MPI, &m_comm);
      std::size_t workspace;
      auto c2c_type = KokkosFFT::Impl::transform_type<ExecutionSpace, T1,
                                              T2>::type();
      cufft_rt = cufftMakePlan2d(m_plan_r2c, nx, ny, c2c_type, &workspace);
      KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftMakePlan2d failed");
    }
    cufftXtSubFormat subformat_forward = is_xslab ? CUFFT_XT_FORMAT_INPLACE : CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
    cufft_rt = cufftXtMalloc(m_plan_r2c, &m_desc, subformat_forward);
    KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftXtMalloc failed");
  }

  ScopedCufftMpSlabPlan(int nx, int ny, int nz, const MPI_Comm& comm, bool is_xslab) : m_comm(comm) {
    cufftResult cufft_rt = cufftCreate(&m_plan_r2c);
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      // R2C + C2R
      cufft_rt = cufftCreate(&m_plan_c2r);
      cufft_rt = cufftMpAttachComm(m_plan_r2c, CUFFT_COMM_MPI, &m_comm);
      cufft_rt = cufftMpAttachComm(m_plan_c2r, CUFFT_COMM_MPI, &m_comm);

      std::size_t workspace;

      auto r2c_type = KokkosFFT::Impl::transform_type<ExecutionSpace, T1,
                                              T2>::type();
      auto c2r_type = KokkosFFT::Impl::transform_type<ExecutionSpace, T2,
                                              T1>::type();
      cufft_rt = cufftMakePlan3d(m_plan_r2c, nx, ny, nz, r2c_type, &workspace);
      cufft_rt = cufftMakePlan3d(m_plan_c2r, nx, ny, nz, c2r_type, &workspace);
      KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftMakePlan2d failed");
    } else {
      // C2C
      cufft_rt = cufftMpAttachComm(m_plan_r2c, CUFFT_COMM_MPI, &m_comm);
      std::size_t workspace;
      auto c2c_type = KokkosFFT::Impl::transform_type<ExecutionSpace, T1,
                                              T2>::type();
      cufft_rt = cufftMakePlan3d(m_plan_r2c, nx, ny, nz, c2c_type, &workspace);
      KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftMakePlan2d failed");
    }
    cufftXtSubFormat subformat_forward = is_xslab ? CUFFT_XT_FORMAT_INPLACE : CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
    cufft_rt = cufftXtMalloc(m_plan_r2c, &m_desc, subformat_forward);
    KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftXtMalloc failed");
  }

  // For Pencil geometry
  ScopedCufftMpSlabPlan(std::array<int, 3>& fft_extents,
                        const std::array<long long, 3>& lower_input,
                        const std::array<long long, 3>& upper_input,
                        const std::array<long long, 3>& lower_output,
                        const std::array<long long, 3>& upper_output,
                        const std::array<long long, 3>& strides_input,
                        const std::array<long long, 3>& strides_output,
                        cufftType type, const MPI_Comm& comm) : m_comm(comm) {
      //cufftResult cufft_rt = cufftCreate(&m_plan);
      //std::size_t workspace;
      //cufft_rt = cufftMpMakePlanDecomposition(m_plan, 3, fft_extents.data(), lower_input.data(), upper_input.data(),
      //                       lower_output.data(), upper_output.data(), strides_input.data(), strides_output.data(),
      //                       type, m_comm, CUFFT_COMM_MPI, &workspace);
      //KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftMpMakePlan3d failed");
    }

  ~ScopedCufftMpSlabPlan() noexcept {
    Kokkos::Profiling::ScopedRegion region(
        "cleanup_plan[TPL_cufft_mp]");
    cufftResult cufft_rt = cufftXtFree(m_desc);
    if (cufft_rt != CUFFT_SUCCESS) Kokkos::abort("cufftXtFree failed");

    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      cufft_rt = cufftDestroy(m_plan_r2c);
      cufft_rt = cufftDestroy(m_plan_c2r);
    } else {
      cufft_rt = cufftDestroy(m_plan_r2c);
    }
    if (cufft_rt != CUFFT_SUCCESS) Kokkos::abort("cufftDestroy failed");
  }

  ScopedCufftMpSlabPlan()                                        = delete;
  ScopedCufftMpSlabPlan(const ScopedCufftMpSlabPlan &)            = delete;
  ScopedCufftMpSlabPlan &operator=(const ScopedCufftMpSlabPlan &) = delete;
  ScopedCufftMpSlabPlan &operator=(ScopedCufftMpSlabPlan &&)      = delete;
  ScopedCufftMpSlabPlan(ScopedCufftMpSlabPlan &&)                 = delete;

  cufftHandle plan([[maybe_unused]] KokkosFFT::Direction direction) const noexcept {
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      return direction == KokkosFFT::Direction::forward ? m_plan_r2c : m_plan_c2r;
    } else {
      return m_plan_r2c;
    }
  }

  cudaLibXtDesc* desc() const noexcept { return m_desc; }
  void commit(const Kokkos::Cuda &exec_space) const {
    cufftResult cufft_rt;
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      cufft_rt = cufftSetStream(m_plan_r2c, exec_space.cuda_stream());
      cufft_rt = cufftSetStream(m_plan_c2r, exec_space.cuda_stream());
    } else {
      cufft_rt = cufftSetStream(m_plan_r2c, exec_space.cuda_stream());
    }
    KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftSetStream failed");
  }
};

template <typename ExecutionSpace, typename T1, typename T2>
struct InternalTplPlanType {
  using type = ScopedCufftMpSlabPlan<ExecutionSpace, T1, T2>;
};

#endif
