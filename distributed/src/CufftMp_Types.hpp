#ifndef CUFFT_MP_TYPES_HPP
#define CUFFT_MP_TYPES_HPP

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include <cufftMp.h>

template <typename ExecutionSpace, typename T1, typename T2>
struct ScopedCufftMpPlan {
  cufftHandle m_plan_f = 0, m_plan_b = 0;
  cudaLibXtDesc *m_desc;
  MPI_Comm m_comm;

 public:
  ScopedCufftMpPlan(int nx, int ny, const MPI_Comm &comm, bool is_xslab)
      : m_comm(comm) {
    cufftResult cufft_rt = cufftCreate(&m_plan_f);
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      cufft_rt = cufftCreate(&m_plan_b);
      cufft_rt = cufftMpAttachComm(m_plan_f, CUFFT_COMM_MPI, &m_comm);
      cufft_rt = cufftMpAttachComm(m_plan_b, CUFFT_COMM_MPI, &m_comm);

      std::size_t workspace;
      auto r2c_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T1, T2>::type();
      auto c2r_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T2, T1>::type();
      cufft_rt = cufftMakePlan2d(m_plan_f, nx, ny, r2c_type, &workspace);
      cufft_rt = cufftMakePlan2d(m_plan_b, nx, ny, c2r_type, &workspace);
      KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftMakePlan2d failed");
    } else {
      cufft_rt = cufftMpAttachComm(m_plan_f, CUFFT_COMM_MPI, &m_comm);
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

  ScopedCufftMpPlan(int nx, int ny, int nz, const MPI_Comm &comm, bool is_xslab)
      : m_comm(comm) {
    cufftResult cufft_rt = cufftCreate(&m_plan_f);
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      cufft_rt = cufftCreate(&m_plan_b);
      cufft_rt = cufftMpAttachComm(m_plan_f, CUFFT_COMM_MPI, &m_comm);
      cufft_rt = cufftMpAttachComm(m_plan_b, CUFFT_COMM_MPI, &m_comm);

      std::size_t workspace;
      auto r2c_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T1, T2>::type();
      auto c2r_type =
          KokkosFFT::Impl::transform_type<ExecutionSpace, T2, T1>::type();
      cufft_rt = cufftMakePlan3d(m_plan_f, nx, ny, nz, r2c_type, &workspace);
      cufft_rt = cufftMakePlan3d(m_plan_b, nx, ny, nz, c2r_type, &workspace);
      KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftMakePlan3d failed");
    } else {
      cufft_rt = cufftMpAttachComm(m_plan_f, CUFFT_COMM_MPI, &m_comm);
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

  ScopedCufftMpPlan()                                     = delete;
  ScopedCufftMpPlan(const ScopedCufftMpPlan &)            = delete;
  ScopedCufftMpPlan &operator=(const ScopedCufftMpPlan &) = delete;
  ScopedCufftMpPlan &operator=(ScopedCufftMpPlan &&)      = delete;
  ScopedCufftMpPlan(ScopedCufftMpPlan &&)                 = delete;

  cufftHandle plan(
      [[maybe_unused]] KokkosFFT::Direction direction) const noexcept {
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      return direction == KokkosFFT::Direction::forward ? m_plan_f : m_plan_b;
    } else {
      return m_plan_f;
    }
  }
  cudaLibXtDesc *desc() const noexcept { return m_desc; }

  ~ScopedCufftMpPlan() noexcept {
    cufftResult cufft_rt = cufftXtFree(m_desc);
    if constexpr (KokkosFFT::Impl::is_real_v<T1>) {
      cufft_rt = cufftDestroy(m_plan_f);
      cufft_rt = cufftDestroy(m_plan_b);
    } else {
      cufft_rt = cufftDestroy(m_plan_f);
    }
  }

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
};

template <typename ExecutionSpace, typename T1, typename T2>
struct InternalTplPlanType {
  using type = ScopedCufftMpPlan<ExecutionSpace, T1, T2>;
};

#endif
