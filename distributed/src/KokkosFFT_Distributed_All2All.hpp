#ifndef KOKKOSFFT_DISTRIBUTED_ALL2ALL_HPP
#define KOKKOSFFT_DISTRIBUTED_ALL2ALL_HPP

#include <mpi.h>
#include "KokkosFFT_Distributed_MPI_Types.hpp"
#include "KokkosFFT_Distributed_MPI_Comm.hpp"
#include "KokkosFFT_Distributed_MPI_All2All.hpp"

#if defined(ENABLE_TPL_NCCL)
#include "nccl/KokkosFFT_Distributed_NCCL_Types.hpp"
#include "nccl/KokkosFFT_Distributed_NCCL_Comm.hpp"
#include "nccl/KokkosFFT_Distributed_NCCL_All2All.hpp"
#elif defined(ENABLE_TPL_ONECCL)
#include "oneCCL/KokkosFFT_Distributed_oneCCL_Types.hpp"
#include "oneCCL/KokkosFFT_Distributed_oneCCL_Comm.hpp"
#include "oneCCL/KokkosFFT_Distributed_oneCCL_All2All.hpp"
#else
namespace KokkosFFT {
namespace Distributed {
namespace Impl {
template <typename ExecutionSpace>
using TplComm = ScopedMPIComm<ExecutionSpace>;
}
}  // namespace Distributed
}  // namespace KokkosFFT
#endif

#endif
