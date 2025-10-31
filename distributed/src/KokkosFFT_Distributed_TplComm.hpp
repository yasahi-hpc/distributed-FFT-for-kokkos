#ifndef KOKKOSFFT_DISTRIBUTED_TPL_COMM_HPP
#define KOKKOSFFT_DISTRIBUTED_TPL_COMM_HPP

#include <mpi.h>

#if defined(ENABLE_TPL_NCCL)
#include "nccl/KokkosFFT_Distributed_NCCL_Types.hpp"
#include "nccl/KokkosFFT_Distributed_NCCL_Comm.hpp"
#include "nccl/KokkosFFT_Distributed_NCCL_All2All.hpp"
#else
#include "KokkosFFT_Distributed_MPI_Types.hpp"
#include "KokkosFFT_Distributed_MPI_Comm.hpp"
#include "KokkosFFT_Distributed_MPI_All2All.hpp"
#endif

#endif
