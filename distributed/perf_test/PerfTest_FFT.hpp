#ifndef KOKKOSFFT_PERFTEST_FFT_HPP
#define KOKKOSFFT_PERFTEST_FFT_HPP

#include <mpi.h>

namespace KokkosFFT {
namespace Distributed {
namespace Benchmark {

/// \brief Register the 3D FFT benchmarks (fft3d/...) and the plan creation
/// benchmarks (plan3d/...) for the number of processes of comm: every size
/// of KOKKOSFFT_BENCHMARK_SIZES, C2C and R2C, float and double, LayoutRight
/// and LayoutLeft Views, the decompositions the process count allows,
/// natural and permuted axes, out-of-place and in-place, the directions,
/// and the plan kinds of the build (native; tpl and factory with a vendor
/// plan). The same on every rank. Select with --benchmark_filter.
void register_fft_benchmarks(MPI_Comm comm);

}  // namespace Benchmark
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
