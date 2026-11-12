#include <cmath>
#include <exception>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <benchmark/benchmark.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed.hpp"
#include "Benchmark_Cases.hpp"
#include "Benchmark_Timing.hpp"
#include "Benchmark_Validation.hpp"
#include "PerfTest_FFT.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Benchmark {

namespace {

using execution_space = Kokkos::DefaultExecutionSpace;

/// \brief Free memory of the device of this rank, in bytes; 0 if unknown
inline std::size_t free_device_memory() {
  std::size_t free = 0, total = 0;
#if defined(KOKKOS_ENABLE_CUDA)
  if (cudaMemGetInfo(&free, &total) != cudaSuccess) free = 0;
#elif defined(KOKKOS_ENABLE_HIP)
  if (hipMemGetInfo(&free, &total) != hipSuccess) free = 0;
#endif
  return free;
}

/// \brief The 3D FFT benchmarks of one value type, transform type and View
/// layout. The Views are unmanaged, on allocations of their own
/// (out-of-place) or on one allocation (in-place), so that both placements
/// use the same plan types.
/// \tparam T float or double
/// \tparam IsR2C Whether the transform is R2C / C2R (C2C otherwise)
/// \tparam LayoutType Layout of the Views
template <typename T, bool IsR2C, typename LayoutType>
struct FFT3D {
  using in_value_type  = std::conditional_t<IsR2C, T, Kokkos::complex<T>>;
  using out_value_type = Kokkos::complex<T>;
  using InViewType = Kokkos::View<in_value_type***, LayoutType, execution_space,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using OutViewType =
      Kokkos::View<out_value_type***, LayoutType, execution_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using RefViewType =
      Kokkos::View<in_value_type***, LayoutType, execution_space>;
  using BasePlanType =
      KokkosFFT::Distributed::Impl::BasePlan<execution_space, InViewType,
                                             OutViewType, 3>;
  using extents_type = std::array<std::size_t, 3>;

  /// \brief Construct the plan of a kind; nullptr with the reason if the
  /// vendor plan is not available for the case
  static std::unique_ptr<BasePlanType> make_plan(
      PlanKind kind, const execution_space& exec, const InViewType& in,
      const OutViewType& out, const KokkosFFT::axis_type<3>& axes,
      const extents_type& in_topology, const extents_type& out_topology,
      MPI_Comm comm, std::string& skip_reason) {
    using namespace KokkosFFT::Distributed;
    using namespace KokkosFFT::Distributed::Impl;
    const Topology<std::size_t, 3> tin(in_topology), tout(out_topology);
    if (kind == PlanKind::factory) {
      return plan_factory(exec, in, out, axes, tin, tout, comm);
    }
    if (kind == PlanKind::tpl) {
#if defined(PRIORITIZE_TPL_PLAN_IF_AVAILABLE)
      if (!is_tpl_available(exec, in, out, axes, tin, tout)) {
        skip_reason = "the vendor plan is not available for this case";
        return nullptr;
      }
      return std::make_unique<
          TplPlan<execution_space, InViewType, OutViewType, 3>>(
          exec, in, out, axes, tin, tout, comm);
#else
      skip_reason = "no vendor plan in this build";
      return nullptr;
#endif
    }

    // The native plans, as plan_factory chooses them without a vendor plan
    KOKKOSFFT_THROW_IF(!are_valid_extents(in, out, axes, tin, tout, comm),
                       "Extents are not valid");
    switch (get_common_topology_type(tin, tout)) {
      case TopologyType::Shared:
        return std::make_unique<
            SharedPlan<execution_space, InViewType, OutViewType, 3>>(
            exec, in, out, axes, tin, tout, comm);
      case TopologyType::Slab:
        return std::make_unique<
            SlabPlan<execution_space, InViewType, OutViewType, 3>>(
            exec, in, out, axes, tin, tout, comm);
      case TopologyType::Pencil:
        return std::make_unique<
            PencilPlan<execution_space, InViewType, OutViewType, 3>>(
            exec, in, out, axes, tin, tout, comm);
      default: throw std::runtime_error("no native plan for these topologies");
    }
  }

  /// \brief The Views of a case and their allocations
  struct Data {
    extents_type m_gin, m_gout, m_in_extents, m_in_starts, m_out_extents,
        m_out_starts;
    Kokkos::View<out_value_type*, execution_space> m_shared;
    Kokkos::View<in_value_type*, execution_space> m_in_alloc;
    Kokkos::View<out_value_type*, execution_space> m_out_alloc;
    InViewType m_in;
    OutViewType m_out;
  };

  /// \brief Global and local extents of a case
  static Data extents(const FFTCase& c, MPI_Comm comm) {
    Data d;
    const auto topologies    = case_topologies(c);
    const auto& in_topology  = topologies[0];
    const auto& out_topology = topologies[1];
    d.m_gin                  = c.m_extents;
    d.m_gout                 = c.m_extents;
    if constexpr (IsR2C) {
      auto& n = d.m_gout.at(static_cast<std::size_t>(c.m_axes[2]));
      n       = n / 2 + 1;
    }
    std::tie(d.m_in_extents, d.m_in_starts) =
        KokkosFFT::Distributed::compute_local_extents_and_starts(
            d.m_gin, in_topology, comm);
    std::tie(d.m_out_extents, d.m_out_starts) =
        KokkosFFT::Distributed::compute_local_extents_and_starts(
            d.m_gout, out_topology, comm);
    return d;
  }

  /// \brief Bytes of the local input and output of a case on this rank
  static std::array<std::size_t, 2> local_bytes(const Data& d) {
    return {
        KokkosFFT::Impl::total_size(d.m_in_extents) * sizeof(in_value_type),
        KokkosFFT::Impl::total_size(d.m_out_extents) * sizeof(out_value_type)};
  }

  /// \brief Whether the case fits in the device memory of every rank: the
  /// Views, the reference copy of the validation, and three times the
  /// output for the buffers of the plans (an estimate). Collective.
  static bool fits(const Data& d, bool with_reference, MPI_Comm comm,
                   std::string& reason) {
    const auto [in_bytes, out_bytes] = local_bytes(d);
    const double need = static_cast<double>(in_bytes + out_bytes +
                                            (with_reference ? in_bytes : 0)) +
                        3.0 * static_cast<double>(out_bytes);
    const double free = static_cast<double>(free_device_memory());
    int ok = (free == 0.0 || need <= 0.9 * free) ? 1 : 0, all_ok = 0;
    ::MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, comm);
    if (all_ok == 0) {
      std::ostringstream oss;
      oss << "needs about " << need / 1e9 << " GB per rank (" << free / 1e9
          << " GB free on this rank)";
      reason = oss.str();
    }
    return all_ok != 0;
  }

  /// \brief Allocate the Views of a case
  static void allocate(Data& d, bool is_inplace) {
    const auto in_layout =
        KokkosFFT::Impl::create_layout<LayoutType>(d.m_in_extents);
    const auto out_layout =
        KokkosFFT::Impl::create_layout<LayoutType>(d.m_out_extents);
    if (is_inplace) {
      const std::size_t n =
          KokkosFFT::Distributed::compute_inplace_allocation_size<
              in_value_type, out_value_type>(d.m_in_extents, d.m_out_extents);
      d.m_shared = Kokkos::View<out_value_type*, execution_space>("shared", n);
      d.m_in  = InViewType(reinterpret_cast<in_value_type*>(d.m_shared.data()),
                           in_layout);
      d.m_out = OutViewType(d.m_shared.data(), out_layout);
    } else {
      d.m_in_alloc = Kokkos::View<in_value_type*, execution_space>(
          "in", KokkosFFT::Impl::total_size(d.m_in_extents));
      d.m_out_alloc = Kokkos::View<out_value_type*, execution_space>(
          "out", KokkosFFT::Impl::total_size(d.m_out_extents));
      d.m_in  = InViewType(d.m_in_alloc.data(), in_layout);
      d.m_out = OutViewType(d.m_out_alloc.data(), out_layout);
    }
  }

  /// \brief The largest local output over the ranks, in bytes
  static double max_out_bytes(const Data& d, MPI_Comm comm) {
    double mine = static_cast<double>(local_bytes(d)[1]), max = 0.0;
    ::MPI_Allreduce(&mine, &max, 1, MPI_DOUBLE, MPI_MAX, comm);
    return max;
  }

  /// \brief Time of the transforms of a case (B2-B8): warm-up, timed
  /// iterations, metrics, then the validation of the plan
  static void run(benchmark::State& state, const FFTCase& c) {
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank      = 0;
    ::MPI_Comm_rank(comm, &rank);
    execution_space exec;

    Data d = extents(c, comm);
    std::string reason;
    if (!fits(d, /*with_reference=*/true, comm, reason)) {
      state.SkipWithMessage(reason);
      return;
    }
    allocate(d, c.m_is_inplace);
    const auto topologies    = case_topologies(c);
    const auto& in_topology  = topologies[0];
    const auto& out_topology = topologies[1];
    auto plan = make_plan(c.m_plan_kind, exec, d.m_in, d.m_out, c.m_axes,
                          in_topology, out_topology, comm, reason);
    if (!plan) {
      state.SkipWithMessage(reason);
      return;
    }
    state.SetLabel(plan->name());

    const InViewType in   = d.m_in;
    const OutViewType out = d.m_out;
    fill_random(exec, in, rank);
    if (c.m_direction == Direction::backward) plan->forward(in, out);

    auto iteration = [&]() {
      switch (c.m_direction) {
        case Direction::forward: plan->forward(in, out); break;
        case Direction::backward: plan->backward(out, in); break;
        default:
          plan->forward(in, out);
          plan->backward(out, in);
          break;
      }
    };
    warmup(exec, comm, iteration);
    for (auto _ : state) {
      state.SetIterationTime(time_max_over_ranks(exec, comm, iteration));
    }

    // Metrics (P6): per iteration
    const double npoints = static_cast<double>(c.m_extents[0]) *
                           static_cast<double>(c.m_extents[1]) *
                           static_cast<double>(c.m_extents[2]);
    const double transforms = c.m_direction == Direction::roundtrip ? 2.0 : 1.0;
    const double flops =
        transforms * (IsR2C ? 2.5 : 5.0) * npoints * std::log2(npoints);
    const double field = max_out_bytes(d, comm);
    double a2a_bytes   = 0.0;
    for (std::size_t g : model_transposes(c)) {
      a2a_bytes += field * static_cast<double>(g - 1) / static_cast<double>(g);
    }
    a2a_bytes *= transforms;
    int nprocs = 1;
    ::MPI_Comm_size(comm, &nprocs);
    state.counters["GPUs"]    = nprocs;
    state.counters["px"]      = static_cast<double>(c.m_px);
    state.counters["py"]      = static_cast<double>(c.m_py);
    state.counters["GFLOP/s"] = benchmark::Counter(
        flops * 1e-9, benchmark::Counter::kIsIterationInvariantRate);
    state.counters["local MB (max)"]      = field * 1e-6;
    state.counters["a2a MB/rank (est)"]   = a2a_bytes * 1e-6;
    state.counters["a2a GB/s/rank (est)"] = benchmark::Counter(
        a2a_bytes * 1e-9, benchmark::Counter::kIsIterationInvariantRate);

    // Validation (P7): the forward of a plane wave, then the round trip of
    // random data. A wrong transform is reported as an error, without time.
    fill_plane_wave(exec, in, d.m_gin, d.m_in_starts);
    plan->forward(in, out);
    const double fwd_error =
        plane_wave_error(exec, out, d.m_gin, d.m_out_starts, IsR2C, comm);

    RefViewType ref("ref",
                    KokkosFFT::Impl::create_layout<LayoutType>(d.m_in_extents));
    fill_random(exec, in, rank);
    Kokkos::deep_copy(exec, ref, in);
    plan->forward(in, out);
    plan->backward(out, in);
    const double rt_error = max_difference(exec, in, ref, comm);
    exec.fence();

    state.counters["fwd error"]       = fwd_error;
    state.counters["roundtrip error"] = rt_error;
    const bool fwd_ok                 = fwd_error <= Tolerances<T>::forward;
    const bool rt_ok                  = rt_error <= Tolerances<T>::roundtrip;
    if (!fwd_ok || !rt_ok) {
      std::ostringstream oss;
      oss << "validation failed: forward error " << fwd_error << " (tolerance "
          << Tolerances<T>::forward << "), round trip error " << rt_error
          << " (tolerance " << Tolerances<T>::roundtrip << ")";
      state.SkipWithError(oss.str());
    }
  }

  /// \brief Time of the construction and destruction of the plan of a case
  /// (B1)
  static void run_creation(benchmark::State& state, const FFTCase& c) {
    MPI_Comm comm = MPI_COMM_WORLD;
    execution_space exec;
    Data d = extents(c, comm);
    std::string reason;
    if (!fits(d, /*with_reference=*/false, comm, reason)) {
      state.SkipWithMessage(reason);
      return;
    }
    allocate(d, c.m_is_inplace);
    const auto topologies    = case_topologies(c);
    const auto& in_topology  = topologies[0];
    const auto& out_topology = topologies[1];

    // Availability first, so that an unavailable vendor plan skips
    {
      auto plan = make_plan(c.m_plan_kind, exec, d.m_in, d.m_out, c.m_axes,
                            in_topology, out_topology, comm, reason);
      if (!plan) {
        state.SkipWithMessage(reason);
        return;
      }
      state.SetLabel(plan->name());
    }
    auto create = [&]() {
      std::string unused;
      auto plan = make_plan(c.m_plan_kind, exec, d.m_in, d.m_out, c.m_axes,
                            in_topology, out_topology, comm, unused);
      exec.fence();
    };
    for (auto _ : state) {
      state.SetIterationTime(time_max_over_ranks(exec, comm, create));
    }
    int nprocs = 1;
    ::MPI_Comm_size(comm, &nprocs);
    state.counters["GPUs"] = nprocs;
    state.counters["px"]   = static_cast<double>(c.m_px);
    state.counters["py"]   = static_cast<double>(c.m_py);
  }
};

/// \brief Run f, with an exception (raised on every rank: plan creation,
/// allocation) reported as the error of the benchmark, so that the next
/// benchmarks still run
template <typename F>
void guarded(benchmark::State& state, const F& f) {
  try {
    f();
  } catch (const std::exception& e) {
    state.SkipWithError(e.what());
  }
}

template <typename T, bool IsR2C, typename LayoutType>
void register_cases(const std::vector<FFTCase>& cases,
                    const std::vector<FFTCase>& creation_cases) {
  const std::string type = IsR2C ? "r2c" : "c2c";
  const std::string real = std::is_same_v<T, float> ? "float" : "double";
  const std::string layout =
      std::is_same_v<LayoutType, Kokkos::LayoutLeft> ? "left" : "right";
  using Bench = FFT3D<T, IsR2C, LayoutType>;
  for (const auto& c : cases) {
    if (IsR2C && c.m_direction == Direction::backward) continue;
    benchmark::RegisterBenchmark(case_name(type, real, layout, c),
                                 [c](benchmark::State& state) {
                                   guarded(state,
                                           [&]() { Bench::run(state, c); });
                                 })
        ->UseManualTime()
        ->Unit(benchmark::kMillisecond)
        ->ComputeStatistics("min", min_statistic);
  }
  for (const auto& c : creation_cases) {
    std::string name = case_name(type, real, layout, c);
    // plan3d/... without the direction
    name.replace(0, 5, "plan3d");
    const std::string dir = "/" + to_string(c.m_direction);
    name.erase(name.find(dir), dir.size());
    benchmark::RegisterBenchmark(
        name,
        [c](benchmark::State& state) {
          guarded(state, [&]() { Bench::run_creation(state, c); });
        })
        ->UseManualTime()
        ->Unit(benchmark::kMillisecond)
        ->ComputeStatistics("min", min_statistic);
  }
}

}  // namespace

void register_fft_benchmarks(MPI_Comm comm) {
  int nprocs = 1;
  ::MPI_Comm_size(comm, &nprocs);
  const auto p    = static_cast<std::size_t>(nprocs);
  const auto grid = pencil_grid(p);

  std::vector<Decomposition> decompositions;
  if (p == 1) {
    decompositions = {Decomposition::local};
  } else {
    decompositions = {Decomposition::slab, Decomposition::slab_same};
    if (grid[1] > 1) {
      decompositions.push_back(Decomposition::pencil);
      decompositions.push_back(Decomposition::pencil_same);
    }
  }

  std::vector<PlanKind> kinds = {PlanKind::native};
#if defined(PRIORITIZE_TPL_PLAN_IF_AVAILABLE)
  kinds.push_back(PlanKind::tpl);
  kinds.push_back(PlanKind::factory);
#endif

  const std::vector<KokkosFFT::axis_type<3>> axes_list = {{0, 1, 2}, {2, 1, 0}};

  std::vector<FFTCase> cases, creation_cases;
  for (const auto& extents : benchmark_sizes()) {
    for (auto decomposition : decompositions) {
      for (const auto& axes : axes_list) {
        for (bool is_inplace : {false, true}) {
          for (auto kind : kinds) {
            FFTCase c;
            c.m_extents          = extents;
            c.m_decomposition    = decomposition;
            c.m_axes             = axes;
            c.m_is_inplace       = is_inplace;
            c.m_plan_kind        = kind;
            const bool is_pencil = decomposition == Decomposition::pencil ||
                                   decomposition == Decomposition::pencil_same;
            c.m_px = decomposition == Decomposition::local ? 1
                     : is_pencil                           ? grid[0]
                                                           : p;
            c.m_py = is_pencil ? grid[1] : 1;

            c.m_direction = Direction::forward;
            creation_cases.push_back(c);
            if (is_inplace) {
              c.m_direction = Direction::roundtrip;
              cases.push_back(c);
            } else {
              for (auto direction : {Direction::forward, Direction::backward,
                                     Direction::roundtrip}) {
                c.m_direction = direction;
                cases.push_back(c);
              }
            }
          }
        }
      }
    }
  }

  register_cases<float, false, Kokkos::LayoutRight>(cases, creation_cases);
  register_cases<double, false, Kokkos::LayoutRight>(cases, creation_cases);
  register_cases<float, true, Kokkos::LayoutRight>(cases, creation_cases);
  register_cases<double, true, Kokkos::LayoutRight>(cases, creation_cases);
  register_cases<float, false, Kokkos::LayoutLeft>(cases, creation_cases);
  register_cases<double, false, Kokkos::LayoutLeft>(cases, creation_cases);
  register_cases<float, true, Kokkos::LayoutLeft>(cases, creation_cases);
  register_cases<double, true, Kokkos::LayoutLeft>(cases, creation_cases);
}

}  // namespace Benchmark
}  // namespace Distributed
}  // namespace KokkosFFT
