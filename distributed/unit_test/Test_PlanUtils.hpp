#ifndef TEST_PLAN_UTILS_HPP
#define TEST_PLAN_UTILS_HPP

#include <array>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <mpi.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_MPI_Extents.hpp"
#include "Test_Utils.hpp"

/// \brief Expected result of a plan case
enum class Expect {
  //! The plan is created, and forward and backward match the reference
  Success,

  //! The constructor of the plan throws std::runtime_error
  Throw,
};

/// \brief Tolerances of allclose
struct Tolerance {
  double m_rtol = 1.e-5;
  double m_atol = 1.e-8;
};

/// \brief One case of a plan test: a plan from the input topology to the
/// output topology over the FFT axes.
///
/// \tparam FFT_DIM Number of FFT axes
template <std::size_t FFT_DIM>
struct PlanCase {
  //! Index of the input topology in the topologies of the test
  std::size_t m_in_topo = 0;

  //! Index of the output topology in the topologies of the test
  std::size_t m_out_topo = 0;

  //! FFT axes
  KokkosFFT::axis_type<FFT_DIM> m_axes = {};

  //! Expected result
  Expect m_expect = Expect::Success;

  //! Tolerances of the forward and backward checks
  Tolerance m_tol_fwd = {}, m_tol_bwd = {};
};

/// \brief Label of a plan case, e.g. "SUCCESS topology0->topology1 axes={0,1}"
template <std::size_t FFT_DIM>
std::string to_string(const PlanCase<FFT_DIM>& plan_case) {
  std::ostringstream oss;
  oss << (plan_case.m_expect == Expect::Success ? "SUCCESS" : "THROW") << " "
      << "topology" << plan_case.m_in_topo << "->topology"
      << plan_case.m_out_topo << " axes={";
  for (std::size_t i = 0; i < FFT_DIM; ++i) {
    oss << (i == 0 ? "" : ",") << plan_case.m_axes.at(i);
  }
  oss << "}";
  return oss.str();
}

/// \brief Print the plan case on rank 0 if KOKKOSFFT_PRINT_PLAN_CASES is set.
/// Used to compare the cases run with tests-baseline/plan_cases_before_np*.txt
template <std::size_t FFT_DIM>
void print_plan_case(const std::string& name,
                     const PlanCase<FFT_DIM>& plan_case) {
  if (std::getenv("KOKKOSFFT_PRINT_PLAN_CASES") == nullptr) return;
  int rank;
  ::MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) return;

  std::ostringstream oss;
  oss << "PLAN_CASE " << name << " " << to_string(plan_case);
  if (plan_case.m_expect == Expect::Success) {
    oss << " | fwd=(" << plan_case.m_tol_fwd.m_rtol << ","
        << plan_case.m_tol_fwd.m_atol << ") bwd=(" << plan_case.m_tol_bwd.m_rtol
        << "," << plan_case.m_tol_bwd.m_atol << ")";
  }
  std::cout << oss.str() << std::endl;
}

/// \brief Allocate a global view and fill it with random numbers. The seed is
/// the same for every test, so that the data is the same on all the ranks.
///
/// \tparam ViewType Type of the global view
/// \param[in] extents Extents of the global view
/// \return The global view
template <typename ViewType>
ViewType make_global_input(
    const std::array<std::size_t, ViewType::rank()>& extents) {
  using LayoutType = typename ViewType::array_layout;
  ViewType gu("gu", KokkosFFT::Impl::create_layout<LayoutType>(extents));
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::fill_random(gu, random_pool, 1.0);
  return gu;
}

/// \brief Global extents of the output of an FFT over the axes. For R2C, the
/// last FFT axis becomes n/2+1.
///
/// \param[in] extents Global extents of the input
/// \param[in] axes FFT axes
/// \param[in] is_R2C Whether the FFT is real to complex
/// \return The global extents of the output
template <std::size_t DIM, std::size_t FFT_DIM>
std::array<std::size_t, DIM> global_out_extents(
    std::array<std::size_t, DIM> extents,
    const KokkosFFT::axis_type<FFT_DIM>& axes, bool is_R2C) {
  int last_axis = axes.back();
  if (last_axis < 0) last_axis += static_cast<int>(DIM);
  auto& n = extents.at(static_cast<std::size_t>(last_axis));
  n       = KokkosFFT::Impl::extent_after_transform(n, is_R2C);
  return extents;
}

/// \brief Reference FFT of the global input over the axes, with the
/// single-node API of KokkosFFT
///
/// \tparam OutViewType Type of the global output view
/// \param[in] exec Kokkos execution space
/// \param[in] gu Global input view
/// \param[in] axes FFT axes
/// \return The global output view
template <typename OutViewType, typename ExecutionSpace, typename InViewType,
          std::size_t FFT_DIM>
OutViewType reference_fft(const ExecutionSpace& exec, const InViewType& gu,
                          const KokkosFFT::axis_type<FFT_DIM>& axes) {
  using LayoutType = typename OutViewType::array_layout;
  constexpr bool is_R2C =
      KokkosFFT::Impl::is_real_v<typename InViewType::non_const_value_type>;

  auto out_extents =
      global_out_extents(KokkosFFT::Impl::extract_extents(gu), axes, is_R2C);
  OutViewType gu_hat("gu_hat",
                     KokkosFFT::Impl::create_layout<LayoutType>(out_extents));
  if constexpr (is_R2C) {
    KokkosFFT::rfftn(exec, gu, gu_hat, axes,
                     KokkosFFT::Normalization::backward);
  } else {
    KokkosFFT::fftn(exec, gu, gu_hat, axes, KokkosFFT::Normalization::backward);
  }
  return gu_hat;
}

template <typename ViewType, std::size_t DIM, std::size_t... Is>
auto local_subview(const ViewType& global,
                   const std::array<std::size_t, DIM>& starts,
                   const std::array<std::size_t, DIM>& extents,
                   std::index_sequence<Is...>) {
  return Kokkos::subview(global, Kokkos::pair<std::size_t, std::size_t>(
                                     starts[Is], starts[Is] + extents[Is])...);
}

/// \brief Copy the block of a global view owned by the current rank on a
/// topology into a new local view
///
/// \param[in] global Global view
/// \param[in] topology Topology (std::array or Topology)
/// \param[in] comm MPI communicator
/// \return The local view
template <typename ViewType, typename TopologyType>
ViewType extract_local(const ViewType& global, const TopologyType& topology,
                       MPI_Comm comm) {
  using LayoutType          = typename ViewType::array_layout;
  constexpr std::size_t DIM = ViewType::rank();

  auto [extents, starts] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          KokkosFFT::Impl::extract_extents(global), topology, comm);
  ViewType local(global.label() + "_local",
                 KokkosFFT::Impl::create_layout<LayoutType>(extents));
  Kokkos::deep_copy(local, local_subview(global, starts, extents,
                                         std::make_index_sequence<DIM>{}));
  return local;
}

/// \brief Call f with the I-th element of a tuple, where I is only known at
/// run time
template <std::size_t I = 0, typename Tuple, typename F>
void visit_topology(const Tuple& topologies, std::size_t idx, F&& f) {
  if constexpr (I < std::tuple_size_v<Tuple>) {
    if (idx == I) {
      f(std::get<I>(topologies));
    } else {
      visit_topology<I + 1>(topologies, idx, std::forward<F>(f));
    }
  } else {
    FAIL() << "topology index " << idx << " is out of range";
  }
}

/// \brief Run one plan case on the given topologies.
///
/// A failure case must throw on all the ranks or on none, otherwise the ranks
/// that do not throw would wait in a collective of the plan. EXPECT is used
/// rather than ASSERT for the same reason: a failed ASSERT only returns on the
/// failing rank.
///
/// \tparam OutViewType Type of the output views
/// \param[in] exec Kokkos execution space
/// \param[in] gu Global input view
/// \param[in] in_topology Input topology
/// \param[in] out_topology Output topology
/// \param[in] plan_case Plan case
/// \param[in] ops Plan operations, see run_plan_cases
template <typename OutViewType, typename ExecutionSpace, typename InViewType,
          typename InTopologyType, typename OutTopologyType,
          std::size_t FFT_DIM, typename PlanOps>
void run_plan_case(const ExecutionSpace& exec, const InViewType& gu,
                   const InTopologyType& in_topology,
                   const OutTopologyType& out_topology,
                   const PlanCase<FFT_DIM>& plan_case, const PlanOps& ops) {
  using InLayoutType  = typename InViewType::array_layout;
  using OutLayoutType = typename OutViewType::array_layout;
  MPI_Comm comm       = MPI_COMM_WORLD;

  auto gu_hat    = reference_fft<OutViewType>(exec, gu, plan_case.m_axes);
  auto u         = extract_local(gu, in_topology, comm);
  auto ref_u_inv = extract_local(gu, in_topology, comm);
  auto ref_u_hat = extract_local(gu_hat, out_topology, comm);

  InViewType u_inv("u_inv", KokkosFFT::Impl::create_layout<InLayoutType>(
                                KokkosFFT::Impl::extract_extents(u)));
  OutViewType u_hat("u_hat", KokkosFFT::Impl::create_layout<OutLayoutType>(
                                 KokkosFFT::Impl::extract_extents(ref_u_hat)));

  if (plan_case.m_expect == Expect::Throw) {
    EXPECT_THROW(
        {
          [[maybe_unused]] auto plan =
              ops.create(exec, u, u_hat, plan_case.m_axes, in_topology,
                         out_topology, comm);
        },
        std::runtime_error);
    return;
  }

  auto plan = ops.create(exec, u, u_hat, plan_case.m_axes, in_topology,
                         out_topology, comm);
  const auto& tol_fwd = plan_case.m_tol_fwd;
  const auto& tol_bwd = plan_case.m_tol_bwd;

  ops.forward(plan, u, u_hat);
  EXPECT_TRUE(allclose(exec, u_hat, ref_u_hat, tol_fwd.m_rtol, tol_fwd.m_atol));

  ops.backward(plan, u_hat, u_inv);
  EXPECT_TRUE(allclose(exec, u_inv, ref_u_inv, tol_bwd.m_rtol, tol_bwd.m_atol));

  ops.check(plan);
}

/// \brief Run the plan cases of a test.
///
/// PlanOps provides:
/// - create(exec, in, out, axes, in_topology, out_topology, comm): returns
///   the plan
/// - forward(plan, in, out) and backward(plan, out, in): execute the plan
/// - check(plan): additional checks of the plan, e.g. its name
///
/// \tparam OutViewType Type of the output views
/// \param[in] name Name of the test, printed with the cases
/// \param[in] exec Kokkos execution space
/// \param[in] gu Global input view
/// \param[in] topologies Topologies of the test, indexed by the cases
/// \param[in] cases Plan cases
/// \param[in] ops Plan operations
template <typename OutViewType, typename ExecutionSpace, typename InViewType,
          typename TopologiesType, std::size_t FFT_DIM, typename PlanOps>
void run_plan_cases(const std::string& name, const ExecutionSpace& exec,
                    const InViewType& gu, const TopologiesType& topologies,
                    const std::vector<PlanCase<FFT_DIM>>& cases,
                    const PlanOps& ops) {
  for (const auto& plan_case : cases) {
    SCOPED_TRACE(name + ": " + to_string(plan_case));
    print_plan_case(name, plan_case);
    visit_topology(topologies, plan_case.m_in_topo, [&](const auto& in_topo) {
      visit_topology(topologies, plan_case.m_out_topo,
                     [&](const auto& out_topo) {
                       run_plan_case<OutViewType>(exec, gu, in_topo, out_topo,
                                                  plan_case, ops);
                     });
    });
  }
}

#endif
