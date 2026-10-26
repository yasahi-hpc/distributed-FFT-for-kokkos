#include <mpi.h>
#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_Plan.hpp"
#include "Test_PlanUtils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using test_types      = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>,
                                    std::pair<double, Kokkos::LayoutLeft>,
                                    std::pair<double, Kokkos::LayoutRight>>;

template <typename T>
struct TestPlan1D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;

  int m_nprocs = 1;

  virtual void SetUp() { ::MPI_Comm_size(MPI_COMM_WORLD, &m_nprocs); }
};

/// \brief Fixture for plans that need a pencil decomposition: skips unless
/// the number of processes is a perfect square of at least 4
template <typename T>
struct TestPlanPencil : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;

  std::size_t m_npx = 1;

  virtual void SetUp() {
    int nprocs = 1;
    ::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const auto npx = static_cast<int>(std::sqrt(nprocs));
    if (nprocs == 1 || npx * npx != nprocs) {
      GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                      "for this test";
    }
    m_npx = static_cast<std::size_t>(npx);
  }
};

/// \brief Plan operations (see run_plan_cases) for the public
/// KokkosFFT::Distributed::Plan
struct PlanOps {
  //! Expected name of the plan implementation
  std::string m_ref_name;

  template <typename ExecutionSpace, typename InViewType, typename OutViewType,
            typename AxesType, typename InTopologyType,
            typename OutTopologyType>
  auto create(const ExecutionSpace& exec, const InViewType& in,
              const OutViewType& out, const AxesType& axes,
              const InTopologyType& in_topology,
              const OutTopologyType& out_topology, MPI_Comm comm) const {
    return KokkosFFT::Distributed::Plan(exec, in, out, axes, in_topology,
                                        out_topology, comm);
  }

  template <typename PlanType, typename InViewType, typename OutViewType>
  void forward(const PlanType& plan, const InViewType& in,
               const OutViewType& out) const {
    execute(plan, in, out, KokkosFFT::Direction::forward);
  }

  template <typename PlanType, typename InViewType, typename OutViewType>
  void backward(const PlanType& plan, const OutViewType& out,
                const InViewType& in) const {
    execute(plan, out, in, KokkosFFT::Direction::backward);
  }

  template <typename PlanType>
  void check([[maybe_unused]] const PlanType& plan) const {
#if !defined(PRIORITIZE_TPL_PLAN_IF_AVAILABLE)
    EXPECT_EQ(plan.name(), m_ref_name);
#endif
  }
};

/// \brief Plan operations that return the plan through its move
/// constructor. The moved-to plan must give the same results.
template <typename Ops>
struct MovedPlanOps : public Ops {
  template <typename... Args>
  auto create(const Args&... args) const {
    auto plan = Ops::create(args...);
    return decltype(plan)(std::move(plan));
  }
};

/// \brief The expected name of the plan implementation: with one process, the
/// plan factory chooses a plan without communication
std::string ref_plan_name(std::size_t nprocs) {
  return nprocs == 1 ? "SharedPlan" : "SlabPlan";
}

/// \brief Run the cases of the 1D FFT of 2D views
/// \tparam IsMoved Whether the plans are returned through their move
/// constructor
template <typename T, typename LayoutType, bool IsMoved>
void test_plan1D_view2D(std::size_t nprocs) {
  using RealView2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;
  using topology_type = std::array<std::size_t, 2>;

  const auto topologies =
      std::make_tuple(topology_type{1, nprocs}, topology_type{nprocs, 1});

  // (in topology, out topology, axes)
  const std::vector<PlanCase<1>> cases = {
      // topo0 -> topo0 with ax = {1}:
      // (n0, n1/p) -> (n0/p, n1) -> ((n0/2+1)/p, n1)
      // Transpose + FFT ax = {1} + Transpose
      {0, 0, {1}},
      // topo 0 -> topo 1 with ax = {0}:
      // (n0, n1/p) -> (n0/2+1, n1/p) -> ((n0 / 2 + 1) / p, n1)
      // FFT ax = {0} + Transpose
      {0, 1, {0}},
      // topology0 (n0, n1/p), ax=1 -> topology1 (n0/p, n1/2+1)
      // topo0 -> topo1 with ax = {1}:
      // (n0, n1/p) -> (n0/p, n1) -> (n0/p, n1/2+1)
      // Transpose + FFT ax = {1}
      {0, 1, {1}},
      // topo1 -> topo0 with ax = {0}: (n0/p, n1) -> (n0/2+1, n1/p)
      // Transpose + FFT ax = {0}
      {1, 0, {0}},
      // topo1 -> topo0 with ax = {1}: (n0/p, n1) -> (n0, n1/p)
      // FFT ax = {1} -> Transpose
      {1, 0, {1}},
      // topo1 -> topo1 with ax = {0}:
      // (n0/p, n1) -> (n0, n1/p) -> ((n0/2+1)/p, n1)
      // Transpose + FFT ax = {0} + Transpose
      {1, 1, {0}},
  };

  const std::array<std::size_t, 2> global_extents{8, 7};
  auto gu = make_global_input<RealView2DType>(global_extents);

  // Same name with and without the move, so that both runs are compared with
  // the same cases of the baseline
  const PlanOps plan_ops{ref_plan_name(nprocs)};
  using OpsType = std::conditional_t<IsMoved, MovedPlanOps<PlanOps>, PlanOps>;
  run_plan_cases<ComplexView2DType>("Test_Plan.cpp test_plan1D_view2D",
                                    execution_space(), gu, topologies, cases,
                                    OpsType{plan_ops});
}

/// \brief Local views of one case: the input, the output, the inverse and
/// the references
template <typename RealViewType, typename ComplexViewType>
struct CaseViews {
  RealViewType m_u, m_u_inv, m_ref_u_inv;
  ComplexViewType m_u_hat, m_ref_u_hat;
};

/// \brief Make the local views of a case from the global input
template <typename ComplexViewType, typename RealViewType,
          typename TopologyType, std::size_t FFT_DIM>
CaseViews<RealViewType, ComplexViewType> make_case_views(
    const RealViewType& gu, const TopologyType& in_topology,
    const TopologyType& out_topology,
    const KokkosFFT::axis_type<FFT_DIM>& axes) {
  using LayoutType = typename RealViewType::array_layout;
  auto gu_hat = reference_fft<ComplexViewType>(execution_space(), gu, axes);

  CaseViews<RealViewType, ComplexViewType> views;
  views.m_u         = extract_local(gu, in_topology, MPI_COMM_WORLD);
  views.m_ref_u_inv = extract_local(gu, in_topology, MPI_COMM_WORLD);
  views.m_ref_u_hat = extract_local(gu_hat, out_topology, MPI_COMM_WORLD);
  views.m_u_inv =
      RealViewType("u_inv", KokkosFFT::Impl::create_layout<LayoutType>(
                                KokkosFFT::Impl::extract_extents(views.m_u)));
  views.m_u_hat = ComplexViewType(
      "u_hat", KokkosFFT::Impl::create_layout<LayoutType>(
                   KokkosFFT::Impl::extract_extents(views.m_ref_u_hat)));
  return views;
}

/// \brief Run a plan forward and backward on the views of a case, and check
/// the results and the name of the plan implementation
/// \param[in] tol_bwd Tolerance of the backward check
template <typename PlanType, typename RealViewType, typename ComplexViewType>
void check_plan(const PlanType& plan,
                const CaseViews<RealViewType, ComplexViewType>& views,
                [[maybe_unused]] const std::string& ref_name,
                const Tolerance& tol_bwd = {}) {
  execution_space exec;
  execute(plan, views.m_u, views.m_u_hat, KokkosFFT::Direction::forward);
  EXPECT_TRUE(allclose(exec, views.m_u_hat, views.m_ref_u_hat));

  execute(plan, views.m_u_hat, views.m_u_inv, KokkosFFT::Direction::backward);
  EXPECT_TRUE(allclose(exec, views.m_u_inv, views.m_ref_u_inv, tol_bwd.m_rtol,
                       tol_bwd.m_atol));

#if !defined(PRIORITIZE_TPL_PLAN_IF_AVAILABLE)
  EXPECT_EQ(plan.name(), ref_name);
#endif
}

/// \brief Move assignment of a Plan: the plan assigned to computes the case
/// of the plan moved from, and the moved-from plan throws when used.
/// The assignment releases the plan assigned to; all the ranks do it.
template <typename T, typename LayoutType>
void test_plan1D_view2D_move_assign(std::size_t nprocs) {
  using RealView2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;
  using PlanType = KokkosFFT::Distributed::Plan<execution_space, RealView2DType,
                                                ComplexView2DType, 1>;
  using topology_type = std::array<std::size_t, 2>;
  using axes_type     = KokkosFFT::axis_type<1>;

  const topology_type topology0{1, nprocs}, topology1{nprocs, 1};
  const axes_type axes_a{1}, axes_b{0};

  const std::array<std::size_t, 2> global_extents{8, 7};
  auto gu = make_global_input<RealView2DType>(global_extents);

  // a: topo0 -> topo0 with ax = {1}, b: topo1 -> topo1 with ax = {0}
  auto views_a =
      make_case_views<ComplexView2DType>(gu, topology0, topology0, axes_a);
  auto views_b =
      make_case_views<ComplexView2DType>(gu, topology1, topology1, axes_b);

  const execution_space exec;
  PlanType plan_a(exec, views_a.m_u, views_a.m_u_hat, axes_a, topology0,
                  topology0, MPI_COMM_WORLD);
  PlanType plan_b(exec, views_b.m_u, views_b.m_u_hat, axes_b, topology1,
                  topology1, MPI_COMM_WORLD);
  check_plan(plan_a, views_a, ref_plan_name(nprocs));

  plan_a = std::move(plan_b);
  check_plan(plan_a, views_b, ref_plan_name(nprocs));

  // The moved-from plan throws, on all the ranks. Using it is the point here.
  // NOLINTBEGIN(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_THROW(execute(plan_b, views_b.m_u, views_b.m_u_hat,
                       KokkosFFT::Direction::forward),
               std::runtime_error);
  EXPECT_THROW(plan_b.name(), std::runtime_error);
  // NOLINTEND(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
}

/// \brief Plans stored in a std::vector, which moves them when it grows
template <typename T, typename LayoutType>
void test_plan1D_view2D_vector(std::size_t nprocs) {
  using RealView2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;
  using PlanType = KokkosFFT::Distributed::Plan<execution_space, RealView2DType,
                                                ComplexView2DType, 1>;
  using topology_type = std::array<std::size_t, 2>;
  using axes_type     = KokkosFFT::axis_type<1>;

  const topology_type topology0{1, nprocs}, topology1{nprocs, 1};
  const axes_type axes_a{1}, axes_b{0};

  const std::array<std::size_t, 2> global_extents{8, 7};
  auto gu = make_global_input<RealView2DType>(global_extents);

  // a: topo0 -> topo1 with ax = {1}, b: topo1 -> topo0 with ax = {0}
  auto views_a =
      make_case_views<ComplexView2DType>(gu, topology0, topology1, axes_a);
  auto views_b =
      make_case_views<ComplexView2DType>(gu, topology1, topology0, axes_b);

  const execution_space exec;
  std::vector<PlanType> plans;
  plans.reserve(1);
  plans.emplace_back(exec, views_a.m_u, views_a.m_u_hat, axes_a, topology0,
                     topology1, MPI_COMM_WORLD);

  // The second plan does not fit: the vector reallocates and moves the first
  const auto capacity = plans.capacity();
  plans.emplace_back(exec, views_b.m_u, views_b.m_u_hat, axes_b, topology1,
                     topology0, MPI_COMM_WORLD);
  EXPECT_GT(plans.capacity(), capacity);

  check_plan(plans.at(0), views_a, ref_plan_name(nprocs));
  check_plan(plans.at(1), views_b, ref_plan_name(nprocs));
}

/// \brief Moves of a Plan that holds a PencilPlan. A PencilPlan owns
/// cartesian, row and column communicators, which it frees (collective
/// MPI_Comm_free) when it is destroyed. Every ownership transition is
/// exercised: move construction, move assignment over a plan that owns a
/// PencilPlan (which is released), move assignment into a moved-from plan,
/// and destruction of the owner and of the moved-from plans. A double free or
/// an unmatched collective would make MPI abort or hang. All the ranks do the
/// same moves in the same order.
template <typename T, typename LayoutType>
void test_plan1D_view3D_pencil_moves(std::size_t npx, std::size_t npy) {
  using RealView3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<T>***, LayoutType, execution_space>;
  using PlanType = KokkosFFT::Distributed::Plan<execution_space, RealView3DType,
                                                ComplexView3DType, 1>;
  using topology_type = std::array<std::size_t, 3>;
  using axes_type     = KokkosFFT::axis_type<1>;

  // x-pencil and y-pencil
  const topology_type topology0{1, npx, npy}, topology1{npx, 1, npy};
  const axes_type axes{0};

  const std::array<std::size_t, 3> global_extents{8, 7, 5};
  auto gu = make_global_input<RealView3DType>(global_extents);

  // a: topo0 -> topo1 with ax = {0}, b: topo1 -> topo0 with ax = {0}
  auto views_a =
      make_case_views<ComplexView3DType>(gu, topology0, topology1, axes);
  auto views_b =
      make_case_views<ComplexView3DType>(gu, topology1, topology0, axes);

  const execution_space exec;
  const std::string ref_name = "PencilPlan";

  // Same tolerance as these cases in Test_PencilPlan.cpp
  const Tolerance tol_bwd{1.0e-5, 1.0e-6};
  PlanType plan_a(exec, views_a.m_u, views_a.m_u_hat, axes, topology0,
                  topology1, MPI_COMM_WORLD);
  PlanType plan_b(exec, views_b.m_u, views_b.m_u_hat, axes, topology1,
                  topology0, MPI_COMM_WORLD);
  check_plan(plan_a, views_a, ref_name, tol_bwd);
  check_plan(plan_b, views_b, ref_name, tol_bwd);

  // Move construction: c owns a's PencilPlan
  PlanType plan_c(std::move(plan_a));
  check_plan(plan_c, views_a, ref_name, tol_bwd);

  // Move assignment over a plan that owns a PencilPlan: a's PencilPlan is
  // released (its communicators are freed), and c owns b's
  plan_c = std::move(plan_b);
  check_plan(plan_c, views_b, ref_name, tol_bwd);

  // Move assignment into a moved-from plan: nothing is released
  plan_a = std::move(plan_c);
  check_plan(plan_a, views_b, ref_name, tol_bwd);

  // The moved-from plans throw, on all the ranks
  // NOLINTBEGIN(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_THROW(plan_b.name(), std::runtime_error);
  EXPECT_THROW(plan_c.name(), std::runtime_error);
  // NOLINTEND(bugprone-use-after-move,clang-analyzer-cplusplus.Move)

  // At the end of the scope, a releases b's PencilPlan, and the moved-from b
  // and c release nothing
}

}  // namespace

TYPED_TEST_SUITE(TestPlan1D, test_types);
TYPED_TEST_SUITE(TestPlanPencil, test_types);

TYPED_TEST(TestPlan1D, View2D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_plan1D_view2D<float_type, layout_type, false>(this->m_nprocs);
}

TYPED_TEST(TestPlan1D, View2D_moved) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_plan1D_view2D<float_type, layout_type, true>(this->m_nprocs);
}

TYPED_TEST(TestPlan1D, View2D_move_assign) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_plan1D_view2D_move_assign<float_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestPlan1D, View2D_vector) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_plan1D_view2D_vector<float_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestPlanPencil, View3D_moves) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_plan1D_view3D_pencil_moves<float_type, layout_type>(this->m_npx,
                                                           this->m_npx);
}
