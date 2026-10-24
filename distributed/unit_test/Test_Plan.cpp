#include <mpi.h>
#include <gtest/gtest.h>
#include <array>
#include <string>
#include <tuple>
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

//  Basically the same fixtures, used for labeling tests
template <typename T>
struct TestPlan1D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;

  int m_rank   = 0;
  int m_nprocs = 1;

  virtual void SetUp() {
    ::MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &m_nprocs);
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

template <typename T, typename LayoutType>
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

  // With one process, the plan factory chooses a plan without communication
  PlanOps ops{nprocs == 1 ? "SharedPlan" : "SlabPlan"};
  run_plan_cases<ComplexView2DType>("Test_Plan.cpp test_plan1D_view2D",
                                    execution_space(), gu, topologies, cases,
                                    ops);
}

}  // namespace

TYPED_TEST_SUITE(TestPlan1D, test_types);

TYPED_TEST(TestPlan1D, View2D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_plan1D_view2D<float_type, layout_type>(this->m_nprocs);
}
