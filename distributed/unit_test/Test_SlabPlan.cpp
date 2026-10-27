#include <mpi.h>
#include <gtest/gtest.h>
#include <array>
#include <string>
#include <tuple>
#include <vector>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_SlabPlan.hpp"
#include "Test_PlanUtils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using test_types      = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>,
                                    std::pair<double, Kokkos::LayoutLeft>,
                                    std::pair<double, Kokkos::LayoutRight>>;

template <typename T>
struct TestSlabPlan : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;

  int m_nprocs = 1;

  virtual void SetUp() {
    ::MPI_Comm_size(MPI_COMM_WORLD, &m_nprocs);

    if (this->m_nprocs > 4) {
      GTEST_SKIP() << "The number of MPI processes should be smaller or equal "
                      "to 4 for this test";
    }
  }
};

// The same fixture, used for labeling tests
template <typename T>
struct TestSlab1D : public TestSlabPlan<T> {};
template <typename T>
struct TestSlab2D : public TestSlabPlan<T> {};
template <typename T>
struct TestSlab3D : public TestSlabPlan<T> {};

/// \brief Plan operations (see run_plan_cases) for SlabPlan
struct SlabPlanOps : public InternalPlanOps {
  template <typename ExecutionSpace, typename InViewType, typename OutViewType,
            typename AxesType, typename InTopologyType,
            typename OutTopologyType>
  auto create(const ExecutionSpace& exec, const InViewType& in,
              const OutViewType& out, const AxesType& axes,
              const InTopologyType& in_topology,
              const OutTopologyType& out_topology, MPI_Comm comm) const {
    return KokkosFFT::Distributed::Impl::SlabPlan<
        ExecutionSpace, InViewType, OutViewType, std::tuple_size_v<AxesType>>(
        exec, in, out, axes, in_topology, out_topology, comm);
  }
};

template <typename T, typename LayoutType>
void test_slab1D_view2D(std::size_t nprocs) {
  using View2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using float_type = KokkosFFT::Impl::base_floating_point_type<T>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<float_type>**, LayoutType, execution_space>;
  using topology_type = std::array<std::size_t, 2>;

  const auto topologies =
      std::make_tuple(topology_type{1, nprocs}, topology_type{nprocs, 1});

  // Not a slab geometry: with one process, no topology is distributed
  const std::vector<PlanCase<1>> failure_cases = {
      {0, 1, {0}, Expect::Throw},
      {0, 1, {1}, Expect::Throw},
      {1, 0, {0}, Expect::Throw},
      {1, 0, {1}, Expect::Throw},
  };

  const std::vector<PlanCase<1>> success_cases = {
      // topo0 -> topo0 with ax = {1}:
      // (n0, n1/p) -> (n0/p, n1) -> ((n0/2+1)/p, n1)
      // Transpose + FFT ax = {1} + Transpose
      {0, 0, {1}},
      // topo 0 -> topo 1 with ax = {0}:
      // (n0, n1/p) -> (n0/2+1, n1/p) -> ((n0 / 2 + 1) / p, n1)
      // FFT ax = {0} + Transpose
      {0, 1, {0}},
      // topology0 (n0, n1/p), ax=1 -> topology1 (n0/p, n1/2+1)
      // topo0 -> topo1 with ax = {1}: (n0, n1/p) -> (n0/p, n1)
      // -> (n0/p, n1/2+1)
      // Transpose + FFT ax = {1}
      {0, 1, {1}},
      // topo1 -> topo0 with ax = {0}: (n0/p, n1) -> (n0/2+1, n1/p)
      // Transpose + FFT ax = {0}
      {1, 0, {0}},
      // topo1 -> topo0 with ax = {1}: (n0/p, n1) -> (n0, n1/p)
      // FFT ax = {1} -> Transpose
      {1, 0, {1}},
      // topo1 -> topo1 with ax = {0}: (n0/p, n1) -> (n0, n1/p)
      // -> (n0/2+1, n1/p)
      // Transpose + FFT ax = {0} + Transpose
      {1, 1, {0}},
  };

  const std::array<std::size_t, 2> global_extents{8, 7};
  auto gu = make_global_input<View2DType>(global_extents);

  const std::string name = "Test_SlabPlan.cpp test_slab1D_view2D";
  const auto& cases      = nprocs == 1 ? failure_cases : success_cases;
  run_plan_cases_all_placements<ComplexView2DType>(
      name, execution_space(), gu, topologies, cases, SlabPlanOps{});
}

template <typename T, typename LayoutType>
void test_slab1D_view3D(std::size_t nprocs) {
  using View3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using float_type = KokkosFFT::Impl::base_floating_point_type<T>;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<float_type>***, LayoutType, execution_space>;
  using topology_type = std::array<std::size_t, 3>;

  const auto topologies =
      std::make_tuple(topology_type{1, 1, nprocs}, topology_type{1, nprocs, 1},
                      topology_type{nprocs, 1, 1});

  // Not a slab geometry: with one process, no topology is distributed
  const std::vector<PlanCase<1>> failure_cases = {
      // topo0 -> topo1 with ax = {0}:
      // (n0, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
      {0, 1, {0}, Expect::Throw},
      // topo0 -> topo2 with ax = {1}:
      // (n0, n1, n2/p) -> (n0/p, n1/2+1, n2)
      {0, 2, {1}, Expect::Throw},
      // topo1 -> topo2 with ax = {2}:
      // (n0, n1/p, n2) -> (n0/p, n1, n2/2+1)
      {1, 2, {2}, Expect::Throw},
      // topo0 -> topo0 with ax = {2}:
      // (n0, n1, n2/p) -> (n0, n1, (n2/2+1)/p)
      {0, 0, {2}, Expect::Throw},
      // topo1 -> topo1 with ax = {1}
      // (n0, n1/p, n2) -> (n0, (n1/2+1)/p, n2)
      {1, 1, {1}, Expect::Throw},
      // topology2 -> topology2 with ax = {1}
      // (n0/p, n1, n2) -> (n0/p, (n1/2+1), n2)
      {2, 2, {1}, Expect::Throw},
  };

  const std::vector<PlanCase<1>> success_cases = {
      // topo 0 -> topo 0 with ax = {0}:
      // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p)
      // FFT ax = {0}
      {0, 0, {0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 0 with ax = {1}:
      // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p)
      // FFT ax = {1}
      {0, 0, {1}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 0 with ax = {2}:
      // (n0, n1, n2/p) -> (n0/p, n1, n2) -> (n0, n1, (n2/2+1)/p)
      // Transpose topo 2 -> FFT ax = {2} -> Transpose topo 0
      {0, 0, {2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 1 with ax = {0}:
      // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> (n0/2+1, n1/p, n2)
      // FFT ax = {0} + Transpose 1
      {0, 1, {0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 1 with ax = {1}:
      // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0, (n1/2+1)/p, n2)
      // FFT ax = {1} -> Transpose topo 1
      {0, 1, {1}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 1 with ax = {2}:
      // (n0, n1, n2/p) -> (n0, n1/p, n2) -> (n0, n1/p, n2/2+1)
      // Transpose topo 1 ->FFT ax = {2}
      {0, 1, {2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 2 with ax = {0}:
      // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
      // FFT ax = {0} -> Transpose topo 2
      {0, 2, {0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 2 with ax = {1}:
      // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0/p, n1/2+1, n2)
      // FFT ax = {1} -> Transpose topo 2
      {0, 2, {1}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 2 with ax = {2}:
      // (n0, n1, n2/p) -> (n0/p, n1, n2) -> (n0/p, n1, n2/2+1)
      // Transpose topo 2 -> FFT ax = {2}
      {0, 2, {2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 1 -> topo 0 with ax = {0}:
      // (n0, n1/p, n2) -> (n0/2+1, n1, n2/p) -> (n0/2+1, n1, n2/p)
      // FFT ax = {0} -> Transpose topo 0
      {1, 0, {0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
  };

  const std::array<std::size_t, 3> global_extents{8, 7, 6};
  auto gu = make_global_input<View3DType>(global_extents);

  const std::string name = "Test_SlabPlan.cpp test_slab1D_view3D";
  const auto& cases      = nprocs == 1 ? failure_cases : success_cases;
  run_plan_cases_all_placements<ComplexView3DType>(
      name, execution_space(), gu, topologies, cases, SlabPlanOps{});
}

template <typename T, typename LayoutType>
void test_slab2D_view2D(std::size_t nprocs) {
  using View2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using float_type = KokkosFFT::Impl::base_floating_point_type<T>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<float_type>**, LayoutType, execution_space>;
  using topology_type = std::array<std::size_t, 2>;

  const auto topologies =
      std::make_tuple(topology_type{1, nprocs}, topology_type{nprocs, 1});

  // Not a slab geometry: with one process, no topology is distributed
  const std::vector<PlanCase<2>> failure_cases = {
      {0, 1, {0, 1}, Expect::Throw},
      {0, 1, {1, 0}, Expect::Throw},
      {1, 0, {0, 1}, Expect::Throw},
      {1, 0, {1, 0}, Expect::Throw},
  };

  const std::vector<PlanCase<2>> success_cases = {
      // topo0 -> topo0 with ax = {0, 1}:
      // (n0, n1/p) -> (n0/p, n1) -> (n0/p, n1/2+1) -> (n0, (n1/2+1)/p)
      // Transpose topo 1 -> FFT ax = {1} -> Transpose topo 0 -> FFT ax = {0}
      {0, 0, {0, 1}},
      // topo0 -> topo0 with ax = {1, 0}:
      // (n0, n1/p) -> (n0/2+1, n1/p) -> ((n0/2+1)/p, n1) -> (n0/2+1, n1/p)
      // FFT ax = {0} -> Transpose topo 1 -> FFT ax = {1}
      // -> Transpose topo 0
      {0, 0, {1, 0}},
      // topo0 -> topo1 with ax = {0, 1}:
      // (n0, n1/p) -> (n0/p, n1) -> (n0/p, n1/2+1) -> (n0, (n1/2+1)/p)
      // -> (n0/p, n1/2+1)
      // Transpose topo 1 -> FFT ax = {1} -> Transpose topo 0 -> FFT ax = {0}
      // -> Transpose topo 1
      {0, 1, {0, 1}},
      // topo0 -> topo1 with ax = {1, 0}:
      // (n0, n1/p) -> (n0/2+1, n1/p) -> ((n0/2+1)/p, n1)
      // FFT ax = {0} -> Transpose topo 1 -> FFT ax = {1}
      {0, 1, {1, 0}},
      // topo1 -> topo0 with ax = {0, 1}:
      // (n0/p, n1) -> (n0/p, n1/2+1) -> (n0, (n1/2+1)/p)
      // FFT ax = {1} -> Transpose topo 0 -> FFT ax = {0}
      {1, 0, {0, 1}},
      // topo1 -> topo0 with ax = {1, 0}:
      // (n0/p, n1) -> (n0, n1/p) -> (n0/2+1, n1/p) -> ((n0/2+1)/p, n1)
      // Transpose topo 0 -> FFT ax = {0} -> Transpose topo 1 -> FFT ax = {1}
      // -> Transpose topo 0
      {1, 0, {1, 0}},
      // topo1 -> topo1 with ax = {0, 1}:
      // (n0/p, n1) -> (n0/p, n1/2+1) -> (n0, (n1/2+1)/p)
      // FFT ax = {1} -> Transpose topo 0 -> FFT ax = {0} -> Transpose topo 1
      {1, 1, {0, 1}},
      // topo1 -> topo1 with ax = {1, 0}:
      // (n0/p, n1) -> (n0, n1/p) -> (n0/2+1, n1/p) -> ((n0/2+1)/p, n1)
      // Transpose topo 0 -> FFT ax = {0} -> Transpose topo 1 -> FFT ax = {1}
      {1, 1, {1, 0}},
  };

  const std::array<std::size_t, 2> global_extents{8, 7};
  auto gu = make_global_input<View2DType>(global_extents);

  const std::string name = "Test_SlabPlan.cpp test_slab2D_view2D";
  const auto& cases      = nprocs == 1 ? failure_cases : success_cases;
  run_plan_cases_all_placements<ComplexView2DType>(
      name, execution_space(), gu, topologies, cases, SlabPlanOps{});
}

template <typename T, typename LayoutType>
void test_slab2D_view3D(std::size_t nprocs) {
  using View3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using float_type = KokkosFFT::Impl::base_floating_point_type<T>;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<float_type>***, LayoutType, execution_space>;
  using topology_type = std::array<std::size_t, 3>;

  const auto topologies =
      std::make_tuple(topology_type{1, 1, nprocs}, topology_type{1, nprocs, 1},
                      topology_type{nprocs, 1, 1});

  // Not a slab geometry: with one process, no topology is distributed
  const std::vector<PlanCase<2>> failure_cases = {
      // topology0 -> topology1
      // (n0, n1, n2/p) -> (n0, (n1/2+1)/p, n2)
      {0, 1, {0, 1}, Expect::Throw},
      // topology0 -> topology2
      // (n0, n1, n2/p) -> (n0/p, n1/2+1, n2)
      {0, 2, {0, 1}, Expect::Throw},
      // topology1 -> topology2
      // (n0, n1/p, n2) -> (n0/p, n1, n2/2+1)
      {1, 2, {1, 2}, Expect::Throw},
      // topology0 -> topology0 with ax = {1, 2}
      // (n0, n1, n2/p) -> (n0, n1, (n2/2+1)/p)
      {0, 0, {1, 2}, Expect::Throw},
      // topology1 -> topology1 with ax = {1, 2}
      // (n0, n1/p, n2) -> (n0, (n1/2+1)/p, n2)
      {1, 1, {2, 1}, Expect::Throw},
      // topology2 -> topology2 with ax = {0, 1}
      // (n0/p, n1, n2) -> (n0/p, (n1/2+1), n2)
      {2, 2, {0, 1}, Expect::Throw},
  };

  const std::vector<PlanCase<2>> success_cases = {
      // topo 0 -> topo 0 with ax = {0, 1}:
      // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p)
      // FFT2 ax = {0, 1}
      {0, 0, {0, 1}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 0 with ax = {0, 2}:
      // (n0, n1, n2/p) -> (n0, n1/p, n2) -> (n0, n1, (n2/2+1)/p)
      // Transpose topo 1 -> FFT2 ax = {0, 2} -> Transpose topo 0
      {0, 0, {0, 2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 0 with ax = {1, 0}:
      // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p)
      // FFT2 ax = {1, 0}
      {0, 0, {1, 0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 0 with ax = {1, 2}:
      // (n0, n1, n2/p) -> (n0/p, n1, n2) -> (n0, n1, (n2/2+1)/p)
      // Transpose topo 2 -> FFT2 ax = {1, 2} -> Transpose topo 0
      {0, 0, {1, 2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 0 with ax = {2, 0}:
      // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> (n0/2+1, n1/p, n2)
      // -> (n0/2+1, n1, n2/p)
      // FFT ax = {0} -> Transpose topo 1 -> FFT ax = {2} -> Transpose topo 0
      {0, 0, {2, 0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 0 with ax = {2, 1}:
      // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0/p, n1/2+1, n2)
      // -> (n0, n1/2+1, n2/p)
      // FFT ax = {1} -> Transpose topo 2 -> FFT ax = {2}
      // -> Transpose topo 0
      {0, 0, {2, 1}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 1 with ax = {0, 1}:
      // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0, (n1/2+1)/p, n2)
      // FFT2 ax = {0, 1} -> Transpose topo 1
      {0, 1, {0, 1}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 1 with ax = {0, 2}:
      // (n0, n1, n2/p) -> (n0, n1/p, n2) -> (n0, n1/p, n2/2+1)
      // Transpose 1 -> FFT2 ax = {0, 2}
      {0, 1, {0, 2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 1 with ax = {1, 0}:
      // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> (n0/2+1, n1/p, n2)
      // FFT2 ax = {1, 0} + Transpose
      {0, 1, {1, 0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 1 with ax = {1, 2}:
      // (n0, n1, n2/p) -> (n0/p, n1, n2) -> (n0/p, n1, n2/2+1)
      // Transpose topo 2 -> FFT2 ax = {1, 2}
      {0, 1, {1, 2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 1 with ax = {2, 0}:
      // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> (n0/2+1, n1/p, n2)
      // FFT ax = {0} -> Transpose topo 1 -> FFT ax = {2}
      {0, 1, {2, 0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 1 with ax = {2, 1}:
      // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0, (n1/2+1)/p, n2)
      // FFT ax = {1} -> Transpose topo 1 -> FFT ax = {2}
      {0, 1, {2, 1}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 2 with ax = {0, 1}:
      // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0/p, n1/2+1, n2)
      // FFT2 ax = {0, 1} -> Transpose topo 2
      {0, 2, {0, 1}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 2 with ax = {0, 2}:
      // (n0, n1, n2/p) -> (n0, n1/p, n2) -> (n0, n1/p, n2/2+1)
      // -> (n0/p, n1, n2/2+1)
      // Transpose topo 1 -> FFT2 ax = {0, 2} -> Transpose topo 2
      {0, 2, {0, 2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 2 with ax = {1, 0}:
      // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
      // FFT2 ax = {1, 0} -> Transpose topo 2
      {0, 2, {1, 0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 2 with ax = {1, 2}:
      // (n0, n1, n2/p) -> (n0/p, n1, n2) -> (n0/p, n1, n2/2+1)
      // Transpose topo 2 -> FFT2 ax = {1, 2}
      {0, 2, {1, 2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 2 with ax = {2, 0}:
      // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
      // FFT ax = {0} -> Transpose topo 2 -> FFT ax = {2}
      {0, 2, {2, 0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 2 with ax = {2, 1}:
      // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0/p, n1/2+1, n2)
      // FFT ax = {1} -> Transpose topo 2 -> FFT ax = {2}
      {0, 2, {2, 1}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 1 -> topo 0 with ax = {0, 1}:
      // (n0, n1/p, n2) -> (n0, n1, n2/p) -> (n0, n1/2+1, n2)
      // Transpose topo 0 -> FFT2 ax = {0, 1}
      {1, 0, {0, 1}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 1 -> topo 0 with ax = {0, 2}:
      // (n0, n1/p, n2) -> (n0, n1/p, n2/2+1) -> (n0, n1, (n2/2+1)/p)
      // FFT ax = {2} -> Transpose topo 0 -> FFT ax = {0}
      {1, 0, {0, 2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 1 -> topo 0 with ax = {1, 0}:
      // (n0, n1/p, n2) -> (n0/2+1, n1/p, n2) -> (n0/2+1, n1, n2/p)
      // FFT ax = {0} -> Transpose topo 0 -> FFT ax = {1}
      // Transpose topo 0 -> FFT2 ax = {1, 0}
      {1, 0, {1, 0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 1 -> topo 0 with ax = {1, 2}:
      // (n0, n1/p, n2) -> (n0, n1/p, n2/2+1) -> (n0, n1, (n2/2+1)/p)
      // FFT ax = {2} -> Transpose topo 0 -> FFT ax = {1}
      {1, 0, {1, 2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 1 -> topo 0 with ax = {2, 0}:
      // (n0, n1/p, n2) -> (n0/2+1, n1/p, n2) -> (n0/2+1, n1, n2/p)
      // FFT2 ax = {2,0} -> Transpose topo 0
      {1, 0, {2, 0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 1 -> topo 0 with ax = {2, 1}:
      // (n0, n1/p, n2) -> (n0/p, n1, n2) -> (n0/p, n1/2+1, n2)
      // -> (n0, n1/2+1, n2/p)
      // Transpose topo 2 -> FFT2 ax = {2, 1} -> Transpose topo 0
      {1, 0, {2, 1}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 1 -> topo 1 with ax = {0, 1}:
      // (n0, n1/p, n2) -> (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0,
      // (n1/2+1)/p, n2) Transpose topo 0 -> FFT2 ax = {0, 1}
      // -> Transpose topo 1
      {1, 1, {0, 1}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 1 -> topo 1 with ax = {0, 2}:
      // (n0, n1/p, n2) ->(n0, n1/p, n2/2+1)
      // FFT2 ax = {0, 2}
      {1, 1, {0, 2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 1 -> topo 1 with ax = {1, 0}:
      // (n0, n1/p, n2) -> (n0/2+1, n1/p, n2) -> (n0/2+1, n1, n2/p)
      // -> (n0/2+1, n1/p, n2)
      // FFT ax = {0} -> Transpose topo 2 -> FFT ax = {1} -> Transpose topo 1
      {1, 1, {1, 0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 1 -> topo 1 with ax = {1, 2}:
      // (n0, n1/p, n2) -> (n0, n1/p, n2/2+1) -> (n0/p, n1, n2/2+1)
      // -> (n0, n1/p, n2/2+1)
      // FFT ax = {2} -> Transpose topo 2 -> FFT ax = {1} -> Transpose topo 1
      {1, 1, {1, 2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 1 -> topo 1 with ax = {2, 0}:
      // (n0, n1/p, n2) -> (n0/2+1, n1/p, n2)
      // FFT2 ax = {2, 0}
      {1, 1, {2, 0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 1 -> topo 1 with ax = {2, 1}:
      // (n0, n1/p, n2) -> (n0, n1/2+1, n2/p) -> (n0/p, n1/2+1, n2)
      // -> (n0, n1/p, n2/2+1)
      // Transpose topo 2 -> FFT2 ax = {2, 1} -> Transpose topo 1
      {1, 1, {2, 1}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 1 -> topo 2 with ax = {0, 1}:
      // (n0, n1/p, n2) -> (n0, n1, n2/p) -> (n0, n1/2+1, n2/p)
      // -> (n0/p, n1/2+1, n2)
      // Transpose topo 0 -> FFT2 ax = {0, 1} -> Transpose topo 2
      {1, 2, {0, 1}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 1 -> topo 2 with ax = {0, 2}:
      // (n0, n1/p, n2) -> (n0, n1/p, n2/2+1)
      // FFT2 ax = {0, 2}
      {1, 2, {0, 2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 1 -> topo 2 with ax = {1, 0}:
      // (n0, n1/p, n2) -> (n0/2+1, n1/p, n2) -> (n0/2+1, n1, n2/p)
      // FFT ax = {0} -> Transpose topo 2 -> FFT ax = {1}
      {1, 2, {1, 0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 1 -> topo 2 with ax = {1, 2}:
      // (n0, n1/p, n2) -> (n0, n1/p, n2/2+1) -> (n0/p, n1, n2/2+1)
      // FFT ax = {2} -> Transpose topo 2 -> FFT ax = {1}
      {1, 2, {1, 2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 1 -> topo 2 with ax = {2, 0}:
      // (n0, n1/p, n2) -> (n0/2+1, n1/p, n2) -> (n0/2+1, n1, n2/p)
      // FFT2 ax = {2, 0} -> Transpose topo 2
      {1, 2, {2, 0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 1 -> topo 2 with ax = {2, 1}:
      // (n0, n1/p, n2) -> (n0/p, n1, n2) -> (n0/p, n1/2+1, n2)
      // Transpose topo 2 -> FFT2 ax = {2, 1}
      {1, 2, {2, 1}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 2 -> topo 0 with ax = {0, 1}:
      // (n0/p, n1, n2) -> (n0, n1, n2/p) -> (n0, n1/2+1, n2/p)
      // Transpose topo 0 -> FFT2 ax = {0, 1}
      {2, 0, {0, 1}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 2 -> topo 0 with ax = {0, 2}:
      // (n0/p, n1, n2) -> (n0/p, n1, n2/2+1) -> (n0, n1, (n2/2+1)/p)
      // FFT ax = {2} -> Transpose topo 0 -> FFT ax = {0}
      {2, 0, {0, 2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 2 -> topo 0 with ax = {1, 0}:
      // (n0/p, n1, n2) -> (n0, n1, n2/p) -> (n0/2+1, n1, n2/p)
      // Transpose topo 0 -> FFT2 ax = {1, 0}
      {2, 0, {1, 0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 2 -> topo 0 with ax = {1, 2}:
      // (n0/p, n1, n2) -> (n0/p, n1, n2/2+1) -> (n0, n1, (n2/2+1)/p)
      // FFT2 ax = {1, 2} -> Transpose topo 0
      {2, 0, {1, 2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 2 -> topo 0 with ax = {2, 0}:
      // (n0/p, n1, n2) -> (n0, n1/p, n2) -> (n0/2+1, n1/p, n2)
      // -> (n0/2+1, n1, n2/p)
      // Transpose topo 1 -> FFT2 ax = {2, 0} -> Transpose topo 0
      {2, 0, {2, 0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 2 -> topo 0 with ax = {2, 1}:
      // (n0/p, n1, n2) -> (n0/p, n1/2+1, n2) -> (n0, n1/2+1, n2/p)
      // FFT2 ax = {2, 1} -> Transpose topo 0
      {2, 0, {2, 1}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 2 -> topo 1 with ax = {0, 1}:
      // (n0/p, n1, n2) -> (n0/p, n1/2+1, n2) -> (n0, (n1/2+1)/p, n2)
      //  FFT ax = {1} -> Transpose topo 1 -> FFT ax = {0}
      {2, 1, {0, 1}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 2 -> topo 1 with ax = {0, 2}:
      // (n0/p, n1, n2) -> (n0/p, n1, n2/2+1) -> (n0, n1/p, n2/2+1)
      // FFT ax = {2} -> Transpose topo 1 -> FFT ax = {0}
      {2, 1, {0, 2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 2 -> topo 1 with ax = {1, 0}:
      // (n0/p, n1, n2) -> (n0, n1, n2/p) -> (n0/2+1, n1, n2/p)
      // -> (n0/2+1, n1/p, n2)
      // Transpose topo 0 -> FFT2 ax = {1, 0} -> Transpose topo 1
      {2, 1, {1, 0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 2 -> topo 1 with ax = {1, 2}:
      // (n0/p, n1, n2) -> (n0/p, n1, n2/2+1) -> (n0, n1/p, n2/2+1)
      // FFT2 ax = {1, 2} -> Transpose topo 1
      {2, 1, {1, 2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 2 -> topo 1 with ax = {2, 0}:
      // (n0/p, n1, n2) -> (n0, n1/p, n2) -> (n0/2+1, n1/p, n2)
      // Transpose topo 1 -> FFT2 ax = {2, 0}
      {2, 1, {2, 0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 2 -> topo 1 with ax = {2, 1}:
      // (n0/p, n1, n2) -> (n0/p, n1/2+1, n2) -> (n0, (n1/2+1)/p, n2)
      // FFT2 ax = {2, 1} -> Transpose topo 1
      {2, 1, {2, 1}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 2 -> topo 2 with ax = {0, 1}:
      // (n0/p, n1, n2) -> (n0/p, n1/2+1, n2) -> (n0, (n1/2+1)/p, n2)
      // -> (n0/p, n1/2+1, n2)
      // FFT ax = {1} -> Transpose topo 1 -> FFT ax = {0} -> Transpose topo 2
      {2, 2, {0, 1}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 2 -> topo 2 with ax = {0, 2}:
      // (n0/p, n1, n2) -> (n0/p, n1, n2/2+1) -> (n0, n1/p, n2/2+1)
      // -> (n0/p, n1, n2/2+1)
      // FFT ax = {2} -> Transpose topo 1 -> FFT ax = {0} -> Transpose topo 2
      {2, 2, {0, 2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 2 -> topo 2 with ax = {1, 0}:
      // (n0/p, n1, n2) -> (n0, n1, n2/p) -> (n0/2+1, n1, n2/p)
      // -> ((n0/2+1)/p, n1, n2)
      // Transpose topo 0 -> FFT2 ax = {1, 0} -> Transpose topo 2
      {2, 2, {1, 0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 2 -> topo 2 with ax = {1, 2}:
      // (n0/p, n1, n2) -> (n0/p, n1, n2/2+1)
      // FFT2 ax = {1, 2}
      {2, 2, {1, 2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 2 -> topo 2 with ax = {2, 0}:
      // (n0/p, n1, n2) -> (n0, n1/p, n2) -> (n0/2+1, n1/p, n2)
      // -> ((n0/2+1)/p, n1, n2)
      // Transpose topo 1 -> FFT2 ax = {2, 0} -> Transpose topo 2
      {2, 2, {2, 0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 2 -> topo 2 with ax = {2, 1}:
      // (n0/p, n1, n2) -> (n0/p, n1/2+1, n2)
      // FFT2 ax = {2, 1}
      {2, 2, {2, 1}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
  };

  const std::array<std::size_t, 3> global_extents{8, 7, 6};
  auto gu = make_global_input<View3DType>(global_extents);

  const std::string name = "Test_SlabPlan.cpp test_slab2D_view3D";
  const auto& cases      = nprocs == 1 ? failure_cases : success_cases;
  run_plan_cases_all_placements<ComplexView3DType>(
      name, execution_space(), gu, topologies, cases, SlabPlanOps{});
}

template <typename T, typename LayoutType>
void test_slab3D_view3D(std::size_t nprocs) {
  using View3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using float_type = KokkosFFT::Impl::base_floating_point_type<T>;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<float_type>***, LayoutType, execution_space>;
  using topology_type = std::array<std::size_t, 3>;

  const auto topologies =
      std::make_tuple(topology_type{1, 1, nprocs}, topology_type{1, nprocs, 1},
                      topology_type{nprocs, 1, 1});

  // Not a slab geometry: with one process, no topology is distributed
  const std::vector<PlanCase<3>> failure_cases = {
      // topology0 -> topology1
      // (n0, n1, n2/p) -> (n0, (n1/2+1)/p, n2)
      {0, 1, {0, 1, 2}, Expect::Throw},
      // topology0 -> topology2
      // (n0, n1, n2/p) -> (n0/p, n1/2+1, n2)
      {0, 2, {0, 1, 2}, Expect::Throw},
      // topology1 -> topology2
      // (n0, n1/p, n2) -> (n0/p, n1, n2/2+1)
      {1, 2, {0, 1, 2}, Expect::Throw},
      // topology0 -> topology0 with ax = {1, 2}
      // (n0, n1, n2/p) -> (n0, n1, (n2/2+1)/p)
      {0, 0, {0, 1, 2}, Expect::Throw},
      // topology1 -> topology1 with ax = {1, 2}
      // (n0, n1/p, n2) -> (n0, (n1/2+1)/p, n2)
      {1, 1, {2, 1, 0}, Expect::Throw},
      // topology2 -> topology2 with ax = {0, 1, 2}
      // (n0/p, n1, n2) -> (n0/p, (n1/2+1), n2)
      {2, 2, {0, 1, 2}, Expect::Throw},
  };

  const std::vector<PlanCase<3>> success_cases = {
      // topo 0 -> topo 0 with ax = {0, 1, 2}:
      // (n0, n1, n2/p) -> (n0/p, n1, n2) -> (n0/p, n1, n2/2+1) -> (n0, n1,
      // (n2/2+1)/p) Transpose -> FFT2 ax = {1, 2} -> Transpose -> FFT ax = {0}
      {0, 0, {0, 1, 2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 0 with ax = {0, 2, 1}:
      // (n0, n1, n2/p) -> (n0/p, n1, n2) -> (n0/p, n1/2+1, n2) -> (n0, n1/2+1,
      // n2/p) FFT ax = {1} -> Transpose topo 1 -> FFT2 ax = {0, 2} -> Transpose
      {0, 0, {0, 2, 1}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 0 with ax = {1, 0, 2}:
      // (n0, n1, n2/p) -> (n0, n1/p, n2) -> (n0, n1/p, n2/2+1) -> (n0, n1,
      // (n2/2+1)/p) Transpose -> FFT2 ax = {0, 2} -> Transpose -> FFT ax = {1}
      {0, 0, {1, 0, 2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 0 with ax = {1, 2, 0}:
      // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
      // -> (n0/2+1,
      // n1, n2/p) FFT ax = {0} -> Transpose topo 2 -> FFT2 ax = {1, 2} ->
      // Transpose
      {0, 0, {1, 2, 0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 0 with ax = {2, 0, 1}:
      // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0, (n1/2+1)/p, n2)
      // -> (n0, n1/2+1, n2/p)
      // FFT2 ax = {0, 1} -> Transpose topo 1 -> FFT ax = {2} ->
      // Transpose
      {0, 0, {2, 0, 1}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 0 with ax = {2, 1, 0}:
      // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
      // -> (n0/2+1,
      // n1, n2/p) FFT2 ax = {1, 0} -> Transpose topo 2 -> FFT ax = {2} ->
      // Transpose
      {0, 0, {2, 1, 0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 1 with ax = {0, 1, 2}:
      // (n0, n1, n2/p) -> (n0/p, n1, n2) -> (n0/p, n1, n2/2+1)
      // -> (n0, n1/p, n2/2+1)
      // Transpose -> FFT2 ax = {1, 2} -> Transpose topo 1 -> FFT ax = {1}
      {0, 1, {0, 1, 2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 1 with ax = {0, 2, 1}:
      // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0, (n1/2+1)/p, n2)
      // FFT ax = {1} -> Transpose topo 1 -> FFT2 ax = {0, 2}
      {0, 1, {0, 2, 1}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 1 with ax = {1, 0, 2}:
      // (n0, n1, n2/p) -> (n0, n1/p, n2) -> (n0, n1/p, n2/2+1)
      // -> (n0/p, n1, n2/2+1) -> (n0, n1/p, n2/2+1)
      // Transpose topo 1 -> FFT2 ax = {0, 2} -> Transpose topo 2
      // -> FFT ax = {1}
      // -> Transpose topo 1
      {0, 1, {1, 0, 2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 1 with ax = {1, 2, 0}:
      // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
      // -> (n0/2+1, n1/p, n2)
      // FFT ax = {0} -> Transpose topo 2 -> FFT2 ax = {1, 2} ->
      // Transpose topo 1
      {0, 1, {1, 2, 0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 1 with ax = {2, 0, 1}:
      // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0, (n1/2+1)/p, n2)
      // FFT2 ax = {0, 1} -> Transpose topo 1 -> FFT ax = {2}
      {0, 1, {2, 0, 1}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 1 with ax = {2, 1, 0}:
      // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> (n0/2+1, n1/p, n2)
      // FFT2 ax = {1, 0} -> Transpose topo 1 -> FFT ax = {2}
      {0, 1, {2, 1, 0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 2 with ax = {0, 1, 2}:
      // (n0, n1, n2/p) -> (n0/p, n1, n2) -> (n0/p, n1, n2/2+1)
      // -> (n0, n1/p, n2/2+1) -> (n0/p, n1, n2/2+1)
      // Transpose topo 2 -> FFT2 ax = {1, 2} -> Transpose topo 1
      // -> FFT ax = {2}
      // Transpose topo 2
      {0, 2, {0, 1, 2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 2 with ax = {0, 2, 1}:
      // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0, (n1/2+1)/p, n2)
      // -> (n0/p, n1/2+1, n2)
      // FFT ax = {1} -> Transpose topo 1 -> FFT2 ax = {0, 2} ->
      // Transpose topo 2
      {0, 2, {0, 2, 1}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 2 with ax = {1, 0, 2}:
      // (n0, n1, n2/p) -> (n0, n1/p, n2) -> (n0, n1/p, n2/2+1)
      // -> (n0/p, n1, n2/2+1)
      // Transpose topo 1 -> FFT2 ax = {0, 2} -> Transpose topo 2
      // -> FFT ax = {1}
      {0, 2, {1, 0, 2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 2 with ax = {1, 2, 0}:
      // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
      // FFT ax = {0} -> Transpose topo 2 -> FFT2 ax = {1, 2}
      {0, 2, {1, 2, 0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 2 with ax = {2, 0, 1}:
      // (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) -> (n0/p, n1/2+1, n2)
      // FFT2 ax = {0, 1} -> Transpose topo 2 -> FFT ax = {2}
      {0, 2, {2, 0, 1}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 2 with ax = {2, 1, 0}:
      // (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) -> ((n0/2+1)/p, n1, n2)
      // FFT2 ax = {0, 1} -> Transpose topo 2 -> FFT ax = {2}
      {0, 2, {2, 1, 0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 2 -> topo 1 with ax = {0, 1, 2}:
      // (n0/p, n1, n2) -> (n0/p, n1, n2/2+1) -> (n0, n1/p, n2/2+1)
      // FFT2 ax = {1, 2} -> Transpose -> FFT ax = {0}
      {2, 1, {0, 1, 2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
  };

  const std::array<std::size_t, 3> global_extents{8, 7, 6};
  auto gu = make_global_input<View3DType>(global_extents);

  const std::string name = "Test_SlabPlan.cpp test_slab3D_view3D";
  const auto& cases      = nprocs == 1 ? failure_cases : success_cases;
  run_plan_cases_all_placements<ComplexView3DType>(
      name, execution_space(), gu, topologies, cases, SlabPlanOps{});
}

template <typename T, typename LayoutType>
void test_slab3D_view4D(std::size_t nprocs) {
  using View4DType        = Kokkos::View<T****, LayoutType, execution_space>;
  using float_type        = KokkosFFT::Impl::base_floating_point_type<T>;
  using ComplexView4DType = Kokkos::View<Kokkos::complex<float_type>****,
                                         LayoutType, execution_space>;
  using topology_type     = std::array<std::size_t, 4>;

  const auto topologies = std::make_tuple(
      topology_type{1, 1, 1, nprocs}, topology_type{1, 1, nprocs, 1},
      topology_type{1, nprocs, 1, 1}, topology_type{nprocs, 1, 1, 1});

  // Not a slab geometry: with one process, no topology is distributed
  const std::vector<PlanCase<3>> failure_cases = {
      // topology1 -> topology2 with ax = {0, 1, 2}
      // (n0, n1/p, n2, n3) -> (n0, n1, (n2/2+1)/p, n3)
      {1, 2, {0, 1, 2}, Expect::Throw},
      // topology0 -> topology0 with ax = {0, 1, 2}
      // (n0, n1, n2, n3/p) -> (n0, n1, n2/2+1, n3/p)
      {0, 0, {0, 1, 2}, Expect::Throw},
      // topology1 -> topology1 with ax = {2, 1, 0}
      // (n0, n1/p, n2, n3) -> (n0/2+1, n1/p, n2, n3)
      {1, 1, {2, 1, 0}, Expect::Throw},
      // topology2 -> topology2 with ax = {0, 1, 2}
      // (n0, n1/p, n2, n3) -> (n0, n1/p, n2/2+1, n3)
      {2, 2, {0, 1, 2}, Expect::Throw},
  };

  const std::vector<PlanCase<3>> success_cases = {
      // topo 0 -> topo 0 with ax = {0, 1, 2}:
      // (n0, n1, n2, n3/p) -> (n0, n1, n2/2+1, n3/p)
      // FFT3 ax = {0, 1, 2}
      {0, 0, {0, 1, 2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 0 with ax = {0, 2, 1}:
      // (n0, n1, n2, n3/p) -> (n0, n1/2+1, n2, n3/p)
      // FFT3 ax = {0, 2, 1}
      {0, 0, {0, 2, 1}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 0 with ax = {1, 0, 2}:
      // (n0, n1, n2, n3/p) -> (n0, n1, n2/2+1, n3/p)
      // FFT3 ax = {1, 0, 2}
      {0, 0, {1, 0, 2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 0 with ax = {1, 2, 0}:
      // (n0, n1, n2, n3/p) -> (n0/2+1, n1, n2, n3/p)
      // FFT3 ax = {1, 2, 0}
      {0, 0, {1, 2, 0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 0 with ax = {2, 0, 1}:
      // (n0, n1, n2, n3/p) -> (n0, n1/2+1, n2, n3/p)
      // FFT3 ax = {2, 0, 1}
      {0, 0, {2, 0, 1}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 0 with ax = {2, 1, 0}:
      // (n0, n1, n2, n3/p) -> (n0/2+1, n1, n2, n3/p)
      // FFT3 ax = {2, 1, 0}
      {0, 0, {2, 1, 0}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 0 with ax = {1, 2, 3}:
      // (n0, n1, n2, n3/p) -> (n0, n1/p, n2, n3) -> (n0, n1/p, n2, n3/2+1)
      // -> (n0, n1, n2, (n3/2+1)/p)
      // Transpose topo 2 -> FFT2 ax = {2, 3} -> Transpose topo 0
      // FFT ax = {1}
      {0, 0, {1, 2, 3}, Expect::Success, {1.0e-5, 1.0e-5}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 0 with ax = {1, 3, 2}:
      // (n0, n1, n2, n3/p) -> (n0, n1, n2/2+1, n3/p)
      // -> (n0, n1, (n2/2+1)/p, n3)
      // -> (n0, n1, n2/2+1, n3/p)
      // FFT ax = {2} -> Transpose topo 1 -> FFT2 ax = {1, 3}
      // -> Transpose topo 0
      {0, 0, {1, 3, 2}, Expect::Success, {1.0e-5, 1.0e-5}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 0 with ax = {2, 1, 3}:
      // (n0, n1, n2, n3/p) -> (n0/p, n1, n2, n3) -> (n0/p, n1, n2, n3/2+1)
      // -> (n0, n1, n2, (n3/2+1)/p)
      // Transpose topo 3 -> FFT3 ax = {2, 1, 3} -> Transpose topo 0
      {0, 0, {2, 1, 3}, Expect::Success, {1.0e-5, 1.0e-6}, {1.0e-5, 1.0e-6}},
      // topo 0 -> topo 1 with ax = {0, 1, 2}:
      // (n0, n1, n2, n3/p) -> (n0, n1, n2/2+1, n3/p) -> (n0, n1, n2/p, n3)
      // FFT3 ax = {0, 1, 2} -> Transpose topo 1
      {0, 1, {0, 1, 2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 1 -> topo 0 with ax = {0, 1, 2}:
      // (n0, n1, n2/p, n3) -> (n0, n1, n2, n3/p) -> (n0, n1, n2/2+1, n3/p)
      // Transpose topo 0 -> FFT3 ax = {0, 1, 2}
      {1, 0, {0, 1, 2}, Expect::Success, {}, {1.0e-5, 1.0e-6}},
      // topo 2 -> topo 1 with ax = {1, 2, 3}:
      // (n0, n1/p, n2, n3) -> (n0, n1/p, n2, n3/2+1) -> (n0, n1, n2/p, n3/2+1)
      // FFT2 ax = {2, 3} -> Transpose -> FFT ax = {1}
      {2, 1, {1, 2, 3}, Expect::Success, {1.0e-5, 1.0e-5}, {1.0e-5, 1.0e-6}},
  };

  const std::array<std::size_t, 4> global_extents{8, 7, 6, 6};
  auto gu = make_global_input<View4DType>(global_extents);

  const std::string name = "Test_SlabPlan.cpp test_slab3D_view4D";
  const auto& cases      = nprocs == 1 ? failure_cases : success_cases;
  run_plan_cases_all_placements<ComplexView4DType>(
      name, execution_space(), gu, topologies, cases, SlabPlanOps{});
}

}  // namespace

TYPED_TEST_SUITE(TestSlab1D, test_types);
TYPED_TEST_SUITE(TestSlab2D, test_types);
TYPED_TEST_SUITE(TestSlab3D, test_types);

TYPED_TEST(TestSlab1D, View2D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_slab1D_view2D<float_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestSlab1D, View2D_C2C) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;

  test_slab1D_view2D<complex_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestSlab1D, View3D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_slab1D_view3D<float_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestSlab1D, View3D_C2C) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;

  test_slab1D_view3D<complex_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestSlab2D, View2D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_slab2D_view2D<float_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestSlab2D, View2D_C2C) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;

  test_slab2D_view2D<complex_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestSlab2D, View3D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_slab2D_view3D<float_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestSlab2D, View3D_C2C) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;

  test_slab2D_view3D<complex_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestSlab3D, View3D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_slab3D_view3D<float_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestSlab3D, View3D_C2C) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;

  test_slab3D_view3D<complex_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestSlab3D, View4D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_slab3D_view4D<float_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestSlab3D, View4D_C2C) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;

  test_slab3D_view4D<complex_type, layout_type>(this->m_nprocs);
}
