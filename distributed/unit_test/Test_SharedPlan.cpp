#include <mpi.h>
#include <gtest/gtest.h>
#include <array>
#include <string>
#include <tuple>
#include <vector>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_SharedPlan.hpp"
#include "Test_PlanUtils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using test_types      = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>,
                                    std::pair<double, Kokkos::LayoutLeft>,
                                    std::pair<double, Kokkos::LayoutRight>>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct TestShared1D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;

  int m_rank   = 0;
  int m_nprocs = 1;

  virtual void SetUp() {
    ::MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &m_nprocs);
  }
};

template <typename T>
struct TestShared2D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;

  int m_rank   = 0;
  int m_nprocs = 1;

  virtual void SetUp() {
    ::MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &m_nprocs);
  }
};

template <typename T>
struct TestShared3D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;

  int m_rank   = 0;
  int m_nprocs = 1;

  virtual void SetUp() {
    ::MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &m_nprocs);
  }
};

/// \brief Plan operations (see run_plan_cases) for SharedPlan
struct SharedPlanOps : public InternalPlanOps {
  template <typename ExecutionSpace, typename InViewType, typename OutViewType,
            typename AxesType, typename InTopologyType,
            typename OutTopologyType>
  auto create(const ExecutionSpace& exec, const InViewType& in,
              const OutViewType& out, const AxesType& axes,
              const InTopologyType& in_topology,
              const OutTopologyType& out_topology, MPI_Comm comm) const {
    return KokkosFFT::Distributed::Impl::SharedPlan<
        ExecutionSpace, InViewType, OutViewType, std::tuple_size_v<AxesType>>(
        exec, in, out, axes, in_topology, out_topology, comm);
  }
};

template <typename T, typename LayoutType>
void test_shared1D_view2D(std::size_t nprocs) {
  using View2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using float_type = KokkosFFT::Impl::base_floating_point_type<T>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<float_type>**, LayoutType, execution_space>;
  using topology_type = std::array<std::size_t, 2>;

  const auto topologies =
      std::make_tuple(topology_type{1, nprocs}, topology_type{nprocs, 1});

  // The FFT axis is distributed on the input or output topology. These are
  // slab geometries, so the shared plan must fail with more than one process.
  const std::vector<PlanCase<1>> failure_cases = {
      // topology0 -> topology1
      // (n0, n1/p) -> ((n0/2+1)/p, n1)
      // This is slab
      {0, 1, {0}, Expect::Throw},
      // topology1 -> topology0
      // (n0/p, n1) -> (n0/2+1, n1/p)
      // This is slab
      {1, 0, {0}, Expect::Throw},
      // topology1 -> topology1
      // (n0/p, n1) -> ((n0/2+1)/p, n1)
      // This is slab
      {1, 1, {0}, Expect::Throw},
      // topology0 (n0, n1/p) -> (n0, (n1/2+1)/p)
      {0, 0, {1}, Expect::Throw},
      // topology0 -> topology1
      // (n0, n1/p) -> (n0/p, n1/2+1)
      // This is slab
      {0, 1, {1}, Expect::Throw},
      // topology1 -> topology0
      // (n0/p, n1) -> (n0, (n1/2+1)/p)
      // This is slab
      {1, 0, {1}, Expect::Throw},
  };

  const std::vector<PlanCase<1>> success_cases = {
      // topology0 (n0, n1/p) -> (n0/2+1, n1/p)
      {0, 0, {0}},
      // topology1 -> topology1
      // (n0/p, n1) -> (n0/p, n1/2+1)
      {1, 1, {1}},
  };

  const std::array<std::size_t, 2> global_extents{8, 7};
  auto gu = make_global_input<View2DType>(global_extents);

  const std::string name = "Test_SharedPlan.cpp test_shared1D_view2D";
  if (nprocs != 1) {
    run_plan_cases<ComplexView2DType>(name, execution_space(), gu, topologies,
                                      failure_cases, SharedPlanOps{});
  }
  run_plan_cases<ComplexView2DType>(name, execution_space(), gu, topologies,
                                    success_cases, SharedPlanOps{});
}

template <typename T, typename LayoutType>
void test_shared2D_view3D(std::size_t nprocs) {
  using View3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using float_type = KokkosFFT::Impl::base_floating_point_type<T>;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<float_type>***, LayoutType, execution_space>;
  using topology_type = std::array<std::size_t, 3>;

  const auto topologies =
      std::make_tuple(topology_type{1, 1, nprocs}, topology_type{1, nprocs, 1},
                      topology_type{nprocs, 1, 1});

  // An FFT axis is distributed on the input or output topology. The shared
  // plan must fail with more than one process.
  const std::vector<PlanCase<2>> failure_cases = {
      // topology0 -> topology1
      // (n0, n1, n2/p) -> (n0, (n1/2+1)/p, n2)
      // This is slab
      {0, 1, {0, 1}, Expect::Throw},
      // topology0 -> topology2
      // (n0, n1, n2/p) -> (n0/p, n1/2+1, n2)
      // This is slab
      {0, 2, {0, 1}, Expect::Throw},
      // topology1 -> topology2
      // (n0, n1/p, n2) -> (n0/p, n1, n2/2+1)
      // This is slab
      {1, 2, {1, 2}, Expect::Throw},
      // topology0 -> topology0 with ax = {1, 2}
      // (n0, n1, n2/p) -> (n0, n1, (n2/2+1)/p)
      // This is slab
      {0, 0, {1, 2}, Expect::Throw},
      // topology1 -> topology1 with ax = {2, 1}
      // (n0, n1/p, n2) -> (n0, (n1/2+1)/p, n2)
      // This is slab
      {1, 1, {2, 1}, Expect::Throw},
      // topology2 -> topology2 with ax = {0, 1}
      // (n0/p, n1, n2) -> (n0/p, (n1/2+1), n2)
      // This is slab
      {2, 2, {0, 1}, Expect::Throw},
  };

  // The FFT axes are local on both topologies
  const Tolerance tol_bwd{1.0e-5, 1.0e-6};
  const std::vector<PlanCase<2>> success_cases = {
      // topology0 (n0, n1, n2/p) -> (n0/2+1, n1, n2/p) axis {1, 0}
      {0, 0, {1, 0}, Expect::Success, {}, tol_bwd},
      // topology0 (n0, n1, n2/p) -> (n0, n1/2+1, n2/p) axis {0, 1}
      {0, 0, {0, 1}, Expect::Success, {}, tol_bwd},
      // topology1 (n0, n1/p, n2) -> topology1 (n0/2+1, n1/p, n2) axis {2, 0}
      {1, 1, {2, 0}, Expect::Success, {}, tol_bwd},
      // topology1 (n0, n1/p, n2) -> topology1 (n0, n1/p, n2/2+1) axis {0, 2}
      {1, 1, {0, 2}, Expect::Success, {}, tol_bwd},
      // topology2 (n0/p, n1, n2) -> topology2 (n0/p, n1/2+1, n2) axis {2, 1}
      {2, 2, {2, 1}, Expect::Success, {}, tol_bwd},
      // topology2 (n0/p, n1, n2) -> topology2 (n0/p, n1, n2/2+1) axis {1, 2}
      {2, 2, {1, 2}, Expect::Success, {}, tol_bwd},
  };

  const std::array<std::size_t, 3> global_extents{8, 7, 5};
  auto gu = make_global_input<View3DType>(global_extents);

  const std::string name = "Test_SharedPlan.cpp test_shared2D_view3D";
  if (nprocs != 1) {
    run_plan_cases<ComplexView3DType>(name, execution_space(), gu, topologies,
                                      failure_cases, SharedPlanOps{});
  }
  run_plan_cases<ComplexView3DType>(name, execution_space(), gu, topologies,
                                    success_cases, SharedPlanOps{});
}

template <typename T, typename LayoutType>
void test_shared3D_view4D(std::size_t nprocs) {
  using View4DType        = Kokkos::View<T****, LayoutType, execution_space>;
  using float_type        = KokkosFFT::Impl::base_floating_point_type<T>;
  using ComplexView4DType = Kokkos::View<Kokkos::complex<float_type>****,
                                         LayoutType, execution_space>;
  using topology_type     = std::array<std::size_t, 4>;

  const auto topologies = std::make_tuple(
      topology_type{1, 1, 1, nprocs}, topology_type{1, 1, nprocs, 1},
      topology_type{1, nprocs, 1, 1}, topology_type{nprocs, 1, 1, 1});

  // An FFT axis is distributed on the input or output topology. The shared
  // plan must fail with more than one process.
  const std::vector<PlanCase<3>> failure_cases = {
      // topology0 -> topology1
      // (n0, n1, n2, n3/p) -> (n0, n1, (n2/2+1)/p, n3)
      {0, 1, {0, 1, 2}, Expect::Throw},
      // topology0 -> topology2
      // (n0, n1, n2, n3/p) -> (n0, (n1/2+1)/p, n2, n3)
      {0, 2, {0, 2, 1}, Expect::Throw},
      // topology0 -> topology3
      // (n0, n1, n2, n3/p) -> ((n0/2+1)/p, n1, n2, n3)
      {0, 3, {2, 1, 0}, Expect::Throw},
      // topology1 -> topology2
      // (n0, n1, n2/p, n3) -> (n0, n1/p, n2, n3/2+1)
      {1, 2, {1, 2, 3}, Expect::Throw},
      // topology1 -> topology3
      // (n0, n1, n2/p, n3) -> (n0/p, n1, n2, n3/2+1)
      {1, 3, {1, 2, 3}, Expect::Throw},
      // topology2 -> topology3
      // (n0, n1/p, n2, n3) -> (n0/p, n1, n2, n3/2+1)
      {2, 3, {1, 2, 3}, Expect::Throw},
      // topology0 -> topology0 with ax = {1, 2, 3}
      // (n0, n1, n2, n3/p) -> (n0, n1, n2, (n3/2+1)/p)
      // This is slab
      {0, 0, {1, 2, 3}, Expect::Throw},
      // topology1 -> topology1 with ax = {1, 3, 2}
      // (n0, n1, n2/p, n3) -> (n0, n1, (n2/2+1)/p, n3)
      // This is slab
      {1, 1, {1, 3, 2}, Expect::Throw},
      // topology2 -> topology2 with ax = {0, 2, 1}
      // (n0, n1/p, n2, n3) -> (n0, (n1/2+1)/p, n2, n3)
      // This is slab
      {2, 2, {0, 2, 1}, Expect::Throw},
      // topology3 -> topology3 with ax = {2, 1, 0}
      // (n0/p, n1, n2, n3) -> ((n0/2+1)/p, n1, n2, n3)
      // This is slab
      {3, 3, {2, 1, 0}, Expect::Throw},
  };

  // The FFT axes are local on both topologies
  const Tolerance tol_bwd{1.0e-5, 1.0e-6};
  const std::vector<PlanCase<3>> success_cases = {
      // topology0 (n0, n1, n2, n3/p) -> (n0/2+1, n1, n2, n3/p) axis {2, 1, 0}
      {0, 0, {2, 1, 0}, Expect::Success, {}, tol_bwd},
      // topology0 (n0, n1, n2, n3/p) -> (n0, n1/2+1, n2, n3/p) axis {2, 0, 1}
      {0, 0, {2, 0, 1}, Expect::Success, {}, tol_bwd},
      // topology0 (n0, n1, n2, n3/p) -> (n0, n1, n2/2+1, n3/p) axis {1, 0, 2}
      {0, 0, {1, 0, 2}, Expect::Success, {}, tol_bwd},
      // topology1 (n0, n1, n2/p, n3) -> (n0/2+1, n1, n2/p, n3) axis {3, 1, 0}
      {1, 1, {3, 1, 0}, Expect::Success, {}, tol_bwd},
      // topology1 (n0, n1, n2/p, n3) -> (n0, n1/2+1, n2/p, n3) axis {3, 0, 1}
      {1, 1, {3, 0, 1}, Expect::Success, {}, tol_bwd},
      // topology1 (n0, n1, n2/p, n3) -> (n0, n1, n2/p, n3/2+1) axis {1, 0, 3}
      {1, 1, {1, 0, 3}, Expect::Success, {}, tol_bwd},
      // topology2 (n0, n1/p, n2, n3) -> (n0/2+1, n1/p, n2, n3) axis {2, 3, 0}
      {2, 2, {2, 3, 0}, Expect::Success, {}, tol_bwd},
      // topology2 (n0, n1/p, n2, n3) -> (n0, n1/p, n2/2+1, n3) axis {3, 0, 2}
      {2, 2, {3, 0, 2}, Expect::Success, {}, tol_bwd},
      // topology2 (n0, n1/p, n2, n3) -> (n0, n1/p, n2, n3/2+1) axis {2, 0, 3}
      {2, 2, {2, 0, 3}, Expect::Success, {}, tol_bwd},
      // topology3 (n0/p, n1, n2, n3) -> (n0/p, n1/2+1, n2, n3) axis {3, 2, 1}
      {3, 3, {3, 2, 1}, Expect::Success, {}, tol_bwd},
      // topology3 (n0/p, n1, n2, n3) -> (n0/p, n1, n2/2+1, n3) axis {3, 1, 2}
      {3, 3, {3, 1, 2}, Expect::Success, {}, tol_bwd},
      // topology3 (n0/p, n1, n2, n3) -> (n0/p, n1, n2, n3/2+1) axis {1, 2, 3}
      {3, 3, {1, 2, 3}, Expect::Success, {}, tol_bwd},
  };

  const std::array<std::size_t, 4> global_extents{8, 7, 5, 6};
  auto gu = make_global_input<View4DType>(global_extents);

  const std::string name = "Test_SharedPlan.cpp test_shared3D_view4D";
  if (nprocs != 1) {
    run_plan_cases<ComplexView4DType>(name, execution_space(), gu, topologies,
                                      failure_cases, SharedPlanOps{});
  }
  run_plan_cases<ComplexView4DType>(name, execution_space(), gu, topologies,
                                    success_cases, SharedPlanOps{});
}

}  // namespace

TYPED_TEST_SUITE(TestShared1D, test_types);
TYPED_TEST_SUITE(TestShared2D, test_types);
TYPED_TEST_SUITE(TestShared3D, test_types);

TYPED_TEST(TestShared1D, View2D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_shared1D_view2D<float_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestShared1D, View2D_C2C) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;

  test_shared1D_view2D<complex_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestShared2D, View3D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_shared2D_view3D<float_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestShared2D, View3D_C2C) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;

  test_shared2D_view3D<complex_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestShared3D, View4D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_shared3D_view4D<float_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestShared3D, View4D_C2C) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;

  test_shared3D_view4D<complex_type, layout_type>(this->m_nprocs);
}
