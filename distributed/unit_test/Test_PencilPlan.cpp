#include <mpi.h>
#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <string>
#include <tuple>
#include <vector>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_PencilPlan.hpp"
#include "Test_PlanUtils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using test_types      = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>,
                                    std::pair<double, Kokkos::LayoutLeft>,
                                    std::pair<double, Kokkos::LayoutRight>>;

//  Basically the same fixtures, used for labeling tests
template <typename T>
struct TestPencil1D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;

  int m_rank   = 0;
  int m_nprocs = 1;
  int m_npx    = 1;

  virtual void SetUp() {
    ::MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &m_nprocs);

    m_npx = std::sqrt(m_nprocs);
    if (this->m_nprocs == 1 || this->m_npx * this->m_npx != this->m_nprocs) {
      GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                      "for this test";
    }
  }
};

template <typename T>
struct TestPencil2D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;

  int m_rank   = 0;
  int m_nprocs = 1;
  int m_npx    = 1;

  virtual void SetUp() {
    ::MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &m_nprocs);

    m_npx = std::sqrt(m_nprocs);
    if (this->m_nprocs == 1 || this->m_npx * this->m_npx != this->m_nprocs) {
      GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                      "for this test";
    }
  }
};

template <typename T>
struct TestPencil3D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;

  int m_rank   = 0;
  int m_nprocs = 1;
  int m_npx    = 1;

  virtual void SetUp() {
    ::MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &m_nprocs);

    m_npx = std::sqrt(m_nprocs);
    if (this->m_nprocs == 1 || this->m_npx * this->m_npx != this->m_nprocs) {
      GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                      "for this test";
    }
  }
};

/// \brief Plan operations (see run_plan_cases) for PencilPlan
struct PencilPlanOps : public InternalPlanOps {
  template <typename ExecutionSpace, typename InViewType, typename OutViewType,
            typename AxesType, typename InTopologyType,
            typename OutTopologyType>
  auto create(const ExecutionSpace& exec, const InViewType& in,
              const OutViewType& out, const AxesType& axes,
              const InTopologyType& in_topology,
              const OutTopologyType& out_topology, MPI_Comm comm) const {
    return KokkosFFT::Distributed::Impl::PencilPlan(
        exec, in, out, axes, in_topology, out_topology, comm);
  }
};

template <typename T, typename LayoutType>
void test_pencil1D_view3D(std::size_t npx, std::size_t npy) {
  using View3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using float_type = KokkosFFT::Impl::base_floating_point_type<T>;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<float_type>***, LayoutType, execution_space>;
  using topology_r_type =
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutRight>;
  using topology_l_type =
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutLeft>;

  // x-pencil, y-pencil and z-pencil
  const auto topologies = std::make_tuple(topology_r_type{1, npx, npy},
                                          topology_r_type{npx, 1, npy},
                                          topology_l_type{npy, npx, 1});
  const Tolerance tol_1e6{1.0e-5, 1.0e-6};

  // Not a pencil geometry: the plan must fail
  const std::vector<PlanCase<1>> failure_cases = {
      // topo0 -> topo1 with ax = {0}:
      // (n0, n1/px, n2/py) -> ((n0/2+1)/px, n1, n2/py)
      {0, 1, {0}, Expect::Throw},
      // topo0 -> topo2 with ax = {1}:
      // (n0, n1/px, n2/py) -> (n0/py, (n1/2+1)/px, n2)
      {0, 2, {1}, Expect::Throw},
      // topo1 -> topo2 with ax = {2}:
      // (n0/px, n1, n2/py) -> (n0/px, n1/py, n2/2+1)
      {1, 2, {2}, Expect::Throw},
      // topo0 -> topo0 with ax = {2}:
      // (n0, n1/px, n2/py) -> (n0, n1/px, (n2/2+1)/py)
      {0, 0, {2}, Expect::Throw},
      // topo1 -> topo1 with ax = {1}
      // (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      {1, 1, {1}, Expect::Throw},
      // topology2 -> topology2 with ax = {1}
      // (n0/px, n1/py, n2) -> (n0/px, (n1/2+1)/py, n2)
      {2, 2, {1}, Expect::Throw},
  };

  const std::vector<PlanCase<1>> success_cases = {
      // topo 0 -> topo 0 with ax = {0}:
      // (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // FFT ax = {0}
      {0, 0, {0}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 0 with ax = {1}:
      // (n0, n1/px, n2/py) -> (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      // -> (n0, (n1/2+1)/px, n2/py)
      // Transpose topo 1 -> FFT ax = {1} -> Transpose topo 0
      {0, 0, {1}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 0 with ax = {2}:
      // (n0, n1/px, n2/py) -> (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
      // -> (n0, n1/px, (n2/2+1)/py)
      // Transpose topo 2 -> FFT ax = {2} -> Transpose topo 0
      {0, 0, {2}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 1 with ax = {0}:
      // (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/px, n1, n2/py)
      // FFT ax = {0} -> Transpose 1
      {0, 1, {0}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 1 with ax = {1}:
      // (n0, n1/px, n2/py) -> (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      // Transpose topo 1 -> FFT ax = {1}
      {0, 1, {1}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 1 with ax = {2}:
      // (n0, n1/px, n2/py) -> (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
      // -> (n0, n1/px, (n2/2+1)/py) -> (n0/px, n1/px, (n2/2+1)/py)
      // Transpose topo 2 -> FFT ax = {2} -> Transpose topo 0
      // -> Transpose topo 1
      {0, 1, {2}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 2 with ax = {0}:
      // (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/py, n1/px, n2)
      // FFT ax = {0} -> Transpose topo 2
      {0, 2, {0}, Expect::Success, {}, tol_1e6},
      //  topo 0 -> topo 2 with ax = {1}:
      //  (n0, n1/px, n2/py) -> (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      //  -> (n0, (n1/2+1)/px, n2/py) -> (n0/py, (n1/2+1)/px, n2)
      //  Transpose topo 1 -> FFT ax = {1} -> Transpose topo 0 -> topo 2
      {0, 2, {1}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 2 with ax = {2}:
      // (n0, n1/px, n2/py) -> (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
      // Transpose topo 2 -> FFT ax = {2}
      {0, 2, {2}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 0 with ax = {0}:
      // (n0/px, n1, n2/py) -> (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // Transpose topo 0 -> FFT ax = {0}
      {1, 0, {0}, Expect::Success, {}, tol_1e6},
  };

  const std::array<std::size_t, 3> global_extents{8, 7, 5};
  auto gu = make_global_input<View3DType>(global_extents);

  const std::string name = "Test_PencilPlan.cpp test_pencil1D_view3D";
  const bool is_failure  = npx == 1 || npy == 1;
  run_plan_cases<ComplexView3DType>(name, execution_space(), gu, topologies,
                                    is_failure ? failure_cases : success_cases,
                                    PencilPlanOps{});
}

template <typename T, typename LayoutType>
void test_pencil2D_view3D(std::size_t npx, std::size_t npy) {
  using View3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using float_type = KokkosFFT::Impl::base_floating_point_type<T>;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<float_type>***, LayoutType, execution_space>;
  using topology_r_type =
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutRight>;
  using topology_l_type =
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutLeft>;

  // x-pencil, y-pencil and z-pencil
  const auto topologies = std::make_tuple(topology_r_type{1, npx, npy},
                                          topology_r_type{npx, 1, npy},
                                          topology_l_type{npy, npx, 1});
  const Tolerance tol_1e6{1.0e-5, 1.0e-6};

  // Not a pencil geometry: the plan must fail
  const std::vector<PlanCase<2>> failure_cases = {
      // topo0 -> topo1 with ax = {0}:
      // (n0, n1/px, n2/py) -> ((n0/2+1)/px, n1, n2/py)
      {0, 1, {0, 1}, Expect::Throw},
      // topo0 -> topo2 with ax = {1}:
      // (n0, n1/px, n2/py) -> (n0/py, (n1/2+1)/px, n2)
      {0, 2, {0, 1}, Expect::Throw},
      // topo1 -> topo2 with ax = {2}:
      // (n0/px, n1, n2/py) -> (n0/px, n1/py, n2/2+1)
      {1, 2, {0, 2}, Expect::Throw},
      // topo0 -> topo0 with ax = {2}:
      // (n0, n1/px, n2/py) -> (n0, n1/px, (n2/2+1)/py)
      {0, 0, {0, 2}, Expect::Throw},
      // topo1 -> topo1 with ax = {1}
      // (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      {1, 1, {0, 1}, Expect::Throw},
      // topology2 -> topology2 with ax = {1}
      // (n0/px, n1/py, n2) -> (n0/px, (n1/2+1)/py, n2)
      {2, 2, {1, 0}, Expect::Throw},
  };

  const std::vector<PlanCase<2>> success_cases = {
      // topo 0 -> topo 0 with ax = {0, 1}:
      // (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // Transpose topo 1 -> FFT ax = {1} -> Transpose topo 0 -> FFT ax = {0}
      {0, 0, {0, 1}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 0 with ax = {0, 2}:
      // (n0, n1/px, n2/py) -> (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
      // -> (n0, n1/px, (n2/2+1)/py)
      // Transpose topo 2 -> FFT ax = {2} -> Transpose topo 0
      // -> FFT ax = {0}
      {0, 0, {0, 2}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 0 with ax = {1, 0}:
      // (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/px, n1, n2/py)
      // -> (n0/2+1, n1/px, n2/py)
      // FFT ax = {0} -> Transpose topo 1 -> FFT ax = {1} -> Transpose topo 0
      {0, 0, {1, 0}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 0 with ax = {1, 2}:
      // (n0, n1/px, n2/py) -> (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
      // -> (n0/py, n1, (n2/2+1)/px) -> (n0/py, n1/px, (n2/2+1))
      // -> (n0, n1/px, (n2/2+1)/py)
      // Transpose topo 2 -> FFT ax = {2} -> Transpose topo 4 -> FFT ax = {1}
      // -> Transpose topo 2 -> Transpose topo 0
      {0, 0, {1, 2}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 0 with ax = {2, 0}:
      // (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/py, n1/px, n2)
      // -> (n0/2+1, n1/px, n2/py)
      // FFT ax = {0} -> Transpose topo 2 -> FFT ax = {2} -> Transpose topo 0
      {0, 0, {2, 0}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 0 with ax = {2, 1}:
      // (n0, n1/px, n2/py) -> (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      // -> (n0/px, (n1/2+1)/py, n2) -> (n0/px, n1/2+1, n2/py)
      // -> (n0, (n1/2+1)/px, n2/py)
      // Transpose topo 1 -> FFT ax = {1} -> Transpose topo 3 -> FFT ax = {2}
      // -> Transpose topo 1 -> Transpose topo 0
      {0, 0, {2, 1}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 1 with ax = {0, 1}:
      // (n0, n1/px, n2/py) -> (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      // -> (n0, (n1/2+1)/px, n2/py) -> (n0/px, n1/2+1, n2/py)
      // Transpose 1 -> FFT ax = {1} -> Transpose topo 0 -> FFT ax = {0}
      // -> Transpose 1
      {0, 1, {0, 1}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 1 with ax = {0, 2}:
      // (n0, n1/px, n2/py) -> (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
      // -> (n0, n1/px, (n2/2+1)/py) -> (n0/px, n1, (n2/2+1)/py)
      // Transpose topo 2 -> FFT ax = {2} -> Transpose topo 0
      // -> FFT ax = {0} -> Transpose topo 1
      {0, 1, {0, 2}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 1 with ax = {1, 0}:
      // (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/px, n1, n2/py)
      // FFT ax = {0} -> Transpose topo 1 -> FFT ax = {1}
      {0, 1, {1, 0}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 1 with ax = {1, 2}:
      // (n0, n1/px, n2/py) -> (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
      // -> (n0, n1/px, (n2/2+1)/py) -> (n0/px, n1, (n2/2+1)/py)
      // Transpose topo 2 -> FFT ax = {2} -> Transpose topo 0
      // -> Transpose topo 1
      // -> FFT ax = {1}
      {0, 1, {1, 2}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 1 with ax = {2, 0}:
      // (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/py, n1/px, n2)
      // -> (n0/2+1, n1/px, n2/py) -> ((n0/2+1)/px, n1, n2/py)
      // FFT ax = {0} -> Transpose topo 2 -> FFT ax = {2} -> Transpose topo 0
      // -> Transpose topo 1
      {0, 1, {2, 0}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 1 with ax = {2, 1}:
      // (n0, n1/px, n2/py) -> (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      // -> (n0/px, (n1/2+1)/py, n2) -> (n0/px, n1/2+1, n2/py)
      // Transpose topo 1 -> FFT ax = {1} -> Transpose topo 3 -> FFT ax = {2}
      // -> Transpose topo 1
      {0, 1, {2, 1}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 2 with ax = {0, 1}:
      // (n0, n1/px, n2/py) -> (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      // -> (n0, (n1/2+1)/px, n2/py) -> (n0/py, (n1/2+1)/px, n2)
      // Transpose topo 1 -> FFT ax = {1} -> Transpose topo 0 -> FFT ax = {0}
      // -> Transpose topo 2
      {0, 2, {0, 1}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 2 with ax = {0, 2}:
      // (n0, n1/px, n2/py) -> (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
      // -> (n0, n1/px, (n2/2+1)/py) -> ((n0/2+1)/py, n1/px, n2)
      // FFT ax = {0} -> Transpose topo 2 -> FFT ax = {2} -> Transpose topo 0
      // -> Transpose topo 2
      {0, 2, {0, 2}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 2 with ax = {1, 0}:
      // (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/px, n1, n2/py)
      // -> (n0/2+1, n1/px, n2/py) -> ((n0/2+1)/py, n1/px, n2)
      // FFT ax = {0} -> Transpose topo 1 -> FFT ax = {1}
      // -> Transpose topo 0 -> Transpose topo 2
      {0, 2, {1, 0}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 2 with ax = {1, 2}:
      // (n0, n1/px, n2/py) -> (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
      // -> (n0/py, n1, (n2/2+1)/px) -> (n0/py, n1/px, n2/2+1)
      // Transpose 2 -> FFT ax = {2} -> Transpose topo 4 -> FFT ax = {1}
      // -> Transpose topo 2
      {0, 2, {1, 2}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 2 with ax = {2, 0}:
      // (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/py, n1/px, n2)
      // FFT ax = {0} -> Transpose topo 2 -> FFT ax = {2}
      {0, 2, {2, 0}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 2 with ax = {2, 1}:
      // (n0, n1/px, n2/py) -> (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      // -> (n0, (n1/2+1)/px, n2/py) -> (n0/py, (n1/2+1)/px, n2)
      // Transpose topo 1 -> FFT ax = {1} -> Transpose topo 0 -> Transpose 2
      // -> FFT ax = {2}
      {0, 2, {2, 1}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 0 with ax = {0, 1}:
      // (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      // -> (n0, (n1/2+1)/px, n2/py)
      // FFT ax = {1} -> Transpose topo 0 -> FFT ax = {0}
      {1, 0, {0, 1}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 0 with ax = {0, 2}:
      // (n0/px, n1, n2/py) -> (n0/px, n1/py, n2) -> (n0/px, n1/py, n2/2+1)
      // -> (n0/px, n1, (n2/2+1)/py) -> (n0, n1/px, (n2/2+1)/py)
      // Transpose 3 -> FFT ax = {2} -> Transpose topo 1 -> Transpose 0
      // -> FFT ax = {0}
      {1, 0, {0, 2}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 0 with ax = {1, 0}:
      // (n0/px, n1, n2/py) -> (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/px, n1, n2/py) -> (n0/2+1, n1/px, n2/py)
      // Transpose 0 -> FFT ax = {0} -> Transpose topo 1 -> FFT ax = {1}
      // -> Transpose 0
      {1, 0, {1, 0}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 0 with ax = {1, 2}:
      // (n0/px, n1, n2/py) -> (n0/px, n1/py, n2) -> (n0/px, n1/py, n2/2+1)
      // -> (n0/px, n1, (n2/2+1)/py) -> (n0, n1/px, (n2/2+1)/py)
      // Transpose 3 -> FFT ax = {2} -> Transpose topo 1 -> FFT ax = {1}
      // -> Transpose 0
      {1, 0, {1, 2}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 0 with ax = {2, 0}:
      // (n0/px, n1, n2/py) -> (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/py, n1/px, n2) -> (n0/2+1, n1/px, n2/py)
      // Transpose 0 -> FFT ax = {0} -> Transpose topo 2 -> FFT ax = {2}
      // -> Transpose 0
      {1, 0, {2, 0}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 0 with ax = {2, 1}:
      // (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      // -> (n0/px, (n1/2+1)/py, n2)
      // -> (n0/px, n1/2+1, n2/py) -> (n0, (n1/2+1)/px, n2/py)
      // FFT ax = {1} -> Transpose topo 3 -> FFT ax = {2}
      // -> Transpose 1 -> Transpose 0
      {1, 0, {2, 1}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 1 with ax = {0, 1}:
      // (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      // -> (n0, (n1/2+1)/px, n2/py)
      // -> (n0/px, n1/2+1, n2/py)
      // FFT ax = {1} -> Transpose topo 0 -> FFT ax = {0}
      // -> Transpose 1
      {1, 1, {0, 1}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 1 with ax = {0, 2}:
      // (n0/px, n1, n2/py) -> (n0/px, n1/py, n2) -> (n0/px, n1/py, n2/2+1)
      // -> (n0, n1/py, (n2/2+1)/px) -> (n0/px, n1/py, n2/2+1)
      // -> (n0/px, n1, (n2/2+1)/py)
      // Transpose 3 -> FFT ax = {2} -> Transpose topo 5 -> FFT ax = {0}
      // -> Transpose 3 -> Transpose 1
      {1, 1, {0, 2}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 1 with ax = {1, 0}:
      // (n0/px, n1, n2/py) -> (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/px, n1, n2/py)
      // Transpose 0 -> FFT ax = {0} -> Transpose topo 1 -> FFT ax = {1}
      {1, 1, {1, 0}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 1 with ax = {1, 2}:
      // (n0/px, n1, n2/py) -> (n0/px, n1/py, n2) -> (n0/px, n1/py, n2/2+1)
      // -> (n0/px, n1, (n2/2+1)/py)
      // Transpose 3 -> FFT ax = {2} -> Transpose topo 1 -> FFT ax = {1}
      {1, 1, {1, 2}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 1 with ax = {2, 0}:
      // (n0/px, n1, n2/py) -> (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/py, n1/px, n2) -> ((n0/2+1), n1/px, n2/py)
      // -> ((n0/2+1)/px, n1, n2/py)
      // Transpose 0 -> FFT ax = {0} -> Transpose topo 2 -> FFT ax = {2}
      // Transpose 0 -> Transpose 1
      {1, 1, {2, 0}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 1 with ax = {2, 1}:
      // (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      // -> (n0/px, (n1/2+1)/py, n2)
      // -> (n0/px, n1/2+1, n2/py)
      // FFT ax = {1} -> Transpose topo 3 -> FFT ax = {2}
      // -> Transpose 1
      {1, 1, {2, 1}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 2 with ax = {0, 1}:
      // (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      // -> (n0, (n1/2+1)/px, n2/py)
      // -> (n0/py, (n1/2+1)/px, n2)
      // FFT ax = {1} -> Transpose topo 0 -> FFT ax = {0}
      // -> Transpose 2
      {1, 2, {0, 1}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 2 with ax = {0, 2}:
      // (n0/px, n1, n2/py) -> (n0/px, n1/py, n2) -> (n0/px, n1/py, n2/2+1)
      // -> (n0, n1/py, (n2/2+1)/px) -> (n0/px, n1/py, n2/2+1)
      // Transpose 3 -> FFT ax = {2} -> Transpose topo 5 -> FFT ax = {0}
      // -> Transpose 4 -> Transpose 2
      {1, 2, {0, 2}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 2 with ax = {1, 0}:
      // (n0/px, n1, n2/py) -> (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/px, n1, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/py, n1/px, n2)
      // Transpose topo 0 -> FFT ax = {0} -> Transpose 1 -> FFT ax = {1}
      // -> Transpose 0 -> Transpose 2
      {1, 2, {1, 0}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 2 with ax = {1, 2}:
      // (n0/px, n1, n2/py) -> (n0/px, n1/py, n2) -> (n0/px, n1/py, n2/2+1)
      // -> (n0/px, n1, (n2/2+1)/py) -> (n0, n1/px, (n2/2+1)/py)
      // -> (n0/py, n1/px, n2/2+1)
      // Transpose topo 3 -> FFT ax = {2} -> Transpose 1 -> FFT ax = {1}
      // -> Transpose 0 -> Transpose 2
      {1, 2, {1, 2}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 2 with ax = {2, 0}:
      // (n0/px, n1, n2/py) -> (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/py, n1/px, n2)
      // Transpose topo 0 -> FFT ax = {0} -> Transpose 2 -> FFT ax = {2}
      {1, 2, {2, 0}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 2 with ax = {2, 1}:
      // (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      // -> (n0, (n1/2+1)/px, n2/py)
      // -> (n0/py, (n1/2+1)/px, n2)
      // FFT ax = {1} -> Transpose 0 -> Transpose 2 -> FFT ax = {2}
      {1, 2, {2, 1}, Expect::Success, {}, tol_1e6},
      // topo 2 -> topo 0 with ax = {0, 1}:
      // (n0/py, n1/px, n2) -> (n0/py, n1, n2/px) -> (n0/py, n1/2+1, n2/px)
      // -> (n0/py, (n1/2+1)/px, n2) -> (n0, (n1/2+1)/px, n2/py)
      // Transpose 4 -> FFT ax = {1} -> Transpose topo 2 -> Transpose 0
      // -> FFT ax = {0}
      {2, 0, {0, 1}, Expect::Success, {}, tol_1e6},
      // topo 2 -> topo 0 with ax = {0, 2}:
      // (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
      // -> (n0, n1/px, (n2/2+1)/py)
      // FFT ax = {2} -> Transpose 0 -> FFT ax = {0}
      {2, 0, {0, 2}, Expect::Success, {}, tol_1e6},
      // topo 2 -> topo 0 with ax = {1, 0}:
      // (n0/py, n1/px, n2) -> (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/px, n1, n2/py) -> (n0/2+1, n1/px, n2/py)
      // Transpose 0 -> FFT ax = {0} -> Transpose 1 -> FFT ax = {1}
      // -> Transpose 0
      {2, 0, {1, 0}, Expect::Success, {}, tol_1e6},
      // topo 2 -> topo 0 with ax = {1, 2}:
      // (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
      // -> (n0/py, n1, (n2/2+1)/px)
      // -> (n0/py, n1/px, n2/2+1) -> (n0, n1/px, (n2/2+1)/py)
      // FFT ax = {2} -> Transpose 4 -> FFT ax = {1} -> Transpose 2
      // -> Transpose 0
      {2, 0, {1, 2}, Expect::Success, {}, tol_1e6},
      // topo 2 -> topo 0 with ax = {2, 0}:
      // (n0/py, n1/px, n2) -> (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/py, n1/px, n2)  -> (n0/2+1, n1/px, n2/py)
      // Transpose 0 -> FFT ax = {0} -> Transpose 2 -> FFT ax = {2}
      // -> Transpose 0
      {2, 0, {2, 0}, Expect::Success, {}, tol_1e6},
      // topo 2 -> topo 0 with ax = {2, 1}:
      // (n0/py, n1/px, n2) -> (n0/py, n1, n2/px) -> (n0/py, n1/2+1, n2/px)
      // -> (n0/py, (n1/2+1)/px, n2) -> (n0, (n1/2+1)/px, n2/py)
      // Transpose 4 -> FFT ax = {1} -> Transpose 2 -> FFT ax = {2}
      // -> Transpose 0
      {2, 0, {2, 1}, Expect::Success, {}, tol_1e6},
      // topo 2 -> topo 1 with ax = {0, 1}:
      // (n0/py, n1/px, n2) -> (n0/py, n1, n2/px) -> (n0/py, n1/2+1, n2/px)
      // -> (n0, (n1/2+1)/py, n2/px) -> (n0/px, (n1/2+1)/py, n2)
      // -> (n0/px, (n1/2+1), n2/py)
      // Transpose 4 -> FFT ax = {1} -> Transpose 5 -> FFT ax = {2}
      // -> Transpose 3 -> Transpose 1
      {2, 1, {0, 1}, Expect::Success, {}, tol_1e6},
      // topo 2 -> topo 1 with ax = {0, 2}:
      // (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
      // -> (n0, n1/px, (n2/2+1)/py)
      // -> (n0/px, n1, (n2/2+1)/py)
      // FFT ax = {2} -> Transpose 0 -> FFT ax = {0} -> Transpose 1
      {2, 1, {0, 2}, Expect::Success, {}, tol_1e6},
      // topo 2 -> topo 1 with ax = {1, 0}:
      // (n0/py, n1/px, n2) -> (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/px, n1, n2/py)
      // Transpose 0 -> FFT ax = {0} -> Transpose 1 -> FFT ax = {1}
      {2, 1, {1, 0}, Expect::Success, {}, tol_1e6},
      // topo 2 -> topo 1 with ax = {1, 2}:
      // (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
      // -> (n0, n1/px, (n2/2+1)/py)
      // -> (n0/px, n1, (n2/2+1)/py)
      // FFT ax = {2} -> Transpose 0 -> Transpose 1 -> FFT ax = {1}
      {2, 1, {1, 2}, Expect::Success, {}, tol_1e6},
      // topo 2 -> topo 1 with ax = {2, 0}
      // (n0/py, n1/px, n2) -> (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/py, n1/px, n2) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/px, n1, n2/py)
      // Transpose 0 -> FFT ax = {0} -> Transpose 2 -> FFT ax = {2}
      // -> Transpose 0 -> Transpose 1
      {2, 1, {2, 0}, Expect::Success, {}, tol_1e6},
      // topo 2 -> topo 1 with ax = {2, 1}
      // (n0/py, n1/px, n2) -> (n0/py, n1, n2/px) -> (n0/py, n1/2+1, n2/px)
      // -> (n0/py, (n1/2+1)/px, n2) -> (n0, (n1/2+1)/px, n2/py)
      // -> (n0/px, n1/2+1, n2/py)
      // Transpose 4 -> FFT ax = {1} -> Transpose 2 -> FFT ax = {2}
      // Transpose 0 -> Transpose 1
      {2, 1, {2, 1}, Expect::Success, {}, tol_1e6},
      // topo 2 -> topo 2 with ax = {0, 1}:
      // (n0/py, n1/px, n2) -> (n0/py, n1, n2/px) -> (n0/py, n1/2+1, n2/px)
      // -> (n0, (n1/2+1)/py, n2/px) -> (n0/py, n1/2+1, n2/px)
      // -> (n0/py, (n1/2+1)/px, n2)
      // Transpose 4 -> FFT ax = {1} -> Transpose 5 -> FFT ax = {0}
      // Transpose 4 -> Transpose 2
      {2, 2, {0, 1}, Expect::Success, {}, tol_1e6},
      // topo 2 -> topo 2 with ax = {0, 2}:
      // (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
      // -> (n0, n1/px, (n2/2+1)/py)
      // -> (n0/py, n1/px, n2/2+1)
      // FFT ax = {2} -> Transpose 0 -> FFT ax = {0} -> Transpose 2
      {2, 2, {0, 2}, Expect::Success, {}, tol_1e6},
      // topo 2 -> topo 2 with ax = {1, 0}:
      // (n0/py, n1/px, n2) -> (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/px, n1, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/py, n1/px, n2)
      // Transpose 0 -> FFT ax = {0} -> Transpose 1 -> FFT ax = {1}
      // Transpose 0 -> Transpose 2
      {2, 2, {1, 0}, Expect::Success, {}, tol_1e6},
      // topo 2 -> topo 2 with ax = {1, 2}:
      // (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
      // -> (n0/py, n1, (n2/2+1)/px)
      // -> (n0/py, n1/px, n2/2+1)
      // FFT ax = {2} -> Transpose 4 -> FFT ax = {1} -> Transpose 2
      {2, 2, {1, 2}, Expect::Success, {}, tol_1e6},
      // topo 2 -> topo 2 with ax = {2, 0}:
      // (n0/py, n1/px, n2) -> (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/py, n1/px, n2)
      // Transpose 0 -> FFT ax = {0} -> Transpose 2 -> FFT ax = {2}
      {2, 2, {2, 0}, Expect::Success, {}, tol_1e6},
      // topo 2 -> topo 2 with ax = {2, 1}:
      // (n0/py, n1/px, n2) -> (n0/py, n1, n2/px) -> (n0/py, n1/2+1, n2/px)
      // -> (n0/py, (n1/2+1)/px, n2)
      // Transpose 4 -> FFT ax = {1} -> Transpose 2 -> FFT ax = {2}
      {2, 2, {2, 1}, Expect::Success, {}, tol_1e6},
  };

  const std::array<std::size_t, 3> global_extents{8, 7, 5};
  auto gu = make_global_input<View3DType>(global_extents);

  const std::string name = "Test_PencilPlan.cpp test_pencil2D_view3D";
  const bool is_failure  = npx == 1 || npy == 1;
  run_plan_cases<ComplexView3DType>(name, execution_space(), gu, topologies,
                                    is_failure ? failure_cases : success_cases,
                                    PencilPlanOps{});
}

template <typename T, typename LayoutType>
void test_pencil2D_view4D(std::size_t npx, std::size_t npy) {
  using View4DType        = Kokkos::View<T****, LayoutType, execution_space>;
  using float_type        = KokkosFFT::Impl::base_floating_point_type<T>;
  using ComplexView4DType = Kokkos::View<Kokkos::complex<float_type>****,
                                         LayoutType, execution_space>;
  using topology_r_type =
      KokkosFFT::Distributed::Topology<std::size_t, 4, Kokkos::LayoutRight>;
  using topology_l_type =
      KokkosFFT::Distributed::Topology<std::size_t, 4, Kokkos::LayoutLeft>;

  // x-pencil, y-pencil and z-pencil
  const auto topologies = std::make_tuple(topology_r_type{1, 1, npx, npy},
                                          topology_r_type{1, npx, 1, npy},
                                          topology_l_type{1, npy, npx, 1});
  const Tolerance tol_1e6{1.0e-5, 1.0e-6};

  // Not a pencil geometry: the plan must fail
  const std::vector<PlanCase<2>> failure_cases = {
      // topo0 -> topo1 with ax = {0}:
      // (n0, n1/px, n2/py) -> ((n0/2+1)/px, n1, n2/py)
      {0, 1, {0, 1}, Expect::Throw},
      // topo0 -> topo2 with ax = {1}:
      // (n0, n1/px, n2/py) -> (n0/py, (n1/2+1)/px, n2)
      {0, 2, {0, 1}, Expect::Throw},
      // topo1 -> topo2 with ax = {2}:
      // (n0/px, n1, n2/py) -> (n0/px, n1/py, n2/2+1)
      {1, 2, {0, 2}, Expect::Throw},
      // topo0 -> topo0 with ax = {2}:
      // (n0, n1/px, n2/py) -> (n0, n1/px, (n2/2+1)/py)
      {0, 0, {0, 2}, Expect::Throw},
      // topo1 -> topo1 with ax = {1}
      // (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      {1, 1, {0, 1}, Expect::Throw},
      // topology2 -> topology2 with ax = {1}
      // (n0/px, n1/py, n2) -> (n0/px, (n1/2+1)/py, n2)
      {2, 2, {1, 0}, Expect::Throw},
  };

  const std::vector<PlanCase<2>> success_cases = {
      // topo 0 -> topo 0 with ax = {0, 1}:
      // (n0, n1, n2/px, n3/py) -> (n0, n1/2+1, n2/px, n3/py)
      // FFT2 ax = {0, 1}
      {0, 0, {0, 1}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 1 with ax = {0, 1}:
      // (n0, n1, n2/px, n3/py) -> (n0, n1/2+1, n2/px, n3/py)
      // -> (n0, (n1/2+1)/px, n2, n3/py)
      // FFT2 ax = {0, 1} -> Transpose topo 1
      {0, 1, {0, 1}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 0 with ax = {0, 1}:
      // (n0, n1/px, n2, n3/py) -> (n0, n1, n2/px, n3/py)
      // -> (n0, n1/2+1, n2/px, n3/py)
      // FFT2 ax = {0, 1} -> Transpose topo 1
      {1, 0, {0, 1}, Expect::Success, {}, tol_1e6},
  };

  const std::array<std::size_t, 4> global_extents{8, 7, 5, 6};
  auto gu = make_global_input<View4DType>(global_extents);

  const std::string name = "Test_PencilPlan.cpp test_pencil2D_view4D";
  const bool is_failure  = npx == 1 || npy == 1;
  run_plan_cases<ComplexView4DType>(name, execution_space(), gu, topologies,
                                    is_failure ? failure_cases : success_cases,
                                    PencilPlanOps{});
}

template <typename T, typename LayoutType>
void test_pencil3D_view3D(std::size_t npx, std::size_t npy) {
  using View3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using float_type = KokkosFFT::Impl::base_floating_point_type<T>;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<float_type>***, LayoutType, execution_space>;
  using topology_r_type =
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutRight>;
  using topology_l_type =
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutLeft>;

  // x-pencil, y-pencil and z-pencil
  const auto topologies = std::make_tuple(
      topology_r_type{1, npx, npy}, topology_r_type{npx, 1, npy},
      topology_l_type{npy, npx, 1}, topology_r_type{npx, npy, 1});
  const Tolerance tol_1e6{1.0e-5, 1.0e-6};

  // Not a pencil geometry: the plan must fail
  const std::vector<PlanCase<3>> failure_cases = {
      // topology0 -> topology1
      // (n0, n1/px, n2/py) -> (n0, (n1/2+1)/p, n2)
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
      // (n0, n1/px, n2/py) -> (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
      // -> (n0/py, n1, (n2/2+1)/px) -> (n0/py, n1/px, n2/2+1)
      // -> (n0, n1/px, (n2/2+1)/py)
      // Transpose topo 2 -> FFT ax = {2} -> Transpose topo 4 -> FFT ax = {1}
      // -> Transpose topo 2 -> Transpose topo 0 -> FFT ax = {0}
      {0, 0, {0, 1, 2}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 0 with ax = {0, 2, 1}:
      // (n0, n1/px, n2/py) -> (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      // -> (n0/px, (n1/2+1)/py, n2) -> (n0/px, n1/2+1, n2/py)
      // -> (n0, n1/px, n2/py)
      // Transpose topo 1 -> FFT ax = {1} -> Transpose topo 3 -> FFT ax = {2}
      // -> Transpose topo 1 -> Transpose topo 0 -> FFT ax = {0}
      {0, 0, {0, 2, 1}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 0 with ax = {1, 0, 2}:
      // (n0, n1/px, n2/py) -> (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
      // -> (n0x, n1/px, (n2/2+1)/py) -> (n0/py, n1/px, n2/2+1)
      // -> (n0x, n1/px, (n2/2+1)/py)
      // Transpose topo 2 -> FFT ax = {2} -> Transpose topo 0
      // -> FFT ax = {0} -> Transpose topo 1 -> FFT ax = {1} -> Transpose topo 0
      {0, 0, {1, 0, 2}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 0 with ax = {1, 2, 0}:
      // (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/py, n1/px, n2) -> ((n0/2+1)/py, n1, n2/px)
      // -> ((n0/2+1)/py, n1/px, n2) -> (n0/2+1, n1/px, n2/py)
      // FFT ax = {0} -> Transpose topo 2 -> FFT ax = {2} -> Transpose topo 4
      // -> FFT ax = {1} -> Transpose topo 2 -> Transpose topo 0
      {0, 0, {1, 2, 0}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 0 with ax = {2, 0, 1}:
      // (n0, n1/px, n2/py) -> (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      // -> (n0, (n1/2+1)/px, n2/py) -> (n0/py, (n1/2+1)/px, n2)
      // -> (n0, (n1/2+1)/px, n2/py)
      // Transpose topo 1 -> FFT ax = {1} -> Transpose topo 0 -> FFT ax = {0}
      // -> Transpose topo 2 -> FFT ax = {2} -> Transpose topo 0
      {0, 0, {2, 0, 1}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 0 with ax = {2, 1, 0}:
      // (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/px, n1, n2/py)
      // -> ((n0/2+1)/px, n1/py, n2) -> ((n0/2+1)/px, n1, n2/py)
      // -> (n0/2+1, n1/px, n2/py)
      // FFT ax = {0} -> Transpose topo 1 -> FFT ax = {1} -> Transpose topo 3
      // -> FFT ax = {2} -> Transpose topo 1 -> Transpose topo 0
      {0, 0, {2, 1, 0}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 1 with ax = {0, 1, 2}:
      // (n0, n1/px, n2/py) -> (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
      // -> (n0/py, n1, (n2/2+1)/px) -> (n0, n1/py, (n2/2+1)/px)
      // -> (n0/px, n1/py, n2/2+1) -> (n0/px, n1, (n2/2+1)/py)
      // Transpose topo 2 -> FFT ax = {2} -> Transpose topo 4 -> FFT ax = {1}
      // -> Transpose topo 5 -> FFT ax = {0} -> Transpose topo 3 -> topo 1
      {0, 1, {0, 1, 2}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 1 with ax = {0, 2, 1}:
      // (n0, n1/px, n2/py) -> (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      // -> (n0/px, (n1/2+1)/py, n2) -> (n0, (n1/2+1)/py, n2/px)
      // -> (n0/px, (n1/2+1)/py, n2) -> (n0/px, n1/2+1, n2/py)
      // Transpose topo 1 -> FFT ax = {1} -> Transpose topo 3 -> FFT ax = {2}
      // -> Transpose topo 5 -> FFT ax = {0} -> Transpose topo 3 -> topo 1
      {0, 1, {0, 2, 1}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 1 with ax = {1, 0, 2}:
      // (n0, n1/px, n2/py) -> (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
      // -> (n0, n1/px, (n2/2+1)/py) -> (n0/px, n1, (n2/2+1)/py)
      // Transpose topo 2 -> FFT ax = {2} -> Transpose topo 0 -> FFT ax = {0}
      // -> Transpose topo 1 -> FFT ax = {1}
      {0, 1, {1, 0, 2}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 1 with ax = {1, 2, 0}:
      // (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/py, n1/px, n2)
      // -> (n0/2+1, n1/px, n2/py) -> ((n0/2+1)/px, n1, n2/py)
      // FFT ax = {0} ->Transpose topo 2 -> FFT ax = {2} -> Transpose topo 0
      // -> Transpose topo 1 -> FFT ax = {1}
      {0, 1, {1, 2, 0}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 1 with ax = {2, 0, 1}:
      // (n0, n1/px, n2/py) -> (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      // -> (n0, (n1/2+1)/px, n2/py) -> (n0/py, (n1/2+1)/px, n2)
      // -> (n0, (n1/2+1)/px, n2/py) -> (n0/px, n1/2+1, n2/py)
      // Transpose 1 -> FFT ax = {1} -> Transpose 0 -> FFT ax = {0}
      // -> Transpose 2 -> FFT ax = {2} -> Transpose 0 -> Transpose 1
      {0, 1, {2, 0, 1}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 1 with ax = {2, 1, 0}:
      // (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/px, n1, n2/py)
      // -> ((n0/2+1)/px, n1/py, n2) -> ((n0/2+1)/px, n1, n2/py)
      // FFT ax = {0} ->Transpose topo 1 -> FFT ax = {1} -> Transpose topo 3
      // -> FFT ax = {2} -> Transpose 1
      {0, 1, {2, 1, 0}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 2 with ax = {0, 1, 2}:
      // (n0, n1/px, n2/py) -> (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
      // -> (n0/py, n1, (n2/2+1)/px) -> (n0, n1/py, (n2/2+1)/px)
      // -> (n0/py, n1, (n2/2+1)/px) -> (n0/py, n1/px, (n2/2+1))
      // Transpose topo 2 -> FFT ax = {2} -> Transpose topo 4 -> FFT ax = {1}
      // Transpose topo 5 -> FFT ax = {0} ->  Transpose topo 4
      // -> Transpose topo 2
      {0, 2, {0, 1, 2}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 2 with ax = {0, 2, 1}:
      // (n0, n1/px, n2/py) -> (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      // -> (n0/px, (n1/2+1)/py, n2) -> (n0, (n1/2+1)/py, n2/px)
      // -> (n0/py, n1/2+1, n2/px) -> (n0/py, (n1/2+1)/px, n2)
      // Transpose 1 -> FFT ax = {1} -> Transpose 3 -> FFT ax = {2}
      // -> Transpose 5 -> FFT ax = {0} -> Transpose 4 -> Transpose 2
      {0, 2, {0, 2, 1}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 2 with ax = {1, 0, 2}:
      // (n0, n1/px, n2/py) -> (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
      // -> (n0, n1/px, (n2/2+1)/py) -> (n0/px, n1, (n2/2+1)/py)
      // -> (n0, n1/px, (n2/2+1)/py) -> (n0/py, n1/px, n2/2+1)
      // Transpose topo 2 -> FFT ax = {2} -> Transpose topo 0 -> FFT ax = {0}
      // Transpose topo 1 -> FFT ax = {1} ->  Transpose topo 0
      // -> Transpose topo 2
      {0, 2, {1, 0, 2}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 2 with ax = {1, 2, 0}:
      // (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/py, n1/px, n2)
      // -> ((n0/2+1)/py, n1, n2/px) -> ((n0/2+1)/py, n1/px, n2)
      // FFT ax = {0} -> Transpose topo 2 -> FFT ax = {2}
      // -> Transpose 4 -> FFT ax = {1} -> Transpose topo 2
      {0, 2, {1, 2, 0}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 2 with ax = {2, 0, 1}:
      // (n0, n1/px, n2/py) -> (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      // -> (n0, (n1/2+1)/px, n2/py) -> (n0/py, (n1/2+1)/px, n2)
      // Transpose 1 -> FFT ax = {1} -> Transpose 0 -> FFT ax = {0}
      // -> Transpose 2 -> FFT ax = {2}
      {0, 2, {2, 0, 1}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 2 with ax = {2, 1, 0}:
      // (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/px, n1, n2/py)
      // ->(n0/2+1, n1/px, n2/py) -> ((n0/2+1)/py, n1/px, n2)
      // FFT ax = {0} -> Transpose topo 1 -> FFT ax = {1}
      // -> Transpose 0 -> Transpose topo 2 -> FFT ax = {2}
      {0, 2, {2, 1, 0}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 3 with ax = {0, 1, 2}:
      // (n0, n1/px, n2/py) -> (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
      // -> (n0/py, n1, (n2/2+1)/px) -> (n0, n1/py, (n2/2+1)/px)
      // -> (n0/px, n1/py, n2/2+1)
      // Transpose topo 2 -> FFT ax = {2} -> Transpose topo 4 -> FFT ax = {1}
      // -> Transpose topo 5 -> FFT ax = {0} -> Transpose topo 3
      {0, 3, {0, 1, 2}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 3 with ax = {0, 2, 1}:
      // (n0, n1/px, n2/py) -> (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      // -> (n0/px, (n1/2+1)/py, n2) -> (n0, (n1/2+1)/py, n2/px)
      // -> (n0/px, (n1/2+1)/py, n2)
      // Transpose topo 1 -> FFT ax = {1} -> Transpose topo 3 -> FFT ax = {2}
      // -> Transpose topo 5 -> FFT ax = {0} -> Transpose topo 3
      {0, 3, {0, 2, 1}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 3 with ax = {1, 0, 2}:
      // (n0, n1/px, n2/py) -> (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
      // -> (n0, n1/px, (n2/2+1)/py) -> (n0/px, n1, (n2/2+1)/py)
      // -> (n0/px, (n1/2+1)/py, n2)
      // Transpose topo 2 -> FFT ax = {2} -> Transpose topo 0 -> FFT ax = {0}
      // -> Transpose topo 1 -> FFT ax = {1} -> Transpose topo 3
      {0, 3, {1, 0, 2}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 3 with ax = {1, 2, 0}:
      // (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/py, n1/px, n2) -> ((n0/2+1)/py, n1, n2/px)
      // -> (n0/2+1, n1/py, n2/px) -> ((n0/2+1)/px, n1/py, n2)
      {0, 3, {1, 2, 0}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 3 with ax = {2, 0, 1}:
      // (n0, n1/px, n2/py) -> (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      // -> (n0, (n1/2+1)/px, n2/py) -> (n0/px, n1/2+1, n2/py)
      // -> (n0/px, (n1/2+1)/py, n2)
      {0, 3, {2, 0, 1}, Expect::Success, {}, tol_1e6},
      // topo 0 -> topo 3 with ax = {2, 1, 0}:
      // (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/px, n1, n2/py) -> ((n0/2+1)/px, n1/py, n2)
      {0, 3, {2, 1, 0}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 0 with ax = {0, 1, 2}:
      // (n0/px, n1, n2/py) -> (n0/px, n1/py, n2) -> (n0/px, n1/py, n2/2+1)
      // -> (n0/px, n1, (n2/2+1)/py) -> (n0, n1/px, (n2/2+1)/py)
      // Transpose 3 -> FFT ax = {2} -> Transpose 1 -> FFT ax = {1}
      // Transpose 0 -> FFT ax = {0}
      {1, 0, {0, 1, 2}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 0 with ax = {0, 2, 1}:
      // (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      // -> (n0/px, (n1/2+1)/py, n2)
      // -> (n0/px, n1/2+1, n2/py) -> (n0, (n1/2+1)/px, n2/py)
      // FFT ax = {1} -> Transpose 3 -> FFT ax = {2} -> Transpose 1
      // -> Transpose 0 -> FFT ax = {0}
      {1, 0, {0, 2, 1}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 0 with ax = {1, 0, 2}:
      // (n0/px, n1, n2/py) -> (n0/px, n1/py, n2) -> (n0/px, n1/py, n2/2+1)
      // -> (n0, n1/py, (n2/2+1)/px) -> (n0/py, n1, (n2/2+1)/px)
      // -> (n0/py, n1/px, n2/2+1) -> (n0, n1/px, (n2/2+1)/py)
      // Transpose 3 -> FFT ax = {2} -> Transpose 5 -> FFT ax = {0}
      // Transpose 4 -> FFT ax = {1} -> Transpose 2 -> Transpose 0
      {1, 0, {1, 0, 2}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 0 with ax = {1, 2, 0}:
      // (n0/px, n1, n2/py) -> (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/py, n1/px, n2) -> ((n0/2+1)/py, n1, n2/px)
      // -> ((n0/2+1)/py, n1/px, n2) -> -> (n0/2+1, n1/px, n2/py)
      // Transpose 0 -> FFT ax = {0} -> Transpose 2 -> FFT ax = {2}
      // Transpose 4 -> FFT ax = {1} -> Transpose 2 -> Transpose 0
      {1, 0, {1, 2, 0}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 0 with ax = {2, 0, 1}:
      // (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      // -> (n0, (n1/2+1)/px, n2/py)
      // -> (n0/py, (n1/2+1)/px, n2) -> (n0, (n1/2+1)/px, n2/py)
      // FFT ax = {1} -> Transpose 0 -> FFT ax = {0} -> Transpose 2
      // -> FFT ax = {2} -> Transpose 0
      {1, 0, {2, 0, 1}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 0 with ax = {2, 1, 0}:
      // (n0/px, n1, n2/py) -> (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/px, n1, n2/py) -> ((n0/2+1)/px, n1/py, n2)
      // -> ((n0/2+1)/px, n1, n2/py) -> -> (n0/2+1, n1/px, n2/py)
      // Transpose 0 -> FFT ax = {0} -> Transpose 1 -> FFT ax = {1}
      // Transpose 3 -> FFT ax = {2} -> Transpose 1 -> Transpose 0
      {1, 0, {2, 1, 0}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 1 with ax = {0, 1, 2}:
      // (n0/px, n1, n2/py) -> (n0/px, n1/py, n2) -> (n0/px, n1/py, n2/2+1)
      // -> (n0/px, n1, (n2/2+1)/py) -> (n0, n1/px, (n2/2+1)/py)
      // -> (n0/px, n1, (n2/2+1)/py)
      // Transpose 3 -> FFT ax = {2} -> Transpose 1 -> FFT ax = {1}
      // Transpose 0 -> FFT ax = {0} -> Transpose 1
      {1, 1, {0, 1, 2}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 1 with ax = {0, 2, 1}:
      // (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      // -> (n0/px, (n1/2+1)/py, n2)
      // -> (n0, (n1/2+1)/py, n2/px) -> (n0/px, (n1/2+1)/py, n2)
      // -> (n0/px, n1/2+1, n2/py)
      // FFT ax = {1} -> Transpose 3 -> FFT ax = {2} -> Transpose 5
      // -> FFT ax = {0} -> Transpose 3 -> Transpose 1
      {1, 1, {0, 2, 1}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 1 with ax = {1, 0, 2}:
      // (n0/px, n1, n2/py) -> (n0/px, n1/py, n2) -> (n0/px, n1/py, n2/2+1)
      // -> (n0, n1/py, (n2/2+1)/px) -> (n0/px, n1/py, n2/2+1)
      // -> (n0/px, n1, (n2/2+1)/py)
      // Transpose 3 -> FFT ax = {2} -> Transpose 5 -> FFT ax = {0}
      // Transpose 3 -> Transpose 1 -> FFT ax = {1}
      {1, 1, {1, 0, 2}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 1 with ax = {1, 2, 0}:
      // (n0/px, n1, n2/py) -> (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/py, n1/px, n2) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/px, n1, n2/py)
      // Transpose 0 -> FFT ax = {0} -> Transpose 2 -> FFT ax = {2}
      // Transpose 0 -> Transpose 1 -> FFT ax = {1}
      {1, 1, {1, 2, 0}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 1 with ax = {2, 0, 1}:
      // (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      // -> (n0, (n1/2+1)/px, n2/py)
      // -> (n0/py, (n1/2+1)/px, n2) -> (n0, (n1/2+1)/px, n2/py)
      // -> (n0/px, n1/2+1, n2/py)
      // FFT ax = {1} -> Transpose 0 -> FFT ax = {0} -> Transpose 2
      // -> FFT ax = {2} -> Transpose 0 -> Transpose 1
      {1, 1, {2, 0, 1}, Expect::Success, {}, tol_1e6},
      {1, 1, {2, 1, 0}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 2 with ax = {0, 1, 2}:
      // (n0/px, n1, n2/py) -> (n0/px, n1/py, n2) -> (n0/px, n1/py, n2/2+1)
      // -> (n0/px, n1, (n2/2+1)/py) -> (n0, n1/px, (n2/2+1)/py)
      // -> (n0/py, n1/px, n2/2+1)
      // Transpose 3 -> FFT ax = {2} -> Transpose 1-> FFT ax = {1}
      // -> Transpose 0 -> FFT ax ={0} -> Transpose 2
      {1, 2, {0, 1, 2}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 2 with ax = {0, 2, 1}:
      // (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py)
      // -> (n0/px, (n1/2+1)/py, n2)
      // -> (n0/px, (n1/2+1)/py, n2) -> (n0, (n1/2+1)/py, n2/px)
      // -> (n0/py, (n1/2+1), n2/px) -> (n0/py, (n1/2+1)/px, n2)
      // FFT ax = {1} -> Transpose 3 -> FFT ax = {2} -> Transpose 5
      // -> FFT ax = {0} -> Transpose 4 -> Transpose 2
      {1, 2, {0, 2, 1}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 2 with ax = {1, 0, 2}:
      // (n0/px, n1, n2/py) -> (n0/px, n1/py, n2) -> (n0/px, n1/py, n2/2+1)
      // -> (n0, n1/py, (n2/2+1)/px) -> (n0/py, n1, (n2/2+1)/px)
      // -> (n0/py, n1/px, n2/2+1)
      {1, 2, {1, 0, 2}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 2 with ax = {1, 2, 0}:
      // (n0/px, n1, n2/py) -> (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/py, n1/px, n2) -> ((n0/2+1)/py, n1, n2/px)
      // -> ((n0/2+1)/py, n1/px, n2)
      // Transpose 0 -> FFT ax = {0} -> Transpose 2 -> FFT ax = {2}
      // Transpose 4 -> FFT ax = {1} -> Transpose 2
      {1, 2, {1, 2, 0}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 2 with ax = {2, 0, 1}:
      // (n0/px, n1, n2/py) -> (n0/px, n1/2+1, n2/py) -> (n0, n1/2+1/px, n2/py)
      // -> (n0/py, n1/2+1/px, n2)
      // FFT ax = {1} -> Transpose 0 -> FFT ax = {0} -> Transpose 2
      // -> FFT ax = {2}
      {1, 2, {2, 0, 1}, Expect::Success, {}, tol_1e6},
      // topo 1 -> topo 2 with ax = {2, 1, 0}:
      // (n0/px, n1, n2/py) -> (n0, n1/px, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/px, n1, n2/py) -> (n0/2+1, n1/px, n2/py)
      // -> ((n0/2+1)/py, n1/px, n2)
      // Transpose 0 -> FFT ax = {0} -> Transpose 1 -> FFT ax = {1}
      // Transpose 0 -> Transpose 2 -> FFT ax = {2}
      {1, 2, {2, 1, 0}, Expect::Success, {}, tol_1e6},
      // topo 2 -> topo 0 with ax = {0, 1, 2}:
      // (n0/py, n1/px, n2) -> (n0/py, n1/px, n2/2+1)
      // -> (n0/py, n1, (n2/1+1)/px)
      // -> (n0/py, n1/px, n2/2+1) -> (n0, n1/px, (n2/1+1)/py)
      // FFT ax = {2} -> Transpose 4 -> FFT ax = {1} -> Transpose 2
      // -> Transpose 0 -> FFT ax = {0}
      {2, 0, {0, 1, 2}, Expect::Success, {}, tol_1e6},
      // topo 2 -> topo 1 with ax = {0, 1, 2}:
      // (n0/p, n1, n2) -> (n0/p, n1, n2/2+1) -> (n0, n1/p, n2/2+1)
      // FFT2 ax = {1, 2} -> Transpose -> FFT ax = {0}
      {2, 1, {0, 1, 2}, Expect::Success, {}, tol_1e6},
      // topo 3 -> topo 0 with ax = {0, 1, 2}:
      // (n0/px, n1/py, n2) -> (n0/px, n1/py, n2/2+1)
      // -> (n0/px, n1, (n2/2+1)/py) -> (n0, n1/px, (n2/2+1)/py)
      // FFT ax = {2} -> Transpose topo 1 -> FFT ax = {1} -> Transpose topo 0
      // -> FFT ax = {0}
      {3, 0, {0, 1, 2}, Expect::Success, {}, tol_1e6},
  };

  const std::array<std::size_t, 3> global_extents{8, 7, 5};
  auto gu = make_global_input<View3DType>(global_extents);

  const std::string name = "Test_PencilPlan.cpp test_pencil3D_view3D";
  const bool is_failure  = npx == 1 || npy == 1;
  run_plan_cases<ComplexView3DType>(name, execution_space(), gu, topologies,
                                    is_failure ? failure_cases : success_cases,
                                    PencilPlanOps{});
}

template <typename T, typename LayoutType>
void test_pencil3D_view4D(std::size_t npx, std::size_t npy) {
  using View4DType        = Kokkos::View<T****, LayoutType, execution_space>;
  using float_type        = KokkosFFT::Impl::base_floating_point_type<T>;
  using ComplexView4DType = Kokkos::View<Kokkos::complex<float_type>****,
                                         LayoutType, execution_space>;
  using topology_r_type =
      KokkosFFT::Distributed::Topology<std::size_t, 4, Kokkos::LayoutRight>;
  using topology_l_type =
      KokkosFFT::Distributed::Topology<std::size_t, 4, Kokkos::LayoutLeft>;

  // x-pencil, y-pencil and z-pencil
  const auto topologies = std::make_tuple(
      topology_r_type{1, 1, npx, npy}, topology_r_type{1, npx, 1, npy},
      topology_l_type{1, npy, npx, 1}, topology_r_type{1, npx, npy, 1});
  const float_type atol = std::is_same_v<float_type, float> ? 3.0e-6 : 1.0e-12;
  const Tolerance tol_atol{1.0e-5, atol};

  // Not a pencil geometry: the plan must fail
  const std::vector<PlanCase<3>> failure_cases = {
      // topology0 -> topology1
      // (n0, n1/px, n2/py) -> (n0, (n1/2+1)/p, n2)
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
      // (n0, n1, n2/px, n3/py) -> (n0/px, n1, n2, n3/py)
      // -> (n0, n1, n2/px, n3/py)
      // Transpose topo 6 -> FFT2 ax = {1,2} -> Transpose topo 0
      // -> FFT ax = {0}
      {0, 0, {0, 1, 2}, Expect::Success, {}, tol_atol},
      // topo 0 -> topo 0 with ax = {1, 2, 0}
      // (n0, n1, n2/px, n3/py) -> (n0/2+1, n1, n2/px, n3/py)
      // -> ((n0/2+1)/px, n1, n2, n3/py) -> (n0/2+1, n1, n2/px, n3/py)
      // FFT ax = {0} -> Transpose topo 6 -> FFT2 ax = {1,2} -> Transpose topo 0
      {0, 0, {1, 2, 0}, Expect::Success, {}, tol_atol},
      // topo 0 -> topo 0 with ax = {1, 2, 3}
      // (n0, n1, n2/px, n3/py) -> (n0/py, n1, n2/px, n3)
      // -> (n0/py, n1, n2/px, n3/2+1) -> (n0/py, n1/px, n2, n3/2+1)
      // -> (n0/py, n1, n2/px, n3/2+1) -> (n0, n1, n2/px, (n3/2+1)/py)
      // Transpose topo 7 -> FFT ax = {3} -> Transpose topo 9
      // -> FFT ax = {2} -> Transpose topo 7 -> FFT ax = {1}
      // -> Transpose topo 1
      {0, 0, {1, 2, 3}, Expect::Success, tol_atol, tol_atol},
      // topo 0 -> topo 0 with ax = {1, 3, 2}
      // (n0, n1, n2/px, n3/py) -> (n0/px, n1, n2, n3/py)
      // -> (n0/px, n1, n2/2+1, n3/py) -> (n0/px, n1/py, n2/2+1, n3)
      // -> (n0/px, n1, n2/2+1, n3/py) -> (n0, n1, (n2/2+1)/px, n3/py)
      // Transpose topo 6 -> FFT ax = {2} -> Transpose topo 10
      // -> FFT ax = {3} -> Transpose topo 6 -> FFT ax = {1}
      // -> Transpose topo 0
      {0, 0, {1, 3, 2}, Expect::Success, tol_atol, tol_atol},
      // topo 0 -> topo 3 with ax = {0, 1, 2}:
      // (n0, n1, n2/px, n3/py) -> (n0, n1/px, n2, n3/py)
      // -> (n0, n1/px, n2/2+1, n3/py) -> (n0, n1, (n2/2+1)/px, n3/py)
      // -> (n0, n1/px, n2/2+1, n3/py) -> (n0, n1/px, (n2/2+1)/py, n3)
      // Transpose topo 1 -> FFT ax = {2} -> Transpose topo 0
      // -> FFT ax = {0, 1} -> Transpose topo 1 -> Transpose topo 3
      {0, 3, {0, 1, 2}, Expect::Success, {}, tol_atol},
      // topo 3 -> topo 0 with ax = {1, 2, 3}:
      // (n0, n1/px, n2/py, n3) -> (n0, n1/px, n2/py, n3/2+1)
      // -> (n0, n1/px, n2, (n3/2+1)/py) -> (n0, n1, n2/px, (n3/2+1)/py)
      // FFT ax = {2} -> Transpose topo 1 -> FFT ax = {1} -> Transpose topo 0
      // -> FFT ax = {0}
      {3, 0, {1, 2, 3}, Expect::Success, tol_atol, tol_atol},
  };

  const std::array<std::size_t, 4> global_extents{8, 7, 5, 6};
  auto gu = make_global_input<View4DType>(global_extents);

  const std::string name = "Test_PencilPlan.cpp test_pencil3D_view4D";
  const bool is_failure  = npx == 1 || npy == 1;
  run_plan_cases<ComplexView4DType>(name, execution_space(), gu, topologies,
                                    is_failure ? failure_cases : success_cases,
                                    PencilPlanOps{});
}

}  // namespace

TYPED_TEST_SUITE(TestPencil1D, test_types);
TYPED_TEST_SUITE(TestPencil2D, test_types);
TYPED_TEST_SUITE(TestPencil3D, test_types);

TYPED_TEST(TestPencil1D, View3D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_pencil1D_view3D<float_type, layout_type>(this->m_npx, this->m_npx);
}

TYPED_TEST(TestPencil1D, View3D_C2C) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;

  test_pencil1D_view3D<complex_type, layout_type>(this->m_npx, this->m_npx);
}

TYPED_TEST(TestPencil2D, View3D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_pencil2D_view3D<float_type, layout_type>(this->m_npx, this->m_npx);
}

TYPED_TEST(TestPencil2D, View3D_C2C) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;

  test_pencil2D_view3D<complex_type, layout_type>(this->m_npx, this->m_npx);
}

TYPED_TEST(TestPencil2D, View4D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_pencil2D_view4D<float_type, layout_type>(this->m_npx, this->m_npx);
}

TYPED_TEST(TestPencil2D, View4D_C2C) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;

  test_pencil2D_view4D<complex_type, layout_type>(this->m_npx, this->m_npx);
}

TYPED_TEST(TestPencil3D, View3D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_pencil3D_view3D<float_type, layout_type>(this->m_npx, this->m_npx);
}

TYPED_TEST(TestPencil3D, View3D_C2C) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;

  test_pencil3D_view3D<complex_type, layout_type>(this->m_npx, this->m_npx);
}

TYPED_TEST(TestPencil3D, View4D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_pencil3D_view4D<float_type, layout_type>(this->m_npx, this->m_npx);
}

TYPED_TEST(TestPencil3D, View4D_C2C) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;

  test_pencil3D_view4D<complex_type, layout_type>(this->m_npx, this->m_npx);
}
