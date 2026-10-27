#include <mpi.h>
#include <gtest/gtest.h>
#include <array>
#include <stdexcept>
#include <string>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_BasePlan.hpp"
#include "KokkosFFT_Distributed_Extents.hpp"
#include "KokkosFFT_Distributed_MPI_Extents.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using layout_types = ::testing::Types<Kokkos::LayoutLeft, Kokkos::LayoutRight>;

template <typename T>
struct TestBasePlan : public ::testing::Test {
  using layout_type = T;

  std::size_t m_nprocs = 1;

  virtual void SetUp() {
    int nprocs = 1;
    ::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    m_nprocs = static_cast<std::size_t>(nprocs);
  }
};

/// \brief Minimal plan to test the constructors of BasePlan. It inherits
/// both constructors (std::array and Topology topologies), and exposes what
/// BasePlan computes.
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM>
class MinimalPlan final
    : public KokkosFFT::Distributed::Impl::BasePlan<ExecutionSpace, InViewType,
                                                    OutViewType, DIM> {
  using base_type =
      KokkosFFT::Distributed::Impl::BasePlan<ExecutionSpace, InViewType,
                                             OutViewType, DIM>;

 public:
  using base_type::base_type;

  void forward(const InViewType& /*in*/,
               const OutViewType& /*out*/) const override {}
  void backward(const OutViewType& /*out*/,
                const InViewType& /*in*/) const override {}
  std::string name() const override { return "MinimalPlan"; }

  auto fft_extents() const { return this->get_fft_extents(); }
  auto norm() const { return this->get_norm(); }
  void check(const InViewType& in, const OutViewType& out) const {
    this->good(in, out);
  }
};

/// \brief BasePlan built from std::array topologies must be the same as
/// built from Topology topologies: same FFT extents, normalization and
/// extents checks
template <typename LayoutType>
void test_base_plan_constructors(std::size_t nprocs) {
  using RealView2DType = Kokkos::View<double**, LayoutType, execution_space>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<double>**, LayoutType, execution_space>;
  using PlanType =
      MinimalPlan<execution_space, RealView2DType, ComplexView2DType, 1>;
  using extents_type  = std::array<std::size_t, 2>;
  using topology_type = KokkosFFT::Distributed::Topology<std::size_t, 2>;

  // (n0, n1/p) -> R2C along axis 1 -> (n0/p, n1/2+1)
  const extents_type global_in_extents{8, 7}, global_out_extents{8, 4};
  const extents_type in_topology{1, nprocs}, out_topology{nprocs, 1};

  auto [in_extents, in_starts] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_in_extents, in_topology, MPI_COMM_WORLD);
  auto [out_extents, out_starts] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents, out_topology, MPI_COMM_WORLD);

  RealView2DType in("in",
                    KokkosFFT::Impl::create_layout<LayoutType>(in_extents));
  ComplexView2DType out(
      "out", KokkosFFT::Impl::create_layout<LayoutType>(out_extents));

  const execution_space exec;
  const KokkosFFT::axis_type<1> axes{1};
  const auto norm = KokkosFFT::Normalization::ortho;

  PlanType plan_from_array(exec, in, out, axes, in_topology, out_topology,
                           MPI_COMM_WORLD, norm);
  PlanType plan_from_topology(exec, in, out, axes, topology_type(in_topology),
                              topology_type(out_topology), MPI_COMM_WORLD,
                              norm);

  // The FFT size is the larger of the global input and output extents
  const std::array<std::size_t, 1> ref_fft_extents{7};
  EXPECT_EQ(plan_from_array.fft_extents(), ref_fft_extents);
  EXPECT_EQ(plan_from_topology.fft_extents(), ref_fft_extents);
  EXPECT_EQ(plan_from_array.norm(), norm);
  EXPECT_EQ(plan_from_topology.norm(), norm);

  // The views of the plan pass the extents check, others do not
  EXPECT_NO_THROW(plan_from_array.check(in, out));
  EXPECT_NO_THROW(plan_from_topology.check(in, out));

  RealView2DType wrong_in("wrong_in", in.extent(0) + 1, in.extent(1));
  EXPECT_THROW(plan_from_array.check(wrong_in, out), std::runtime_error);
  EXPECT_THROW(plan_from_topology.check(wrong_in, out), std::runtime_error);
}

/// \brief In-place detection and validation: a plan is in-place when its
/// input and output Views point to the same data. It must be executed in the
/// placement it was created for, and partially overlapping Views are
/// rejected, at construction and at execution.
template <typename LayoutType>
void test_base_plan_placement(std::size_t nprocs) {
  using RealView2DType = Kokkos::View<double**, LayoutType, execution_space>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<double>**, LayoutType, execution_space>;
  using BufferType = Kokkos::View<Kokkos::complex<double>*, execution_space>;
  using PlanType =
      MinimalPlan<execution_space, RealView2DType, ComplexView2DType, 1>;
  using extents_type = std::array<std::size_t, 2>;

  // (n0, n1/p) -> R2C along axis 1 -> (n0/p, n1/2+1)
  const extents_type global_in_extents{8, 7}, global_out_extents{8, 4};
  const extents_type in_topology{1, nprocs}, out_topology{nprocs, 1};

  auto [in_extents, in_starts] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_in_extents, in_topology, MPI_COMM_WORLD);
  auto [out_extents, out_starts] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          global_out_extents, out_topology, MPI_COMM_WORLD);
  const auto in_layout = KokkosFFT::Impl::create_layout<LayoutType>(in_extents);
  const auto out_layout =
      KokkosFFT::Impl::create_layout<LayoutType>(out_extents);

  // Out-of-place: separate allocations
  RealView2DType in("in", in_layout);
  ComplexView2DType out("out", out_layout);

  // In-place: one allocation, large enough for both Views on this rank, and
  // one more element for the partially overlapping View below
  const std::size_t size =
      KokkosFFT::Distributed::compute_inplace_allocation_size<
          double, Kokkos::complex<double>>(in_extents, out_extents);
  BufferType buffer("buffer", size + 1);
  RealView2DType in_inplace(reinterpret_cast<double*>(buffer.data()),
                            in_layout);
  ComplexView2DType out_inplace(buffer.data(), out_layout);

  // Starts one element after the input: overlaps it partially
  ComplexView2DType out_partial(buffer.data() + 1, out_layout);

  const execution_space exec;
  const KokkosFFT::axis_type<1> axes{1};
  const auto norm = KokkosFFT::Normalization::backward;

  PlanType plan(exec, in, out, axes, in_topology, out_topology, MPI_COMM_WORLD,
                norm);
  PlanType plan_inplace(exec, in_inplace, out_inplace, axes, in_topology,
                        out_topology, MPI_COMM_WORLD, norm);
  EXPECT_FALSE(plan.is_inplace());
  EXPECT_TRUE(plan_inplace.is_inplace());

  // Each plan accepts its own placement only
  EXPECT_NO_THROW(plan.check(in, out));
  EXPECT_NO_THROW(plan_inplace.check(in_inplace, out_inplace));
  EXPECT_THROW(plan.check(in_inplace, out_inplace), std::runtime_error);
  EXPECT_THROW(plan_inplace.check(in, out), std::runtime_error);

  // Partially overlapping Views: rejected at construction and at execution
  EXPECT_THROW(
      {
        PlanType plan_partial(exec, in_inplace, out_partial, axes, in_topology,
                              out_topology, MPI_COMM_WORLD, norm);
      },
      std::runtime_error);
  EXPECT_THROW(plan.check(in_inplace, out_partial), std::runtime_error);
  EXPECT_THROW(plan_inplace.check(in_inplace, out_partial), std::runtime_error);
}

}  // namespace

TYPED_TEST_SUITE(TestBasePlan, layout_types);

TYPED_TEST(TestBasePlan, constructors) {
  using layout_type = typename TestFixture::layout_type;
  test_base_plan_constructors<layout_type>(this->m_nprocs);
}

TYPED_TEST(TestBasePlan, placement) {
  using layout_type = typename TestFixture::layout_type;
  test_base_plan_placement<layout_type>(this->m_nprocs);
}
