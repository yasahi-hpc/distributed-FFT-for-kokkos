#include <type_traits>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_Plan.hpp"

// Copy and move semantics of the plan classes (refactor-classes.md).
// Only the public Plan is movable. The internal classes are constructible
// only from their arguments: they are built in place (std::make_unique in
// plan_factory), and never copied or moved.
//
// The checks are compile-time only. The traits do not depend on the value
// type, the layouts or DIM, so one instantiation per class is enough.

namespace {
namespace Impl = KokkosFFT::Distributed::Impl;

using execution_space = Kokkos::DefaultExecutionSpace;
using RealViewType =
    Kokkos::View<double***, Kokkos::LayoutRight, execution_space>;
using ComplexViewType = Kokkos::View<Kokkos::complex<double>***,
                                     Kokkos::LayoutRight, execution_space>;

using PlanType = KokkosFFT::Distributed::Plan<execution_space, RealViewType,
                                              ComplexViewType, 1>;
using BasePlanType =
    Impl::BasePlan<execution_space, RealViewType, ComplexViewType, 1>;
using SharedPlanType =
    Impl::SharedPlan<execution_space, RealViewType, ComplexViewType, 1>;
using SlabPlanType =
    Impl::SlabPlan<execution_space, RealViewType, ComplexViewType, 1>;
using PencilPlanType =
    Impl::PencilPlan<execution_space, RealViewType, ComplexViewType, 1,
                     Kokkos::LayoutRight, Kokkos::LayoutLeft>;

using SharedPlanExecutorType =
    Impl::SharedPlanExecutor<execution_space, RealViewType, ComplexViewType, 1>;
using SlabPlanExecutorType =
    Impl::SlabPlanExecutor<execution_space, RealViewType, ComplexViewType, 1>;
using PencilPlanExecutorType =
    Impl::PencilPlanExecutor<execution_space, RealViewType, ComplexViewType, 1,
                             Kokkos::LayoutRight, Kokkos::LayoutLeft>;
using BlockExecutorType =
    Impl::BlockExecutor<execution_space, RealViewType, ComplexViewType>;
using FFTStageType =
    Impl::FFTStage<execution_space, RealViewType, ComplexViewType>;
using PencilCommsType = Impl::PencilComms<execution_space>;

/// \brief Neither default constructible, nor copyable
template <typename T>
inline constexpr bool is_non_copyable_v =
    !std::is_default_constructible_v<T> && !std::is_copy_constructible_v<T> &&
    !std::is_copy_assignable_v<T>;

/// \brief Constructible only from its arguments: neither default
/// constructible, nor copyable, nor movable
template <typename T>
inline constexpr bool is_pinned_v =
    is_non_copyable_v<T> && !std::is_move_constructible_v<T> &&
    !std::is_move_assignable_v<T>;

// Public plan: movable, not copyable, not default constructible.
// The move only moves the std::unique_ptr to the internal plan, so it is
// noexcept, which std::vector<Plan> needs to move its elements.
static_assert(is_non_copyable_v<PlanType>);
static_assert(std::is_nothrow_move_constructible_v<PlanType>);
static_assert(std::is_nothrow_move_assignable_v<PlanType>);

// Internal classes: constructible only from their arguments. They own
// communicators, FFT library plans and buffers that must not move.
static_assert(is_pinned_v<SharedPlanType>);
static_assert(is_pinned_v<SlabPlanType>);
static_assert(is_pinned_v<PencilPlanType>);
static_assert(is_pinned_v<SharedPlanExecutorType>);
static_assert(is_pinned_v<SlabPlanExecutorType>);
static_assert(is_pinned_v<PencilPlanExecutorType>);
static_assert(is_pinned_v<BlockExecutorType>);
static_assert(is_pinned_v<FFTStageType>);
static_assert(is_pinned_v<PencilCommsType>);

// BasePlan is abstract, so it is never constructible; only its assignments
// can be checked
static_assert(std::is_abstract_v<BasePlanType>);
static_assert(!std::is_copy_assignable_v<BasePlanType>);
static_assert(!std::is_move_assignable_v<BasePlanType>);

}  // namespace

// The checks above are compile-time; this test only records them in the
// test report
TEST(TestPlanTraits, copy_and_move_semantics) { SUCCEED(); }
