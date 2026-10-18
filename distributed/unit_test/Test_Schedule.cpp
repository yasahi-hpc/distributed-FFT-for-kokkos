#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <mpi.h>
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_Schedule.hpp"
#include "KokkosFFT_Distributed_PencilPlan.hpp"
#include "KokkosFFT_Distributed_MPI_Extents.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;

constexpr std::size_t DIM = 3;
using extents_type        = std::array<std::size_t, DIM>;
using map_type            = std::array<std::size_t, DIM>;
using BlockInfoType       = KokkosFFT::Distributed::Impl::BlockInfo<DIM>;
using BlockStepType       = KokkosFFT::Distributed::Impl::BlockStep<DIM>;
using ViewSpecType        = KokkosFFT::Distributed::Impl::ViewSpec<DIM>;
using KokkosFFT::Distributed::Impl::BlockType;
using KokkosFFT::Distributed::Impl::BufferId;

const map_type identity_map = {0, 1, 2}, permuted_map = {0, 2, 1};

// Distinct extents, so that the extents of each view are checked as well
const extents_type e0 = {8, 6, 4}, e1 = {4, 12, 4}, e2 = {12, 4, 4},
                   e3 = {4, 4, 12}, e4 = {6, 8, 4}, e5 = {4, 6, 8},
                   e6 = {2, 6, 16};

/// \brief Make a transpose block. Only the members used by build_schedule are
/// set.
BlockInfoType trans_block(const extents_type& in_extents,
                          const extents_type& out_extents,
                          std::size_t comm_axis = 0) {
  BlockInfoType block;
  block.m_block_type  = BlockType::Transpose;
  block.m_in_extents  = in_extents;
  block.m_out_extents = out_extents;
  block.m_comm_axis   = comm_axis;
  return block;
}

/// \brief Make an FFT block. Only the members used by build_schedule are set.
BlockInfoType fft_block(const extents_type& in_extents,
                        const extents_type& out_extents,
                        const map_type& map = identity_map) {
  BlockInfoType block;
  block.m_block_type  = BlockType::FFT;
  block.m_in_extents  = in_extents;
  block.m_out_extents = out_extents;
  block.m_in_map      = map;
  block.m_out_map     = map;
  return block;
}

/// \brief Make a reference step
BlockStepType ref_step(BlockType type, std::size_t sub_idx,
                       const ViewSpecType& src, const ViewSpecType& dst,
                       std::optional<ViewSpecType> src_T = std::nullopt,
                       std::optional<ViewSpecType> dst_T = std::nullopt,
                       std::size_t comm_axis             = 0) {
  BlockStepType step;
  step.m_type      = type;
  step.m_sub_idx   = sub_idx;
  step.m_comm_axis = comm_axis;
  step.m_src       = src;
  step.m_dst       = dst;
  step.m_src_T     = src_T;
  step.m_dst_T     = dst_T;
  return step;
}

// Views before (input type) and after (output type) the first FFT
ViewSpecType in_spec(BufferId id, const extents_type& extents) {
  return ViewSpecType{id, extents, true};
}
ViewSpecType out_spec(BufferId id, const extents_type& extents) {
  return ViewSpecType{id, extents, false};
}

constexpr auto Trans = BlockType::Transpose;
constexpr auto FFT   = BlockType::FFT;

// FFT only: Slab/Pencil, 1D/2D/3D, e.g. FFT on the input topology
void test_schedule_fft() {
  auto steps =
      KokkosFFT::Distributed::Impl::build_schedule<DIM>({fft_block(e0, e1)});
  std::vector<BlockStepType> ref = {ref_step(
      FFT, 0, in_spec(BufferId::UserIn, e0), out_spec(BufferId::UserOut, e1))};
  EXPECT_EQ(steps, ref);
}

// FFT only with local transposes of the input and output
void test_schedule_fft_with_maps() {
  auto steps = KokkosFFT::Distributed::Impl::build_schedule<DIM>(
      {fft_block(e0, e1, permuted_map)});
  std::vector<BlockStepType> ref = {ref_step(
      FFT, 0, in_spec(BufferId::UserIn, e0), out_spec(BufferId::UserOut, e1),
      in_spec(BufferId::Send, e0), out_spec(BufferId::Recv, e1))};
  EXPECT_EQ(steps, ref);
}

// Transpose + FFT: the transpose writes into m_in_T (Send) even though the
// FFT is the last block, since the data still has the input value type
void test_schedule_trans_fft() {
  auto steps = KokkosFFT::Distributed::Impl::build_schedule<DIM>(
      {trans_block(e0, e1, 1), fft_block(e1, e2)});
  std::vector<BlockStepType> ref = {
      ref_step(Trans, 0, in_spec(BufferId::UserIn, e0),
               in_spec(BufferId::Send, e1), std::nullopt, std::nullopt, 1),
      ref_step(FFT, 0, in_spec(BufferId::Send, e1),
               out_spec(BufferId::UserOut, e2))};
  EXPECT_EQ(steps, ref);
}

// FFT + Transpose: the FFT writes into m_out_T (Recv)
void test_schedule_fft_trans() {
  auto steps = KokkosFFT::Distributed::Impl::build_schedule<DIM>(
      {fft_block(e0, e1), trans_block(e1, e2)});
  std::vector<BlockStepType> ref = {
      ref_step(FFT, 0, in_spec(BufferId::UserIn, e0),
               out_spec(BufferId::Recv, e1)),
      ref_step(Trans, 0, out_spec(BufferId::Recv, e1),
               out_spec(BufferId::UserOut, e2))};
  EXPECT_EQ(steps, ref);
}

// Transpose + FFT + Transpose
void test_schedule_trans_fft_trans() {
  auto steps = KokkosFFT::Distributed::Impl::build_schedule<DIM>(
      {trans_block(e0, e1), fft_block(e1, e2), trans_block(e2, e3)});
  std::vector<BlockStepType> ref = {
      ref_step(Trans, 0, in_spec(BufferId::UserIn, e0),
               in_spec(BufferId::Send, e1)),
      ref_step(FFT, 0, in_spec(BufferId::Send, e1),
               out_spec(BufferId::Recv, e2)),
      ref_step(Trans, 1, out_spec(BufferId::Recv, e2),
               out_spec(BufferId::UserOut, e3))};
  EXPECT_EQ(steps, ref);
}

// Transpose + FFT + Transpose + Transpose (Pencil 1D): the second transpose
// writes into m_out_T2 (Send)
void test_schedule_trans_fft_trans_trans() {
  auto steps = KokkosFFT::Distributed::Impl::build_schedule<DIM>(
      {trans_block(e0, e1, 0), fft_block(e1, e2), trans_block(e2, e3, 1),
       trans_block(e3, e4, 0)});
  std::vector<BlockStepType> ref = {
      ref_step(Trans, 0, in_spec(BufferId::UserIn, e0),
               in_spec(BufferId::Send, e1), std::nullopt, std::nullopt, 0),
      ref_step(FFT, 0, in_spec(BufferId::Send, e1),
               out_spec(BufferId::Recv, e2)),
      ref_step(Trans, 1, out_spec(BufferId::Recv, e2),
               out_spec(BufferId::Send, e3), std::nullopt, std::nullopt, 1),
      ref_step(Trans, 2, out_spec(BufferId::Send, e3),
               out_spec(BufferId::UserOut, e4), std::nullopt, std::nullopt, 0)};
  EXPECT_EQ(steps, ref);
}

// FFT + Transpose + FFT (Slab/Pencil 2D): the transpose writes directly into
// the output and the last FFT runs in-place on it
void test_schedule_fft_trans_fft() {
  auto steps = KokkosFFT::Distributed::Impl::build_schedule<DIM>(
      {fft_block(e0, e1), trans_block(e1, e2), fft_block(e2, e2)});
  std::vector<BlockStepType> ref = {
      ref_step(FFT, 0, in_spec(BufferId::UserIn, e0),
               out_spec(BufferId::Recv, e1)),
      ref_step(Trans, 0, out_spec(BufferId::Recv, e1),
               out_spec(BufferId::UserOut, e2)),
      ref_step(FFT, 1, out_spec(BufferId::UserOut, e2),
               out_spec(BufferId::UserOut, e2))};
  EXPECT_EQ(steps, ref);
}

// FFT + Transpose + FFT with out-map: the last FFT runs in-place on the buffer
// before transposing locally into the output
void test_schedule_fft_trans_fft_with_out_map() {
  auto steps = KokkosFFT::Distributed::Impl::build_schedule<DIM>(
      {fft_block(e0, e1), trans_block(e1, e2),
       fft_block(e2, e2, permuted_map)});
  std::vector<BlockStepType> ref = {
      ref_step(FFT, 0, in_spec(BufferId::UserIn, e0),
               out_spec(BufferId::Recv, e1)),
      ref_step(Trans, 0, out_spec(BufferId::Recv, e1),
               out_spec(BufferId::Send, e2)),
      ref_step(FFT, 1, out_spec(BufferId::Send, e2),
               out_spec(BufferId::UserOut, e2), std::nullopt,
               out_spec(BufferId::Send, e2))};
  EXPECT_EQ(steps, ref);
}

// Transpose + FFT + Transpose + FFT + Transpose (Slab/Pencil 2D)
void test_schedule_trans_fft_trans_fft_trans() {
  auto steps = KokkosFFT::Distributed::Impl::build_schedule<DIM>(
      {trans_block(e0, e1), fft_block(e1, e2), trans_block(e2, e3),
       fft_block(e3, e3), trans_block(e3, e4)});
  std::vector<BlockStepType> ref = {
      ref_step(Trans, 0, in_spec(BufferId::UserIn, e0),
               in_spec(BufferId::Send, e1)),
      ref_step(FFT, 0, in_spec(BufferId::Send, e1),
               out_spec(BufferId::Recv, e2)),
      ref_step(Trans, 1, out_spec(BufferId::Recv, e2),
               out_spec(BufferId::Send, e3)),
      ref_step(FFT, 1, out_spec(BufferId::Send, e3),
               out_spec(BufferId::Send, e3)),
      ref_step(Trans, 2, out_spec(BufferId::Send, e3),
               out_spec(BufferId::UserOut, e4))};
  EXPECT_EQ(steps, ref);
}

// FFT + Transpose + Transpose + FFT (Pencil 2D/3D)
void test_schedule_fft_trans_trans_fft() {
  auto steps = KokkosFFT::Distributed::Impl::build_schedule<DIM>(
      {fft_block(e0, e1), trans_block(e1, e2), trans_block(e2, e3),
       fft_block(e3, e3)});
  std::vector<BlockStepType> ref = {
      ref_step(FFT, 0, in_spec(BufferId::UserIn, e0),
               out_spec(BufferId::Recv, e1)),
      ref_step(Trans, 0, out_spec(BufferId::Recv, e1),
               out_spec(BufferId::Send, e2)),
      ref_step(Trans, 1, out_spec(BufferId::Send, e2),
               out_spec(BufferId::UserOut, e3)),
      ref_step(FFT, 1, out_spec(BufferId::UserOut, e3),
               out_spec(BufferId::UserOut, e3))};
  EXPECT_EQ(steps, ref);
}

// Transpose + FFT + Transpose + FFT + Transpose + FFT (Pencil 3D)
void test_schedule_three_ffts() {
  auto steps = KokkosFFT::Distributed::Impl::build_schedule<DIM>(
      {trans_block(e0, e1), fft_block(e1, e2), trans_block(e2, e3),
       fft_block(e3, e3), trans_block(e3, e4), fft_block(e4, e4)});
  std::vector<BlockStepType> ref = {
      ref_step(Trans, 0, in_spec(BufferId::UserIn, e0),
               in_spec(BufferId::Send, e1)),
      ref_step(FFT, 0, in_spec(BufferId::Send, e1),
               out_spec(BufferId::Recv, e2)),
      ref_step(Trans, 1, out_spec(BufferId::Recv, e2),
               out_spec(BufferId::Send, e3)),
      ref_step(FFT, 1, out_spec(BufferId::Send, e3),
               out_spec(BufferId::Send, e3)),
      ref_step(Trans, 2, out_spec(BufferId::Send, e3),
               out_spec(BufferId::UserOut, e4)),
      ref_step(FFT, 2, out_spec(BufferId::UserOut, e4),
               out_spec(BufferId::UserOut, e4))};
  EXPECT_EQ(steps, ref);
}

// Transpose + Transpose + FFT (not produced by the block analyses today, see
// issue_topology.md): both transposes keep the input value type
void test_schedule_trans_trans_fft() {
  auto steps = KokkosFFT::Distributed::Impl::build_schedule<DIM>(
      {trans_block(e0, e1), trans_block(e1, e6), fft_block(e6, e5)});
  std::vector<BlockStepType> ref = {
      ref_step(Trans, 0, in_spec(BufferId::UserIn, e0),
               in_spec(BufferId::Send, e1)),
      ref_step(Trans, 1, in_spec(BufferId::Send, e1),
               in_spec(BufferId::Recv, e6)),
      ref_step(FFT, 0, in_spec(BufferId::Recv, e6),
               out_spec(BufferId::UserOut, e5))};
  EXPECT_EQ(steps, ref);
}

/// \brief Fixture for the comparison against the pencil plans, which needs
/// a perfect square number of MPI processes larger than 1
template <typename T>
struct TestSchedulePencil : public ::testing::Test {
  using value_type  = typename T::first_type;
  using layout_type = typename T::second_type;

  std::size_t m_npx = 1;

  virtual void SetUp() {
    int nprocs;
    ::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    m_npx = static_cast<std::size_t>(std::sqrt(nprocs));
    if (nprocs == 1 || m_npx * m_npx != static_cast<std::size_t>(nprocs)) {
      GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                      "for this test";
    }
  }
};

using pencil_types =
    ::testing::Types<std::pair<double, Kokkos::LayoutLeft>,
                     std::pair<double, Kokkos::LayoutRight>,
                     std::pair<Kokkos::complex<double>, Kokkos::LayoutLeft>,
                     std::pair<Kokkos::complex<double>, Kokkos::LayoutRight>>;

/// \brief Compare the schedule with the buffers selected by a Pencil 2D/3D
/// plan: the m_in_out_ptr chain built by set_block_impl, the views m_in_T and
/// m_out_T, and the conditions under which forward_impl writes directly into
/// the output View.
/// \return true if the plan could be constructed for the given topologies
template <typename ValueType, typename LayoutType, std::size_t FFT_DIM,
          typename InLayoutType, typename OutLayoutType>
bool compare_schedule_with_pencil_plan(
    const KokkosFFT::Distributed::Topology<std::size_t, DIM, InLayoutType>&
        in_topology,
    const KokkosFFT::Distributed::Topology<std::size_t, DIM, OutLayoutType>&
        out_topology,
    const KokkosFFT::axis_type<FFT_DIM>& axes) {
  using float_type = KokkosFFT::Impl::base_floating_point_type<ValueType>;
  using InViewType = Kokkos::View<ValueType***, LayoutType, execution_space>;
  using OutViewType =
      Kokkos::View<Kokkos::complex<float_type>***, LayoutType, execution_space>;
  using PlanType = KokkosFFT::Distributed::Impl::PencilInternalPlan<
      execution_space, InViewType, OutViewType, FFT_DIM, InLayoutType,
      OutLayoutType>;

  constexpr bool is_R2C = KokkosFFT::Impl::is_real_v<ValueType>;
  extents_type gin_extents{8, 7, 5}, gout_extents = gin_extents;
  const auto last_axis       = static_cast<std::size_t>(axes.back());
  gout_extents.at(last_axis) = KokkosFFT::Impl::extent_after_transform(
      gin_extents.at(last_axis), is_R2C);

  auto [in_extents, in_starts] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          gin_extents, in_topology, MPI_COMM_WORLD);
  auto [out_extents, out_starts] =
      KokkosFFT::Distributed::compute_local_extents_and_starts(
          gout_extents, out_topology, MPI_COMM_WORLD);
  InViewType in("in", KokkosFFT::Impl::create_layout<LayoutType>(in_extents));
  OutViewType out("out",
                  KokkosFFT::Impl::create_layout<LayoutType>(out_extents));

  // Invalid combinations are covered by Test_PencilPlan
  std::unique_ptr<PlanType> plan;
  try {
    plan =
        std::make_unique<PlanType>(execution_space(), in, out, axes,
                                   in_topology, out_topology, MPI_COMM_WORLD);
  } catch (const std::runtime_error&) {
    return false;
  }

  const auto& block_infos = plan->m_block_analyses->m_block_infos;
  const auto& chain       = plan->m_in_out_ptr;
  auto steps = KokkosFFT::Distributed::Impl::build_schedule(block_infos);

  EXPECT_EQ(steps.size(), block_infos.size());
  EXPECT_EQ(chain.size(), block_infos.size());
  if (steps.size() != chain.size()) return true;

  auto* send = plan->m_send_buffer_allocation.data();
  auto* recv = plan->m_recv_buffer_allocation.data();
  auto ptr   = [&](BufferId id) { return id == BufferId::Send ? send : recv; };
  auto is_buffer = [](BufferId id) {
    return id == BufferId::Send || id == BufferId::Recv;
  };

  // Conditions under which forward_impl writes into the output View
  const std::size_t last = block_infos.size() - 1;
  const auto identity = KokkosFFT::Impl::index_sequence<std::size_t, DIM, 0>();
  const bool has_out_map = block_infos.back().m_block_type == BlockType::FFT &&
                           block_infos.back().m_out_map != identity;
  const bool is_last_fft = block_infos.back().m_block_type == BlockType::FFT;

  std::size_t nb_fft = 0;
  for (std::size_t i = 0; i < steps.size(); ++i) {
    const auto& step                 = steps.at(i);
    const auto& block                = block_infos.at(i);
    const auto [chain_in, chain_out] = chain.at(i);
    SCOPED_TRACE("block " + std::to_string(i));

    EXPECT_EQ(step.m_type, block.m_block_type);
    EXPECT_EQ(step.m_sub_idx, block.m_block_idx);
    EXPECT_EQ(step.m_src.m_extents, block.m_in_extents);
    EXPECT_EQ(step.m_dst.m_extents, block.m_out_extents);
    EXPECT_EQ(step.m_src.m_is_input_type, nb_fft == 0);
    // The chain does not record the input of the first block and of the first
    // FFT (nullptr), which read the input View or m_in_T (checked below)
    if (is_buffer(step.m_src.m_id) && chain_in != nullptr) {
      EXPECT_EQ(ptr(step.m_src.m_id), chain_in);
    }

    if (block.m_block_type == BlockType::Transpose) {
      EXPECT_EQ(step.m_comm_axis, block.m_comm_axis);
      const bool writes_out =
          i != 0 &&
          (i == last || (i + 1 == last && !has_out_map && is_last_fft));
      EXPECT_EQ(step.m_dst.m_id == BufferId::UserOut, writes_out);
      if (is_buffer(step.m_dst.m_id)) {
        EXPECT_EQ(ptr(step.m_dst.m_id), chain_out);
      }
      if (i == 0) {
        EXPECT_EQ(step.m_src.m_id, BufferId::UserIn);
        EXPECT_EQ(ptr(step.m_dst.m_id),
                  reinterpret_cast<Kokkos::complex<float_type>*>(
                      plan->m_in_T.data()));
      }
    } else {
      EXPECT_EQ(step.m_dst.m_id == BufferId::UserOut, i == last);
      if (nb_fft == 0) {
        // The first FFT reads m_in_T (or the input) and writes m_out_T
        EXPECT_EQ(step.m_src.m_extents,
                  KokkosFFT::Impl::extract_extents(plan->m_in_T));
        EXPECT_EQ(step.m_dst.m_extents,
                  KokkosFFT::Impl::extract_extents(plan->m_out_T));
        if (i > 0) {
          EXPECT_EQ(ptr(step.m_src.m_id),
                    reinterpret_cast<Kokkos::complex<float_type>*>(
                        plan->m_in_T.data()));
        }
        if (step.m_dst.m_id != BufferId::UserOut) {
          EXPECT_EQ(ptr(step.m_dst.m_id), plan->m_out_T.data());
          EXPECT_EQ(ptr(step.m_dst.m_id), chain_out);
        }
        if (step.m_src_T) {
          EXPECT_EQ(ptr(step.m_src_T->m_id),
                    reinterpret_cast<Kokkos::complex<float_type>*>(
                        plan->m_in_T.data()));
        }
        if (step.m_dst_T) {
          EXPECT_EQ(ptr(step.m_dst_T->m_id), plan->m_out_T.data());
        }
      } else {
        // The following FFTs run in-place, on the output View if the last
        // FFT has no out-map
        EXPECT_EQ(step.m_src.m_id == BufferId::UserOut,
                  i == last && !has_out_map);
        EXPECT_FALSE(step.m_src_T.has_value());
        EXPECT_EQ(step.m_dst_T.has_value(), i == last && has_out_map);
        if (step.m_dst_T) {
          EXPECT_EQ(ptr(step.m_dst_T->m_id), chain_in);
        }
      }
      ++nb_fft;
    }
  }
  return true;
}

template <typename ValueType, typename LayoutType, std::size_t FFT_DIM>
void test_schedule_vs_pencil_plan(std::size_t npx, std::size_t npy) {
  using topology_r_type =
      KokkosFFT::Distributed::Topology<std::size_t, DIM, Kokkos::LayoutRight>;
  using topology_l_type =
      KokkosFFT::Distributed::Topology<std::size_t, DIM, Kokkos::LayoutLeft>;

  // x-pencil, y-pencil and z-pencils with both topology layouts
  auto topologies = std::make_tuple(
      topology_r_type{1, npx, npy}, topology_r_type{npx, 1, npy},
      topology_r_type{npx, npy, 1}, topology_l_type{npy, npx, 1});

  std::vector<KokkosFFT::axis_type<FFT_DIM>> all_axes;
  if constexpr (FFT_DIM == 2) {
    all_axes = {{0, 1}, {0, 2}, {1, 0}, {1, 2}, {2, 0}, {2, 1}};
  } else {
    all_axes = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2},
                {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  }

  auto for_each_topology = [&](auto&& f) {
    std::apply([&](const auto&... topology) { (f(topology), ...); },
               topologies);
  };

  std::size_t nb_checked = 0;
  for (const auto& axes : all_axes) {
    for_each_topology([&](const auto& in_topology) {
      for_each_topology([&](const auto& out_topology) {
        nb_checked +=
            compare_schedule_with_pencil_plan<ValueType, LayoutType, FFT_DIM>(
                in_topology, out_topology, axes);
      });
    });
  }
  EXPECT_GT(nb_checked, 0u);
}

}  // namespace

TEST(TestSchedule, FFT) { test_schedule_fft(); }
TEST(TestSchedule, FFTWithMaps) { test_schedule_fft_with_maps(); }
TEST(TestSchedule, TransFFT) { test_schedule_trans_fft(); }
TEST(TestSchedule, FFTTrans) { test_schedule_fft_trans(); }
TEST(TestSchedule, TransFFTTrans) { test_schedule_trans_fft_trans(); }
TEST(TestSchedule, TransFFTTransTrans) {
  test_schedule_trans_fft_trans_trans();
}
TEST(TestSchedule, FFTTransFFT) { test_schedule_fft_trans_fft(); }
TEST(TestSchedule, FFTTransFFTWithOutMap) {
  test_schedule_fft_trans_fft_with_out_map();
}
TEST(TestSchedule, TransFFTTransFFTTrans) {
  test_schedule_trans_fft_trans_fft_trans();
}
TEST(TestSchedule, FFTTransTransFFT) { test_schedule_fft_trans_trans_fft(); }
TEST(TestSchedule, ThreeFFTs) { test_schedule_three_ffts(); }
TEST(TestSchedule, TransTransFFT) { test_schedule_trans_trans_fft(); }

TYPED_TEST_SUITE(TestSchedulePencil, pencil_types);

TYPED_TEST(TestSchedulePencil, Pencil2D) {
  using value_type  = typename TestFixture::value_type;
  using layout_type = typename TestFixture::layout_type;
  test_schedule_vs_pencil_plan<value_type, layout_type, 2>(this->m_npx,
                                                           this->m_npx);
}

TYPED_TEST(TestSchedulePencil, Pencil3D) {
  using value_type  = typename TestFixture::value_type;
  using layout_type = typename TestFixture::layout_type;
  test_schedule_vs_pencil_plan<value_type, layout_type, 3>(this->m_npx,
                                                           this->m_npx);
}
