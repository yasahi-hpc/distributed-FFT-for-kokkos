#include <array>
#include <mpi.h>
#include <optional>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_Schedule.hpp"
#include "KokkosFFT_Distributed_BlockExecutor.hpp"
#include "KokkosFFT_Distributed_DistributedInternalPlan.hpp"

namespace {
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

// A plan without any block or without any FFT block is invalid
void test_schedule_invalid_blocks() {
  EXPECT_THROW(
      {
        [[maybe_unused]] auto steps =
            KokkosFFT::Distributed::Impl::build_schedule<DIM>({});
      },
      std::runtime_error);

  EXPECT_THROW(
      {
        [[maybe_unused]] auto steps =
            KokkosFFT::Distributed::Impl::build_schedule<DIM>(
                {trans_block(e0, e1), trans_block(e1, e2)});
      },
      std::runtime_error);
}

// A plan without communication (SharedComms) must never execute a transpose:
// BlockExecutor throws instead of compiling the transpose path for it
void test_transpose_without_communication() {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using ViewType = Kokkos::View<Kokkos::complex<double>***, Kokkos::LayoutRight,
                                execution_space>;
  using ExecutorType =
      KokkosFFT::Distributed::Impl::BlockExecutor<execution_space, ViewType,
                                                  ViewType>;
  using CommsType = KokkosFFT::Distributed::Impl::SharedComms<execution_space>;

  BlockInfoType transpose    = trans_block(e0, e0);
  transpose.m_buffer_extents = {1, 8, 6, 4};
  transpose.m_in_map         = identity_map;
  transpose.m_out_map        = identity_map;
  BlockInfoType fft          = fft_block(e0, e0);
  fft.m_fft_dim              = 1;

  execution_space exec;
  const std::size_t max_buffer_size = 8 * 6 * 4 * 2;
  ExecutorType executor(exec, {transpose, fft}, max_buffer_size);
  CommsType comms(exec, extents_type{1, 1, 1}, MPI_COMM_WORLD);

  ViewType in("in", 8, 6, 4), out("out", 8, 6, 4);
  EXPECT_THROW({ executor.forward(comms, in, out); }, std::runtime_error);
  EXPECT_THROW({ executor.backward(comms, out, in); }, std::runtime_error);
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
TEST(TestSchedule, InvalidBlocks) { test_schedule_invalid_blocks(); }

TEST(TestBlockExecutor, TransposeWithoutCommunication) {
  test_transpose_without_communication();
}
