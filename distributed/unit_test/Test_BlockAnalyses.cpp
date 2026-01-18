#include <mpi.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_BlockAnalyses.hpp"

namespace {
using layout_types = ::testing::Types<Kokkos::LayoutLeft, Kokkos::LayoutRight>;

template <typename T>
struct TestProposeBlock : public ::testing::Test {
  using layout_type = T;

  std::size_t m_rank                    = 0;
  std::vector<std::size_t> m_axes       = {2};
  std::vector<std::size_t> m_empty_axes = {};

  virtual void SetUp() {
    int rank, nprocs;
    ::MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    m_rank = rank;
  }
};

void test_count_blocks() {
  using BlockInfoType = KokkosFFT::Distributed::Impl::BlockInfo<3>;
  std::vector<BlockInfoType> block_infos;

  BlockInfoType block1;
  block1.m_block_type = KokkosFFT::Distributed::Impl::BlockType::FFT;
  block_infos.push_back(block1);

  BlockInfoType block2;
  block2.m_block_type = KokkosFFT::Distributed::Impl::BlockType::Transpose;
  block_infos.push_back(block2);

  BlockInfoType block3;
  block3.m_block_type = KokkosFFT::Distributed::Impl::BlockType::FFT;
  block_infos.push_back(block3);

  auto fft_count = KokkosFFT::Distributed::Impl::count_blocks(
      block_infos, KokkosFFT::Distributed::Impl::BlockType::FFT);
  auto transpose_count = KokkosFFT::Distributed::Impl::count_blocks(
      block_infos, KokkosFFT::Distributed::Impl::BlockType::Transpose);

  EXPECT_EQ(fft_count, 2);
  EXPECT_EQ(transpose_count, 1);
}

void test_count_fft_dims() {
  using BlockInfoType = KokkosFFT::Distributed::Impl::BlockInfo<3>;
  std::vector<BlockInfoType> block_infos;

  BlockInfoType block1;
  block1.m_block_type = KokkosFFT::Distributed::Impl::BlockType::FFT;
  block1.m_fft_dim    = 2;
  block_infos.push_back(block1);

  auto total_fft_dims =
      KokkosFFT::Distributed::Impl::count_fft_dims(block_infos);
  EXPECT_EQ(total_fft_dims, 2);

  BlockInfoType block2;
  block2.m_block_type = KokkosFFT::Distributed::Impl::BlockType::Transpose;
  block_infos.push_back(block2);

  BlockInfoType block3;
  block3.m_block_type = KokkosFFT::Distributed::Impl::BlockType::FFT;
  block3.m_fft_dim    = 1;
  block_infos.push_back(block3);

  total_fft_dims = KokkosFFT::Distributed::Impl::count_fft_dims(block_infos);
  EXPECT_EQ(total_fft_dims, 3);
}

void test_propose_fft_block() {
  using BlockInfoType            = KokkosFFT::Distributed::Impl::BlockInfo<3>;
  std::array<std::size_t, 3> map = {0, 1, 2};
  std::array<std::size_t, 3> in_extents  = {16, 16, 16};
  std::array<std::size_t, 3> out_extents = {9, 16, 16};
  std::size_t fft_dim                    = 3;
  std::size_t block_idx                  = 0;

  auto [block, max_buffer_size] =
      KokkosFFT::Distributed::Impl::propose_fft_block<BlockInfoType>(
          map, in_extents, out_extents, fft_dim, block_idx);
  EXPECT_EQ(block.m_block_type, KokkosFFT::Distributed::Impl::BlockType::FFT);
  EXPECT_EQ(block.m_fft_dim, fft_dim);
  EXPECT_EQ(block.m_in_extents, in_extents);
  EXPECT_EQ(block.m_out_extents, out_extents);
  EXPECT_EQ(block.m_in_map, map);
  EXPECT_EQ(block.m_out_map, map);
  EXPECT_EQ(block.m_block_idx, block_idx);

  std::size_t expected_max_buffer_size =
      KokkosFFT::Impl::total_size(out_extents) * 2;
  EXPECT_EQ(max_buffer_size, expected_max_buffer_size);
}

template <typename LayoutType, std::size_t DIM = 3>
void test_propose_transpose_block(std::size_t rank,
                                  std::vector<std::size_t> axes,
                                  bool is_layout_right, bool is_last) {
  using BlockInfoType = KokkosFFT::Distributed::Impl::BlockInfo<DIM>;
  std::array<std::size_t, DIM> map            = {2, 0, 1};
  std::array<std::size_t, DIM> in_extents     = {16, 16, 16};
  std::array<std::size_t, DIM> in_topology    = {4, 4, 1};
  std::array<std::size_t, DIM> out_topology   = {4, 1, 4};
  std::array<std::size_t, DIM> global_extents = {64, 64, 16};
  std::size_t in_axis                         = 1;
  std::size_t out_axis                        = 2;
  std::size_t comm_axis                       = 1;
  std::size_t block_idx                       = 3;
  std::size_t size_factor                     = 2;

  auto [block, max_buffer_size] =
      KokkosFFT::Distributed::Impl::propose_transpose_block<BlockInfoType,
                                                            LayoutType>(
          map, in_topology, out_topology, in_extents, global_extents, axes,
          rank, is_layout_right, is_last, in_axis, out_axis, comm_axis,
          block_idx, size_factor);

  EXPECT_EQ(block.m_block_type,
            KokkosFFT::Distributed::Impl::BlockType::Transpose);
  EXPECT_EQ(block.m_in_map, map);
  EXPECT_EQ(block.m_in_axis, in_axis);
  EXPECT_EQ(block.m_out_axis, out_axis);
  EXPECT_EQ(block.m_comm_axis, comm_axis);
  EXPECT_EQ(block.m_in_topology, in_topology);
  EXPECT_EQ(block.m_out_topology, out_topology);
  EXPECT_EQ(block.m_in_extents, in_extents);
  EXPECT_EQ(block.m_block_idx, block_idx);

  // Check out_map
  if (is_last) {
    auto ref_out_map = KokkosFFT::Impl::index_sequence<std::size_t, DIM, 0>();
    EXPECT_EQ(block.m_out_map, ref_out_map);
  } else {
    auto ref_out_map =
        axes.empty()
            ? KokkosFFT::Distributed::Impl::get_dst_map<LayoutType>(map,
                                                                    out_axis)
            : KokkosFFT::Distributed::Impl::get_dst_map<LayoutType>(map, axes);
    EXPECT_EQ(block.m_out_map, ref_out_map);
  }

  // Compute expected out_extents
  auto ref_out_extents = KokkosFFT::Distributed::Impl::compute_next_extents(
      global_extents, out_topology, block.m_out_map, rank, is_layout_right);
  EXPECT_EQ(block.m_out_extents, ref_out_extents);

  // Compute expected buffer_extents
  auto ref_buffer_extents =
      KokkosFFT::Distributed::Impl::compute_buffer_extents<LayoutType>(
          global_extents, in_topology, out_topology);
  EXPECT_EQ(block.m_buffer_extents, ref_buffer_extents);

  // Compute expected max_buffer_size
  std::vector<std::size_t> all_max_buffer_sizes = {
      KokkosFFT::Impl::total_size(block.m_in_extents),
      KokkosFFT::Impl::total_size(block.m_buffer_extents),
      KokkosFFT::Impl::total_size(block.m_out_extents)};
  std::size_t ref_max_buffer_size =
      *std::max_element(all_max_buffer_sizes.begin(),
                        all_max_buffer_sizes.end()) *
      size_factor;
  EXPECT_EQ(max_buffer_size, ref_max_buffer_size);
}

}  // namespace

TEST(TestBlockAnalysesHelpers, count_blocks) { test_count_blocks(); }

TEST(TestBlockAnalysesHelpers, count_fft_dims) { test_count_fft_dims(); }

TEST(TestBlockAnalysesHelpers, propose_fft_block) { test_propose_fft_block(); }

TYPED_TEST_SUITE(TestProposeBlock, layout_types);

TYPED_TEST(TestProposeBlock, propose_transpose_block_right_last) {
  using layout_type = typename TestFixture::layout_type;
  test_propose_transpose_block<layout_type>(this->m_rank, this->m_axes, true,
                                            true);
}

TYPED_TEST(TestProposeBlock, propose_transpose_block_right_not_last) {
  using layout_type = typename TestFixture::layout_type;
  test_propose_transpose_block<layout_type>(this->m_rank, this->m_axes, true,
                                            false);
}

TYPED_TEST(TestProposeBlock, propose_transpose_block_left_last) {
  using layout_type = typename TestFixture::layout_type;
  test_propose_transpose_block<layout_type>(this->m_rank, this->m_axes, false,
                                            true);
}

TYPED_TEST(TestProposeBlock, propose_transpose_block_left_not_last) {
  using layout_type = typename TestFixture::layout_type;
  test_propose_transpose_block<layout_type>(this->m_rank, this->m_axes, false,
                                            false);
}

TYPED_TEST(TestProposeBlock, propose_transpose_empty_axes_block_right_last) {
  using layout_type = typename TestFixture::layout_type;
  test_propose_transpose_block<layout_type>(this->m_rank, this->m_empty_axes,
                                            true, true);
}

TYPED_TEST(TestProposeBlock,
           propose_transpose_empty_axes_block_right_not_last) {
  using layout_type = typename TestFixture::layout_type;
  test_propose_transpose_block<layout_type>(this->m_rank, this->m_empty_axes,
                                            true, false);
}

TYPED_TEST(TestProposeBlock, propose_transpose_empty_axes_block_left_last) {
  using layout_type = typename TestFixture::layout_type;
  test_propose_transpose_block<layout_type>(this->m_rank, this->m_empty_axes,
                                            false, true);
}

TYPED_TEST(TestProposeBlock, propose_transpose_empty_axes_block_left_not_last) {
  using layout_type = typename TestFixture::layout_type;
  test_propose_transpose_block<layout_type>(this->m_rank, this->m_empty_axes,
                                            false, false);
}
