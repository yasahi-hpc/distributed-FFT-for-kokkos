#include <mpi.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_BlockAnalyses.hpp"

namespace {
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

}  // namespace

TEST(TEST_BLOCK_ANALYSES_HELPERS, count_blocks) { test_count_blocks(); }

TEST(TEST_BLOCK_ANALYSES_HELPERS, count_fft_dims) { test_count_fft_dims(); }

TEST(TEST_BLOCK_ANALYSES_HELPERS, propose_fft_block) {
  test_propose_fft_block();
}
