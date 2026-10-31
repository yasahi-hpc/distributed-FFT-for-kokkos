#include <array>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_BlockAnalyses.hpp"
#include "Test_BlockAnalyses_Utils.hpp"

namespace {
using layout_types =
    ::testing::Types<std::pair<Kokkos::LayoutLeft, Kokkos::LayoutLeft>,
                     std::pair<Kokkos::LayoutLeft, Kokkos::LayoutRight>,
                     std::pair<Kokkos::LayoutRight, Kokkos::LayoutLeft>,
                     std::pair<Kokkos::LayoutRight, Kokkos::LayoutRight>>;

/// \brief Fixture for the propose_*_block helpers.
/// layout_type is the layout of the Input/Output Views, whereas layout_type2 is
/// the layout of the topology, i.e. the mapping from a rank to its Cartesian
/// coordinates.
template <typename T>
struct TestProposeBlock : public ::testing::Test {
  using layout_type  = typename T::first_type;
  using layout_type2 = typename T::second_type;

  bool m_is_layout_right = std::is_same_v<layout_type2, Kokkos::LayoutRight>;
  std::vector<std::size_t> m_axes       = {2};
  std::vector<std::size_t> m_empty_axes = {};
};

void test_count_blocks() {
  using BlockInfoType = KokkosFFT::Distributed::Impl::BlockInfo<3>;
  using KokkosFFT::Distributed::Impl::BlockType;
  std::vector<BlockInfoType> block_infos;

  // An empty plan does not include any block
  EXPECT_EQ(
      KokkosFFT::Distributed::Impl::count_blocks(block_infos, BlockType::FFT),
      0);
  EXPECT_EQ(KokkosFFT::Distributed::Impl::count_blocks(block_infos,
                                                       BlockType::Transpose),
            0);

  BlockInfoType block1;
  block1.m_block_type = BlockType::FFT;
  block_infos.push_back(block1);

  BlockInfoType block2;
  block2.m_block_type = BlockType::Transpose;
  block_infos.push_back(block2);

  BlockInfoType block3;
  block3.m_block_type = BlockType::FFT;
  block_infos.push_back(block3);

  auto fft_count =
      KokkosFFT::Distributed::Impl::count_blocks(block_infos, BlockType::FFT);
  auto transpose_count = KokkosFFT::Distributed::Impl::count_blocks(
      block_infos, BlockType::Transpose);

  EXPECT_EQ(fft_count, 2);
  EXPECT_EQ(transpose_count, 1);

  // Adding a transpose block must not change the number of FFT blocks
  BlockInfoType block4;
  block4.m_block_type = BlockType::Transpose;
  block_infos.push_back(block4);

  EXPECT_EQ(
      KokkosFFT::Distributed::Impl::count_blocks(block_infos, BlockType::FFT),
      2);
  EXPECT_EQ(KokkosFFT::Distributed::Impl::count_blocks(block_infos,
                                                       BlockType::Transpose),
            2);
}

void test_count_fft_dims() {
  using BlockInfoType = KokkosFFT::Distributed::Impl::BlockInfo<3>;
  using KokkosFFT::Distributed::Impl::BlockType;
  std::vector<BlockInfoType> block_infos;

  // An empty plan does not perform any FFT
  EXPECT_EQ(KokkosFFT::Distributed::Impl::count_fft_dims(block_infos), 0);

  BlockInfoType block1;
  block1.m_block_type = BlockType::FFT;
  block1.m_fft_dim    = 2;
  block_infos.push_back(block1);

  auto total_fft_dims =
      KokkosFFT::Distributed::Impl::count_fft_dims(block_infos);
  EXPECT_EQ(total_fft_dims, 2);

  // A transpose block does not contribute to the number of FFT dimensions,
  // even if m_fft_dim is set
  BlockInfoType block2;
  block2.m_block_type = BlockType::Transpose;
  block2.m_fft_dim    = 3;
  block_infos.push_back(block2);

  EXPECT_EQ(KokkosFFT::Distributed::Impl::count_fft_dims(block_infos), 2);

  BlockInfoType block3;
  block3.m_block_type = BlockType::FFT;
  block3.m_fft_dim    = 1;
  block_infos.push_back(block3);

  total_fft_dims = KokkosFFT::Distributed::Impl::count_fft_dims(block_infos);
  EXPECT_EQ(total_fft_dims, 3);
}

/// \brief Test propose_fft_block for a C2C and an R2C like transform.
/// The map is not the identity to make sure that it is stored as it is for
/// both the input and the output.
void test_propose_fft_block() {
  [[maybe_unused]] constexpr std::size_t DIM = 3;
  using BlockInfoType = KokkosFFT::Distributed::Impl::BlockInfo<DIM>;
  using extents_type  = std::array<std::size_t, DIM>;

  extents_type map{2, 0, 1};
  extents_type in_extents{16, 16, 16}, out_extents{9, 16, 16};
  std::size_t fft_dim = 3, block_idx = 2;

  // R2C like transform (in and out extents differ)
  auto [block, max_buffer_size] =
      KokkosFFT::Distributed::Impl::propose_fft_block<BlockInfoType>(
          map, in_extents, out_extents, fft_dim, block_idx);

  auto ref_block =
      make_fft_block(in_extents, out_extents, map, fft_dim, block_idx);
  EXPECT_EQ(block, ref_block);

  // The buffer must be large enough to store the complex output
  EXPECT_EQ(max_buffer_size, std::size_t(9 * 16 * 16 * 2));

  // C2C like transform (in and out extents are identical)
  auto [block2, max_buffer_size2] =
      KokkosFFT::Distributed::Impl::propose_fft_block<BlockInfoType>(
          map, in_extents, in_extents, fft_dim, 0);

  auto ref_block2 = make_fft_block(in_extents, in_extents, map, fft_dim, 0);
  EXPECT_EQ(block2, ref_block2);
  EXPECT_EQ(max_buffer_size2, std::size_t(16 * 16 * 16 * 2));
}

/// \brief Test propose_transpose_block against hard-coded reference values.
///
/// Configuration (3D View distributed over 16 processes):
///   global extents: (66, 64, 18)
///   in topology:    (4, 4, 1), out topology: (4, 1, 4)
///   in map:         (2, 0, 1),  in axis: 2, out axis: 1, comm axis: 1
///
/// The global extents are deliberately not divisible by the topology, so that
/// the reference local extents depend on the Cartesian coordinates of the
/// current rank, and thus on the layout of the topology (is_layout_right).
/// With the out topology (4, 1, 4) and a rank smaller than 4, the coordinates
/// are (0, 0, rank) for a LayoutRight topology and (rank, 0, 0) for a
/// LayoutLeft one.
///
/// \tparam LayoutType Layout of the Input/Output Views
/// \param[in] rank Rank of the process (must be smaller than 4)
/// \param[in] axes Axes of the FFT performed after this transpose
/// \param[in] is_layout_right Whether the topology is LayoutRight
/// \param[in] is_last Whether this is the last block of the plan
template <typename LayoutType>
void test_propose_transpose_block(std::size_t rank,
                                  std::vector<std::size_t> axes,
                                  bool is_layout_right, bool is_last) {
  [[maybe_unused]] constexpr std::size_t DIM = 3;
  using BlockInfoType       = KokkosFFT::Distributed::Impl::BlockInfo<DIM>;
  using extents_type        = std::array<std::size_t, DIM>;
  using buffer_extents_type = std::array<std::size_t, DIM + 1>;
  constexpr bool is_left    = std::is_same_v<LayoutType, Kokkos::LayoutLeft>;

  extents_type map{2, 0, 1};
  extents_type in_extents{17, 16, 18}, global_extents{66, 64, 18};
  extents_type in_topology{4, 4, 1}, out_topology{4, 1, 4};
  std::size_t in_axis = 2, out_axis = 1, comm_axis = 1, block_idx = 3,
              size_factor = 2;

  auto [block, max_buffer_size] =
      KokkosFFT::Distributed::Impl::propose_transpose_block<BlockInfoType,
                                                            LayoutType>(
          map, in_topology, out_topology, in_extents, global_extents, axes,
          rank, is_layout_right, is_last, in_axis, out_axis, comm_axis,
          block_idx, size_factor);

  // The map of the output View is:
  // - the identity if this is the last block
  // - the map bringing the FFT axes innermost, if the FFT axes are given
  // - the map bringing the out_axis innermost, if the FFT axes are empty
  extents_type ref_out_map{0, 1, 2};
  if (!is_last) {
    if (axes.empty()) {
      // permutation of (2, 0, 1) by the out_axis 1
      ref_out_map = is_left ? extents_type{1, 2, 0} : extents_type{2, 0, 1};
    } else {
      // permutation of (2, 0, 1) by the axes (2)
      ref_out_map = is_left ? extents_type{2, 0, 1} : extents_type{0, 1, 2};
    }
  }

  // Local extents of the global extents (66, 64, 18) over the out topology
  // (4, 1, 4). 66 = 4 * 16 + 2 and 18 = 4 * 4 + 2, so the first two
  // coordinates own one more element than the others.
  std::size_t coord0 = is_layout_right ? 0 : rank;
  std::size_t coord2 = is_layout_right ? rank : 0;
  extents_type ref_local_extents{coord0 < 2 ? 17u : 16u, 64u,
                                 coord2 < 2 ? 5u : 4u};
  auto ref_out_extents =
      KokkosFFT::Impl::compute_mapped_extents(ref_local_extents, ref_out_map);

  // The buffer is shaped from the merged topology (4, 4, 4) and the number of
  // processes involved in the all2all (4)
  buffer_extents_type ref_buffer_extents =
      is_left ? buffer_extents_type{17, 16, 5, 4}
              : buffer_extents_type{4, 17, 16, 5};

  auto ref_block = make_transpose_block(in_topology, out_topology, in_extents,
                                        ref_out_extents, ref_buffer_extents,
                                        map, ref_out_map, in_axis, out_axis,
                                        comm_axis, block_idx);
  EXPECT_EQ(block, ref_block);

  // in: 17 * 16 * 18 = 4896, buffer: 4 * 17 * 16 * 5 = 5440 and
  // out: at most 17 * 64 * 5 = 5440. The buffer is always the largest one.
  EXPECT_EQ(max_buffer_size, std::size_t(5440) * size_factor);
}

/// \brief Run test_propose_transpose_block for every rank of the out topology
/// along the comm axis (0, 1, 2 and 3). propose_transpose_block takes the rank
/// as a value, so a single process covers all the ranks.
///
/// \tparam LayoutType Layout of the Input/Output Views
/// \param[in] axes Axes of the FFT performed after this transpose
/// \param[in] is_layout_right Whether the topology is LayoutRight
/// \param[in] is_last Whether this is the last block of the plan
template <typename LayoutType>
void test_propose_transpose_block_all_ranks(std::vector<std::size_t> axes,
                                            bool is_layout_right,
                                            bool is_last) {
  for (std::size_t rank = 0; rank < 4; ++rank) {
    SCOPED_TRACE("rank " + std::to_string(rank));
    test_propose_transpose_block<LayoutType>(rank, axes, is_layout_right,
                                             is_last);
  }
}

}  // namespace

TEST(TestBlockAnalysesHelpers, count_blocks) { test_count_blocks(); }

TEST(TestBlockAnalysesHelpers, count_fft_dims) { test_count_fft_dims(); }

TEST(TestBlockAnalysesHelpers, propose_fft_block) { test_propose_fft_block(); }

TYPED_TEST_SUITE(TestProposeBlock, layout_types);

TYPED_TEST(TestProposeBlock, propose_transpose_block_last) {
  using layout_type = typename TestFixture::layout_type;
  test_propose_transpose_block_all_ranks<layout_type>(
      this->m_axes, this->m_is_layout_right, true);
}

TYPED_TEST(TestProposeBlock, propose_transpose_block_not_last) {
  using layout_type = typename TestFixture::layout_type;
  test_propose_transpose_block_all_ranks<layout_type>(
      this->m_axes, this->m_is_layout_right, false);
}

TYPED_TEST(TestProposeBlock, propose_transpose_block_empty_axes_last) {
  using layout_type = typename TestFixture::layout_type;
  test_propose_transpose_block_all_ranks<layout_type>(
      this->m_empty_axes, this->m_is_layout_right, true);
}

TYPED_TEST(TestProposeBlock, propose_transpose_block_empty_axes_not_last) {
  using layout_type = typename TestFixture::layout_type;
  test_propose_transpose_block_all_ranks<layout_type>(
      this->m_empty_axes, this->m_is_layout_right, false);
}
