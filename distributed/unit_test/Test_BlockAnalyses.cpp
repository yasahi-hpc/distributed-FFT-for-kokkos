#include <algorithm>
#include <array>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include <mpi.h>
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_BlockAnalyses.hpp"

namespace {
using layout_types =
    ::testing::Types<std::pair<Kokkos::LayoutLeft, Kokkos::LayoutLeft>,
                     std::pair<Kokkos::LayoutLeft, Kokkos::LayoutRight>,
                     std::pair<Kokkos::LayoutRight, Kokkos::LayoutLeft>,
                     std::pair<Kokkos::LayoutRight, Kokkos::LayoutRight>>;

/// \brief Fixture for the propose_*_block helpers.
/// layout_type is the layout of the Input/Output Views, whereas layout_type2 is
/// the layout of the topology, i.e. the mapping from an MPI rank to its
/// Cartesian coordinates.
template <typename T>
struct TestProposeBlock : public ::testing::Test {
  using layout_type  = typename T::first_type;
  using layout_type2 = typename T::second_type;

  std::size_t m_rank     = 0;
  bool m_is_layout_right = std::is_same_v<layout_type2, Kokkos::LayoutRight>;
  std::vector<std::size_t> m_axes       = {2};
  std::vector<std::size_t> m_empty_axes = {};

  virtual void SetUp() {
    int rank, nprocs;
    ::MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    m_rank = static_cast<std::size_t>(rank);

    if (nprocs > 4) {
      GTEST_SKIP() << "The number of MPI processes should be smaller or equal "
                      "to 4 for this test";
    }
  }
};

/// \brief Fixture for SlabBlockAnalysesInternal and
/// PencilBlockAnalysesInternal. It is parameterized in the same way as
/// TestProposeBlock: layout_type is the layout of the Input/Output Views and
/// layout_type2 is the layout of the pencil topologies. The slab analyses do
/// not take the layout of the topology, hence they only depend on layout_type.
template <typename T>
struct TestBlockAnalyses : public ::testing::Test {
  using layout_type  = typename T::first_type;
  using layout_type2 = typename T::second_type;

  virtual void SetUp() {
    int nprocs;
    ::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // The block analyses only depend on the topologies given, which are fixed
    // in these tests. The MPI communicator is only used to get the rank of the
    // current process, which must be within the topology.
    if (nprocs > 4) {
      GTEST_SKIP() << "The number of MPI processes should be smaller or equal "
                      "to 4 for this test";
    }
  }
};

/// \brief Check that a map is a permutation of (0, 1, ..., DIM-1)
template <std::size_t DIM>
bool is_permutation(const std::array<std::size_t, DIM>& map) {
  std::array<std::size_t, DIM> identity{};
  std::iota(identity.begin(), identity.end(), std::size_t(0));
  return std::is_permutation(map.begin(), map.end(), identity.begin());
}

/// \brief Make a reference FFT block.
/// The topologies, the buffer extents and the axes of a transpose are unused
/// for an FFT block and thus kept default initialized.
template <std::size_t DIM>
auto make_fft_block(const std::array<std::size_t, DIM>& in_extents,
                    const std::array<std::size_t, DIM>& out_extents,
                    const std::array<std::size_t, DIM>& map,
                    std::size_t fft_dim, std::size_t block_idx) {
  KokkosFFT::Distributed::Impl::BlockInfo<DIM> block;
  block.m_block_type  = KokkosFFT::Distributed::Impl::BlockType::FFT;
  block.m_in_extents  = in_extents;
  block.m_out_extents = out_extents;
  block.m_in_map      = map;
  block.m_out_map     = map;
  block.m_fft_dim     = fft_dim;
  block.m_block_idx   = block_idx;
  return block;
}

/// \brief Make a reference Transpose block.
/// The FFT dimension is unused for a transpose block and thus kept default
/// initialized.
template <std::size_t DIM>
auto make_transpose_block(
    const std::array<std::size_t, DIM>& in_topology,
    const std::array<std::size_t, DIM>& out_topology,
    const std::array<std::size_t, DIM>& in_extents,
    const std::array<std::size_t, DIM>& out_extents,
    const std::array<std::size_t, DIM + 1>& buffer_extents,
    const std::array<std::size_t, DIM>& in_map,
    const std::array<std::size_t, DIM>& out_map, std::size_t in_axis,
    std::size_t out_axis, std::size_t comm_axis, std::size_t block_idx) {
  KokkosFFT::Distributed::Impl::BlockInfo<DIM> block;
  block.m_block_type     = KokkosFFT::Distributed::Impl::BlockType::Transpose;
  block.m_in_topology    = in_topology;
  block.m_out_topology   = out_topology;
  block.m_in_extents     = in_extents;
  block.m_out_extents    = out_extents;
  block.m_buffer_extents = buffer_extents;
  block.m_in_map         = in_map;
  block.m_out_map        = out_map;
  block.m_in_axis        = in_axis;
  block.m_out_axis       = out_axis;
  block.m_comm_axis      = comm_axis;
  block.m_block_idx      = block_idx;
  return block;
}

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
  constexpr std::size_t DIM = 3;
  using BlockInfoType       = KokkosFFT::Distributed::Impl::BlockInfo<DIM>;
  using extents_type        = std::array<std::size_t, DIM>;

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
/// \param[in] rank MPI rank (must be smaller than 4)
/// \param[in] axes Axes of the FFT performed after this transpose
/// \param[in] is_layout_right Whether the topology is LayoutRight
/// \param[in] is_last Whether this is the last block of the plan
template <typename LayoutType>
void test_propose_transpose_block(std::size_t rank,
                                  std::vector<std::size_t> axes,
                                  bool is_layout_right, bool is_last) {
  constexpr std::size_t DIM = 3;
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
  extents_type ref_local_extents{coord0 < 2 ? 17 : 16, 64, coord2 < 2 ? 5 : 4};
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

/// \brief Test SlabBlockAnalysesInternal for a 1D FFT over the axis 2 of a 3D
/// View, which is distributed over 4 processes.
///
/// in topology (1, 1, 4) -> out topology (4, 1, 1), axes (2)
/// The FFT axis is distributed on the input topology, so a transpose is needed
/// before the FFT can be performed:
/// Transpose (1, 1, 4) -> (4, 1, 1) -> FFT over the axis 2
///
/// \tparam ValueType Value type of the input data (real for R2C)
/// \tparam LayoutType Layout of the Input/Output Views
template <typename ValueType, typename LayoutType>
void test_slab_analyses_1D() {
  constexpr std::size_t DIM = 3, FFT_DIM = 1;
  using extents_type        = std::array<std::size_t, DIM>;
  using buffer_extents_type = std::array<std::size_t, DIM + 1>;
  using axes_type           = std::array<std::size_t, FFT_DIM>;
  using AnalysesType = KokkosFFT::Distributed::Impl::SlabBlockAnalysesInternal<
      ValueType, LayoutType, std::size_t, DIM, FFT_DIM>;

  constexpr bool is_left = std::is_same_v<LayoutType, Kokkos::LayoutLeft>;
  constexpr bool is_R2C  = KokkosFFT::Impl::is_real_v<ValueType>;

  axes_type axes{2};
  extents_type gin_extents{8, 12, 16};
  extents_type gout_extents{8, 12, is_R2C ? 9 : 16};
  extents_type in_topology{1, 1, 4}, out_topology{4, 1, 1};

  // Local extents of the input over the in topology
  extents_type in_extents{8, 12, 4};

  AnalysesType analyses(in_extents, gin_extents, gout_extents, in_topology,
                        out_topology, axes, MPI_COMM_WORLD);

  // Transpose block: the FFT over the axis 2 follows, so the axis 2 is brought
  // innermost
  extents_type in_map{0, 1, 2};
  extents_type out_map =
      is_left ? extents_type{2, 0, 1} : extents_type{0, 1, 2};
  auto trans_out_extents =
      KokkosFFT::Impl::compute_mapped_extents(extents_type{2, 12, 16}, out_map);
  buffer_extents_type buffer_extents = is_left
                                           ? buffer_extents_type{2, 12, 4, 4}
                                           : buffer_extents_type{4, 2, 12, 4};

  // The transpose is performed on the real input data for R2C
  std::size_t trans_size_factor = is_R2C ? 1 : 2;

  // FFT block: the input is the output of the transpose and the output is
  // shrunk along the FFT axis for R2C
  auto fft_out_extents = KokkosFFT::Impl::compute_mapped_extents(
      extents_type{2, 12, is_R2C ? 9 : 16}, out_map);

  std::vector<KokkosFFT::Distributed::Impl::BlockInfo<DIM>> ref_block_infos = {
      make_transpose_block(in_topology, out_topology, in_extents,
                           trans_out_extents, buffer_extents, in_map, out_map,
                           /*in_axis=*/0, /*out_axis=*/2, /*comm_axis=*/0,
                           /*block_idx=*/0),
      make_fft_block(trans_out_extents, fft_out_extents, out_map,
                     /*fft_dim=*/1, /*block_idx=*/0)};

  EXPECT_EQ(analyses.m_block_infos, ref_block_infos);

  // All the extents of the transpose block have 8 * 12 * 16 / 4 = 384 elements
  // and the FFT block needs 2 * 12 * 16 (or 2 * 12 * 9 for R2C) complex numbers
  std::size_t ref_fft_buffer_size =
      (is_R2C ? std::size_t(216) : std::size_t(384)) * 2;
  std::size_t ref_max_buffer_size =
      std::max(std::size_t(384) * trans_size_factor, ref_fft_buffer_size);
  EXPECT_EQ(analyses.m_max_buffer_size, ref_max_buffer_size);
}

/// \brief Test SlabBlockAnalysesInternal for a 3D FFT on a 3D View, which is
/// distributed over 4 processes.
///
/// in topology (1, 1, 4) -> out topology (1, 1, 4), axes (0, 1, 2)
/// Only the axes 0 and 1 can be transformed on the input topology, so the plan
/// is:
/// Transpose (1, 1, 4) -> (4, 1, 1) -> FFT over the axes (1, 2)
/// -> Transpose (4, 1, 1) -> (1, 1, 4) -> FFT over the axis 0
///
/// \tparam LayoutType Layout of the Input/Output Views
template <typename LayoutType>
void test_slab_analyses_3D() {
  constexpr std::size_t DIM = 3, FFT_DIM = 3;
  using value_type          = Kokkos::complex<double>;
  using extents_type        = std::array<std::size_t, DIM>;
  using buffer_extents_type = std::array<std::size_t, DIM + 1>;
  using axes_type           = std::array<std::size_t, FFT_DIM>;
  using AnalysesType = KokkosFFT::Distributed::Impl::SlabBlockAnalysesInternal<
      value_type, LayoutType, std::size_t, DIM, FFT_DIM>;

  constexpr bool is_left = std::is_same_v<LayoutType, Kokkos::LayoutLeft>;

  axes_type axes{0, 1, 2};
  extents_type gin_extents{8, 12, 16}, gout_extents{8, 12, 16};
  extents_type in_topology{1, 1, 4}, out_topology{1, 1, 4};
  extents_type mid_topology{4, 1, 1};
  extents_type in_extents{8, 12, 4};

  AnalysesType analyses(in_extents, gin_extents, gout_extents, in_topology,
                        out_topology, axes, MPI_COMM_WORLD);

  // First transpose: the FFT over the axes (1, 2) follows
  extents_type map0{0, 1, 2};
  extents_type map1 = is_left ? extents_type{2, 1, 0} : extents_type{0, 1, 2};
  auto extents1 =
      KokkosFFT::Impl::compute_mapped_extents(extents_type{2, 12, 16}, map1);
  buffer_extents_type buffer0 = is_left ? buffer_extents_type{2, 12, 4, 4}
                                        : buffer_extents_type{4, 2, 12, 4};

  // Second transpose: the FFT over the axis 0 follows
  extents_type map2 = is_left ? extents_type{0, 2, 1} : extents_type{1, 2, 0};
  auto extents2 =
      KokkosFFT::Impl::compute_mapped_extents(extents_type{8, 12, 4}, map2);
  buffer_extents_type buffer1 = is_left ? buffer_extents_type{2, 12, 4, 4}
                                        : buffer_extents_type{4, 2, 12, 4};

  std::vector<KokkosFFT::Distributed::Impl::BlockInfo<DIM>> ref_block_infos = {
      make_transpose_block(in_topology, mid_topology, in_extents, extents1,
                           buffer0, map0, map1, /*in_axis=*/0, /*out_axis=*/2,
                           /*comm_axis=*/0, /*block_idx=*/0),
      make_fft_block(extents1, extents1, map1, /*fft_dim=*/2, /*block_idx=*/0),
      make_transpose_block(mid_topology, out_topology, extents1, extents2,
                           buffer1, map1, map2, /*in_axis=*/2, /*out_axis=*/0,
                           /*comm_axis=*/0, /*block_idx=*/1),
      make_fft_block(extents2, extents2, map2, /*fft_dim=*/1, /*block_idx=*/1)};

  EXPECT_EQ(analyses.m_block_infos, ref_block_infos);

  // All the extents involved have 8 * 12 * 16 / 4 = 384 complex numbers
  EXPECT_EQ(analyses.m_max_buffer_size, std::size_t(384) * 2);
}

/// \brief Test SlabBlockAnalysesInternal for a plan which does not need any
/// transpose, i.e. the FFT axis is already local on the input topology.
///
/// in topology (4, 1, 1) -> out topology (4, 1, 1), axes (2)
/// The plan consists of a single FFT block, whose extents are permuted with
/// the map bringing the FFT axis innermost. That map is the only quantity of
/// this test which is not hard-coded, since it is computed from the axes and
/// the layout of the Views by KokkosFFT.
///
/// \tparam ValueType Value type of the input data (real for R2C)
/// \tparam LayoutType Layout of the Input/Output Views
template <typename ValueType, typename LayoutType>
void test_slab_analyses_without_transpose() {
  constexpr std::size_t DIM = 3, FFT_DIM = 1;
  using extents_type = std::array<std::size_t, DIM>;
  using axes_type    = std::array<std::size_t, FFT_DIM>;
  using AnalysesType = KokkosFFT::Distributed::Impl::SlabBlockAnalysesInternal<
      ValueType, LayoutType, std::size_t, DIM, FFT_DIM>;

  constexpr bool is_left = std::is_same_v<LayoutType, Kokkos::LayoutLeft>;
  constexpr bool is_R2C  = KokkosFFT::Impl::is_real_v<ValueType>;

  axes_type axes{2};
  extents_type gin_extents{8, 12, 16};
  extents_type gout_extents{8, 12, is_R2C ? 9u : 16u};
  extents_type in_topology{4, 1, 1}, out_topology{4, 1, 1};
  extents_type in_extents{2, 12, 16};

  AnalysesType analyses(in_extents, gin_extents, gout_extents, in_topology,
                        out_topology, axes, MPI_COMM_WORLD);

  ASSERT_EQ(analyses.m_block_infos.size(), std::size_t(1));
  auto block = analyses.m_block_infos.at(0);

  EXPECT_EQ(block.m_block_type, KokkosFFT::Distributed::Impl::BlockType::FFT);
  EXPECT_EQ(block.m_fft_dim, std::size_t(1));
  EXPECT_EQ(block.m_block_idx, std::size_t(0));
  EXPECT_EQ(block.m_in_map, block.m_out_map);
  EXPECT_TRUE(is_permutation(block.m_in_map));

  // The map must bring the FFT axis to the innermost position
  std::size_t innermost =
      is_left ? block.m_in_map.front() : block.m_in_map.back();
  EXPECT_EQ(innermost, std::size_t(2));

  // The extents are the local extents permuted with the map of the block
  EXPECT_EQ(block.m_in_extents, KokkosFFT::Impl::compute_mapped_extents(
                                    in_extents, block.m_in_map));
  EXPECT_EQ(block.m_out_extents,
            KokkosFFT::Impl::compute_mapped_extents(
                extents_type{2, 12, is_R2C ? 9u : 16u}, block.m_out_map));

  // 2 * 12 * 16 (or 2 * 12 * 9 for R2C) complex numbers
  EXPECT_EQ(analyses.m_max_buffer_size,
            (is_R2C ? std::size_t(216) : std::size_t(384)) * 2);
}

/// \brief Test PencilBlockAnalysesInternal for a 1D FFT over the axis 2 of a 3D
/// View, which is distributed over 8 processes.
///
/// in topology (1, 2, 4) -> out topology (2, 4, 1), axes (2)
/// The FFT axis is only local on the output topology, which cannot be reached
/// by a single all2all. Hence the plan is
/// Transpose (1, 2, 4) -> (2, 1, 4) -> Transpose (2, 1, 4) -> (2, 4, 1)
/// -> FFT over the axis 2
///
/// \tparam LayoutType Layout of the Input/Output Views
/// \tparam TopologyLayoutType Layout of the pencil topologies
template <typename LayoutType, typename TopologyLayoutType>
void test_pencil_analyses_1D() {
  constexpr std::size_t DIM = 3, FFT_DIM = 1;
  using value_type          = Kokkos::complex<double>;
  using extents_type        = std::array<std::size_t, DIM>;
  using buffer_extents_type = std::array<std::size_t, DIM + 1>;
  using axes_type           = std::array<std::size_t, FFT_DIM>;
  using topology_type =
      KokkosFFT::Distributed::Topology<std::size_t, DIM, TopologyLayoutType>;
  using AnalysesType =
      KokkosFFT::Distributed::Impl::PencilBlockAnalysesInternal<
          value_type, LayoutType, std::size_t, DIM, FFT_DIM, TopologyLayoutType,
          TopologyLayoutType>;

  constexpr bool is_left = std::is_same_v<LayoutType, Kokkos::LayoutLeft>;
  constexpr bool is_topology_right =
      std::is_same_v<TopologyLayoutType, Kokkos::LayoutRight>;

  axes_type axes{2};
  extents_type gin_extents{8, 12, 16}, gout_extents{8, 12, 16};
  topology_type in_topology{1, 2, 4}, out_topology{2, 4, 1};
  extents_type mid_topology{2, 1, 4};
  extents_type in_extents{8, 6, 4};

  AnalysesType analyses(in_extents, gin_extents, gout_extents, in_topology,
                        out_topology, axes, MPI_COMM_WORLD);

  // First transpose: no FFT follows, so the out_axis 1 is brought innermost
  extents_type map0{0, 1, 2};
  extents_type map1 = is_left ? extents_type{1, 0, 2} : extents_type{0, 2, 1};
  auto extents1 =
      KokkosFFT::Impl::compute_mapped_extents(extents_type{4, 12, 4}, map1);
  buffer_extents_type buffer0 = is_left ? buffer_extents_type{4, 6, 4, 2}
                                        : buffer_extents_type{2, 4, 6, 4};

  // Second transpose: the FFT over the axis 2 follows
  extents_type map2 = is_left ? extents_type{2, 1, 0} : extents_type{0, 1, 2};
  auto extents2 =
      KokkosFFT::Impl::compute_mapped_extents(extents_type{4, 3, 16}, map2);
  buffer_extents_type buffer1 = is_left ? buffer_extents_type{4, 3, 4, 4}
                                        : buffer_extents_type{4, 4, 3, 4};

  // The axis of the all2all is the axis of the topology which is exchanged.
  // It depends on the layout of the topology, since the first non-one element
  // of a LayoutRight topology is the second one of a LayoutLeft topology.
  std::size_t comm_axis0 = is_topology_right ? 0 : 1;
  std::size_t comm_axis1 = is_topology_right ? 1 : 0;

  std::vector<KokkosFFT::Distributed::Impl::BlockInfo<DIM>> ref_block_infos = {
      make_transpose_block(in_topology.array(), mid_topology, in_extents,
                           extents1, buffer0, map0, map1, /*in_axis=*/0,
                           /*out_axis=*/1, comm_axis0, /*block_idx=*/0),
      make_transpose_block(mid_topology, out_topology.array(), extents1,
                           extents2, buffer1, map1, map2, /*in_axis=*/1,
                           /*out_axis=*/2, comm_axis1, /*block_idx=*/1),
      make_fft_block(extents2, extents2, map2, /*fft_dim=*/1,
                     /*block_idx=*/0)};

  EXPECT_EQ(analyses.m_block_infos, ref_block_infos);

  // All the extents involved have 8 * 12 * 16 / 8 = 192 complex numbers
  EXPECT_EQ(analyses.m_max_buffer_size, std::size_t(192) * 2);
}

/// \brief Test PencilBlockAnalysesInternal for a 2D FFT over the axes (1, 2) of
/// a 3D View, which is distributed over 8 processes.
///
/// in topology (1, 2, 4) -> out topology (2, 4, 1), axes (1, 2)
/// None of the FFT axes is local on the input topology and the output topology
/// cannot be reached directly from the topology where the last FFT is
/// performed. Hence the plan is
/// Transpose (1, 2, 4) -> (4, 2, 1) -> FFT over the axis 2
/// -> Transpose (4, 2, 1) -> (4, 1, 2) -> FFT over the axis 1
/// -> Transpose (4, 1, 2) -> (1, 4, 2) -> Transpose (1, 4, 2) -> (2, 4, 1)
/// The last transpose is the last block of the plan and thus restores the
/// identity map.
///
/// \tparam LayoutType Layout of the Input/Output Views
/// \tparam TopologyLayoutType Layout of the pencil topologies
template <typename LayoutType, typename TopologyLayoutType>
void test_pencil_analyses_2D() {
  constexpr std::size_t DIM = 3, FFT_DIM = 2;
  using value_type          = Kokkos::complex<double>;
  using extents_type        = std::array<std::size_t, DIM>;
  using buffer_extents_type = std::array<std::size_t, DIM + 1>;
  using axes_type           = std::array<std::size_t, FFT_DIM>;
  using topology_type =
      KokkosFFT::Distributed::Topology<std::size_t, DIM, TopologyLayoutType>;
  using AnalysesType =
      KokkosFFT::Distributed::Impl::PencilBlockAnalysesInternal<
          value_type, LayoutType, std::size_t, DIM, FFT_DIM, TopologyLayoutType,
          TopologyLayoutType>;

  constexpr bool is_left = std::is_same_v<LayoutType, Kokkos::LayoutLeft>;
  constexpr bool is_topology_right =
      std::is_same_v<TopologyLayoutType, Kokkos::LayoutRight>;

  axes_type axes{1, 2};
  extents_type gin_extents{8, 12, 16}, gout_extents{8, 12, 16};
  topology_type in_topology{1, 2, 4}, out_topology{2, 4, 1};
  extents_type topology1{4, 2, 1}, topology2{4, 1, 2}, topology3{1, 4, 2};
  extents_type in_extents{8, 6, 4};

  AnalysesType analyses(in_extents, gin_extents, gout_extents, in_topology,
                        out_topology, axes, MPI_COMM_WORLD);

  // First transpose: the FFT over the axis 2 follows
  extents_type map0{0, 1, 2};
  extents_type map1 = is_left ? extents_type{2, 0, 1} : extents_type{0, 1, 2};
  auto extents1 =
      KokkosFFT::Impl::compute_mapped_extents(extents_type{2, 6, 16}, map1);
  buffer_extents_type buffer0 = is_left ? buffer_extents_type{2, 6, 4, 4}
                                        : buffer_extents_type{4, 2, 6, 4};

  // Second transpose: the FFT over the axis 1 follows
  extents_type map2 = is_left ? extents_type{1, 2, 0} : extents_type{0, 2, 1};
  auto extents2 =
      KokkosFFT::Impl::compute_mapped_extents(extents_type{2, 12, 8}, map2);
  buffer_extents_type buffer1 = is_left ? buffer_extents_type{2, 6, 8, 2}
                                        : buffer_extents_type{2, 2, 6, 8};

  // Third transpose: no FFT follows, so the out_axis 0 is brought innermost
  extents_type map3 = is_left ? extents_type{0, 1, 2} : extents_type{2, 1, 0};
  auto extents3 =
      KokkosFFT::Impl::compute_mapped_extents(extents_type{8, 3, 8}, map3);
  buffer_extents_type buffer2 = is_left ? buffer_extents_type{2, 3, 8, 4}
                                        : buffer_extents_type{4, 2, 3, 8};

  // Last transpose: the map of the output View is the identity
  extents_type map4{0, 1, 2};
  extents_type extents4{4, 3, 16};
  buffer_extents_type buffer3 = is_left ? buffer_extents_type{4, 3, 8, 2}
                                        : buffer_extents_type{2, 4, 3, 8};

  std::size_t comm_axis0 = is_topology_right ? 1 : 0;
  std::size_t comm_axis1 = is_topology_right ? 0 : 1;
  std::size_t comm_axis2 = is_topology_right ? 1 : 0;
  std::size_t comm_axis3 = is_topology_right ? 0 : 1;

  std::vector<KokkosFFT::Distributed::Impl::BlockInfo<DIM>> ref_block_infos = {
      make_transpose_block(in_topology.array(), topology1, in_extents, extents1,
                           buffer0, map0, map1, /*in_axis=*/0, /*out_axis=*/2,
                           comm_axis0, /*block_idx=*/0),
      make_fft_block(extents1, extents1, map1, /*fft_dim=*/1, /*block_idx=*/0),
      make_transpose_block(topology1, topology2, extents1, extents2, buffer1,
                           map1, map2, /*in_axis=*/2, /*out_axis=*/1,
                           comm_axis1, /*block_idx=*/1),
      make_fft_block(extents2, extents2, map2, /*fft_dim=*/1, /*block_idx=*/1),
      make_transpose_block(topology2, topology3, extents2, extents3, buffer2,
                           map2, map3, /*in_axis=*/1, /*out_axis=*/0,
                           comm_axis2, /*block_idx=*/2),
      make_transpose_block(topology3, out_topology.array(), extents3, extents4,
                           buffer3, map3, map4, /*in_axis=*/0, /*out_axis=*/2,
                           comm_axis3, /*block_idx=*/3)};

  EXPECT_EQ(analyses.m_block_infos, ref_block_infos);

  // All the extents involved have 8 * 12 * 16 / 8 = 192 complex numbers
  EXPECT_EQ(analyses.m_max_buffer_size, std::size_t(192) * 2);
}

/// \brief The slab analyses must fail if the topologies are not slabs and the
/// pencil analyses must fail if the topologies are not pencils.
/// \tparam LayoutType Layout of the Input/Output Views
/// \tparam TopologyLayoutType Layout of the pencil topologies
template <typename LayoutType, typename TopologyLayoutType>
void test_analyses_invalid_topologies() {
  constexpr std::size_t DIM = 3, FFT_DIM = 1;
  using value_type   = Kokkos::complex<double>;
  using extents_type = std::array<std::size_t, DIM>;
  using axes_type    = std::array<std::size_t, FFT_DIM>;
  using topology_type =
      KokkosFFT::Distributed::Topology<std::size_t, DIM, TopologyLayoutType>;
  using SlabAnalysesType =
      KokkosFFT::Distributed::Impl::SlabBlockAnalysesInternal<
          value_type, LayoutType, std::size_t, DIM, FFT_DIM>;
  using PencilAnalysesType =
      KokkosFFT::Distributed::Impl::PencilBlockAnalysesInternal<
          value_type, LayoutType, std::size_t, DIM, FFT_DIM, TopologyLayoutType,
          TopologyLayoutType>;

  axes_type axes{2};
  extents_type gin_extents{8, 12, 16}, gout_extents{8, 12, 16};
  extents_type slab_in_topology{1, 1, 4}, slab_out_topology{4, 1, 1};
  extents_type slab_in_extents{8, 12, 4};
  topology_type pencil_in_topology{1, 2, 4}, pencil_out_topology{2, 4, 1};
  extents_type pencil_in_extents{8, 6, 4};

  // Pencil topologies are not accepted by the slab analyses
  EXPECT_THROW(
      {
        [[maybe_unused]] SlabAnalysesType analyses(
            pencil_in_extents, gin_extents, gout_extents,
            pencil_in_topology.array(), pencil_out_topology.array(), axes,
            MPI_COMM_WORLD);
      },
      std::runtime_error);

  // Slab topologies are not accepted by the pencil analyses
  EXPECT_THROW(
      {
        [[maybe_unused]] PencilAnalysesType analyses(
            slab_in_extents, gin_extents, gout_extents,
            topology_type(slab_in_topology), topology_type(slab_out_topology),
            axes, MPI_COMM_WORLD);
      },
      std::runtime_error);
}

}  // namespace

TEST(TestBlockAnalysesHelpers, count_blocks) { test_count_blocks(); }

TEST(TestBlockAnalysesHelpers, count_fft_dims) { test_count_fft_dims(); }

TEST(TestBlockAnalysesHelpers, propose_fft_block) { test_propose_fft_block(); }

TYPED_TEST_SUITE(TestProposeBlock, layout_types);

TYPED_TEST(TestProposeBlock, propose_transpose_block_last) {
  using layout_type = typename TestFixture::layout_type;
  test_propose_transpose_block<layout_type>(this->m_rank, this->m_axes,
                                            this->m_is_layout_right, true);
}

TYPED_TEST(TestProposeBlock, propose_transpose_block_not_last) {
  using layout_type = typename TestFixture::layout_type;
  test_propose_transpose_block<layout_type>(this->m_rank, this->m_axes,
                                            this->m_is_layout_right, false);
}

TYPED_TEST(TestProposeBlock, propose_transpose_block_empty_axes_last) {
  using layout_type = typename TestFixture::layout_type;
  test_propose_transpose_block<layout_type>(this->m_rank, this->m_empty_axes,
                                            this->m_is_layout_right, true);
}

TYPED_TEST(TestProposeBlock, propose_transpose_block_empty_axes_not_last) {
  using layout_type = typename TestFixture::layout_type;
  test_propose_transpose_block<layout_type>(this->m_rank, this->m_empty_axes,
                                            this->m_is_layout_right, false);
}

TYPED_TEST_SUITE(TestBlockAnalyses, layout_types);

TYPED_TEST(TestBlockAnalyses, slab_analyses_1D_C2C) {
  using layout_type = typename TestFixture::layout_type;
  test_slab_analyses_1D<Kokkos::complex<double>, layout_type>();
}

TYPED_TEST(TestBlockAnalyses, slab_analyses_1D_R2C) {
  using layout_type = typename TestFixture::layout_type;
  test_slab_analyses_1D<double, layout_type>();
}

TYPED_TEST(TestBlockAnalyses, slab_analyses_3D_C2C) {
  using layout_type = typename TestFixture::layout_type;
  test_slab_analyses_3D<layout_type>();
}

TYPED_TEST(TestBlockAnalyses, slab_analyses_without_transpose_C2C) {
  using layout_type = typename TestFixture::layout_type;
  test_slab_analyses_without_transpose<Kokkos::complex<double>, layout_type>();
}

TYPED_TEST(TestBlockAnalyses, slab_analyses_without_transpose_R2C) {
  using layout_type = typename TestFixture::layout_type;
  test_slab_analyses_without_transpose<double, layout_type>();
}

TYPED_TEST(TestBlockAnalyses, pencil_analyses_1D_C2C) {
  using layout_type  = typename TestFixture::layout_type;
  using layout_type2 = typename TestFixture::layout_type2;
  test_pencil_analyses_1D<layout_type, layout_type2>();
}

TYPED_TEST(TestBlockAnalyses, pencil_analyses_2D_C2C) {
  using layout_type  = typename TestFixture::layout_type;
  using layout_type2 = typename TestFixture::layout_type2;
  test_pencil_analyses_2D<layout_type, layout_type2>();
}

TYPED_TEST(TestBlockAnalyses, analyses_invalid_topologies) {
  using layout_type  = typename TestFixture::layout_type;
  using layout_type2 = typename TestFixture::layout_type2;
  test_analyses_invalid_topologies<layout_type, layout_type2>();
}
