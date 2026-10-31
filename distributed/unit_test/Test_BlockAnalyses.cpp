#include <algorithm>
#include <array>
#include <type_traits>
#include <utility>
#include <vector>

#include <mpi.h>
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

/// \brief Fixture for SlabBlockAnalysesInternal and
/// PencilBlockAnalysesInternal. It is parameterized in the same way as
/// TestProposeBlock (Test_BlockAnalysesHelpers.cpp): layout_type is the layout
/// of the Input/Output Views and layout_type2 is the layout of the pencil
/// topologies. The slab analyses do not take the layout of the topology, hence
/// they only depend on layout_type.
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
  [[maybe_unused]] constexpr std::size_t DIM = 3, FFT_DIM = 1;
  using extents_type        = std::array<std::size_t, DIM>;
  using buffer_extents_type = std::array<std::size_t, DIM + 1>;
  using axes_type           = std::array<std::size_t, FFT_DIM>;
  using AnalysesType = KokkosFFT::Distributed::Impl::SlabBlockAnalysesInternal<
      ValueType, LayoutType, std::size_t, DIM, FFT_DIM>;

  constexpr bool is_left = std::is_same_v<LayoutType, Kokkos::LayoutLeft>;
  constexpr bool is_R2C  = KokkosFFT::Impl::is_real_v<ValueType>;

  axes_type axes{2};
  extents_type gin_extents{8, 12, 16};
  extents_type gout_extents{8, 12, is_R2C ? 9u : 16u};
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
      extents_type{2, 12, is_R2C ? 9u : 16u}, out_map);

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
  [[maybe_unused]] constexpr std::size_t DIM = 3, FFT_DIM = 3;
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
  [[maybe_unused]] constexpr std::size_t DIM = 3, FFT_DIM = 1;
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

/// \brief Test SharedBlockAnalysesInternal, where every FFT axis is local and
/// no data is exchanged between the ranks.
///
/// 1. Shared topology (1, 1, 1), axes (2): a single FFT block on the global
///    extents, whose extents are permuted with the map bringing the FFT axis
///    innermost.
/// 2. Batch-distributed topology (4, 1, 1), axes (2): the same single FFT
///    block as proposed by SlabBlockAnalysesInternal, which has no transpose
///    either for these topologies.
///
/// \tparam ValueType Value type of the input data (real for R2C)
/// \tparam LayoutType Layout of the Input/Output Views
template <typename ValueType, typename LayoutType>
void test_shared_analyses() {
  [[maybe_unused]] constexpr std::size_t DIM = 3, FFT_DIM = 1;
  using extents_type = std::array<std::size_t, DIM>;
  using axes_type    = std::array<std::size_t, FFT_DIM>;
  using SharedAnalysesType =
      KokkosFFT::Distributed::Impl::SharedBlockAnalysesInternal<
          ValueType, LayoutType, std::size_t, DIM, FFT_DIM>;
  using SlabAnalysesType =
      KokkosFFT::Distributed::Impl::SlabBlockAnalysesInternal<
          ValueType, LayoutType, std::size_t, DIM, FFT_DIM>;

  constexpr bool is_left = std::is_same_v<LayoutType, Kokkos::LayoutLeft>;
  constexpr bool is_R2C  = KokkosFFT::Impl::is_real_v<ValueType>;

  axes_type axes{2};
  extents_type gin_extents{8, 12, 16};
  extents_type gout_extents{8, 12, is_R2C ? 9u : 16u};

  // 1. Shared topology: the local extents are the global extents. This
  // topology describes a single rank (as chosen by the plan factory), so the
  // case is only run with one rank. The condition is the same on all ranks,
  // which keeps the MPI collectives of the case 2 matched.
  int nprocs;
  ::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  if (nprocs == 1) {
    extents_type shared_topology{1, 1, 1};
    SharedAnalysesType analyses(gin_extents, gin_extents, gout_extents,
                                shared_topology, shared_topology, axes,
                                MPI_COMM_WORLD);

    EXPECT_EQ(analyses.m_block_infos.size(), std::size_t(1));
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

    // The extents are the global extents permuted with the map of the block
    EXPECT_EQ(block.m_in_extents, KokkosFFT::Impl::compute_mapped_extents(
                                      gin_extents, block.m_in_map));
    EXPECT_EQ(block.m_out_extents, KokkosFFT::Impl::compute_mapped_extents(
                                       gout_extents, block.m_out_map));

    // 8 * 12 * 16 (or 8 * 12 * 9 for R2C) complex numbers
    EXPECT_EQ(analyses.m_max_buffer_size,
              (is_R2C ? std::size_t(864) : std::size_t(1536)) * 2);
  }

  // 2. Batch-distributed topology: same block as the slab analyses
  extents_type batch_topology{4, 1, 1};
  extents_type in_extents{2, 12, 16};
  SharedAnalysesType batch_analyses(in_extents, gin_extents, gout_extents,
                                    batch_topology, batch_topology, axes,
                                    MPI_COMM_WORLD);
  SlabAnalysesType slab_analyses(in_extents, gin_extents, gout_extents,
                                 batch_topology, batch_topology, axes,
                                 MPI_COMM_WORLD);

  EXPECT_EQ(batch_analyses.m_block_infos, slab_analyses.m_block_infos);
  EXPECT_EQ(batch_analyses.m_max_buffer_size, slab_analyses.m_max_buffer_size);
}

/// \brief SharedBlockAnalysesInternal must fail if the input and output
/// topologies differ or if they are distributed along an FFT axis.
/// \tparam LayoutType Layout of the Input/Output Views
template <typename LayoutType>
void test_shared_analyses_invalid_topologies() {
  [[maybe_unused]] constexpr std::size_t DIM = 3, FFT_DIM = 1;
  using value_type   = Kokkos::complex<double>;
  using extents_type = std::array<std::size_t, DIM>;
  using axes_type    = std::array<std::size_t, FFT_DIM>;
  using AnalysesType =
      KokkosFFT::Distributed::Impl::SharedBlockAnalysesInternal<
          value_type, LayoutType, std::size_t, DIM, FFT_DIM>;

  axes_type axes{2};
  extents_type gin_extents{8, 12, 16}, gout_extents{8, 12, 16};

  // The input and output topologies differ
  extents_type topology0{4, 1, 1}, topology1{1, 4, 1};
  EXPECT_THROW(
      {
        [[maybe_unused]] AnalysesType analyses(
            extents_type{2, 12, 16}, gin_extents, gout_extents, topology0,
            topology1, axes, MPI_COMM_WORLD);
      },
      std::runtime_error);

  // The topologies are distributed along the FFT axis
  extents_type topology2{1, 1, 4};
  EXPECT_THROW(
      {
        [[maybe_unused]] AnalysesType analyses(
            extents_type{8, 12, 4}, gin_extents, gout_extents, topology2,
            topology2, axes, MPI_COMM_WORLD);
      },
      std::runtime_error);
}

/// \brief Test PencilBlockAnalysesInternal for a 1D FFT over the axis 2 of a 3D
/// View, which is distributed over 8 processes.
///
/// in topology (1, 2, 4) -> out topology (2, 4, 1), axes (2)
/// The FFT axis is only local on the output topology, which cannot be reached
/// by a single all2all. Hence the plan is
/// Transpose (1, 2, 4) -> (2, 1, 4) -> Transpose (2, 1, 4) -> (2, 4, 1)
/// -> FFT over the axis 2
/// Both transposes are performed before the FFT. For R2C, they move real
/// data with the global input extents, and only the FFT output has the global
/// output extents.
///
/// \tparam T Value type of the input View (double for R2C)
/// \tparam LayoutType Layout of the Input/Output Views
/// \tparam TopologyLayoutType Layout of the pencil topologies
template <typename T, typename LayoutType, typename TopologyLayoutType>
void test_pencil_analyses_1D() {
  [[maybe_unused]] constexpr std::size_t DIM = 3, FFT_DIM = 1;
  using value_type          = T;
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
  constexpr bool is_R2C = KokkosFFT::Impl::is_real_v<value_type>;

  axes_type axes{2};
  const std::size_t n2h = is_R2C ? 9 : 16;
  extents_type gin_extents{8, 12, 16}, gout_extents{8, 12, n2h};
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

  // Second transpose: the FFT over the axis 2 follows. No FFT has been
  // performed yet, so the global input extents are used.
  extents_type map2 = is_left ? extents_type{2, 1, 0} : extents_type{0, 1, 2};
  auto extents2 =
      KokkosFFT::Impl::compute_mapped_extents(extents_type{4, 3, 16}, map2);
  auto fft_out_extents =
      KokkosFFT::Impl::compute_mapped_extents(extents_type{4, 3, n2h}, map2);
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
      make_fft_block(extents2, fft_out_extents, map2, /*fft_dim=*/1,
                     /*block_idx=*/0)};

  EXPECT_EQ(analyses.m_block_infos, ref_block_infos);

  // C2C: all the extents involved have 8 * 12 * 16 / 8 = 192 complex numbers.
  // R2C: the transposes move 192 real numbers, and the FFT output has
  // 4 * 3 * 9 = 108 complex numbers, i.e. 216 real numbers.
  EXPECT_EQ(analyses.m_max_buffer_size,
            is_R2C ? std::size_t(216) : std::size_t(192) * 2);
}

/// \brief Test PencilBlockAnalysesInternal for a 2D FFT over the axes (1, 2) of
/// a 3D View, which is distributed over 8 processes.
///
/// in topology (1, 2, 4) -> out topology (2, 4, 1), axes (1, 2)
/// None of the FFT axes is local on the input topology and the output topology
/// cannot be reached directly from the topology where the last FFT is
/// performed. Both of the following plans are minimal (4 transposes), and the
/// one found depends on the layout of the topology:
/// LayoutRight (two transposes before the first FFT):
/// Transpose (1, 2, 4) -> (2, 1, 4) -> Transpose (2, 1, 4) -> (2, 4, 1)
/// -> FFT over the axis 2 -> Transpose (2, 4, 1) -> (2, 1, 4)
/// -> FFT over the axis 1 -> Transpose (2, 1, 4) -> (2, 4, 1)
/// LayoutLeft:
/// Transpose (1, 2, 4) -> (4, 2, 1) -> FFT over the axis 2
/// -> Transpose (4, 2, 1) -> (1, 2, 4) -> Transpose (1, 2, 4) -> (2, 1, 4)
/// -> FFT over the axis 1 -> Transpose (2, 1, 4) -> (2, 4, 1)
/// The last transpose is the last block of the plan and thus restores the
/// identity map.
///
/// \tparam LayoutType Layout of the Input/Output Views
/// \tparam TopologyLayoutType Layout of the pencil topologies
template <typename LayoutType, typename TopologyLayoutType>
void test_pencil_analyses_2D() {
  [[maybe_unused]] constexpr std::size_t DIM = 3, FFT_DIM = 2;
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
  extents_type in_extents{8, 6, 4};

  AnalysesType analyses(in_extents, gin_extents, gout_extents, in_topology,
                        out_topology, axes, MPI_COMM_WORLD);

  // The map of the input View and of the output View is the identity
  extents_type map0{0, 1, 2}, map4{0, 1, 2};
  extents_type extents4{4, 3, 16};

  std::vector<KokkosFFT::Distributed::Impl::BlockInfo<DIM>> ref_block_infos;
  if constexpr (is_topology_right) {
    extents_type topology1{2, 1, 4};

    // First transpose: no FFT follows, so the out_axis 1 is brought innermost
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

    // Third transpose: the FFT over the axis 1 follows
    extents_type map3 = is_left ? extents_type{1, 2, 0} : extents_type{0, 2, 1};
    auto extents3 =
        KokkosFFT::Impl::compute_mapped_extents(extents_type{4, 12, 4}, map3);
    buffer_extents_type buffer2 = is_left ? buffer_extents_type{4, 3, 4, 4}
                                          : buffer_extents_type{4, 4, 3, 4};

    buffer_extents_type buffer3 = is_left ? buffer_extents_type{4, 3, 4, 4}
                                          : buffer_extents_type{4, 4, 3, 4};

    ref_block_infos = {
        make_transpose_block(in_topology.array(), topology1, in_extents,
                             extents1, buffer0, map0, map1, /*in_axis=*/0,
                             /*out_axis=*/1, /*comm_axis=*/0, /*block_idx=*/0),
        make_transpose_block(topology1, out_topology.array(), extents1,
                             extents2, buffer1, map1, map2, /*in_axis=*/1,
                             /*out_axis=*/2, /*comm_axis=*/1, /*block_idx=*/1),
        make_fft_block(extents2, extents2, map2, /*fft_dim=*/1,
                       /*block_idx=*/0),
        make_transpose_block(out_topology.array(), topology1, extents2,
                             extents3, buffer2, map2, map3, /*in_axis=*/2,
                             /*out_axis=*/1, /*comm_axis=*/1, /*block_idx=*/2),
        make_fft_block(extents3, extents3, map3, /*fft_dim=*/1,
                       /*block_idx=*/1),
        make_transpose_block(topology1, out_topology.array(), extents3,
                             extents4, buffer3, map3, map4, /*in_axis=*/1,
                             /*out_axis=*/2, /*comm_axis=*/1,
                             /*block_idx=*/3)};
  } else {
    extents_type topology1{4, 2, 1}, topology2{1, 2, 4}, topology3{2, 1, 4};

    // First transpose: the FFT over the axis 2 follows
    extents_type map1 = is_left ? extents_type{2, 0, 1} : extents_type{0, 1, 2};
    auto extents1 =
        KokkosFFT::Impl::compute_mapped_extents(extents_type{2, 6, 16}, map1);
    buffer_extents_type buffer0 = is_left ? buffer_extents_type{2, 6, 4, 4}
                                          : buffer_extents_type{4, 2, 6, 4};

    // Second transpose: no FFT follows, so the out_axis 0 is brought innermost
    extents_type map2 = is_left ? extents_type{0, 2, 1} : extents_type{1, 2, 0};
    auto extents2 =
        KokkosFFT::Impl::compute_mapped_extents(extents_type{8, 6, 4}, map2);
    buffer_extents_type buffer1 = is_left ? buffer_extents_type{2, 6, 4, 4}
                                          : buffer_extents_type{4, 2, 6, 4};

    // Third transpose: the FFT over the axis 1 follows
    extents_type map3 = is_left ? extents_type{1, 0, 2} : extents_type{2, 0, 1};
    auto extents3 =
        KokkosFFT::Impl::compute_mapped_extents(extents_type{4, 12, 4}, map3);
    buffer_extents_type buffer2 = is_left ? buffer_extents_type{4, 6, 4, 2}
                                          : buffer_extents_type{2, 4, 6, 4};

    buffer_extents_type buffer3 = is_left ? buffer_extents_type{4, 3, 4, 4}
                                          : buffer_extents_type{4, 4, 3, 4};

    ref_block_infos = {
        make_transpose_block(in_topology.array(), topology1, in_extents,
                             extents1, buffer0, map0, map1, /*in_axis=*/0,
                             /*out_axis=*/2, /*comm_axis=*/0, /*block_idx=*/0),
        make_fft_block(extents1, extents1, map1, /*fft_dim=*/1,
                       /*block_idx=*/0),
        make_transpose_block(topology1, topology2, extents1, extents2, buffer1,
                             map1, map2, /*in_axis=*/2, /*out_axis=*/0,
                             /*comm_axis=*/0, /*block_idx=*/1),
        make_transpose_block(topology2, topology3, extents2, extents3, buffer2,
                             map2, map3, /*in_axis=*/0, /*out_axis=*/1,
                             /*comm_axis=*/1, /*block_idx=*/2),
        make_fft_block(extents3, extents3, map3, /*fft_dim=*/1,
                       /*block_idx=*/1),
        make_transpose_block(topology3, out_topology.array(), extents3,
                             extents4, buffer3, map3, map4, /*in_axis=*/1,
                             /*out_axis=*/2, /*comm_axis=*/0,
                             /*block_idx=*/3)};
  }

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
  [[maybe_unused]] constexpr std::size_t DIM = 3, FFT_DIM = 1;
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
  test_pencil_analyses_1D<Kokkos::complex<double>, layout_type, layout_type2>();
}

TYPED_TEST(TestBlockAnalyses, pencil_analyses_1D_R2C) {
  using layout_type  = typename TestFixture::layout_type;
  using layout_type2 = typename TestFixture::layout_type2;
  test_pencil_analyses_1D<double, layout_type, layout_type2>();
}

TYPED_TEST(TestBlockAnalyses, pencil_analyses_2D_C2C) {
  using layout_type  = typename TestFixture::layout_type;
  using layout_type2 = typename TestFixture::layout_type2;
  test_pencil_analyses_2D<layout_type, layout_type2>();
}

TYPED_TEST(TestBlockAnalyses, shared_analyses_C2C) {
  using layout_type = typename TestFixture::layout_type;
  test_shared_analyses<Kokkos::complex<double>, layout_type>();
}

TYPED_TEST(TestBlockAnalyses, shared_analyses_R2C) {
  using layout_type = typename TestFixture::layout_type;
  test_shared_analyses<double, layout_type>();
}

TYPED_TEST(TestBlockAnalyses, shared_analyses_invalid_topologies) {
  using layout_type = typename TestFixture::layout_type;
  test_shared_analyses_invalid_topologies<layout_type>();
}

TYPED_TEST(TestBlockAnalyses, analyses_invalid_topologies) {
  using layout_type  = typename TestFixture::layout_type;
  using layout_type2 = typename TestFixture::layout_type2;
  test_analyses_invalid_topologies<layout_type, layout_type2>();
}
