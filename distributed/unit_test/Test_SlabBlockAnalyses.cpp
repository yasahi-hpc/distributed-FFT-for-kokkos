#include <mpi.h>
#include <gtest/gtest.h>
#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include "KokkosFFT_Distributed_SlabBlockAnalyses.hpp"
#include "Test_Utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using test_types = ::testing::Types<std::pair<double, Kokkos::LayoutRight>>;
// using test_types      = ::testing::Types<std::pair<float,
// Kokkos::LayoutLeft>,
//                                     std::pair<float, Kokkos::LayoutRight>,
//                                     std::pair<double, Kokkos::LayoutLeft>,
//                                     std::pair<double, Kokkos::LayoutRight>>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct TestSlabBlockAnalyses1D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;

  int m_rank   = 0;
  int m_nprocs = 1;

  virtual void SetUp() {
    ::MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &m_nprocs);
  }
};

template <typename T>
struct TestSlabBlockAnalyses2D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;

  int m_rank   = 0;
  int m_nprocs = 1;

  virtual void SetUp() {
    ::MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &m_nprocs);
  }
};

template <typename T>
struct TestSlabBlockAnalyses3D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;

  int m_rank   = 0;
  int m_nprocs = 1;

  virtual void SetUp() {
    ::MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &m_nprocs);
  }
};

template <typename T, typename LayoutType>
void test_slab_analyses_1D_view2D(std::size_t nprocs) {
  using axes_type           = std::array<std::size_t, 1>;
  using extents_type        = std::array<std::size_t, 2>;
  using buffer_extents_type = std::array<std::size_t, 3>;
  using topology_type       = std::array<std::size_t, 2>;
  using SlabBlockAnalysesType =
      SlabBlockAnalysesInternal<T, LayoutType, std::size_t, 2, 1>;
  using BlockInfoType = BlockInfo<2>;

  topology_type topology0{1, nprocs};
  topology_type topology1{nprocs, 1};

  extents_type map01{0, 1}, map10{1, 0};

  const std::size_t n0 = 8, n1 = 7;
  extents_type global_in_extents{n0, n1};
  extents_type global_out_extents_ax0, global_out_extents_ax1;
  if (KokkosFFT::Impl::is_real_v<T>) {
    global_out_extents_ax0 = {n0 / 2 + 1, n1};
    global_out_extents_ax1 = {n0, n1 / 2 + 1};
  } else {
    global_out_extents_ax0 = global_in_extents;
    global_out_extents_ax1 = global_in_extents;
  }

  auto [in_extents_t0, in_starts_t0] =
      get_local_extents(global_in_extents, topology0, MPI_COMM_WORLD);
  auto [in_extents_t1, in_starts_t1] =
      get_local_extents(global_in_extents, topology1, MPI_COMM_WORLD);
  auto [out_extents_t0_ax0, out_starts_t0_ax0] =
      get_local_extents(global_out_extents_ax0, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax0, out_starts_t1_ax0] =
      get_local_extents(global_out_extents_ax0, topology1, MPI_COMM_WORLD);
  auto [out_extents_t0_ax1, out_starts_t0_ax1] =
      get_local_extents(global_out_extents_ax1, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax1, out_starts_t1_ax1] =
      get_local_extents(global_out_extents_ax1, topology1, MPI_COMM_WORLD);

  auto [out_extents_t0_ax0_eq, out_starts_t0_ax0_eq] = get_local_extents(
      global_out_extents_ax0, topology0, MPI_COMM_WORLD, true);

  axes_type axes0 = {0}, axes1 = {1};

  buffer_extents_type buffer_in, buffer_out_ax0, buffer_out_ax1;
  if (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    buffer_in = {(n0 - 1) / nprocs + 1, (n1 - 1) / nprocs + 1, nprocs};
    if (KokkosFFT::Impl::is_real_v<T>) {
      buffer_out_ax0 = {(n0 / 2) / nprocs + 1, (n1 - 1) / nprocs + 1, nprocs};
      buffer_out_ax1 = {(n0 - 1) / nprocs + 1, (n1 / 2) / nprocs + 1, nprocs};
    } else {
      buffer_out_ax0 = buffer_in;
      buffer_out_ax1 = buffer_in;
    }
  } else {
    buffer_in = {nprocs, (n0 - 1) / nprocs + 1, (n1 - 1) / nprocs + 1};
    if (KokkosFFT::Impl::is_real_v<T>) {
      buffer_out_ax0 = {nprocs, (n0 / 2) / nprocs + 1, (n1 - 1) / nprocs + 1};
      buffer_out_ax1 = {nprocs, (n0 - 1) / nprocs + 1, (n1 / 2) / nprocs + 1};
    } else {
      buffer_out_ax0 = buffer_in;
      buffer_out_ax1 = buffer_in;
    }
  }

  if (nprocs == 1) {
    // Failure tests because these are shared topologies
    ASSERT_THROW(
        {
          SlabBlockAnalysesType slab_block_analyses(
              in_extents_t0, out_extents_t0_ax0, global_in_extents,
              global_out_extents_ax0, topology0, topology0, axes0);
        },
        std::runtime_error);

    ASSERT_THROW(
        {
          SlabBlockAnalysesType slab_block_analyses(
              in_extents_t0, out_extents_t1_ax0, global_in_extents,
              global_out_extents_ax0, topology0, topology1, axes0);
        },
        std::runtime_error);

    ASSERT_THROW(
        {
          SlabBlockAnalysesType slab_block_analyses(
              in_extents_t1, out_extents_t0_ax0, global_in_extents,
              global_out_extents_ax0, topology1, topology0, axes0);
        },
        std::runtime_error);

    ASSERT_THROW(
        {
          SlabBlockAnalysesType slab_block_analyses(
              in_extents_t1, out_extents_t1_ax0, global_in_extents,
              global_out_extents_ax0, topology1, topology1, axes0);
        },
        std::runtime_error);

    ASSERT_THROW(
        {
          SlabBlockAnalysesType slab_block_analyses(
              in_extents_t0, out_extents_t0_ax1, global_in_extents,
              global_out_extents_ax1, topology0, topology0, axes1);
        },
        std::runtime_error);

    ASSERT_THROW(
        {
          SlabBlockAnalysesType slab_block_analyses(
              in_extents_t0, out_extents_t1_ax1, global_in_extents,
              global_out_extents_ax1, topology0, topology1, axes1);
        },
        std::runtime_error);

    ASSERT_THROW(
        {
          SlabBlockAnalysesType slab_block_analyses(
              in_extents_t1, out_extents_t0_ax1, global_in_extents,
              global_out_extents_ax1, topology1, topology0, axes1);
        },
        std::runtime_error);

    ASSERT_THROW(
        {
          SlabBlockAnalysesType slab_block_analyses(
              in_extents_t1, out_extents_t1_ax1, global_in_extents,
              global_out_extents_ax1, topology1, topology1, axes1);
        },
        std::runtime_error);
  } else {
    // E.g. {1, P} and ax=0
    {
      SlabBlockAnalysesType slab_0_0_ax0(
          in_extents_t0, out_extents_t0_ax0, global_in_extents,
          global_out_extents_ax0, topology0, topology0, axes0);
      ASSERT_EQ(slab_0_0_ax0.m_block_infos.size(), 1);
      ASSERT_EQ(slab_0_0_ax0.m_op_type, OperationType::F);
      ASSERT_EQ(slab_0_0_ax0.m_max_buffer_size,
                get_size(out_extents_t0_ax0_eq) * 2);
      auto block_0_0_ax0 = slab_0_0_ax0.m_block_infos.at(0);
      BlockInfoType ref_block_0_0_ax0;
      ref_block_0_0_ax0.m_in_extents  = in_extents_t0;
      ref_block_0_0_ax0.m_out_extents = out_extents_t0_ax0;
      ref_block_0_0_ax0.m_block_type  = BlockType::FFT;
      ref_block_0_0_ax0.m_axes        = to_vector(axes0);
      EXPECT_EQ(block_0_0_ax0, ref_block_0_0_ax0);
    }

    // E.g. {1, P} FFT (ax=0) -> {P, 1}
    {
      SlabBlockAnalysesType slab_0_1_ax0(
          in_extents_t0, out_extents_t1_ax0, global_in_extents,
          global_out_extents_ax0, topology0, topology1, axes0);

      ASSERT_EQ(slab_0_1_ax0.m_block_infos.size(), 2);
      ASSERT_EQ(slab_0_1_ax0.m_op_type, OperationType::FT);
      ASSERT_EQ(slab_0_1_ax0.m_max_buffer_size, get_size(buffer_out_ax0) * 2);

      auto block0_0_1_ax0 = slab_0_1_ax0.m_block_infos.at(0);
      BlockInfoType ref_block0_0_1_ax0;
      ref_block0_0_1_ax0.m_in_extents  = in_extents_t0;
      ref_block0_0_1_ax0.m_out_extents = out_extents_t0_ax0;
      ref_block0_0_1_ax0.m_block_type  = BlockType::FFT;
      ref_block0_0_1_ax0.m_axes        = to_vector(axes0);
      EXPECT_EQ(block0_0_1_ax0, ref_block0_0_1_ax0);

      auto block1_0_1_ax0 = slab_0_1_ax0.m_block_infos.at(1);
      BlockInfoType ref_block1_0_1_ax0;
      ref_block1_0_1_ax0.m_in_topology    = topology0;
      ref_block1_0_1_ax0.m_out_topology   = topology1;
      ref_block1_0_1_ax0.m_in_extents     = out_extents_t0_ax0;
      ref_block1_0_1_ax0.m_out_extents    = out_extents_t1_ax0;
      ref_block1_0_1_ax0.m_buffer_extents = buffer_out_ax0;
      ref_block1_0_1_ax0.m_in_map         = map01;
      ref_block1_0_1_ax0.m_out_map        = map01;
      ref_block1_0_1_ax0.m_in_axis        = 0;
      ref_block1_0_1_ax0.m_out_axis       = 1;
      ref_block1_0_1_ax0.m_block_type     = BlockType::Transpose;
      EXPECT_EQ(block1_0_1_ax0, ref_block1_0_1_ax0);
    }

    // E.g. {P, 1} -> {1, P} FFT (ax=0)
    {
      SlabBlockAnalysesType slab_1_0_ax0(
          in_extents_t1, out_extents_t0_ax0, global_in_extents,
          global_out_extents_ax0, topology1, topology0, axes0);

      ASSERT_EQ(slab_1_0_ax0.m_block_infos.size(), 2);
      ASSERT_EQ(slab_1_0_ax0.m_op_type, OperationType::TF);
      if (KokkosFFT::Impl::is_real_v<T>) {
        ASSERT_EQ(slab_1_0_ax0.m_max_buffer_size,
                  get_size(out_extents_t0_ax0_eq) * 2);
      } else {
        ASSERT_EQ(slab_1_0_ax0.m_max_buffer_size, get_size(buffer_out_ax0) * 2);
      }
      auto block0_1_0_ax0 = slab_1_0_ax0.m_block_infos.at(0);
      BlockInfoType ref_block0_1_0_ax0;
      ref_block0_1_0_ax0.m_in_topology  = topology1;
      ref_block0_1_0_ax0.m_out_topology = topology0;
      ref_block0_1_0_ax0.m_in_extents   = in_extents_t1;
      if (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
        ref_block0_1_0_ax0.m_out_extents = in_extents_t0;
        ref_block0_1_0_ax0.m_out_map     = map01;
      } else {
        ref_block0_1_0_ax0.m_out_extents =
            extents_type{in_extents_t0.at(1), in_extents_t0.at(0)};
        ref_block0_1_0_ax0.m_out_map = map10;
      }
      ref_block0_1_0_ax0.m_buffer_extents = buffer_in;
      ref_block0_1_0_ax0.m_in_map         = map01;
      ref_block0_1_0_ax0.m_in_axis        = 1;
      ref_block0_1_0_ax0.m_out_axis       = 0;
      ref_block0_1_0_ax0.m_block_type     = BlockType::Transpose;

      EXPECT_EQ(block0_1_0_ax0, ref_block0_1_0_ax0);

      auto block1_1_0_ax0 = slab_1_0_ax0.m_block_infos.at(1);
      BlockInfoType ref_block1_1_0_ax0;
      ref_block1_1_0_ax0.m_block_type = BlockType::FFT;
      if (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
        ref_block1_1_0_ax0.m_in_extents  = in_extents_t0;
        ref_block1_1_0_ax0.m_out_extents = out_extents_t0_ax0;
        ref_block1_1_0_ax0.m_axes        = to_vector(axes0);
      } else {
        ref_block1_1_0_ax0.m_in_extents  = {in_extents_t0.at(1),
                                            in_extents_t0.at(0)};
        ref_block1_1_0_ax0.m_out_extents = {out_extents_t0_ax0.at(1),
                                            out_extents_t0_ax0.at(0)};
        ref_block1_1_0_ax0.m_axes        = to_vector(axes1);
      }
      EXPECT_EQ(block1_1_0_ax0, ref_block1_1_0_ax0);
    }

    // E.g. {P, 1} -> {1, P} FFT (ax=0) -> {P, 1}
    {
      SlabBlockAnalysesType slab_1_1_ax0(
          in_extents_t1, out_extents_t1_ax0, global_in_extents,
          global_out_extents_ax0, topology1, topology1, axes0);

      ASSERT_EQ(slab_1_1_ax0.m_block_infos.size(), 3);
      ASSERT_EQ(slab_1_1_ax0.m_op_type, OperationType::TFT);
      ASSERT_EQ(slab_1_1_ax0.m_max_buffer_size, get_size(buffer_out_ax0) * 2);
      auto block0_1_1_ax0 = slab_1_1_ax0.m_block_infos.at(0);
      BlockInfoType ref_block0_1_1_ax0;
      ref_block0_1_1_ax0.m_in_topology  = topology1;
      ref_block0_1_1_ax0.m_out_topology = topology0;
      ref_block0_1_1_ax0.m_in_extents   = in_extents_t1;
      if (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
        ref_block0_1_1_ax0.m_out_extents = in_extents_t0;
        ref_block0_1_1_ax0.m_out_map     = map01;
      } else {
        ref_block0_1_1_ax0.m_out_extents =
            extents_type{in_extents_t0.at(1), in_extents_t0.at(0)};
        ref_block0_1_1_ax0.m_out_map = map10;
      }
      ref_block0_1_1_ax0.m_buffer_extents = buffer_in;
      ref_block0_1_1_ax0.m_in_map         = map01;
      ref_block0_1_1_ax0.m_in_axis        = 1;
      ref_block0_1_1_ax0.m_out_axis       = 0;
      ref_block0_1_1_ax0.m_block_type     = BlockType::Transpose;

      EXPECT_EQ(block0_1_1_ax0, ref_block0_1_1_ax0);

      auto block1_1_1_ax0 = slab_1_1_ax0.m_block_infos.at(1);
      BlockInfoType ref_block1_1_1_ax0;
      ref_block1_1_1_ax0.m_block_type = BlockType::FFT;
      if (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
        ref_block1_1_1_ax0.m_in_extents  = in_extents_t0;
        ref_block1_1_1_ax0.m_out_extents = out_extents_t0_ax0;
        ref_block1_1_1_ax0.m_axes        = to_vector(axes0);
      } else {
        ref_block1_1_1_ax0.m_in_extents  = {in_extents_t0.at(1),
                                            in_extents_t0.at(0)};
        ref_block1_1_1_ax0.m_out_extents = {out_extents_t0_ax0.at(1),
                                            out_extents_t0_ax0.at(0)};
        ref_block1_1_1_ax0.m_axes        = to_vector(axes1);
      }
      EXPECT_EQ(block1_1_1_ax0, ref_block1_1_1_ax0);

      auto block2_1_1_ax0 = slab_1_1_ax0.m_block_infos.at(2);
      BlockInfoType ref_block2_1_1_ax0;
      ref_block2_1_1_ax0.m_in_topology    = topology0;
      ref_block2_1_1_ax0.m_out_topology   = topology1;
      ref_block2_1_1_ax0.m_in_extents     = ref_block1_1_1_ax0.m_out_extents;
      ref_block2_1_1_ax0.m_out_extents    = out_extents_t1_ax0;
      ref_block2_1_1_ax0.m_buffer_extents = buffer_out_ax0;
      ref_block2_1_1_ax0.m_in_map         = ref_block0_1_1_ax0.m_out_map;
      ref_block2_1_1_ax0.m_out_map        = map01;
      ref_block2_1_1_ax0.m_in_axis        = 0;
      ref_block2_1_1_ax0.m_out_axis       = 1;
      ref_block2_1_1_ax0.m_block_type     = BlockType::Transpose;
      EXPECT_EQ(block2_1_1_ax0, ref_block2_1_1_ax0);
    }

    // E.g. {1, P} ax=1
    // {1, P} -> {P, 1} + FFT -> {1, P}
    {
      SlabBlockAnalysesType slab_0_0_ax1(
          in_extents_t0, out_extents_t0_ax1, global_in_extents,
          global_out_extents_ax1, topology0, topology0, axes1);

      ASSERT_EQ(slab_0_0_ax1.m_block_infos.size(), 3);
      ASSERT_EQ(slab_0_0_ax1.m_op_type, OperationType::TFT);
      if (KokkosFFT::Impl::is_real_v<T>) {
        ASSERT_EQ(slab_0_0_ax1.m_max_buffer_size,
                  get_size(out_extents_t1_ax1) * 2);
      } else {
        ASSERT_EQ(slab_0_0_ax1.m_max_buffer_size, get_size(buffer_out_ax1) * 2);
      }
      auto block0_0_0_ax1 = slab_0_0_ax1.m_block_infos.at(0);
      BlockInfoType ref_block0_0_0_ax1;
      ref_block0_0_0_ax1.m_in_topology  = topology0;
      ref_block0_0_0_ax1.m_out_topology = topology1;
      ref_block0_0_0_ax1.m_in_extents   = in_extents_t0;
      if (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
        ref_block0_0_0_ax1.m_out_extents =
            extents_type{in_extents_t1.at(1), in_extents_t1.at(0)};
        ref_block0_0_0_ax1.m_out_map = map10;
      } else {
        ref_block0_0_0_ax1.m_out_extents = in_extents_t1;
        ref_block0_0_0_ax1.m_out_map     = map01;
      }
      ref_block0_0_0_ax1.m_buffer_extents = buffer_in;
      ref_block0_0_0_ax1.m_in_map         = map01;
      ref_block0_0_0_ax1.m_in_axis        = 0;
      ref_block0_0_0_ax1.m_out_axis       = 1;
      ref_block0_0_0_ax1.m_block_type     = BlockType::Transpose;
      EXPECT_EQ(block0_0_0_ax1, ref_block0_0_0_ax1);

      auto block1_0_0_ax1 = slab_0_0_ax1.m_block_infos.at(1);
      BlockInfoType ref_block1_0_0_ax1;
      ref_block1_0_0_ax1.m_block_type = BlockType::FFT;
      if (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
        ref_block1_0_0_ax1.m_in_extents  = {in_extents_t1.at(1),
                                            in_extents_t1.at(0)};
        ref_block1_0_0_ax1.m_out_extents = {out_extents_t1_ax1.at(1),
                                            out_extents_t1_ax1.at(0)};
        ref_block1_0_0_ax1.m_axes        = to_vector(axes0);
      } else {
        ref_block1_0_0_ax1.m_in_extents  = in_extents_t1;
        ref_block1_0_0_ax1.m_out_extents = out_extents_t1_ax1;
        ref_block1_0_0_ax1.m_axes        = to_vector(axes1);
      }
      EXPECT_EQ(block1_0_0_ax1, ref_block1_0_0_ax1);

      auto block2_0_0_ax1 = slab_0_0_ax1.m_block_infos.at(2);
      BlockInfoType ref_block2_0_0_ax1;
      ref_block2_0_0_ax1.m_in_topology    = topology1;
      ref_block2_0_0_ax1.m_out_topology   = topology0;
      ref_block2_0_0_ax1.m_in_extents     = ref_block1_0_0_ax1.m_out_extents;
      ref_block2_0_0_ax1.m_out_extents    = out_extents_t0_ax1;
      ref_block2_0_0_ax1.m_buffer_extents = buffer_out_ax1;
      ref_block2_0_0_ax1.m_in_map         = ref_block0_0_0_ax1.m_out_map;
      ref_block2_0_0_ax1.m_out_map        = map01;
      ref_block2_0_0_ax1.m_in_axis        = 1;
      ref_block2_0_0_ax1.m_out_axis       = 0;
      ref_block2_0_0_ax1.m_block_type     = BlockType::Transpose;
      EXPECT_EQ(block2_0_0_ax1, ref_block2_0_0_ax1);
    }

    // E.g. {1, P} ax=1 -> {P, 1}
    // {1, P} -> {P, 1} + FFT
    {
      SlabBlockAnalysesType slab_0_1_ax1(
          in_extents_t0, out_extents_t1_ax1, global_in_extents,
          global_out_extents_ax1, topology0, topology1, axes1);

      ASSERT_EQ(slab_0_1_ax1.m_block_infos.size(), 2);
      ASSERT_EQ(slab_0_1_ax1.m_op_type, OperationType::TF);
      if (KokkosFFT::Impl::is_real_v<T>) {
        ASSERT_EQ(slab_0_1_ax1.m_max_buffer_size,
                  get_size(out_extents_t1_ax1) * 2);
      } else {
        ASSERT_EQ(slab_0_1_ax1.m_max_buffer_size, get_size(buffer_in) * 2);
      }
      auto block0_0_1_ax1 = slab_0_1_ax1.m_block_infos.at(0);
      BlockInfoType ref_block0_0_1_ax1;
      ref_block0_0_1_ax1.m_in_topology  = topology0;
      ref_block0_0_1_ax1.m_out_topology = topology1;
      ref_block0_0_1_ax1.m_in_extents   = in_extents_t0;
      if (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
        ref_block0_0_1_ax1.m_out_extents =
            extents_type{in_extents_t1.at(1), in_extents_t1.at(0)};
        ref_block0_0_1_ax1.m_out_map = map10;
      } else {
        ref_block0_0_1_ax1.m_out_extents = in_extents_t1;
        ref_block0_0_1_ax1.m_out_map     = map01;
      }
      ref_block0_0_1_ax1.m_buffer_extents = buffer_in;
      ref_block0_0_1_ax1.m_in_map         = map01;
      ref_block0_0_1_ax1.m_in_axis        = 0;
      ref_block0_0_1_ax1.m_out_axis       = 1;
      ref_block0_0_1_ax1.m_block_type     = BlockType::Transpose;
      EXPECT_EQ(block0_0_1_ax1, ref_block0_0_1_ax1);

      auto block1_0_1_ax1 = slab_0_1_ax1.m_block_infos.at(1);
      BlockInfoType ref_block1_0_1_ax1;
      ref_block1_0_1_ax1.m_block_type = BlockType::FFT;
      if (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
        ref_block1_0_1_ax1.m_in_extents  = {in_extents_t1.at(1),
                                            in_extents_t1.at(0)};
        ref_block1_0_1_ax1.m_out_extents = {out_extents_t1_ax1.at(1),
                                            out_extents_t1_ax1.at(0)};
        ref_block1_0_1_ax1.m_axes        = to_vector(axes0);
      } else {
        ref_block1_0_1_ax1.m_in_extents  = in_extents_t1;
        ref_block1_0_1_ax1.m_out_extents = out_extents_t1_ax1;
        ref_block1_0_1_ax1.m_axes        = to_vector(axes1);
      }

      EXPECT_EQ(block1_0_1_ax1, ref_block1_0_1_ax1);
    }

    // E.g. {P, 1} ax=1 -> {1, P}
    // {P, 1} + FFT -> {1, P}
    {
      SlabBlockAnalysesType slab_1_0_ax1(
          in_extents_t1, out_extents_t0_ax1, global_in_extents,
          global_out_extents_ax1, topology1, topology0, axes1);

      ASSERT_EQ(slab_1_0_ax1.m_block_infos.size(), 2);
      ASSERT_EQ(slab_1_0_ax1.m_op_type, OperationType::FT);
      ASSERT_EQ(slab_1_0_ax1.m_max_buffer_size, get_size(buffer_out_ax1) * 2);
      auto block0_1_0_ax1 = slab_1_0_ax1.m_block_infos.at(0);
      BlockInfoType ref_block0_1_0_ax1;
      ref_block0_1_0_ax1.m_in_extents  = in_extents_t1;
      ref_block0_1_0_ax1.m_out_extents = out_extents_t1_ax1;
      ref_block0_1_0_ax1.m_block_type  = BlockType::FFT;
      ref_block0_1_0_ax1.m_axes        = to_vector(axes1);
      EXPECT_EQ(block0_1_0_ax1, ref_block0_1_0_ax1);

      auto block1_1_0_ax1 = slab_1_0_ax1.m_block_infos.at(1);
      BlockInfoType ref_block1_1_0_ax1;
      ref_block1_1_0_ax1.m_in_topology    = topology1;
      ref_block1_1_0_ax1.m_out_topology   = topology0;
      ref_block1_1_0_ax1.m_in_extents     = out_extents_t1_ax1;
      ref_block1_1_0_ax1.m_out_extents    = out_extents_t0_ax1;
      ref_block1_1_0_ax1.m_buffer_extents = buffer_out_ax1;
      ref_block1_1_0_ax1.m_in_map         = map01;
      ref_block1_1_0_ax1.m_out_map        = map01;
      ref_block1_1_0_ax1.m_in_axis        = 1;
      ref_block1_1_0_ax1.m_out_axis       = 0;
      ref_block1_1_0_ax1.m_block_type     = BlockType::Transpose;
      EXPECT_EQ(block1_1_0_ax1, ref_block1_1_0_ax1);
    }

    // E.g. {P, 1} and ax=1
    {
      SlabBlockAnalysesType slab_1_1_ax1(
          in_extents_t1, out_extents_t1_ax1, global_in_extents,
          global_out_extents_ax1, topology1, topology1, axes1);

      ASSERT_EQ(slab_1_1_ax1.m_block_infos.size(), 1);
      ASSERT_EQ(slab_1_1_ax1.m_op_type, OperationType::F);
      ASSERT_EQ(slab_1_1_ax1.m_max_buffer_size,
                get_size(out_extents_t1_ax1) * 2);
      auto block_1_1_ax1 = slab_1_1_ax1.m_block_infos.at(0);
      BlockInfoType ref_block_1_1_ax1;
      ref_block_1_1_ax1.m_in_extents  = in_extents_t1;
      ref_block_1_1_ax1.m_out_extents = out_extents_t1_ax1;
      ref_block_1_1_ax1.m_block_type  = BlockType::FFT;
      ref_block_1_1_ax1.m_axes        = to_vector(axes1);

      EXPECT_EQ(block_1_1_ax1, ref_block_1_1_ax1);
    }
  }
}

template <typename T, typename LayoutType>
void test_slab_analyses_2D_view2D(std::size_t nprocs) {
  using axes_type           = std::array<std::size_t, 2>;
  using extents_type        = std::array<std::size_t, 2>;
  using buffer_extents_type = std::array<std::size_t, 3>;
  using topology_type       = std::array<std::size_t, 2>;
  using SlabBlockAnalysesType =
      SlabBlockAnalysesInternal<T, LayoutType, std::size_t, 2, 2>;

  topology_type topology0{1, nprocs};
  topology_type topology1{nprocs, 1};

  const std::size_t n0 = 8, n1 = 7;
  extents_type global_in_extents{n0, n1},
      global_out_extents_ax01{n0, n1 / 2 + 1},
      global_out_extents_ax10{n0 / 2 + 1, n1};

  auto [in_extents_t0, in_starts_t0] =
      get_local_extents(global_in_extents, topology0, MPI_COMM_WORLD);
  auto [in_extents_t1, in_starts_t1] =
      get_local_extents(global_in_extents, topology1, MPI_COMM_WORLD);
  auto [out_extents_t0_ax01, out_starts_t0_ax01] =
      get_local_extents(global_out_extents_ax01, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax01, out_starts_t1_ax01] =
      get_local_extents(global_out_extents_ax01, topology1, MPI_COMM_WORLD);
  auto [out_extents_t0_ax10, out_starts_t0_ax10] =
      get_local_extents(global_out_extents_ax10, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax10, out_starts_t1_ax10] =
      get_local_extents(global_out_extents_ax10, topology1, MPI_COMM_WORLD);

  axes_type axes01 = {0, 1};
  axes_type axes10 = {1, 0};

  buffer_extents_type buffer_real, buffer_complex_ax01, buffer_complex_ax10;
  if (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    buffer_real = {(n0 - 1) / nprocs + 1, (n1 - 1) / nprocs + 1, nprocs};
    buffer_complex_ax01 = {(n0 - 1) / nprocs + 1, (n1 / 2) / nprocs + 1,
                           nprocs};
    buffer_complex_ax10 = {(n0 / 2) / nprocs + 1, (n1 - 1) / nprocs + 1,
                           nprocs};
  } else {
    buffer_real = {nprocs, (n0 - 1) / nprocs + 1, (n1 - 1) / nprocs + 1};
    buffer_complex_ax01 = {nprocs, (n0 - 1) / nprocs + 1,
                           (n1 / 2) / nprocs + 1};
    buffer_complex_ax10 = {nprocs, (n0 / 2) / nprocs + 1,
                           (n1 - 1) / nprocs + 1};
  }

  if (nprocs == 1) {
    // Failure tests because these are shared topologies
    ASSERT_THROW(
        {
          SlabBlockAnalysesType slab_block_analyses(
              in_extents_t0, out_extents_t0_ax01, global_in_extents,
              global_out_extents_ax01, topology0, topology0, axes01);
        },
        std::runtime_error);

    ASSERT_THROW(
        {
          SlabBlockAnalysesType slab_block_analyses(
              in_extents_t0, out_extents_t1_ax01, global_in_extents,
              global_out_extents_ax01, topology0, topology1, axes01);
        },
        std::runtime_error);

    ASSERT_THROW(
        {
          SlabBlockAnalysesType slab_block_analyses(
              in_extents_t1, out_extents_t0_ax01, global_in_extents,
              global_out_extents_ax01, topology1, topology0, axes01);
        },
        std::runtime_error);

    ASSERT_THROW(
        {
          SlabBlockAnalysesType slab_block_analyses(
              in_extents_t1, out_extents_t1_ax01, global_in_extents,
              global_out_extents_ax01, topology1, topology1, axes01);
        },
        std::runtime_error);

    ASSERT_THROW(
        {
          SlabBlockAnalysesType slab_block_analyses(
              in_extents_t0, out_extents_t0_ax10, global_in_extents,
              global_out_extents_ax10, topology0, topology0, axes10);
        },
        std::runtime_error);

    ASSERT_THROW(
        {
          SlabBlockAnalysesType slab_block_analyses(
              in_extents_t0, out_extents_t1_ax10, global_in_extents,
              global_out_extents_ax10, topology0, topology1, axes10);
        },
        std::runtime_error);

    ASSERT_THROW(
        {
          SlabBlockAnalysesType slab_block_analyses(
              in_extents_t1, out_extents_t0_ax10, global_in_extents,
              global_out_extents_ax10, topology1, topology0, axes10);
        },
        std::runtime_error);

    ASSERT_THROW(
        {
          SlabBlockAnalysesType slab_block_analyses(
              in_extents_t1, out_extents_t1_ax10, global_in_extents,
              global_out_extents_ax10, topology1, topology1, axes10);
        },
        std::runtime_error);
  } else {
    // E.g. {1, P} + FFT (ax=0) -> {P, 1} + FFT (ax=1)
    // {1, P} -> {P, 1} + FFT (ax=1) -> {1, P} + FFT (ax=0)
    SlabBlockAnalysesType slab_0_0_ax01(
        in_extents_t0, out_extents_t0_ax01, global_in_extents,
        global_out_extents_ax01, topology0, topology0, axes01);

    ASSERT_EQ(slab_0_0_ax01.m_block_infos.size(), 4);
    ASSERT_EQ(slab_0_0_ax01.m_op_type, OperationType::TFTF);
    if (KokkosFFT::Impl::is_real_v<T>) {
      ASSERT_EQ(slab_0_0_ax01.m_max_buffer_size,
                get_size(buffer_complex_ax01) * 2);
    } else {
      ASSERT_EQ(slab_0_0_ax01.m_max_buffer_size, get_size(buffer_real) * 2);
    }
    auto block0_0_0_ax01 = slab_0_0_ax01.m_block_infos.at(0);

    ASSERT_EQ(block0_0_0_ax01.m_in_topology, topology0);
    ASSERT_EQ(block0_0_0_ax01.m_out_topology, topology1);
    ASSERT_EQ(block0_0_0_ax01.m_in_extents, in_extents_t0);
    ASSERT_EQ(block0_0_0_ax01.m_out_extents, in_extents_t1);
    ASSERT_EQ(block0_0_0_ax01.m_buffer_extents, buffer_real);
    ASSERT_EQ(block0_0_0_ax01.m_block_type, BlockType::Transpose);

    auto block1_0_0_ax01 = slab_0_0_ax01.m_block_infos.at(1);

    ASSERT_EQ(block1_0_0_ax01.m_in_topology, topology1);
    ASSERT_EQ(block1_0_0_ax01.m_out_topology, topology1);
    ASSERT_EQ(block1_0_0_ax01.m_in_extents, in_extents_t1);
    ASSERT_EQ(block1_0_0_ax01.m_out_extents, out_extents_t1_ax01);
    ASSERT_EQ(block1_0_0_ax01.m_buffer_extents, buffer_extents_type{0});
    ASSERT_EQ(block1_0_0_ax01.m_block_type, BlockType::FFT);

    auto block2_0_0_ax01 = slab_0_0_ax01.m_block_infos.at(2);

    EXPECT_EQ(block2_0_0_ax01.m_in_topology, topology1);
    EXPECT_EQ(block2_0_0_ax01.m_out_topology, topology0);
    EXPECT_EQ(block2_0_0_ax01.m_in_extents, out_extents_t1_ax01);
    EXPECT_EQ(block2_0_0_ax01.m_out_extents, out_extents_t0_ax01);
    EXPECT_EQ(block2_0_0_ax01.m_buffer_extents, buffer_complex_ax01);
    EXPECT_EQ(block2_0_0_ax01.m_block_type, BlockType::Transpose);

    auto block3_0_0_ax01 = slab_0_0_ax01.m_block_infos.at(3);

    EXPECT_EQ(block3_0_0_ax01.m_in_topology, topology0);
    EXPECT_EQ(block3_0_0_ax01.m_out_topology, topology0);
    EXPECT_EQ(block3_0_0_ax01.m_in_extents, out_extents_t0_ax01);
    EXPECT_EQ(block3_0_0_ax01.m_out_extents, out_extents_t0_ax01);
    EXPECT_EQ(block3_0_0_ax01.m_buffer_extents, buffer_extents_type{0});
    EXPECT_EQ(block3_0_0_ax01.m_block_type, BlockType::FFT);

    // E.g. {1, P} + FFT (ax=1) -> {P, 1} + FFT (ax=0)
    // {1, P} -> {P, 1} + FFT (ax=1) -> {1, P} + FFT (ax=0) -> {P, 1}
    SlabBlockAnalysesType slab_0_1_ax01(
        in_extents_t0, out_extents_t1_ax01, global_in_extents,
        global_out_extents_ax01, topology0, topology1, axes01);

    ASSERT_EQ(slab_0_1_ax01.m_block_infos.size(), 5);
    auto block0_0_1_ax01 = slab_0_1_ax01.m_block_infos.at(0);

    EXPECT_EQ(block0_0_1_ax01.m_in_topology, topology0);
    EXPECT_EQ(block0_0_1_ax01.m_out_topology, topology1);
    EXPECT_EQ(block0_0_1_ax01.m_in_extents, in_extents_t0);
    EXPECT_EQ(block0_0_1_ax01.m_out_extents, in_extents_t1);
    EXPECT_EQ(block0_0_1_ax01.m_buffer_extents, buffer_real);
    EXPECT_EQ(block0_0_1_ax01.m_block_type, BlockType::Transpose);

    auto block1_0_1_ax01 = slab_0_1_ax01.m_block_infos.at(1);

    EXPECT_EQ(block1_0_1_ax01.m_in_topology, topology1);
    EXPECT_EQ(block1_0_1_ax01.m_out_topology, topology1);
    EXPECT_EQ(block1_0_1_ax01.m_in_extents, in_extents_t1);
    EXPECT_EQ(block1_0_1_ax01.m_out_extents, out_extents_t1_ax01);
    EXPECT_EQ(block1_0_1_ax01.m_buffer_extents, buffer_extents_type{0});
    EXPECT_EQ(block1_0_1_ax01.m_block_type, BlockType::FFT);

    auto block2_0_1_ax01 = slab_0_1_ax01.m_block_infos.at(2);

    EXPECT_EQ(block2_0_1_ax01.m_in_topology, topology1);
    EXPECT_EQ(block2_0_1_ax01.m_out_topology, topology0);
    EXPECT_EQ(block2_0_1_ax01.m_in_extents, out_extents_t1_ax01);
    EXPECT_EQ(block2_0_1_ax01.m_out_extents, out_extents_t0_ax01);
    EXPECT_EQ(block2_0_1_ax01.m_buffer_extents, buffer_complex_ax01);
    EXPECT_EQ(block2_0_1_ax01.m_block_type, BlockType::Transpose);

    auto block3_0_1_ax01 = slab_0_1_ax01.m_block_infos.at(3);

    EXPECT_EQ(block3_0_1_ax01.m_in_topology, topology0);
    EXPECT_EQ(block3_0_1_ax01.m_out_topology, topology0);
    EXPECT_EQ(block3_0_1_ax01.m_in_extents, out_extents_t0_ax01);
    EXPECT_EQ(block3_0_1_ax01.m_out_extents, out_extents_t0_ax01);
    EXPECT_EQ(block3_0_1_ax01.m_buffer_extents, buffer_extents_type{0});
    EXPECT_EQ(block3_0_1_ax01.m_block_type, BlockType::FFT);

    auto block4_0_1_ax01 = slab_0_1_ax01.m_block_infos.at(4);

    EXPECT_EQ(block4_0_1_ax01.m_in_topology, topology0);
    EXPECT_EQ(block4_0_1_ax01.m_out_topology, topology1);
    EXPECT_EQ(block4_0_1_ax01.m_in_extents, out_extents_t0_ax01);
    EXPECT_EQ(block4_0_1_ax01.m_out_extents, out_extents_t1_ax01);
    EXPECT_EQ(block4_0_1_ax01.m_buffer_extents, buffer_complex_ax01);
    EXPECT_EQ(block4_0_1_ax01.m_block_type, BlockType::Transpose);

    // E.g. {P, 1} + FFT (ax=1) -> {1, P} + FFT (ax=0)
    // {P, 1} + FFT (ax=1) -> {1, P} + FFT (ax=0)
    SlabBlockAnalysesType slab_1_0_ax01(
        in_extents_t1, out_extents_t0_ax01, global_in_extents,
        global_out_extents_ax01, topology1, topology0, axes01);

    ASSERT_EQ(slab_1_0_ax01.m_block_infos.size(), 3);
    auto block0_1_0_ax01 = slab_1_0_ax01.m_block_infos.at(0);

    EXPECT_EQ(block0_1_0_ax01.m_in_topology, topology1);
    EXPECT_EQ(block0_1_0_ax01.m_out_topology, topology1);
    EXPECT_EQ(block0_1_0_ax01.m_in_extents, in_extents_t1);
    EXPECT_EQ(block0_1_0_ax01.m_out_extents, out_extents_t1_ax01);
    EXPECT_EQ(block0_1_0_ax01.m_buffer_extents, buffer_extents_type{0});
    EXPECT_EQ(block0_1_0_ax01.m_block_type, BlockType::FFT);

    auto block1_1_0_ax01 = slab_1_0_ax01.m_block_infos.at(1);

    EXPECT_EQ(block1_1_0_ax01.m_in_topology, topology1);
    EXPECT_EQ(block1_1_0_ax01.m_out_topology, topology0);
    EXPECT_EQ(block1_1_0_ax01.m_in_extents, out_extents_t1_ax01);
    EXPECT_EQ(block1_1_0_ax01.m_out_extents, out_extents_t0_ax01);
    EXPECT_EQ(block1_1_0_ax01.m_buffer_extents, buffer_complex_ax01);
    EXPECT_EQ(block1_1_0_ax01.m_block_type, BlockType::Transpose);

    auto block2_1_0_ax01 = slab_1_0_ax01.m_block_infos.at(2);

    EXPECT_EQ(block2_1_0_ax01.m_in_topology, topology0);
    EXPECT_EQ(block2_1_0_ax01.m_out_topology, topology0);
    EXPECT_EQ(block2_1_0_ax01.m_in_extents, out_extents_t0_ax01);
    EXPECT_EQ(block2_1_0_ax01.m_out_extents, out_extents_t0_ax01);
    EXPECT_EQ(block2_1_0_ax01.m_buffer_extents, buffer_extents_type{0});
    EXPECT_EQ(block2_1_0_ax01.m_block_type, BlockType::FFT);

    // E.g. {P, 1} + FFT (ax=1) -> {1, P} + FFT (ax=0)
    // {P, 1} + FFT (ax=1) -> {1, P} + FFT (ax=0) -> {P, 1}
    SlabBlockAnalysesType slab_1_1_ax01(
        in_extents_t1, out_extents_t1_ax01, global_in_extents,
        global_out_extents_ax01, topology1, topology1, axes01);

    ASSERT_EQ(slab_1_1_ax01.m_block_infos.size(), 4);
    auto block0_1_1_ax01 = slab_1_1_ax01.m_block_infos.at(0);

    EXPECT_EQ(block0_1_1_ax01.m_in_topology, topology1);
    EXPECT_EQ(block0_1_1_ax01.m_out_topology, topology1);
    EXPECT_EQ(block0_1_1_ax01.m_in_extents, in_extents_t1);
    EXPECT_EQ(block0_1_1_ax01.m_out_extents, out_extents_t1_ax01);
    EXPECT_EQ(block0_1_1_ax01.m_buffer_extents, buffer_extents_type{0});
    EXPECT_EQ(block0_1_1_ax01.m_block_type, BlockType::FFT);

    auto block1_1_1_ax01 = slab_1_1_ax01.m_block_infos.at(1);

    EXPECT_EQ(block1_1_1_ax01.m_in_topology, topology1);
    EXPECT_EQ(block1_1_1_ax01.m_out_topology, topology0);
    EXPECT_EQ(block1_1_1_ax01.m_in_extents, out_extents_t1_ax01);
    EXPECT_EQ(block1_1_1_ax01.m_out_extents, out_extents_t0_ax01);
    EXPECT_EQ(block1_1_1_ax01.m_buffer_extents, buffer_complex_ax01);
    EXPECT_EQ(block1_1_1_ax01.m_block_type, BlockType::Transpose);

    auto block2_1_1_ax01 = slab_1_1_ax01.m_block_infos.at(2);

    EXPECT_EQ(block2_1_1_ax01.m_in_topology, topology0);
    EXPECT_EQ(block2_1_1_ax01.m_out_topology, topology0);
    EXPECT_EQ(block2_1_1_ax01.m_in_extents, out_extents_t0_ax01);
    EXPECT_EQ(block2_1_1_ax01.m_out_extents, out_extents_t0_ax01);
    EXPECT_EQ(block2_1_1_ax01.m_buffer_extents, buffer_extents_type{0});
    EXPECT_EQ(block2_1_1_ax01.m_block_type, BlockType::FFT);

    auto block3_1_1_ax01 = slab_1_1_ax01.m_block_infos.at(3);

    EXPECT_EQ(block3_1_1_ax01.m_in_topology, topology0);
    EXPECT_EQ(block3_1_1_ax01.m_out_topology, topology1);
    EXPECT_EQ(block3_1_1_ax01.m_in_extents, out_extents_t0_ax01);
    EXPECT_EQ(block3_1_1_ax01.m_out_extents, out_extents_t1_ax01);
    EXPECT_EQ(block3_1_1_ax01.m_buffer_extents, buffer_complex_ax01);
    EXPECT_EQ(block3_1_1_ax01.m_block_type, BlockType::Transpose);

    // Axes {1, 0}
    // E.g. {1, P} + FFT (ax=0) -> {P, 1} + FFT (ax=1)
    // {1, P} FFT (ax=0) -> {P, 1} + FFT (ax=1) -> {1, P}
    SlabBlockAnalysesType slab_0_0_ax10(
        in_extents_t0, out_extents_t0_ax10, global_in_extents,
        global_out_extents_ax10, topology0, topology0, axes10);

    ASSERT_EQ(slab_0_0_ax10.m_block_infos.size(), 4);
    ASSERT_EQ(slab_0_0_ax10.m_op_type, OperationType::FTFT);
    ASSERT_EQ(slab_0_0_ax10.m_max_buffer_size,
              get_size(buffer_complex_ax10) * 2);
    auto block0_0_0_ax10 = slab_0_0_ax10.m_block_infos.at(0);

    ASSERT_EQ(block0_0_0_ax10.m_in_topology, topology0);
    ASSERT_EQ(block0_0_0_ax10.m_out_topology, topology0);
    ASSERT_EQ(block0_0_0_ax10.m_in_extents, in_extents_t0);
    ASSERT_EQ(block0_0_0_ax10.m_out_extents, out_extents_t0_ax10);
    ASSERT_EQ(block0_0_0_ax10.m_buffer_extents, buffer_extents_type{0});
    ASSERT_EQ(block0_0_0_ax10.m_block_type, BlockType::FFT);

    auto block1_0_0_ax10 = slab_0_0_ax10.m_block_infos.at(1);

    EXPECT_EQ(block1_0_0_ax10.m_in_topology, topology0);
    EXPECT_EQ(block1_0_0_ax10.m_out_topology, topology1);
    EXPECT_EQ(block1_0_0_ax10.m_in_extents, out_extents_t0_ax10);
    EXPECT_EQ(block1_0_0_ax10.m_out_extents, out_extents_t1_ax10);
    EXPECT_EQ(block1_0_0_ax10.m_buffer_extents, buffer_complex_ax10);
    EXPECT_EQ(block1_0_0_ax10.m_block_type, BlockType::Transpose);

    auto block2_0_0_ax10 = slab_0_0_ax10.m_block_infos.at(2);

    ASSERT_EQ(block2_0_0_ax10.m_in_topology, topology1);
    ASSERT_EQ(block2_0_0_ax10.m_out_topology, topology1);
    ASSERT_EQ(block2_0_0_ax10.m_in_extents, out_extents_t1_ax10);
    ASSERT_EQ(block2_0_0_ax10.m_out_extents, out_extents_t1_ax10);
    ASSERT_EQ(block2_0_0_ax10.m_buffer_extents, buffer_extents_type{0});
    ASSERT_EQ(block2_0_0_ax10.m_block_type, BlockType::FFT);

    auto block3_0_0_ax10 = slab_0_0_ax10.m_block_infos.at(3);

    EXPECT_EQ(block3_0_0_ax10.m_in_topology, topology1);
    EXPECT_EQ(block3_0_0_ax10.m_out_topology, topology0);
    EXPECT_EQ(block3_0_0_ax10.m_in_extents, out_extents_t1_ax10);
    EXPECT_EQ(block3_0_0_ax10.m_out_extents, out_extents_t0_ax10);
    EXPECT_EQ(block3_0_0_ax10.m_buffer_extents, buffer_complex_ax10);
    EXPECT_EQ(block3_0_0_ax10.m_block_type, BlockType::Transpose);

    // Axes {1, 0}
    // E.g. {1, P} + FFT (ax=0) -> {P, 1} + FFT (ax=1)
    // {1, P} FFT (ax=0) -> {P, 1} + FFT (ax=1)
    SlabBlockAnalysesType slab_0_1_ax10(
        in_extents_t0, out_extents_t1_ax10, global_in_extents,
        global_out_extents_ax10, topology0, topology1, axes10);

    ASSERT_EQ(slab_0_1_ax10.m_block_infos.size(), 3);
    ASSERT_EQ(slab_0_1_ax10.m_op_type, OperationType::FTF);
    ASSERT_EQ(slab_0_1_ax10.m_max_buffer_size,
              get_size(buffer_complex_ax10) * 2);
    auto block0_0_1_ax10 = slab_0_1_ax10.m_block_infos.at(0);

    ASSERT_EQ(block0_0_1_ax10.m_in_topology, topology0);
    ASSERT_EQ(block0_0_1_ax10.m_out_topology, topology0);
    ASSERT_EQ(block0_0_1_ax10.m_in_extents, in_extents_t0);
    ASSERT_EQ(block0_0_1_ax10.m_out_extents, out_extents_t0_ax10);
    ASSERT_EQ(block0_0_1_ax10.m_buffer_extents, buffer_extents_type{0});
    ASSERT_EQ(block0_0_1_ax10.m_block_type, BlockType::FFT);

    auto block1_0_1_ax10 = slab_0_1_ax10.m_block_infos.at(1);

    EXPECT_EQ(block1_0_1_ax10.m_in_topology, topology0);
    EXPECT_EQ(block1_0_1_ax10.m_out_topology, topology1);
    EXPECT_EQ(block1_0_1_ax10.m_in_extents, out_extents_t0_ax10);
    EXPECT_EQ(block1_0_1_ax10.m_out_extents, out_extents_t1_ax10);
    EXPECT_EQ(block1_0_1_ax10.m_buffer_extents, buffer_complex_ax10);
    EXPECT_EQ(block1_0_1_ax10.m_block_type, BlockType::Transpose);

    auto block2_0_1_ax10 = slab_0_1_ax10.m_block_infos.at(2);

    ASSERT_EQ(block2_0_1_ax10.m_in_topology, topology1);
    ASSERT_EQ(block2_0_1_ax10.m_out_topology, topology1);
    ASSERT_EQ(block2_0_1_ax10.m_in_extents, out_extents_t1_ax10);
    ASSERT_EQ(block2_0_1_ax10.m_out_extents, out_extents_t1_ax10);
    ASSERT_EQ(block2_0_1_ax10.m_buffer_extents, buffer_extents_type{0});
    ASSERT_EQ(block2_0_1_ax10.m_block_type, BlockType::FFT);

    // Axes {1, 0}
    // E.g. {P, 1} -> {1, P} + FFT (ax=0) -> {P, 1} + FFT (ax=1) -> {1, P}
    // {P, 1} -> {1, P} + FFT (ax=0) -> {P, 1} + FFT (ax=1) -> {1, P}
    SlabBlockAnalysesType slab_1_0_ax10(
        in_extents_t1, out_extents_t0_ax10, global_in_extents,
        global_out_extents_ax10, topology1, topology0, axes10);

    ASSERT_EQ(slab_1_0_ax10.m_block_infos.size(), 5);
    ASSERT_EQ(slab_1_0_ax10.m_op_type, OperationType::TFTFT);
    if (KokkosFFT::Impl::is_real_v<T>) {
      ASSERT_EQ(slab_1_0_ax10.m_max_buffer_size,
                get_size(buffer_complex_ax10) * 2);
    } else {
      ASSERT_EQ(slab_1_0_ax10.m_max_buffer_size, get_size(buffer_real) * 2);
    }

    auto block0_1_0_ax10 = slab_1_0_ax10.m_block_infos.at(0);

    EXPECT_EQ(block0_1_0_ax10.m_in_topology, topology1);
    EXPECT_EQ(block0_1_0_ax10.m_out_topology, topology0);
    EXPECT_EQ(block0_1_0_ax10.m_in_extents, in_extents_t1);
    EXPECT_EQ(block0_1_0_ax10.m_out_extents, in_extents_t0);
    EXPECT_EQ(block0_1_0_ax10.m_buffer_extents, buffer_real);
    EXPECT_EQ(block0_1_0_ax10.m_block_type, BlockType::Transpose);

    auto block1_1_0_ax10 = slab_1_0_ax10.m_block_infos.at(1);

    ASSERT_EQ(block1_1_0_ax10.m_in_topology, topology0);
    ASSERT_EQ(block1_1_0_ax10.m_out_topology, topology0);
    ASSERT_EQ(block1_1_0_ax10.m_in_extents, in_extents_t0);
    ASSERT_EQ(block1_1_0_ax10.m_out_extents, out_extents_t0_ax10);
    ASSERT_EQ(block1_1_0_ax10.m_buffer_extents, buffer_extents_type{0});
    ASSERT_EQ(block1_1_0_ax10.m_block_type, BlockType::FFT);

    auto block2_1_0_ax10 = slab_1_0_ax10.m_block_infos.at(2);

    EXPECT_EQ(block2_1_0_ax10.m_in_topology, topology0);
    EXPECT_EQ(block2_1_0_ax10.m_out_topology, topology1);
    EXPECT_EQ(block2_1_0_ax10.m_in_extents, out_extents_t0_ax10);
    EXPECT_EQ(block2_1_0_ax10.m_out_extents, out_extents_t1_ax10);
    EXPECT_EQ(block2_1_0_ax10.m_buffer_extents, buffer_complex_ax10);
    EXPECT_EQ(block2_1_0_ax10.m_block_type, BlockType::Transpose);

    auto block3_1_0_ax10 = slab_1_0_ax10.m_block_infos.at(3);

    ASSERT_EQ(block3_1_0_ax10.m_in_topology, topology1);
    ASSERT_EQ(block3_1_0_ax10.m_out_topology, topology1);
    ASSERT_EQ(block3_1_0_ax10.m_in_extents, out_extents_t1_ax10);
    ASSERT_EQ(block3_1_0_ax10.m_out_extents, out_extents_t1_ax10);
    ASSERT_EQ(block3_1_0_ax10.m_buffer_extents, buffer_extents_type{0});
    ASSERT_EQ(block3_1_0_ax10.m_block_type, BlockType::FFT);

    auto block4_1_0_ax10 = slab_1_0_ax10.m_block_infos.at(4);

    EXPECT_EQ(block4_1_0_ax10.m_in_topology, topology1);
    EXPECT_EQ(block4_1_0_ax10.m_out_topology, topology0);
    EXPECT_EQ(block4_1_0_ax10.m_in_extents, out_extents_t1_ax10);
    EXPECT_EQ(block4_1_0_ax10.m_out_extents, out_extents_t0_ax10);
    EXPECT_EQ(block4_1_0_ax10.m_buffer_extents, buffer_complex_ax10);
    EXPECT_EQ(block4_1_0_ax10.m_block_type, BlockType::Transpose);

    // Axes {1, 0}
    // E.g. {P, 1} -> {1, P} + FFT (ax=0) -> {P, 1} + FFT (ax=1)
    // {P, 1} -> {1, P} + FFT (ax=0) -> {P, 1} + FFT (ax=1)
    SlabBlockAnalysesType slab_1_1_ax10(
        in_extents_t1, out_extents_t1_ax10, global_in_extents,
        global_out_extents_ax10, topology1, topology1, axes10);

    ASSERT_EQ(slab_1_1_ax10.m_block_infos.size(), 4);
    ASSERT_EQ(slab_1_1_ax10.m_op_type, OperationType::TFTF);
    if (KokkosFFT::Impl::is_real_v<T>) {
      ASSERT_EQ(slab_1_1_ax10.m_max_buffer_size,
                get_size(buffer_complex_ax10) * 2);
    } else {
      ASSERT_EQ(slab_1_1_ax10.m_max_buffer_size, get_size(buffer_real) * 2);
    }

    auto block0_1_1_ax10 = slab_1_1_ax10.m_block_infos.at(0);

    EXPECT_EQ(block0_1_1_ax10.m_in_topology, topology1);
    EXPECT_EQ(block0_1_1_ax10.m_out_topology, topology0);
    EXPECT_EQ(block0_1_1_ax10.m_in_extents, in_extents_t1);
    EXPECT_EQ(block0_1_1_ax10.m_out_extents, in_extents_t0);
    EXPECT_EQ(block0_1_1_ax10.m_buffer_extents, buffer_real);
    EXPECT_EQ(block0_1_1_ax10.m_block_type, BlockType::Transpose);

    auto block1_1_1_ax10 = slab_1_1_ax10.m_block_infos.at(1);

    ASSERT_EQ(block1_1_1_ax10.m_in_topology, topology0);
    ASSERT_EQ(block1_1_1_ax10.m_out_topology, topology0);
    ASSERT_EQ(block1_1_1_ax10.m_in_extents, in_extents_t0);
    ASSERT_EQ(block1_1_1_ax10.m_out_extents, out_extents_t0_ax10);
    ASSERT_EQ(block1_1_1_ax10.m_buffer_extents, buffer_extents_type{0});
    ASSERT_EQ(block1_1_1_ax10.m_block_type, BlockType::FFT);

    auto block2_1_1_ax10 = slab_1_1_ax10.m_block_infos.at(2);

    EXPECT_EQ(block2_1_1_ax10.m_in_topology, topology0);
    EXPECT_EQ(block2_1_1_ax10.m_out_topology, topology1);
    EXPECT_EQ(block2_1_1_ax10.m_in_extents, out_extents_t0_ax10);
    EXPECT_EQ(block2_1_1_ax10.m_out_extents, out_extents_t1_ax10);
    EXPECT_EQ(block2_1_1_ax10.m_buffer_extents, buffer_complex_ax10);
    EXPECT_EQ(block2_1_1_ax10.m_block_type, BlockType::Transpose);

    auto block3_1_1_ax10 = slab_1_1_ax10.m_block_infos.at(3);

    ASSERT_EQ(block3_1_1_ax10.m_in_topology, topology1);
    ASSERT_EQ(block3_1_1_ax10.m_out_topology, topology1);
    ASSERT_EQ(block3_1_1_ax10.m_in_extents, out_extents_t1_ax10);
    ASSERT_EQ(block3_1_1_ax10.m_out_extents, out_extents_t1_ax10);
    ASSERT_EQ(block3_1_1_ax10.m_buffer_extents, buffer_extents_type{0});
    ASSERT_EQ(block3_1_1_ax10.m_block_type, BlockType::FFT);
  }
}

/*
template <typename T, typename LayoutType>
void test_slab_analyses_2D_view3D(std::size_t nprocs) {
  using axes_type           = std::array<std::size_t, 2>;
  using extents_type        = std::array<std::size_t, 3>;
  using buffer_extents_type = std::array<std::size_t, 4>;
  using topology_type       = std::array<std::size_t, 3>;
  using SlabBlockAnalysesType =
      SlabBlockAnalysesInternal<T, LayoutType, std::size_t, 3, 2>;

  topology_type topology0{1, 1, nprocs};
  topology_type topology1{1, nprocs, 1};
  topology_type topology2{nprocs, 1, 1};

  std::vector<topology_type> all_topologies = {topology0, topology1, topology2};

  axes_type axes01 = {0, 1};
  axes_type axes02 = {0, 2};
  axes_type axes10 = {1, 0};
  axes_type axes12 = {1, 2};
  axes_type axes20 = {2, 0};
  axes_type axes21 = {2, 1};

  std::vector<axes_type> all_axes = {axes01, axes02, axes10,
                                     axes12, axes20, axes21};

  const std::size_t n0 = 8, n1 = 7, n2 = 10;
  extents_type global_in_extents{n0, n1, n2},
      global_out_extents_ax0{n0 / 2 + 1, n1, n2},
      global_out_extents_ax1{n0, n1 / 2 + 1, n2},
      global_out_extents_ax2{n0, n1, n2 / 2 + 1};

  std::vector<extents_type> all_global_out_extents = {
      global_out_extents_ax0, global_out_extents_ax1, global_out_extents_ax2};

  if (nprocs == 1) {
    // Failure tests because these are shared topologies
    for (const auto& axes : all_axes) {
      for (const auto& in_topo : all_topologies) {
        for (const auto& out_topo : all_topologies) {
          auto [in_extents, in_starts] =
              get_local_extents(global_in_extents, in_topo, MPI_COMM_WORLD);
          extents_type global_out_extents;
          if (axes == axes10 || axes == axes20) {
            global_out_extents = global_out_extents_ax0;
          } else if (axes == axes01 || axes == axes21) {
            global_out_extents = global_out_extents_ax1;
          } else {
            global_out_extents = global_out_extents_ax2;
          }
          auto [out_extents, out_starts] =
              get_local_extents(global_out_extents, out_topo, MPI_COMM_WORLD);

          EXPECT_THROW(
              {
                SlabBlockAnalysesType slab_block_analyses(
                    in_extents, out_extents, global_in_extents,
                    global_out_extents, in_topo, out_topo, axes);
              },
              std::runtime_error);
        }
      }
    }
  } else {
    auto [in_extents_t0, in_starts_t0] =
        get_local_extents(global_in_extents, topology0, MPI_COMM_WORLD);
    auto [in_extents_t1, in_starts_t1] =
        get_local_extents(global_in_extents, topology1, MPI_COMM_WORLD);
    auto [in_extents_t2, in_starts_t2] =
        get_local_extents(global_in_extents, topology2, MPI_COMM_WORLD);
    auto [out_extents_t0_ax0, out_starts_t0_ax0] =
        get_local_extents(global_out_extents_ax0, topology0, MPI_COMM_WORLD);
    auto [out_extents_t1_ax0, out_starts_t1_ax0] =
        get_local_extents(global_out_extents_ax0, topology1, MPI_COMM_WORLD);
    auto [out_extents_t2_ax0, out_starts_t2_ax0] =
        get_local_extents(global_out_extents_ax0, topology2, MPI_COMM_WORLD);
    auto [out_extents_t0_ax1, out_starts_t0_ax1] =
        get_local_extents(global_out_extents_ax1, topology0, MPI_COMM_WORLD);
    auto [out_extents_t1_ax1, out_starts_t1_ax1] =
        get_local_extents(global_out_extents_ax1, topology1, MPI_COMM_WORLD);
    auto [out_extents_t2_ax1, out_starts_t2_ax1] =
        get_local_extents(global_out_extents_ax1, topology2, MPI_COMM_WORLD);
    auto [out_extents_t0_ax2, out_starts_t0_ax2] =
        get_local_extents(global_out_extents_ax2, topology0, MPI_COMM_WORLD);
    auto [out_extents_t1_ax2, out_starts_t1_ax2] =
        get_local_extents(global_out_extents_ax2, topology1, MPI_COMM_WORLD);
    auto [out_extents_t2_ax2, out_starts_t2_ax2] =
        get_local_extents(global_out_extents_ax2, topology2, MPI_COMM_WORLD);

    // Topology 0 -> 0
    {
      auto buffer_real_01 = get_buffer_extents<LayoutType>(
          global_in_extents, topology0, topology1);
      auto buffer_complex_01_ax2 = get_buffer_extents<LayoutType>(
          global_out_extents_ax2, topology0, topology1);
      // E.g. {1, 1, P} + FFT2 {ax=0,1}
      SlabBlockAnalysesType slab_0_0_ax01(
          in_extents_t0, out_extents_t0_ax1, global_in_extents,
          global_out_extents_ax1, topology0, topology0, axes01);

      ASSERT_EQ(slab_0_0_ax01.m_block_infos.size(), 1);
      auto block0_0_0_ax01 = slab_0_0_ax01.m_block_infos.at(0);

      EXPECT_EQ(block0_0_0_ax01.m_in_topology, topology0);
      EXPECT_EQ(block0_0_0_ax01.m_out_topology, topology0);
      EXPECT_EQ(block0_0_0_ax01.m_in_extents, in_extents_t0);
      EXPECT_EQ(block0_0_0_ax01.m_out_extents, out_extents_t0_ax1);
      EXPECT_EQ(block0_0_0_ax01.m_buffer_extents, buffer_extents_type{0});
      EXPECT_EQ(block0_0_0_ax01.m_block_type, BlockType::FFT2);

      // E.g. {1, 1, P} + FFT2 {ax=0,1}
      SlabBlockAnalysesType slab_0_0_ax10(
          in_extents_t0, out_extents_t0_ax0, global_in_extents,
          global_out_extents_ax0, topology0, topology0, axes10);

      ASSERT_EQ(slab_0_0_ax10.m_block_infos.size(), 1);
      auto block0_0_0_ax10 = slab_0_0_ax10.m_block_infos.at(0);

      EXPECT_EQ(block0_0_0_ax10.m_in_topology, topology0);
      EXPECT_EQ(block0_0_0_ax10.m_out_topology, topology0);
      EXPECT_EQ(block0_0_0_ax10.m_in_extents, in_extents_t0);
      EXPECT_EQ(block0_0_0_ax10.m_out_extents, out_extents_t0_ax0);
      EXPECT_EQ(block0_0_0_ax10.m_buffer_extents, buffer_extents_type{0});
      EXPECT_EQ(block0_0_0_ax10.m_block_type, BlockType::FFT2);

      // E.g. {1, 1, P} -> {1, P, 1} FFT {ax=2} -> {1, 1, P} + FFT {ax=0}
      // T + F + T + F
      SlabBlockAnalysesType slab_0_0_ax02(
          in_extents_t0, out_extents_t0_ax2, global_in_extents,
          global_out_extents_ax2, topology0, topology0, axes02);

      ASSERT_EQ(slab_0_0_ax02.m_block_infos.size(), 4);
      auto block0_0_0_ax02 = slab_0_0_ax02.m_block_infos.at(0);

      EXPECT_EQ(block0_0_0_ax02.m_in_topology, topology0);
      EXPECT_EQ(block0_0_0_ax02.m_out_topology, topology1);
      EXPECT_EQ(block0_0_0_ax02.m_in_extents, in_extents_t0);
      EXPECT_EQ(block0_0_0_ax02.m_out_extents, in_extents_t1);
      EXPECT_EQ(block0_0_0_ax02.m_buffer_extents, buffer_real_01);
      EXPECT_EQ(block0_0_0_ax02.m_block_type, BlockType::Transpose);

      auto block1_0_0_ax02 = slab_0_0_ax02.m_block_infos.at(1);

      EXPECT_EQ(block1_0_0_ax02.m_in_topology, topology1);
      EXPECT_EQ(block1_0_0_ax02.m_out_topology, topology1);
      EXPECT_EQ(block1_0_0_ax02.m_in_extents, in_extents_t1);
      EXPECT_EQ(block1_0_0_ax02.m_out_extents, out_extents_t1_ax2);
      EXPECT_EQ(block1_0_0_ax02.m_buffer_extents, buffer_extents_type{0});
      EXPECT_EQ(block1_0_0_ax02.m_block_type, BlockType::FFT);

      auto block2_0_0_ax02 = slab_0_0_ax02.m_block_infos.at(2);

      EXPECT_EQ(block2_0_0_ax02.m_in_topology, topology1);
      EXPECT_EQ(block2_0_0_ax02.m_out_topology, topology0);
      EXPECT_EQ(block2_0_0_ax02.m_in_extents, out_extents_t1_ax2);
      EXPECT_EQ(block2_0_0_ax02.m_out_extents, out_extents_t0_ax2);
      EXPECT_EQ(block2_0_0_ax02.m_buffer_extents, buffer_complex_01_ax2);
      EXPECT_EQ(block2_0_0_ax02.m_block_type, BlockType::Transpose);

      auto block3_0_0_ax02 = slab_0_0_ax02.m_block_infos.at(3);

      EXPECT_EQ(block3_0_0_ax02.m_in_topology, topology0);
      EXPECT_EQ(block3_0_0_ax02.m_out_topology, topology0);
      EXPECT_EQ(block3_0_0_ax02.m_in_extents, out_extents_t0_ax2);
      EXPECT_EQ(block3_0_0_ax02.m_out_extents, out_extents_t0_ax2);
      EXPECT_EQ(block3_0_0_ax02.m_buffer_extents, buffer_extents_type{0});
      EXPECT_EQ(block3_0_0_ax02.m_block_type, BlockType::FFT);
    }
  }
}
*/

template <typename T, typename LayoutType>
void test_slab_analyses_3D_view3D(std::size_t nprocs) {
  using axes_type           = std::array<std::size_t, 3>;
  using extents_type        = std::array<std::size_t, 3>;
  using buffer_extents_type = std::array<std::size_t, 4>;
  using topology_type       = std::array<std::size_t, 3>;
  using SlabBlockAnalysesType =
      SlabBlockAnalysesInternal<T, LayoutType, std::size_t, 3, 3>;
  using BlockInfoType = BlockInfo<3>;

  topology_type topology0{1, 1, nprocs};
  topology_type topology1{1, nprocs, 1};
  topology_type topology2{nprocs, 1, 1};

  std::vector<topology_type> all_topologies = {topology0, topology1, topology2};

  axes_type ax012 = {0, 1, 2}, ax021 = {0, 2, 1}, ax102 = {1, 0, 2},
            ax120 = {1, 2, 0}, ax201 = {2, 0, 1}, ax210 = {2, 1, 0};

  extents_type map012{0, 1, 2}, map021{0, 2, 1}, map102{1, 0, 2},
      map120{1, 2, 0}, map201{2, 0, 1}, map210{2, 1, 0};

  std::vector<axes_type> all_axes = {ax012, ax021, ax102, ax120, ax201, ax210};

  const std::size_t n0 = 8, n1 = 7, n2 = 10;
  extents_type global_in_extents{n0, n1, n2}, global_out_extents_ax0,
      global_out_extents_ax1, global_out_extents_ax2;

  if (KokkosFFT::Impl::is_real_v<T>) {
    global_out_extents_ax0 = {n0 / 2 + 1, n1, n2};
    global_out_extents_ax1 = {n0, n1 / 2 + 1, n2};
    global_out_extents_ax2 = {n0, n1, n2 / 2 + 1};
  } else {
    global_out_extents_ax0 = global_in_extents;
    global_out_extents_ax1 = global_in_extents;
    global_out_extents_ax2 = global_in_extents;
  }

  buffer_extents_type buffer_real_0_1, buffer_real_0_2, buffer_real_1_2,
      buffer_complex_0_1_ax0, buffer_complex_0_1_ax1, buffer_complex_0_1_ax2,
      buffer_complex_0_2_ax0, buffer_complex_0_2_ax1, buffer_complex_0_2_ax2,
      buffer_complex_1_2_ax0, buffer_complex_1_2_ax1, buffer_complex_1_2_ax2;

  if (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    buffer_real_0_1 = {n0, (n1 - 1) / nprocs + 1, (n2 - 1) / nprocs + 1,
                       nprocs};
    buffer_real_0_2 = {(n0 - 1) / nprocs + 1, n1, (n2 - 1) / nprocs + 1,
                       nprocs};
    buffer_real_1_2 = {(n0 - 1) / nprocs + 1, (n1 - 1) / nprocs + 1, n2,
                       nprocs};
    if (KokkosFFT::Impl::is_real_v<T>) {
      buffer_complex_0_1_ax0 = {n0 / 2 + 1, (n1 - 1) / nprocs + 1,
                                (n2 - 1) / nprocs + 1, nprocs};
      buffer_complex_0_1_ax1 = {n0, (n1 / 2) / nprocs + 1,
                                (n2 - 1) / nprocs + 1, nprocs};
      buffer_complex_0_1_ax2 = {n0, (n1 - 1) / nprocs + 1,
                                (n2 / 2) / nprocs + 1, nprocs};
      buffer_complex_0_2_ax0 = {n0 / 2 + 1, (n1 - 1) / nprocs + 1,
                                (n2 - 1) / nprocs + 1, nprocs};
      buffer_complex_0_2_ax1 = {n0, (n1 / 2) / nprocs + 1,
                                (n2 - 1) / nprocs + 1, nprocs};
      buffer_complex_0_2_ax2 = {n0, (n1 - 1) / nprocs + 1,
                                (n2 / 2) / nprocs + 1, nprocs};
      buffer_complex_1_2_ax0 = {(n0 / 2) / nprocs + 1, (n1 - 1) / nprocs + 1,
                                n2, nprocs};
      buffer_complex_1_2_ax1 = {(n0 - 1) / nprocs + 1, (n1 / 2) / nprocs + 1,
                                n2, nprocs};
      buffer_complex_1_2_ax2 = {(n0 - 1) / nprocs + 1, (n1 - 1) / nprocs + 1,
                                n2 / 2 + 1, nprocs};
    } else {
      buffer_complex_0_1_ax0 = buffer_real_0_1;
      buffer_complex_0_1_ax1 = buffer_real_0_1;
      buffer_complex_0_1_ax2 = buffer_real_0_1;
      buffer_complex_0_2_ax0 = buffer_real_0_2;
      buffer_complex_0_2_ax1 = buffer_real_0_2;
      buffer_complex_0_2_ax2 = buffer_real_0_2;
      buffer_complex_1_2_ax0 = buffer_real_1_2;
      buffer_complex_1_2_ax1 = buffer_real_1_2;
      buffer_complex_1_2_ax2 = buffer_real_1_2;
    }
  } else {
    buffer_real_0_1 = {nprocs, n0, (n1 - 1) / nprocs + 1,
                       (n2 - 1) / nprocs + 1};
    buffer_real_0_2 = {nprocs, (n0 - 1) / nprocs + 1, n1,
                       (n2 - 1) / nprocs + 1};
    buffer_real_1_2 = {nprocs, (n0 - 1) / nprocs + 1, (n1 - 1) / nprocs + 1,
                       n2};
    if (KokkosFFT::Impl::is_real_v<T>) {
      buffer_complex_0_1_ax0 = {nprocs, n0 / 2 + 1, (n1 - 1) / nprocs + 1,
                                (n2 - 1) / nprocs + 1};
      buffer_complex_0_1_ax1 = {nprocs, n0, (n1 / 2) / nprocs + 1,
                                (n2 - 1) / nprocs + 1};
      buffer_complex_0_1_ax2 = {nprocs, n0, (n1 - 1) / nprocs + 1,
                                (n2 / 2) / nprocs + 1};
      buffer_complex_0_2_ax0 = {nprocs, n0 / 2 + 1, (n1 - 1) / nprocs + 1,
                                (n2 - 1) / nprocs + 1};
      buffer_complex_0_2_ax1 = {nprocs, (n1 / 2) / nprocs + 1,
                                (n2 - 1) / nprocs + 1};
      buffer_complex_0_2_ax2 = {nprocs, (n1 - 1) / nprocs + 1,
                                (n2 / 2) / nprocs + 1};
      buffer_complex_1_2_ax0 = {nprocs, (n0 / 2) / nprocs + 1,
                                (n1 - 1) / nprocs + 1, n2};
      buffer_complex_1_2_ax1 = {nprocs, (n0 - 1) / nprocs + 1,
                                (n1 / 2) / nprocs + 1, n2};
      buffer_complex_1_2_ax2 = {nprocs, (n0 - 1) / nprocs + 1,
                                (n1 - 1) / nprocs + 1, n2 / 2 + 1};
    } else {
      buffer_complex_0_1_ax0 = buffer_real_0_1;
      buffer_complex_0_1_ax1 = buffer_real_0_1;
      buffer_complex_0_1_ax2 = buffer_real_0_1;
      buffer_complex_0_2_ax0 = buffer_real_0_2;
      buffer_complex_0_2_ax1 = buffer_real_0_2;
      buffer_complex_0_2_ax2 = buffer_real_0_2;
      buffer_complex_1_2_ax0 = buffer_real_1_2;
      buffer_complex_1_2_ax1 = buffer_real_1_2;
      buffer_complex_1_2_ax2 = buffer_real_1_2;
    }
  }

  if (nprocs == 1) {
    // Failure tests because these are shared topologies
    for (const auto& axes : all_axes) {
      for (const auto& in_topo : all_topologies) {
        for (const auto& out_topo : all_topologies) {
          auto [in_extents, in_starts] =
              get_local_extents(global_in_extents, in_topo, MPI_COMM_WORLD);
          extents_type global_out_extents;
          if (axes == ax120 || axes == ax210) {
            global_out_extents = global_out_extents_ax0;
          } else if (axes == ax021 || axes == ax201) {
            global_out_extents = global_out_extents_ax1;
          } else {
            global_out_extents = global_out_extents_ax2;
          }
          auto [out_extents, out_starts] =
              get_local_extents(global_out_extents, out_topo, MPI_COMM_WORLD);

          EXPECT_THROW(
              {
                SlabBlockAnalysesType slab_block_analyses(
                    in_extents, out_extents, global_in_extents,
                    global_out_extents, in_topo, out_topo, axes);
              },
              std::runtime_error);
        }
      }
    }
  } else {
    auto [in_extents_t0, in_starts_t0] =
        get_local_extents(global_in_extents, topology0, MPI_COMM_WORLD);
    auto [in_extents_t1, in_starts_t1] =
        get_local_extents(global_in_extents, topology1, MPI_COMM_WORLD);
    auto [in_extents_t2, in_starts_t2] =
        get_local_extents(global_in_extents, topology2, MPI_COMM_WORLD);
    auto [out_extents_t0_ax0, out_starts_t0_ax0] =
        get_local_extents(global_out_extents_ax0, topology0, MPI_COMM_WORLD);
    auto [out_extents_t1_ax0, out_starts_t1_ax0] =
        get_local_extents(global_out_extents_ax0, topology1, MPI_COMM_WORLD);
    auto [out_extents_t2_ax0, out_starts_t2_ax0] =
        get_local_extents(global_out_extents_ax0, topology2, MPI_COMM_WORLD);
    auto [out_extents_t0_ax1, out_starts_t0_ax1] =
        get_local_extents(global_out_extents_ax1, topology0, MPI_COMM_WORLD);
    auto [out_extents_t1_ax1, out_starts_t1_ax1] =
        get_local_extents(global_out_extents_ax1, topology1, MPI_COMM_WORLD);
    auto [out_extents_t2_ax1, out_starts_t2_ax1] =
        get_local_extents(global_out_extents_ax1, topology2, MPI_COMM_WORLD);
    auto [out_extents_t0_ax2, out_starts_t0_ax2] =
        get_local_extents(global_out_extents_ax2, topology0, MPI_COMM_WORLD);
    auto [out_extents_t1_ax2, out_starts_t1_ax2] =
        get_local_extents(global_out_extents_ax2, topology1, MPI_COMM_WORLD);
    auto [out_extents_t2_ax2, out_starts_t2_ax2] =
        get_local_extents(global_out_extents_ax2, topology2, MPI_COMM_WORLD);
    // Axes {0, 1, 2}
    // {P, 1, 1} + FFT {ax=1,2} -> {1, P, 1} + FFT (ax=0) -> (local transpose)
    // FFT2 + Transpose + FFT
    SlabBlockAnalysesType slab_2_1_ax012(
        in_extents_t2, out_extents_t1_ax2, global_in_extents,
        global_out_extents_ax2, topology2, topology1, ax012);

    EXPECT_EQ(slab_2_1_ax012.m_block_infos.size(), 3);
    EXPECT_EQ(slab_2_1_ax012.m_op_type, OperationType::FTF);
    EXPECT_EQ(slab_2_1_ax012.m_max_buffer_size,
              get_size(buffer_complex_1_2_ax2) * 2);

    auto block0_2_1_ax012 = slab_2_1_ax012.m_block_infos.at(0);
    BlockInfoType ref_block0_2_1_ax012;
    if (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
      ref_block0_2_1_ax012.m_in_map  = map201;
      ref_block0_2_1_ax012.m_out_map = map201;
      ref_block0_2_1_ax012.m_in_extents =
          get_mapped_extents(in_extents_t2, map201);
      ref_block0_2_1_ax012.m_out_extents =
          get_mapped_extents(out_extents_t2_ax2, map201);
    } else {
      ref_block0_2_1_ax012.m_in_map      = map012;
      ref_block0_2_1_ax012.m_out_map     = map012;
      ref_block0_2_1_ax012.m_in_extents  = in_extents_t2;
      ref_block0_2_1_ax012.m_out_extents = out_extents_t2_ax2;
    }
    ref_block0_2_1_ax012.m_axes       = {1, 2};
    ref_block0_2_1_ax012.m_block_type = BlockType::FFT;
    EXPECT_EQ(block0_2_1_ax012, ref_block0_2_1_ax012);

    auto block1_2_1_ax012 = slab_2_1_ax012.m_block_infos.at(1);
    BlockInfoType ref_block1_2_1_ax012;
    ref_block1_2_1_ax012.m_in_topology  = topology2;
    ref_block1_2_1_ax012.m_out_topology = topology1;
    ref_block1_2_1_ax012.m_in_extents   = block0_2_1_ax012.m_out_extents;
    if (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
      ref_block1_2_1_ax012.m_out_map = map012;
      ref_block1_2_1_ax012.m_out_extents =
          get_mapped_extents(out_extents_t1_ax2, map012);
    } else {
      ref_block1_2_1_ax012.m_out_map = map120;
      ref_block1_2_1_ax012.m_out_extents =
          get_mapped_extents(out_extents_t1_ax2, map120);
    }
    ref_block1_2_1_ax012.m_buffer_extents = buffer_complex_1_2_ax2;
    ref_block1_2_1_ax012.m_in_map         = block0_2_1_ax012.m_out_map;
    ref_block1_2_1_ax012.m_in_axis        = 1;
    ref_block1_2_1_ax012.m_out_axis       = 0;
    ref_block1_2_1_ax012.m_block_type     = BlockType::Transpose;
    EXPECT_EQ(block1_2_1_ax012, ref_block1_2_1_ax012);

    // EXPECT_EQ(block1_2_1_ax012.m_in_topology,
    // ref_block1_2_1_ax012.m_in_topology);
    // EXPECT_EQ(block1_2_1_ax012.m_out_topology,
    // ref_block1_2_1_ax012.m_out_topology);
    // EXPECT_EQ(block1_2_1_ax012.m_in_extents,
    // ref_block1_2_1_ax012.m_in_extents);
    // EXPECT_EQ(block1_2_1_ax012.m_out_extents,
    // ref_block1_2_1_ax012.m_out_extents); EXPECT_EQ(block1_2_1_ax012.m_in_map,
    // ref_block1_2_1_ax012.m_in_map); EXPECT_EQ(block1_2_1_ax012.m_out_map,
    // ref_block1_2_1_ax012.m_out_map);
    // EXPECT_EQ(block1_2_1_ax012.m_buffer_extents,
    // ref_block1_2_1_ax012.m_buffer_extents);
    // EXPECT_EQ(block1_2_1_ax012.m_in_axis, ref_block1_2_1_ax012.m_in_axis);
    // EXPECT_EQ(block1_2_1_ax012.m_out_axis, ref_block1_2_1_ax012.m_out_axis);
    // EXPECT_EQ(block1_2_1_ax012.m_block_type,
    // ref_block1_2_1_ax012.m_block_type);

    auto block2_2_1_ax012 = slab_2_1_ax012.m_block_infos.at(2);
    BlockInfoType ref_block2_2_1_ax012;
    if (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
      ref_block2_2_1_ax012.m_in_extents = get_mapped_extents(
          out_extents_t1_ax2, ref_block1_2_1_ax012.m_out_map);
      ref_block2_2_1_ax012.m_out_extents = get_mapped_extents(
          out_extents_t1_ax2, ref_block1_2_1_ax012.m_out_map);
    } else {
      ref_block2_2_1_ax012.m_in_extents = get_mapped_extents(
          out_extents_t1_ax2, ref_block1_2_1_ax012.m_out_map);
      ref_block2_2_1_ax012.m_out_extents = get_mapped_extents(
          out_extents_t1_ax2, ref_block1_2_1_ax012.m_out_map);
    }
    ref_block2_2_1_ax012.m_axes       = {0};
    ref_block2_2_1_ax012.m_block_type = BlockType::FFT;
    EXPECT_EQ(block2_2_1_ax012, ref_block2_2_1_ax012);

    // EXPECT_EQ(block2_2_1_ax012.m_in_extents,
    // ref_block2_2_1_ax012.m_in_extents);
    // EXPECT_EQ(block2_2_1_ax012.m_out_extents,
    // ref_block2_2_1_ax012.m_out_extents); EXPECT_EQ(block2_2_1_ax012.m_axes,
    // ref_block2_2_1_ax012.m_axes); EXPECT_EQ(block2_2_1_ax012.m_block_type,
    // ref_block2_2_1_ax012.m_block_type);
  }

  /*
  auto [in_extents_t0, in_starts_t0] =
      get_local_extents(global_in_extents, topology0, MPI_COMM_WORLD);
  auto [in_extents_t1, in_starts_t1] =
      get_local_extents(global_in_extents, topology1, MPI_COMM_WORLD);
  auto [in_extents_t2, in_starts_t2] =
      get_local_extents(global_in_extents, topology2, MPI_COMM_WORLD);
  auto [out_extents_t0_ax0, out_starts_t0_ax0] =
      get_local_extents(global_out_extents_ax0, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax0, out_starts_t1_ax0] =
      get_local_extents(global_out_extents_ax0, topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax0, out_starts_t2_ax0] =
      get_local_extents(global_out_extents_ax0, topology2, MPI_COMM_WORLD);
  auto [out_extents_t0_ax1, out_starts_t0_ax1] =
      get_local_extents(global_out_extents_ax1, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax1, out_starts_t1_ax1] =
      get_local_extents(global_out_extents_ax1, topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax1, out_starts_t2_ax1] =
      get_local_extents(global_out_extents_ax1, topology2, MPI_COMM_WORLD);
  auto [out_extents_t0_ax2, out_starts_t0_ax2] =
      get_local_extents(global_out_extents_ax2, topology0, MPI_COMM_WORLD);
  auto [out_extents_t1_ax2, out_starts_t1_ax2] =
      get_local_extents(global_out_extents_ax2, topology1, MPI_COMM_WORLD);
  auto [out_extents_t2_ax2, out_starts_t2_ax2] =
      get_local_extents(global_out_extents_ax2, topology2, MPI_COMM_WORLD);
  */
}

}  // namespace

TYPED_TEST_SUITE(TestSlabBlockAnalyses1D, test_types);
TYPED_TEST_SUITE(TestSlabBlockAnalyses2D, test_types);
TYPED_TEST_SUITE(TestSlabBlockAnalyses3D, test_types);

TYPED_TEST(TestSlabBlockAnalyses1D, View2D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_slab_analyses_1D_view2D<float_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestSlabBlockAnalyses1D, View2D_C2C) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;

  test_slab_analyses_1D_view2D<complex_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestSlabBlockAnalyses2D, View2D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_slab_analyses_2D_view2D<float_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestSlabBlockAnalyses2D, View2D_C2C) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;

  test_slab_analyses_2D_view2D<complex_type, layout_type>(this->m_nprocs);
}

/*
TYPED_TEST(TestSlabBlockAnalyses2D, View3D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_slab_analyses_2D_view3D<float_type, layout_type>(this->m_nprocs);
}
  */

TYPED_TEST(TestSlabBlockAnalyses3D, View3D_R2C) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_slab_analyses_3D_view3D<float_type, layout_type>(this->m_nprocs);
}

TYPED_TEST(TestSlabBlockAnalyses3D, View3D_C2C) {
  using float_type   = typename TestFixture::float_type;
  using layout_type  = typename TestFixture::layout_type;
  using complex_type = Kokkos::complex<float_type>;

  test_slab_analyses_3D_view3D<complex_type, layout_type>(this->m_nprocs);
}
