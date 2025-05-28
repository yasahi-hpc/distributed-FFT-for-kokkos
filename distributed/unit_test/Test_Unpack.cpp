#include <mpi.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "PackUnpack.hpp"
#include "Test_Utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using test_types      = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>,
                                    std::pair<double, Kokkos::LayoutLeft>,
                                    std::pair<double, Kokkos::LayoutRight>>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct TestUnpack : public ::testing::Test {
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
void test_unpack_view2D(int rank, int nprocs, int order = 0) {
  using SrcView2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using DstView3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using map_type      = std::array<std::size_t, 2>;

  const int n0 = 8, n1 = 7;
  const int n0_local = ((n0 - 1) / nprocs) + 1;
  const int n1_local = ((n1 - 1) / nprocs) + 1;

  map_type dst_map = (order == 0) ? map_type({0, 1}) : map_type({1, 0});

  std::string rank_str = std::to_string(rank);

  int n0_recv = 0, n1_recv = 0, n2_recv = 0;
  int n0_xpencil = 0, n1_xpencil = 0;
  int n0_ypencil = 0, n1_ypencil = 0;
  if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    n0_recv = n0_local;
    n1_recv = n1_local;
    n2_recv = nprocs;
  } else {
    n0_recv = nprocs;
    n1_recv = n0_local;
    n2_recv = n1_local;
  }

  if (order == 0) {
    // n0, n1
    n0_xpencil = n0;
    n1_xpencil = n1_local;
    n0_ypencil = n0_local;
    n1_ypencil = n1;
  } else {
    // n1, n0
    n0_xpencil = n1_local;
    n1_xpencil = n0;
    n0_ypencil = n1;
    n1_ypencil = n0_local;
  }
  DstView3DType xrecv("xrecv", n0_recv, n1_recv, n2_recv);
  DstView3DType yrecv("yrecv", n0_recv, n1_recv, n2_recv);

  SrcView2DType xpencil("xpencil" + rank_str, n0_xpencil, n1_xpencil),
      xpencil_ref("xpencil_ref" + rank_str, n0_xpencil, n1_xpencil),
      ypencil("ypencil" + rank_str, n0_ypencil, n1_ypencil),
      ypencil_ref("ypencil_ref" + rank_str, n0_ypencil, n1_ypencil);

  double dx = M_PI * 2.0 / static_cast<double>(n0);
  double dy = M_PI * 2.0 / static_cast<double>(n1);

  auto h_xrecv       = Kokkos::create_mirror_view(xrecv);
  auto h_yrecv       = Kokkos::create_mirror_view(yrecv);
  auto h_xpencil_ref = Kokkos::create_mirror_view(xpencil_ref);
  auto h_ypencil_ref = Kokkos::create_mirror_view(ypencil_ref);

  for (int i1 = 0; i1 < n1_local; i1++) {
    for (int i0 = 0; i0 < n0_local; i0++) {
      for (int p = 0; p < nprocs; p++) {
        int gi0   = i0 + n0_local * p;
        int li0   = i0 + n0_local * rank;
        int gi1   = i1 + n1_local * p;
        int li1   = i1 + n1_local * rank;
        double gx = gi0 * dx;
        double lx = li0 * dx;
        double gy = gi1 * dy;
        double ly = li1 * dy;
        if (gi0 < n0 && li1 < n1) {
          auto tmp_xpencil_ref = std::cos(gx) * std::sin(ly);
          if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
            h_xrecv(i0, i1, p) = tmp_xpencil_ref;
          } else {
            h_xrecv(p, i0, i1) = tmp_xpencil_ref;
          }

          if (order == 0) {
            h_xpencil_ref(gi0, i1) = tmp_xpencil_ref;
          } else {
            h_xpencil_ref(i1, gi0) = tmp_xpencil_ref;
          }
        }
        if (li0 < n0 && gi1 < n1) {
          auto tmp_ypencil_ref = std::cos(lx) * std::sin(gy);
          if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
            h_yrecv(i0, i1, p) = tmp_ypencil_ref;
          } else {
            h_yrecv(p, i0, i1) = tmp_ypencil_ref;
          }
          if (order == 0) {
            h_ypencil_ref(i0, gi1) = tmp_ypencil_ref;
          } else {
            h_ypencil_ref(gi1, i0) = tmp_ypencil_ref;
          }
        }
      }
    }
  }

  Kokkos::deep_copy(xrecv, h_xrecv);
  Kokkos::deep_copy(yrecv, h_yrecv);
  Kokkos::deep_copy(xpencil_ref, h_xpencil_ref);
  Kokkos::deep_copy(ypencil_ref, h_ypencil_ref);

  execution_space exec;
  unpack(exec, xrecv, xpencil, dst_map, 0);
  unpack(exec, yrecv, ypencil, dst_map, 1);

  auto h_xpencil =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), xpencil);
  auto h_ypencil =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ypencil);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;

  // Check xpencil is correct
  for (int i1 = 0; i1 < xpencil.extent(1); i1++) {
    for (int i0 = 0; i0 < xpencil.extent(0); i0++) {
      auto diff = Kokkos::abs(h_xpencil(i0, i1) - h_xpencil_ref(i0, i1));
      EXPECT_LE(diff, epsilon);
    }
  }

  // Check ypencil is correct
  for (int i1 = 0; i1 < ypencil.extent(1); i1++) {
    for (int i0 = 0; i0 < ypencil.extent(0); i0++) {
      auto diff = Kokkos::abs(h_ypencil(i0, i1) - h_ypencil_ref(i0, i1));
      EXPECT_LE(diff, epsilon);
    }
  }
}

template <typename T, typename LayoutType>
void test_unpack_view3D(int rank, int nprocs, int order = 0) {
  using SrcView3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using DstView4DType = Kokkos::View<T****, LayoutType, execution_space>;
  using map_type      = std::array<std::size_t, 3>;

  const int n0 = 16, n1 = 15, n2 = 17;
  const int n0_local = ((n0 - 1) / nprocs) + 1;
  const int n1_local = ((n1 - 1) / nprocs) + 1;
  const int n2_local = ((n2 - 1) / nprocs) + 1;

  map_type dst_map = (order == 0)   ? map_type({0, 1, 2})
                     : (order == 1) ? map_type({0, 2, 1})
                     : (order == 2) ? map_type({1, 0, 2})
                     : (order == 3) ? map_type({1, 2, 0})
                     : (order == 4) ? map_type({2, 0, 1})
                                    : map_type({2, 1, 0});

  std::string rank_str = std::to_string(rank);

  int n0_recv = 0, n1_recv = 0, n2_recv = 0, n3_recv = 0;
  int n0_xpencil = 0, n1_xpencil = 0, n2_xpencil = 0;
  int n0_ypencil = 0, n1_ypencil = 0, n2_ypencil = 0;
  int n0_zpencil = 0, n1_zpencil = 0, n2_zpencil = 0;
  if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    n0_recv = n0_local;
    n1_recv = n1_local;
    n2_recv = n2_local;
    n3_recv = nprocs;
  } else {
    n0_recv = nprocs;
    n1_recv = n0_local;
    n2_recv = n1_local;
    n3_recv = n2_local;
  }

  if (order == 0) {
    // n0, n1, n2
    n0_xpencil = n0;
    n1_xpencil = n1_local;
    n2_xpencil = n2_local;
    n0_ypencil = n0_local;
    n1_ypencil = n1;
    n2_ypencil = n2_local;
    n0_zpencil = n0_local;
    n1_zpencil = n1_local;
    n2_zpencil = n2;
  } else if (order == 1) {
    // n0, n2, n1
    n0_xpencil = n0;
    n1_xpencil = n2_local;
    n2_xpencil = n1_local;
    n0_ypencil = n0_local;
    n1_ypencil = n2_local;
    n2_ypencil = n1;
    n0_zpencil = n0_local;
    n1_zpencil = n2;
    n2_zpencil = n1_local;
  } else if (order == 2) {
    // n1, n0, n2
    n0_xpencil = n1_local;
    n1_xpencil = n0;
    n2_xpencil = n2_local;
    n0_ypencil = n1;
    n1_ypencil = n0_local;
    n2_ypencil = n2_local;
    n0_zpencil = n1_local;
    n1_zpencil = n0_local;
    n2_zpencil = n2;
  } else if (order == 3) {
    // n1, n2, n0
    n0_xpencil = n1_local;
    n1_xpencil = n2_local;
    n2_xpencil = n0;
    n0_ypencil = n1;
    n1_ypencil = n2_local;
    n2_ypencil = n0_local;
    n0_zpencil = n1_local;
    n1_zpencil = n2;
    n2_zpencil = n0_local;
  } else if (order == 4) {
    // n2, n0, n1
    n0_xpencil = n2_local;
    n1_xpencil = n0;
    n2_xpencil = n1_local;
    n0_ypencil = n2_local;
    n1_ypencil = n0_local;
    n2_ypencil = n1;
    n0_zpencil = n2;
    n1_zpencil = n0_local;
    n2_zpencil = n1_local;
  } else {
    // n2, n1, n0
    n0_xpencil = n2_local;
    n1_xpencil = n1_local;
    n2_xpencil = n0;
    n0_ypencil = n2_local;
    n1_ypencil = n1;
    n2_ypencil = n0_local;
    n0_zpencil = n2;
    n1_zpencil = n1_local;
    n2_zpencil = n0_local;
  }

  DstView4DType xrecv("xrecv", n0_recv, n1_recv, n2_recv, n3_recv);
  DstView4DType yrecv("yrecv", n0_recv, n1_recv, n2_recv, n3_recv);
  DstView4DType zrecv("zrecv", n0_recv, n1_recv, n2_recv, n3_recv);

  SrcView3DType xpencil("xpencil" + rank_str, n0_xpencil, n1_xpencil,
                        n2_xpencil),
      xpencil_ref("xpencil_ref" + rank_str, n0_xpencil, n1_xpencil, n2_xpencil),
      ypencil("ypencil" + rank_str, n0_ypencil, n1_ypencil, n2_ypencil),
      ypencil_ref("ypencil_ref" + rank_str, n0_ypencil, n1_ypencil, n2_ypencil),
      zpencil("zpencil" + rank_str, n0_zpencil, n1_zpencil, n2_zpencil),
      zpencil_ref("zpencil_ref" + rank_str, n0_zpencil, n1_zpencil, n2_zpencil);

  double dx = M_PI * 2.0 / static_cast<double>(n0);
  double dy = M_PI * 2.0 / static_cast<double>(n1);
  double dz = M_PI * 2.0 / static_cast<double>(n2);

  auto h_xrecv       = Kokkos::create_mirror_view(xrecv);
  auto h_yrecv       = Kokkos::create_mirror_view(yrecv);
  auto h_zrecv       = Kokkos::create_mirror_view(zrecv);
  auto h_xpencil_ref = Kokkos::create_mirror_view(xpencil_ref);
  auto h_ypencil_ref = Kokkos::create_mirror_view(ypencil_ref);
  auto h_zpencil_ref = Kokkos::create_mirror_view(zpencil_ref);

  for (int i2 = 0; i2 < n2_local; i2++) {
    for (int i1 = 0; i1 < n1_local; i1++) {
      for (int i0 = 0; i0 < n0_local; i0++) {
        for (int p = 0; p < nprocs; p++) {
          int gi0   = i0 + n0_local * p;
          int li0   = i0 + n0_local * rank;
          int gi1   = i1 + n1_local * p;
          int li1   = i1 + n1_local * rank;
          int gi2   = i2 + n2_local * p;
          int li2   = i2 + n2_local * rank;
          double gx = gi0 * dx;
          double lx = li0 * dx;
          double gy = gi1 * dy;
          double ly = li1 * dy;
          double gz = gi2 * dz;
          double lz = li2 * dz;
          if (gi0 < n0 && li1 < n1 && li2 < n2) {
            auto tmp_xpencil_ref = std::cos(gx) * std::sin(ly) * std::sin(lz);
            if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
              h_xrecv(i0, i1, i2, p) = tmp_xpencil_ref;
            } else {
              h_xrecv(p, i0, i1, i2) = tmp_xpencil_ref;
            }

            if (order == 0) {
              h_xpencil_ref(gi0, i1, i2) = tmp_xpencil_ref;
            } else if (order == 1) {
              h_xpencil_ref(gi0, i2, i1) = tmp_xpencil_ref;
            } else if (order == 2) {
              h_xpencil_ref(i1, gi0, i2) = tmp_xpencil_ref;
            } else if (order == 3) {
              h_xpencil_ref(i1, i2, gi0) = tmp_xpencil_ref;
            } else if (order == 4) {
              h_xpencil_ref(i2, gi0, i1) = tmp_xpencil_ref;
            } else {
              h_xpencil_ref(i2, i1, gi0) = tmp_xpencil_ref;
            }
          }
          if (li0 < n0 && gi1 < n1 && li2 < n2) {
            auto tmp_ypencil_ref = std::cos(lx) * std::sin(gy) * std::sin(lz);
            if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
              h_yrecv(i0, i1, i2, p) = tmp_ypencil_ref;
            } else {
              h_yrecv(p, i0, i1, i2) = tmp_ypencil_ref;
            }
            if (order == 0) {
              h_ypencil_ref(i0, gi1, i2) = tmp_ypencil_ref;
            } else if (order == 1) {
              h_ypencil_ref(i0, i2, gi1) = tmp_ypencil_ref;
            } else if (order == 2) {
              h_ypencil_ref(gi1, i0, i2) = tmp_ypencil_ref;
            } else if (order == 3) {
              h_ypencil_ref(gi1, i2, i0) = tmp_ypencil_ref;
            } else if (order == 4) {
              h_ypencil_ref(i2, i0, gi1) = tmp_ypencil_ref;
            } else {
              h_ypencil_ref(i2, gi1, i0) = tmp_ypencil_ref;
            }
          }
          if (li0 < n0 && li1 < n1 && gi2 < n2) {
            auto tmp_zpencil_ref = std::cos(lx) * std::sin(ly) * std::sin(gz);
            if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
              h_zrecv(i0, i1, i2, p) = tmp_zpencil_ref;
            } else {
              h_zrecv(p, i0, i1, i2) = tmp_zpencil_ref;
            }
            if (order == 0) {
              h_zpencil_ref(i0, i1, gi2) = tmp_zpencil_ref;
            } else if (order == 1) {
              h_zpencil_ref(i0, gi2, i1) = tmp_zpencil_ref;
            } else if (order == 2) {
              h_zpencil_ref(i1, i0, gi2) = tmp_zpencil_ref;
            } else if (order == 3) {
              h_zpencil_ref(i1, gi2, i0) = tmp_zpencil_ref;
            } else if (order == 4) {
              h_zpencil_ref(gi2, i0, i1) = tmp_zpencil_ref;
            } else {
              h_zpencil_ref(gi2, i1, i0) = tmp_zpencil_ref;
            }
          }
        }
      }
    }
  }

  Kokkos::deep_copy(xrecv, h_xrecv);
  Kokkos::deep_copy(yrecv, h_yrecv);
  Kokkos::deep_copy(zrecv, h_zrecv);
  Kokkos::deep_copy(xpencil_ref, h_xpencil_ref);
  Kokkos::deep_copy(ypencil_ref, h_ypencil_ref);
  Kokkos::deep_copy(zpencil_ref, h_zpencil_ref);

  execution_space exec;
  unpack(exec, xrecv, xpencil, dst_map, 0);
  unpack(exec, yrecv, ypencil, dst_map, 1);
  unpack(exec, zrecv, zpencil, dst_map, 2);

  auto h_xpencil =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), xpencil);
  auto h_ypencil =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ypencil);
  auto h_zpencil =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), zpencil);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;

  // Check xpencil is correct
  for (int i2 = 0; i2 < xpencil.extent(2); i2++) {
    for (int i1 = 0; i1 < xpencil.extent(1); i1++) {
      for (int i0 = 0; i0 < xpencil.extent(0); i0++) {
        auto diff =
            Kokkos::abs(h_xpencil(i0, i1, i2) - h_xpencil_ref(i0, i1, i2));
        EXPECT_LE(diff, epsilon);
      }
    }
  }

  // Check ypencil is correct
  for (int i2 = 0; i2 < ypencil.extent(2); i2++) {
    for (int i1 = 0; i1 < ypencil.extent(1); i1++) {
      for (int i0 = 0; i0 < ypencil.extent(0); i0++) {
        auto diff =
            Kokkos::abs(h_ypencil(i0, i1, i2) - h_ypencil_ref(i0, i1, i2));
        EXPECT_LE(diff, epsilon);
      }
    }
  }

  // Check zpencil is correct
  for (int i2 = 0; i2 < zpencil.extent(2); i2++) {
    for (int i1 = 0; i1 < zpencil.extent(1); i1++) {
      for (int i0 = 0; i0 < zpencil.extent(0); i0++) {
        auto diff =
            Kokkos::abs(h_zpencil(i0, i1, i2) - h_zpencil_ref(i0, i1, i2));
        EXPECT_LE(diff, epsilon);
      }
    }
  }
}

}  // namespace

TYPED_TEST_SUITE(TestUnpack, test_types);

TYPED_TEST(TestUnpack, View2D_01) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_unpack_view2D<float_type, layout_type>(this->m_rank, this->m_nprocs, 1);
}

TYPED_TEST(TestUnpack, View2D_10) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_unpack_view2D<float_type, layout_type>(this->m_rank, this->m_nprocs, -1);
}

TYPED_TEST(TestUnpack, View3D_012) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_unpack_view3D<float_type, layout_type>(this->m_rank, this->m_nprocs, 0);
}

TYPED_TEST(TestUnpack, View3D_021) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_unpack_view3D<float_type, layout_type>(this->m_rank, this->m_nprocs, 1);
}

TYPED_TEST(TestUnpack, View3D_102) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_unpack_view3D<float_type, layout_type>(this->m_rank, this->m_nprocs, 2);
}

TYPED_TEST(TestUnpack, View3D_120) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_unpack_view3D<float_type, layout_type>(this->m_rank, this->m_nprocs, 3);
}

TYPED_TEST(TestUnpack, View3D_201) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_unpack_view3D<float_type, layout_type>(this->m_rank, this->m_nprocs, 4);
}

TYPED_TEST(TestUnpack, View3D_210) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_unpack_view3D<float_type, layout_type>(this->m_rank, this->m_nprocs, 5);
}
