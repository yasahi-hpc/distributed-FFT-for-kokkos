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
struct TestPack : public ::testing::Test {
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
void test_pack_view2D(int rank, int nprocs, int order = 0) {
  using SrcView2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using DstView3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using map_type      = std::array<std::size_t, 2>;

  const int n0 = 8, n1 = 7;
  const int n0_local = ((n0 - 1) / nprocs) + 1;
  const int n1_local = ((n1 - 1) / nprocs) + 1;

  map_type src_map = (order == 0) ? map_type({0, 1}) : map_type({1, 0});

  int n0_send = 0, n1_send = 0, n2_send = 0;
  int n0_xpencil = 0, n1_xpencil = 0;
  int n0_ypencil = 0, n1_ypencil = 0;

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

  if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    n0_send = n0_local;
    n1_send = n1_local;
    n2_send = nprocs;
  } else {
    n0_send = nprocs;
    n1_send = n0_local;
    n2_send = n1_local;
  }

  DstView3DType xsend("xsend", n0_send, n1_send, n2_send),
      xsend_ref("xsend_ref", n0_send, n1_send, n2_send);
  DstView3DType ysend("ysend", n0_send, n1_send, n2_send),
      ysend_ref("ysend_ref", n0_send, n1_send, n2_send);

  SrcView2DType xpencil("xpencil", n0_xpencil, n1_xpencil),
      ypencil("ypencil", n0_ypencil, n1_ypencil);

  double dx = M_PI * 2.0 / static_cast<double>(n0);
  double dy = M_PI * 2.0 / static_cast<double>(n1);

  auto h_xpencil   = Kokkos::create_mirror_view(xpencil);
  auto h_ypencil   = Kokkos::create_mirror_view(ypencil);
  auto h_xsend_ref = Kokkos::create_mirror_view(xsend_ref);
  auto h_ysend_ref = Kokkos::create_mirror_view(ysend_ref);

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
          auto tmp_xpencil = std::cos(gx) * std::sin(ly);
          if (order == 0) {
            h_xpencil(gi0, i1) = tmp_xpencil;
          } else {
            h_xpencil(i1, gi0) = tmp_xpencil;
          }
          if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
            h_xsend_ref(i0, i1, p) = tmp_xpencil;
          } else {
            h_xsend_ref(p, i0, i1) = tmp_xpencil;
          }
        }
        if (li0 < n0 && gi1 < n1) {
          auto tmp_ypencil = std::cos(lx) * std::sin(gy);
          if (order == 0) {
            h_ypencil(i0, gi1) = tmp_ypencil;
          } else {
            h_ypencil(gi1, i0) = tmp_ypencil;
          }
          if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
            h_ysend_ref(i0, i1, p) = tmp_ypencil;
          } else {
            h_ysend_ref(p, i0, i1) = tmp_ypencil;
          }
        }
      }
    }
  }

  Kokkos::deep_copy(xpencil, h_xpencil);
  Kokkos::deep_copy(ypencil, h_ypencil);

  execution_space exec;
  pack(exec, xpencil, xsend, src_map, 0);
  pack(exec, ypencil, ysend, src_map, 1);

  auto h_xsend =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), xsend);
  auto h_ysend =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ysend);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;

  // Check xsend is correct
  for (int i2 = 0; i2 < xsend.extent(2); i2++) {
    for (int i1 = 0; i1 < xsend.extent(1); i1++) {
      for (int i0 = 0; i0 < xsend.extent(0); i0++) {
        auto diff = Kokkos::abs(h_xsend(i0, i1, i2) - h_xsend_ref(i0, i1, i2));
        EXPECT_LE(diff, epsilon);
      }
    }
  }

  // Check ysend is correct
  for (int i2 = 0; i2 < ysend.extent(2); i2++) {
    for (int i1 = 0; i1 < ysend.extent(1); i1++) {
      for (int i0 = 0; i0 < ysend.extent(0); i0++) {
        auto diff = Kokkos::abs(h_ysend(i0, i1, i2) - h_ysend_ref(i0, i1, i2));
        EXPECT_LE(diff, epsilon);
      }
    }
  }
}

template <typename T, typename LayoutType>
void test_pack_view3D(int rank, int nprocs, int order = 0) {
  using SrcView3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using DstView4DType = Kokkos::View<T****, LayoutType, execution_space>;
  using map_type      = std::array<std::size_t, 3>;

  const int n0 = 16, n1 = 15, n2 = 17;
  const int n0_local = ((n0 - 1) / nprocs) + 1;
  const int n1_local = ((n1 - 1) / nprocs) + 1;
  const int n2_local = ((n2 - 1) / nprocs) + 1;

  map_type src_map = (order == 0)   ? map_type({0, 1, 2})
                     : (order == 1) ? map_type({0, 2, 1})
                     : (order == 2) ? map_type({1, 0, 2})
                     : (order == 3) ? map_type({1, 2, 0})
                     : (order == 4) ? map_type({2, 0, 1})
                                    : map_type({2, 1, 0});

  std::string rank_str = std::to_string(rank);

  int n0_send = 0, n1_send = 0, n2_send = 0, n3_send = 0;
  int n0_xpencil = 0, n1_xpencil = 0, n2_xpencil = 0;
  int n0_ypencil = 0, n1_ypencil = 0, n2_ypencil = 0;
  int n0_zpencil = 0, n1_zpencil = 0, n2_zpencil = 0;
  if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    n0_send = n0_local;
    n1_send = n1_local;
    n2_send = n2_local;
    n3_send = nprocs;
  } else {
    n0_send = nprocs;
    n1_send = n0_local;
    n2_send = n1_local;
    n3_send = n2_local;
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

    n0_ypencil = n0;
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

  DstView4DType xsend("xsend", n0_send, n1_send, n2_send, n3_send),
      xsend_ref("xsend_ref", n0_send, n1_send, n2_send, n3_send);
  DstView4DType ysend("ysend", n0_send, n1_send, n2_send, n3_send),
      ysend_ref("ysend_ref", n0_send, n1_send, n2_send, n3_send);
  DstView4DType zsend("zsend", n0_send, n1_send, n2_send, n3_send),
      zsend_ref("zsend_ref", n0_send, n1_send, n2_send, n3_send);

  SrcView3DType xpencil("xpencil" + rank_str, n0_xpencil, n1_xpencil,
                        n2_xpencil),
      ypencil("ypencil" + rank_str, n0_ypencil, n1_ypencil, n2_ypencil),
      zpencil("zpencil" + rank_str, n0_zpencil, n1_zpencil, n2_zpencil);

  double dx = M_PI * 2.0 / static_cast<double>(n0);
  double dy = M_PI * 2.0 / static_cast<double>(n1);
  double dz = M_PI * 2.0 / static_cast<double>(n2);

  auto h_xpencil   = Kokkos::create_mirror_view(xpencil);
  auto h_ypencil   = Kokkos::create_mirror_view(ypencil);
  auto h_zpencil   = Kokkos::create_mirror_view(zpencil);
  auto h_xsend_ref = Kokkos::create_mirror_view(xsend_ref);
  auto h_ysend_ref = Kokkos::create_mirror_view(ysend_ref);
  auto h_zsend_ref = Kokkos::create_mirror_view(zsend_ref);

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
              h_xsend_ref(i0, i1, i2, p) = tmp_xpencil_ref;
            } else {
              h_xsend_ref(p, i0, i1, i2) = tmp_xpencil_ref;
            }
            if (order == 0) {
              h_xpencil(gi0, i1, i2) = tmp_xpencil_ref;
            } else if (order == 1) {
              h_xpencil(gi0, i2, i1) = tmp_xpencil_ref;
            } else if (order == 2) {
              h_xpencil(i1, gi0, i2) = tmp_xpencil_ref;
            } else if (order == 3) {
              h_xpencil(i1, i2, gi0) = tmp_xpencil_ref;
            } else if (order == 4) {
              h_xpencil(i2, gi0, i1) = tmp_xpencil_ref;
            } else {
              h_xpencil(i2, i1, gi0) = tmp_xpencil_ref;
            }
          }
          if (li0 < n0 && gi1 < n1 && li2 < n2) {
            auto tmp_ypencil_ref = std::cos(lx) * std::sin(gy) * std::sin(lz);
            if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
              h_ysend_ref(i0, i1, i2, p) = tmp_ypencil_ref;
            } else {
              h_ysend_ref(p, i0, i1, i2) = tmp_ypencil_ref;
            }
            if (order == 0) {
              h_ypencil(i0, gi1, i2) = tmp_ypencil_ref;
            } else if (order == 1) {
              h_ypencil(i0, i2, gi1) = tmp_ypencil_ref;
            } else if (order == 2) {
              h_ypencil(gi1, i0, i2) = tmp_ypencil_ref;
            } else if (order == 3) {
              h_ypencil(gi1, i2, i0) = tmp_ypencil_ref;
            } else if (order == 4) {
              h_ypencil(i2, i0, gi1) = tmp_ypencil_ref;
            } else {
              h_ypencil(i2, gi1, i0) = tmp_ypencil_ref;
            }
          }
          if (li0 < n0 && li1 < n1 && gi2 < n2) {
            auto tmp_zpencil_ref = std::cos(lx) * std::sin(ly) * std::sin(gz);
            if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
              h_zsend_ref(i0, i1, i2, p) = tmp_zpencil_ref;
            } else {
              h_zsend_ref(p, i0, i1, i2) = tmp_zpencil_ref;
            }
            if (order == 0) {
              h_zpencil(i0, i1, gi2) = tmp_zpencil_ref;
            } else if (order == 1) {
              h_zpencil(i0, gi2, i1) = tmp_zpencil_ref;
            } else if (order == 2) {
              h_zpencil(i1, i0, gi2) = tmp_zpencil_ref;
            } else if (order == 3) {
              h_zpencil(i1, gi2, i0) = tmp_zpencil_ref;
            } else if (order == 4) {
              h_zpencil(gi2, i0, i1) = tmp_zpencil_ref;
            } else {
              h_zpencil(gi2, i1, i0) = tmp_zpencil_ref;
            }
          }
        }
      }
    }
  }

  Kokkos::deep_copy(xpencil, h_xpencil);
  Kokkos::deep_copy(ypencil, h_ypencil);
  Kokkos::deep_copy(zpencil, h_zpencil);

  execution_space exec;
  pack(exec, xpencil, xsend, src_map, 0);
  pack(exec, ypencil, ysend, src_map, 1);
  pack(exec, zpencil, zsend, src_map, 2);

  auto h_xsend =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), xsend);
  auto h_ysend =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ysend);
  auto h_zsend =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), zsend);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;

  // Check xsend is correct
  for (int i3 = 0; i3 < xsend.extent(3); i3++) {
    for (int i2 = 0; i2 < xsend.extent(2); i2++) {
      for (int i1 = 0; i1 < xsend.extent(1); i1++) {
        for (int i0 = 0; i0 < xsend.extent(0); i0++) {
          auto diff = Kokkos::abs(h_xsend(i0, i1, i2, i3) -
                                  h_xsend_ref(i0, i1, i2, i3));
          EXPECT_LE(diff, epsilon);
        }
      }
    }
  }

  // Check ysend is correct
  for (int i3 = 0; i3 < ysend.extent(3); i3++) {
    for (int i2 = 0; i2 < ysend.extent(2); i2++) {
      for (int i1 = 0; i1 < ysend.extent(1); i1++) {
        for (int i0 = 0; i0 < ysend.extent(0); i0++) {
          auto diff = Kokkos::abs(h_ysend(i0, i1, i2, i3) -
                                  h_ysend_ref(i0, i1, i2, i3));
          EXPECT_LE(diff, epsilon);
        }
      }
    }
  }

  // Check zsend is correct
  for (int i3 = 0; i3 < zsend.extent(3); i3++) {
    for (int i2 = 0; i2 < zsend.extent(2); i2++) {
      for (int i1 = 0; i1 < zsend.extent(1); i1++) {
        for (int i0 = 0; i0 < zsend.extent(0); i0++) {
          auto diff = Kokkos::abs(h_zsend(i0, i1, i2, i3) -
                                  h_zsend_ref(i0, i1, i2, i3));
          EXPECT_LE(diff, epsilon);
        }
      }
    }
  }
}

}  // namespace

TYPED_TEST_SUITE(TestPack, test_types);

TYPED_TEST(TestPack, View2D_01) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_pack_view2D<float_type, layout_type>(this->m_rank, this->m_nprocs, 0);
}

TYPED_TEST(TestPack, View2D_10) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_pack_view2D<float_type, layout_type>(this->m_rank, this->m_nprocs, 1);
}

TYPED_TEST(TestPack, View3D_012) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_pack_view3D<float_type, layout_type>(this->m_rank, this->m_nprocs, 0);
}

TYPED_TEST(TestPack, View3D_021) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_pack_view3D<float_type, layout_type>(this->m_rank, this->m_nprocs, 1);
}

TYPED_TEST(TestPack, View3D_102) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_pack_view3D<float_type, layout_type>(this->m_rank, this->m_nprocs, 2);
}

TYPED_TEST(TestPack, View3D_120) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_pack_view3D<float_type, layout_type>(this->m_rank, this->m_nprocs, 3);
}

TYPED_TEST(TestPack, View3D_201) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_pack_view3D<float_type, layout_type>(this->m_rank, this->m_nprocs, 4);
}

TYPED_TEST(TestPack, View3D_210) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_pack_view3D<float_type, layout_type>(this->m_rank, this->m_nprocs, 5);
}
