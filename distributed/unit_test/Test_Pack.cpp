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
void test_pack_view2D(int rank, int nprocs) {
  using SrcView2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using DstView3DType = Kokkos::View<T***, LayoutType, execution_space>;

  const int n0 = 16, n1 = 15;
  const int n0_local = ((n0 - 1) / nprocs) + 1;
  const int n1_local = ((n1 - 1) / nprocs) + 1;

  std::string rank_str = std::to_string(rank);
  SrcView2DType xpencil("xpencil" + rank_str, n0, n1_local),
      ypencil("ypencil" + rank_str, n0_local, n1);

  int n0_xsend = 0, n1_xsend = 0, n2_xsend = 0;
  int n0_ysend = 0, n1_ysend = 0, n2_ysend = 0;
  if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    n0_xsend = n0_local;
    n1_xsend = n1_local;
    n2_xsend = nprocs;
    n0_ysend = n0_local;
    n1_ysend = n1_local;
    n2_ysend = nprocs;
  } else {
    n0_xsend = nprocs;
    n1_xsend = n0_local;
    n2_xsend = n1_local;
    n0_ysend = nprocs;
    n1_ysend = n0_local;
    n2_ysend = n1_local;
  }
  DstView3DType xsend("xsend", n0_xsend, n1_xsend, n2_xsend),
      xsend_ref("xsend_ref", n0_xsend, n1_xsend, n2_xsend);
  DstView3DType ysend("ysend", n0_ysend, n1_ysend, n2_ysend),
      ysend_ref("ysend_ref", n0_ysend, n1_ysend, n2_ysend);

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
          h_xpencil(gi0, i1) = std::cos(gx) * std::sin(ly);
          if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
            h_xsend_ref(i0, i1, p) = h_xpencil(gi0, i1);
          } else {
            h_xsend_ref(p, i0, i1) = h_xpencil(gi0, i1);
          }
        }
        if (li0 < n0 && gi1 < n1) {
          h_ypencil(i0, gi1) = std::cos(lx) * std::sin(gy);
          if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
            h_ysend_ref(i0, i1, p) = h_ypencil(i0, gi1);
          } else {
            h_ysend_ref(p, i0, i1) = h_ypencil(i0, gi1);
          }
        }
      }
    }
  }

  Kokkos::deep_copy(xpencil, h_xpencil);
  Kokkos::deep_copy(ypencil, h_ypencil);

  execution_space exec;
  pack(exec, xsend, xpencil, 0);
  pack(exec, ysend, ypencil, 1);

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
void test_pack_view3D_XtoY(int rank, int nprocs) {
  using SrcView3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using DstView4DType = Kokkos::View<T****, LayoutType, execution_space>;

  const int n0 = 8, n1 = 15, n2 = 7;
  const int n0_local = ((n0 - 1) / nprocs) + 1;
  const int n1_local = ((n1 - 1) / nprocs) + 1;

  std::string rank_str = std::to_string(rank);
  SrcView3DType xpencil("xpencil" + rank_str, n0, n1_local, n2),
      ypencil("ypencil" + rank_str, n0_local, n1, n2);

  int n0_send = 0, n1_send = 0, n2_send = 0, n3_send = 0;
  if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    n0_send = n0_local;
    n1_send = n1_local;
    n2_send = n2;
    n3_send = nprocs;
  } else {
    n0_send = nprocs;
    n1_send = n0_local;
    n2_send = n1_local;
    n3_send = n2;
  }
  DstView4DType xsend("xsend", n0_send, n1_send, n2_send, n3_send),
      xsend_ref("xsend_ref", n0_send, n1_send, n2_send, n3_send);
  DstView4DType ysend("ysend", n0_send, n1_send, n2_send, n3_send),
      ysend_ref("ysend_ref", n0_send, n1_send, n2_send, n3_send);

  double dx = M_PI * 2.0 / static_cast<double>(n0);
  double dy = M_PI * 2.0 / static_cast<double>(n1);
  double dz = M_PI * 2.0 / static_cast<double>(n2);

  auto h_xpencil   = Kokkos::create_mirror_view(xpencil);
  auto h_ypencil   = Kokkos::create_mirror_view(ypencil);
  auto h_xsend_ref = Kokkos::create_mirror_view(xsend_ref);
  auto h_ysend_ref = Kokkos::create_mirror_view(ysend_ref);

  for (int i2 = 0; i2 < n2; i2++) {
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
          double z  = i2 * dz;

          if (gi0 < n0 && li1 < n1) {
            h_xpencil(gi0, i1, i2) = std::cos(gx) * std::sin(ly) * std::sin(z);
            if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
              h_xsend_ref(i0, i1, i2, p) = h_xpencil(gi0, i1, i2);
            } else {
              h_xsend_ref(p, i0, i1, i2) = h_xpencil(gi0, i1, i2);
            }
          }
          if (li0 < n0 && gi1 < n1) {
            h_ypencil(i0, gi1, i2) = std::cos(lx) * std::sin(gy) * std::sin(z);
            if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
              h_ysend_ref(i0, i1, i2, p) = h_ypencil(i0, gi1, i2);
            } else {
              h_ysend_ref(p, i0, i1, i2) = h_ypencil(i0, gi1, i2);
            }
          }
        }
      }
    }
  }

  Kokkos::deep_copy(xpencil, h_xpencil);
  Kokkos::deep_copy(ypencil, h_ypencil);

  execution_space exec;
  pack(exec, xsend, xpencil, 0);
  pack(exec, ysend, ypencil, 1);

  auto h_xsend =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), xsend);
  auto h_ysend =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ysend);

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
}

template <typename T, typename LayoutType>
void test_pack_view3D_XtoZ(int rank, int nprocs) {
  using SrcView3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using DstView4DType = Kokkos::View<T****, LayoutType, execution_space>;

  const int n0 = 8, n1 = 15, n2 = 7;
  const int n0_local = ((n0 - 1) / nprocs) + 1;
  const int n2_local = ((n2 - 1) / nprocs) + 1;

  std::string rank_str = std::to_string(rank);
  SrcView3DType xpencil("xpencil" + rank_str, n0, n1, n2_local),
      zpencil("zpencil" + rank_str, n0_local, n1, n2);

  int n0_send = 0, n1_send = 0, n2_send = 0, n3_send = 0;
  if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    n0_send = n0_local;
    n1_send = n1;
    n2_send = n2_local;
    n3_send = nprocs;
  } else {
    n0_send = nprocs;
    n1_send = n0_local;
    n2_send = n1;
    n3_send = n2_local;
  }
  DstView4DType xsend("xsend", n0_send, n1_send, n2_send, n3_send),
      xsend_ref("xsend_ref", n0_send, n1_send, n2_send, n3_send);
  DstView4DType zsend("zsend", n0_send, n1_send, n2_send, n3_send),
      zsend_ref("zsend_ref", n0_send, n1_send, n2_send, n3_send);

  double dx = M_PI * 2.0 / static_cast<double>(n0);
  double dy = M_PI * 2.0 / static_cast<double>(n1);
  double dz = M_PI * 2.0 / static_cast<double>(n2);

  auto h_xpencil   = Kokkos::create_mirror_view(xpencil);
  auto h_zpencil   = Kokkos::create_mirror_view(zpencil);
  auto h_xsend_ref = Kokkos::create_mirror_view(xsend_ref);
  auto h_zsend_ref = Kokkos::create_mirror_view(zsend_ref);

  for (int i2 = 0; i2 < n2_local; i2++) {
    for (int i1 = 0; i1 < n1; i1++) {
      for (int i0 = 0; i0 < n0_local; i0++) {
        for (int p = 0; p < nprocs; p++) {
          int gi0   = i0 + n0_local * p;
          int li0   = i0 + n0_local * rank;
          int gi2   = i2 + n2_local * p;
          int li2   = i2 + n2_local * rank;
          double gx = gi0 * dx;
          double lx = li0 * dx;
          double gz = gi2 * dz;
          double lz = li2 * dz;
          double y  = i1 * dy;

          if (gi0 < n0 && li2 < n2) {
            h_xpencil(gi0, i1, i2) = std::cos(gx) * std::sin(y) * std::sin(lz);
            if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
              h_xsend_ref(i0, i1, i2, p) = h_xpencil(gi0, i1, i2);
            } else {
              h_xsend_ref(p, i0, i1, i2) = h_xpencil(gi0, i1, i2);
            }
          }
          if (li0 < n0 && gi2 < n2) {
            h_zpencil(i0, i1, gi2) = std::cos(lx) * std::sin(y) * std::sin(gz);
            if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
              h_zsend_ref(i0, i1, i2, p) = h_zpencil(i0, i1, gi2);
            } else {
              h_zsend_ref(p, i0, i1, i2) = h_zpencil(i0, i1, gi2);
            }
          }
        }
      }
    }
  }

  Kokkos::deep_copy(xpencil, h_xpencil);
  Kokkos::deep_copy(zpencil, h_zpencil);

  execution_space exec;
  pack(exec, xsend, xpencil, 0);
  pack(exec, zsend, zpencil, 2);

  auto h_xsend =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), xsend);
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

template <typename T, typename LayoutType>
void test_pack_view3D_YtoZ(int rank, int nprocs) {
  using SrcView3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using DstView4DType = Kokkos::View<T****, LayoutType, execution_space>;

  const int n0 = 8, n1 = 15, n2 = 7;
  const int n1_local = ((n1 - 1) / nprocs) + 1;
  const int n2_local = ((n2 - 1) / nprocs) + 1;

  std::string rank_str = std::to_string(rank);
  SrcView3DType ypencil("ypencil" + rank_str, n0, n1, n2_local),
      zpencil("zpencil" + rank_str, n0, n1_local, n2);

  int n0_send = 0, n1_send = 0, n2_send = 0, n3_send = 0;
  if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    n0_send = n0;
    n1_send = n1_local;
    n2_send = n2_local;
    n3_send = nprocs;
  } else {
    n0_send = nprocs;
    n1_send = n0;
    n2_send = n1_local;
    n3_send = n2_local;
  }
  DstView4DType ysend("ysend", n0_send, n1_send, n2_send, n3_send),
      ysend_ref("ysend_ref", n0_send, n1_send, n2_send, n3_send);
  DstView4DType zsend("zsend", n0_send, n1_send, n2_send, n3_send),
      zsend_ref("zsend_ref", n0_send, n1_send, n2_send, n3_send);

  double dx = M_PI * 2.0 / static_cast<double>(n0);
  double dy = M_PI * 2.0 / static_cast<double>(n1);
  double dz = M_PI * 2.0 / static_cast<double>(n2);

  auto h_ypencil   = Kokkos::create_mirror_view(ypencil);
  auto h_zpencil   = Kokkos::create_mirror_view(zpencil);
  auto h_ysend_ref = Kokkos::create_mirror_view(ysend_ref);
  auto h_zsend_ref = Kokkos::create_mirror_view(zsend_ref);

  for (int i2 = 0; i2 < n2_local; i2++) {
    for (int i1 = 0; i1 < n1_local; i1++) {
      for (int i0 = 0; i0 < n0; i0++) {
        for (int p = 0; p < nprocs; p++) {
          int gi1   = i1 + n1_local * p;
          int li1   = i1 + n1_local * rank;
          int gi2   = i2 + n2_local * p;
          int li2   = i2 + n2_local * rank;
          double gy = gi1 * dy;
          double ly = li1 * dy;
          double gz = gi2 * dz;
          double lz = li2 * dz;
          double x  = i0 * dx;

          if (gi1 < n1 && li2 < n2) {
            h_ypencil(i0, gi1, i2) = std::cos(x) * std::sin(gy) * std::sin(lz);
            if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
              h_ysend_ref(i0, i1, i2, p) = h_ypencil(i0, gi1, i2);
            } else {
              h_ysend_ref(p, i0, i1, i2) = h_ypencil(i0, gi1, i2);
            }
          }
          if (li1 < n1 && gi2 < n2) {
            h_zpencil(i0, i1, gi2) = std::cos(x) * std::sin(ly) * std::sin(gz);
            if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
              h_zsend_ref(i0, i1, i2, p) = h_zpencil(i0, i1, gi2);
            } else {
              h_zsend_ref(p, i0, i1, i2) = h_zpencil(i0, i1, gi2);
            }
          }
        }
      }
    }
  }

  Kokkos::deep_copy(ypencil, h_ypencil);
  Kokkos::deep_copy(zpencil, h_zpencil);

  execution_space exec;
  pack(exec, ysend, ypencil, 1);
  pack(exec, zsend, zpencil, 2);

  auto h_ysend =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ysend);
  auto h_zsend =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), zsend);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;

  // Check xsend is correct
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

TYPED_TEST(TestPack, View2D) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_pack_view2D<float_type, layout_type>(this->m_rank, this->m_nprocs);
}

TYPED_TEST(TestPack, View3D_XtoY) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_pack_view3D_XtoY<float_type, layout_type>(this->m_rank, this->m_nprocs);
}

TYPED_TEST(TestPack, View3D_XtoZ) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_pack_view3D_XtoZ<float_type, layout_type>(this->m_rank, this->m_nprocs);
}

TYPED_TEST(TestPack, View3D_YtoZ) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_pack_view3D_YtoZ<float_type, layout_type>(this->m_rank, this->m_nprocs);
}
