#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "All2All.hpp"
#include "Test_Utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using test_types      = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>,
                                    std::pair<double, Kokkos::LayoutLeft>,
                                    std::pair<double, Kokkos::LayoutRight>>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct TestAll2All : public ::testing::Test {
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
void test_all2all_view2D_XtoY_Left(int rank, int nprocs, int direction) {
  using View3DType = Kokkos::View<T***, Kokkos::LayoutLeft, execution_space>;

  const int n0 = 4, n1 = 3;
  const int n0_local = ((n0 - 1) / nprocs) + 1;
  const int n1_local = ((n1 - 1) / nprocs) + 1;
  const int n1_pad   = n1_local * nprocs;

  std::string rank_str = std::to_string(rank);
  View3DType send("send" + rank_str, n0_local, n1_local, nprocs),
      recv("recv" + rank_str, n0_local, n1_local, nprocs),
      ref("ref" + rank_str, n0_local, n1_local, nprocs);

  double dx = M_PI * 2.0 / static_cast<double>(n0);
  double dy = M_PI * 2.0 / static_cast<double>(n1);

  auto h_send = Kokkos::create_mirror_view(send);
  auto h_ref  = Kokkos::create_mirror_view(ref);
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

        if (direction == 1) {
          double send_tmp =
              gi0 < n0 && gi1 < n1 ? std::cos(gx) * std::sin(ly) : 0.0;
          double ref_tmp =
              gi0 < n0 && gi1 < n1 ? std::cos(lx) * std::sin(gy) : 0.0;
          h_send(i0, i1, p) = send_tmp;
          h_ref(i0, i1, p)  = ref_tmp;
        } else {
          double send_tmp =
              gi0 < n0 && gi1 < n1 ? std::cos(lx) * std::sin(gy) : 0.0;
          double ref_tmp =
              gi0 < n0 && gi1 < n1_pad ? std::cos(gx) * std::sin(ly) : 0.0;

          h_send(i0, i1, p) = send_tmp;
          h_ref(i0, i1, p)  = ref_tmp;
        }
      }
    }
  }

  Kokkos::deep_copy(send, h_send);
  Kokkos::deep_copy(ref, h_ref);

  All2All<execution_space, View3DType> all2all(send, recv);
  all2all(send, recv);

  auto h_recv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), recv);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;
  for (int i1 = 0; i1 < n1_local; i1++) {
    for (int i0 = 0; i0 < n0_local; i0++) {
      for (int p = 0; p < nprocs; p++) {
        auto diff = Kokkos::abs(h_recv(i0, i1, p) - h_ref(i0, i1, p));
        EXPECT_LE(diff, epsilon);
      }
    }
  }
}

template <typename T>
void test_all2all_view2D_XtoY_Right(int rank, int nprocs, int direction) {
  using View3DType = Kokkos::View<T***, Kokkos::LayoutRight, execution_space>;

  const int n0 = 4, n1 = 3;
  const int n0_local = ((n0 - 1) / nprocs) + 1;
  const int n1_local = ((n1 - 1) / nprocs) + 1;
  const int n1_pad   = n1_local * nprocs;

  View3DType send("send", nprocs, n0_local, n1_local),
      recv("recv", nprocs, n0_local, n1_local),
      ref("ref", nprocs, n0_local, n1_local);

  double dx = M_PI * 2.0 / static_cast<double>(n0);
  double dy = M_PI * 2.0 / static_cast<double>(n1);

  auto h_send = Kokkos::create_mirror_view(send);
  auto h_ref  = Kokkos::create_mirror_view(ref);
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

        if (direction == 1) {
          double send_tmp =
              gi0 < n0 && gi1 < n1 ? std::cos(gx) * std::sin(ly) : 0.0;
          double ref_tmp =
              gi0 < n0 && gi1 < n1 ? std::cos(lx) * std::sin(gy) : 0.0;

          h_send(p, i0, i1) = send_tmp;
          h_ref(p, i0, i1)  = ref_tmp;
        } else {
          double send_tmp =
              gi0 < n0 && gi1 < n1 ? std::cos(lx) * std::sin(gy) : 0.0;
          double ref_tmp =
              gi0 < n0 && gi1 < n1_pad ? std::cos(gx) * std::sin(ly) : 0.0;

          h_send(p, i0, i1) = send_tmp;
          h_ref(p, i0, i1)  = ref_tmp;
        }
      }
    }
  }

  Kokkos::deep_copy(send, h_send);
  Kokkos::deep_copy(ref, h_ref);

  All2All<execution_space, View3DType> all2all(send, recv);
  all2all(send, recv);

  auto h_recv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), recv);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;
  for (int i1 = 0; i1 < n1_local; i1++) {
    for (int i0 = 0; i0 < n0_local; i0++) {
      for (int p = 0; p < nprocs; p++) {
        auto diff = Kokkos::abs(h_recv(p, i0, i1) - h_ref(p, i0, i1));
        EXPECT_LE(diff, epsilon);
      }
    }
  }
}

template <typename T>
void test_all2all_view3D_XtoY_Left(int rank, int nprocs, int direction) {
  using View4DType = Kokkos::View<T****, Kokkos::LayoutLeft, execution_space>;

  const int n0 = 4, n1 = 3, n2 = 5;
  const int n0_local = ((n0 - 1) / nprocs) + 1;
  const int n1_local = ((n1 - 1) / nprocs) + 1;
  const int n1_pad   = n1_local * nprocs;

  View4DType send("send", n0_local, n1_local, n2, nprocs),
      recv("recv", n0_local, n1_local, n2, nprocs),
      ref("ref", n0_local, n1_local, n2, nprocs);

  double dx = M_PI * 2.0 / static_cast<double>(n0);
  double dy = M_PI * 2.0 / static_cast<double>(n1);
  double dz = M_PI * 2.0 / static_cast<double>(n2);

  auto h_send = Kokkos::create_mirror_view(send);
  auto h_ref  = Kokkos::create_mirror_view(ref);
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

          if (direction == 1) {
            double send_tmp = gi0 < n0 && gi1 < n1
                                  ? std::cos(gx) * std::sin(ly) * std::sin(z)
                                  : 0.0;
            double ref_tmp  = gi0 < n0 && gi1 < n1
                                  ? std::cos(lx) * std::sin(gy) * std::sin(z)
                                  : 0.0;

            h_send(i0, i1, i2, p) = send_tmp;
            h_ref(i0, i1, i2, p)  = ref_tmp;
          } else {
            double send_tmp = gi0 < n0 && gi1 < n1
                                  ? std::cos(lx) * std::sin(gy) * std::sin(z)
                                  : 0.0;
            double ref_tmp  = gi0 < n0 && gi1 < n1_pad
                                  ? std::cos(gx) * std::sin(ly) * std::sin(z)
                                  : 0.0;

            h_send(i0, i1, i2, p) = send_tmp;
            h_ref(i0, i1, i2, p)  = ref_tmp;
          }
        }
      }
    }
  }

  Kokkos::deep_copy(send, h_send);
  Kokkos::deep_copy(ref, h_ref);

  All2All<execution_space, View4DType> all2all(send, recv);
  all2all(send, recv);

  auto h_recv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), recv);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;
  for (int i2 = 0; i2 < n2; i2++) {
    for (int i1 = 0; i1 < n1_local; i1++) {
      for (int i0 = 0; i0 < n0_local; i0++) {
        for (int p = 0; p < nprocs; p++) {
          auto diff = Kokkos::abs(h_recv(i0, i1, i2, p) - h_ref(i0, i1, i2, p));
          EXPECT_LE(diff, epsilon);
        }
      }
    }
  }
}

template <typename T>
void test_all2all_view3D_XtoY_Right(int rank, int nprocs, int direction) {
  using View4DType = Kokkos::View<T****, Kokkos::LayoutRight, execution_space>;

  const int n0 = 4, n1 = 3, n2 = 5;
  const int n0_local = ((n0 - 1) / nprocs) + 1;
  const int n1_local = ((n1 - 1) / nprocs) + 1;
  const int n1_pad   = n1_local * nprocs;

  View4DType send("send", nprocs, n0_local, n1_local, n2),
      recv("recv", nprocs, n0_local, n1_local, n2),
      ref("ref", nprocs, n0_local, n1_local, n2);

  double dx = M_PI * 2.0 / static_cast<double>(n0);
  double dy = M_PI * 2.0 / static_cast<double>(n1);
  double dz = M_PI * 2.0 / static_cast<double>(n2);

  auto h_send = Kokkos::create_mirror_view(send);
  auto h_ref  = Kokkos::create_mirror_view(ref);
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

          if (direction == 1) {
            double send_tmp = gi0 < n0 && gi1 < n1
                                  ? std::cos(gx) * std::sin(ly) * std::sin(z)
                                  : 0.0;
            double ref_tmp  = gi0 < n0 && gi1 < n1_pad
                                  ? std::cos(lx) * std::sin(gy) * std::sin(z)
                                  : 0.0;

            h_send(p, i0, i1, i2) = send_tmp;
            h_ref(p, i0, i1, i2)  = ref_tmp;
          } else {
            double send_tmp = gi0 < n0 && gi1 < n1
                                  ? std::cos(lx) * std::sin(gy) * std::sin(z)
                                  : 0.0;
            double ref_tmp  = gi0 < n0 && gi1 < n1_pad
                                  ? std::cos(gx) * std::sin(ly) * std::sin(z)
                                  : 0.0;

            h_send(p, i0, i1, i2) = send_tmp;
            h_ref(p, i0, i1, i2)  = ref_tmp;
          }
        }
      }
    }
  }

  Kokkos::deep_copy(send, h_send);
  Kokkos::deep_copy(ref, h_ref);

  All2All<execution_space, View4DType> all2all(send, recv);
  all2all(send, recv);

  auto h_recv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), recv);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;
  for (int i2 = 0; i2 < n2; i2++) {
    for (int i1 = 0; i1 < n1_local; i1++) {
      for (int i0 = 0; i0 < n0_local; i0++) {
        for (int p = 0; p < nprocs; p++) {
          auto diff = Kokkos::abs(h_recv(p, i0, i1, i2) - h_ref(p, i0, i1, i2));
          EXPECT_LE(diff, epsilon);
        }
      }
    }
  }
}

template <typename T>
void test_all2all_view3D_XtoZ_Left(int rank, int nprocs, int direction) {
  using View4DType = Kokkos::View<T****, Kokkos::LayoutLeft, execution_space>;

  const int n0 = 4, n1 = 3, n2 = 5;
  const int n0_local = ((n0 - 1) / nprocs) + 1;
  const int n2_local = ((n2 - 1) / nprocs) + 1;
  const int n2_pad   = n2_local * nprocs;

  View4DType send("send", n0_local, n1, n2_local, nprocs),
      recv("recv", n0_local, n1, n2_local, nprocs),
      ref("ref", n0_local, n1, n2_local, nprocs);

  double dx = M_PI * 2.0 / static_cast<double>(n0);
  double dy = M_PI * 2.0 / static_cast<double>(n1);
  double dz = M_PI * 2.0 / static_cast<double>(n2);

  auto h_send = Kokkos::create_mirror_view(send);
  auto h_ref  = Kokkos::create_mirror_view(ref);
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

          if (direction == 1) {
            double send_tmp = gi0 < n0 && gi2 < n2
                                  ? std::cos(gx) * std::sin(y) * std::sin(lz)
                                  : 0.0;
            double ref_tmp  = gi0 < n0 && gi2 < n2_pad
                                  ? std::cos(lx) * std::sin(y) * std::sin(gz)
                                  : 0.0;

            h_send(i0, i1, i2, p) = send_tmp;
            h_ref(i0, i1, i2, p)  = ref_tmp;
          } else {
            double send_tmp = gi0 < n0 && gi2 < n2
                                  ? std::cos(lx) * std::sin(y) * std::sin(gz)
                                  : 0.0;
            double ref_tmp  = gi0 < n0 && gi2 < n2_pad
                                  ? std::cos(gx) * std::sin(y) * std::sin(lz)
                                  : 0.0;

            h_send(i0, i1, i2, p) = send_tmp;
            h_ref(i0, i1, i2, p)  = ref_tmp;
          }
        }
      }
    }
  }

  Kokkos::deep_copy(send, h_send);
  Kokkos::deep_copy(ref, h_ref);

  All2All<execution_space, View4DType> all2all(send, recv);
  all2all(send, recv);

  auto h_recv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), recv);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;
  for (int i2 = 0; i2 < n2_local; i2++) {
    for (int i1 = 0; i1 < n1; i1++) {
      for (int i0 = 0; i0 < n0_local; i0++) {
        for (int p = 0; p < nprocs; p++) {
          auto diff = Kokkos::abs(h_recv(i0, i1, i2, p) - h_ref(i0, i1, i2, p));
          EXPECT_LE(diff, epsilon);
        }
      }
    }
  }
}

template <typename T>
void test_all2all_view3D_XtoZ_Right(int rank, int nprocs, int direction) {
  using View4DType = Kokkos::View<T****, Kokkos::LayoutRight, execution_space>;

  const int n0 = 4, n1 = 3, n2 = 5;
  const int n0_local = ((n0 - 1) / nprocs) + 1;
  const int n2_local = ((n2 - 1) / nprocs) + 1;
  const int n2_pad   = n2_local * nprocs;

  View4DType send("send", nprocs, n0_local, n1, n2_local),
      recv("recv", nprocs, n0_local, n1, n2_local),
      ref("ref", nprocs, n0_local, n1, n2_local);

  double dx = M_PI * 2.0 / static_cast<double>(n0);
  double dy = M_PI * 2.0 / static_cast<double>(n1);
  double dz = M_PI * 2.0 / static_cast<double>(n2);

  auto h_send = Kokkos::create_mirror_view(send);
  auto h_ref  = Kokkos::create_mirror_view(ref);
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
          if (direction == 1) {
            double send_tmp = gi0 < n0 && gi2 < n2
                                  ? std::cos(gx) * std::sin(y) * std::sin(lz)
                                  : 0.0;
            double ref_tmp  = gi0 < n0 && gi2 < n2_pad
                                  ? std::cos(lx) * std::sin(y) * std::sin(gz)
                                  : 0.0;

            h_send(p, i0, i1, i2) = send_tmp;
            h_ref(p, i0, i1, i2)  = ref_tmp;
          } else {
            double send_tmp = gi0 < n0 && gi2 < n2
                                  ? std::cos(lx) * std::sin(y) * std::sin(gz)
                                  : 0.0;
            double ref_tmp  = gi0 < n0 && gi2 < n2_pad
                                  ? std::cos(gx) * std::sin(y) * std::sin(lz)
                                  : 0.0;

            h_send(p, i0, i1, i2) = send_tmp;
            h_ref(p, i0, i1, i2)  = ref_tmp;
          }
        }
      }
    }
  }

  Kokkos::deep_copy(send, h_send);
  Kokkos::deep_copy(ref, h_ref);

  All2All<execution_space, View4DType> all2all(send, recv);
  all2all(send, recv);

  auto h_recv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), recv);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;
  for (int i2 = 0; i2 < n2_local; i2++) {
    for (int i1 = 0; i1 < n1; i1++) {
      for (int i0 = 0; i0 < n0_local; i0++) {
        for (int p = 0; p < nprocs; p++) {
          auto diff = Kokkos::abs(h_recv(p, i0, i1, i2) - h_ref(p, i0, i1, i2));
          EXPECT_LE(diff, epsilon);
        }
      }
    }
  }
}

template <typename T>
void test_all2all_view3D_YtoZ_Left(int rank, int nprocs, int direction) {
  using View4DType = Kokkos::View<T****, Kokkos::LayoutLeft, execution_space>;

  const int n0 = 4, n1 = 3, n2 = 5;
  const int n1_local = ((n1 - 1) / nprocs) + 1;
  const int n2_local = ((n2 - 1) / nprocs) + 1;
  const int n1_pad   = n1_local * nprocs;
  const int n2_pad   = n2_local * nprocs;

  View4DType send("send", n0, n1_local, n2_local, nprocs),
      recv("recv", n0, n1_local, n2_local, nprocs),
      ref("ref", n0, n1_local, n2_local, nprocs);

  double dx = M_PI * 2.0 / static_cast<double>(n0);
  double dy = M_PI * 2.0 / static_cast<double>(n1);
  double dz = M_PI * 2.0 / static_cast<double>(n2);

  auto h_send = Kokkos::create_mirror_view(send);
  auto h_ref  = Kokkos::create_mirror_view(ref);
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

          if (direction == 1) {
            double send_tmp = gi1 < n1 && gi2 < n2
                                  ? std::cos(x) * std::sin(gy) * std::sin(lz)
                                  : 0.0;
            double ref_tmp  = gi1 < n1_pad && gi2 < n2_pad
                                  ? std::cos(x) * std::sin(ly) * std::sin(gz)
                                  : 0.0;

            h_send(i0, i1, i2, p) = send_tmp;
            h_ref(i0, i1, i2, p)  = ref_tmp;
          } else {
            double send_tmp = gi1 < n1 && gi2 < n2
                                  ? std::cos(x) * std::sin(ly) * std::sin(gz)
                                  : 0.0;
            double ref_tmp  = gi1 < n1_pad && gi2 < n2_pad
                                  ? std::cos(x) * std::sin(gy) * std::sin(lz)
                                  : 0.0;

            h_send(i0, i1, i2, p) = send_tmp;
            h_ref(i0, i1, i2, p)  = ref_tmp;
          }
        }
      }
    }
  }

  Kokkos::deep_copy(send, h_send);
  Kokkos::deep_copy(ref, h_ref);

  All2All<execution_space, View4DType> all2all(send, recv);
  all2all(send, recv);

  auto h_recv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), recv);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;
  for (int i2 = 0; i2 < n2_local; i2++) {
    for (int i1 = 0; i1 < n1_local; i1++) {
      for (int i0 = 0; i0 < n0; i0++) {
        for (int p = 0; p < nprocs; p++) {
          auto diff = Kokkos::abs(h_recv(i0, i1, i2, p) - h_ref(i0, i1, i2, p));
          EXPECT_LE(diff, epsilon);
        }
      }
    }
  }
}

template <typename T>
void test_all2all_view3D_YtoZ_Right(int rank, int nprocs, int direction) {
  using View4DType = Kokkos::View<T****, Kokkos::LayoutRight, execution_space>;

  const int n0 = 4, n1 = 3, n2 = 5;
  const int n1_local = ((n1 - 1) / nprocs) + 1;
  const int n2_local = ((n2 - 1) / nprocs) + 1;
  const int n1_pad   = n1_local * nprocs;
  const int n2_pad   = n2_local * nprocs;

  View4DType send("send", nprocs, n0, n1_local, n2_local),
      recv("recv", nprocs, n0, n1_local, n2_local),
      ref("ref", nprocs, n0, n1_local, n2_local);

  double dx = M_PI * 2.0 / static_cast<double>(n0);
  double dy = M_PI * 2.0 / static_cast<double>(n1);
  double dz = M_PI * 2.0 / static_cast<double>(n2);

  auto h_send = Kokkos::create_mirror_view(send);
  auto h_ref  = Kokkos::create_mirror_view(ref);
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
          if (direction == 1) {
            double send_tmp = gi1 < n1 && gi2 < n2
                                  ? std::cos(x) * std::sin(gy) * std::sin(lz)
                                  : 0.0;
            double ref_tmp  = gi1 < n1_pad && gi2 < n2_pad
                                  ? std::cos(x) * std::sin(ly) * std::sin(gz)
                                  : 0.0;

            h_send(p, i0, i1, i2) = send_tmp;
            h_ref(p, i0, i1, i2)  = ref_tmp;
          } else {
            double send_tmp = gi1 < n1 && gi2 < n2
                                  ? std::cos(x) * std::sin(ly) * std::sin(gz)
                                  : 0.0;
            double ref_tmp  = gi1 < n1_pad && gi2 < n2_pad
                                  ? std::cos(x) * std::sin(gy) * std::sin(lz)
                                  : 0.0;

            h_send(p, i0, i1, i2) = send_tmp;
            h_ref(p, i0, i1, i2)  = ref_tmp;
          }
        }
      }
    }
  }

  Kokkos::deep_copy(send, h_send);
  Kokkos::deep_copy(ref, h_ref);

  All2All<execution_space, View4DType> all2all(send, recv);
  all2all(send, recv);

  auto h_recv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), recv);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;
  for (int i2 = 0; i2 < n2_local; i2++) {
    for (int i1 = 0; i1 < n1_local; i1++) {
      for (int i0 = 0; i0 < n0; i0++) {
        for (int p = 0; p < nprocs; p++) {
          auto diff = Kokkos::abs(h_recv(p, i0, i1, i2) - h_ref(p, i0, i1, i2));
          EXPECT_LE(diff, epsilon);
        }
      }
    }
  }
}

}  // namespace

TYPED_TEST_SUITE(TestAll2All, test_types);

TYPED_TEST(TestAll2All, View2D_XtoY) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  if constexpr (std::is_same_v<layout_type, Kokkos::LayoutLeft>) {
    test_all2all_view2D_XtoY_Left<float_type>(this->m_rank, this->m_nprocs, 1);
  } else {
    test_all2all_view2D_XtoY_Right<float_type>(this->m_rank, this->m_nprocs, 1);
  }
}

TYPED_TEST(TestAll2All, View2D_YtoX) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  if constexpr (std::is_same_v<layout_type, Kokkos::LayoutLeft>) {
    test_all2all_view2D_XtoY_Left<float_type>(this->m_rank, this->m_nprocs, -1);
  } else {
    test_all2all_view2D_XtoY_Right<float_type>(this->m_rank, this->m_nprocs,
                                               -1);
  }
}

TYPED_TEST(TestAll2All, View3D_XtoY) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  if constexpr (std::is_same_v<layout_type, Kokkos::LayoutLeft>) {
    test_all2all_view3D_XtoY_Left<float_type>(this->m_rank, this->m_nprocs, 1);
  } else {
    test_all2all_view3D_XtoY_Right<float_type>(this->m_rank, this->m_nprocs, 1);
  }
}

TYPED_TEST(TestAll2All, View3D_YtoX) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  if constexpr (std::is_same_v<layout_type, Kokkos::LayoutLeft>) {
    test_all2all_view3D_XtoY_Left<float_type>(this->m_rank, this->m_nprocs, -1);
  } else {
    test_all2all_view3D_XtoY_Right<float_type>(this->m_rank, this->m_nprocs,
                                               -1);
  }
}

TYPED_TEST(TestAll2All, View3D_XtoZ) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  if constexpr (std::is_same_v<layout_type, Kokkos::LayoutLeft>) {
    test_all2all_view3D_XtoZ_Left<float_type>(this->m_rank, this->m_nprocs, 1);
  } else {
    test_all2all_view3D_XtoZ_Right<float_type>(this->m_rank, this->m_nprocs, 1);
  }
}

TYPED_TEST(TestAll2All, View3D_ZtoX) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  if constexpr (std::is_same_v<layout_type, Kokkos::LayoutLeft>) {
    test_all2all_view3D_XtoZ_Left<float_type>(this->m_rank, this->m_nprocs, -1);
  } else {
    test_all2all_view3D_XtoZ_Right<float_type>(this->m_rank, this->m_nprocs,
                                               -1);
  }
}

TYPED_TEST(TestAll2All, View3D_YtoZ) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  if constexpr (std::is_same_v<layout_type, Kokkos::LayoutLeft>) {
    test_all2all_view3D_YtoZ_Left<float_type>(this->m_rank, this->m_nprocs, 1);
  } else {
    test_all2all_view3D_YtoZ_Right<float_type>(this->m_rank, this->m_nprocs, 1);
  }
}

TYPED_TEST(TestAll2All, View3D_ZtoY) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  if constexpr (std::is_same_v<layout_type, Kokkos::LayoutLeft>) {
    test_all2all_view3D_YtoZ_Left<float_type>(this->m_rank, this->m_nprocs, -1);
  } else {
    test_all2all_view3D_YtoZ_Right<float_type>(this->m_rank, this->m_nprocs,
                                               -1);
  }
}
