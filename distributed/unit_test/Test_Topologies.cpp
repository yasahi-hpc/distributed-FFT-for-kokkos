#include <mpi.h>
#include <gtest/gtest.h>
#include <iostream>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_Topologies.hpp"
#include "Test_Utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
class TopologyParamTests : public ::testing::TestWithParam<int> {};
class SlabParamTests : public ::testing::TestWithParam<int> {};
class PencilParamTests : public ::testing::TestWithParam<int> {};

/// \brief Convert topology type to string for better test failure messages.
/// \param[in] type The topology type to convert.
/// \return The string representation of the topology type.
inline std::string topology_type_to_string(
    KokkosFFT::Distributed::Impl::TopologyType type) {
  using KokkosFFT::Distributed::Impl::TopologyType;
  switch (type) {
    case TopologyType::Empty: return "Empty";
    case TopologyType::Shared: return "Shared";
    case TopologyType::Slab: return "Slab";
    case TopologyType::Pencil: return "Pencil";
    case TopologyType::Brick: return "Brick";
    case TopologyType::Invalid: return "Invalid";
    default: return "Unknown";
  }
}

/// \brief Generate error message for topology type test failures.
/// \tparam TopologyType The type of the topology input.
/// \param[in] topology The input topology that caused the failure.
/// \param[in] ref The expected topology type that should have been returned.
/// \return Error message including the input topology, expected topology type,
/// and actual topology type.
template <typename TopologyType>
std::string error_to_topology_type(
    const TopologyType& topology,
    KokkosFFT::Distributed::Impl::TopologyType ref) {
  std::string msg;
  msg += "Input topology: (";
  msg += std::to_string(topology.at(0));
  for (std::size_t i = 1; i < topology.size(); ++i) {
    msg += ", " + std::to_string(topology.at(i));
  }
  msg += "), should be: " + topology_type_to_string(ref) + ", but got: " +
         topology_type_to_string(
             KokkosFFT::Distributed::Impl::to_topology_type(topology));
  return msg;
}

/// \brief Generate error message for get_common_topology_type test failures.
/// \tparam Topology1Type The type of the first topology input.
/// \tparam Topology2Type The type of the second topology input.
/// \param[in] topo1 The first input topology that caused the failure.
/// \param[in] topo2 The second input topology that caused the failure.
/// \param[in] ref The expected common topology type that should have been
/// returned.
/// \return Error message including the input topologies, expected common
/// topology type, and actual common topology type.
template <typename Topology1Type, typename Topology2Type>
std::string error_get_common_topology_type(
    const Topology1Type& topo1, const Topology2Type& topo2,
    KokkosFFT::Distributed::Impl::TopologyType ref) {
  std::string msg;
  msg += "Input topologies: ";
  msg += "(" + std::to_string(topo1.at(0));
  for (std::size_t i = 1; i < topo1.size(); ++i) {
    msg += ", " + std::to_string(topo1.at(i));
  }
  msg += ") and (";
  msg += std::to_string(topo2.at(0));
  for (std::size_t i = 1; i < topo2.size(); ++i) {
    msg += ", " + std::to_string(topo2.at(i));
  }
  msg +=
      "), should be: " + topology_type_to_string(ref) + ", but got: " +
      topology_type_to_string(
          KokkosFFT::Distributed::Impl::get_common_topology_type(topo1, topo2));
  return msg;
}

/// \brief Generate error message for is_topology test failures.
/// \tparam TopologyType The type of the topology input.
/// \param[in] topology The input topology that caused the failure.
/// \param[in] specified The topology type that was expected to be identified or
/// not identified.
/// \param[in] expected Whether the input topology was expected to be identified
/// as the specified topology type.
/// \return Error message including the input topology, specified topology type,
/// and whether it was expected to be identified.
template <typename TopologyType>
std::string error_is_topology(
    const TopologyType& topology,
    KokkosFFT::Distributed::Impl::TopologyType specified, bool expected) {
  std::string msg;
  msg += "Input topology: (";
  msg += std::to_string(topology.at(0));
  for (std::size_t i = 1; i < topology.size(); ++i) {
    msg += ", " + std::to_string(topology.at(i));
  }
  if (expected) {
    msg += "), should be identified as " + topology_type_to_string(specified) +
           ", but it is not.";
  } else {
    msg += "), should not be identified as " +
           topology_type_to_string(specified) + ", but it is.";
  }
  return msg;
}

/// \brief Generate error message for are_topologies test failures.
/// \tparam Topology1Type The type of the first topology input.
/// \tparam Topology2Type The type of the second topology input.
/// \param[in] topo1 The first input topology that caused the failure.
/// \param[in] topo2 The second input topology that caused the failure.
/// \param[in] specified The topology type that was expected to be identified or
/// not identified.
/// \param[in] expected Whether the input topologies were expected to be
/// identified as the specified topology type.
/// \return Error message including the input topologies, specified topology
/// type, and whether they were expected to be identified.
template <typename Topology1Type, typename Topology2Type>
std::string error_are_topologies(
    const Topology1Type& topo1, const Topology2Type& topo2,
    KokkosFFT::Distributed::Impl::TopologyType specified, bool expected) {
  std::string msg;
  msg += "Input topologies: ";
  msg += "(" + std::to_string(topo1.at(0));
  for (std::size_t i = 1; i < topo1.size(); ++i) {
    msg += ", " + std::to_string(topo1.at(i));
  }
  msg += ") and (";
  msg += std::to_string(topo2.at(0));
  for (std::size_t i = 1; i < topo2.size(); ++i) {
    msg += ", " + std::to_string(topo2.at(i));
  }
  if (expected) {
    msg += "), should be identified as " + topology_type_to_string(specified) +
           ", but it is not.";
  } else {
    msg += "), should not be identified as " +
           topology_type_to_string(specified) + ", but it is.";
  }
  return msg;
}

/// \brief Generate error message for in_out_axes test failures.
/// \tparam Topology1Type The type of the first topology input.
/// \tparam Topology2Type The type of the second topology input.
/// \param[in] topo1 The first input topology that caused the failure.
/// \param[in] topo2 The second input topology that caused the failure.
/// \param[in] expected The expected in/out axes.
/// \return Error message including the input topologies and the expected in/out
/// axes.
template <typename Topology1Type, typename Topology2Type>
std::string error_in_out_axes(const Topology1Type& topo1,
                              const Topology2Type& topo2,
                              std::tuple<std::size_t, std::size_t> expected) {
  auto actual = KokkosFFT::Distributed::Impl::slab_in_out_axes(topo1, topo2);
  std::string msg;
  msg += "Input topologies: ";
  msg += "(" + std::to_string(topo1.at(0));
  for (std::size_t i = 1; i < topo1.size(); ++i) {
    msg += ", " + std::to_string(topo1.at(i));
  }
  msg += ") and (";
  msg += std::to_string(topo2.at(0));
  for (std::size_t i = 1; i < topo2.size(); ++i) {
    msg += ", " + std::to_string(topo2.at(i));
  }
  msg += ")";
  msg += "), should have in/out axes: (" +
         std::to_string(std::get<0>(expected)) + ", " +
         std::to_string(std::get<1>(expected)) + "), but got: (" +
         std::to_string(std::get<0>(actual)) + ", " +
         std::to_string(std::get<1>(actual)) + ").";
  return msg;
}

/// \brief Generate error message for mid_topology test failures.
/// \tparam TopologyType The type of the topology input.
/// \param[in] topo1 The first input topology that caused the failure.
/// \param[in] topo2 The second input topology that caused the failure.
/// \param[in] expected The expected topology
/// \return Error message including the input topologies and the expected
/// topology
template <typename TopologyType>
std::string error_mid_topology(const TopologyType& topo1,
                               const TopologyType& topo2,
                               const TopologyType& expected) {
  auto actual = KokkosFFT::Distributed::Impl::propose_mid_array(topo1, topo2);
  std::string msg;
  msg += "Input topologies: ";
  msg += "(" + std::to_string(topo1.at(0));
  for (std::size_t i = 1; i < topo1.size(); ++i) {
    msg += ", " + std::to_string(topo1.at(i));
  }
  msg += ") and (";
  msg += std::to_string(topo2.at(0));
  for (std::size_t i = 1; i < topo2.size(); ++i) {
    msg += ", " + std::to_string(topo2.at(i));
  }
  msg += "), should have a mid topology: (" + std::to_string(expected.at(0));
  for (std::size_t i = 1; i < expected.size(); ++i) {
    msg += ", " + std::to_string(expected.at(i));
  }
  msg += "), but got (" + std::to_string(actual.at(0));
  for (std::size_t i = 1; i < actual.size(); ++i) {
    msg += ", " + std::to_string(actual.at(i));
  }
  msg += ").";
  return msg;
}

/// \brief Generate error message for decompose_axes test failures.
/// \tparam iType The index type used for the topology.
/// \tparam DIM The dimensionality of the topology.
/// \tparam FFT_DIM The dimensionality of the FFT axes.
/// \param[in] topologies The input topologies that caused the failure.
/// \param[in] axes The axes along which the FFT is performed.
/// \param[in] expected The expected decomposed axes
/// \return Error message including the input topologies, FFT axes, and the
/// expected decomposition
template <typename iType, std::size_t DIM, std::size_t FFT_DIM>
std::string error_decompose_axes(
    const std::vector<std::array<std::size_t, DIM>>& topologies,
    const std::array<iType, FFT_DIM>& axes,
    const std::vector<std::vector<iType>>& expected) {
  auto actual = KokkosFFT::Distributed::Impl::decompose_axes(topologies, axes);
  std::string msg;
  msg += "Input topologies: ";
  msg += "(";
  for (std::size_t i = 0; i < topologies.size(); ++i) {
    msg += "(" + std::to_string(topologies.at(i).at(0));
    for (std::size_t j = 1; j < topologies.at(i).size(); ++j) {
      msg += ", " + std::to_string(topologies.at(i).at(j));
    }
    msg += ")";
    if (i != topologies.size() - 1) {
      msg += " and ";
    }
  }
  msg += "), with FFT axes: (";
  msg += std::to_string(axes.at(0));
  for (std::size_t i = 1; i < axes.size(); ++i) {
    msg += ", " + std::to_string(axes.at(i));
  }
  msg += "), should have decomposed axes: (";
  for (std::size_t i = 0; i < expected.size(); ++i) {
    msg += "(";
    for (std::size_t j = 0; j < expected.at(i).size(); ++j) {
      msg += std::to_string(expected.at(i).at(j));
      if (j != expected.at(i).size() - 1) {
        msg += ", ";
      }
    }
    msg += ")";
    if (i != expected.size() - 1) {
      msg += " and ";
    }
  }
  msg += ").";

  return msg;
}

/// \brief Generate error message for compute_trans_axis test failures.
/// \tparam iType The index type
/// \tparam DIM The dimension
/// \param[in] in_topology The input topology
/// \param[in] out_topology The output topology
/// \param[in] first_non_one The first non-one element in the input or output
/// \param[in] expected The expected transformation axis (0 or 1)
/// \return Error message including the input topologies and the expected in/out
/// axes.
template <typename iType, std::size_t DIM>
std::string error_trans_axis(const std::array<iType, DIM>& in_topology,
                             const std::array<iType, DIM>& out_topology,
                             iType first_non_one, iType expected) {
  auto actual = KokkosFFT::Distributed::Impl::compute_trans_axis(
      in_topology, out_topology, first_non_one);
  std::string msg;
  msg += "Input topologies: ";
  msg += "(" + std::to_string(in_topology.at(0));
  for (std::size_t i = 1; i < in_topology.size(); ++i) {
    msg += ", " + std::to_string(in_topology.at(i));
  }
  msg += ") and (";
  msg += std::to_string(out_topology.at(0));
  for (std::size_t i = 1; i < out_topology.size(); ++i) {
    msg += ", " + std::to_string(out_topology.at(i));
  }
  msg += "), should have trans_axis: " + std::to_string(expected) +
         ", but got: " + std::to_string(actual) + ".";
  return msg;
}

/// \brief Generate error message for get_all_slab_topologies and
/// get_all_pencil_topologies test failures.
/// \tparam iType The index type used for the topology.
/// \tparam DIM The dimensionality of the topology.
/// \tparam FFT_DIM The dimensionality of the FFT axes.
/// \param[in] in_topology The input topology.
/// \param[out] out_topology The output topology.
/// \param[in] axes The axes along which the FFT is performed.
/// \param[in] actual The computed vector of all possible topologies that can be
/// formed
/// \param[in] expected The expected vector of all possible topologies that can
/// be formed
/// \return Error message including the input topologies, FFT axes, and the
/// expected vector of topologies
template <typename iType, std::size_t DIM, std::size_t FFT_DIM>
std::string error_all_topologies(
    const std::array<std::size_t, DIM>& in_topology,
    const std::array<std::size_t, DIM>& out_topology,
    const std::array<iType, FFT_DIM>& axes,
    const std::vector<std::array<std::size_t, DIM>>& actual,
    const std::vector<std::array<std::size_t, DIM>>& expected) {
  std::string msg;
  msg += "Input topologies: ";
  msg += "(";
  for (std::size_t i = 0; i < in_topology.size(); ++i) {
    msg += std::to_string(in_topology.at(i));
    if (i != in_topology.size() - 1) {
      msg += ", ";
    }
  }
  msg += ") and (";
  for (std::size_t i = 0; i < out_topology.size(); ++i) {
    msg += std::to_string(out_topology.at(i));
    if (i != out_topology.size() - 1) {
      msg += ", ";
    }
  }
  msg += "), with FFT axes: (";
  msg += std::to_string(axes.at(0));
  for (std::size_t i = 1; i < axes.size(); ++i) {
    msg += ", " + std::to_string(axes.at(i));
  }
  msg += "), should have topologies: (";
  for (std::size_t i = 0; i < expected.size(); ++i) {
    msg += "(";
    for (std::size_t j = 0; j < expected.at(i).size(); ++j) {
      msg += std::to_string(expected.at(i).at(j));
      if (j != expected.at(i).size() - 1) {
        msg += ", ";
      }
    }
    msg += ")";
    if (i != expected.size() - 1) {
      msg += " and ";
    }
  }
  msg += "), but got: (";
  for (std::size_t i = 0; i < actual.size(); ++i) {
    msg += "(";
    for (std::size_t j = 0; j < actual.at(i).size(); ++j) {
      msg += std::to_string(actual.at(i).at(j));
      if (j != actual.at(i).size() - 1) {
        msg += ", ";
      }
    }
    msg += ")";
    if (i != actual.size() - 1) {
      msg += " and ";
    }
  }
  msg += ").";

  return msg;
}

/// \brief Generate error message for get_all_pencil_topologies and
/// get_all_pencil_topologies test failures.
/// \tparam Topology1Type The type of the first topology input.
/// \tparam Topology2Type The type of the second topology input.
/// \tparam iType The index type used for the topology.
/// \tparam DIM The dimensionality of the topology.
/// \tparam FFT_DIM The dimensionality of the FFT axes.
/// \param[in] in_topology The input topology.
/// \param[out] out_topology The output topology.
/// \param[in] axes The axes along which the FFT is performed.
/// \param[in] actual The computed vector of all possible topologies that can be
/// formed
/// \param[in] expected The expected vector of all possible topologies that can
/// be formed
/// \return Error message including the input topologies, FFT axes, and the
/// expected vector of topologies
template <typename Topology1Type, typename Topology2Type,
          typename TopoAndAxesType, typename iType, std::size_t FFT_DIM>
std::string error_all_pencil_topologies(const Topology1Type& in_topology,
                                        const Topology2Type& out_topology,
                                        const std::array<iType, FFT_DIM>& axes,
                                        const TopoAndAxesType& actual,
                                        const TopoAndAxesType& expected) {
  auto print_topo_and_axes = [](const TopoAndAxesType& t) {
    auto [all_topologies, all_trans_axes, all_layouts] = t;
    std::string msg;
    msg += "topologies: (";
    for (std::size_t i = 0; i < all_topologies.size(); ++i) {
      msg += "(";
      for (std::size_t j = 0; j < all_topologies.at(i).size(); ++j) {
        msg += std::to_string(all_topologies.at(i).at(j));
        if (j != all_topologies.at(i).size() - 1) {
          msg += ", ";
        }
      }
      msg += ")";
      if (i != all_topologies.size() - 1) {
        msg += " and ";
      }
    }
    msg += "), trans_axes: (";
    for (std::size_t i = 0; i < all_trans_axes.size(); ++i) {
      msg += std::to_string(all_trans_axes.at(i));
      if (i != all_trans_axes.size() - 1) {
        msg += ", ";
      }
    }
    msg += "), layouts: (";
    for (std::size_t i = 0; i < all_layouts.size(); ++i) {
      msg += std::to_string(all_layouts.at(i));
      if (i != all_layouts.size() - 1) {
        msg += ", ";
      }
    }
    msg += ")";
    return msg;
  };

  std::string msg;
  msg += "Input topologies: ";
  msg += "(";
  for (std::size_t i = 0; i < in_topology.size(); ++i) {
    msg += std::to_string(in_topology.at(i));
    if (i != in_topology.size() - 1) {
      msg += ", ";
    }
  }
  msg += ") and (";
  for (std::size_t i = 0; i < out_topology.size(); ++i) {
    msg += std::to_string(out_topology.at(i));
    if (i != out_topology.size() - 1) {
      msg += ", ";
    }
  }
  msg += "), with FFT axes: (";
  msg += std::to_string(axes.at(0));
  for (std::size_t i = 1; i < axes.size(); ++i) {
    msg += ", " + std::to_string(axes.at(i));
  }

  msg += "), should have: " + print_topo_and_axes(expected);
  msg += ", but got: " + print_topo_and_axes(actual);

  return msg;
}

template <bool is_std_array>
void test_to_topology_type(std::size_t nprocs) {
  using KokkosFFT::Distributed::Impl::TopologyType;
  using topology1D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 1>,
      KokkosFFT::Distributed::Topology<std::size_t, 1, Kokkos::LayoutRight>>;
  using topology2D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 2>,
      KokkosFFT::Distributed::Topology<std::size_t, 2, Kokkos::LayoutLeft>>;
  using topology3D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 3>,
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutRight>>;
  using topology4D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 4>,
      KokkosFFT::Distributed::Topology<std::size_t, 4, Kokkos::LayoutLeft>>;

  using topology1D_and_ref1D_type = std::tuple<topology1D_type, TopologyType>;
  using topology2D_and_ref2D_type = std::tuple<topology2D_type, TopologyType>;
  using topology3D_and_ref3D_type = std::tuple<topology3D_type, TopologyType>;
  using topology4D_and_ref4D_type = std::tuple<topology4D_type, TopologyType>;

  const std::size_t p0 = 2, p1 = 3, p2 = 4, p3 = 5;

  topology1D_type topology1{nprocs};
  topology2D_type topology2_1{p0, nprocs}, topology2_2{nprocs, p1};
  topology3D_type topology3_1{p0, p1, nprocs}, topology3_2{p0, nprocs, p2},
      topology3_3{nprocs, p1, p2}, topology3_4{nprocs, nprocs, p2},
      topology3_5{nprocs, nprocs, nprocs};
  topology4D_type topology4_1{p0, p1, nprocs, nprocs},
      topology4_2{p0, nprocs, p2, nprocs}, topology4_3{p0, nprocs, nprocs, p3},
      topology4_4{nprocs, p1, p2, nprocs}, topology4_5{nprocs, p1, nprocs, p3},
      topology4_6{nprocs, nprocs, p2, p3},
      topology4_7{p0, nprocs, nprocs, nprocs},
      topology4_8{nprocs, nprocs, nprocs, nprocs};

  if (nprocs == 1) {
    // 1D topology
    std::vector<topology1D_and_ref1D_type> topo1D_test_cases = {
        {topology1, KokkosFFT::Distributed::Impl::TopologyType::Shared},
        {topology1D_type{0},
         KokkosFFT::Distributed::Impl::TopologyType::Empty}};
    for (const auto& [topo, ref] : topo1D_test_cases) {
      auto topo_type = KokkosFFT::Distributed::Impl::to_topology_type(topo);
      EXPECT_EQ(topo_type, ref) << error_to_topology_type(topo, ref);
    }

    // 2D topology
    std::vector<topology2D_and_ref2D_type> topo2D_test_cases = {
        {topology2_1, TopologyType::Slab},
        {topology2_2, TopologyType::Slab},
        {topology2D_type{0, nprocs}, TopologyType::Empty},
        {topology2D_type{nprocs, 0}, TopologyType::Empty}};
    for (const auto& [topo, ref] : topo2D_test_cases) {
      auto topo_type = KokkosFFT::Distributed::Impl::to_topology_type(topo);
      EXPECT_EQ(topo_type, ref) << error_to_topology_type(topo, ref);
    }

    // 3D topology
    std::vector<topology3D_and_ref3D_type> topo3D_test_cases = {
        {topology3_1, TopologyType::Pencil},
        {topology3_2, TopologyType::Pencil},
        {topology3_3, TopologyType::Pencil},
        {topology3_4, TopologyType::Slab},
        {topology3_5, TopologyType::Shared}};
    for (const auto& [topo, ref] : topo3D_test_cases) {
      auto topo_type = KokkosFFT::Distributed::Impl::to_topology_type(topo);
      EXPECT_EQ(topo_type, ref) << error_to_topology_type(topo, ref);
    }

    // 4D topology
    std::vector<topology4D_and_ref4D_type> topo4D_test_cases = {
        {topology4_1, TopologyType::Pencil},
        {topology4_2, TopologyType::Pencil},
        {topology4_3, TopologyType::Pencil},
        {topology4_4, TopologyType::Pencil},
        {topology4_5, TopologyType::Pencil},
        {topology4_6, TopologyType::Pencil},
        {topology4_7, TopologyType::Slab},
        {topology4_8, TopologyType::Shared}};
    for (const auto& [topo, ref] : topo4D_test_cases) {
      auto topo_type = KokkosFFT::Distributed::Impl::to_topology_type(topo);
      EXPECT_EQ(topo_type, ref) << error_to_topology_type(topo, ref);
    }
  } else {
    // 1D topology
    std::vector<topology1D_and_ref1D_type> topo1D_test_cases = {
        {topology1, TopologyType::Slab}};
    for (const auto& [topo, ref] : topo1D_test_cases) {
      auto topo_type = KokkosFFT::Distributed::Impl::to_topology_type(topo);
      EXPECT_EQ(topo_type, ref) << error_to_topology_type(topo, ref);
    }

    // 2D topology
    std::vector<topology2D_and_ref2D_type> topo2D_test_cases = {
        {topology2_1, TopologyType::Pencil},
        {topology2_2, TopologyType::Pencil}};
    for (const auto& [topo, ref] : topo2D_test_cases) {
      auto topo_type = KokkosFFT::Distributed::Impl::to_topology_type(topo);
      EXPECT_EQ(topo_type, ref) << error_to_topology_type(topo, ref);
    }

    // 3D topology
    std::vector<topology3D_and_ref3D_type> topo3D_test_cases = {
        {topology3_1, TopologyType::Brick},
        {topology3_2, TopologyType::Brick},
        {topology3_3, TopologyType::Brick},
        {topology3_4, TopologyType::Brick},
        {topology3_5, TopologyType::Brick}};
    for (const auto& [topo, ref] : topo3D_test_cases) {
      auto topo_type = KokkosFFT::Distributed::Impl::to_topology_type(topo);
      EXPECT_EQ(topo_type, ref) << error_to_topology_type(topo, ref);
    }

    // 4D topology
    std::vector<topology4D_and_ref4D_type> topo4D_test_cases = {
        {topology4_1, TopologyType::Invalid},
        {topology4_2, TopologyType::Invalid},
        {topology4_3, TopologyType::Invalid},
        {topology4_4, TopologyType::Invalid},
        {topology4_5, TopologyType::Invalid},
        {topology4_6, TopologyType::Invalid},
        {topology4_7, TopologyType::Invalid},
        {topology4_8, TopologyType::Invalid}};
    for (const auto& [topo, ref] : topo4D_test_cases) {
      auto topo_type = KokkosFFT::Distributed::Impl::to_topology_type(topo);
      EXPECT_EQ(topo_type, ref) << error_to_topology_type(topo, ref);
    }
  }
}

template <bool is_std_array>
void test_get_common_topology_type(std::size_t nprocs) {
  using KokkosFFT::Distributed::Impl::TopologyType;
  using topology1D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 1>,
      KokkosFFT::Distributed::Topology<std::size_t, 1, Kokkos::LayoutRight>>;
  using topology2D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 2>,
      KokkosFFT::Distributed::Topology<std::size_t, 2, Kokkos::LayoutLeft>>;
  using topology3D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 3>,
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutRight>>;
  using topology4D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 4>,
      KokkosFFT::Distributed::Topology<std::size_t, 4, Kokkos::LayoutLeft>>;

  using topology1D_and_ref1D_type =
      std::tuple<topology1D_type, topology1D_type, TopologyType>;
  using topology2D_and_ref2D_type =
      std::tuple<topology2D_type, topology2D_type, TopologyType>;
  using topology3D_and_ref3D_type =
      std::tuple<topology3D_type, topology3D_type, TopologyType>;
  using topology4D_and_ref4D_type =
      std::tuple<topology4D_type, topology4D_type, TopologyType>;

  const std::size_t p0 = 2, p1 = 3, p2 = 4, p3 = 5;

  topology1D_type topology1{nprocs};
  topology2D_type topology2_1{p0, nprocs}, topology2_2{nprocs, p1};
  topology3D_type topology3_1{p0, p1, nprocs}, topology3_2{p0, nprocs, p2},
      topology3_3{nprocs, p1, p2}, topology3_4{nprocs, nprocs, p2},
      topology3_5{nprocs, nprocs, nprocs};
  topology4D_type topology4_1{p0, p1, nprocs, nprocs},
      topology4_2{p0, nprocs, p2, nprocs}, topology4_3{p0, nprocs, nprocs, p3},
      topology4_4{nprocs, p1, p2, nprocs}, topology4_5{nprocs, p1, nprocs, p3},
      topology4_6{nprocs, nprocs, p2, p3},
      topology4_7{p0, nprocs, nprocs, nprocs},
      topology4_8{nprocs, nprocs, nprocs, nprocs};

  if (nprocs == 1) {
    // 1D topology
    std::vector<topology1D_and_ref1D_type> topo1D_test_cases = {
        {topology1, topology1,
         KokkosFFT::Distributed::Impl::TopologyType::Shared},
        {topology1D_type{0}, topology1,
         KokkosFFT::Distributed::Impl::TopologyType::Empty}};
    for (const auto& [topo1, topo2, ref] : topo1D_test_cases) {
      auto topo_type =
          KokkosFFT::Distributed::Impl::get_common_topology_type(topo1, topo2);
      EXPECT_EQ(topo_type, ref)
          << error_get_common_topology_type(topo1, topo2, ref);
    }

    // 2D topology
    std::vector<topology2D_and_ref2D_type> topo2D_test_cases = {
        {topology2_1, topology2_1,
         KokkosFFT::Distributed::Impl::TopologyType::Slab},
        {topology2_2, topology2_2,
         KokkosFFT::Distributed::Impl::TopologyType::Slab},
        {topology2_1, topology2_2,
         KokkosFFT::Distributed::Impl::TopologyType::Slab},
        {topology2D_type{0, nprocs}, topology2_1,
         KokkosFFT::Distributed::Impl::TopologyType::Empty},
        {topology2D_type{nprocs, 0}, topology2_2,
         KokkosFFT::Distributed::Impl::TopologyType::Empty}};

    for (const auto& [topo1, topo2, ref] : topo2D_test_cases) {
      auto topo_type =
          KokkosFFT::Distributed::Impl::get_common_topology_type(topo1, topo2);
      EXPECT_EQ(topo_type, ref)
          << error_get_common_topology_type(topo1, topo2, ref);
    }

    // 3D topology
    std::vector<topology3D_and_ref3D_type> topo3D_test_cases = {
        {topology3_1, topology3_1, TopologyType::Pencil},
        {topology3_2, topology3_2, TopologyType::Pencil},
        {topology3_3, topology3_3, TopologyType::Pencil},
        {topology3_4, topology3_4, TopologyType::Slab},
        {topology3_5, topology3_5, TopologyType::Shared},
        {topology3_1, topology3_2, TopologyType::Pencil},
        {topology3_1, topology3_3, TopologyType::Pencil},
        {topology3_1, topology3_4, TopologyType::Invalid},
        {topology3_1, topology3_5, TopologyType::Invalid}};
    for (const auto& [topo1, topo2, ref] : topo3D_test_cases) {
      auto topo_type =
          KokkosFFT::Distributed::Impl::get_common_topology_type(topo1, topo2);
      EXPECT_EQ(topo_type, ref)
          << error_get_common_topology_type(topo1, topo2, ref);
    }

    // 4D topology
    std::vector<topology4D_and_ref4D_type> topo4D_test_cases = {
        {topology4_1, topology4_1, TopologyType::Pencil},
        {topology4_2, topology4_2, TopologyType::Pencil},
        {topology4_3, topology4_3, TopologyType::Pencil},
        {topology4_4, topology4_4, TopologyType::Pencil},
        {topology4_5, topology4_5, TopologyType::Pencil},
        {topology4_6, topology4_6, TopologyType::Pencil},
        {topology4_7, topology4_7, TopologyType::Slab},
        {topology4_8, topology4_8, TopologyType::Shared},
        {topology4_1, topology4_2, TopologyType::Pencil},
        {topology4_1, topology4_3, TopologyType::Pencil},
        {topology4_1, topology4_4, TopologyType::Pencil},
        {topology4_1, topology4_5, TopologyType::Pencil},
        {topology4_1, topology4_6, TopologyType::Pencil},
        {topology4_2, topology4_3, TopologyType::Pencil},
        {topology4_2, topology4_4, TopologyType::Pencil},
        {topology4_2, topology4_5, TopologyType::Pencil},
        {topology4_2, topology4_6, TopologyType::Pencil},
        {topology4_3, topology4_4, TopologyType::Pencil},
        {topology4_3, topology4_5, TopologyType::Pencil},
        {topology4_3, topology4_6, TopologyType::Pencil},
        {topology4_4, topology4_5, TopologyType::Pencil},
        {topology4_4, topology4_6, TopologyType::Pencil},
        {topology4_5, topology4_6, TopologyType::Pencil},
        {topology4_7, topology4_7, TopologyType::Slab},
        {topology4_8, topology4_8, TopologyType::Shared}};

    for (const auto& [topo1, topo2, ref] : topo4D_test_cases) {
      auto topo_type =
          KokkosFFT::Distributed::Impl::get_common_topology_type(topo1, topo2);
      EXPECT_EQ(topo_type, ref)
          << error_get_common_topology_type(topo1, topo2, ref);
    }
  } else {
    // 1D topology
    std::vector<topology1D_and_ref1D_type> topo1D_test_cases = {
        {topology1, topology1, TopologyType::Slab}};
    for (const auto& [topo1, topo2, ref] : topo1D_test_cases) {
      auto topo_type =
          KokkosFFT::Distributed::Impl::get_common_topology_type(topo1, topo2);
      EXPECT_EQ(topo_type, ref)
          << error_get_common_topology_type(topo1, topo2, ref);
    }

    // 2D topology
    std::vector<topology2D_and_ref2D_type> topo2D_test_cases = {
        {topology2_1, topology2_1, TopologyType::Pencil},
        {topology2_2, topology2_2, TopologyType::Pencil},
        {topology2_1, topology2_2, TopologyType::Pencil}};
    for (const auto& [topo1, topo2, ref] : topo2D_test_cases) {
      auto topo_type =
          KokkosFFT::Distributed::Impl::get_common_topology_type(topo1, topo2);
      EXPECT_EQ(topo_type, ref)
          << error_get_common_topology_type(topo1, topo2, ref);
    }

    // 3D topology
    std::vector<topology3D_and_ref3D_type> topo3D_test_cases = {
        {topology3_1, topology3_1, TopologyType::Brick},
        {topology3_2, topology3_2, TopologyType::Brick},
        {topology3_3, topology3_3, TopologyType::Brick},
        {topology3_4, topology3_4, TopologyType::Brick},
        {topology3_5, topology3_5, TopologyType::Brick},
        {topology3_1, topology3_2, TopologyType::Brick},
        {topology3_1, topology3_3, TopologyType::Brick},
        {topology3_1, topology3_4, TopologyType::Brick},
        {topology3_1, topology3_5, TopologyType::Brick}};
    for (const auto& [topo1, topo2, ref] : topo3D_test_cases) {
      auto topo_type =
          KokkosFFT::Distributed::Impl::get_common_topology_type(topo1, topo2);
      EXPECT_EQ(topo_type, ref)
          << error_get_common_topology_type(topo1, topo2, ref);
    }

    // 4D topology
    std::vector<topology4D_and_ref4D_type> topo4D_test_cases = {
        {topology4_1, topology4_1, TopologyType::Invalid},
        {topology4_2, topology4_2, TopologyType::Invalid},
        {topology4_3, topology4_3, TopologyType::Invalid},
        {topology4_4, topology4_4, TopologyType::Invalid},
        {topology4_5, topology4_5, TopologyType::Invalid},
        {topology4_6, topology4_6, TopologyType::Invalid},
        {topology4_7, topology4_7, TopologyType::Invalid},
        {topology4_8, topology4_8, TopologyType::Invalid},
        {topology4_1, topology4_2, TopologyType::Invalid},
        {topology4_1, topology4_3, TopologyType::Invalid},
        {topology4_1, topology4_4, TopologyType::Invalid},
        {topology4_1, topology4_5, TopologyType::Invalid},
        {topology4_1, topology4_6, TopologyType::Invalid},
        {topology4_2, topology4_3, TopologyType::Invalid},
        {topology4_2, topology4_4, TopologyType::Invalid},
        {topology4_2, topology4_5, TopologyType::Invalid}};

    for (const auto& [topo1, topo2, ref] : topo4D_test_cases) {
      auto topo_type =
          KokkosFFT::Distributed::Impl::get_common_topology_type(topo1, topo2);
      EXPECT_EQ(topo_type, ref)
          << error_get_common_topology_type(topo1, topo2, ref);
    }
  }
}

template <bool is_std_array>
void test_is_topology(std::size_t nprocs) {
  using KokkosFFT::Distributed::Impl::TopologyType;
  using topology1D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 1>,
      KokkosFFT::Distributed::Topology<std::size_t, 1, Kokkos::LayoutRight>>;
  using topology2D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 2>,
      KokkosFFT::Distributed::Topology<std::size_t, 2, Kokkos::LayoutLeft>>;
  using topology3D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 3>,
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutRight>>;
  using topology4D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 4>,
      KokkosFFT::Distributed::Topology<std::size_t, 4, Kokkos::LayoutLeft>>;

  using topology1D_and_ref1D_type = std::tuple<topology1D_type, TopologyType>;
  using topology2D_and_ref2D_type = std::tuple<topology2D_type, TopologyType>;
  using topology3D_and_ref3D_type = std::tuple<topology3D_type, TopologyType>;
  using topology4D_and_ref4D_type = std::tuple<topology4D_type, TopologyType>;
  const std::size_t p0 = 2, p1 = 3, p2 = 4, p3 = 5;

  topology1D_type topology1{nprocs};
  topology2D_type topology2_1{p0, nprocs}, topology2_2{nprocs, p1};
  topology3D_type topology3_1{p0, p1, nprocs}, topology3_2{p0, nprocs, p2},
      topology3_3{nprocs, p1, p2}, topology3_4{nprocs, nprocs, p2},
      topology3_5{nprocs, nprocs, nprocs};
  topology4D_type topology4_1{p0, p1, nprocs, nprocs},
      topology4_2{p0, nprocs, p2, nprocs}, topology4_3{p0, nprocs, nprocs, p3},
      topology4_4{nprocs, p1, p2, nprocs}, topology4_5{nprocs, p1, nprocs, p3},
      topology4_6{nprocs, nprocs, p2, p3},
      topology4_7{p0, nprocs, nprocs, nprocs},
      topology4_8{nprocs, nprocs, nprocs, nprocs};

  if (nprocs == 1) {
    // 1D topology is shared
    std::vector<topology1D_and_ref1D_type> topo1D_test_cases = {
        {topology1, TopologyType::Shared},
        {topology1D_type{0}, TopologyType::Empty}};
    for (const auto& [topo, ref_topo_type] : topo1D_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if (topo_type == ref_topo_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, false);
        }
      }
    }

    // 2D topology is slab
    std::vector<topology2D_and_ref2D_type> topo2D_test_cases = {
        {topology2_1, TopologyType::Slab},
        {topology2_2, TopologyType::Slab},
        {topology2D_type{0, nprocs}, TopologyType::Empty},
        {topology2D_type{nprocs, 0}, TopologyType::Empty}};

    for (const auto& [topo, ref_topo_type] : topo2D_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if (topo_type == ref_topo_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, false);
        }
      }
    }

    // 3D case
    // Pencil topologies
    std::vector<topology3D_and_ref3D_type> topo3D_test_cases = {
        {topology3_1, TopologyType::Pencil},
        {topology3_2, TopologyType::Pencil},
        {topology3_3, TopologyType::Pencil},
        {topology3_4, TopologyType::Slab},
        {topology3_5, TopologyType::Shared},
        {topology3D_type{0, p1, p2}, TopologyType::Empty},
        {topology3D_type{p0, 0, p2}, TopologyType::Empty},
        {topology3D_type{p0, p1, 0}, TopologyType::Empty}};

    for (const auto& [topo, ref_topo_type] : topo3D_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if (topo_type == ref_topo_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, false);
        }
      }
    }

    // 4D case
    std::vector<topology4D_and_ref4D_type> topo4D_test_cases = {
        {topology4_1, TopologyType::Pencil},
        {topology4_2, TopologyType::Pencil},
        {topology4_3, TopologyType::Pencil},
        {topology4_4, TopologyType::Pencil},
        {topology4_5, TopologyType::Pencil},
        {topology4_6, TopologyType::Pencil},
        {topology4_7, TopologyType::Slab},
        {topology4_8, TopologyType::Shared}};

    for (const auto& [topo, ref_topo_type] : topo4D_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if (topo_type == ref_topo_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, false);
        }
      }
    }
  } else {
    // 1D topology is slab
    std::vector<topology1D_and_ref1D_type> topo1D_test_cases = {
        {topology1, TopologyType::Slab},
        {topology1D_type{0}, TopologyType::Empty}};
    for (const auto& [topo, ref_topo_type] : topo1D_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if (topo_type == ref_topo_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, false);
        }
      }
    }

    // 2D topology is pencil
    std::vector<topology2D_and_ref2D_type> topo2D_test_cases = {
        {topology2_1, TopologyType::Pencil},
        {topology2_2, TopologyType::Pencil},
        {topology2D_type{0, nprocs}, TopologyType::Empty},
        {topology2D_type{nprocs, 0}, TopologyType::Empty}};

    for (const auto& [topo, ref_topo_type] : topo2D_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if (topo_type == ref_topo_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, false);
        }
      }
    }

    // 3D topology
    std::vector<topology3D_and_ref3D_type> topo3D_test_cases = {
        {topology3_1, TopologyType::Brick},
        {topology3_2, TopologyType::Brick},
        {topology3_3, TopologyType::Brick},
        {topology3_4, TopologyType::Brick},
        {topology3_5, TopologyType::Brick},
        {topology3_1, TopologyType::Brick},
        {topology3_2, TopologyType::Brick},
        {topology3_3, TopologyType::Brick},
        {topology3_4, TopologyType::Brick},
        {topology3_5, TopologyType::Brick},
        {topology3D_type{0, p1, p2}, TopologyType::Empty},
        {topology3D_type{p0, 0, p2}, TopologyType::Empty},
        {topology3D_type{p0, p1, 0}, TopologyType::Empty}};

    for (const auto& [topo, ref_topo_type] : topo3D_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if (topo_type == ref_topo_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, false);
        }
      }
    }

    // 4D topology
    std::vector<topology4D_and_ref4D_type> topo4D_test_cases = {
        {topology4_1, TopologyType::Invalid},
        {topology4_2, TopologyType::Invalid},
        {topology4_3, TopologyType::Invalid},
        {topology4_4, TopologyType::Invalid},
        {topology4_5, TopologyType::Invalid},
        {topology4_6, TopologyType::Invalid},
        {topology4_7, TopologyType::Invalid},
        {topology4_8, TopologyType::Invalid}};

    for (const auto& [topo, ref_topo_type] : topo4D_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if (topo_type == ref_topo_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, false);
        }
      }
    }
  }
}

template <bool is_std_array>
void test_are_topologies(std::size_t nprocs) {
  using KokkosFFT::Distributed::Impl::TopologyType;
  using topology1D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 1>,
      KokkosFFT::Distributed::Topology<std::size_t, 1, Kokkos::LayoutRight>>;
  using topology2D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 2>,
      KokkosFFT::Distributed::Topology<std::size_t, 2, Kokkos::LayoutLeft>>;
  using topology3D_r_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 3>,
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutRight>>;
  using topology3D_l_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 3>,
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutLeft>>;
  using topology4D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 4>,
      KokkosFFT::Distributed::Topology<std::size_t, 4, Kokkos::LayoutLeft>>;

  using topology1D_and_ref1D_type =
      std::tuple<topology1D_type, topology1D_type, TopologyType, bool>;
  using topology2D_and_ref2D_type =
      std::tuple<topology2D_type, topology2D_type, TopologyType, bool>;
  using topology3D_rr_and_ref3D_type =
      std::tuple<topology3D_r_type, topology3D_r_type, TopologyType, bool>;
  using topology3D_rl_and_ref3D_type =
      std::tuple<topology3D_r_type, topology3D_l_type, TopologyType, bool>;
  using topology4D_and_ref4D_type =
      std::tuple<topology4D_type, topology4D_type, TopologyType, bool>;

  const std::size_t p0 = 2, p1 = 3, p2 = 4, p3 = 5;

  topology1D_type topology1{nprocs};
  topology2D_type topology2_1{p0, nprocs}, topology2_2{nprocs, p1};
  topology3D_r_type topology3_1{p0, p1, nprocs}, topology3_2{p0, nprocs, p2},
      topology3_3{nprocs, p1, p2}, topology3_4{nprocs, nprocs, p2},
      topology3_5{nprocs, nprocs, nprocs};
  topology3D_l_type topology3_6{p0, p1, nprocs}, topology3_7{p0, nprocs, p2},
      topology3_8{nprocs, p1, p2}, topology3_9{nprocs, nprocs, p2},
      topology3_10{nprocs, nprocs, nprocs};
  topology4D_type topology4_1{p0, p1, nprocs, nprocs},
      topology4_2{p0, nprocs, p2, nprocs}, topology4_3{p0, nprocs, nprocs, p3},
      topology4_4{nprocs, p1, p2, nprocs}, topology4_5{nprocs, p1, nprocs, p3},
      topology4_6{nprocs, nprocs, p2, p3},
      topology4_7{p0, nprocs, nprocs, nprocs},
      topology4_8{nprocs, nprocs, nprocs, nprocs};

  if (nprocs == 1) {
    // 1D topology is shared
    std::vector<topology1D_and_ref1D_type> topo1D_test_cases = {
        {topology1, topology1, TopologyType::Shared, true},
        {topology1D_type{0}, topology1D_type{0}, TopologyType::Empty, true},
        {topology1, topology1D_type{0}, TopologyType::Invalid, false}};
    for (const auto& [topo1, topo2, ref_topo_type, is_same_type] :
         topo1D_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if ((topo_type == ref_topo_type) && is_same_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, false);
        }
      }
    }

    // 2D topology is slab
    std::vector<topology2D_and_ref2D_type> topo2D_test_cases = {
        {topology2_1, topology2_1, TopologyType::Slab, true},
        {topology2_2, topology2_2, TopologyType::Slab, true},
        {topology2_1, topology2_2, TopologyType::Slab, true},
        {topology2_1, topology2D_type{0, nprocs}, TopologyType::Invalid, false},
        {topology2_1, topology2D_type{nprocs, 0}, TopologyType::Invalid,
         false}};

    for (const auto& [topo1, topo2, ref_topo_type, is_same_type] :
         topo2D_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if ((topo_type == ref_topo_type) && is_same_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, false);
        }
      }
    }

    // 3D case Pencil topologies
    std::vector<topology3D_rr_and_ref3D_type> topo3D_rr_test_cases = {
        {topology3_1, topology3_1, TopologyType::Pencil, true},
        {topology3_2, topology3_2, TopologyType::Pencil, true},
        {topology3_3, topology3_3, TopologyType::Pencil, true},
        {topology3_4, topology3_4, TopologyType::Slab, true},
        {topology3_5, topology3_5, TopologyType::Shared, true},
        {topology3_1, topology3_2, TopologyType::Pencil, true},
        {topology3_2, topology3_3, TopologyType::Pencil, true},
        {topology3_3, topology3_1, TopologyType::Pencil, true},
        {topology3D_r_type{0, p1, p2}, topology3D_r_type{0, p1, p2},
         TopologyType::Empty, true},
        {topology3D_r_type{p0, 0, p2}, topology3D_r_type{p0, 0, p2},
         TopologyType::Empty, true},
        {topology3D_r_type{p0, p1, 0}, topology3D_r_type{p0, p1, 0},
         TopologyType::Empty, true}};
    for (const auto& [topo1, topo2, ref_topo_type, is_same_type] :
         topo3D_rr_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if ((topo_type == ref_topo_type) && is_same_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, false);
        }
      }
    }
    std::vector<topology3D_rl_and_ref3D_type> topo3D_rl_test_cases = {
        {topology3_1, topology3_6, TopologyType::Pencil, true},
        {topology3_2, topology3_7, TopologyType::Pencil, true},
        {topology3_3, topology3_8, TopologyType::Pencil, true},
        {topology3_4, topology3_9, TopologyType::Slab, true},
        {topology3_5, topology3_10, TopologyType::Shared, true}};
    for (const auto& [topo1, topo2, ref_topo_type, is_same_type] :
         topo3D_rl_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if (topo_type == ref_topo_type && is_same_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, false);
        }
      }
    }

    // 4D case
    std::vector<topology4D_and_ref4D_type> topo4D_test_cases = {
        {topology4_1, topology4_1, TopologyType::Pencil, true},
        {topology4_2, topology4_2, TopologyType::Pencil, true},
        {topology4_3, topology4_3, TopologyType::Pencil, true},
        {topology4_4, topology4_4, TopologyType::Pencil, true},
        {topology4_5, topology4_5, TopologyType::Pencil, true},
        {topology4_6, topology4_6, TopologyType::Pencil, true},
        {topology4_7, topology4_7, TopologyType::Slab, true},
        {topology4_8, topology4_8, TopologyType::Shared, true}};
    for (const auto& [topo1, topo2, ref_topo_type, is_same_type] :
         topo4D_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if ((topo_type == ref_topo_type) && is_same_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, false);
        }
      }
    }
  } else {
    // 1D topology
    std::vector<topology1D_and_ref1D_type> topo1D_test_cases = {
        {topology1, topology1, TopologyType::Slab, true},
        {topology1D_type{0}, topology1D_type{0}, TopologyType::Empty, true},
        {topology1, topology1D_type{0}, TopologyType::Invalid, false}};
    for (const auto& [topo1, topo2, ref_topo_type, is_same_type] :
         topo1D_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if ((topo_type == ref_topo_type) && is_same_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, false);
        }
      }
    }

    // 2D topology
    std::vector<topology2D_and_ref2D_type> topo2D_test_cases = {
        {topology2_1, topology2_1, TopologyType::Pencil, true},
        {topology2_2, topology2_2, TopologyType::Pencil, true},
        {topology2_1, topology2_2, TopologyType::Pencil, true},
        {topology2_1, topology2D_type{0, nprocs}, TopologyType::Invalid, false},
        {topology2_1, topology2D_type{nprocs, 0}, TopologyType::Invalid,
         false}};
    for (const auto& [topo1, topo2, ref_topo_type, is_same_type] :
         topo2D_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if ((topo_type == ref_topo_type) && is_same_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, false);
        }
      }
    }

    // 3D topology
    std::vector<topology3D_rr_and_ref3D_type> topo3D_rr_test_cases = {
        {topology3_1, topology3_1, TopologyType::Brick, true},
        {topology3_2, topology3_2, TopologyType::Brick, true},
        {topology3_3, topology3_3, TopologyType::Brick, true},
        {topology3_4, topology3_4, TopologyType::Brick, true},
        {topology3_5, topology3_5, TopologyType::Brick, true},
        {topology3_1, topology3_2, TopologyType::Brick, true},
        {topology3_2, topology3_3, TopologyType::Brick, true},
        {topology3_3, topology3_1, TopologyType::Brick, true},
        {topology3D_r_type{0, p1, p2}, topology3D_r_type{0, p1, p2},
         TopologyType::Empty, true},
        {topology3D_r_type{p0, 0, p2}, topology3D_r_type{p0, 0, p2},
         TopologyType::Empty, true},
        {topology3D_r_type{p0, p1, 0}, topology3D_r_type{p0, p1, 0},
         TopologyType::Empty, true}};
    for (const auto& [topo1, topo2, ref_topo_type, is_same_type] :
         topo3D_rr_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if ((topo_type == ref_topo_type) && is_same_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, false);
        }
      }
    }

    std::vector<topology3D_rl_and_ref3D_type> topo3D_rl_test_cases = {
        {topology3_1, topology3_6, TopologyType::Brick, true},
        {topology3_2, topology3_7, TopologyType::Brick, true},
        {topology3_3, topology3_8, TopologyType::Brick, true},
        {topology3_4, topology3_9, TopologyType::Brick, true},
        {topology3_5, topology3_10, TopologyType::Brick, true}};
    for (const auto& [topo1, topo2, ref_topo_type, is_same_type] :
         topo3D_rl_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if ((topo_type == ref_topo_type) && is_same_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, false);
        }
      }
    }

    // 4D topology
    std::vector<topology4D_and_ref4D_type> topo4D_test_cases = {
        {topology4_1, topology4_1, TopologyType::Invalid, true},
        {topology4_2, topology4_2, TopologyType::Invalid, true},
        {topology4_3, topology4_3, TopologyType::Invalid, true},
        {topology4_4, topology4_4, TopologyType::Invalid, true},
        {topology4_5, topology4_5, TopologyType::Invalid, true},
        {topology4_6, topology4_6, TopologyType::Invalid, true},
        {topology4_7, topology4_7, TopologyType::Invalid, true},
        {topology4_8, topology4_8, TopologyType::Invalid, true}};
    for (const auto& [topo1, topo2, ref_topo_type, is_same_type] :
         topo4D_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if ((topo_type == ref_topo_type) && is_same_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, false);
        }
      }
    }
  }
}

void test_slab_in_out_axes_2D(std::size_t nprocs) {
  using topo_type = std::array<std::size_t, 2>;
  using topo_and_ref_type =
      std::tuple<topo_type, topo_type, std::tuple<std::size_t, std::size_t>>;
  topo_type topo0{1, nprocs}, topo1{nprocs, 1}, topo2{nprocs, 7}, topo3{1, 1};

  if (nprocs == 1) {
    // Failure tests because of size 1 case
    std::vector<topo_and_ref_type> topo_test_cases = {{topo0, topo1, {0, 1}},
                                                      {topo1, topo0, {1, 0}}};
    for (const auto& [topo_in, topo_out, ref_inout_axes] : topo_test_cases) {
      EXPECT_THROW(
          {
            [[maybe_unused]] auto inout_axes =
                KokkosFFT::Distributed::Impl::slab_in_out_axes(topo_in,
                                                               topo_out);
          },
          std::runtime_error);
    }
  } else {
    std::vector<topo_and_ref_type> topo_test_cases = {{topo0, topo1, {0, 1}},
                                                      {topo1, topo0, {1, 0}}};

    for (const auto& [topo_in, topo_out, ref_inout_axes] : topo_test_cases) {
      auto inout_axes =
          KokkosFFT::Distributed::Impl::slab_in_out_axes(topo_in, topo_out);
      EXPECT_EQ(inout_axes, ref_inout_axes)
          << error_in_out_axes(topo_in, topo_out, ref_inout_axes);
    }
  }

  // Failure tests because of shape mismatch (or size 1 case)
  std::vector<topo_and_ref_type> topo_failure_test_cases = {
      {topo0, topo2, {0, 1}}, {topo0, topo3, {0, 1}}};
  for (const auto& [topo_in, topo_out, ref_inout_axes] :
       topo_failure_test_cases) {
    EXPECT_THROW(
        {
          [[maybe_unused]] auto inout_axes =
              KokkosFFT::Distributed::Impl::slab_in_out_axes(topo_in, topo_out);
        },
        std::runtime_error);
  }
}

void test_slab_in_out_axes_3D(std::size_t nprocs) {
  using topo_type = std::array<std::size_t, 3>;
  using topo_and_ref_type =
      std::tuple<topo_type, topo_type, std::tuple<std::size_t, std::size_t>>;
  topo_type topo0{1, 1, nprocs}, topo1{1, nprocs, 1}, topo2{nprocs, 1, 1},
      topo3{1, nprocs, 7}, topo4{1, 1, 1};

  if (nprocs == 1) {
    // Failure tests because of size 1 case
    std::vector<topo_and_ref_type> topo_test_cases = {
        {topo0, topo1, {0, 1}}, {topo0, topo2, {0, 2}}, {topo1, topo0, {1, 0}},
        {topo1, topo2, {1, 2}}, {topo2, topo0, {2, 0}}, {topo2, topo1, {2, 1}}};
    for (const auto& [topo_in, topo_out, ref_inout_axes] : topo_test_cases) {
      EXPECT_THROW(
          {
            [[maybe_unused]] auto inout_axes =
                KokkosFFT::Distributed::Impl::slab_in_out_axes(topo_in,
                                                               topo_out);
          },
          std::runtime_error);
    }
  } else {
    std::vector<topo_and_ref_type> topo_test_cases = {
        {topo0, topo1, {1, 2}}, {topo0, topo2, {0, 2}}, {topo1, topo0, {2, 1}},
        {topo1, topo2, {0, 1}}, {topo2, topo0, {2, 0}}, {topo2, topo1, {1, 0}}};
    for (const auto& [topo_in, topo_out, ref_inout_axes] : topo_test_cases) {
      auto inout_axes =
          KokkosFFT::Distributed::Impl::slab_in_out_axes(topo_in, topo_out);
      EXPECT_EQ(inout_axes, ref_inout_axes)
          << error_in_out_axes(topo_in, topo_out, ref_inout_axes);
    }
  }

  // Failure tests because of shape mismatch (or size 1 case)
  std::vector<topo_and_ref_type> topo_failure_test_cases = {
      {topo0, topo3, {0, 1}}, {topo0, topo4, {0, 1}}};
  for (const auto& [topo_in, topo_out, ref_inout_axes] :
       topo_failure_test_cases) {
    EXPECT_THROW(
        {
          [[maybe_unused]] auto inout_axes =
              KokkosFFT::Distributed::Impl::slab_in_out_axes(topo_in, topo_out);
        },
        std::runtime_error);
  }
}

void test_decompose_axes_slab(std::size_t nprocs) {
  using topo3D_type       = std::array<std::size_t, 3>;
  using topo4D_type       = std::array<std::size_t, 4>;
  using axes_type         = std::array<std::size_t, 3>;
  using vec_topo3D_type   = std::vector<topo3D_type>;
  using vec_topo4D_type   = std::vector<topo4D_type>;
  using vec_axes_type     = std::vector<std::size_t>;
  using vec_vec_axes_type = std::vector<vec_axes_type>;
  using topo3D_and_ref_type =
      std::tuple<vec_topo3D_type, axes_type, vec_vec_axes_type>;
  using topo4D_and_ref_type =
      std::tuple<vec_topo4D_type, axes_type, vec_vec_axes_type>;

  // 3D topologies
  topo3D_type topo0{1, 1, nprocs}, topo1{1, nprocs, 1}, topo2{nprocs, 1, 1};

  // 4D topologies
  topo4D_type topo3{1, 1, 1, nprocs}, topo4{1, 1, nprocs, 1};

  axes_type axes012{0, 1, 2}, axes021{0, 2, 1}, axes102{1, 0, 2},
      axes120{1, 2, 0}, axes201{2, 0, 1}, axes210{2, 1, 0};

  std::vector<axes_type> all_axes{axes012, axes021, axes102,
                                  axes120, axes201, axes210};

  if (nprocs == 1) {
    for (const auto& axes : all_axes) {
      vec_vec_axes_type ref_all_axes2{KokkosFFT::Impl::to_vector(axes), {}},
          ref_all_axes3{KokkosFFT::Impl::to_vector(axes), {}, {}};
      // 3D case
      std::vector<topo3D_and_ref_type> topo3D_test_cases = {
          {vec_topo3D_type{topo0, topo1}, axes, ref_all_axes2},
          {vec_topo3D_type{topo0, topo2}, axes, ref_all_axes2},
          {vec_topo3D_type{topo1, topo2}, axes, ref_all_axes2},
          {vec_topo3D_type{topo2, topo0, topo2}, axes, ref_all_axes3}};
      for (const auto& [topos3D, axes3D, ref_axes3D] : topo3D_test_cases) {
        auto all_axes_3D =
            KokkosFFT::Distributed::Impl::decompose_axes(topos3D, axes3D);
        EXPECT_EQ(all_axes_3D, ref_axes3D)
            << error_decompose_axes(topos3D, axes3D, ref_axes3D);
      }

      // 4D case
      std::vector<topo4D_and_ref_type> topo4D_test_cases = {
          {vec_topo4D_type{topo3, topo4}, axes, ref_all_axes2},
          {vec_topo4D_type{topo4, topo3}, axes, ref_all_axes2}};

      for (const auto& [topos4D, axes4D, ref_axes4D] : topo4D_test_cases) {
        auto all_axes_4D =
            KokkosFFT::Distributed::Impl::decompose_axes(topos4D, axes4D);
        EXPECT_EQ(all_axes_4D, ref_axes4D)
            << error_decompose_axes(topos4D, axes4D, ref_axes4D);
      }
    }
  } else {
    vec_vec_axes_type ref_all_axes_2_0_2{vec_axes_type{2, 1}, vec_axes_type{0},
                                         vec_axes_type{}},
        ref_all_axes_3_4{vec_axes_type{0, 1, 2}, vec_axes_type{}},
        ref_all_axes_4_3_ax210{vec_axes_type{1, 0}, vec_axes_type{2}},
        ref_all_axes_4_3_ax012{vec_axes_type{}, vec_axes_type{0, 1, 2}};
    // 3D case
    std::vector<topo3D_and_ref_type> topo3D_test_cases{
        {vec_topo3D_type{topo2, topo0, topo2}, axes021, ref_all_axes_2_0_2}};
    for (const auto& [topos3D, axes3D, ref_axes3D] : topo3D_test_cases) {
      auto all_axes_3D =
          KokkosFFT::Distributed::Impl::decompose_axes(topos3D, axes3D);
      EXPECT_EQ(all_axes_3D, ref_axes3D)
          << error_decompose_axes(topos3D, axes3D, ref_axes3D);
    }

    // 4D case
    std::vector<topo4D_and_ref_type> topo4D_test_cases{
        {vec_topo4D_type{topo3, topo4}, axes012, ref_all_axes_3_4},
        {vec_topo4D_type{topo4, topo3}, axes210, ref_all_axes_4_3_ax210},
        {vec_topo4D_type{topo4, topo3}, axes012, ref_all_axes_4_3_ax012}};

    for (const auto& [topos4D, axes4D, ref_axes4D] : topo4D_test_cases) {
      auto all_axes_4D =
          KokkosFFT::Distributed::Impl::decompose_axes(topos4D, axes4D);
      EXPECT_EQ(all_axes_4D, ref_axes4D)
          << error_decompose_axes(topos4D, axes4D, ref_axes4D);
    }
  }
}

void test_decompose_axes_pencil(std::size_t nprocs) {
  using topo_type         = std::array<std::size_t, 3>;
  using axes_type         = std::array<std::size_t, 3>;
  using vec_axes_type     = std::vector<std::size_t>;
  using vec_topo_type     = std::vector<topo_type>;
  using vec_vec_axes_type = std::vector<vec_axes_type>;
  using topo_and_ref_type =
      std::tuple<vec_topo_type, axes_type, vec_vec_axes_type>;
  std::size_t np0 = 4;

  // 3D topologies
  topo_type topo0{1, nprocs, np0}, topo1{nprocs, 1, np0}, topo2{np0, nprocs, 1},
      topo3{nprocs, np0, 1}, topo4{np0, 1, nprocs};

  axes_type axes012{0, 1, 2}, axes021{0, 2, 1}, axes102{1, 0, 2},
      axes120{1, 2, 0}, axes201{2, 0, 1}, axes210{2, 1, 0};
  std::vector<axes_type> all_axes = {axes012, axes021, axes102,
                                     axes120, axes201, axes210};
  if (nprocs == 1) {
    // Slab geometry
    std::vector<topo_and_ref_type> topo_test_cases = {
        {std::vector<topo_type>{topo0, topo2, topo4, topo2, topo0},
         axes012,
         {{}, vec_axes_type{1, 2}, {}, {}, vec_axes_type{0}}},
        {std::vector<topo_type>{topo0, topo1, topo3, topo1, topo0},
         axes021,
         {vec_axes_type{1}, {}, vec_axes_type{0, 2}, {}, {}}},
        {std::vector<topo_type>{topo0, topo2, topo0, topo1, topo0},
         axes102,
         {{}, vec_axes_type{2}, vec_axes_type{1, 0}, {}, {}}},
        {std::vector<topo_type>{topo0, topo1, topo0, topo2, topo0},
         axes201,
         {vec_axes_type{0, 1}, {}, {}, vec_axes_type{2}, {}}},
        {std::vector<topo_type>{topo0, topo2, topo0, topo1},
         axes102,
         {{}, vec_axes_type{2}, vec_axes_type{1, 0}, {}}}};

    for (const auto& [topos, axes, ref_axes] : topo_test_cases) {
      auto all_axes = KokkosFFT::Distributed::Impl::decompose_axes(topos, axes);
      EXPECT_EQ(all_axes, ref_axes)
          << error_decompose_axes(topos, axes, ref_axes);
    }
  } else {
    // Pencil geometry
    std::vector<topo_and_ref_type> topo_test_cases = {
        {std::vector<topo_type>{topo0, topo2, topo4, topo2, topo0},
         axes012,
         {{}, vec_axes_type{2}, vec_axes_type{1}, {}, vec_axes_type{0}}},
        {std::vector<topo_type>{topo0, topo1, topo3, topo1, topo0},
         axes021,
         {{}, vec_axes_type{1}, vec_axes_type{2}, {}, vec_axes_type{0}}},
        {std::vector<topo_type>{topo0, topo2, topo0, topo1, topo0},
         axes102,
         {{}, vec_axes_type{2}, vec_axes_type{0}, vec_axes_type{1}, {}}},
        {std::vector<topo_type>{topo0, topo1, topo0, topo2, topo0},
         axes201,
         {{}, vec_axes_type{1}, vec_axes_type{0}, vec_axes_type{2}, {}}},
        {std::vector<topo_type>{topo0, topo2, topo0, topo1},
         axes102,
         {{}, vec_axes_type{2}, vec_axes_type{0}, vec_axes_type{1}}}};

    for (const auto& [topos, axes, ref_axes] : topo_test_cases) {
      auto all_axes = KokkosFFT::Distributed::Impl::decompose_axes(topos, axes);
      EXPECT_EQ(all_axes, ref_axes)
          << error_decompose_axes(topos, axes, ref_axes);
    }
  }
}

void test_compute_trans_axis(std::size_t nprocs) {
  using topo3D_type         = std::array<std::size_t, 3>;
  using topo4D_type         = std::array<std::size_t, 4>;
  using topo3D_and_ref_type = std::tuple<topo3D_type, topo3D_type, std::size_t>;
  using topo4D_and_ref_type = std::tuple<topo4D_type, topo4D_type, std::size_t>;

  std::size_t np0 = 4;

  // 3D topologies
  topo3D_type topo0{1, nprocs, np0}, topo1{nprocs, 1, np0},
      topo2{np0, nprocs, 1};

  // 4D topologies
  topo4D_type topo3{1, 1, np0, nprocs}, topo4{1, np0, 1, nprocs};

  if (nprocs == 1 || nprocs == np0) {
    // Failure tests because these are not pencils for nprocs == 1 or they
    // include identical non-one elements for nprocs == np0
    std::vector<topo3D_and_ref_type> topo3D_failure_test_cases = {
        {topo0, topo1, 0}, {topo0, topo2, 1}, {topo1, topo0, 0},
        {topo1, topo2, 1}, {topo2, topo0, 1}, {topo2, topo1, 1}};

    for (const auto& [topo_in, topo_out, ref_trans_axis] :
         topo3D_failure_test_cases) {
      EXPECT_THROW(
          {
            [[maybe_unused]] auto trans_axis =
                KokkosFFT::Distributed::Impl::compute_trans_axis(
                    topo_in, topo_out, nprocs);
          },
          std::runtime_error);
    }

    std::vector<topo4D_and_ref_type> topo4D_failure_test_cases = {
        {topo3, topo4, 0}, {topo4, topo3, 0}};
    for (const auto& [topo_in, topo_out, ref_trans_axis] :
         topo4D_failure_test_cases) {
      EXPECT_THROW(
          {
            [[maybe_unused]] auto trans_axis =
                KokkosFFT::Distributed::Impl::compute_trans_axis(
                    topo_in, topo_out, nprocs);
          },
          std::runtime_error);
    }
  } else {
    // 3D case
    std::vector<topo3D_and_ref_type> topo3D_test_cases = {{topo0, topo1, 0},
                                                          {topo0, topo2, 1},
                                                          {topo1, topo0, 0},
                                                          {topo2, topo0, 1}};

    for (const auto& [topo_in, topo_out, ref_trans_axis] : topo3D_test_cases) {
      auto trans_axis = KokkosFFT::Distributed::Impl::compute_trans_axis(
          topo_in, topo_out, nprocs);
      EXPECT_EQ(trans_axis, ref_trans_axis)
          << error_trans_axis(topo_in, topo_out, nprocs, ref_trans_axis);
    }

    std::vector<topo3D_and_ref_type> topo3D_failure_test_cases = {
        {topo1, topo2, 0}, {topo2, topo1, 1}};

    for (const auto& [topo_in, topo_out, ref_trans_axis] :
         topo3D_failure_test_cases) {
      EXPECT_THROW(
          {
            [[maybe_unused]] auto trans_axis =
                KokkosFFT::Distributed::Impl::compute_trans_axis(
                    topo_in, topo_out, nprocs);
          },
          std::runtime_error);
    }

    // 4D case
    std::vector<topo4D_and_ref_type> topo4D_test_cases = {{topo3, topo4, 0},
                                                          {topo4, topo3, 0}};

    for (const auto& [topo_in, topo_out, ref_trans_axis] : topo4D_test_cases) {
      auto trans_axis = KokkosFFT::Distributed::Impl::compute_trans_axis(
          topo_in, topo_out, np0);
      EXPECT_EQ(trans_axis, ref_trans_axis)
          << error_trans_axis(topo_in, topo_out, np0, ref_trans_axis);
    }
  }
}

void test_pencil_in_out_axes_3D(std::size_t nprocs) {
  using topo_type = std::array<std::size_t, 3>;
  using topo_and_ref_type =
      std::tuple<topo_type, topo_type, std::tuple<std::size_t, std::size_t>>;
  topo_type topo0{1, 1, nprocs}, topo1{1, nprocs, 1}, topo2{nprocs, 1, 1},
      topo3{nprocs, 1, 2}, topo4{nprocs, 2, 1};

  if (nprocs == 1) {
    // Failure tests because of size 1 case
    std::vector<topo_and_ref_type> topo_and_ref_vec = {
        {topo0, topo1, {1, 2}}, {topo0, topo2, {0, 2}}, {topo1, topo0, {2, 1}},
        {topo1, topo2, {0, 1}}, {topo2, topo0, {2, 0}}, {topo2, topo1, {1, 0}}};
    for (const auto& [topo_in, topo_out, ref_in_out] : topo_and_ref_vec) {
      EXPECT_THROW(
          {
            [[maybe_unused]] auto inout_axis =
                KokkosFFT::Distributed::Impl::pencil_in_out_axes(topo_in,
                                                                 topo_out);
          },
          std::runtime_error);
    }
  } else {
    std::vector<topo_and_ref_type> topo_and_ref_vec = {
        {topo0, topo1, {1, 2}}, {topo0, topo2, {0, 2}}, {topo1, topo0, {2, 1}},
        {topo1, topo2, {0, 1}}, {topo2, topo0, {2, 0}}, {topo2, topo1, {1, 0}},
        {topo3, topo4, {1, 2}}, {topo4, topo3, {2, 1}}};
    for (const auto& [topo_in, topo_out, ref_inout_axes] : topo_and_ref_vec) {
      auto inout_axes =
          KokkosFFT::Distributed::Impl::pencil_in_out_axes(topo_in, topo_out);
      EXPECT_EQ(inout_axes, ref_inout_axes)
          << error_in_out_axes(topo_in, topo_out, ref_inout_axes);
    }
  }

  // Failure tests because of shape mismatch (or size 1 case)
  std::vector<topo_and_ref_type> topo_failure_test_cases = {
      {topo3, topo0, {0, 1}}, {topo3, topo1, {0, 1}}, {topo3, topo2, {0, 2}}};
  for (const auto& [topo_in, topo_out, ref_inout_axes] :
       topo_failure_test_cases) {
    EXPECT_THROW(
        {
          [[maybe_unused]] auto inout_axes =
              KokkosFFT::Distributed::Impl::pencil_in_out_axes(topo_in,
                                                               topo_out);
        },
        std::runtime_error);
  }
}

void test_get_mid_array_pencil_3D(std::size_t nprocs) {
  using topo_type         = std::array<std::size_t, 3>;
  using topo_and_ref_type = std::tuple<topo_type, topo_type, topo_type>;
  topo_type topo0{nprocs, 1, 8}, topo1{nprocs, 8, 1}, topo2{8, nprocs, 1},
      topo3{1, 2, nprocs}, topo4{2, nprocs, 1};

  if (nprocs == 1) {
    // Failure tests because only two elements differ
    std::vector<topo_and_ref_type> topo_and_ref_vec = {
        {topo0, topo1, topo_type{}}, {topo0, topo2, topo_type{}},
        {topo1, topo0, topo_type{}}, {topo1, topo2, topo_type{}},
        {topo2, topo0, topo_type{}}, {topo2, topo1, topo_type{}}};
    for (const auto& [topo_in, topo_out, ref_mid] : topo_and_ref_vec) {
      EXPECT_THROW(
          {
            [[maybe_unused]] auto mid =
                KokkosFFT::Distributed::Impl::propose_mid_array(topo_in,
                                                                topo_out);
          },
          std::runtime_error);
    }
  } else {
    // Failure tests because only two elements differ
    std::vector<topo_and_ref_type> topo_failure_test_cases = {
        {topo0, topo1, topo_type{}},
        {topo1, topo0, topo_type{}},
        {topo1, topo2, topo_type{}},
        {topo2, topo1, topo_type{}}};
    for (const auto& [topo_in, topo_out, ref_mid] : topo_failure_test_cases) {
      EXPECT_THROW(
          {
            [[maybe_unused]] auto mid =
                KokkosFFT::Distributed::Impl::propose_mid_array(topo_in,
                                                                topo_out);
          },
          std::runtime_error);
    }
    topo_type ref_mid02{1, nprocs, 8}, ref_mid34{2, 1, nprocs};
    std::vector<topo_and_ref_type> topo_test_cases = {
        {topo0, topo2, ref_mid02},
        {topo2, topo0, ref_mid02},
        {topo3, topo4, ref_mid34},
        {topo4, topo3, ref_mid34}};
    for (const auto& [topo_in, topo_out, ref_mid] : topo_test_cases) {
      auto mid =
          KokkosFFT::Distributed::Impl::propose_mid_array(topo_in, topo_out);
      EXPECT_EQ(mid, ref_mid) << error_mid_topology(topo_in, topo_out, ref_mid);
    }
  }
}

void test_get_mid_array_pencil_4D(std::size_t nprocs) {
  using topo_type         = std::array<std::size_t, 4>;
  using topo_and_ref_type = std::tuple<topo_type, topo_type, topo_type>;
  topo_type topo0{1, 1, nprocs, 8}, topo1{1, nprocs, 1, 8},
      topo2{1, 8, nprocs, 1}, topo3{1, nprocs, 8, 1}, topo4{1, 8, 1, nprocs},
      topo5{1, 1, 8, nprocs};

  if (nprocs == 1) {
    // Failure tests because only two elements differ
    std::vector<topo_and_ref_type> topo_and_ref_vec = {
        {topo0, topo1, topo_type{}}, {topo0, topo2, topo_type{}},
        {topo1, topo0, topo_type{}}, {topo1, topo2, topo_type{}},
        {topo2, topo0, topo_type{}}, {topo2, topo1, topo_type{}}};
    for (const auto& [topo_in, topo_out, ref_mid] : topo_and_ref_vec) {
      EXPECT_THROW(
          {
            [[maybe_unused]] auto mid =
                KokkosFFT::Distributed::Impl::propose_mid_array(topo_in,
                                                                topo_out);
          },
          std::runtime_error);
    }
  } else {
    // Failure tests because only two elements differ
    std::vector<topo_and_ref_type> topo_failure_test_cases = {
        {topo0, topo1, topo_type{}}, {topo0, topo2, topo_type{}},
        {topo0, topo5, topo_type{}}, {topo1, topo3, topo_type{}},
        {topo1, topo4, topo_type{}}, {topo2, topo3, topo_type{}},
        {topo2, topo4, topo_type{}}, {topo3, topo5, topo_type{}},
        {topo4, topo5, topo_type{}}};
    for (const auto& [topo_in, topo_out, ref_mid] : topo_failure_test_cases) {
      EXPECT_THROW(
          {
            [[maybe_unused]] auto mid =
                KokkosFFT::Distributed::Impl::propose_mid_array(topo_in,
                                                                topo_out);
          },
          std::runtime_error);
    }

    std::vector<topo_and_ref_type> topo_test_cases = {
        {topo0, topo3, topo1}, {topo0, topo4, topo2}, {topo1, topo2, topo0},
        {topo1, topo5, topo3}, {topo2, topo5, topo4}, {topo3, topo4, topo5}};
    for (const auto& [topo_in, topo_out, ref_mid] : topo_test_cases) {
      auto mid =
          KokkosFFT::Distributed::Impl::propose_mid_array(topo_in, topo_out);
      EXPECT_EQ(mid, ref_mid) << error_mid_topology(topo_in, topo_out, ref_mid);
    }
  }
}

void test_get_all_slab_topologies1D_3DView(std::size_t nprocs) {
  using topo_type     = std::array<std::size_t, 3>;
  using axes_type     = std::array<std::size_t, 1>;
  using vec_topo_type = std::vector<topo_type>;
  using topo_and_ref_type =
      std::tuple<topo_type, topo_type, axes_type, vec_topo_type>;

  topo_type topo0{1, 1, nprocs}, topo1{1, nprocs, 1}, topo2{nprocs, 1, 1};
  axes_type axes0{0}, axes1{1}, axes2{2};
  std::vector<axes_type> all_axes{axes0, axes1, axes2};

  if (nprocs == 1) {
    for (const auto& axes : all_axes) {
      // Failure tests because these are shared topologies
      for (const auto& topo_in : vec_topo_type{topo0, topo1, topo2}) {
        for (const auto& topo_out : vec_topo_type{topo0, topo1, topo2}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto all_slab_topologies =
                    KokkosFFT::Distributed::Impl::get_all_slab_topologies(
                        topo_in, topo_out, axes);
              },
              std::runtime_error);
        }
      }
    }
  } else {
    std::vector<topo_and_ref_type> topo_test_cases = {
        {topo0, topo0, axes0, vec_topo_type{topo0}},
        {topo0, topo0, axes1, vec_topo_type{topo0}},
        {topo0, topo0, axes2, vec_topo_type{topo0, topo2, topo0}},
        {topo0, topo1, axes0, vec_topo_type{topo0, topo1}},
        {topo0, topo1, axes1, vec_topo_type{topo0, topo1}},
        {topo0, topo1, axes2, vec_topo_type{topo0, topo1}},
        {topo0, topo2, axes0, vec_topo_type{topo0, topo2}},
        {topo0, topo2, axes1, vec_topo_type{topo0, topo2}},
        {topo0, topo2, axes2, vec_topo_type{topo0, topo2}},
        {topo1, topo0, axes0, vec_topo_type{topo1, topo0}},
        {topo1, topo0, axes1, vec_topo_type{topo1, topo0}},
        {topo1, topo0, axes2, vec_topo_type{topo1, topo0}},
        {topo1, topo1, axes0, vec_topo_type{topo1}},
        {topo1, topo1, axes1, vec_topo_type{topo1, topo2, topo1}},
        {topo1, topo1, axes2, vec_topo_type{topo1}},
        {topo1, topo2, axes0, vec_topo_type{topo1, topo2}},
        {topo1, topo2, axes1, vec_topo_type{topo1, topo2}},
        {topo1, topo2, axes2, vec_topo_type{topo1, topo2}},
        {topo2, topo0, axes0, vec_topo_type{topo2, topo0}},
        {topo2, topo0, axes1, vec_topo_type{topo2, topo0}},
        {topo2, topo0, axes2, vec_topo_type{topo2, topo0}},
        {topo2, topo1, axes0, vec_topo_type{topo2, topo1}},
        {topo2, topo1, axes1, vec_topo_type{topo2, topo1}},
        {topo2, topo1, axes2, vec_topo_type{topo2, topo1}},
        {topo2, topo2, axes0, vec_topo_type{topo2, topo1, topo2}},
        {topo2, topo2, axes1, vec_topo_type{topo2}},
        {topo2, topo2, axes2, vec_topo_type{topo2}}};
    for (const auto& [topo_in, topo_out, axes, ref_topos] : topo_test_cases) {
      auto topos = KokkosFFT::Distributed::Impl::get_all_slab_topologies(
          topo_in, topo_out, axes);
      EXPECT_EQ(topos, ref_topos)
          << error_all_topologies(topo_in, topo_out, axes, topos, ref_topos);
    }
  }
}

void test_get_all_slab_topologies2D_2DView(std::size_t nprocs) {
  using topo_type     = std::array<std::size_t, 2>;
  using axes_type     = std::array<std::size_t, 2>;
  using vec_topo_type = std::vector<topo_type>;
  using topo_and_ref_type =
      std::tuple<topo_type, topo_type, axes_type, vec_topo_type>;
  topo_type topo0{1, nprocs}, topo1{nprocs, 1};

  axes_type axes01{0, 1}, axes10{1, 0};
  std::vector<axes_type> all_axes{axes01, axes10};

  if (nprocs == 1) {
    for (const auto& axes : all_axes) {
      // Failure tests because these are shared topologies
      for (const auto& topo_in : vec_topo_type{topo0, topo1}) {
        for (const auto& topo_out : vec_topo_type{topo0, topo1}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto all_slab_topologies =
                    KokkosFFT::Distributed::Impl::get_all_slab_topologies(
                        topo_in, topo_out, axes);
              },
              std::runtime_error);
        }
      }
    }
  } else {
    std::vector<topo_and_ref_type> topo_test_cases = {
        {topo0, topo0, axes01, vec_topo_type{topo0, topo1, topo0}},
        {topo0, topo0, axes10, vec_topo_type{topo0, topo1, topo0}},
        {topo0, topo1, axes01, vec_topo_type{topo0, topo1, topo0, topo1}},
        {topo0, topo1, axes10, vec_topo_type{topo0, topo1}},
        {topo1, topo0, axes01, vec_topo_type{topo1, topo0}},
        {topo1, topo0, axes10, vec_topo_type{topo1, topo0, topo1, topo0}},
        {topo1, topo1, axes01, vec_topo_type{topo1, topo0, topo1}},
        {topo1, topo1, axes10, vec_topo_type{topo1, topo0, topo1}}};
    for (const auto& [topo_in, topo_out, axes, ref_topos] : topo_test_cases) {
      auto topos = KokkosFFT::Distributed::Impl::get_all_slab_topologies(
          topo_in, topo_out, axes);
      EXPECT_EQ(topos, ref_topos)
          << error_all_topologies(topo_in, topo_out, axes, topos, ref_topos);
    }
  }
}

void test_get_all_slab_topologies2D_3DView(std::size_t nprocs) {
  using topo_type     = std::array<std::size_t, 3>;
  using axes_type     = std::array<std::size_t, 2>;
  using vec_topo_type = std::vector<topo_type>;
  using topo_and_ref_type =
      std::tuple<topo_type, topo_type, axes_type, vec_topo_type>;
  topo_type topo0{1, 1, nprocs}, topo1{1, nprocs, 1}, topo2{nprocs, 1, 1};

  axes_type axes01{0, 1}, axes02{0, 2}, axes10{1, 0}, axes12{1, 2},
      axes20{2, 0}, axes21{2, 1};

  std::vector<axes_type> all_axes{axes01, axes02, axes10,
                                  axes12, axes20, axes21};

  if (nprocs == 1) {
    for (const auto& axes : all_axes) {
      // Failure tests because these are shared topologies
      for (const auto& topo_in : vec_topo_type{topo0, topo1, topo2}) {
        for (const auto& topo_out : vec_topo_type{topo0, topo1, topo2}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto all_slab_topologies =
                    KokkosFFT::Distributed::Impl::get_all_slab_topologies(
                        topo_in, topo_out, axes);
              },
              std::runtime_error);
        }
      }
    }
  } else {
    std::vector<topo_and_ref_type> topo_test_cases = {
        {topo0, topo0, axes01, vec_topo_type{topo0}},
        {topo0, topo0, axes02, vec_topo_type{topo0, topo1, topo0}},
        {topo0, topo0, axes10, vec_topo_type{topo0}},
        {topo0, topo0, axes12, vec_topo_type{topo0, topo2, topo0}},
        {topo0, topo0, axes20, vec_topo_type{topo0, topo2, topo0}},
        {topo0, topo0, axes21, vec_topo_type{topo0, topo2, topo0}},
        {topo0, topo1, axes01, vec_topo_type{topo0, topo1}},
        {topo0, topo1, axes02, vec_topo_type{topo0, topo1}},
        {topo0, topo1, axes10, vec_topo_type{topo0, topo1}},
        {topo0, topo1, axes12, vec_topo_type{topo0, topo2, topo1}},
        {topo0, topo1, axes20, vec_topo_type{topo0, topo1}},
        {topo0, topo1, axes21, vec_topo_type{topo0, topo1}},
        {topo0, topo2, axes01, vec_topo_type{topo0, topo2}},
        {topo0, topo2, axes02, vec_topo_type{topo0, topo1, topo2}},
        {topo0, topo2, axes10, vec_topo_type{topo0, topo2}},
        {topo0, topo2, axes12, vec_topo_type{topo0, topo2}},
        {topo0, topo2, axes20, vec_topo_type{topo0, topo2}},
        {topo0, topo2, axes21, vec_topo_type{topo0, topo2}},
        {topo1, topo0, axes01, vec_topo_type{topo1, topo0}},
        {topo1, topo0, axes02, vec_topo_type{topo1, topo0}},
        {topo1, topo0, axes10, vec_topo_type{topo1, topo0}},
        {topo1, topo0, axes12, vec_topo_type{topo1, topo0}},
        {topo1, topo0, axes20, vec_topo_type{topo1, topo0}},
        {topo1, topo0, axes21, vec_topo_type{topo1, topo2, topo0}},
        {topo1, topo1, axes01, vec_topo_type{topo1, topo0, topo1}},
        {topo1, topo1, axes02, vec_topo_type{topo1}},
        {topo1, topo1, axes10, vec_topo_type{topo1, topo2, topo1}},
        {topo1, topo1, axes12, vec_topo_type{topo1, topo2, topo1}},
        {topo1, topo1, axes20, vec_topo_type{topo1}},
        {topo1, topo1, axes21, vec_topo_type{topo1, topo2, topo1}},
        {topo1, topo2, axes01, vec_topo_type{topo1, topo0, topo2}},
        {topo1, topo2, axes02, vec_topo_type{topo1, topo2}},
        {topo1, topo2, axes10, vec_topo_type{topo1, topo2}},
        {topo1, topo2, axes12, vec_topo_type{topo1, topo2}},
        {topo1, topo2, axes20, vec_topo_type{topo1, topo2}},
        {topo1, topo2, axes21, vec_topo_type{topo1, topo2}},
        {topo2, topo0, axes01, vec_topo_type{topo2, topo0}},
        {topo2, topo0, axes02, vec_topo_type{topo2, topo0}},
        {topo2, topo0, axes10, vec_topo_type{topo2, topo0}},
        {topo2, topo0, axes12, vec_topo_type{topo2, topo0}},
        {topo2, topo0, axes20, vec_topo_type{topo2, topo1, topo0}},
        {topo2, topo0, axes21, vec_topo_type{topo2, topo0}},
        {topo2, topo1, axes01, vec_topo_type{topo2, topo1}},
        {topo2, topo1, axes02, vec_topo_type{topo2, topo1}},
        {topo2, topo1, axes10, vec_topo_type{topo2, topo0, topo1}},
        {topo2, topo1, axes12, vec_topo_type{topo2, topo1}},
        {topo2, topo1, axes20, vec_topo_type{topo2, topo1}},
        {topo2, topo1, axes21, vec_topo_type{topo2, topo1}},
        {topo2, topo2, axes01, vec_topo_type{topo2, topo1, topo2}},
        {topo2, topo2, axes02, vec_topo_type{topo2, topo1, topo2}},
        {topo2, topo2, axes10, vec_topo_type{topo2, topo0, topo2}},
        {topo2, topo2, axes12, vec_topo_type{topo2}},
        {topo2, topo2, axes20, vec_topo_type{topo2, topo1, topo2}},
        {topo2, topo2, axes21, vec_topo_type{topo2}}};
    for (const auto& [topo_in, topo_out, axes, ref_topos] : topo_test_cases) {
      auto topos = KokkosFFT::Distributed::Impl::get_all_slab_topologies(
          topo_in, topo_out, axes);
      EXPECT_EQ(topos, ref_topos)
          << error_all_topologies(topo_in, topo_out, axes, topos, ref_topos);
    }
  }
}

void test_get_all_slab_topologies3D_3DView(std::size_t nprocs) {
  using topo_type     = std::array<std::size_t, 3>;
  using axes_type     = std::array<std::size_t, 3>;
  using vec_topo_type = std::vector<topo_type>;
  using topo_and_ref_type =
      std::tuple<topo_type, topo_type, axes_type, vec_topo_type>;
  topo_type topo0{1, 1, nprocs}, topo1{1, nprocs, 1}, topo2{nprocs, 1, 1};

  axes_type axes012{0, 1, 2}, axes021{0, 2, 1}, axes102{1, 0, 2},
      axes120{1, 2, 0}, axes201{2, 0, 1}, axes210{2, 1, 0};

  std::vector<axes_type> all_axes{axes012, axes021, axes102,
                                  axes120, axes201, axes210};

  if (nprocs == 1) {
    for (const auto& axes : all_axes) {
      // Failure tests because these are shared topologies
      for (const auto& topo_in : vec_topo_type{topo0, topo1, topo2}) {
        for (const auto& topo_out : vec_topo_type{topo0, topo1, topo2}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto all_slab_topologies =
                    KokkosFFT::Distributed::Impl::get_all_slab_topologies(
                        topo_in, topo_out, axes);
              },
              std::runtime_error);
        }
      }
    }
  } else {
    std::vector<topo_and_ref_type> topo_test_cases = {
        {topo0, topo0, axes012, vec_topo_type{topo0, topo2, topo0}},
        {topo0, topo0, axes021, vec_topo_type{topo0, topo1, topo0}},
        {topo0, topo0, axes102, vec_topo_type{topo0, topo1, topo0}},
        {topo0, topo0, axes120, vec_topo_type{topo0, topo2, topo0}},
        {topo0, topo0, axes201, vec_topo_type{topo0, topo1, topo0}},
        {topo0, topo0, axes210, vec_topo_type{topo0, topo2, topo0}},
        {topo0, topo1, axes012, vec_topo_type{topo0, topo2, topo1}},
        {topo0, topo1, axes021, vec_topo_type{topo0, topo1}},
        {topo0, topo1, axes102, vec_topo_type{topo0, topo1, topo2, topo1}},
        {topo0, topo1, axes120, vec_topo_type{topo0, topo2, topo1}},
        {topo0, topo1, axes201, vec_topo_type{topo0, topo1}},
        {topo0, topo1, axes210, vec_topo_type{topo0, topo1}},
        {topo0, topo2, axes012, vec_topo_type{topo0, topo2, topo1, topo2}},
        {topo0, topo2, axes021, vec_topo_type{topo0, topo1, topo2}},
        {topo0, topo2, axes102, vec_topo_type{topo0, topo1, topo2}},
        {topo0, topo2, axes120, vec_topo_type{topo0, topo2}},
        {topo0, topo2, axes201, vec_topo_type{topo0, topo2}},
        {topo0, topo2, axes210, vec_topo_type{topo0, topo2}},
        {topo1, topo0, axes012, vec_topo_type{topo1, topo0}},
        {topo1, topo0, axes021, vec_topo_type{topo1, topo2, topo0}},
        {topo1, topo0, axes102, vec_topo_type{topo1, topo0}},
        {topo1, topo0, axes120, vec_topo_type{topo1, topo0}},
        {topo1, topo0, axes201, vec_topo_type{topo1, topo0, topo2, topo0}},
        {topo1, topo0, axes210, vec_topo_type{topo1, topo2, topo0}},
        {topo1, topo1, axes012, vec_topo_type{topo1, topo0, topo1}},
        {topo1, topo1, axes021, vec_topo_type{topo1, topo2, topo1}},
        {topo1, topo1, axes102, vec_topo_type{topo1, topo0, topo1}},
        {topo1, topo1, axes120, vec_topo_type{topo1, topo2, topo1}},
        {topo1, topo1, axes201, vec_topo_type{topo1, topo0, topo1}},
        {topo1, topo1, axes210, vec_topo_type{topo1, topo2, topo1}},
        {topo1, topo2, axes012, vec_topo_type{topo1, topo0, topo2}},
        {topo1, topo2, axes021, vec_topo_type{topo1, topo2, topo1, topo2}},
        {topo1, topo2, axes102, vec_topo_type{topo1, topo2}},
        {topo1, topo2, axes120, vec_topo_type{topo1, topo2}},
        {topo1, topo2, axes201, vec_topo_type{topo1, topo0, topo2}},
        {topo1, topo2, axes210, vec_topo_type{topo1, topo2}},
        {topo2, topo0, axes012, vec_topo_type{topo2, topo0}},
        {topo2, topo0, axes021, vec_topo_type{topo2, topo0}},
        {topo2, topo0, axes102, vec_topo_type{topo2, topo0}},
        {topo2, topo0, axes120, vec_topo_type{topo2, topo1, topo0}},
        {topo2, topo0, axes201, vec_topo_type{topo2, topo1, topo0}},
        {topo2, topo0, axes210, vec_topo_type{topo2, topo0, topo2, topo0}},
        {topo2, topo1, axes012, vec_topo_type{topo2, topo1}},
        {topo2, topo1, axes021, vec_topo_type{topo2, topo1}},
        {topo2, topo1, axes102, vec_topo_type{topo2, topo0, topo1}},
        {topo2, topo1, axes120, vec_topo_type{topo2, topo1, topo2, topo1}},
        {topo2, topo1, axes201, vec_topo_type{topo2, topo1}},
        {topo2, topo1, axes210, vec_topo_type{topo2, topo0, topo1}},
        {topo2, topo2, axes012, vec_topo_type{topo2, topo0, topo2}},
        {topo2, topo2, axes021, vec_topo_type{topo2, topo1, topo2}},
        {topo2, topo2, axes102, vec_topo_type{topo2, topo0, topo2}},
        {topo2, topo2, axes120, vec_topo_type{topo2, topo1, topo2}},
        {topo2, topo2, axes201, vec_topo_type{topo2, topo1, topo2}},
        {topo2, topo2, axes210, vec_topo_type{topo2, topo0, topo2}}};
    for (const auto& [topo_in, topo_out, axes, ref_topos] : topo_test_cases) {
      auto topos = KokkosFFT::Distributed::Impl::get_all_slab_topologies(
          topo_in, topo_out, axes);
      EXPECT_EQ(topos, ref_topos)
          << error_all_topologies(topo_in, topo_out, axes, topos, ref_topos);
    }
  }
}

void test_get_all_slab_topologies3D_4DView(std::size_t nprocs) {
  using topo_type     = std::array<std::size_t, 4>;
  using axes_type     = std::array<std::size_t, 3>;
  using vec_topo_type = std::vector<topo_type>;
  using topo_and_ref_type =
      std::tuple<topo_type, topo_type, axes_type, vec_topo_type>;

  topo_type topo0{1, 1, 1, nprocs}, topo1{1, 1, nprocs, 1},
      topo2{1, nprocs, 1, 1}, topo3{nprocs, 1, 1, 1};

  axes_type axes012{0, 1, 2}, axes021{0, 2, 1}, axes102{1, 0, 2},
      axes120{1, 2, 0}, axes201{2, 0, 1}, axes210{2, 1, 0}, axes123{1, 2, 3},
      axes132{1, 3, 2};

  std::vector<axes_type> all_axes{axes012, axes021, axes102, axes120,
                                  axes201, axes210, axes123, axes132};

  if (nprocs == 1) {
    for (const auto& axes : all_axes) {
      // Failure tests because these are shared topologies
      for (const auto& topo_in : vec_topo_type{topo0, topo1, topo2}) {
        for (const auto& topo_out : vec_topo_type{topo0, topo1, topo2}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto all_slab_topologies =
                    KokkosFFT::Distributed::Impl::get_all_slab_topologies(
                        topo_in, topo_out, axes);
              },
              std::runtime_error);
        }
      }
    }
  } else {
    std::vector<topo_and_ref_type> topo_test_cases = {
        {topo0, topo0, axes012, vec_topo_type{topo0}},
        {topo0, topo0, axes021, vec_topo_type{topo0}},
        {topo0, topo0, axes102, vec_topo_type{topo0}},
        {topo0, topo0, axes120, vec_topo_type{topo0}},
        {topo0, topo0, axes201, vec_topo_type{topo0}},
        {topo0, topo0, axes210, vec_topo_type{topo0}},
        {topo0, topo0, axes123, vec_topo_type{topo0, topo2, topo0}},
        {topo0, topo0, axes132, vec_topo_type{topo0, topo1, topo0}},
        {topo0, topo1, axes012, vec_topo_type{topo0, topo1}},
        {topo0, topo1, axes021, vec_topo_type{topo0, topo1}},
        {topo0, topo1, axes102, vec_topo_type{topo0, topo1}},
        {topo0, topo1, axes120, vec_topo_type{topo0, topo1}},
        {topo0, topo1, axes201, vec_topo_type{topo0, topo1}},
        {topo0, topo1, axes210, vec_topo_type{topo0, topo1}},
        {topo0, topo2, axes012, vec_topo_type{topo0, topo2}},
        {topo0, topo2, axes021, vec_topo_type{topo0, topo2}},
        {topo0, topo2, axes102, vec_topo_type{topo0, topo2}},
        {topo0, topo2, axes120, vec_topo_type{topo0, topo2}},
        {topo0, topo2, axes201, vec_topo_type{topo0, topo2}},
        {topo0, topo2, axes210, vec_topo_type{topo0, topo2}},
        {topo0, topo3, axes012, vec_topo_type{topo0, topo3}},
        {topo0, topo3, axes021, vec_topo_type{topo0, topo3}},
        {topo0, topo3, axes102, vec_topo_type{topo0, topo3}},
        {topo0, topo3, axes120, vec_topo_type{topo0, topo3}},
        {topo0, topo3, axes201, vec_topo_type{topo0, topo3}},
        {topo0, topo3, axes210, vec_topo_type{topo0, topo3}},
        {topo0, topo3, axes123, vec_topo_type{topo0, topo3}},
        {topo0, topo3, axes132, vec_topo_type{topo0, topo3}},
        {topo1, topo0, axes012, vec_topo_type{topo1, topo0}},
        {topo1, topo0, axes021, vec_topo_type{topo1, topo0}},
        {topo1, topo0, axes102, vec_topo_type{topo1, topo0}},
        {topo1, topo0, axes120, vec_topo_type{topo1, topo0}},
        {topo1, topo0, axes201, vec_topo_type{topo1, topo0}},
        {topo1, topo0, axes210, vec_topo_type{topo1, topo0}},
        {topo1, topo1, axes012, vec_topo_type{topo1, topo3, topo1}},
        {topo1, topo1, axes021, vec_topo_type{topo1, topo2, topo1}},
        {topo1, topo1, axes102, vec_topo_type{topo1, topo2, topo1}},
        {topo1, topo1, axes120, vec_topo_type{topo1, topo3, topo1}},
        {topo1, topo1, axes201, vec_topo_type{topo1, topo3, topo1}},
        {topo1, topo1, axes210, vec_topo_type{topo1, topo3, topo1}},
        {topo1, topo1, axes123, vec_topo_type{topo1, topo0, topo1}},
        {topo1, topo1, axes132, vec_topo_type{topo1, topo2, topo1}},
        {topo1, topo2, axes012, vec_topo_type{topo1, topo3, topo2}},
        {topo1, topo2, axes021, vec_topo_type{topo1, topo2}},
        {topo1, topo2, axes102, vec_topo_type{topo1, topo2, topo3, topo2}},
        {topo1, topo2, axes120, vec_topo_type{topo1, topo3, topo2}},
        {topo1, topo2, axes201, vec_topo_type{topo1, topo2}},
        {topo1, topo2, axes210, vec_topo_type{topo1, topo2}},
        {topo1, topo2, axes123, vec_topo_type{topo1, topo0, topo2}},
        {topo1, topo2, axes132, vec_topo_type{topo1, topo2, topo1, topo2}},
        {topo1, topo3, axes012, vec_topo_type{topo1, topo3, topo2, topo3}},
        {topo1, topo3, axes021, vec_topo_type{topo1, topo2, topo3}},
        {topo1, topo3, axes102, vec_topo_type{topo1, topo2, topo3}},
        {topo1, topo3, axes120, vec_topo_type{topo1, topo3}},
        {topo1, topo3, axes201, vec_topo_type{topo1, topo3}},
        {topo1, topo3, axes210, vec_topo_type{topo1, topo3}},
        {topo1, topo3, axes123, vec_topo_type{topo1, topo3}},
        {topo1, topo3, axes132, vec_topo_type{topo1, topo3}},
        {topo2, topo0, axes012, vec_topo_type{topo2, topo0}},
        {topo2, topo0, axes021, vec_topo_type{topo2, topo0}},
        {topo2, topo0, axes102, vec_topo_type{topo2, topo0}},
        {topo2, topo0, axes120, vec_topo_type{topo2, topo0}},
        {topo2, topo0, axes201, vec_topo_type{topo2, topo0}},
        {topo2, topo0, axes210, vec_topo_type{topo2, topo0}},
        {topo2, topo0, axes123, vec_topo_type{topo2, topo0}},
        {topo2, topo0, axes132, vec_topo_type{topo2, topo0}},
        {topo2, topo1, axes012, vec_topo_type{topo2, topo1}},
        {topo2, topo1, axes021, vec_topo_type{topo2, topo3, topo1}},
        {topo2, topo1, axes102, vec_topo_type{topo2, topo1}},
        {topo2, topo1, axes120, vec_topo_type{topo2, topo1}},
        {topo2, topo1, axes201, vec_topo_type{topo2, topo1, topo3, topo1}},
        {topo2, topo1, axes210, vec_topo_type{topo2, topo3, topo1}},
        {topo2, topo1, axes123, vec_topo_type{topo2, topo1}},
        {topo2, topo1, axes132, vec_topo_type{topo2, topo1}},
        {topo2, topo2, axes012, vec_topo_type{topo2, topo1, topo2}},
        {topo2, topo2, axes021, vec_topo_type{topo2, topo3, topo2}},
        {topo2, topo2, axes102, vec_topo_type{topo2, topo3, topo2}},
        {topo2, topo2, axes120, vec_topo_type{topo2, topo3, topo2}},
        {topo2, topo2, axes201, vec_topo_type{topo2, topo1, topo2}},
        {topo2, topo2, axes210, vec_topo_type{topo2, topo3, topo2}},
        {topo2, topo3, axes012, vec_topo_type{topo2, topo1, topo3}},
        {topo2, topo3, axes021, vec_topo_type{topo2, topo3, topo2, topo3}},
        {topo2, topo3, axes102, vec_topo_type{topo2, topo3}},
        {topo2, topo3, axes120, vec_topo_type{topo2, topo3}},
        {topo2, topo3, axes201, vec_topo_type{topo2, topo1, topo3}},
        {topo2, topo3, axes210, vec_topo_type{topo2, topo3}},
        {topo3, topo0, axes012, vec_topo_type{topo3, topo0}},
        {topo3, topo0, axes021, vec_topo_type{topo3, topo0}},
        {topo3, topo0, axes102, vec_topo_type{topo3, topo0}},
        {topo3, topo0, axes120, vec_topo_type{topo3, topo0}},
        {topo3, topo0, axes201, vec_topo_type{topo3, topo0}},
        {topo3, topo0, axes210, vec_topo_type{topo3, topo0}},
        {topo3, topo0, axes123, vec_topo_type{topo3, topo0}},
        {topo3, topo0, axes132, vec_topo_type{topo3, topo0}},
        {topo3, topo1, axes012, vec_topo_type{topo3, topo1}},
        {topo3, topo1, axes021, vec_topo_type{topo3, topo1}},
        {topo3, topo1, axes102, vec_topo_type{topo3, topo1}},
        {topo3, topo1, axes120, vec_topo_type{topo3, topo2, topo1}},
        {topo3, topo1, axes201, vec_topo_type{topo3, topo2, topo1}},
        {topo3, topo1, axes210, vec_topo_type{topo3, topo1, topo3, topo1}},
        {topo3, topo1, axes123, vec_topo_type{topo3, topo1}},
        {topo3, topo1, axes132, vec_topo_type{topo3, topo1}},
        {topo3, topo2, axes012, vec_topo_type{topo3, topo2}},
        {topo3, topo2, axes021, vec_topo_type{topo3, topo2}},
        {topo3, topo2, axes102, vec_topo_type{topo3, topo1, topo2}},
        {topo3, topo2, axes120, vec_topo_type{topo3, topo2, topo3, topo2}},
        {topo3, topo2, axes201, vec_topo_type{topo3, topo2}},
        {topo3, topo2, axes210, vec_topo_type{topo3, topo1, topo2}},
        {topo3, topo3, axes012, vec_topo_type{topo3, topo2, topo3}},
        {topo3, topo3, axes021, vec_topo_type{topo3, topo2, topo3}},
        {topo3, topo3, axes102, vec_topo_type{topo3, topo1, topo3}},
        {topo3, topo3, axes120, vec_topo_type{topo3, topo2, topo3}},
        {topo3, topo3, axes201, vec_topo_type{topo3, topo2, topo3}},
        {topo3, topo3, axes210, vec_topo_type{topo3, topo1, topo3}}};
    for (const auto& [topo_in, topo_out, axes, ref_topos] : topo_test_cases) {
      auto topos = KokkosFFT::Distributed::Impl::get_all_slab_topologies(
          topo_in, topo_out, axes);
      EXPECT_EQ(topos, ref_topos)
          << error_all_topologies(topo_in, topo_out, axes, topos, ref_topos);
    }
  }
}

void test_get_all_pencil_topologies1D_3DView(std::size_t nprocs) {
  using topo_type     = std::array<std::size_t, 3>;
  using axes_type     = std::array<std::size_t, 1>;
  using vec_topo_type = std::vector<topo_type>;
  using topo_r_type =
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutRight>;
  using topo_l_type =
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutLeft>;
  using vec_topo_r_type = std::vector<topo_r_type>;
  using vec_topo_l_type = std::vector<topo_l_type>;
  using vec_axis_type   = std::vector<std::size_t>;
  using vec_layout_type = std::vector<std::size_t>;
  using topo_rr_and_ref_type =
      std::tuple<topo_r_type, topo_r_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  using topo_rl_and_ref_type =
      std::tuple<topo_r_type, topo_l_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  using topo_lr_and_ref_type =
      std::tuple<topo_l_type, topo_r_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  using topo_ll_and_ref_type =
      std::tuple<topo_l_type, topo_l_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;

  std::size_t np0 = 4;

  topo_r_type topo0{1, nprocs, np0}, topo1{nprocs, 1, np0},
      topo2{nprocs, np0, 1};
  topo_l_type topo3{np0, nprocs, 1}, topo4{np0, 1, nprocs},
      topo5{1, np0, nprocs};
  topo_type ref_topo0 = topo0.array(), ref_topo1 = topo1.array(),
            ref_topo2 = topo2.array(), ref_topo3 = topo3.array(),
            ref_topo4 = topo4.array(), ref_topo5 = topo5.array();

  axes_type axes0{0}, axes1{1}, axes2{2};
  std::vector<axes_type> all_axes{axes0, axes1, axes2};

  if (nprocs == 1) {
    for (const auto& axes : all_axes) {
      // Failure tests because only two elements differ (slabs)
      for (const auto& topo_r_in : vec_topo_r_type{topo0, topo1, topo2}) {
        for (const auto& topo_r_out : vec_topo_r_type{topo0, topo1, topo2}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        topo_r_in, topo_r_out, axes);
              },
              std::runtime_error);
        }
        for (const auto& topo_l_out : vec_topo_l_type{topo3, topo4}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        topo_r_in, topo_l_out, axes);
              },
              std::runtime_error);
        }
      }
      for (const auto& topo_l_in : vec_topo_l_type{topo3, topo4}) {
        for (const auto& topo_r_out : vec_topo_r_type{topo0, topo1, topo2}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        topo_l_in, topo_r_out, axes);
              },
              std::runtime_error);
        }
        for (const auto& topo_l_out : vec_topo_l_type{topo3, topo4}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        topo_l_in, topo_l_out, axes);
              },
              std::runtime_error);
        }
      }
    }
  } else {
    std::vector<topo_rr_and_ref_type> topo_rr_test_cases = {
        {topo0, topo0, axes0, vec_axis_type{}, vec_layout_type{1},
         vec_topo_type{ref_topo0}},
        {topo0, topo0, axes1, vec_axis_type{0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0}},
        {topo0, topo0, axes2, vec_axis_type{1, 1}, vec_layout_type{1, 0, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0}},
        {topo0, topo1, axes0, vec_axis_type{0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo0, ref_topo1}},
        {topo0, topo1, axes1, vec_axis_type{0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo0, ref_topo1}},
        {topo0, topo1, axes2, vec_axis_type{1, 1, 0},
         vec_layout_type{1, 0, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo1}},
        {topo0, topo2, axes0, vec_axis_type{0, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes1, vec_axis_type{0, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes2, vec_axis_type{0, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2}},
        {topo1, topo0, axes0, vec_axis_type{0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo0}},
        {topo1, topo0, axes1, vec_axis_type{0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo0}},
        {topo1, topo0, axes2, vec_axis_type{1, 1, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo1, ref_topo0}},
        {topo1, topo1, axes0, vec_axis_type{0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1}},
        {topo1, topo1, axes1, vec_axis_type{}, vec_layout_type{1},
         vec_topo_type{ref_topo1}},
        {topo1, topo1, axes2, vec_axis_type{1, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo1}},
        {topo1, topo2, axes0, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1, ref_topo2}},
        {topo1, topo2, axes1, vec_axis_type{1}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo2}},
        {topo1, topo2, axes2, vec_axis_type{1}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo2}},
        {topo2, topo0, axes0, vec_axis_type{1, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo0}},
        {topo2, topo0, axes1, vec_axis_type{1, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo0}},
        {topo2, topo0, axes2, vec_axis_type{1, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo0}},
        {topo2, topo1, axes0, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 0, 1, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2, ref_topo1}},
        {topo2, topo1, axes1, vec_axis_type{1}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo2, ref_topo1}},
        {topo2, topo1, axes2, vec_axis_type{1}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo2, ref_topo1}},
        {topo2, topo2, axes0, vec_axis_type{0, 0}, vec_layout_type{1, 0, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2}},
        {topo2, topo2, axes1, vec_axis_type{1, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo2}},
        {topo2, topo2, axes2, vec_axis_type{}, vec_layout_type{1},
         vec_topo_type{ref_topo2}}};
    for (const auto& [topo_r_in, topo_r_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_rr_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              topo_r_in, topo_r_out, axes);
      auto ref_topo_and_axes =
          std::make_tuple(ref_topos, ref_axes, ref_layouts);
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_r_in, topo_r_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }

    std::vector<topo_rl_and_ref_type> topo_rl_test_cases = {
        {topo0, topo3, axes0, vec_axis_type{1}, vec_layout_type{1, 0},
         vec_topo_type{ref_topo0, ref_topo3}},
        {topo0, topo3, axes1, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 1, 1, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo3}},
        {topo0, topo3, axes2, vec_axis_type{1}, vec_layout_type{1, 0},
         vec_topo_type{ref_topo0, ref_topo3}},
        {topo0, topo4, axes0, vec_axis_type{1, 0}, vec_layout_type{1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4}},
        {topo0, topo4, axes1, vec_axis_type{1, 0}, vec_layout_type{1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4}},
        {topo0, topo4, axes2, vec_axis_type{1, 0}, vec_layout_type{1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4}}};
    for (const auto& [topo_r_in, topo_l_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_rl_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              topo_r_in, topo_l_out, axes);
      auto ref_topo_and_axes =
          std::make_tuple(ref_topos, ref_axes, ref_layouts);
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_r_in, topo_l_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }

    std::vector<topo_lr_and_ref_type> topo_lr_test_cases = {
        {topo3, topo0, axes0, vec_axis_type{1}, vec_layout_type{0, 1},
         vec_topo_type{ref_topo3, ref_topo0}},
        {topo3, topo0, axes1, vec_axis_type{0, 0, 1},
         vec_layout_type{0, 0, 0, 1},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo3, ref_topo0}},
        {topo3, topo0, axes2, vec_axis_type{1}, vec_layout_type{0, 1},
         vec_topo_type{ref_topo3, ref_topo0}},
        {topo4, topo0, axes0, vec_axis_type{0, 1}, vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo4, ref_topo3, ref_topo0}},
        {topo4, topo0, axes1, vec_axis_type{0, 1}, vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo4, ref_topo3, ref_topo0}},
        {topo4, topo0, axes2, vec_axis_type{0, 1}, vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo4, ref_topo3, ref_topo0}}};
    for (const auto& [topo_l_in, topo_r_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_lr_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              topo_l_in, topo_r_out, axes);
      auto ref_topo_and_axes =
          std::make_tuple(ref_topos, ref_axes, ref_layouts);
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_l_in, topo_r_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }

    std::vector<topo_ll_and_ref_type> topo_ll_test_cases = {
        {topo3, topo3, axes0, vec_axis_type{1, 1}, vec_layout_type{0, 1, 0},
         vec_topo_type{ref_topo3, ref_topo0, ref_topo3}},
        {topo3, topo3, axes1, vec_axis_type{0, 0}, vec_layout_type{0, 0, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo3}},
        {topo3, topo3, axes2, vec_axis_type{}, vec_layout_type{0},
         vec_topo_type{ref_topo3}}};
    for (const auto& [topo_l_in, topo_l_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_ll_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              topo_l_in, topo_l_out, axes);
      auto ref_topo_and_axes =
          std::make_tuple(ref_topos, ref_axes, ref_layouts);
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_l_in, topo_l_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }
  }
}

void test_get_all_pencil_topologies2D_3DView(std::size_t nprocs) {
  using topo_type     = std::array<std::size_t, 3>;
  using axes_type     = std::array<std::size_t, 2>;
  using vec_topo_type = std::vector<topo_type>;
  using topo_r_type =
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutRight>;
  using topo_l_type =
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutLeft>;
  using vec_topo_r_type = std::vector<topo_r_type>;
  using vec_topo_l_type = std::vector<topo_l_type>;
  using vec_axis_type   = std::vector<std::size_t>;
  using vec_layout_type = std::vector<std::size_t>;
  using topo_rr_and_ref_type =
      std::tuple<topo_r_type, topo_r_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  using topo_rl_and_ref_type =
      std::tuple<topo_r_type, topo_l_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  using topo_lr_and_ref_type =
      std::tuple<topo_l_type, topo_r_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  using topo_ll_and_ref_type =
      std::tuple<topo_l_type, topo_l_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  std::size_t np0 = 4;

  topo_r_type topo0{1, nprocs, np0}, topo1{nprocs, 1, np0},
      topo2{nprocs, np0, 1};
  topo_l_type topo3{np0, nprocs, 1}, topo4{np0, 1, nprocs},
      topo5{1, np0, nprocs};

  topo_type ref_topo0 = topo0.array(), ref_topo1 = topo1.array(),
            ref_topo2 = topo2.array(), ref_topo3 = topo3.array(),
            ref_topo4 = topo4.array(), ref_topo5 = topo5.array();

  axes_type axes01{0, 1}, axes02{0, 2}, axes10{1, 0}, axes12{1, 2},
      axes20{2, 0}, axes21{2, 1};
  std::vector<axes_type> all_axes{axes01, axes02, axes10,
                                  axes12, axes20, axes21};

  if (nprocs == 1) {
    for (const auto& axes : all_axes) {
      // Failure tests because only two elements differ (slabs)
      for (const auto& topo_r_in : vec_topo_r_type{topo0, topo1, topo2}) {
        for (const auto& topo_r_out : vec_topo_r_type{topo0, topo1, topo2}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        topo_r_in, topo_r_out, axes);
              },
              std::runtime_error);
        }
        for (const auto& topo_l_out : vec_topo_l_type{topo3, topo4, topo5}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        topo_r_in, topo_l_out, axes);
              },
              std::runtime_error);
        }
      }
      for (const auto& topo_l_in : vec_topo_l_type{topo3, topo4, topo5}) {
        for (const auto& topo_r_out : vec_topo_r_type{topo0, topo1, topo2}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        topo_l_in, topo_r_out, axes);
              },
              std::runtime_error);
        }
        for (const auto& topo_l_out : vec_topo_l_type{topo3, topo4, topo5}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        topo_l_in, topo_l_out, axes);
              },
              std::runtime_error);
        }
      }
    }
  } else {
    std::vector<topo_rr_and_ref_type> topo_rr_test_cases = {
        {topo0, topo0, axes01, vec_axis_type{0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0}},
        {topo0, topo0, axes02, vec_axis_type{1, 1}, vec_layout_type{1, 0, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0}},
        {topo0, topo0, axes10, vec_axis_type{0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0}},
        {topo0, topo0, axes12, vec_axis_type{1, 0, 0, 1},
         vec_layout_type{1, 0, 0, 0, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4, ref_topo3, ref_topo0}},
        {topo0, topo0, axes20, vec_axis_type{1, 1}, vec_layout_type{1, 0, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0}},
        {topo0, topo0, axes21, vec_axis_type{0, 1, 1, 0},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo1, ref_topo0}},
        {topo0, topo1, axes01, vec_axis_type{0, 0, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo1}},
        {topo0, topo1, axes02, vec_axis_type{1, 1, 0},
         vec_layout_type{1, 0, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo1}},
        {topo0, topo1, axes10, vec_axis_type{0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo0, ref_topo1}},
        {topo0, topo1, axes12, vec_axis_type{1, 1, 0},
         vec_layout_type{1, 0, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo1}},
        {topo0, topo1, axes20, vec_axis_type{1, 1, 0},
         vec_layout_type{1, 0, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo1}},
        {topo0, topo1, axes21, vec_axis_type{0, 1, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo1}},
        {topo0, topo2, axes01, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes02, vec_axis_type{1, 1, 0, 1},
         vec_layout_type{1, 0, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes10, vec_axis_type{0, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes12, vec_axis_type{1, 0, 1, 0},
         vec_layout_type{1, 0, 0, 0, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4, ref_topo5, ref_topo2}},
        {topo0, topo2, axes20, vec_axis_type{0, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes21, vec_axis_type{0, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2}},
        {topo1, topo0, axes01, vec_axis_type{0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo0}},
        {topo1, topo0, axes02, vec_axis_type{1, 1, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo1, ref_topo0}},
        {topo1, topo0, axes10, vec_axis_type{0, 0, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1, ref_topo0}},
        {topo1, topo0, axes12, vec_axis_type{1, 1, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo1, ref_topo0}},
        {topo1, topo0, axes20, vec_axis_type{0, 1, 1},
         vec_layout_type{1, 1, 0, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo3, ref_topo0}},
        {topo1, topo0, axes21, vec_axis_type{1, 1, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo1, ref_topo0}},
        {topo1, topo1, axes01, vec_axis_type{0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1}},
        {topo1, topo1, axes02, vec_axis_type{1, 0, 0, 1},
         vec_layout_type{1, 1, 0, 1, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo5, ref_topo2, ref_topo1}},
        {topo1, topo1, axes10, vec_axis_type{0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1}},
        {topo1, topo1, axes12, vec_axis_type{1, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo1}},
        {topo1, topo1, axes20, vec_axis_type{0, 1, 1, 0},
         vec_layout_type{1, 1, 0, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo3, ref_topo0, ref_topo1}},
        {topo1, topo1, axes21, vec_axis_type{1, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo1}},
        {topo1, topo2, axes01, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1, ref_topo2}},
        {topo1, topo2, axes02, vec_axis_type{1, 0, 0},
         vec_layout_type{1, 1, 0, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo5, ref_topo2}},
        {topo1, topo2, axes10, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1, ref_topo2}},
        {topo1, topo2, axes12, vec_axis_type{1, 1, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo1, ref_topo2}},
        {topo1, topo2, axes20, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1, ref_topo2}},
        {topo1, topo2, axes21, vec_axis_type{1}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo2}},
        {topo2, topo0, axes01, vec_axis_type{1, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo0}},
        {topo2, topo0, axes02, vec_axis_type{1, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo0}},
        {topo2, topo0, axes10, vec_axis_type{0, 1, 0, 1},
         vec_layout_type{1, 0, 0, 0, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo4, ref_topo3, ref_topo0}},
        {topo2, topo0, axes12, vec_axis_type{1, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo0}},
        {topo2, topo0, axes20, vec_axis_type{0, 0, 1, 0},
         vec_layout_type{1, 0, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2, ref_topo1, ref_topo0}},
        {topo2, topo0, axes21, vec_axis_type{1, 1, 1, 0},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo2, ref_topo1, ref_topo0}},
        {topo2, topo1, axes01, vec_axis_type{1, 0, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo0, ref_topo1}},
        {topo2, topo1, axes02, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 0, 1, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2, ref_topo1}},
        {topo2, topo1, axes10, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 0, 1, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2, ref_topo1}},
        {topo2, topo1, axes12, vec_axis_type{1}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo2, ref_topo1}},
        {topo2, topo1, axes20, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 0, 1, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2, ref_topo1}},
        {topo2, topo1, axes21, vec_axis_type{1, 1, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo2, ref_topo1}},
        {topo2, topo2, axes01, vec_axis_type{1, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo0, ref_topo1, ref_topo2}},
        {topo2, topo2, axes02, vec_axis_type{0, 0}, vec_layout_type{1, 0, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2}},
        {topo2, topo2, axes10, vec_axis_type{0, 1, 1, 0},
         vec_layout_type{1, 0, 0, 0, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo4, ref_topo5, ref_topo2}},
        {topo2, topo2, axes12, vec_axis_type{1, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo2}},
        {topo2, topo2, axes20, vec_axis_type{0, 0}, vec_layout_type{1, 0, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2}},
        {topo2, topo2, axes21, vec_axis_type{1, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo2}}};
    for (const auto& [topo_r_in, topo_r_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_rr_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              topo_r_in, topo_r_out, axes);
      auto ref_topo_and_axes =
          std::make_tuple(ref_topos, ref_axes, ref_layouts);
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_r_in, topo_r_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }

    std::vector<topo_rl_and_ref_type> topo_rl_test_cases = {
        {topo0, topo3, axes01, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 1, 1, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo3}},
        {topo0, topo3, axes02, vec_axis_type{1, 1, 1},
         vec_layout_type{1, 0, 1, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo3}},
        {topo0, topo3, axes10, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 1, 1, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo3}},
        {topo0, topo3, axes12, vec_axis_type{1, 0, 0},
         vec_layout_type{1, 0, 0, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4, ref_topo3}},
        {topo0, topo3, axes20, vec_axis_type{1}, vec_layout_type{1, 0},
         vec_topo_type{ref_topo0, ref_topo3}},
        {topo0, topo3, axes21, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 1, 1, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo3}},
        {topo0, topo4, axes01, vec_axis_type{0, 0, 1, 0},
         vec_layout_type{1, 1, 1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo3, ref_topo4}},
        {topo0, topo4, axes02, vec_axis_type{1, 1, 1, 0},
         vec_layout_type{1, 0, 1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo3, ref_topo4}},
        {topo0, topo4, axes10, vec_axis_type{1, 0}, vec_layout_type{1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4}},
        {topo0, topo4, axes12, vec_axis_type{1, 0}, vec_layout_type{1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4}},
        {topo0, topo4, axes20, vec_axis_type{1, 0}, vec_layout_type{1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4}},
        {topo0, topo4, axes21, vec_axis_type{0, 1, 0, 1},
         vec_layout_type{1, 1, 1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo5, ref_topo4}}};
    for (const auto& [topo_r_in, topo_l_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_rl_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              topo_r_in, topo_l_out, axes);
      auto ref_topo_and_axes =
          std::make_tuple(ref_topos, ref_axes, ref_layouts);
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_r_in, topo_l_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }

    std::vector<topo_lr_and_ref_type> topo_lr_test_cases = {
        {topo3, topo0, axes01, vec_axis_type{0, 0, 1},
         vec_layout_type{0, 0, 0, 1},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo3, ref_topo0}},
        {topo3, topo0, axes02, vec_axis_type{1}, vec_layout_type{0, 1},
         vec_topo_type{ref_topo3, ref_topo0}},
        {topo3, topo0, axes10, vec_axis_type{1, 0, 0},
         vec_layout_type{0, 1, 1, 1},
         vec_topo_type{ref_topo3, ref_topo0, ref_topo1, ref_topo0}},
        {topo3, topo0, axes12, vec_axis_type{0, 0, 1},
         vec_layout_type{0, 0, 0, 1},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo3, ref_topo0}},
        {topo3, topo0, axes20, vec_axis_type{1, 1, 1},
         vec_layout_type{0, 1, 0, 1},
         vec_topo_type{ref_topo3, ref_topo0, ref_topo3, ref_topo0}},
        {topo3, topo0, axes21, vec_axis_type{0, 0, 1},
         vec_layout_type{0, 0, 0, 1},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo3, ref_topo0}},
        {topo4, topo0, axes01, vec_axis_type{0, 1}, vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo4, ref_topo3, ref_topo0}},
        {topo4, topo0, axes02, vec_axis_type{0, 1}, vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo4, ref_topo3, ref_topo0}},
        {topo4, topo0, axes10, vec_axis_type{1, 1, 0, 1},
         vec_layout_type{0, 0, 0, 0, 1},
         vec_topo_type{ref_topo4, ref_topo5, ref_topo4, ref_topo3, ref_topo0}},
        {topo4, topo0, axes12, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{0, 0, 0, 0, 1},
         vec_topo_type{ref_topo4, ref_topo3, ref_topo4, ref_topo3, ref_topo0}},
        {topo4, topo0, axes20, vec_axis_type{1, 0, 1, 0},
         vec_layout_type{0, 0, 1, 1, 1},
         vec_topo_type{ref_topo4, ref_topo5, ref_topo2, ref_topo1, ref_topo0}},
        {topo4, topo0, axes21, vec_axis_type{0, 1}, vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo4, ref_topo3, ref_topo0}}};
    for (const auto& [topo_l_in, topo_r_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_lr_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              topo_l_in, topo_r_out, axes);
      auto ref_topo_and_axes =
          std::make_tuple(ref_topos, ref_axes, ref_layouts);
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_l_in, topo_r_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }

    std::vector<topo_ll_and_ref_type> topo_ll_test_cases = {
        {topo3, topo3, axes01, vec_axis_type{0, 1, 1, 0},
         vec_layout_type{0, 0, 0, 0, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo5, ref_topo4, ref_topo3}},
        {topo3, topo3, axes02, vec_axis_type{1, 1}, vec_layout_type{0, 1, 0},
         vec_topo_type{ref_topo3, ref_topo0, ref_topo3}},
        {topo3, topo3, axes10, vec_axis_type{1, 0, 0, 1},
         vec_layout_type{0, 1, 1, 1, 0},
         vec_topo_type{ref_topo3, ref_topo0, ref_topo1, ref_topo0, ref_topo3}},
        {topo3, topo3, axes12, vec_axis_type{0, 0}, vec_layout_type{0, 0, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo3}},
        {topo3, topo3, axes20, vec_axis_type{1, 1}, vec_layout_type{0, 1, 0},
         vec_topo_type{ref_topo3, ref_topo0, ref_topo3}},
        {topo3, topo3, axes21, vec_axis_type{0, 0}, vec_layout_type{0, 0, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo3}},
        {topo3, topo4, axes01, vec_axis_type{0, 1, 1},
         vec_layout_type{0, 0, 0, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo5, ref_topo4}},
        {topo3, topo4, axes02, vec_axis_type{1, 1, 0},
         vec_layout_type{0, 1, 0, 0},
         vec_topo_type{ref_topo3, ref_topo0, ref_topo3, ref_topo4}},
        {topo3, topo4, axes10, vec_axis_type{1, 1, 0},
         vec_layout_type{0, 1, 0, 0},
         vec_topo_type{ref_topo3, ref_topo0, ref_topo3, ref_topo4}},
        {topo3, topo4, axes12, vec_axis_type{0}, vec_layout_type{0, 0},
         vec_topo_type{ref_topo3, ref_topo4}},
        {topo3, topo4, axes20, vec_axis_type{1, 1, 0},
         vec_layout_type{0, 1, 0, 0},
         vec_topo_type{ref_topo3, ref_topo0, ref_topo3, ref_topo4}},
        {topo3, topo4, axes21, vec_axis_type{0, 0, 0},
         vec_layout_type{0, 0, 0, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo3, ref_topo4}}};
    for (const auto& [topo_l_in, topo_l_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_ll_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              topo_l_in, topo_l_out, axes);
      auto ref_topo_and_axes =
          std::make_tuple(ref_topos, ref_axes, ref_layouts);
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_l_in, topo_l_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }
  }
}

void test_get_all_pencil_topologies2D_4DView(std::size_t nprocs) {
  using topo_type     = std::array<std::size_t, 4>;
  using axes_type     = std::array<std::size_t, 2>;
  using vec_topo_type = std::vector<topo_type>;
  using topo_r_type =
      KokkosFFT::Distributed::Topology<std::size_t, 4, Kokkos::LayoutRight>;
  using topo_l_type =
      KokkosFFT::Distributed::Topology<std::size_t, 4, Kokkos::LayoutLeft>;
  using vec_topo_r_type = std::vector<topo_r_type>;
  using vec_topo_l_type = std::vector<topo_l_type>;
  using vec_axis_type   = std::vector<std::size_t>;
  using vec_layout_type = std::vector<std::size_t>;
  using topo_rr_and_ref_type =
      std::tuple<topo_r_type, topo_r_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  using topo_rl_and_ref_type =
      std::tuple<topo_r_type, topo_l_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  using topo_lr_and_ref_type =
      std::tuple<topo_l_type, topo_r_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  using topo_ll_and_ref_type =
      std::tuple<topo_l_type, topo_l_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;

  std::size_t np0 = 4;
  topo_r_type topo0{1, 1, nprocs, np0}, topo1{1, nprocs, 1, np0},
      topo2{1, nprocs, np0, 1}, topo3{nprocs, 1, 1, np0},
      topo4{nprocs, 1, np0, 1}, topo5{nprocs, np0, 1, 1};

  topo_l_type topo6{1, np0, nprocs, 1}, topo7{1, np0, 1, nprocs},
      topo8{1, 1, np0, nprocs}, topo9{np0, 1, nprocs, 1},
      topo10{np0, nprocs, 1, 1};

  topo_type ref_topo0 = topo0.array(), ref_topo1 = topo1.array(),
            ref_topo2 = topo2.array(), ref_topo3 = topo3.array(),
            ref_topo4 = topo4.array(), ref_topo5 = topo5.array(),
            ref_topo6 = topo6.array(), ref_topo7 = topo7.array(),
            ref_topo8 = topo8.array(), ref_topo9 = topo9.array(),
            ref_topo10 = topo10.array();

  axes_type axes01{0, 1}, axes02{0, 2}, axes10{1, 0}, axes12{1, 2},
      axes20{2, 0}, axes21{2, 1}, axes13{1, 3}, axes23{2, 3}, axes03{0, 3};

  std::vector<axes_type> all_axes{axes01, axes02, axes10, axes12, axes20,
                                  axes21, axes13, axes23, axes03};

  if (nprocs == 1) {
    for (const auto& axes : all_axes) {
      // Failure tests because only two elements differ (slabs)
      for (const auto& topo_r_in :
           vec_topo_r_type{topo0, topo1, topo2, topo3, topo4, topo5}) {
        for (const auto& topo_r_out :
             vec_topo_r_type{topo0, topo1, topo2, topo3, topo4, topo5}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        topo_r_in, topo_r_out, axes);
              },
              std::runtime_error);
        }
        for (const auto& topo_l_out :
             vec_topo_l_type{topo6, topo7, topo8, topo9, topo10}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        topo_r_in, topo_l_out, axes);
              },
              std::runtime_error);
        }
      }
      for (const auto& topo_l_in :
           vec_topo_l_type{topo6, topo7, topo8, topo9, topo10}) {
        for (const auto& topo_r_out :
             vec_topo_r_type{topo0, topo1, topo2, topo3, topo4, topo5}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        topo_l_in, topo_r_out, axes);
              },
              std::runtime_error);
        }
        for (const auto& topo_l_out :
             vec_topo_l_type{topo6, topo7, topo8, topo9, topo10}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        topo_l_in, topo_l_out, axes);
              },
              std::runtime_error);
        }
      }
    }
  } else {
    std::vector<topo_rr_and_ref_type> topo_rr_test_cases = {
        {topo0, topo0, axes01, vec_axis_type{}, vec_layout_type{1},
         vec_topo_type{ref_topo0}},
        {topo0, topo0, axes02, vec_axis_type{0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0}},
        {topo0, topo0, axes10, vec_axis_type{}, vec_layout_type{1},
         vec_topo_type{ref_topo0}},
        {topo0, topo0, axes12, vec_axis_type{0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0}},
        {topo0, topo0, axes20, vec_axis_type{0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0}},
        {topo0, topo0, axes21, vec_axis_type{0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0}},
        {topo0, topo0, axes13, vec_axis_type{1, 1}, vec_layout_type{1, 0, 1},
         vec_topo_type{ref_topo0, ref_topo9, ref_topo0}},
        {topo0, topo0, axes23, vec_axis_type{1, 0, 0, 1},
         vec_layout_type{1, 0, 0, 0, 1},
         vec_topo_type{ref_topo0, ref_topo9, ref_topo10, ref_topo9, ref_topo0}},
        {topo0, topo0, axes03, vec_axis_type{1, 1}, vec_layout_type{1, 0, 1},
         vec_topo_type{ref_topo0, ref_topo9, ref_topo0}},
        {topo0, topo1, axes01, vec_axis_type{0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo0, ref_topo1}},
        {topo0, topo1, axes02, vec_axis_type{0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo0, ref_topo1}},
        {topo0, topo1, axes10, vec_axis_type{0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo0, ref_topo1}},
        {topo0, topo1, axes12, vec_axis_type{0, 0, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo1}},
        {topo0, topo1, axes20, vec_axis_type{0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo0, ref_topo1}},
        {topo0, topo1, axes21, vec_axis_type{0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo0, ref_topo1}},
        {topo0, topo1, axes13, vec_axis_type{1, 1, 0},
         vec_layout_type{1, 0, 1, 1},
         vec_topo_type{ref_topo0, ref_topo6, ref_topo0, ref_topo1}},
        {topo0, topo1, axes23, vec_axis_type{1, 1, 0},
         vec_layout_type{1, 0, 1, 1},
         vec_topo_type{ref_topo0, ref_topo6, ref_topo0, ref_topo1}},
        {topo0, topo1, axes03, vec_axis_type{1, 1, 0},
         vec_layout_type{1, 0, 1, 1},
         vec_topo_type{ref_topo0, ref_topo6, ref_topo0, ref_topo1}},
        {topo0, topo2, axes01, vec_axis_type{0, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes02, vec_axis_type{0, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes10, vec_axis_type{0, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes12, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes20, vec_axis_type{0, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes21, vec_axis_type{0, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes13, vec_axis_type{1, 1, 0, 1},
         vec_layout_type{1, 0, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo6, ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes23, vec_axis_type{1, 0, 1, 0},
         vec_layout_type{1, 0, 0, 0, 1},
         vec_topo_type{ref_topo0, ref_topo6, ref_topo7, ref_topo8, ref_topo2}},
        {topo0, topo2, axes03, vec_axis_type{0, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2}},
        {topo1, topo0, axes01, vec_axis_type{0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo0}},
        {topo1, topo0, axes02, vec_axis_type{0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo0}},
        {topo1, topo0, axes10, vec_axis_type{0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo0}},
        {topo1, topo0, axes12, vec_axis_type{0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo0}},
        {topo1, topo0, axes20, vec_axis_type{0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo0}},
        {topo1, topo0, axes21, vec_axis_type{0, 0, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1, ref_topo0}},
        {topo1, topo0, axes13, vec_axis_type{1, 1, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo1, ref_topo0}},
        {topo1, topo0, axes23, vec_axis_type{1, 1, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo1, ref_topo0}},
        {topo1, topo0, axes03, vec_axis_type{1, 1, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo1, ref_topo0}},
        {topo1, topo1, axes01, vec_axis_type{0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo1}},
        {topo1, topo1, axes02, vec_axis_type{}, vec_layout_type{1},
         vec_topo_type{ref_topo1}},
        {topo1, topo1, axes10, vec_axis_type{0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo1}},
        {topo1, topo1, axes12, vec_axis_type{0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo1}},
        {topo1, topo1, axes20, vec_axis_type{}, vec_layout_type{1},
         vec_topo_type{ref_topo1}},
        {topo1, topo1, axes21, vec_axis_type{0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo1}},
        {topo1, topo1, axes13, vec_axis_type{1, 0, 0, 1},
         vec_layout_type{1, 0, 0, 0, 1},
         vec_topo_type{ref_topo1, ref_topo10, ref_topo9, ref_topo10,
                       ref_topo1}},
        {topo1, topo1, axes23, vec_axis_type{1, 1}, vec_layout_type{1, 0, 1},
         vec_topo_type{ref_topo1, ref_topo10, ref_topo1}},
        {topo1, topo1, axes03, vec_axis_type{1, 1}, vec_layout_type{1, 0, 1},
         vec_topo_type{ref_topo1, ref_topo10, ref_topo1}},
        {topo1, topo2, axes01, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1, ref_topo2}},
        {topo1, topo2, axes02, vec_axis_type{1}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo2}},
        {topo1, topo2, axes10, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1, ref_topo2}},
        {topo1, topo2, axes12, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1, ref_topo2}},
        {topo1, topo2, axes20, vec_axis_type{1}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo2}},
        {topo1, topo2, axes21, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1, ref_topo2}},
        {topo1, topo2, axes13, vec_axis_type{1, 0, 0},
         vec_layout_type{1, 1, 0, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo8, ref_topo2}},
        {topo1, topo2, axes23, vec_axis_type{1, 1, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo1, ref_topo2}},
        {topo1, topo2, axes03, vec_axis_type{1}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo2}},
        {topo2, topo2, axes01, vec_axis_type{0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo4, ref_topo2}},
        {topo2, topo2, axes02, vec_axis_type{1, 1}, vec_layout_type{1, 0, 1},
         vec_topo_type{ref_topo2, ref_topo10, ref_topo2}},
        {topo2, topo2, axes10, vec_axis_type{0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo4, ref_topo2}},
        {topo2, topo2, axes12, vec_axis_type{1, 0, 0, 1},
         vec_layout_type{1, 0, 0, 0, 1},
         vec_topo_type{ref_topo2, ref_topo10, ref_topo9, ref_topo10,
                       ref_topo2}},
        {topo2, topo2, axes20, vec_axis_type{1, 1}, vec_layout_type{1, 0, 1},
         vec_topo_type{ref_topo2, ref_topo10, ref_topo2}},
        {topo2, topo2, axes21, vec_axis_type{0, 1, 1, 0},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo4, ref_topo5, ref_topo4, ref_topo2}},
        {topo2, topo2, axes13, vec_axis_type{0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo4, ref_topo2}},
        {topo2, topo2, axes23, vec_axis_type{1, 1}, vec_layout_type{1, 0, 1},
         vec_topo_type{ref_topo2, ref_topo10, ref_topo2}},
        {topo2, topo2, axes03, vec_axis_type{}, vec_layout_type{1},
         vec_topo_type{ref_topo2}}};
    for (const auto& [topo_r_in, topo_r_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_rr_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              topo_r_in, topo_r_out, axes);
      auto ref_topo_and_axes =
          std::make_tuple(ref_topos, ref_axes, ref_layouts);
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_r_in, topo_r_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }

    std::vector<topo_rl_and_ref_type> topo_rl_test_cases = {
        {topo0, topo6, axes01, vec_axis_type{1}, vec_layout_type{1, 0},
         vec_topo_type{ref_topo0, ref_topo6}},
        {topo0, topo6, axes02, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 1, 1, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo6}},
        {topo0, topo6, axes10, vec_axis_type{1}, vec_layout_type{1, 0},
         vec_topo_type{ref_topo0, ref_topo6}},
        {topo0, topo6, axes12, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 1, 1, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo6}},
        {topo0, topo6, axes20, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 1, 1, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo6}},
        {topo0, topo6, axes21, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 1, 1, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo6}},
        {topo0, topo6, axes13, vec_axis_type{1, 1, 1},
         vec_layout_type{1, 0, 1, 0},
         vec_topo_type{ref_topo0, ref_topo6, ref_topo0, ref_topo6}},
        {topo0, topo6, axes23, vec_axis_type{1, 0, 0},
         vec_layout_type{1, 0, 0, 0},
         vec_topo_type{ref_topo0, ref_topo6, ref_topo7, ref_topo6}},
        {topo0, topo6, axes03, vec_axis_type{1}, vec_layout_type{1, 0},
         vec_topo_type{ref_topo0, ref_topo6}}};
    for (const auto& [topo_r_in, topo_l_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_rl_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              topo_r_in, topo_l_out, axes);
      auto ref_topo_and_axes =
          std::make_tuple(ref_topos, ref_axes, ref_layouts);
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_r_in, topo_l_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }

    std::vector<topo_lr_and_ref_type> topo_lr_test_cases = {
        {topo7, topo0, axes01, vec_axis_type{0, 1}, vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo7, ref_topo6, ref_topo0}},
        {topo7, topo0, axes02, vec_axis_type{0, 1}, vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo7, ref_topo6, ref_topo0}},
        {topo7, topo0, axes10, vec_axis_type{0, 1}, vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo7, ref_topo6, ref_topo0}},
        {topo7, topo0, axes12, vec_axis_type{0, 1}, vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo7, ref_topo6, ref_topo0}},
        {topo7, topo0, axes20, vec_axis_type{0, 1}, vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo7, ref_topo6, ref_topo0}},
        {topo7, topo0, axes21, vec_axis_type{1, 1, 0, 1},
         vec_layout_type{0, 0, 0, 0, 1},
         vec_topo_type{ref_topo7, ref_topo8, ref_topo7, ref_topo6, ref_topo0}},
        {topo7, topo0, axes13, vec_axis_type{0, 1}, vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo7, ref_topo6, ref_topo0}},
        {topo7, topo0, axes23, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{0, 0, 0, 0, 1},
         vec_topo_type{ref_topo7, ref_topo6, ref_topo7, ref_topo6, ref_topo0}},
        {topo7, topo0, axes03, vec_axis_type{0, 1}, vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo7, ref_topo6, ref_topo0}}};
    for (const auto& [topo_l_in, topo_r_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_lr_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              topo_l_in, topo_r_out, axes);
      auto ref_topo_and_axes =
          std::make_tuple(ref_topos, ref_axes, ref_layouts);
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_l_in, topo_r_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }

    std::vector<topo_ll_and_ref_type> topo_ll_test_cases = {
        {topo7, topo6, axes01, vec_axis_type{1, 1, 0},
         vec_layout_type{0, 0, 0, 0},
         vec_topo_type{ref_topo7, ref_topo8, ref_topo7, ref_topo6}},
        {topo7, topo6, axes02, vec_axis_type{0}, vec_layout_type{0, 0},
         vec_topo_type{ref_topo7, ref_topo6}},
        {topo7, topo6, axes10, vec_axis_type{1, 1, 0},
         vec_layout_type{0, 0, 0, 0},
         vec_topo_type{ref_topo7, ref_topo8, ref_topo7, ref_topo6}},
        {topo7, topo6, axes12, vec_axis_type{1, 1, 0},
         vec_layout_type{0, 0, 0, 0},
         vec_topo_type{ref_topo7, ref_topo8, ref_topo7, ref_topo6}},
        {topo7, topo6, axes20, vec_axis_type{0}, vec_layout_type{0, 0},
         vec_topo_type{ref_topo7, ref_topo6}},
        {topo7, topo6, axes21, vec_axis_type{1, 1, 0},
         vec_layout_type{0, 0, 0, 0},
         vec_topo_type{ref_topo7, ref_topo8, ref_topo7, ref_topo6}},
        {topo7, topo6, axes13, vec_axis_type{0, 1, 1},
         vec_layout_type{0, 0, 1, 0},
         vec_topo_type{ref_topo7, ref_topo6, ref_topo0, ref_topo6}},
        {topo7, topo6, axes23, vec_axis_type{0, 0, 0},
         vec_layout_type{0, 0, 0, 0},
         vec_topo_type{ref_topo7, ref_topo6, ref_topo7, ref_topo6}},
        {topo7, topo6, axes03, vec_axis_type{0}, vec_layout_type{0, 0},
         vec_topo_type{ref_topo7, ref_topo6}}};
    for (const auto& [topo_l_in, topo_l_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_ll_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              topo_l_in, topo_l_out, axes);
      auto ref_topo_and_axes =
          std::make_tuple(ref_topos, ref_axes, ref_layouts);
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_l_in, topo_l_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }
  }
}

void test_get_all_pencil_topologies3D_3DView(std::size_t nprocs) {
  using topo_type     = std::array<std::size_t, 3>;
  using axes_type     = std::array<std::size_t, 3>;
  using vec_topo_type = std::vector<topo_type>;
  using topo_r_type =
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutRight>;
  using topo_l_type =
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutLeft>;
  using vec_topo_r_type = std::vector<topo_r_type>;
  using vec_topo_l_type = std::vector<topo_l_type>;
  using vec_axis_type   = std::vector<std::size_t>;
  using vec_layout_type = std::vector<std::size_t>;
  using topo_rr_and_ref_type =
      std::tuple<topo_r_type, topo_r_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  using topo_rl_and_ref_type =
      std::tuple<topo_r_type, topo_l_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  using topo_lr_and_ref_type =
      std::tuple<topo_l_type, topo_r_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  using topo_ll_and_ref_type =
      std::tuple<topo_l_type, topo_l_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  std::size_t np0 = 4;

  topo_r_type topo0{1, nprocs, np0}, topo1{nprocs, 1, np0},
      topo2{nprocs, np0, 1};
  topo_l_type topo3{np0, nprocs, 1}, topo4{np0, 1, nprocs},
      topo5{1, np0, nprocs};

  topo_type ref_topo0 = topo0.array(), ref_topo1 = topo1.array(),
            ref_topo2 = topo2.array(), ref_topo3 = topo3.array(),
            ref_topo4 = topo4.array(), ref_topo5 = topo5.array();

  axes_type axes012{0, 1, 2}, axes021{0, 2, 1}, axes102{1, 0, 2},
      axes120{1, 2, 0}, axes201{2, 0, 1}, axes210{2, 1, 0};

  std::vector<axes_type> all_axes{axes012, axes021, axes102,
                                  axes120, axes201, axes210};

  if (nprocs == 1) {
    for (const auto& axes : all_axes) {
      // Failure tests because only two elements differ (slabs)
      for (const auto& topo_r_in : vec_topo_r_type{topo0, topo1, topo2}) {
        for (const auto& topo_r_out : vec_topo_r_type{topo0, topo1, topo2}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        topo_r_in, topo_r_out, axes);
              },
              std::runtime_error);
        }
        for (const auto& topo_l_out : vec_topo_l_type{topo3, topo4, topo5}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        topo_r_in, topo_l_out, axes);
              },
              std::runtime_error);
        }
      }
      for (const auto& topo_l_in : vec_topo_l_type{topo3, topo4, topo5}) {
        for (const auto& topo_r_out : vec_topo_r_type{topo0, topo1, topo2}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        topo_l_in, topo_r_out, axes);
              },
              std::runtime_error);
        }
        for (const auto& topo_l_out : vec_topo_l_type{topo3, topo4, topo5}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        topo_l_in, topo_l_out, axes);
              },
              std::runtime_error);
        }
      }
    }
  } else {
    std::vector<topo_rr_and_ref_type> topo_rr_test_cases = {
        {topo0, topo0, axes012, vec_axis_type{1, 0, 0, 1},
         vec_layout_type{1, 0, 0, 0, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4, ref_topo3, ref_topo0}},
        {topo0, topo0, axes021, vec_axis_type{0, 1, 1, 0},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo1, ref_topo0}},
        {topo0, topo0, axes102, vec_axis_type{1, 1, 0, 0},
         vec_layout_type{1, 0, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo1, ref_topo0}},
        {topo0, topo0, axes120, vec_axis_type{1, 0, 0, 1},
         vec_layout_type{1, 0, 0, 0, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4, ref_topo3, ref_topo0}},
        {topo0, topo0, axes201, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{1, 1, 1, 0, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo3, ref_topo0}},
        {topo0, topo0, axes210, vec_axis_type{0, 1, 1, 0},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo1, ref_topo0}},
        {topo0, topo1, axes012, vec_axis_type{1, 0, 1, 0, 1},
         vec_layout_type{1, 0, 0, 0, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4, ref_topo5, ref_topo2,
                       ref_topo1}},
        {topo0, topo1, axes021, vec_axis_type{0, 1, 0, 0, 1},
         vec_layout_type{1, 1, 1, 0, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo5, ref_topo2,
                       ref_topo1}},
        {topo0, topo1, axes102, vec_axis_type{1, 1, 0},
         vec_layout_type{1, 0, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo1}},
        {topo0, topo1, axes120, vec_axis_type{1, 1, 0},
         vec_layout_type{1, 0, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo1}},
        {topo0, topo1, axes201, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{1, 1, 1, 0, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo3, ref_topo0,
                       ref_topo1}},
        {topo0, topo1, axes210, vec_axis_type{0, 1, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo1}},
        {topo0, topo2, axes012, vec_axis_type{1, 0, 1, 0},
         vec_layout_type{1, 0, 0, 0, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4, ref_topo5, ref_topo2}},
        {topo0, topo2, axes021, vec_axis_type{0, 1, 0, 0},
         vec_layout_type{1, 1, 1, 0, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo5, ref_topo2}},
        {topo0, topo2, axes102, vec_axis_type{1, 1, 0, 1},
         vec_layout_type{1, 0, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes120, vec_axis_type{1, 0, 1, 0},
         vec_layout_type{1, 0, 0, 0, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4, ref_topo5, ref_topo2}},
        {topo0, topo2, axes201, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes210, vec_axis_type{0, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2}},
        {topo1, topo0, axes012, vec_axis_type{1, 1, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo1, ref_topo0}},
        {topo1, topo0, axes021, vec_axis_type{1, 1, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo1, ref_topo0}},
        {topo1, topo0, axes102, vec_axis_type{1, 0, 1, 0, 1},
         vec_layout_type{1, 1, 0, 0, 0, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo5, ref_topo4, ref_topo3,
                       ref_topo0}},
        {topo1, topo0, axes120, vec_axis_type{0, 1, 0, 0, 1},
         vec_layout_type{1, 1, 0, 0, 0, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo3, ref_topo4, ref_topo3,
                       ref_topo0}},
        {topo1, topo0, axes201, vec_axis_type{0, 1, 1},
         vec_layout_type{1, 1, 0, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo3, ref_topo0}},
        {topo1, topo0, axes210, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{1, 1, 1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1, ref_topo2, ref_topo1,
                       ref_topo0}},
        {topo1, topo1, axes012, vec_axis_type{1, 1, 0, 0},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo1, ref_topo0, ref_topo1}},
        {topo1, topo1, axes021, vec_axis_type{1, 0, 0, 1},
         vec_layout_type{1, 1, 0, 1, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo5, ref_topo2, ref_topo1}},
        {topo1, topo1, axes102, vec_axis_type{1, 0, 0, 1},
         vec_layout_type{1, 1, 0, 1, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo5, ref_topo2, ref_topo1}},
        {topo1, topo1, axes120, vec_axis_type{0, 1, 1, 0},
         vec_layout_type{1, 1, 0, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo3, ref_topo0, ref_topo1}},
        {topo1, topo1, axes201, vec_axis_type{0, 1, 1, 0},
         vec_layout_type{1, 1, 0, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo3, ref_topo0, ref_topo1}},
        {topo1, topo1, axes210, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1, ref_topo2, ref_topo1}},
        {topo1, topo2, axes012, vec_axis_type{1, 1, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo1, ref_topo0, ref_topo1,
                       ref_topo2}},
        {topo1, topo2, axes021, vec_axis_type{1, 0, 0},
         vec_layout_type{1, 1, 0, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo5, ref_topo2}},
        {topo1, topo2, axes102, vec_axis_type{1, 0, 1, 1, 0},
         vec_layout_type{1, 1, 0, 0, 0, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo5, ref_topo4, ref_topo5,
                       ref_topo2}},
        {topo1, topo2, axes120, vec_axis_type{0, 1, 0, 1, 0},
         vec_layout_type{1, 1, 0, 0, 0, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo3, ref_topo4, ref_topo5,
                       ref_topo2}},
        {topo1, topo2, axes201, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1, ref_topo2}},
        {topo1, topo2, axes210, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1, ref_topo2}},
        {topo2, topo0, axes012, vec_axis_type{1, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo0}},
        {topo2, topo0, axes021, vec_axis_type{1, 1, 1, 0},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo2, ref_topo1, ref_topo0}},
        {topo2, topo0, axes102, vec_axis_type{0, 1, 0, 1},
         vec_layout_type{1, 0, 0, 0, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo4, ref_topo3, ref_topo0}},
        {topo2, topo0, axes120, vec_axis_type{0, 0, 1, 0},
         vec_layout_type{1, 0, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2, ref_topo1, ref_topo0}},
        {topo2, topo0, axes201, vec_axis_type{1, 0, 1, 1},
         vec_layout_type{1, 1, 1, 0, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo0, ref_topo3, ref_topo0}},
        {topo2, topo0, axes210, vec_axis_type{0, 1, 0, 1},
         vec_layout_type{1, 0, 0, 0, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo4, ref_topo3, ref_topo0}},
        {topo2, topo1, axes012, vec_axis_type{1, 0, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo0, ref_topo1}},
        {topo2, topo1, axes021, vec_axis_type{1, 1, 0, 0, 1},
         vec_layout_type{1, 1, 1, 0, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo2, ref_topo5, ref_topo2,
                       ref_topo1}},
        {topo2, topo1, axes102, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 0, 1, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2, ref_topo1}},
        {topo2, topo1, axes120, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 0, 1, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2, ref_topo1}},
        {topo2, topo1, axes201, vec_axis_type{1, 0, 1, 1, 0},
         vec_layout_type{1, 1, 1, 0, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo0, ref_topo3, ref_topo0,
                       ref_topo1}},
        {topo2, topo1, axes210, vec_axis_type{0, 1, 0, 1, 0},
         vec_layout_type{1, 0, 0, 0, 1, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo4, ref_topo3, ref_topo0,
                       ref_topo1}},
        {topo2, topo2, axes012, vec_axis_type{1, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo0, ref_topo1, ref_topo2}},
        {topo2, topo2, axes021, vec_axis_type{1, 1, 0, 0},
         vec_layout_type{1, 1, 1, 0, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo2, ref_topo5, ref_topo2}},
        {topo2, topo2, axes102, vec_axis_type{0, 1, 1, 0},
         vec_layout_type{1, 0, 0, 0, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo4, ref_topo5, ref_topo2}},
        {topo2, topo2, axes120, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{1, 0, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2, ref_topo1, ref_topo2}},
        {topo2, topo2, axes201, vec_axis_type{1, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo0, ref_topo1, ref_topo2}},
        {topo2, topo2, axes210, vec_axis_type{0, 1, 1, 0},
         vec_layout_type{1, 0, 0, 0, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo4, ref_topo5, ref_topo2}}};

    for (const auto& [topo_r_in, topo_r_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_rr_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              topo_r_in, topo_r_out, axes);
      auto ref_topo_and_axes =
          std::make_tuple(ref_topos, ref_axes, ref_layouts);
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_r_in, topo_r_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }

    std::vector<topo_rl_and_ref_type> topo_rl_test_cases = {
        {topo0, topo3, axes012, vec_axis_type{1, 0, 1, 1, 0},
         vec_layout_type{1, 0, 0, 0, 0, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4, ref_topo5, ref_topo4,
                       ref_topo3}},
        {topo0, topo3, axes021, vec_axis_type{0, 1, 0, 1, 0},
         vec_layout_type{1, 1, 1, 0, 0, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo5, ref_topo4,
                       ref_topo3}},
        {topo0, topo3, axes102, vec_axis_type{1, 1, 0, 0, 1},
         vec_layout_type{1, 0, 1, 1, 1, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo1, ref_topo0,
                       ref_topo3}},
        {topo0, topo3, axes120, vec_axis_type{1, 0, 0},
         vec_layout_type{1, 0, 0, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4, ref_topo3}},
        {topo0, topo3, axes201, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 1, 1, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo3}},
        {topo0, topo3, axes210, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 1, 1, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo3}},
        {topo0, topo4, axes012, vec_axis_type{1, 0, 1, 1},
         vec_layout_type{1, 0, 0, 0, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4, ref_topo5, ref_topo4}},
        {topo0, topo4, axes021, vec_axis_type{0, 1, 0, 1},
         vec_layout_type{1, 1, 1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo5, ref_topo4}},
        {topo0, topo4, axes102, vec_axis_type{1, 1, 1, 0},
         vec_layout_type{1, 0, 1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo3, ref_topo4}},
        {topo0, topo4, axes120, vec_axis_type{1, 0}, vec_layout_type{1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4}},
        {topo0, topo4, axes201, vec_axis_type{0, 0, 1, 0},
         vec_layout_type{1, 1, 1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo3, ref_topo4}},
        {topo0, topo4, axes210, vec_axis_type{0, 1, 0, 1},
         vec_layout_type{1, 1, 1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo5, ref_topo4}}};

    for (const auto& [topo_r_in, topo_l_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_rl_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              topo_r_in, topo_l_out, axes);
      auto ref_topo_and_axes =
          std::make_tuple(ref_topos, ref_axes, ref_layouts);
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_r_in, topo_l_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }

    std::vector<topo_lr_and_ref_type> topo_lr_test_cases = {
        {topo3, topo1, axes012, vec_axis_type{0, 1, 0, 1},
         vec_layout_type{0, 0, 0, 1, 1},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo5, ref_topo2, ref_topo1}},
        {topo3, topo1, axes021, vec_axis_type{0, 0, 1, 0},
         vec_layout_type{0, 0, 0, 1, 1},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo3, ref_topo0, ref_topo1}},
        {topo3, topo1, axes102, vec_axis_type{1, 0}, vec_layout_type{0, 1, 1},
         vec_topo_type{ref_topo3, ref_topo0, ref_topo1}},
        {topo3, topo1, axes120, vec_axis_type{1, 1, 1, 0},
         vec_layout_type{0, 1, 0, 1, 1},
         vec_topo_type{ref_topo3, ref_topo0, ref_topo3, ref_topo0, ref_topo1}},
        {topo3, topo1, axes201, vec_axis_type{0, 1, 0, 1},
         vec_layout_type{0, 0, 0, 1, 1},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo5, ref_topo2, ref_topo1}},
        {topo3, topo1, axes210, vec_axis_type{1, 0, 1, 1},
         vec_layout_type{0, 1, 1, 1, 1},
         vec_topo_type{ref_topo3, ref_topo0, ref_topo1, ref_topo2, ref_topo1}}};

    for (const auto& [topo_l_in, topo_r_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_lr_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              topo_l_in, topo_r_out, axes);
      auto ref_topo_and_axes =
          std::make_tuple(ref_topos, ref_axes, ref_layouts);
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_l_in, topo_r_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }

    std::vector<topo_ll_and_ref_type> topo_ll_test_cases = {
        {topo3, topo3, axes012, vec_axis_type{0, 1, 1, 0},
         vec_layout_type{0, 0, 0, 0, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo5, ref_topo4, ref_topo3}},
        {topo3, topo3, axes021, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{0, 0, 0, 1, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo3, ref_topo0, ref_topo3}},
        {topo3, topo3, axes102, vec_axis_type{1, 0, 0, 1},
         vec_layout_type{0, 1, 1, 1, 0},
         vec_topo_type{ref_topo3, ref_topo0, ref_topo1, ref_topo0, ref_topo3}},
        {topo3, topo3, axes120, vec_axis_type{1, 1, 0, 0},
         vec_layout_type{0, 1, 0, 0, 0},
         vec_topo_type{ref_topo3, ref_topo0, ref_topo3, ref_topo4, ref_topo3}},
        {topo3, topo3, axes201, vec_axis_type{0, 1, 1, 0},
         vec_layout_type{0, 0, 0, 0, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo5, ref_topo4, ref_topo3}},
        {topo3, topo3, axes210, vec_axis_type{1, 0, 0, 1},
         vec_layout_type{0, 1, 1, 1, 0},
         vec_topo_type{ref_topo3, ref_topo0, ref_topo1, ref_topo0, ref_topo3}},
        {topo3, topo4, axes012, vec_axis_type{0, 1, 1},
         vec_layout_type{0, 0, 0, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo5, ref_topo4}},
        {topo3, topo4, axes021, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{0, 0, 0, 1, 0, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo3, ref_topo0, ref_topo3,
                       ref_topo4}},
        {topo3, topo4, axes102, vec_axis_type{1, 1, 0},
         vec_layout_type{0, 1, 0, 0},
         vec_topo_type{ref_topo3, ref_topo0, ref_topo3, ref_topo4}},
        {topo3, topo4, axes120, vec_axis_type{1, 1, 0},
         vec_layout_type{0, 1, 0, 0},
         vec_topo_type{ref_topo3, ref_topo0, ref_topo3, ref_topo4}},
        {topo3, topo4, axes201, vec_axis_type{0, 1, 0, 0, 1},
         vec_layout_type{0, 0, 0, 1, 0, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo5, ref_topo2, ref_topo5,
                       ref_topo4}},
        {topo3, topo4, axes210, vec_axis_type{1, 0, 1, 0, 1},
         vec_layout_type{0, 1, 1, 1, 0, 0},
         vec_topo_type{ref_topo3, ref_topo0, ref_topo1, ref_topo2, ref_topo5,
                       ref_topo4}}};

    for (const auto& [topo_l_in, topo_l_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_ll_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              topo_l_in, topo_l_out, axes);
      auto ref_topo_and_axes =
          std::make_tuple(ref_topos, ref_axes, ref_layouts);
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_l_in, topo_l_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }
  }
}

void test_get_all_pencil_topologies3D_4DView(std::size_t nprocs) {
  using topology_type   = std::array<std::size_t, 4>;
  using topologies_type = std::vector<topology_type>;
  using topology_r_type =
      KokkosFFT::Distributed::Topology<std::size_t, 4, Kokkos::LayoutRight>;
  using topology_l_type =
      KokkosFFT::Distributed::Topology<std::size_t, 4, Kokkos::LayoutLeft>;
  using vec_axis_type        = std::vector<std::size_t>;
  using layouts_type         = std::vector<std::size_t>;
  std::size_t np0            = 4;
  topology_r_type topology0  = {1, 1, nprocs, np0},
                  topology1  = {1, nprocs, 1, np0},
                  topology3  = {1, nprocs, np0, 1},
                  topology6  = {nprocs, 1, 1, np0},
                  topology8  = {nprocs, 1, np0, 1},
                  topology10 = {nprocs, np0, 1, 1};

  topology_l_type topology2 = {1, np0, nprocs, 1},
                  topology4 = {1, np0, 1, nprocs},
                  topology5 = {1, 1, np0, nprocs},
                  topology7 = {np0, 1, nprocs, 1},
                  topology9 = {np0, nprocs, 1, 1};

  topology_type ref_topo0 = topology0.array(), ref_topo1 = topology1.array(),
                ref_topo2 = topology2.array(), ref_topo3 = topology3.array(),
                ref_topo4 = topology4.array(), ref_topo5 = topology5.array(),
                ref_topo6 = topology6.array(), ref_topo7 = topology7.array(),
                ref_topo8 = topology8.array(), ref_topo9 = topology9.array(),
                ref_topo10 = topology10.array();

  using axes_type   = std::array<std::size_t, 3>;
  axes_type axes012 = {0, 1, 2}, axes021 = {0, 2, 1}, axes102 = {1, 0, 2},
            axes120 = {1, 2, 0}, axes201 = {2, 0, 1}, axes210 = {2, 1, 0},
            axes123 = {1, 2, 3}, axes132 = {1, 3, 2};

  std::vector<axes_type> all_axes = {axes012, axes021, axes102, axes120,
                                     axes201, axes210, axes123, axes132};

  if (nprocs == 1) {
    for (const auto& axes : all_axes) {
      // Failure tests because only two elements differ (slabs)
      EXPECT_THROW(
          {
            [[maybe_unused]] auto topologies_and_axes_0_1 =
                KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                    topology0, topology1, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto topologies_and_axes_0_2 =
                KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                    topology0, topology2, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto topologies_and_axes_1_0 =
                KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                    topology1, topology0, axes);
          },
          std::runtime_error);
      EXPECT_THROW(
          {
            [[maybe_unused]] auto topologies_and_axes_2_0 =
                KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                    topology2, topology0, axes);
          },
          std::runtime_error);
    }
  } else {
    // topology0 to topology0
    auto topologies_and_axes_0_0_012 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology0, axes012);
    auto topologies_and_axes_0_0_021 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology0, axes021);
    auto topologies_and_axes_0_0_102 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology0, axes102);
    auto topologies_and_axes_0_0_120 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology0, axes120);
    auto topologies_and_axes_0_0_201 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology0, axes201);
    auto topologies_and_axes_0_0_210 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology0, axes210);
    auto topologies_and_axes_0_0_123 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology0, axes123);
    auto topologies_and_axes_0_0_132 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology0, axes132);

    auto ref_topologies_and_axes_0_0_012 =
        std::make_tuple(topologies_type{ref_topo0, ref_topo6, ref_topo0},
                        vec_axis_type{0, 0}, layouts_type{1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_0_012, ref_topologies_and_axes_0_0_012);

    auto ref_topologies_and_axes_0_0_021 =
        std::make_tuple(topologies_type{ref_topo0, ref_topo6, ref_topo0},
                        vec_axis_type{0, 0}, layouts_type{1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_0_021, ref_topologies_and_axes_0_0_021);

    auto ref_topologies_and_axes_0_0_102 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo6, ref_topo1, ref_topo0},
        vec_axis_type{0, 0, 0}, layouts_type{1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_0_102, ref_topologies_and_axes_0_0_102);

    auto ref_topologies_and_axes_0_0_120 =
        std::make_tuple(topologies_type{ref_topo0, ref_topo6, ref_topo0},
                        vec_axis_type{0, 0}, layouts_type{1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_0_120, ref_topologies_and_axes_0_0_120);

    auto ref_topologies_and_axes_0_0_201 =
        std::make_tuple(topologies_type{ref_topo0, ref_topo6, ref_topo0},
                        vec_axis_type{0, 0}, layouts_type{1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_0_201, ref_topologies_and_axes_0_0_201);

    auto ref_topologies_and_axes_0_0_210 =
        std::make_tuple(topologies_type{ref_topo0, ref_topo6, ref_topo0},
                        vec_axis_type{0, 0}, layouts_type{1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_0_210, ref_topologies_and_axes_0_0_210);

    auto ref_topologies_and_axes_0_0_123 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo7, ref_topo9, ref_topo7, ref_topo0},
        vec_axis_type{1, 0, 0, 1}, layouts_type{1, 0, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_0_0_123, ref_topologies_and_axes_0_0_123);

    auto ref_topologies_and_axes_0_0_132 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo6, ref_topo10, ref_topo6, ref_topo0},
        vec_axis_type{0, 1, 1, 0}, layouts_type{1, 1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_0_132, ref_topologies_and_axes_0_0_132);

    // topology0 to topology1
    auto topologies_and_axes_0_1_012 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology1, axes012);
    auto topologies_and_axes_0_1_021 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology1, axes021);
    auto topologies_and_axes_0_1_102 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology1, axes102);
    auto topologies_and_axes_0_1_120 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology1, axes120);
    auto topologies_and_axes_0_1_201 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology1, axes201);
    auto topologies_and_axes_0_1_210 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology1, axes210);
    auto topologies_and_axes_0_1_123 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology1, axes123);
    auto topologies_and_axes_0_1_132 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology1, axes132);

    auto ref_topologies_and_axes_0_1_012 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo1},
        vec_axis_type{0, 0, 0}, layouts_type{1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_1_012, ref_topologies_and_axes_0_1_012);

    auto ref_topologies_and_axes_0_1_021 =
        std::make_tuple(topologies_type{ref_topo0, ref_topo1}, vec_axis_type{0},
                        layouts_type{1, 1});
    EXPECT_EQ(topologies_and_axes_0_1_021, ref_topologies_and_axes_0_1_021);

    auto ref_topologies_and_axes_0_1_102 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo1},
        vec_axis_type{0, 0, 0}, layouts_type{1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_1_102, ref_topologies_and_axes_0_1_102);

    auto ref_topologies_and_axes_0_1_120 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo1},
        vec_axis_type{0, 0, 0}, layouts_type{1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_1_120, ref_topologies_and_axes_0_1_120);

    auto ref_topologies_and_axes_0_1_201 =
        std::make_tuple(topologies_type{ref_topo0, ref_topo1}, vec_axis_type{0},
                        layouts_type{1, 1});
    EXPECT_EQ(topologies_and_axes_0_1_201, ref_topologies_and_axes_0_1_201);

    auto ref_topologies_and_axes_0_1_210 =
        std::make_tuple(topologies_type{ref_topo0, ref_topo1}, vec_axis_type{0},
                        layouts_type{1, 1});
    EXPECT_EQ(topologies_and_axes_0_1_210, ref_topologies_and_axes_0_1_210);

    auto ref_topologies_and_axes_0_1_123 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2, ref_topo4, ref_topo5, ref_topo3,
                        ref_topo1},
        vec_axis_type{1, 0, 1, 0, 1}, layouts_type{1, 0, 0, 0, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_1_123, ref_topologies_and_axes_0_1_123);

    auto ref_topologies_and_axes_0_1_132 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo3, ref_topo5, ref_topo3,
                        ref_topo1},
        vec_axis_type{0, 1, 0, 0, 1}, layouts_type{1, 1, 1, 0, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_1_132, ref_topologies_and_axes_0_1_132);

    // topology0 to topology2
    auto topologies_and_axes_0_2_012 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology2, axes012);
    auto topologies_and_axes_0_2_021 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology2, axes021);
    auto topologies_and_axes_0_2_102 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology2, axes102);
    auto topologies_and_axes_0_2_120 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology2, axes120);
    auto topologies_and_axes_0_2_201 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology2, axes201);
    auto topologies_and_axes_0_2_210 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology2, axes210);
    auto topologies_and_axes_0_2_123 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology2, axes123);
    auto topologies_and_axes_0_2_132 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology2, axes132);

    auto ref_topologies_and_axes_0_2_012 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{0, 0, 1}, layouts_type{1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_0_2_012, ref_topologies_and_axes_0_2_012);

    auto ref_topologies_and_axes_0_2_021 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{0, 0, 1}, layouts_type{1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_0_2_021, ref_topologies_and_axes_0_2_021);

    auto ref_topologies_and_axes_0_2_102 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{0, 0, 1}, layouts_type{1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_0_2_102, ref_topologies_and_axes_0_2_102);

    auto ref_topologies_and_axes_0_2_120 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{0, 0, 1}, layouts_type{1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_0_2_120, ref_topologies_and_axes_0_2_120);

    auto ref_topologies_and_axes_0_2_201 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{0, 0, 1}, layouts_type{1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_0_2_201, ref_topologies_and_axes_0_2_201);

    auto ref_topologies_and_axes_0_2_210 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{0, 0, 1}, layouts_type{1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_0_2_210, ref_topologies_and_axes_0_2_210);
    auto ref_topologies_and_axes_0_2_123 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2, ref_topo4, ref_topo5, ref_topo4,
                        ref_topo2},
        vec_axis_type{1, 0, 1, 1, 0}, layouts_type{1, 0, 0, 0, 0, 0});
    EXPECT_EQ(topologies_and_axes_0_2_123, ref_topologies_and_axes_0_2_123);
    auto ref_topologies_and_axes_0_2_132 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo3, ref_topo5, ref_topo4,
                        ref_topo2},
        vec_axis_type{0, 1, 0, 1, 0}, layouts_type{1, 1, 1, 0, 0, 0});
    EXPECT_EQ(topologies_and_axes_0_2_132, ref_topologies_and_axes_0_2_132);

    // topology0 to topology3
    auto topologies_and_axes_0_3_012 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology3, axes012);
    auto topologies_and_axes_0_3_021 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology3, axes021);
    auto topologies_and_axes_0_3_102 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology3, axes102);
    auto topologies_and_axes_0_3_120 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology3, axes120);
    auto topologies_and_axes_0_3_201 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology3, axes201);
    auto topologies_and_axes_0_3_210 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology3, axes210);
    auto topologies_and_axes_0_3_123 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology3, axes123);
    auto topologies_and_axes_0_3_132 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology0, topology3, axes132);

    auto ref_topologies_and_axes_0_3_012 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo1, ref_topo3},
        vec_axis_type{0, 0, 0, 1}, layouts_type{1, 1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_3_012, ref_topologies_and_axes_0_3_012);

    auto ref_topologies_and_axes_0_3_021 =
        std::make_tuple(topologies_type{ref_topo0, ref_topo1, ref_topo3},
                        vec_axis_type{0, 1}, layouts_type{1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_3_021, ref_topologies_and_axes_0_3_021);

    auto ref_topologies_and_axes_0_3_102 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo1, ref_topo3},
        vec_axis_type{0, 0, 0, 1}, layouts_type{1, 1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_3_102, ref_topologies_and_axes_0_3_102);

    auto ref_topologies_and_axes_0_3_120 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo0, ref_topo1, ref_topo3},
        vec_axis_type{0, 0, 0, 1}, layouts_type{1, 1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_3_120, ref_topologies_and_axes_0_3_120);

    auto ref_topologies_and_axes_0_3_201 =
        std::make_tuple(topologies_type{ref_topo0, ref_topo1, ref_topo3},
                        vec_axis_type{0, 1}, layouts_type{1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_3_201, ref_topologies_and_axes_0_3_201);

    auto ref_topologies_and_axes_0_3_210 =
        std::make_tuple(topologies_type{ref_topo0, ref_topo1, ref_topo3},
                        vec_axis_type{0, 1}, layouts_type{1, 1, 1});
    EXPECT_EQ(topologies_and_axes_0_3_210, ref_topologies_and_axes_0_3_210);

    auto ref_topologies_and_axes_0_3_123 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo2, ref_topo4, ref_topo5, ref_topo3},
        vec_axis_type{1, 0, 1, 0}, layouts_type{1, 0, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_0_3_123, ref_topologies_and_axes_0_3_123);

    auto ref_topologies_and_axes_0_3_132 = std::make_tuple(
        topologies_type{ref_topo0, ref_topo1, ref_topo3, ref_topo5, ref_topo3},
        vec_axis_type{0, 1, 0, 0}, layouts_type{1, 1, 1, 0, 1});
    EXPECT_EQ(topologies_and_axes_0_3_132, ref_topologies_and_axes_0_3_132);

    // topology1 to topology0
    auto topologies_and_axes_1_0_012 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology1, topology0, axes012);
    auto topologies_and_axes_1_0_021 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology1, topology0, axes021);
    auto topologies_and_axes_1_0_102 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology1, topology0, axes102);
    auto topologies_and_axes_1_0_120 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology1, topology0, axes120);
    auto topologies_and_axes_1_0_201 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology1, topology0, axes201);
    auto topologies_and_axes_1_0_210 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology1, topology0, axes210);
    auto topologies_and_axes_1_0_123 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology1, topology0, axes123);
    auto topologies_and_axes_1_0_132 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology1, topology0, axes132);

    auto ref_topologies_and_axes_1_0_012 =
        std::make_tuple(topologies_type{ref_topo1, ref_topo0}, vec_axis_type{0},
                        layouts_type{1, 1});
    EXPECT_EQ(topologies_and_axes_1_0_012, ref_topologies_and_axes_1_0_012);

    auto ref_topologies_and_axes_1_0_021 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo1, ref_topo0},
        vec_axis_type{0, 0, 0}, layouts_type{1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_0_021, ref_topologies_and_axes_1_0_021);

    auto ref_topologies_and_axes_1_0_102 =
        std::make_tuple(topologies_type{ref_topo1, ref_topo0}, vec_axis_type{0},
                        layouts_type{1, 1});
    EXPECT_EQ(topologies_and_axes_1_0_102, ref_topologies_and_axes_1_0_102);

    auto ref_topologies_and_axes_1_0_120 =
        std::make_tuple(topologies_type{ref_topo1, ref_topo0}, vec_axis_type{0},
                        layouts_type{1, 1});
    EXPECT_EQ(topologies_and_axes_1_0_120, ref_topologies_and_axes_1_0_120);

    auto ref_topologies_and_axes_1_0_201 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo1, ref_topo0},
        vec_axis_type{0, 0, 0}, layouts_type{1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_0_201, ref_topologies_and_axes_1_0_201);

    auto ref_topologies_and_axes_1_0_210 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo1, ref_topo0},
        vec_axis_type{0, 0, 0}, layouts_type{1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_0_210, ref_topologies_and_axes_1_0_210);

    auto ref_topologies_and_axes_1_0_123 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo3, ref_topo1, ref_topo0},
        vec_axis_type{1, 1, 0}, layouts_type{1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_0_123, ref_topologies_and_axes_1_0_123);

    auto ref_topologies_and_axes_1_0_132 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo3, ref_topo1, ref_topo0},
        vec_axis_type{1, 1, 0}, layouts_type{1, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_0_132, ref_topologies_and_axes_1_0_132);

    // topology1 to topology1
    auto topologies_and_axes_1_1_012 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology1, topology1, axes012);
    auto topologies_and_axes_1_1_021 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology1, topology1, axes021);
    auto topologies_and_axes_1_1_102 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology1, topology1, axes102);
    auto topologies_and_axes_1_1_120 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology1, topology1, axes120);
    auto topologies_and_axes_1_1_201 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology1, topology1, axes201);
    auto topologies_and_axes_1_1_210 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology1, topology1, axes210);
    auto topologies_and_axes_1_1_123 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology1, topology1, axes123);
    auto topologies_and_axes_1_1_132 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology1, topology1, axes132);

    auto ref_topologies_and_axes_1_1_012 =
        std::make_tuple(topologies_type{ref_topo1, ref_topo6, ref_topo1},
                        vec_axis_type{0, 0}, layouts_type{1, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_1_012, ref_topologies_and_axes_1_1_012);

    auto ref_topologies_and_axes_1_1_021 =
        std::make_tuple(topologies_type{ref_topo1, ref_topo6, ref_topo1},
                        vec_axis_type{0, 0}, layouts_type{1, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_1_021, ref_topologies_and_axes_1_1_021);

    auto ref_topologies_and_axes_1_1_102 =
        std::make_tuple(topologies_type{ref_topo1, ref_topo6, ref_topo1},
                        vec_axis_type{0, 0}, layouts_type{1, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_1_102, ref_topologies_and_axes_1_1_102);

    auto ref_topologies_and_axes_1_1_120 =
        std::make_tuple(topologies_type{ref_topo1, ref_topo6, ref_topo1},
                        vec_axis_type{0, 0}, layouts_type{1, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_1_120, ref_topologies_and_axes_1_1_120);

    auto ref_topologies_and_axes_1_1_201 =
        std::make_tuple(topologies_type{ref_topo1, ref_topo6, ref_topo1},
                        vec_axis_type{0, 0}, layouts_type{1, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_1_201, ref_topologies_and_axes_1_1_201);

    auto ref_topologies_and_axes_1_1_210 =
        std::make_tuple(topologies_type{ref_topo1, ref_topo6, ref_topo1},
                        vec_axis_type{0, 0}, layouts_type{1, 1, 1});
    EXPECT_EQ(topologies_and_axes_1_1_210, ref_topologies_and_axes_1_1_210);

    auto ref_topologies_and_axes_1_1_123 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo9, ref_topo7, ref_topo9, ref_topo1},
        vec_axis_type{1, 0, 0, 1}, layouts_type{1, 0, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_1_1_123, ref_topologies_and_axes_1_1_123);

    auto ref_topologies_and_axes_1_1_132 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo9, ref_topo7, ref_topo9, ref_topo1},
        vec_axis_type{1, 0, 0, 1}, layouts_type{1, 0, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_1_1_132, ref_topologies_and_axes_1_1_132);

    // topology1 to topology2
    auto topologies_and_axes_1_2_012 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology1, topology2, axes012);
    auto topologies_and_axes_1_2_021 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology1, topology2, axes021);
    auto topologies_and_axes_1_2_102 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology1, topology2, axes102);
    auto topologies_and_axes_1_2_120 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology1, topology2, axes120);
    auto topologies_and_axes_1_2_201 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology1, topology2, axes201);
    auto topologies_and_axes_1_2_210 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology1, topology2, axes210);
    auto topologies_and_axes_1_2_123 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology1, topology2, axes123);
    auto topologies_and_axes_1_2_132 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology1, topology2, axes132);

    auto ref_topologies_and_axes_1_2_012 =
        std::make_tuple(topologies_type{ref_topo1, ref_topo0, ref_topo2},
                        vec_axis_type{0, 1}, layouts_type{1, 1, 0});
    EXPECT_EQ(topologies_and_axes_1_2_012, ref_topologies_and_axes_1_2_012);

    auto ref_topologies_and_axes_1_2_021 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{0, 0, 0, 1}, layouts_type{1, 1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_1_2_021, ref_topologies_and_axes_1_2_021);

    auto ref_topologies_and_axes_1_2_102 =
        std::make_tuple(topologies_type{ref_topo1, ref_topo0, ref_topo2},
                        vec_axis_type{0, 1}, layouts_type{1, 1, 0});
    EXPECT_EQ(topologies_and_axes_1_2_102, ref_topologies_and_axes_1_2_102);

    auto ref_topologies_and_axes_1_2_120 =
        std::make_tuple(topologies_type{ref_topo1, ref_topo0, ref_topo2},
                        vec_axis_type{0, 1}, layouts_type{1, 1, 0});

    EXPECT_EQ(topologies_and_axes_1_2_120, ref_topologies_and_axes_1_2_120);

    auto ref_topologies_and_axes_1_2_201 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{0, 0, 0, 1}, layouts_type{1, 1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_1_2_201, ref_topologies_and_axes_1_2_201);

    auto ref_topologies_and_axes_1_2_210 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo0, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{0, 0, 0, 1}, layouts_type{1, 1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_1_2_210, ref_topologies_and_axes_1_2_210);

    auto ref_topologies_and_axes_1_2_123 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo3, ref_topo1, ref_topo0, ref_topo2},
        vec_axis_type{1, 1, 0, 1}, layouts_type{1, 1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_1_2_123, ref_topologies_and_axes_1_2_123);

    auto ref_topologies_and_axes_1_2_132 = std::make_tuple(
        topologies_type{ref_topo1, ref_topo3, ref_topo5, ref_topo4, ref_topo2},
        vec_axis_type{1, 0, 1, 0}, layouts_type{1, 1, 0, 0, 0});
    EXPECT_EQ(topologies_and_axes_1_2_132, ref_topologies_and_axes_1_2_132);

    // topology2 to topology0
    auto topologies_and_axes_2_0_012 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology2, topology0, axes012);
    auto topologies_and_axes_2_0_021 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology2, topology0, axes021);
    auto topologies_and_axes_2_0_102 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology2, topology0, axes102);
    auto topologies_and_axes_2_0_120 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology2, topology0, axes120);
    auto topologies_and_axes_2_0_201 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology2, topology0, axes201);
    auto topologies_and_axes_2_0_210 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology2, topology0, axes210);
    auto topologies_and_axes_2_0_123 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology2, topology0, axes123);
    auto topologies_and_axes_2_0_132 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology2, topology0, axes132);

    auto ref_topologies_and_axes_2_0_012 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo5, ref_topo0},
        vec_axis_type{0, 1, 1}, layouts_type{0, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_2_0_012, ref_topologies_and_axes_2_0_012);

    auto ref_topologies_and_axes_2_0_021 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0, ref_topo1, ref_topo0},
        vec_axis_type{1, 0, 0}, layouts_type{0, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_0_021, ref_topologies_and_axes_2_0_021);

    auto ref_topologies_and_axes_2_0_102 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo2, ref_topo0},
        vec_axis_type{0, 0, 1}, layouts_type{0, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_2_0_102, ref_topologies_and_axes_2_0_102);

    auto ref_topologies_and_axes_2_0_120 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo2, ref_topo0},
        vec_axis_type{0, 0, 1}, layouts_type{0, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_2_0_120, ref_topologies_and_axes_2_0_120);

    auto ref_topologies_and_axes_2_0_201 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0, ref_topo1, ref_topo0},
        vec_axis_type{1, 0, 0}, layouts_type{0, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_0_201, ref_topologies_and_axes_2_0_201);

    auto ref_topologies_and_axes_2_0_210 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo0, ref_topo1, ref_topo0},
        vec_axis_type{1, 0, 0}, layouts_type{0, 1, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_0_210, ref_topologies_and_axes_2_0_210);

    auto ref_topologies_and_axes_2_0_123 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo2, ref_topo0},
        vec_axis_type{0, 0, 1}, layouts_type{0, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_2_0_123, ref_topologies_and_axes_2_0_123);

    auto ref_topologies_and_axes_2_0_132 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo2, ref_topo0},
        vec_axis_type{0, 0, 1}, layouts_type{0, 0, 0, 1});
    EXPECT_EQ(topologies_and_axes_2_0_132, ref_topologies_and_axes_2_0_132);

    // topology2 to topology1
    auto topologies_and_axes_2_1_012 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology2, topology1, axes012);
    auto topologies_and_axes_2_1_021 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology2, topology1, axes021);
    auto topologies_and_axes_2_1_102 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology2, topology1, axes102);
    auto topologies_and_axes_2_1_120 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology2, topology1, axes120);
    auto topologies_and_axes_2_1_201 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology2, topology1, axes201);
    auto topologies_and_axes_2_1_210 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology2, topology1, axes210);
    auto topologies_and_axes_2_1_123 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology2, topology1, axes123);
    auto topologies_and_axes_2_1_132 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology2, topology1, axes132);

    auto ref_topologies_and_axes_2_1_012 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo5, ref_topo3, ref_topo1},
        vec_axis_type{0, 1, 0, 1}, layouts_type{0, 0, 0, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_1_012, ref_topologies_and_axes_2_1_012);

    auto ref_topologies_and_axes_2_1_021 =
        std::make_tuple(topologies_type{ref_topo2, ref_topo0, ref_topo1},
                        vec_axis_type{1, 0}, layouts_type{0, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_1_021, ref_topologies_and_axes_2_1_021);

    auto ref_topologies_and_axes_2_1_102 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo5, ref_topo3, ref_topo1},
        vec_axis_type{0, 1, 0, 1}, layouts_type{0, 0, 0, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_1_102, ref_topologies_and_axes_2_1_102);

    auto ref_topologies_and_axes_2_1_120 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo5, ref_topo3, ref_topo1},
        vec_axis_type{0, 1, 0, 1}, layouts_type{0, 0, 0, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_1_120, ref_topologies_and_axes_2_1_120);

    auto ref_topologies_and_axes_2_1_201 =
        std::make_tuple(topologies_type{ref_topo2, ref_topo0, ref_topo1},
                        vec_axis_type{1, 0}, layouts_type{0, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_1_201, ref_topologies_and_axes_2_1_201);

    auto ref_topologies_and_axes_2_1_210 =
        std::make_tuple(topologies_type{ref_topo2, ref_topo0, ref_topo1},
                        vec_axis_type{1, 0}, layouts_type{0, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_1_210, ref_topologies_and_axes_2_1_210);

    auto ref_topologies_and_axes_2_1_123 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo5, ref_topo3, ref_topo1},
        vec_axis_type{0, 1, 0, 1}, layouts_type{0, 0, 0, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_1_123, ref_topologies_and_axes_2_1_123);

    auto ref_topologies_and_axes_2_1_132 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo4, ref_topo2, ref_topo0, ref_topo1},
        vec_axis_type{0, 0, 1, 0}, layouts_type{0, 0, 0, 1, 1});
    EXPECT_EQ(topologies_and_axes_2_1_132, ref_topologies_and_axes_2_1_132);

    // topology2 to topology2
    auto topologies_and_axes_2_2_012 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology2, topology2, axes012);
    auto topologies_and_axes_2_2_021 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology2, topology2, axes021);
    auto topologies_and_axes_2_2_102 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology2, topology2, axes102);
    auto topologies_and_axes_2_2_120 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology2, topology2, axes120);
    auto topologies_and_axes_2_2_201 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology2, topology2, axes201);
    auto topologies_and_axes_2_2_210 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology2, topology2, axes210);
    auto topologies_and_axes_2_2_123 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology2, topology2, axes123);
    auto topologies_and_axes_2_2_132 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology2, topology2, axes132);

    auto ref_topologies_and_axes_2_2_012 =
        std::make_tuple(topologies_type{ref_topo2, ref_topo10, ref_topo8,
                                        ref_topo10, ref_topo2},
                        vec_axis_type{0, 1, 1, 0}, layouts_type{0, 1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_2_2_012, ref_topologies_and_axes_2_2_012);

    auto ref_topologies_and_axes_2_2_021 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo7, ref_topo9, ref_topo7, ref_topo2},
        vec_axis_type{1, 0, 0, 1}, layouts_type{0, 0, 0, 0, 0});
    EXPECT_EQ(topologies_and_axes_2_2_021, ref_topologies_and_axes_2_2_021);

    auto ref_topologies_and_axes_2_2_102 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo10, ref_topo2, ref_topo7, ref_topo2},
        vec_axis_type{0, 0, 1, 1}, layouts_type{0, 1, 0, 0, 0});
    EXPECT_EQ(topologies_and_axes_2_2_102, ref_topologies_and_axes_2_2_102);

    auto ref_topologies_and_axes_2_2_120 =
        std::make_tuple(topologies_type{ref_topo2, ref_topo10, ref_topo8,
                                        ref_topo10, ref_topo2},
                        vec_axis_type{0, 1, 1, 0}, layouts_type{0, 1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_2_2_120, ref_topologies_and_axes_2_2_120);

    auto ref_topologies_and_axes_2_2_201 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo7, ref_topo2, ref_topo10, ref_topo2},
        vec_axis_type{1, 1, 0, 0}, layouts_type{0, 0, 0, 1, 0});
    EXPECT_EQ(topologies_and_axes_2_2_201, ref_topologies_and_axes_2_2_201);

    auto ref_topologies_and_axes_2_2_210 = std::make_tuple(
        topologies_type{ref_topo2, ref_topo7, ref_topo9, ref_topo7, ref_topo2},
        vec_axis_type{1, 0, 0, 1}, layouts_type{0, 0, 0, 0, 0});
    EXPECT_EQ(topologies_and_axes_2_2_210, ref_topologies_and_axes_2_2_210);

    auto ref_topologies_and_axes_2_2_123 =
        std::make_tuple(topologies_type{ref_topo2, ref_topo10, ref_topo8,
                                        ref_topo10, ref_topo2},
                        vec_axis_type{0, 1, 1, 0}, layouts_type{0, 1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_2_2_123, ref_topologies_and_axes_2_2_123);

    auto ref_topologies_and_axes_2_2_132 =
        std::make_tuple(topologies_type{ref_topo2, ref_topo10, ref_topo8,
                                        ref_topo10, ref_topo2},
                        vec_axis_type{0, 1, 1, 0}, layouts_type{0, 1, 1, 1, 0});
    EXPECT_EQ(topologies_and_axes_2_2_132, ref_topologies_and_axes_2_2_132);

    // topology3 to topology0
    auto topologies_and_axes_3_0_123 =
        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
            topology3, topology0, axes123);

    auto ref_topologies_and_axes_3_0_123 =
        std::make_tuple(topologies_type{ref_topo3, ref_topo1, ref_topo0},
                        vec_axis_type{1, 0}, layouts_type{1, 1, 1});
    EXPECT_EQ(topologies_and_axes_3_0_123, ref_topologies_and_axes_3_0_123);
  }
}

}  // namespace

TEST_P(TopologyParamTests, GetTopologyType_std_array) {
  int n0 = GetParam();
  test_to_topology_type<true>(n0);
}

TEST_P(TopologyParamTests, GetTopologyType_topology) {
  int n0 = GetParam();
  test_to_topology_type<false>(n0);
}

TEST_P(TopologyParamTests, GetCommonTopologyType_std_array) {
  int n0 = GetParam();
  test_get_common_topology_type<true>(n0);
}

TEST_P(TopologyParamTests, GetCommonTopologyType_topology) {
  int n0 = GetParam();
  test_get_common_topology_type<false>(n0);
}

TEST_P(TopologyParamTests, is_topology_std_array) {
  int n0 = GetParam();
  test_is_topology<true>(n0);
}

TEST_P(TopologyParamTests, is_topology_topology) {
  int n0 = GetParam();
  test_is_topology<false>(n0);
}

TEST_P(TopologyParamTests, are_topologies_std_array) {
  int n0 = GetParam();
  test_are_topologies<true>(n0);
}

TEST_P(TopologyParamTests, are_topologies_topology) {
  int n0 = GetParam();
  test_are_topologies<false>(n0);
}

TEST_P(TopologyParamTests, decompose_axes_slab) {
  int n0 = GetParam();
  test_decompose_axes_slab(n0);
}

TEST_P(TopologyParamTests, decompose_axes_pencil) {
  int n0 = GetParam();
  test_decompose_axes_pencil(n0);
}

TEST_P(TopologyParamTests, compute_trans_axis) {
  int n0 = GetParam();
  test_compute_trans_axis(n0);
}

TEST_P(TopologyParamTests, slab_in_out_axes_2D) {
  int n0 = GetParam();
  test_slab_in_out_axes_2D(n0);
}

TEST_P(TopologyParamTests, slab_in_out_axes_3D) {
  int n0 = GetParam();
  test_slab_in_out_axes_3D(n0);
}

TEST_P(TopologyParamTests, pencil_in_out_axes_3D) {
  int n0 = GetParam();
  test_pencil_in_out_axes_3D(n0);
}

TEST_P(TopologyParamTests, get_mid_array_3D) {
  int n0 = GetParam();
  test_get_mid_array_pencil_3D(n0);
}

TEST_P(TopologyParamTests, get_mid_array_4D) {
  int n0 = GetParam();
  test_get_mid_array_pencil_4D(n0);
}

INSTANTIATE_TEST_SUITE_P(TopologyTests, TopologyParamTests,
                         ::testing::Values(1, 2, 3, 4, 5, 6));

TEST_P(SlabParamTests, GetAllSlabTopologies1D_3DView) {
  int n0 = GetParam();
  test_get_all_slab_topologies1D_3DView(n0);
}

TEST_P(SlabParamTests, GetAllSlabTopologies2D_2DView) {
  int n0 = GetParam();
  test_get_all_slab_topologies2D_2DView(n0);
}

TEST_P(SlabParamTests, GetAllSlabTopologies2D_3DView) {
  int n0 = GetParam();
  test_get_all_slab_topologies2D_3DView(n0);
}

TEST_P(SlabParamTests, GetAllSlabTopologies3D_3DView) {
  int n0 = GetParam();
  test_get_all_slab_topologies3D_3DView(n0);
}

TEST_P(SlabParamTests, GetAllSlabTopologies3D_4DView) {
  int n0 = GetParam();
  test_get_all_slab_topologies3D_4DView(n0);
}

INSTANTIATE_TEST_SUITE_P(SlabTests, SlabParamTests,
                         ::testing::Values(1, 2, 3, 4, 5, 6));

TEST_P(PencilParamTests, GetAllPencilTopologies1D_3DView) {
  int n0 = GetParam();
  test_get_all_pencil_topologies1D_3DView(n0);
}

TEST_P(PencilParamTests, GetAllPencilTopologies2D_3DView) {
  int n0 = GetParam();
  test_get_all_pencil_topologies2D_3DView(n0);
}

TEST_P(PencilParamTests, GetAllPencilTopologies2D_4DView) {
  int n0 = GetParam();
  test_get_all_pencil_topologies2D_4DView(n0);
}

TEST_P(PencilParamTests, GetAllPencilTopologies3D_3DView) {
  int n0 = GetParam();
  test_get_all_pencil_topologies3D_3DView(n0);
}

TEST_P(PencilParamTests, GetAllPencilTopologies3D_4DView) {
  int n0 = GetParam();
  test_get_all_pencil_topologies3D_4DView(n0);
}

INSTANTIATE_TEST_SUITE_P(PencilTests, PencilParamTests,
                         ::testing::Values(1, 2, 3, 4, 5, 6));
