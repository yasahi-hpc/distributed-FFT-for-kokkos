#ifndef KOKKOSFFT_DISTRIBUTED_UTILS_HPP
#define KOKKOSFFT_DISTRIBUTED_UTILS_HPP

#include <algorithm>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_Types.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief Count the number of components that are not equal to one in a
/// container
/// \tparam ContainerType The type of the container (e.g., std::array,
/// std::vector)
/// \param[in] values The container of values
/// \return The number of components that are not equal to one
template <typename ContainerType>
auto count_non_ones(const ContainerType& values) {
  using value_type =
      std::remove_cv_t<std::remove_reference_t<decltype(*values.begin())>>;

  static_assert(
      std::is_integral_v<value_type>,
      "count_non_ones: Container value type must be an integral type");
  return std::count_if(values.cbegin(), values.cend(),
                       [](value_type val) { return val != 1; });
}

/// \brief Extract the different indices between two arrays
/// \tparam ContainerType The type of the container (e.g., std::array,
/// std::vector)
///
/// \param[in] a The first array
/// \param[in] b The second array
/// \return A vector of indices where the arrays differ
template <typename ContainerType>
auto extract_different_indices(const ContainerType& a, const ContainerType& b) {
  using value_type =
      std::remove_cv_t<std::remove_reference_t<decltype(*a.begin())>>;
  static_assert(std::is_integral_v<value_type>,
                "extract_different_indices: Container value type must be an "
                "integral type");
  KOKKOSFFT_THROW_IF(a.size() != b.size(),
                     "Containers must have the same size.");

  std::vector<std::size_t> diffs;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a.at(i) != b.at(i)) {
      diffs.push_back(i);
    }
  }
  return diffs;
}

/// \brief Extract the different values between two arrays
/// \tparam ContainerType The type of the container (e.g., std::array,
/// std::vector)
///
/// \param[in] a The first array
/// \param[in] b The second array
/// \return A set of values where the arrays differ
template <typename ContainerType>
auto extract_different_value_set(const ContainerType& a,
                                 const ContainerType& b) {
  using value_type =
      std::remove_cv_t<std::remove_reference_t<decltype(*a.begin())>>;
  static_assert(std::is_integral_v<value_type>,
                "extract_different_indices: Container value type must be an "
                "integral type");
  KOKKOSFFT_THROW_IF(a.size() != b.size(),
                     "Containers must have the same size.");

  std::vector<value_type> diffs;
  for (std::size_t i = 0; i < a.size(); ++i) {
    diffs.push_back(a.at(i));
    diffs.push_back(b.at(i));
  }
  return std::set<value_type>(diffs.begin(), diffs.end());
}

/// \brief Extract the indices where the values are not ones
/// \tparam ContainerType The type of the container (e.g., std::array,
/// std::vector)
///
/// \param[in] a The first array
/// \param[in] b The second array
/// \return A vector of indices where the one of the values are not ones
template <typename ContainerType>
auto extract_non_one_indices(const ContainerType& a, const ContainerType& b) {
  using value_type = std::remove_cv_t<std::remove_reference_t<decltype(a[0])>>;
  static_assert(
      std::is_integral_v<value_type>,
      "extract_non_one_indices: Container value type must be an integral type");
  KOKKOSFFT_THROW_IF(a.size() != b.size(),
                     "Containers must have the same size.");

  std::vector<std::size_t> non_one_indices;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] != 1 || b[i] != 1) {
      non_one_indices.push_back(i);
    }
  }
  return non_one_indices;
}

/// \brief Extract the non-one values
/// \tparam ContainerType The type of the container (e.g., std::array,
/// std::vector)
///
/// \param[in] a The first array
/// \return A vector of non-one values
template <typename ContainerType>
auto extract_non_one_values(const ContainerType& a) {
  using value_type = std::remove_cv_t<std::remove_reference_t<decltype(a[0])>>;
  static_assert(
      std::is_integral_v<value_type>,
      "extract_non_one_values: Container value type must be an integral type");
  std::vector<value_type> non_ones;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] != 1) {
      non_ones.push_back(a[i]);
    }
  }
  return non_ones;
}

/// \brief Check if the non-one elements are identical
/// \tparam ContainerType The type of the container (e.g., std::array,
/// std::vector)
/// \param[in] non_ones The vector of non-one elements
/// \return True if the non-one elements are identical, false otherwise
/// Note: This function assumes that the size of non_ones is 2
template <typename ContainerType>
bool has_identical_non_ones(const ContainerType& non_ones) {
  using value_type =
      std::remove_cv_t<std::remove_reference_t<decltype(*non_ones.begin())>>;
  static_assert(std::is_integral_v<value_type>,
                "has_identical_non_ones: Container value type must be an "
                "integral type");

  // If there are less than 2 non-one elements, return false
  if (count_non_ones(non_ones) < 2) return false;
  if (non_ones.size() == 2 &&
      std::set<value_type>(non_ones.begin(), non_ones.end()).size() == 1) {
    return true;
  }
  return false;
}

/// \brief Swap two elements in an array and return a new array
/// \tparam ContainerType The type of the container (e.g., std::array,
/// std::vector)
/// \tparam iType The type of the index in the array
///
/// \param[in] arr The array to be swapped
/// \param[in] i The index of the first element to be swapped
/// \param[in] j The index of the second element to be swapped
/// \return A new array with the elements swapped at indices i and j
template <typename ContainerType, typename iType>
ContainerType swap_elements(const ContainerType& arr, iType i, iType j) {
  static_assert(std::is_integral_v<iType>,
                "swap_elements: Index type must be an integral type");
  ContainerType result = arr;
  std::swap(result.at(i), result.at(j));
  return result;
}

/// \brief Merge two topologies into one
/// \tparam ContainerType The type of the container (e.g., std::array,
/// std::vector)
///
/// \param[in] in_topology The input topology
/// \param[in] out_topology The output topology
/// \return The merged topology
/// \throws std::runtime_error if the topologies do not have the same size or
/// are not convertible pencils
template <typename ContainerType>
auto merge_topology(const ContainerType& in_topology,
                    const ContainerType& out_topology) {
  auto in_size  = KokkosFFT::Impl::total_size(in_topology);
  auto out_size = KokkosFFT::Impl::total_size(out_topology);

  KOKKOSFFT_THROW_IF(in_size != out_size,
                     "Input and output topologies must have the same size.");

  if (in_size == 1) return in_topology;

  auto mismatched_extents = [](ContainerType in_topology,
                               ContainerType out_topology) -> std::string {
    std::string message;
    message += "in_topology (";
    message += std::to_string(in_topology.at(0));
    for (std::size_t r = 1; r < in_topology.size(); r++) {
      message += ",";
      message += std::to_string(in_topology.at(r));
    }
    message += "), ";
    message += "out_topology (";
    message += std::to_string(out_topology.at(0));
    for (std::size_t r = 1; r < out_topology.size(); r++) {
      message += ",";
      message += std::to_string(out_topology.at(r));
    }
    message += ")";
    return message;
  };

  // Check if two topologies are two convertible pencils
  auto diff_indices = extract_different_indices(in_topology, out_topology);
  KOKKOSFFT_THROW_IF(
      diff_indices.size() != 2,
      "Input and output topologies must differ exactly two positions: " +
          mismatched_extents(in_topology, out_topology));

  ContainerType topology = in_topology;
  for (std::size_t i = 0; i < in_topology.size(); i++) {
    topology.at(i) = std::max(in_topology.at(i), out_topology.at(i));
  }
  return topology;
}

/// \brief Get the non-one extent where the two topologies differ
/// In practice, compare the merged topology with one of the input
/// For example, if the two topologies are (1, p0, 1) and (p0, 1, 1),
/// the merged topology is (p0, p0, 1). The two topologies differ at the first
/// position, and the non-one extent is p0
///
/// \tparam ContainerType The type of the container (e.g., std::array,
/// std::vector)
/// \param[in] in_topology The input topology
/// \param[in] out_topology The output topology
/// \return The non-one extent where the two topologies differ. If both
/// topologies are ones, returns 1
/// \throws std::runtime_error if the topologies do not differ at exactly one
/// position
template <typename ContainerType>
auto diff_topology(const ContainerType& in_topology,
                   const ContainerType& out_topology) {
  using value_type =
      std::remove_cv_t<std::remove_reference_t<decltype(in_topology.at(0))>>;
  auto in_size  = KokkosFFT::Impl::total_size(in_topology);
  auto out_size = KokkosFFT::Impl::total_size(out_topology);

  if (in_size == 1 && out_size == 1) return value_type(1);

  auto diff_indices = extract_different_indices(in_topology, out_topology);
  KOKKOSFFT_THROW_IF(
      diff_indices.size() != 1,
      "Input and output topologies must differ exactly one positions.");
  auto diff_idx = diff_indices.at(0);

  // Returning the non-one extent
  return std::max(in_topology.at(diff_idx), out_topology.at(diff_idx));
}

/// \brief Get the axis to transpose to convert one topology to another
/// \tparam iType The index type
/// \tparam DIM The dimension
///
/// \param[in] in_topology The input topology
/// \param[in] out_topology The output topology
/// \return The axis to transpose
template <typename iType, std::size_t DIM = 1>
auto get_trans_axis(const std::array<iType, DIM>& in_topology,
                    const std::array<iType, DIM>& out_topology,
                    iType first_non_one) {
  auto in_non_ones  = extract_non_one_values(in_topology);
  auto out_non_ones = extract_non_one_values(out_topology);
  KOKKOSFFT_THROW_IF(
      in_non_ones.size() != 2 || out_non_ones.size() != 2,
      "Input and output topologies must have exactly two non-one "
      "elements.");
  KOKKOSFFT_THROW_IF(has_identical_non_ones(in_non_ones) ||
                         has_identical_non_ones(out_non_ones),
                     "Input and output topologies must not have identical "
                     "non-one elements.");

  std::vector<iType> diff_indices =
      extract_different_indices(in_topology, out_topology);
  KOKKOSFFT_THROW_IF(
      diff_indices.size() != 2,
      "Input and output topologies must differ exactly two positions");

  iType exchange_non_one = 0;
  for (auto diff_idx : diff_indices) {
    if (in_topology.at(diff_idx) > 1) {
      exchange_non_one = in_topology.at(diff_idx);
      break;
    }
  }
  iType trans_axis = exchange_non_one == first_non_one ? 0 : 1;
  return trans_axis;
}

/// \brief Convert a std::vector to std::array (rvalue version)
/// \tparam T The type of the elements in the std::vector
/// \tparam N The size of the std::array
/// \param[in, out] vec The input std::vector
/// \return A std::array containing the elements of the input vector
template <typename T, std::size_t N>
auto to_array(std::vector<T>&& vec) {
  KOKKOSFFT_THROW_IF(
      vec.size() != N,
      "to_array: Vector size must match the specified dimension.");
  std::array<T, N> arr;
  std::move(vec.begin(), vec.end(), arr.begin());
  return arr;
}

/// \brief Convert a std::vector to std::array (lvalue version)
/// \tparam T The type of the elements in the std::vector
/// \tparam N The size of the std::array
/// \param[in] vec The input std::vector
/// \return A std::array containing the elements of the input vector
template <std::size_t N, typename T>
auto to_array(const std::vector<T>& vec) {
  KOKKOSFFT_THROW_IF(
      vec.size() != N,
      "to_array: Vector size must match the specified dimension.");
  std::array<T, N> arr;
  std::copy(vec.begin(), vec.end(), arr.begin());
  return arr;
}

/// \brief Calculate the permuted extents based on the map
///
/// Example
/// View extents: (n0, n1, n2, n3)
/// map: (0, 2, 3, 1)
/// Next extents: (n0, n2, n3, n1)
///
/// \tparam ContainerType The container type
/// \tparam iType The integer type used for extents
/// \tparam DIM The number of dimensions of the extents.
///
/// \param[in] extents Extents of the View.
/// \param[in] map A map representing how the data is permuted
/// \return A extents of the permuted view
/// \throws std::runtime_error if the size of map is not equal to DIM
template <typename ContainerType, typename iType, std::size_t DIM>
auto get_mapped_extents(const std::array<iType, DIM>& extents,
                        const ContainerType& map) {
  using value_type =
      std::remove_cv_t<std::remove_reference_t<decltype(map.at(0))>>;
  static_assert(std::is_integral_v<value_type>,
                "get_mapped_extents: Map container value type must be an "
                "integral type");
  KOKKOSFFT_THROW_IF(
      map.size() != DIM,
      "get_mapped_extents: extents size must be equal to map size.");
  std::array<iType, DIM> mapped_extents;
  std::transform(
      map.begin(), map.end(), mapped_extents.begin(),
      [&](std::size_t mapped_idx) { return extents.at(mapped_idx); });

  return mapped_extents;
}

/// \brief Compute the larger extents. Larger one corresponds to
/// the extents to FFT library. This is a helper for vendor library
/// which supports 2D or 3D non-batched FFTs.
///
/// Example
/// in extents: (8, 7, 8)
/// out extents: (8, 7, 5)
///
/// \tparam iType The integer type used for extents
/// \tparam DIM The number of dimensions of the extents.
/// \tparam FFT_DIM The number of dimensions of the FFT.
///
/// \param[in] in_extents Extents of the global input View.
/// \param[in] out_extents Extents of the global output View.
/// \return A extents of the permuted view
template <typename iType, std::size_t DIM, std::size_t FFT_DIM>
auto compute_fft_extents(const std::array<iType, DIM>& in_extents,
                         const std::array<iType, DIM>& out_extents,
                         const std::array<iType, FFT_DIM>& axes) {
  static_assert(std::is_integral_v<iType>,
                "compute_fft_extents: iType must be an integral type");
  static_assert(
      FFT_DIM >= 1 && FFT_DIM <= KokkosFFT::MAX_FFT_DIM,
      "compute_fft_extents: the Rank of FFT axes must be between 1 and 3");
  static_assert(
      DIM >= FFT_DIM,
      "compute_fft_extents: View rank must be larger than or equal to "
      "the Rank of FFT axes");

  std::array<iType, FFT_DIM> fft_extents;
  std::transform(axes.begin(), axes.end(), fft_extents.begin(),
                 [&](iType axis) {
                   return std::max(in_extents.at(axis), out_extents.at(axis));
                 });

  return fft_extents;
}

/// \brief Calculate the axes to have contiguous axes
/// based on the layout
///
/// Example
/// Axes: (1, 3, 2)
/// LayoutLeft -> (2, 1, 0)
/// LayoutRight -> (-3, -2, -1)
///
/// \tparam Layout The layout type
/// \tparam ContainerType The container type
///
/// \param[in] axes Axes of the transform
/// \param[in] rank The rank of the View
/// \return A vector of contiguous axes used for FFT
/// \throws std::runtime_error if the size of axes is greater than rank
template <typename Layout, typename ContainerType>
auto get_contiguous_axes(const ContainerType& axes,
                         [[maybe_unused]] std::size_t rank) {
  using value_type =
      std::remove_cv_t<std::remove_reference_t<decltype(axes.at(0))>>;
  std::vector<value_type> contiguous_axes(axes.size());

  if (std::is_same_v<Layout, Kokkos::LayoutLeft>) {
    // Construct the contiguous data and then reverse it
    // Reverse the axes to have the inner most axes first
    // (0, 1, 2) -> (2, 1, 0)
    std::iota(contiguous_axes.begin(), contiguous_axes.end(), 0);
    std::reverse(contiguous_axes.begin(), contiguous_axes.end());
  } else {
    for (std::size_t i = 0; i < axes.size(); ++i) {
      int negative_axis = -int(axes.size()) + int(i);
      contiguous_axes.at(i) =
          KokkosFFT::Impl::convert_negative_axis(negative_axis, rank);
    }
  }
  return contiguous_axes;
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
