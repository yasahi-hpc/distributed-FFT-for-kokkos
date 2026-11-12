#ifndef KOKKOSFFT_BENCHMARK_CASES_HPP
#define KOKKOSFFT_BENCHMARK_CASES_HPP

#include <array>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include <KokkosFFT.hpp>

namespace KokkosFFT {
namespace Distributed {
namespace Benchmark {

/// \brief Which plan a benchmark constructs
enum class PlanKind {
  //! The native plan of the topologies (SharedPlan, SlabPlan, PencilPlan),
  //! constructed directly
  native,
  //! The vendor plan (TplPlan: cuFFTMp, rocFFT-MPI), skipped where
  //! is_tpl_available rejects the case
  tpl,
  //! What Plan picks (plan_factory): tpl where available, native otherwise
  factory
};

/// \brief How the data is distributed before and after the transform
enum class Decomposition {
  //! One process, no distribution (the reference of the parallel efficiency)
  local,
  //! Slabs: the first axis of the FFT distributed in, the second out
  slab,
  //! Pencils on a px x py grid
  pencil,
  //! The same slab in and out
  slab_same,
  //! The same pencil in and out
  pencil_same
};

/// \brief What one timed iteration executes
enum class Direction {
  //! A forward transform (out-of-place only: in-place, it would transform
  //! its own output again)
  forward,
  //! A backward transform (out-of-place C2C only: a C2R may overwrite its
  //! input)
  backward,
  //! A forward and a backward transform (any placement; the data stays
  //! bounded with the backward normalization)
  roundtrip
};

inline std::string to_string(PlanKind kind) {
  switch (kind) {
    case PlanKind::native: return "native";
    case PlanKind::tpl: return "tpl";
    default: return "factory";
  }
}

inline std::string to_string(Decomposition d) {
  switch (d) {
    case Decomposition::local: return "local";
    case Decomposition::slab: return "slab";
    case Decomposition::pencil: return "pencil";
    case Decomposition::slab_same: return "slab_same";
    default: return "pencil_same";
  }
}

inline std::string to_string(Direction d) {
  switch (d) {
    case Direction::forward: return "forward";
    case Direction::backward: return "backward";
    default: return "roundtrip";
  }
}

/// \brief One case of the 3D FFT benchmarks (the value type and the View
/// layout are template parameters of the benchmark)
struct FFTCase {
  //! Global extents of the input
  std::array<std::size_t, 3> m_extents = {};
  Decomposition m_decomposition        = Decomposition::slab;
  //! FFT axes: {0, 1, 2} (natural) or a permutation
  KokkosFFT::axis_type<3> m_axes = {0, 1, 2};
  bool m_is_inplace              = false;
  Direction m_direction          = Direction::forward;
  PlanKind m_plan_kind           = PlanKind::native;
  //! Process grid: px x py (py = 1 for slabs)
  std::size_t m_px = 1, m_py = 1;
};

/// \brief Whether the axes are {0, 1, 2}
inline bool is_natural(const KokkosFFT::axis_type<3>& axes) {
  return axes[0] == 0 && axes[1] == 1 && axes[2] == 2;
}

/// \brief The most square grid px x py = nprocs with px >= py > 1, or
/// {nprocs, 1} if there is none (nprocs prime)
inline std::array<std::size_t, 2> pencil_grid(std::size_t nprocs) {
  std::size_t py =
      static_cast<std::size_t>(std::sqrt(static_cast<double>(nprocs)));
  while (py > 1 && nprocs % py != 0) --py;
  return {nprocs / py, py};
}

/// \brief Input and output topologies of a case. With the axes {a0, a1, a2}
/// (a2 transformed first, a0 last), the input keeps a2 local and the output
/// keeps a0 local:
/// - slab / pencil: in (a0: px, a1: py, a2: 1), out (a0: 1, a1: px, a2: py);
///   for natural axes and py = 1, {px, 1, 1} to {1, px, 1};
/// - slab_same / pencil_same: the input topology in and out;
/// - local: all ones.
inline std::array<std::array<std::size_t, 3>, 2> case_topologies(
    const FFTCase& c) {
  std::array<std::size_t, 3> in{1, 1, 1}, out{1, 1, 1};
  if (c.m_decomposition == Decomposition::local) return {in, out};
  const auto a0 = static_cast<std::size_t>(c.m_axes[0]);
  const auto a1 = static_cast<std::size_t>(c.m_axes[1]);
  const auto a2 = static_cast<std::size_t>(c.m_axes[2]);
  in[a0]        = c.m_px;
  in[a1]        = c.m_py;
  in[a2]        = 1;
  out[a0]       = 1;
  out[a1]       = c.m_px;
  out[a2]       = c.m_py;
  if (c.m_decomposition == Decomposition::slab_same ||
      c.m_decomposition == Decomposition::pencil_same) {
    out = in;
  }
  return {in, out};
}

/// \brief Model of the all-to-alls of one transform: the size of the group
/// of each one. The minimal schedule: a slab needs one transpose over all
/// the processes, a pencil two (over py, then px processes), the same
/// distribution in and out goes there and back. The plans' actual schedules
/// are not exposed; this is an estimate.
inline std::vector<std::size_t> model_transposes(const FFTCase& c) {
  switch (c.m_decomposition) {
    case Decomposition::local: return {};
    case Decomposition::slab: return {c.m_px};
    case Decomposition::pencil: return {c.m_py, c.m_px};
    case Decomposition::slab_same: return {c.m_px, c.m_px};
    default: return {c.m_py, c.m_px, c.m_px, c.m_py};
  }
}

/// \brief Name of a benchmark case, e.g.
/// fft3d/c2c/double/right/pencil/natural/out-of-place/forward/native/N:256
inline std::string case_name(const std::string& type, const std::string& real,
                             const std::string& layout, const FFTCase& c) {
  std::ostringstream oss;
  oss << "fft3d/" << type << "/" << real << "/" << layout << "/"
      << to_string(c.m_decomposition) << "/"
      << (is_natural(c.m_axes) ? "natural" : "permuted") << "/"
      << (c.m_is_inplace ? "in-place" : "out-of-place") << "/"
      << to_string(c.m_direction) << "/" << to_string(c.m_plan_kind);
  if (c.m_extents[0] == c.m_extents[1] && c.m_extents[1] == c.m_extents[2]) {
    oss << "/N:" << c.m_extents[0];
  } else {
    oss << "/N:" << c.m_extents[0] << "x" << c.m_extents[1] << "x"
        << c.m_extents[2];
  }
  return oss.str();
}

/// \brief The sizes of the FFT benchmarks: KOKKOSFFT_BENCHMARK_SIZES, a comma
/// separated list of N (cubes N^3) or NxMxL, default "256,512,1024"
inline std::vector<std::array<std::size_t, 3>> benchmark_sizes() {
  const char* env = std::getenv("KOKKOSFFT_BENCHMARK_SIZES");
  std::string list =
      (env != nullptr && *env != '\0') ? std::string(env) : "256,512,1024";
  std::vector<std::array<std::size_t, 3>> sizes;
  std::stringstream ss(list);
  for (std::string item; std::getline(ss, item, ',');) {
    if (item.empty()) continue;
    std::array<std::size_t, 3> e{};
    std::stringstream is(item);
    std::string part;
    std::size_t i = 0;
    while (std::getline(is, part, 'x') && i < 3) {
      e[i++] = static_cast<std::size_t>(std::stoul(part));
    }
    if (i == 1) e[1] = e[2] = e[0];
    if (i == 1 || i == 3) sizes.push_back(e);
  }
  return sizes;
}

}  // namespace Benchmark
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
