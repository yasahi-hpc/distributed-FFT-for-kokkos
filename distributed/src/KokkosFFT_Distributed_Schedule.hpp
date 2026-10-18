#ifndef KOKKOSFFT_DISTRIBUTED_SCHEDULE_HPP
#define KOKKOSFFT_DISTRIBUTED_SCHEDULE_HPP

#include <array>
#include <cstddef>
#include <optional>
#include <vector>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Asserts.hpp"
#include "KokkosFFT_Distributed_Types.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief Identifies the memory a view of a block lives in
enum class BufferId {
  //! The input View of the plan (forward input / backward output)
  UserIn,

  //! The output View of the plan (forward output / backward input)
  UserOut,

  //! The send buffer allocation
  Send,

  //! The receive buffer allocation
  Recv,
};

/// \brief Description of a view used by a block: the buffer it lives in, its
/// extents and its value type. The view itself is created from the buffer
/// allocation when the block is executed.
///
/// \tparam DIM The rank of the views
template <std::size_t DIM>
struct ViewSpec {
  using extents_type = std::array<std::size_t, DIM>;

  //! The buffer the view lives in
  BufferId m_id = BufferId::UserIn;

  //! The extents of the view
  extents_type m_extents = {};

  //! True before the first FFT, i.e. the view has the input value type of the
  //! plan (InViewType, may be real). False afterwards (OutViewType).
  bool m_is_input_type = true;

  bool operator==(const ViewSpec& other) const {
    return m_id == other.m_id && m_extents == other.m_extents &&
           m_is_input_type == other.m_is_input_type;
  }

  bool operator!=(const ViewSpec& other) const { return !(*this == other); }
};

/// \brief One step of the schedule of a distributed FFT plan. The views are
/// given for the forward direction. The backward direction executes the
/// steps in reverse order with m_src and m_dst swapped.
///
/// \tparam DIM The rank of the views
template <std::size_t DIM>
struct BlockStep {
  //! The type of the block (Transpose or FFT)
  BlockType m_type = BlockType::Transpose;

  //! The index of the transpose block or of the FFT stage
  std::size_t m_sub_idx = 0;

  //! The axis of the communicator used by a transpose (pencil only)
  std::size_t m_comm_axis = 0;

  //! The input and output views of the block (forward direction)
  ViewSpec<DIM> m_src, m_dst;

  //! FFT only: the scratch views for the local transposes of the input and
  //! the output. Only set if the FFT stage has an in-map or an out-map.
  std::optional<ViewSpec<DIM>> m_src_T, m_dst_T;

  bool operator==(const BlockStep& other) const {
    return m_type == other.m_type && m_sub_idx == other.m_sub_idx &&
           m_comm_axis == other.m_comm_axis && m_src == other.m_src &&
           m_dst == other.m_dst && m_src_T == other.m_src_T &&
           m_dst_T == other.m_dst_T;
  }

  bool operator!=(const BlockStep& other) const { return !(*this == other); }
};

/// \brief Build the schedule of a distributed FFT plan from its blocks. The
/// schedule decides, once, in which buffer every block reads and writes.
///
/// The data starts in UserIn and ends in UserOut. In between, it ping-pongs
/// between the send and receive buffers:
/// 1. A transpose reading UserIn writes into Send. A transpose reading
///    Send/Recv writes into the other buffer, since TransBlock forbids its
///    input to alias the send buffer and its output to alias the receive
///    buffer.
/// 2. The first FFT writes into the buffer it does not read from (Recv for
///    UserIn/Send). Its scratch views for the local transposes are Send (input
///    type) and that buffer.
/// 3. The following FFTs run in-place.
/// 4. The last block writes into UserOut. If the last block is an FFT without
///    out-map, the transpose just before it already writes into UserOut and
///    the FFT runs in-place on UserOut.
///
/// \tparam DIM The rank of the views
///
/// \param[in] block_infos The blocks of the plan
/// \return The schedule, one step per block
/// \throws std::runtime_error if a step would read and write the same buffer
/// where this is not allowed
template <std::size_t DIM>
std::vector<BlockStep<DIM>> build_schedule(
    const std::vector<BlockInfo<DIM>>& block_infos) {
  const auto src_map = KokkosFFT::Impl::index_sequence<std::size_t, DIM, 0>();
  auto other         = [](BufferId id) {
    return id == BufferId::Send ? BufferId::Recv : BufferId::Send;
  };
  auto is_buffer = [](BufferId id) {
    return id == BufferId::Send || id == BufferId::Recv;
  };

  std::vector<BlockStep<DIM>> steps;
  BufferId current     = BufferId::UserIn;  // where the data lives now
  bool fft_done        = false;
  std::size_t nb_trans = 0, nb_fft = 0;
  const std::size_t nb_blocks = block_infos.size();

  for (std::size_t i = 0; i < nb_blocks; ++i) {
    const auto& block  = block_infos.at(i);
    const bool is_last = i + 1 == nb_blocks;

    BlockStep<DIM> step;
    step.m_type      = block.m_block_type;
    step.m_comm_axis = block.m_comm_axis;

    if (block.m_block_type == BlockType::Transpose) {
      // The next block is the last FFT and runs in-place on UserOut
      bool is_followed_by_inplace_last_fft = false;
      if (i + 2 == nb_blocks) {
        const auto& next = block_infos.at(i + 1);
        is_followed_by_inplace_last_fft =
            next.m_block_type == BlockType::FFT && next.m_out_map == src_map;
      }

      BufferId dst = BufferId::Send;
      if (is_last || (fft_done && is_followed_by_inplace_last_fft)) {
        dst = BufferId::UserOut;
      } else if (is_buffer(current)) {
        dst = other(current);
      }

      KOKKOSFFT_THROW_IF(current == dst,
                         "build_schedule: transpose must not be in-place");

      step.m_sub_idx = nb_trans++;
      step.m_src     = {current, block.m_in_extents, !fft_done};
      step.m_dst     = {dst, block.m_out_extents, !fft_done};
      current        = dst;
    } else {
      const bool has_in_map  = i == 0 && block.m_in_map != src_map;
      const bool has_out_map = is_last && block.m_out_map != src_map;

      step.m_sub_idx = nb_fft;
      step.m_src     = {current, block.m_in_extents, !fft_done};

      if (nb_fft == 0) {
        // The first FFT is out-of-place (it may be real to complex)
        const BufferId out_buffer =
            current == BufferId::Recv ? BufferId::Send : BufferId::Recv;
        const BufferId dst = is_last ? BufferId::UserOut : out_buffer;
        KOKKOSFFT_THROW_IF(current == dst,
                           "build_schedule: first FFT must not be in-place");

        step.m_dst = {dst, block.m_out_extents, false};
        if (has_in_map) {
          step.m_src_T =
              ViewSpec<DIM>{BufferId::Send, block.m_in_extents, true};
        }
        if (has_out_map) {
          step.m_dst_T = ViewSpec<DIM>{out_buffer, block.m_out_extents, false};
        }
      } else {
        // The following FFTs are in-place
        KOKKOSFFT_THROW_IF(!is_buffer(current) && current != BufferId::UserOut,
                           "build_schedule: FFT must follow a transpose");
        KOKKOSFFT_THROW_IF(has_out_map && !is_buffer(current),
                           "build_schedule: FFT with out-map must not run on "
                           "the output View");

        step.m_dst = {is_last ? BufferId::UserOut : current,
                      block.m_out_extents, false};
        if (has_out_map) {
          step.m_dst_T = ViewSpec<DIM>{current, block.m_out_extents, false};
        }
      }

      fft_done = true;
      ++nb_fft;
      current = step.m_dst.m_id;
    }
    steps.push_back(step);
  }

  return steps;
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
