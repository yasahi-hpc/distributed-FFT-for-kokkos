#ifndef KOKKOSFFT_DISTRIBUTED_BLOCK_EXECUTOR_HPP
#define KOKKOSFFT_DISTRIBUTED_BLOCK_EXECUTOR_HPP

#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Asserts.hpp"
#include "KokkosFFT_Distributed_Types.hpp"
#include "KokkosFFT_Distributed_Schedule.hpp"
#include "KokkosFFT_Distributed_TransBlock.hpp"
#include "KokkosFFT_Distributed_InternalPlan.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief Create an unmanaged view on the send or receive buffer
/// \tparam ViewType The type of the view to create
/// \tparam AllocationViewType The type of the buffer allocations
/// \tparam DIM The rank of the view
///
/// \param[in] spec The description of the view (must be Send or Recv)
/// \param[in] send The send buffer allocation
/// \param[in] recv The receive buffer allocation
/// \return The view on the buffer
template <typename ViewType, typename AllocationViewType, std::size_t DIM>
ViewType make_view(const ViewSpec<DIM>& spec, const AllocationViewType& send,
                   const AllocationViewType& recv) {
  using value_type = typename ViewType::non_const_value_type;
  using LayoutType = typename ViewType::array_layout;
  KOKKOSFFT_THROW_IF(
      spec.m_id != BufferId::Send && spec.m_id != BufferId::Recv,
      "make_view: only views on the send or receive buffer can be created");
  auto* ptr = spec.m_id == BufferId::Send ? send.data() : recv.data();
  return ViewType(reinterpret_cast<value_type*>(ptr),
                  KokkosFFT::Impl::create_layout<LayoutType>(spec.m_extents));
}

/// \brief Executes the blocks of a distributed FFT plan following its
/// schedule. It owns the send/receive buffers, the FFT workspace, the
/// transpose blocks and the FFT stages. The views on the buffers are created
/// from the schedule each time a block is executed.
///
/// \tparam ExecutionSpace Kokkos execution space
/// \tparam InViewType Input Kokkos view type of the plan
/// \tparam OutViewType Output Kokkos view type of the plan
template <typename ExecutionSpace, typename InViewType, typename OutViewType>
class BlockExecutor {
  static constexpr std::size_t DIM = InViewType::rank();

  using in_value_type = typename InViewType::non_const_value_type;
  using float_type   = KokkosFFT::Impl::base_floating_point_type<in_value_type>;
  using complex_type = Kokkos::complex<float_type>;
  using LayoutType   = typename InViewType::array_layout;

  using AllocationViewType =
      Kokkos::View<complex_type*, LayoutType, ExecutionSpace>;
  using BlockInfoType  = BlockInfo<DIM>;
  using BlockStepType  = BlockStep<DIM>;
  using ViewSpecType   = ViewSpec<DIM>;
  using TransBlockType = TransBlock<ExecutionSpace, DIM>;

  // First FFT to perform can be Real to Complex
  using FFTStageType0 = FFTStage<ExecutionSpace, InViewType, OutViewType>;

  // Following FFTs to perform must be Complex to Complex
  using FFTStageType1 = FFTStage<ExecutionSpace, OutViewType, OutViewType>;

  ExecutionSpace m_exec_space;

  // Which buffer every block reads and writes
  std::vector<BlockStepType> m_schedule;

  // Buffer Allocations
  AllocationViewType m_send_buffer_allocation, m_recv_buffer_allocation;
  std::vector<AllocationViewType> m_fft_buffer_allocations;

  // Internal transpose blocks
  std::vector<std::unique_ptr<TransBlockType>> m_trans_blocks;

  // Internal FFT stages
  std::unique_ptr<FFTStageType0> m_fft_stage0;
  std::vector<std::unique_ptr<FFTStageType1>> m_fft_stages;

 public:
  /// \brief BlockExecutor constructor
  /// \param[in] exec_space Kokkos execution space
  /// \param[in] block_infos The blocks of the plan
  /// \param[in] max_buffer_size The size of the send and receive buffers in
  /// the number of real values
  explicit BlockExecutor(const ExecutionSpace& exec_space,
                         const std::vector<BlockInfoType>& block_infos,
                         std::size_t max_buffer_size)
      : m_exec_space(exec_space), m_schedule(build_schedule(block_infos)) {
    // Allocate buffer views
    std::size_t complex_alloc_size = (max_buffer_size + 1) / 2;
    m_send_buffer_allocation =
        AllocationViewType("send_buffer_allocation", complex_alloc_size);
    m_recv_buffer_allocation =
        AllocationViewType("recv_buffer_allocation", complex_alloc_size);

    const std::size_t last_block_idx = block_infos.size() - 1;
    for (std::size_t block_idx = 0; block_idx < block_infos.size();
         ++block_idx) {
      const auto& block = block_infos.at(block_idx);
      const auto& step  = m_schedule.at(block_idx);
      check_buffer_size(step);

      if (step.m_type == BlockType::Transpose) {
        m_trans_blocks.push_back(std::make_unique<TransBlockType>(
            m_exec_space, block.m_buffer_extents, block.m_in_map,
            block.m_in_axis, block.m_out_map, block.m_out_axis));
        continue;
      }

      // Local transposes are needed only for the first and the last blocks
      auto set_maps = [&](auto& stage) {
        if (block_idx == 0) stage.set_in_map(block.m_in_map);
        if (block_idx == last_block_idx) stage.set_out_map(block.m_out_map);
      };

      if (step.m_sub_idx == 0) {
        // The first FFT is out-of-place
        auto in_T = buffer_view<InViewType>(
            ViewSpecType{BufferId::Send, block.m_in_extents, true});
        auto out_T = buffer_view<OutViewType>(
            ViewSpecType{BufferId::Recv, block.m_out_extents, false});
        m_fft_stage0 = std::make_unique<FFTStageType0>(m_exec_space, in_T,
                                                       out_T, block.m_fft_dim);
        set_maps(*m_fft_stage0);
      } else {
        // The following FFTs are in-place. If the FFT runs on the output
        // View, the plan is made on the send buffer instead.
        const BufferId id =
            is_buffer(step.m_src.m_id) ? step.m_src.m_id : BufferId::Send;
        auto view = buffer_view<OutViewType>(
            ViewSpecType{id, block.m_out_extents, false});
        m_fft_stages.push_back(std::make_unique<FFTStageType1>(
            m_exec_space, view, view, block.m_fft_dim));
        set_maps(*m_fft_stages.back());
      }
    }

    // Using aligned data
    m_fft_buffer_allocations =
        allocate_fft_work_areas<complex_type, AllocationViewType>(*m_fft_stage0,
                                                                  m_fft_stages);
  }

  /// \brief Forward FFT
  /// \tparam CommsType Callable returning the communicator for a comm axis
  /// \param[in] comms The communicators of the transposes
  /// \param[in] in Input view
  /// \param[out] out Output view
  template <typename CommsType>
  void forward(const CommsType& comms, const InViewType& in,
               const OutViewType& out) const {
    for (const auto& step : m_schedule) {
      execute(step, comms, in, out, KokkosFFT::Direction::forward);
    }
  }

  /// \brief Backward FFT
  /// \tparam CommsType Callable returning the communicator for a comm axis
  /// \param[in] comms The communicators of the transposes
  /// \param[in] out Input view of the backward FFT
  /// \param[out] in Output view of the backward FFT
  template <typename CommsType>
  void backward(const CommsType& comms, const OutViewType& out,
                const InViewType& in) const {
    for (auto it = m_schedule.rbegin(); it != m_schedule.rend(); ++it) {
      execute(*it, comms, in, out, KokkosFFT::Direction::backward);
    }
  }

 private:
  static bool is_buffer(BufferId id) {
    return id == BufferId::Send || id == BufferId::Recv;
  }

  /// \brief Check that the views of a step fit into the buffers
  void check_buffer_size(const BlockStepType& step) const {
    auto check = [&](const ViewSpecType& spec) {
      if (!is_buffer(spec.m_id)) return;
      // Views of the input type may be real and take half the space
      const bool is_real =
          spec.m_is_input_type && KokkosFFT::Impl::is_real_v<in_value_type>;
      std::size_t size = KokkosFFT::Impl::total_size(spec.m_extents);
      if (is_real) size = (size + 1) / 2;
      KOKKOSFFT_THROW_IF(size > m_send_buffer_allocation.size(),
                         "BlockExecutor: a view does not fit into the buffer");
    };
    check(step.m_src);
    check(step.m_dst);
    if (step.m_src_T) check(*step.m_src_T);
    if (step.m_dst_T) check(*step.m_dst_T);
  }

  template <typename ViewType>
  ViewType buffer_view(const ViewSpecType& spec) const {
    return make_view<ViewType>(spec, m_send_buffer_allocation,
                               m_recv_buffer_allocation);
  }

  /// \brief The view described by spec: the input/output View of the plan or
  /// a view on a buffer
  template <typename ViewType>
  ViewType get_view(const ViewSpecType& spec, const InViewType& in,
                    const OutViewType& out) const {
    if (spec.m_id == BufferId::UserIn) {
      if constexpr (std::is_same_v<ViewType, InViewType>) {
        return in;
      } else {
        KOKKOSFFT_THROW_IF(true, "BlockExecutor: input View has wrong type");
      }
    } else if (spec.m_id == BufferId::UserOut) {
      if constexpr (std::is_same_v<ViewType, OutViewType>) {
        return out;
      } else {
        KOKKOSFFT_THROW_IF(true, "BlockExecutor: output View has wrong type");
      }
    }
    return buffer_view<ViewType>(spec);
  }

  /// \brief The scratch view for a local transpose, or an empty view if the
  /// stage has no local transpose
  template <typename ViewType>
  ViewType get_scratch(const std::optional<ViewSpecType>& spec) const {
    return spec ? buffer_view<ViewType>(*spec) : ViewType();
  }

  /// \brief Execute a transpose block. The send buffer must not alias the
  /// input and the receive buffer must not alias the output (TransBlock).
  template <typename ViewType, typename CommsType>
  void transpose(const BlockStepType& step, const CommsType& comms,
                 const InViewType& in, const OutViewType& out,
                 KokkosFFT::Direction direction) const {
    const bool is_forward = direction == KokkosFFT::Direction::forward;
    const auto& src_spec  = is_forward ? step.m_src : step.m_dst;
    const auto& dst_spec  = is_forward ? step.m_dst : step.m_src;
    auto src              = get_view<ViewType>(src_spec, in, out);
    auto dst              = get_view<ViewType>(dst_spec, in, out);

    const bool use_recv_as_send =
        src_spec.m_id == BufferId::Send || dst_spec.m_id == BufferId::Recv;
    const auto& send =
        use_recv_as_send ? m_recv_buffer_allocation : m_send_buffer_allocation;
    const auto& recv =
        use_recv_as_send ? m_send_buffer_allocation : m_recv_buffer_allocation;

    (*m_trans_blocks.at(step.m_sub_idx))(comms(step.m_comm_axis), src, dst,
                                         send, recv, direction);
  }

  template <typename CommsType>
  void execute(const BlockStepType& step, const CommsType& comms,
               const InViewType& in, const OutViewType& out,
               KokkosFFT::Direction direction) const {
    const bool is_forward = direction == KokkosFFT::Direction::forward;
    if (step.m_type == BlockType::Transpose) {
      if (step.m_src.m_is_input_type) {
        transpose<InViewType>(step, comms, in, out, direction);
      } else {
        transpose<OutViewType>(step, comms, in, out, direction);
      }
    } else if (step.m_sub_idx == 0) {
      auto src   = get_view<InViewType>(step.m_src, in, out);
      auto dst   = get_view<OutViewType>(step.m_dst, in, out);
      auto src_T = get_scratch<InViewType>(step.m_src_T);
      auto dst_T = get_scratch<OutViewType>(step.m_dst_T);
      if (is_forward) {
        m_fft_stage0->forward(src, dst, src_T, dst_T);
      } else {
        m_fft_stage0->backward(dst, src, src_T, dst_T);
      }
    } else {
      auto src          = get_view<OutViewType>(step.m_src, in, out);
      auto dst          = get_view<OutViewType>(step.m_dst, in, out);
      auto dst_T        = get_scratch<OutViewType>(step.m_dst_T);
      const auto& stage = *m_fft_stages.at(step.m_sub_idx - 1);
      if (is_forward) {
        stage.forward(src, dst, dst_T, dst_T);
      } else {
        stage.backward(dst, src, dst_T, dst_T);
      }
    }
  }
};

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
