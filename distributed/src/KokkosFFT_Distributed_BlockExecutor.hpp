#ifndef KOKKOSFFT_DISTRIBUTED_BLOCK_EXECUTOR_HPP
#define KOKKOSFFT_DISTRIBUTED_BLOCK_EXECUTOR_HPP

#include <algorithm>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Asserts.hpp"
#include "KokkosFFT_Distributed_Types.hpp"
#include "KokkosFFT_Distributed_Extents.hpp"
#include "KokkosFFT_Distributed_Schedule.hpp"
#include "KokkosFFT_Distributed_TransBlock.hpp"
#include "KokkosFFT_Distributed_FFTStage.hpp"

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
  /// \param[in] in Input view of the plan
  /// \param[in] out Output view of the plan
  /// \param[in] block_infos The blocks of the plan
  /// \param[in] max_buffer_size The size of the send and receive buffers in
  /// the number of real values. The buffers are only allocated if the
  /// schedule uses them.
  explicit BlockExecutor(const ExecutionSpace& exec_space, const InViewType& in,
                         const OutViewType& out,
                         const std::vector<BlockInfoType>& block_infos,
                         std::size_t max_buffer_size)
      : m_exec_space(exec_space),
        m_schedule(build_schedule(block_infos, stage_user_views(in, out))) {
    // Allocate the buffers the schedule uses. The transposes use both of
    // them, as send and receive buffers, with the size given by the block
    // analyses. Without transposes (a single FFT block), each buffer only
    // holds the scratch views of the block, and is sized to them: a block
    // without local transpose and without staging uses neither buffer.
    const bool has_transpose = std::any_of(
        m_schedule.begin(), m_schedule.end(),
        [](const auto& step) { return step.m_type == BlockType::Transpose; });
    const std::size_t transpose_alloc_size = (max_buffer_size + 1) / 2;
    m_send_buffer_allocation               = AllocationViewType(
        "send_buffer_allocation", has_transpose
                                                    ? transpose_alloc_size
                                                    : required_buffer_size(BufferId::Send));
    m_recv_buffer_allocation = AllocationViewType(
        "recv_buffer_allocation", has_transpose
                                      ? transpose_alloc_size
                                      : required_buffer_size(BufferId::Recv));

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
        // The first FFT is planned on the views it runs on: the scratch views
        // of its local transposes or staging copy if any, else the views of
        // the block. They are out-of-place, except for an in-place C2C plan
        // made of a single block without local transposes, whose FFT then
        // runs in-place on the shared data of the input and output Views.
        const auto& in_spec  = step.m_src_T ? *step.m_src_T : step.m_src;
        const auto& out_spec = step.m_dst_T ? *step.m_dst_T : step.m_dst;
        auto plan_in         = get_view<InViewType>(in_spec, in, out);
        auto plan_out        = get_view<OutViewType>(out_spec, in, out);
        KOKKOSFFT_THROW_IF(
            KokkosFFT::Impl::extract_extents(plan_in) != in_spec.m_extents ||
                KokkosFFT::Impl::extract_extents(plan_out) !=
                    out_spec.m_extents,
            "BlockExecutor: the input and output Views must have the extents "
            "of the first FFT block");
        m_fft_stage0 = std::make_unique<FFTStageType0>(
            m_exec_space, plan_in, plan_out, block.m_fft_dim);
        if (step.m_src_T && block.m_in_map == identity_map()) {
          m_fft_stage0->set_in_staging();
        }
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

  // Not copyable, not movable: its FFT stages were planned on views of
  // its own buffers.
  BlockExecutor()                                = delete;
  BlockExecutor(const BlockExecutor&)            = delete;
  BlockExecutor& operator=(const BlockExecutor&) = delete;
  BlockExecutor(BlockExecutor&&)                 = delete;
  BlockExecutor& operator=(BlockExecutor&&)      = delete;
  ~BlockExecutor()                               = default;

  /// \brief Forward FFT
  /// \tparam CommsType Communicators of the transposes (SlabComms,
  /// PencilComms or SharedComms): a callable returning the communicator for a
  /// comm axis, with a static has_communication flag
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
  /// \tparam CommsType Communicators of the transposes (SlabComms,
  /// PencilComms or SharedComms): a callable returning the communicator for a
  /// comm axis, with a static has_communication flag
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

  /// \brief The size of the send buffer, in complex values. 0 if the
  /// schedule does not use it.
  std::size_t buffer_size() const { return m_send_buffer_allocation.size(); }

  /// \brief The size of the receive buffer, in complex values. 0 if the
  /// schedule does not use it.
  std::size_t recv_buffer_size() const {
    return m_recv_buffer_allocation.size();
  }

 private:
  static bool is_buffer(BufferId id) {
    return id == BufferId::Send || id == BufferId::Recv;
  }

  /// \brief Whether the schedule must stage the input and output Views:
  /// an in-place plan with a real input (build_schedule, rule 5)
  static bool stage_user_views(const InViewType& in, const OutViewType& out) {
    return KokkosFFT::Impl::is_real_v<in_value_type> && check_inplace(in, out);
  }

  static auto identity_map() {
    return KokkosFFT::Impl::index_sequence<std::size_t, DIM, 0>();
  }

  /// \brief The size of a view of the schedule, in complex values. Views of
  /// the input type may be real and take half the space.
  static std::size_t complex_size(const ViewSpecType& spec) {
    const bool is_real =
        spec.m_is_input_type && KokkosFFT::Impl::is_real_v<in_value_type>;
    std::size_t size = KokkosFFT::Impl::total_size(spec.m_extents);
    return is_real ? (size + 1) / 2 : size;
  }

  /// \brief The largest view of the schedule in a buffer, in complex values.
  /// 0 if no view lives in it.
  std::size_t required_buffer_size(BufferId id) const {
    std::size_t size = 0;
    auto update      = [&](const ViewSpecType& spec) {
      if (spec.m_id == id) size = std::max(size, complex_size(spec));
    };
    for (const auto& step : m_schedule) {
      update(step.m_src);
      update(step.m_dst);
      if (step.m_src_T) update(*step.m_src_T);
      if (step.m_dst_T) update(*step.m_dst_T);
    }
    return size;
  }

  /// \brief Check that the views of a step fit into their buffers
  void check_buffer_size(const BlockStepType& step) const {
    auto check = [&](const ViewSpecType& spec) {
      if (!is_buffer(spec.m_id)) return;
      const auto& allocation = spec.m_id == BufferId::Send
                                   ? m_send_buffer_allocation
                                   : m_recv_buffer_allocation;
      KOKKOSFFT_THROW_IF(complex_size(spec) > allocation.size(),
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
      // Communicators without communication (SharedComms) cannot run a
      // transpose, and the transpose path is not even compiled for them.
      if constexpr (CommsType::has_communication) {
        if (step.m_src.m_is_input_type) {
          transpose<InViewType>(step, comms, in, out, direction);
        } else {
          transpose<OutViewType>(step, comms, in, out, direction);
        }
      } else {
        KOKKOSFFT_THROW_IF(true,
                           "BlockExecutor: a transpose block cannot be "
                           "executed without communication");
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
