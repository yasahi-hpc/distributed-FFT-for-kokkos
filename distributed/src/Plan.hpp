#ifndef PLAN_HPP
#define PLAN_HPP

#include <vector>
#include <memory>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include "Block.hpp"
#include "Mapping.hpp"
#include "MPI_Helper.hpp"
#include "Extents.hpp"

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1>
class Plan {
  using execSpace     = ExecutionSpace;
  using in_value_type = typename InViewType::non_const_value_type;
  using float_type   = KokkosFFT::Impl::base_floating_point_type<in_value_type>;
  using complex_type = Kokkos::complex<float_type>;
  using fft_data_type =
      KokkosFFT::Impl::add_pointer_n_t<complex_type, InViewType::rank()>;
  using buffer_data_type =
      KokkosFFT::Impl::add_pointer_n_t<complex_type, InViewType::rank() + 1>;

  using axes_type           = KokkosFFT::axis_type<DIM>;
  using map_type            = std::array<std::size_t, InViewType::rank()>;
  using topology_type       = std::array<std::size_t, InViewType::rank()>;
  using extents_type        = std::array<std::size_t, InViewType::rank()>;
  using buffer_extents_type = std::array<std::size_t, InViewType::rank() + 1>;

  using pencil_axes_type    = std::tuple<std::size_t, std::size_t>;
  using paired_map_type     = std::tuple<map_type, map_type>;
  using paired_extents_type = std::tuple<extents_type, extents_type>;
  using paired_data_type    = std::tuple<complex_type*, complex_type*>;
  using LayoutType          = typename InViewType::array_layout;

  using FFTViewType =
      Kokkos::View<fft_data_type, typename InViewType::array_layout,
                   typename InViewType::execution_space>;
  using BufferViewType =
      Kokkos::View<buffer_data_type, typename InViewType::array_layout,
                   typename InViewType::execution_space>;

  using FFTForwardBlockType =
      FFTForwardBlock<ExecutionSpace, FFTViewType, FFTViewType, FFTViewType,
                      BufferViewType>;
  using FFTBackwardBlockType =
      FFTBackwardBlock<ExecutionSpace, FFTViewType, FFTViewType, FFTViewType,
                       BufferViewType>;

  using FFTForwardPlanType =
      KokkosFFT::Plan<ExecutionSpace, InViewType, FFTViewType, 1>;
  using FFTBackwardPlanType =
      KokkosFFT::Plan<ExecutionSpace, FFTViewType, InViewType, 1>;

  using AllocationViewType =
      Kokkos::View<complex_type*, LayoutType, ExecutionSpace>;

  ExecutionSpace m_exec_space;
  InViewType m_in;
  FFTViewType m_fft;
  OutViewType m_out;
  AllocationViewType m_send_buffer_allocation;
  AllocationViewType m_recv_buffer_allocation;
  AllocationViewType m_pencil_allocation0;
  AllocationViewType m_pencil_allocation1;

  axes_type m_axes;
  topology_type m_in_topology;
  topology_type m_out_topology;
  MPI_Comm m_comm;

  std::vector<std::unique_ptr<FFTForwardBlockType>> m_forward_blocks;
  std::vector<std::unique_ptr<FFTBackwardBlockType>> m_backward_blocks;
  std::unique_ptr<FFTForwardPlanType> m_forward_plan;
  std::unique_ptr<FFTBackwardPlanType> m_backward_plan;

 public:
  explicit Plan(const ExecutionSpace& exec_space, const InViewType& in,
                const OutViewType& out, const axes_type& axes,
                const topology_type& in_topology,
                const topology_type& out_topology, const MPI_Comm& comm)
      : m_exec_space(exec_space),
        m_in(in),
        m_out(out),
        m_axes(axes),
        m_in_topology(in_topology),
        m_out_topology(out_topology),
        m_comm(comm) {
    initialize();
  }

  void forward(const InViewType& in, const OutViewType& out) {
    // This part should be fixed
    // forward_block should directly modify out
    KokkosFFT::execute(*m_forward_plan, in, m_fft);
    for (const auto& forward_block : m_forward_blocks) {
      (*forward_block)();
    }
  }

  void backward(const OutViewType& out, const InViewType& in) {
    for (const auto& backward_block : m_backward_blocks) {
      (*backward_block)();
    }
    KokkosFFT::execute(*m_backward_plan, m_fft, in);
  }

 private:
  void initialize() {
    // check that topologies are pencils

    // First get global shape to define buffer and next shape
    auto gin_shape  = get_global_shape(m_in, m_in_topology, m_comm);
    auto gout_shape = get_global_shape(m_out, m_out_topology, m_comm);

    const std::size_t rank = InViewType::rank();

    map_type src_map = {};
    for (std::size_t i = 0; i < rank; ++i) {
      src_map[i] = i;
    }
    auto all_topologies =
        get_shuffled_topologies(m_in_topology, m_out_topology, m_axes);
    auto in_hat_extents = get_next_extents(gout_shape, m_in_topology, src_map);

    extents_type current_extents = in_hat_extents;
    map_type current_map         = src_map;
    std::vector<buffer_extents_type> all_buffer_extents;
    std::vector<extents_type> all_pencil_extents;
    std::vector<paired_map_type> all_maps;
    std::vector<pencil_axes_type> all_pencil_axes;
    std::vector<paired_extents_type> all_paired_extents;
    std::vector<paired_data_type> all_paired_data;

    for (std::size_t i = 0; i < all_topologies.size(); i++) {
      // There are valid intermediate topology
      if (i + 2 < all_topologies.size()) {
        map_type mid_topology =
            get_mid_array(all_topologies.at(i), all_topologies.at(i + 2));
        auto [in_axis01, out_axis01] =
            get_pencil(all_topologies.at(i), mid_topology);
        auto [in_axis12, out_axis12] =
            get_pencil(mid_topology, all_topologies.at(i + 2));
        map_type mid_map01 =
            get_dst_map<LayoutType, rank>(current_map, out_axis01);
        map_type mid_map12 =
            get_dst_map<LayoutType, rank>(mid_map01, out_axis12);
        all_pencil_axes.push_back(pencil_axes_type{in_axis01, out_axis01});
        all_pencil_axes.push_back(pencil_axes_type{in_axis12, out_axis12});
        all_maps.push_back(paired_map_type{current_map, mid_map01});
        all_maps.push_back(paired_map_type{mid_map01, mid_map12});

        // Evaluate buffer extents based on the topology
        auto buffer_01_extents = get_buffer_extents<LayoutType>(
            gout_shape, all_topologies.at(i), mid_topology);
        auto buffer_12_extents = get_buffer_extents<LayoutType>(
            gout_shape, mid_topology, all_topologies.at(i + 2));
        all_buffer_extents.push_back(buffer_01_extents);
        all_buffer_extents.push_back(buffer_12_extents);

        // Evaluate next pencil extents
        auto next_01_extents =
            get_next_extents(gout_shape, mid_topology, mid_map01);
        auto next_12_extents =
            get_next_extents(gout_shape, all_topologies.at(i + 2), mid_map12);
        all_pencil_extents.push_back(next_01_extents);
        all_pencil_extents.push_back(next_12_extents);

        all_paired_extents.push_back(
            paired_extents_type{current_extents, next_01_extents});
        all_paired_extents.push_back(
            paired_extents_type{next_01_extents, next_12_extents});

        // Update the current topology
        current_map     = mid_map01;
        current_extents = next_01_extents;
      }
    }

    // Get the required buffer and pencil sizes
    auto buffer_size = get_required_allocation_size(all_buffer_extents);
    auto pencil_size = get_required_allocation_size(all_pencil_extents);

    // Allocate buffer views
    m_send_buffer_allocation =
        AllocationViewType("send_buffer_allocation", buffer_size);
    m_recv_buffer_allocation =
        AllocationViewType("recv_buffer_allocation", buffer_size);
    m_pencil_allocation0 =
        AllocationViewType("pencil_allocation0", pencil_size);
    m_pencil_allocation1 =
        AllocationViewType("pencil_allocation1", pencil_size);

    m_fft =
        FFTViewType("fft_buffer",
                    KokkosFFT::Impl::create_layout<LayoutType>(in_hat_extents));

    all_paired_data.push_back(
        paired_data_type{m_fft.data(), m_pencil_allocation0.data()});
    all_paired_data.push_back(paired_data_type{m_pencil_allocation0.data(),
                                               m_pencil_allocation1.data()});

    // Create a Cartesian communicator
    std::vector<int> dims;
    for (auto& dim : m_in_topology) {
      if (dim > 1) {
        dims.push_back(static_cast<int>(dim));
      }
    }

    int periods[2] = {1, 1};  // Periodic in all directions
    ::MPI_Comm cart_comm;
    ::MPI_Cart_create(m_comm, 2, dims.data(), periods, /*reorder=*/1,
                      &cart_comm);

    // split into row‐ and col‐ communicators
    ::MPI_Comm row_comm, col_comm;

    int remain_dims[2];

    // keep Y‐axis for row_comm (all procs with same px)
    remain_dims[0] = 1;
    remain_dims[1] = 0;

    ::MPI_Cart_sub(cart_comm, remain_dims, &row_comm);

    // keep X‐axis for col_comm (all procs with same py)
    remain_dims[0] = 0;
    remain_dims[1] = 1;
    ::MPI_Cart_sub(cart_comm, remain_dims, &col_comm);

    std::vector<MPI_Comm> comms = {col_comm, row_comm};

    for (std::size_t i = 0; i < all_buffer_extents.size(); i++) {
      auto [in_extents, out_extents] = all_paired_extents.at(i);
      auto [in_data, out_data]       = all_paired_data.at(i);
      auto [in_axis, out_axis]       = all_pencil_axes.at(i);
      auto [in_map, out_map]         = all_maps.at(i);

      // Make unmanaged view using the allocation
      FFTViewType in_pencil(
          in_data, KokkosFFT::Impl::create_layout<LayoutType>(in_extents));
      FFTViewType out_pencil(
          out_data, KokkosFFT::Impl::create_layout<LayoutType>(out_extents));

      BufferViewType send_buffer(
          m_send_buffer_allocation.data(),
          KokkosFFT::Impl::create_layout<LayoutType>(all_buffer_extents.at(i)));
      BufferViewType recv_buffer(
          m_recv_buffer_allocation.data(),
          KokkosFFT::Impl::create_layout<LayoutType>(all_buffer_extents.at(i)));

      // Make a plan
      m_forward_blocks.push_back(std::make_unique<FFTForwardBlockType>(
          m_exec_space, in_pencil, out_pencil, out_pencil, send_buffer,
          recv_buffer, in_map, in_axis, out_map, out_axis, comms.at(i)));
      m_backward_blocks.push_back(std::make_unique<FFTBackwardBlockType>(
          m_exec_space, out_pencil, out_pencil, in_pencil, send_buffer,
          recv_buffer, out_map, out_axis, in_map, in_axis, comms.at(i)));
    }

    // Backward operation order should be reversed
    std::reverse(m_backward_blocks.begin(), m_backward_blocks.end());

    int fft_axis   = m_axes.back();
    m_forward_plan = std::make_unique<FFTForwardPlanType>(
        m_exec_space, m_in, m_fft, KokkosFFT::Direction::forward, fft_axis);
    m_backward_plan = std::make_unique<FFTBackwardPlanType>(
        m_exec_space, m_fft, m_in, KokkosFFT::Direction::backward, fft_axis);
  }
};

#endif
