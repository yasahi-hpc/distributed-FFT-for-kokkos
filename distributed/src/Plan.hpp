#ifndef PLAN_HPP
#define PLAN_HPP

#include <vector>
#include <memory>
#include <iostream>
#include <sstream>
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
  using int_map_type        = std::array<int, InViewType::rank()>;
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
  FFTViewType m_pencil0, m_pencil1, m_pencil2;
  OutViewType m_out;
  AllocationViewType m_send_buffer_allocation;
  AllocationViewType m_recv_buffer_allocation;

  axes_type m_axes;
  int_map_type m_map_forward, m_map_backward;
  topology_type m_in_topology;
  topology_type m_out_topology;
  MPI_Comm m_comm;
  extents_type m_in_extents, m_out_extents;

  std::unique_ptr<FFTForwardBlockType> m_forward0;
  std::unique_ptr<FFTForwardBlockType> m_forward1;
  std::unique_ptr<FFTBackwardBlockType> m_backward0;
  std::unique_ptr<FFTBackwardBlockType> m_backward1;

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
        m_comm(comm),
        m_in_extents(KokkosFFT::Impl::extract_extents(in)),
        m_out_extents(KokkosFFT::Impl::extract_extents(out)) {
    initialize();
  }

  void forward(const InViewType& in, const OutViewType& out) {
    good(in, out);
    KokkosFFT::execute(*m_forward_plan, in, m_pencil0);

    // First transpose + FFT
    (*m_forward0)(m_pencil0, m_pencil1);

    // Second transpose + FFT
    (*m_forward1)(m_pencil1, m_pencil2);

    safe_transpose(m_exec_space, m_pencil2, out, m_map_forward);
  }

  void backward(const OutViewType& out, const InViewType& in) {
    good(in, out);
    safe_transpose(m_exec_space, out, m_pencil2, m_map_backward);

    // First IFFT + transpose
    (*m_backward1)(m_pencil2, m_pencil1);

    // Second IFFT + transpose
    (*m_backward0)(m_pencil1, m_pencil0);

    KokkosFFT::execute(*m_backward_plan, m_pencil0, in);
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

    // Allocate buffer views
    m_send_buffer_allocation =
        AllocationViewType("send_buffer_allocation", buffer_size);
    m_recv_buffer_allocation =
        AllocationViewType("recv_buffer_allocation", buffer_size);

    m_pencil0 = FFTViewType(
        "pencil0", KokkosFFT::Impl::create_layout<LayoutType>(in_hat_extents));
    m_pencil1 = FFTViewType(
        "pencil1",
        KokkosFFT::Impl::create_layout<LayoutType>(all_pencil_extents.at(0)));
    m_pencil2 = FFTViewType(
        "pencil2",
        KokkosFFT::Impl::create_layout<LayoutType>(all_pencil_extents.at(1)));

    // Create a Cartesian communicator
    std::vector<int> dims;
    for (auto& dim : m_in_topology) {
      if (dim > 1) {
        dims.push_back(static_cast<int>(dim));
      }
    }

    int periods[2] = {1, 1};  // Periodic in all directions
    ::MPI_Comm cart_comm;
    ::MPI_Cart_create(m_comm, 2, dims.data(), periods, 1, &cart_comm);

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

    // Make blocks
    BufferViewType send_buffer0(
        m_send_buffer_allocation.data(),
        KokkosFFT::Impl::create_layout<LayoutType>(all_buffer_extents.at(0)));
    BufferViewType recv_buffer0(
        m_recv_buffer_allocation.data(),
        KokkosFFT::Impl::create_layout<LayoutType>(all_buffer_extents.at(0)));
    BufferViewType send_buffer1(
        m_send_buffer_allocation.data(),
        KokkosFFT::Impl::create_layout<LayoutType>(all_buffer_extents.at(1)));
    BufferViewType recv_buffer1(
        m_recv_buffer_allocation.data(),
        KokkosFFT::Impl::create_layout<LayoutType>(all_buffer_extents.at(1)));

    auto [in_axis0, out_axis0] = all_pencil_axes.at(0);
    auto [in_axis1, out_axis1] = all_pencil_axes.at(1);
    auto [in_map0, out_map0]   = all_maps.at(0);
    auto [in_map1, out_map1]   = all_maps.at(1);

    m_forward0 = std::make_unique<FFTForwardBlockType>(
        m_exec_space, m_pencil0, m_pencil1, m_pencil1, send_buffer0,
        recv_buffer0, in_map0, in_axis0, out_map0, out_axis0, comms.at(0));
    m_backward0 = std::make_unique<FFTBackwardBlockType>(
        m_exec_space, m_pencil1, m_pencil1, m_pencil0, send_buffer0,
        recv_buffer0, out_map0, out_axis0, in_map0, in_axis0, comms.at(0));

    m_forward1 = std::make_unique<FFTForwardBlockType>(
        m_exec_space, m_pencil1, m_pencil2, m_pencil2, send_buffer1,
        recv_buffer1, in_map1, in_axis1, out_map1, out_axis1, comms.at(1));
    m_backward1 = std::make_unique<FFTBackwardBlockType>(
        m_exec_space, m_pencil2, m_pencil2, m_pencil1, send_buffer1,
        recv_buffer1, out_map1, out_axis1, in_map1, in_axis1, comms.at(1));

    // Make maps
    for (std::size_t i = 0; i < rank; ++i) {
      m_map_forward.at(i)  = KokkosFFT::Impl::get_index(out_map1, i);
      m_map_backward.at(i) = out_map1.at(i);
    }

    // Make FFT plans
    int fft_axis   = m_axes.back();
    m_forward_plan = std::make_unique<FFTForwardPlanType>(
        m_exec_space, m_in, m_pencil0, KokkosFFT::Direction::forward, fft_axis);
    m_backward_plan = std::make_unique<FFTBackwardPlanType>(
        m_exec_space, m_pencil0, m_in, KokkosFFT::Direction::backward,
        fft_axis);
  }

  void good(const InViewType& in, const OutViewType& out) const {
    auto in_extents  = KokkosFFT::Impl::extract_extents(in);
    auto out_extents = KokkosFFT::Impl::extract_extents(out);

    KOKKOSFFT_THROW_IF(in_extents != m_in_extents,
                       "extents of input View for plan and "
                       "execution are not identical.");

    KOKKOSFFT_THROW_IF(out_extents != m_out_extents,
                       "extents of output View for plan and "
                       "execution are not identical.");
  }
};

#endif
