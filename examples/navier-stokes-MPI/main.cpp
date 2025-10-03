#include <iostream>
#include <memory>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>
#include "../utils/math_utils.hpp"
#include "../utils/io_utils.hpp"
#include "KokkosFFT_Distributed.hpp"

using execution_space      = Kokkos::DefaultExecutionSpace;
using host_execution_space = Kokkos::DefaultHostExecutionSpace;
template <typename T>
using View1D = Kokkos::View<T*, Kokkos::LayoutRight, execution_space>;
template <typename T>
using View2D = Kokkos::View<T**, Kokkos::LayoutRight, execution_space>;
template <typename T>
using View3D = Kokkos::View<T***, Kokkos::LayoutRight, execution_space>;

using extents_type = std::array<std::size_t, 3>;

using range3D_type = Kokkos::MDRangePolicy<
    execution_space,
    Kokkos::Rank<3, Kokkos::Iterate::Right, Kokkos::Iterate::Right>,
    Kokkos::IndexType<int>>;

// \brief A class to represent the grid used in the Navier-Stokes equation.
// Wavenumber space grids are in z-pencil format whereas real space grids
// are in global format
struct Grid {
  //! Global MPI rank
  int m_rank;

  //! Number of processes in x direction (px)
  int m_px;

  //! Number of processes in y direction (py)
  int m_py;

  //! Local MPI rank in x direction (rx)
  int m_rx;

  //! Local MPI rank in y direction (ry)
  int m_ry;

  //! Input topology of the parallelization
  extents_type m_in_topology;

  //! Output topology of the parallelization
  extents_type m_out_topology;

  //! Grid in x direction (nx)
  View1D<double> m_x;

  //! Grid in y direction (ny))
  View1D<double> m_y;

  //! Grid in z direction (nz)
  View1D<double> m_z;

  //! Wavenumber in x direction (nkx * 2 + 1)
  View1D<double> m_kx;

  //! Wavenumber in y direction (nky * 2 + 1)
  View1D<double> m_ky;

  //! Wavenumber in z direction (nkz + 1)
  View1D<double> m_kzh;

  //! Wavenumber squared (nkx * 2 + 1, nky * 2 + 1, nkz + 1)
  View3D<double> m_ksq;

  //! Inverse of k^2 for pressure projection (nkx * 2 + 1, nky * 2 + 1, nkz + 1)
  View3D<double> m_inv_ksq;

  //! Mask for dealiasing (nkx * 2 + 1, nky * 2 + 1, nkz + 1)
  View3D<double> m_alias_mask;

  // \brief Constructor of a Grid class
  // \param[in] rank Global MPI rank of the process.
  // \param[in] px Number of processes in the x-direction.
  // \param[in] py Number of processes in the y-direction.
  // \param[in] nx Number of grid points in the x-direction.
  // \param[in] ny Number of grid points in the y-direction.
  // \param[in] nz Number of grid points in the z-direction.
  // \param[in] lx Length of the domain in the x-direction.
  // \param[in] ly Length of the domain in the y-direction.
  // \param[in] lz Length of the domain in the z-direction.
  Grid(int rank, int px, int py, int nx, int ny, int nz, double lx, double ly,
       double lz)
      : m_rank(rank), m_px(px), m_py(py) {
    // Check that parallelization is valid
    if (nx % px != 0 || ny % py != 0 || nz % py != 0) {
      throw std::runtime_error(
          "Grid size must be divisible by the number of processes in each "
          "direction.");
    }

    std::array<std::size_t, 2> topology{std::size_t(px), std::size_t(py)};
    auto coord = KokkosFFT::Distributed::rank_to_coord(topology, rank);
    m_rx = coord.at(0), m_ry = coord.at(1);

    // Z-pencil or X-slab (if py==1)
    m_in_topology = {std::size_t(px), std::size_t(py), std::size_t(1)};

    // X-pencil or Y-slab (if py==1)
    m_out_topology = {std::size_t(1), std::size_t(px), std::size_t(py)};

    // Grid and Wavenumbers
    execution_space exec;
    m_x = Math::linspace(exec, 0.0, lx * M_PI, nx, /*endpoint=*/false);
    m_y = Math::linspace(exec, 0.0, ly * M_PI, ny, /*endpoint=*/false);
    m_z = Math::linspace(exec, 0.0, lz * M_PI, nz, /*endpoint=*/false);

    // Wavenumbers
    auto [out_extents, out_starts] = KokkosFFT::Distributed::get_local_extents(
        extents_type(
            {std::size_t(nx), std::size_t(ny), std::size_t(nz / 2 + 1)}),
        m_out_topology, MPI_COMM_WORLD);
    auto [nkx, nky, nkz] = out_extents;
    m_kx                 = View1D<double>("kx", nkx);
    m_ky                 = View1D<double>("ky", nky);
    m_kzh                = View1D<double>("kzh", nkz);
    m_ksq                = View3D<double>("ksq", nkx, nky, nkz);
    m_inv_ksq            = View3D<double>("inv_ksq", nkx, nky, nkz);
    m_alias_mask         = View3D<double>("alias_mask", nkx, nky, nkz);

    host_execution_space host_exec;

    // [0, dkx, 2*dkx, ..., nkx * dkx, -nkx * dkx, ..., -dkx]
    double dkx = lx / static_cast<double>(2 * nx);
    auto h_kx  = KokkosFFT::fftfreq(host_exec, nx, dkx);

    // [0, dky, 2*dky, ..., nky * dky, -nky * dkyx, ..., -dky]
    double dky = ly / static_cast<double>(2 * ny);
    auto h_gky = KokkosFFT::fftfreq(host_exec, ny, dky);

    // [0, dkz, 2*dkz, ..., nkz * dkz]
    double dkz  = lz / static_cast<double>(2 * nz);
    auto h_gkzh = KokkosFFT::rfftfreq(host_exec, nz, dkz);

    // kx**2 + ky**2 + kz**2
    auto h_ksq     = Kokkos::create_mirror_view(m_ksq);
    auto h_inv_ksq = Kokkos::create_mirror_view(m_inv_ksq);
    for (int ikz = 0; ikz < h_ksq.extent_int(2); ikz++) {
      for (int iky = 0; iky < h_ksq.extent_int(1); iky++) {
        for (int ikx = 0; ikx < h_ksq.extent_int(0); ikx++) {
          int giky             = iky + out_starts.at(1);  // Global index in y
          int gikz             = ikz + out_starts.at(2);  // Global index in z
          h_ksq(ikx, iky, ikz) = h_kx(ikx) * h_kx(ikx) +
                                 +h_gky(giky) * h_gky(giky) +
                                 h_gkzh(gikz) * h_gkzh(gikz);
          h_inv_ksq(ikx, iky, ikz) = (ikx == 0 && giky == 0 && gikz == 0)
                                         ? 0.0
                                         : 1.0 / (h_ksq(ikx, iky, ikz));
        }
      }
    }

    // Dealiasing Mask (2/3 rule)
    // Keep modes k < k_max * (2/3)
    int kx_max = nx / 2, ky_max = ny / 2, kz_max = nz / 2;
    int cutoff_x = int(kx_max * 2 / 3);
    int cutoff_y = int(ky_max * 2 / 3);
    int cutoff_z = int(kz_max * 2 / 3);

    // Indices for FFT frequencies: 0, 1, ..., N/2-1, -N/2, ..., -1
    // We want to zero out frequencies |k_i| > cutoff
    auto h_kx_ind = KokkosFFT::fftfreq(host_exec, nx, 1.0);
    auto h_ky_ind = KokkosFFT::fftfreq(host_exec, ny, 1.0);
    auto h_kz_ind = KokkosFFT::rfftfreq(host_exec, nz, 1.0);

    auto h_alias_mask = Kokkos::create_mirror_view(m_alias_mask);
    for (int ikz = 0; ikz < h_alias_mask.extent_int(2); ikz++) {
      for (int iky = 0; iky < h_alias_mask.extent_int(1); iky++) {
        for (int ikx = 0; ikx < h_alias_mask.extent_int(0); ikx++) {
          int giky    = iky + out_starts.at(1);  // Global index in y
          int gikz    = ikz + out_starts.at(2);  // Global index in z
          bool cutoff = (Kokkos::abs(h_kx_ind(ikx) * nx) > cutoff_x) ||
                        (Kokkos::abs(h_ky_ind(giky) * ny) > cutoff_y) ||
                        (Kokkos::abs(h_kz_ind(gikz) * nz) > cutoff_z);
          h_alias_mask(ikx, iky, ikz) = static_cast<double>(!cutoff);
        }
      }
    }

    auto h_ky  = Kokkos::create_mirror_view(m_ky);
    auto h_kzh = Kokkos::create_mirror_view(m_kzh);
    for (int iky = 0; iky < m_ky.extent_int(0); iky++) {
      int giky  = iky + out_starts.at(1);  // Global index in y
      h_ky(iky) = h_gky(giky);
    }
    for (int ikz = 0; ikz < m_kzh.extent_int(0); ikz++) {
      int gikz   = ikz + out_starts.at(2);  // Global index in z
      h_kzh(ikz) = h_gkzh(gikz);
    }

    Kokkos::deep_copy(m_kx, h_kx);
    Kokkos::deep_copy(m_ky, h_ky);
    Kokkos::deep_copy(m_kzh, h_kzh);
    Kokkos::deep_copy(m_ksq, h_ksq);
    Kokkos::deep_copy(m_inv_ksq, h_inv_ksq);
    Kokkos::deep_copy(m_alias_mask, h_alias_mask);
  }
};

// \brief Apply the dealiase mask in Fourier space
//
// \tparam ViewType The type of the view
// \tparam MaskViewType The type of the mask view
// \param[in,out] view View to be modified
// \param[in] mask The dealiase mask
template <typename ViewType, typename MaskViewType>
void dealias(const ViewType& view, const MaskViewType& mask) {
  static_assert(ViewType::rank() == 3, "dealias: View rank should be 3");
  const int n0 = mask.extent(0), n1 = mask.extent(1), n2 = mask.extent(2);

  range3D_type policy({0, 0, 0}, {n0, n1, n2});
  Kokkos::parallel_for(
      "dealias", policy, KOKKOS_LAMBDA(int i0, int i1, int i2) {
        auto dealias_mask = mask(i0, i1, i2);
        view(i0, i1, i2) *= dealias_mask;
      });
}

// \brief Projects the velocity field onto the divergence-free space in Fourier
// domain. u^_proj = u^ - k (k \dot u^) / k^2
//
// \tparam ViewType The type of the first field.
// \param[in] grid The grid in Fourier space
// \param[in, out] uk The Fourier representation of the velocity field u
// \param[in, out] vk The Fourier representation of the velocity field v
// \param[in, out] wk The Fourier representation of the velocity field w
template <typename ViewType>
void projection(const Grid& grid, const ViewType& uk, const ViewType& vk,
                const ViewType& wk) {
  auto kx      = grid.m_kx;
  auto ky      = grid.m_ky;
  auto kzh     = grid.m_kzh;
  auto inv_ksq = grid.m_inv_ksq;
  const int n0 = uk.extent(0), n1 = uk.extent(1), n2 = uk.extent(2);
  range3D_type policy({0, 0, 0}, {n0, n1, n2});
  Kokkos::parallel_for(
      "projection", policy, KOKKOS_LAMBDA(int i0, int i1, int i2) {
        Kokkos::complex<double> div_hat =
            (kx(i0) * uk(i0, i1, i2) + ky(i1) * vk(i0, i1, i2) +
             kzh(i2) * wk(i0, i1, i2)) *
            inv_ksq(i0, i1, i2);
        uk(i0, i1, i2) -= kx(i0) * div_hat;
        vk(i0, i1, i2) -= ky(i1) * div_hat;
        wk(i0, i1, i2) -= kzh(i2) * div_hat;
      });
}

// \brief A class to represent the variables used in the Navier-Stokes equations
struct Variables {
  //! Velocity fields in the x, y and z directions. (z-pencil format)
  View3D<Kokkos::complex<double>> m_uk, m_vk, m_wk;

  //! Time derivative of Velocity fields in the x, y and z directions. (z-pencil
  //! format)
  View3D<Kokkos::complex<double>> m_dukdt, m_dvkdt, m_dwkdt;

  //! Buffer view to store the spatial derivatives of u (z-pencil format)
  View3D<Kokkos::complex<double>> m_dukdx, m_dukdy, m_dukdz, m_dvkdx, m_dvkdy,
      m_dvkdz, m_dwkdx, m_dwkdy, m_dwkdz;

  //! Buffer view to store the velocity fields in the x, y and z directions
  //! (x-pencil format)
  View3D<double> m_u, m_v, m_w;

  //! Buffer view to store the time derivative of u, v, w (x-pencil format)
  View3D<double> m_dudx, m_dudy, m_dudz, m_dvdx, m_dvdy, m_dvdz, m_dwdx, m_dwdy,
      m_dwdz;

  // \brief Constructor of a Variables class
  // Taylor-Green vortex is used as the initial condition.
  // \param[in] grid Grid in Fourier space
  // \param[in] suppress_diag If true, suppresses value initialization.
  // \param[in] v0 Initial velocity magnitude. Defaults to 1.0.
  Variables(const Grid& grid, bool suppress_diag, double v0 = 1.0) {
    auto h_x =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grid.m_x);
    auto h_y =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grid.m_y);
    auto h_z =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grid.m_z);

    std::size_t nx = h_x.extent(0), ny = h_y.extent(0), nz = h_z.extent(0);

    // Compute the extents in z-pencils
    auto [nin0, nin1, nin2] = KokkosFFT::Distributed::get_local_shape(
        extents_type({nx, ny, nz}), grid.m_in_topology, MPI_COMM_WORLD);

    // Compute the extents in x-pencils
    auto [nout0, nout1, nout2] = KokkosFFT::Distributed::get_local_shape(
        extents_type({nx, ny, nz / 2 + 1}), grid.m_out_topology,
        MPI_COMM_WORLD);

    // Create views in z-pencil format
    m_u    = View3D<double>("u", nin0, nin1, nin2);
    m_v    = View3D<double>("v", nin0, nin1, nin2);
    m_w    = View3D<double>("w", nin0, nin1, nin2);
    m_dudx = View3D<double>("dudx", nin0, nin1, nin2);
    m_dudy = View3D<double>("dudy", nin0, nin1, nin2);
    m_dudz = View3D<double>("dudz", nin0, nin1, nin2);
    m_dvdx = View3D<double>("dvdx", nin0, nin1, nin2);
    m_dvdy = View3D<double>("dvdy", nin0, nin1, nin2);
    m_dvdz = View3D<double>("dvdz", nin0, nin1, nin2);
    m_dwdx = View3D<double>("dwdx", nin0, nin1, nin2);
    m_dwdy = View3D<double>("dwdy", nin0, nin1, nin2);
    m_dwdz = View3D<double>("dwdz", nin0, nin1, nin2);

    // Create views in x-pencil format
    m_uk    = View3D<Kokkos::complex<double>>("uk", nout0, nout1, nout2);
    m_vk    = View3D<Kokkos::complex<double>>("vk", nout0, nout1, nout2);
    m_wk    = View3D<Kokkos::complex<double>>("wk", nout0, nout1, nout2);
    m_dukdt = View3D<Kokkos::complex<double>>("dukdt", nout0, nout1, nout2);
    m_dvkdt = View3D<Kokkos::complex<double>>("dvkdt", nout0, nout1, nout2);
    m_dwkdt = View3D<Kokkos::complex<double>>("dwkdt", nout0, nout1, nout2);
    m_dukdx = View3D<Kokkos::complex<double>>("dukdx", nout0, nout1, nout2);
    m_dukdy = View3D<Kokkos::complex<double>>("dukdy", nout0, nout1, nout2);
    m_dukdz = View3D<Kokkos::complex<double>>("dukdz", nout0, nout1, nout2);
    m_dvkdx = View3D<Kokkos::complex<double>>("dvkdx", nout0, nout1, nout2);
    m_dvkdy = View3D<Kokkos::complex<double>>("dvkdy", nout0, nout1, nout2);
    m_dvkdz = View3D<Kokkos::complex<double>>("dvkdz", nout0, nout1, nout2);
    m_dwkdx = View3D<Kokkos::complex<double>>("dwkdx", nout0, nout1, nout2);
    m_dwkdy = View3D<Kokkos::complex<double>>("dwkdy", nout0, nout1, nout2);
    m_dwkdz = View3D<Kokkos::complex<double>>("dwkdz", nout0, nout1, nout2);

    // Data in z-pencil format
    // (nx/px, ny/py, nz)
    auto h_u = Kokkos::create_mirror_view(m_u);
    auto h_v = Kokkos::create_mirror_view(m_v);
    auto h_w = Kokkos::create_mirror_view(m_w);
    for (std::size_t iz = 0; iz < nin2; iz++) {
      for (std::size_t iy = 0; iy < nin1; iy++) {
        for (std::size_t ix = 0; ix < nin0; ix++) {
          std::size_t gix = ix + grid.m_rx * nin0;  // Global index in x
          std::size_t giy = iy + grid.m_ry * nin1;  // Global index in y

          double x = h_x(gix), y = h_y(giy), z = h_z(iz);
          h_u(ix, iy, iz) =
              v0 * Kokkos::sin(x) * Kokkos::cos(y) * Kokkos::cos(z);
          h_v(ix, iy, iz) =
              -v0 * Kokkos::cos(x) * Kokkos::sin(y) * Kokkos::cos(z);
          h_w(ix, iy, iz) = 0.0;
        }
      }
    }
    Kokkos::deep_copy(m_u, h_u);
    Kokkos::deep_copy(m_v, h_v);
    Kokkos::deep_copy(m_w, h_w);

    // If suppress_diag is true, we do not initialize values
    // This allows outputs from kokkos-tools simpler
    if (!suppress_diag) {
      using execution_space = Kokkos::DefaultExecutionSpace;
      execution_space exec;
      KokkosFFT::Distributed::Plan plan(
          exec, m_u, m_uk, KokkosFFT::axis_type<3>{0, 1, 2}, grid.m_in_topology,
          grid.m_out_topology, MPI_COMM_WORLD);

      KokkosFFT::Distributed::execute(plan, m_u, m_uk,
                                      KokkosFFT::Direction::forward);
      KokkosFFT::Distributed::execute(plan, m_v, m_vk,
                                      KokkosFFT::Direction::forward);
      KokkosFFT::Distributed::execute(plan, m_w, m_wk,
                                      KokkosFFT::Direction::forward);

      dealias(m_uk, grid.m_alias_mask);
      dealias(m_vk, grid.m_alias_mask);
      dealias(m_wk, grid.m_alias_mask);

      projection(grid, m_uk, m_vk, m_wk);
    }
  }
};

class NavierStokes {
  using OdeSolverType =
      Math::RK4th<execution_space, View1D<Kokkos::complex<double>>>;
  using DistributedPlanType =
      KokkosFFT::Distributed::Plan<execution_space, View3D<double>,
                                   View3D<Kokkos::complex<double>>, 3>;

  //! The ODE solver used in the simulation
  std::unique_ptr<OdeSolverType> m_ode_x, m_ode_y, m_ode_z;

  //! The distributed FFT plan used in the simulation
  std::unique_ptr<DistributedPlanType> m_plan;

  //! Radial bins of wavenumbers and energy
  View1D<double> m_k_bins, m_k_vals, m_E_spec;

  //! The rank of the current process.
  const int m_rank;

  ///@{
  //! The number of processes in each direction
  const int m_px, m_py;
  ///@}

  //! The total number of iterations.
  const int m_nbiter;

  //! The time step size.
  const double m_dt;

  //! The viscosity coefficient.
  const double m_nu;

  ///@{
  //! The directory to output diagnostic data.
  std::string m_base_out_dir, m_out_dir;
  ;
  ///@}

  //! If true, suppresses diagnostics.
  const bool m_suppress_diag;

  //! The grid used in the simulation
  Grid m_grid;

  //! The variables used in the simulation
  Variables m_variables;

  double m_coef;
  double m_time = 0.0;
  int m_diag_it = 0, m_diag_steps = 50;
  int m_num_bins = 50;

 public:
  // \brief Constructor of a HasegawaWakatani class
  // \param[in] rank The rank of the current process.
  // \param[in] px The number of processors in x direction.
  // \param[in] py The number of processors in y direction.
  // \param[in] nx The number of grid points in each direction.
  // \param[in] lx The length of the domain in each direction.
  // \param[in] nbiter The total number of iterations.
  // \param[in] dt The time step size.
  // \param[in] nu The viscosity coefficient.
  // \param[in] out_dir The directory to output diagnostic data.
  // \param[in] suppress_diag If true, suppresses diagnostics.
  NavierStokes(int rank, int px, int py, int nx, double lx, int nbiter,
               double dt, double nu, const std::string& out_dir,
               bool suppress_diag)
      : m_rank(rank),
        m_px(px),
        m_py(py),
        m_nbiter(nbiter),
        m_dt(dt),
        m_nu(nu),
        m_base_out_dir(out_dir),
        m_suppress_diag(suppress_diag),
        m_grid(rank, px, py, nx, nx, nx, lx, lx, lx),
        m_variables(m_grid, suppress_diag) {
    execution_space exec;
    View1D<Kokkos::complex<double>> uk_flatten(m_variables.m_uk.data(),
                                               m_variables.m_uk.size());
    m_ode_x = std::make_unique<OdeSolverType>(exec, uk_flatten, dt);
    View1D<Kokkos::complex<double>> vk_flatten(m_variables.m_vk.data(),
                                               m_variables.m_vk.size());
    m_ode_y = std::make_unique<OdeSolverType>(exec, vk_flatten, dt);
    View1D<Kokkos::complex<double>> wk_flatten(m_variables.m_wk.data(),
                                               m_variables.m_wk.size());
    m_ode_z = std::make_unique<OdeSolverType>(exec, wk_flatten, dt);

    if (!m_suppress_diag) {
      namespace fs = std::filesystem;
      // Make a directory for this parallelization (px, py)
      m_out_dir = m_base_out_dir + "_px" + std::to_string(m_px) + "_py" +
                  std::to_string(m_py);

      if (m_rank == 0) {
        IO::mkdir(m_out_dir, fs::perms::owner_all | fs::perms::group_read |
                                 fs::perms::group_exec |
                                 fs::perms::others_read |
                                 fs::perms::others_exec);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // Create FFT plans
    m_plan = std::make_unique<DistributedPlanType>(
        exec, m_variables.m_u, m_variables.m_uk,
        KokkosFFT::axis_type<3>{0, 1, 2}, m_grid.m_in_topology,
        m_grid.m_out_topology, MPI_COMM_WORLD);

    // Preparation for diagnostics
    // Define radial bins (e.g., 0 to k_max)
    auto h_ksq =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m_grid.m_ksq);

    const int n0 = h_ksq.extent(0), n1 = h_ksq.extent(1), n2 = h_ksq.extent(2);
    double k_max = 0.0;
    for (int i0 = 0; i0 < n0; i0++) {
      for (int i1 = 0; i1 < n1; i1++) {
        for (int i2 = 0; i2 < n2; i2++) {
          k_max = Kokkos::max(k_max, Kokkos::sqrt(h_ksq(i0, i1, i2)));
        }
      }
    }
    double gk_max = 0.0;
    MPI_Allreduce(&k_max, &gk_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    k_max = gk_max;

    m_k_bins = View1D<double>("k_bins", m_num_bins + 1);
    m_k_vals = View1D<double>("k_vals", m_num_bins);
    m_E_spec = View1D<double>("E_spec", m_num_bins);

    m_k_bins =
        Math::linspace(exec, 0.0, k_max, m_num_bins + 1, /*endpoint=*/true);
    auto h_k_bins =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m_k_bins);
    auto h_k_vals = Kokkos::create_mirror_view(m_k_vals);
    for (int i = 0; i < m_num_bins; i++) {
      h_k_vals(i) = 0.5 * (h_k_bins(i) + h_k_bins(i + 1));  // midpoints
    }
    Kokkos::deep_copy(m_k_vals, h_k_vals);

    // Coefficient to calculate total kinetic energy
    m_coef =
        Kokkos::pow(lx * M_PI, 3) / Kokkos::pow(static_cast<double>(nx), 6);
  }

  // \brief Runs the simulation for the specified number of iterations.
  void run() {
    m_time = 0.0;
    for (int iter = 0; iter < m_nbiter; iter++) {
      if (!m_suppress_diag) diag(iter);
      solve();
      m_time += m_dt;
    }
  }

  // \brief Advances the simulation by one time step.
  void solve() {
    for (int step = 0; step < m_ode_x->order(); step++) {
      rhs(m_variables.m_uk, m_variables.m_vk, m_variables.m_wk,
          m_variables.m_dukdt, m_variables.m_dvkdt, m_variables.m_dwkdt);

      // Flatten Views for time integral
      View1D<Kokkos::complex<double>> uk(m_variables.m_uk.data(),
                                         m_variables.m_uk.size()),
          vk(m_variables.m_vk.data(), m_variables.m_vk.size()),
          wk(m_variables.m_wk.data(), m_variables.m_wk.size()),
          dukdt(m_variables.m_dukdt.data(), m_variables.m_dukdt.size()),
          dvkdt(m_variables.m_dvkdt.data(), m_variables.m_dvkdt.size()),
          dwkdt(m_variables.m_dwkdt.data(), m_variables.m_dwkdt.size());
      m_ode_x->advance(dukdt, uk, step);
      m_ode_y->advance(dvkdt, vk, step);
      m_ode_z->advance(dwkdt, wk, step);
    }
  }

  // \brief Computes the RHS of Navier-Stokes equation
  //
  // \tparam ViewType The type of the velocity field.
  // \param[in] uk The Fourier representation of the velocity field u
  // \param[in] vk The Fourier representation of the velocity field v
  // \param[in] wk The Fourier representation of the velocity field w
  // \param[out] dukdt The x component of the RHS of the vorticity equation.
  // \param[out] dvkdt The y component of the RHS of the vorticity equation.
  // \param[out] dwkdt The z component of the RHS of the vorticity equation.
  template <typename ViewType>
  void rhs(const ViewType& uk, const ViewType& vk, const ViewType& wk,
           const ViewType& dukdt, const ViewType& dvkdt,
           const ViewType& dwkdt) {
    // Apply dealiasing filter before inverse transform
    dealias(uk, m_grid.m_alias_mask);
    dealias(vk, m_grid.m_alias_mask);
    dealias(wk, m_grid.m_alias_mask);

    // Calculate velocity derivatives in Fourier space
    auto dukdx = m_variables.m_dukdx;
    auto dukdy = m_variables.m_dukdy;
    auto dukdz = m_variables.m_dukdz;
    auto dvkdx = m_variables.m_dvkdx;
    auto dvkdy = m_variables.m_dvkdy;
    auto dvkdz = m_variables.m_dvkdz;
    auto dwkdx = m_variables.m_dwkdx;
    auto dwkdy = m_variables.m_dwkdy;
    auto dwkdz = m_variables.m_dwkdz;
    derivative(uk, dukdx, dukdy, dukdz);
    derivative(vk, dvkdx, dvkdy, dvkdz);
    derivative(wk, dwkdx, dwkdy, dwkdz);

    auto dudx = m_variables.m_dudx;
    auto dudy = m_variables.m_dudy;
    auto dudz = m_variables.m_dudz;
    auto dvdx = m_variables.m_dvdx;
    auto dvdy = m_variables.m_dvdy;
    auto dvdz = m_variables.m_dvdz;
    auto dwdx = m_variables.m_dwdx;
    auto dwdy = m_variables.m_dwdy;
    auto dwdz = m_variables.m_dwdz;

    // Inverse FFT to get derivatives in real space
    // X-pencil to Z-pencil format

    // Since the input can be modified,
    // we need to copy uk to dukdt and perform FFT on dukdt
    auto u = m_variables.m_u;
    auto v = m_variables.m_v;
    auto w = m_variables.m_w;
    Kokkos::deep_copy(dukdt, uk);
    Kokkos::deep_copy(dvkdt, vk);
    Kokkos::deep_copy(dwkdt, wk);
    KokkosFFT::Distributed::execute(*m_plan, dukdt, u,
                                    KokkosFFT::Direction::backward);
    KokkosFFT::Distributed::execute(*m_plan, dvkdt, v,
                                    KokkosFFT::Direction::backward);
    KokkosFFT::Distributed::execute(*m_plan, dwkdt, w,
                                    KokkosFFT::Direction::backward);
    KokkosFFT::Distributed::execute(*m_plan, dukdx, dudx,
                                    KokkosFFT::Direction::backward);
    KokkosFFT::Distributed::execute(*m_plan, dukdy, dudy,
                                    KokkosFFT::Direction::backward);
    KokkosFFT::Distributed::execute(*m_plan, dukdz, dudz,
                                    KokkosFFT::Direction::backward);
    KokkosFFT::Distributed::execute(*m_plan, dvkdx, dvdx,
                                    KokkosFFT::Direction::backward);
    KokkosFFT::Distributed::execute(*m_plan, dvkdy, dvdy,
                                    KokkosFFT::Direction::backward);
    KokkosFFT::Distributed::execute(*m_plan, dvkdz, dvdz,
                                    KokkosFFT::Direction::backward);
    KokkosFFT::Distributed::execute(*m_plan, dwkdx, dwdx,
                                    KokkosFFT::Direction::backward);
    KokkosFFT::Distributed::execute(*m_plan, dwkdy, dwdy,
                                    KokkosFFT::Direction::backward);
    KokkosFFT::Distributed::execute(*m_plan, dwkdz, dwdz,
                                    KokkosFFT::Direction::backward);

    // Calculate nonlinear advection terms in real space
    // in Z-pencil or X-slab (if py==1)
    advection(u, v, w, dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz);

    // FFT of nonlinear terms (stored in dukdt)
    // Z-pencil to X-pencil format
    KokkosFFT::Distributed::execute(*m_plan, u, dukdt,
                                    KokkosFFT::Direction::forward);
    KokkosFFT::Distributed::execute(*m_plan, v, dvkdt,
                                    KokkosFFT::Direction::forward);
    KokkosFFT::Distributed::execute(*m_plan, w, dwkdt,
                                    KokkosFFT::Direction::forward);

    dealias(dukdt, m_grid.m_alias_mask);
    dealias(dvkdt, m_grid.m_alias_mask);
    dealias(dwkdt, m_grid.m_alias_mask);

    // Calculate viscous diffusion term in Fourier space
    // And combine terms (stored in dukdt)
    combine(dukdt, dvkdt, dwkdt, uk, vk, wk);

    // Project the RHS onto the divergence-free space
    // [NOTE] This is an external function, which is
    // unit-tested
    projection(m_grid, dukdt, dvkdt, dwkdt);
  }

  // \brief Computes the derivative in Fourier space
  //
  // \tparam ViewType The type of the first field.
  // \param[in] uk The Fourier representation of the velocity field
  // \param[out] dukdx The x derivative of the velocity field.
  // \param[out] dukdy The y derivative of the velocity field.
  // \param[out] dukdz The z derivative of the velocity field.
  template <typename ViewType>
  void derivative(const ViewType& uk, const ViewType& dukdx,
                  const ViewType& dukdy, const ViewType& dukdz) {
    auto kx  = m_grid.m_kx;
    auto ky  = m_grid.m_ky;
    auto kzh = m_grid.m_kzh;

    const Kokkos::complex<double> z(0.0, 1.0);  // Imaginary unit
    const int n0 = uk.extent(0), n1 = uk.extent(1), n2 = uk.extent(2);

    range3D_type policy({0, 0, 0}, {n0, n1, n2});
    Kokkos::parallel_for(
        "derivative", policy, KOKKOS_LAMBDA(int i0, int i1, int i2) {
          Kokkos::complex<double> ikx = z * kx(i0), iky = z * ky(i1),
                                  ikz = z * kzh(i2);
          dukdx(i0, i1, i2)           = ikx * uk(i0, i1, i2);
          dukdy(i0, i1, i2)           = iky * uk(i0, i1, i2);
          dukdz(i0, i1, i2)           = ikz * uk(i0, i1, i2);
        });
  }

  // \brief Computes the nonlinear advection term in real space
  // u.grad(u) = u * du/dx + v * dv/dx + w * dw/dx
  // The operation is made in x-pencil or xz-slab (if py==1) format
  //
  // \tparam ViewType The type of the velocity field.
  // \param[in,out] u On entrance, the velocity field. On exit, the
  // nonlinear advection term
  // \param[in,out] v On entrance, the velocity field. On exit, the
  // nonlinear advection term
  // \param[in,out] w On entrance, the velocity field. On exit, the
  // nonlinear advection term
  // \param[in] dudx The x derivative of velocity field u.
  // \param[in] dudy The y derivative of velocity field u.
  // \param[in] dudz The z derivative of velocity field u.
  // \param[in] dvdx The x derivative of velocity field v.
  // \param[in] dvdy The y derivative of velocity field v.
  // \param[in] dvdz The z derivative of velocity field v.
  // \param[in] dwdx The x derivative of velocity field w.
  // \param[in] dwdy The y derivative of velocity field w.
  // \param[in] dwdz The z derivative of velocity field w.
  template <typename ViewType>
  void advection(const ViewType& u, const ViewType& v, const ViewType& w,
                 const ViewType& dudx, const ViewType& dudy,
                 const ViewType& dudz, const ViewType& dvdx,
                 const ViewType& dvdy, const ViewType& dvdz,
                 const ViewType& dwdx, const ViewType& dwdy,
                 const ViewType& dwdz) {
    const int n0 = u.extent(0), n1 = u.extent(1), n2 = u.extent(2);

    range3D_type policy({0, 0, 0}, {n0, n1, n2});
    Kokkos::parallel_for(
        "advection", policy, KOKKOS_LAMBDA(int i0, int i1, int i2) {
          double u_tmp = u(i0, i1, i2), v_tmp = v(i0, i1, i2),
                 w_tmp = w(i0, i1, i2);

          u(i0, i1, i2) = u_tmp * dudx(i0, i1, i2) + v_tmp * dudy(i0, i1, i2) +
                          w_tmp * dudz(i0, i1, i2);
          v(i0, i1, i2) = u_tmp * dvdx(i0, i1, i2) + v_tmp * dvdy(i0, i1, i2) +
                          w_tmp * dvdz(i0, i1, i2);
          w(i0, i1, i2) = u_tmp * dwdx(i0, i1, i2) + v_tmp * dwdy(i0, i1, i2) +
                          w_tmp * dwdz(i0, i1, i2);
        });
  }

  // \brief Computes the RHS excluding pressure in Fourier space
  // - advection_term (-u.grad(u))_hat + viscosity term (-nu * k^2 * u_hat)
  //
  // \tparam ViewType The type of the velocity field.
  // \param[in,out] advk_x On entrance, the x component of the advection term.
  // On exit, the x component of RHS excluding pressure
  // \param[in,out] advk_y On entrance, the y component of the advection term.
  // On exit, the y component of RHS excluding pressure
  // \param[in,out] advk_z On entrance, the z component of the advection term.
  // On exit, the z component of RHS excluding pressure
  // \param[in] uk The x component of the velocity field.
  // \param[in] vk The y component of the velocity field.
  // \param[in] wk The z component of the velocity field.
  template <typename ViewType>
  void combine(const ViewType& advk_x, const ViewType& advk_y,
               const ViewType& advk_z, const ViewType& uk, const ViewType& vk,
               const ViewType& wk) {
    const int n0 = uk.extent(0), n1 = uk.extent(1), n2 = uk.extent(2);
    const double nu = m_nu;
    auto ksq        = m_grid.m_ksq;

    range3D_type policy({0, 0, 0}, {n0, n1, n2});
    Kokkos::parallel_for(
        "rhs", policy, KOKKOS_LAMBDA(int i0, int i1, int i2) {
          double visc        = -nu * ksq(i0, i1, i2);
          advk_x(i0, i1, i2) = -advk_x(i0, i1, i2) + visc * uk(i0, i1, i2);
          advk_y(i0, i1, i2) = -advk_y(i0, i1, i2) + visc * vk(i0, i1, i2);
          advk_z(i0, i1, i2) = -advk_z(i0, i1, i2) + visc * wk(i0, i1, i2);
        });
  }

  // \brief Performs diagnostics at a given simulation time.
  // \param[in] iter The current iteration number.
  void diag(const int iter) {
    if (iter % m_diag_steps == 0) {
      diag_fields(m_diag_it);
      diag_energy(m_diag_it);
      m_diag_it += 1;
    }
  }

  // \brief Prepare Views to be saved to a binary file
  // \param[in] iter The current iteration number.
  void diag_fields(const int iter) {
    // Inverse FFT to get velocity in real space
    Kokkos::deep_copy(m_variables.m_dukdt, m_variables.m_uk);
    Kokkos::deep_copy(m_variables.m_dvkdt, m_variables.m_vk);
    Kokkos::deep_copy(m_variables.m_dwkdt, m_variables.m_wk);
    KokkosFFT::Distributed::execute(*m_plan, m_variables.m_dukdt,
                                    m_variables.m_u,
                                    KokkosFFT::Direction::backward);
    KokkosFFT::Distributed::execute(*m_plan, m_variables.m_dvkdt,
                                    m_variables.m_v,
                                    KokkosFFT::Direction::backward);
    KokkosFFT::Distributed::execute(*m_plan, m_variables.m_dwkdt,
                                    m_variables.m_w,
                                    KokkosFFT::Direction::backward);

    to_binary_file("u", m_variables.m_u, iter);
    to_binary_file("v", m_variables.m_v, iter);
    to_binary_file("w", m_variables.m_w, iter);
  }

  // \brief Saves the kinetic energy
  // \param[in] iter The current iteration number.
  void diag_energy(const int iter) {
    // Cleanup bins
    Kokkos::deep_copy(m_E_spec, 0.0);
    auto uk      = m_variables.m_uk;
    auto vk      = m_variables.m_vk;
    auto wk      = m_variables.m_wk;
    auto ksq     = m_grid.m_ksq;
    auto E_spec  = m_E_spec;
    auto k_bins  = m_k_bins;
    const int n0 = uk.extent(0), n1 = uk.extent(1), n2 = uk.extent(2);
    const int num_bins = m_E_spec.extent(0);

    range3D_type policy({0, 0, 0}, {n0, n1, n2});
    double coef   = m_coef;
    double energy = 0.0;
    Kokkos::parallel_reduce(
        "total-energy", policy,
        KOKKOS_LAMBDA(int i0, int i1, int i2, double& l_energy) {
          auto k_mag = Kokkos::sqrt(ksq(i0, i1, i2));
          auto u_tmp = uk(i0, i1, i2), v_tmp = vk(i0, i1, i2),
               w_tmp = wk(i0, i1, i2);
          double factor =
              i2 == 0 ? 1.0
                      : 2.0;  // account for Hermitian symmetry in z direction
          auto Ek = 0.5 * factor *
                    (Kokkos::abs(u_tmp * u_tmp) + Kokkos::abs(v_tmp * v_tmp) +
                     Kokkos::abs(w_tmp * w_tmp));
          l_energy += Ek * coef;  // total energy
          for (int i = 0; i < num_bins; i++) {
            bool mask = (k_mag >= k_bins(i)) && (k_mag < k_bins(i + 1));
            if (mask) {
              Kokkos::atomic_add(&E_spec(i), Ek);
            }
          }
        },
        energy);

    // Gather data at rank == 0
    View1D<double> gE_spec("gE_spec", E_spec.layout());
    MPI_Reduce(E_spec.data(), gE_spec.data(), E_spec.size(), MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);

    double genergy = 0.0;
    MPI_Reduce(&energy, &genergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    energy = genergy;

    // Output should be made from master process
    if (m_rank == 0) {
      std::cout << "Step: " << iter * m_diag_steps << "/" << m_nbiter
                << ", Time: " << m_time << ", Kinetic Energy: " << energy
                << std::endl;

      to_binary_file("E_spec", E_spec, iter);
      to_binary_file("k_vals", m_k_vals, iter);
    }
  }

  // \brief Saves a View to a binary file
  //
  // \tparam ViewType The type of the field to be saved.
  // \param[in] label The label of the field.
  // \param[in] value The field to be saved.
  // \param[in] iter The current iteration number.
  template <typename ViewType>
  void to_binary_file(const std::string& label, const ViewType& value,
                      const int iter) {
    using value_type = typename ViewType::non_const_value_type;
    using elem_type =
        KokkosFFT::Impl::add_pointer_n_t<value_type, ViewType::rank()>;

    Kokkos::View<elem_type, Kokkos::LayoutRight, execution_space> out(
        label, value.layout());
    Kokkos::deep_copy(out, value);

    std::string file_name =
        m_out_dir + "/" + label + "_rx" + IO::zfill(m_grid.m_rx, 3) + "_ry" +
        IO::zfill(m_grid.m_ry, 3) + "_" + IO::zfill(iter, 10) + ".dat";
    auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    IO::to_binary(file_name, h_out);
  }
};

int main(int argc, char* argv[]) {
  ::MPI_Init(&argc, &argv);
  int rank, nprocs;
  ::MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  Kokkos::initialize(argc, argv);
  {
    auto kwargs = IO::parse_args(argc, argv);
    std::string out_dir =
        IO::get_arg<std::string>(kwargs, "out_dir", "data_kokkos");
    int px             = IO::get_arg(kwargs, "px", 2);
    int py             = IO::get_arg(kwargs, "py", 1);
    int nx             = IO::get_arg(kwargs, "nx", 32);
    int nbiter         = IO::get_arg(kwargs, "nbiter", 500);
    double lx          = IO::get_arg(kwargs, "lx", 2.0);
    double dt          = IO::get_arg(kwargs, "dt", 0.01);
    double Re          = IO::get_arg(kwargs, "Re", 100.0);
    bool suppress_diag = IO::get_arg(kwargs, "suppress_diag", false);
    const double nu    = 1.0 / Re;

    // Make sure parallelization is valid
    if (px * py != nprocs) {
      throw std::runtime_error(
          "Total number of process must be equal to px * py.\n nprocs: " +
          std::to_string(nprocs) + ", px: " + std::to_string(px) +
          ", py: " + std::to_string(py));
    }

    if (rank == 0) {
      std::cout << "NS3D (px * py): (" << px << ", " << py
                << "), nbiter: " << nbiter << ", nx: " << nx << ", dt: " << dt
                << ", Re: " << Re << std::endl;
    }

    NavierStokes model(rank, px, py, nx, lx, nbiter, dt, nu, out_dir,
                       suppress_diag);
    Kokkos::Timer timer;
    model.run();
    Kokkos::fence();
    double seconds = timer.seconds();
    if (rank == 0) {
      std::cout << "Elapsed time: " << seconds << " [s]" << std::endl;
    }
  }
  Kokkos::finalize();
  ::MPI_Finalize();

  return 0;
}
