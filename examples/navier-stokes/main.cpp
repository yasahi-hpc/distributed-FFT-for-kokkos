#include <vector>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>
#include "math_utils.hpp"
#include "io_utils.hpp"

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL)
constexpr int TILE0 = 1, TILE1 = 4, TILE2 = 32;
#else
constexpr int TILE0 = 4, TILE1 = 4, TILE2 = 4;
#endif
constexpr int DIM = 3;

using execution_space      = Kokkos::DefaultExecutionSpace;
using host_execution_space = Kokkos::DefaultHostExecutionSpace;
template <typename T>
using View1D = Kokkos::View<T*, Kokkos::LayoutRight, execution_space>;
template <typename T>
using View2D = Kokkos::View<T**, Kokkos::LayoutRight, execution_space>;
template <typename T>
using View3D = Kokkos::View<T***, Kokkos::LayoutRight, execution_space>;
template <typename T>
using View4D = Kokkos::View<T****, Kokkos::LayoutRight, execution_space>;
template <typename T>
using View5D = Kokkos::View<T*****, Kokkos::LayoutRight, execution_space>;

using range3D_type = Kokkos::MDRangePolicy<
    execution_space,
    Kokkos::Rank<3, Kokkos::Iterate::Right, Kokkos::Iterate::Right>,
    Kokkos::IndexType<int>>;

// \brief A class to represent the grid used in the Navier-Stokes equation.
struct Grid {
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
  // \param[in] nx Number of grid points in the x-direction.
  // \param[in] ny Number of grid points in the y-direction.
  // \param[in] nz Number of grid points in the z-direction.
  // \param[in] lx Length of the domain in the x-direction.
  // \param[in] ly Length of the domain in the y-direction.
  // \param[in] lz Length of the domain in the z-direction.
  Grid(int nx, int ny, int nz, double lx, double ly, double lz) {
    // Grid and Wavenumbers
    execution_space exec;
    m_x = Math::linspace(exec, 0.0, lx * M_PI, nx, /*endpoint=*/false);
    m_y = Math::linspace(exec, 0.0, ly * M_PI, ny, /*endpoint=*/false);
    m_z = Math::linspace(exec, 0.0, lz * M_PI, nz, /*endpoint=*/false);

    // Wavenumbers
    // int nkx2 = nx * 2 + 1, nky2 = ny * 2 + 1, nkzh = nz + 1;
    int nzh      = nz / 2 + 1;
    m_kx         = View1D<double>("kx", nx);
    m_ky         = View1D<double>("ky", ny);
    m_kzh        = View1D<double>("kzh", nzh);
    m_ksq        = View3D<double>("ksq", nx, ny, nzh);
    m_inv_ksq    = View3D<double>("inv_ksq", nx, ny, nzh);
    m_alias_mask = View3D<double>("alias_mask", nx, ny, nzh);

    host_execution_space host_exec;

    // [0, dkx, 2*dkx, ..., nkx * dkx, -nkx * dkx, ..., -dkx]
    double dkx = lx / static_cast<double>(2 * nx);
    auto h_kx  = KokkosFFT::fftfreq(host_exec, nx, dkx);

    // [0, dky, 2*dky, ..., nky * dky, -nky * dkyx, ..., -dky]
    double dky = ly / static_cast<double>(2 * ny);
    auto h_ky  = KokkosFFT::fftfreq(host_exec, ny, dky);

    // [0, dkz, 2*dkz, ..., nkz * dkz]
    double dkz = lz / static_cast<double>(2 * nz);
    auto h_kzh = KokkosFFT::rfftfreq(host_exec, nz, dkz);

    // kx**2 + ky**2 + kz**2
    auto h_ksq     = Kokkos::create_mirror_view(m_ksq);
    auto h_inv_ksq = Kokkos::create_mirror_view(m_inv_ksq);
    for (int ikz = 0; ikz < nzh; ikz++) {
      for (int iky = 0; iky < ny; iky++) {
        for (int ikx = 0; ikx < nx; ikx++) {
          h_ksq(ikx, iky, ikz) = h_kx(ikx) * h_kx(ikx) +
                                 +h_ky(iky) * h_ky(iky) +
                                 h_kzh(ikz) * h_kzh(ikz);
          h_inv_ksq(ikx, iky, ikz) = (ikx == 0 && iky == 0 && ikz == 0)
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
    for (int ikz = 0; ikz < nzh; ikz++) {
      for (int iky = 0; iky < ny; iky++) {
        for (int ikx = 0; ikx < nx; ikx++) {
          bool cutoff = (Kokkos::abs(h_kx_ind(ikx) * nx) > cutoff_x) ||
                        (Kokkos::abs(h_ky_ind(iky) * ny) > cutoff_y) ||
                        (Kokkos::abs(h_kz_ind(ikz) * nz) > cutoff_z);
          h_alias_mask(ikx, iky, ikz) = static_cast<double>(!cutoff);
        }
      }
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
  static_assert(ViewType::rank() == 4, "dealias: View rank should be 4");
  if constexpr (ViewType::rank() == 4) {
    const int n0 = mask.extent(0), n1 = mask.extent(1), n2 = mask.extent(2);

    range3D_type policy({0, 0, 0}, {n0, n1, n2});
    Kokkos::parallel_for(
        "dealias", policy, KOKKOS_LAMBDA(int i0, int i1, int i2) {
          auto dealias_mask = mask(i0, i1, i2);
          for (int d = 0; d < DIM; d++) {
            view(d, i0, i1, i2) *= dealias_mask;
          }
        });
  }
}

// \brief Projects the velocity field onto the divergence-free space in Fourier
// domain. u^_proj = u^ - k (k \dot u^) / k^2
//
// \tparam ViewType The type of the first field.
// \param[in] uk The first field.
template <typename ViewType>
void projection(const Grid& grid, const ViewType& uk) {
  auto kx      = grid.m_kx;
  auto ky      = grid.m_ky;
  auto kzh     = grid.m_kzh;
  auto inv_ksq = grid.m_inv_ksq;
  const int n0 = uk.extent(1), n1 = uk.extent(2), n2 = uk.extent(3);
  range3D_type policy({0, 0, 0}, {n0, n1, n2});
  Kokkos::parallel_for(
      "projection", policy, KOKKOS_LAMBDA(int i0, int i1, int i2) {
        double k_tmp[DIM] = {kx(i0), ky(i1), kzh(i2)};
        Kokkos::complex<double> div_hat =
            (k_tmp[0] * uk(0, i0, i1, i2) + k_tmp[1] * uk(1, i0, i1, i2) +
             k_tmp[2] * uk(2, i0, i1, i2)) *
            inv_ksq(i0, i1, i2);
        for (int d = 0; d < DIM; d++) {
          uk(d, i0, i1, i2) -= k_tmp[d] * div_hat;
        }
      });
}

// \brief A class to represent the variables used in the Navier-Stokes equations
struct Variables {
  //! Velocity fields in the x, y and z directions.
  View4D<Kokkos::complex<double>> m_uk;

  //! Time derivative of Velocity fields in the x, y and z directions.
  View4D<Kokkos::complex<double>> m_dukdt;

  // \brief Constructor of a Variables class
  // Taylor-Green vortex is used as the initial condition.
  // \param[in] grid Grid in Fourier space
  // \param[in] v0 Initial velocity magnitude. Defaults to 1.0.
  Variables(const Grid& grid, double v0 = 1.0) {
    auto h_x =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grid.m_x);
    auto h_y =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grid.m_y);
    auto h_z =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grid.m_z);

    int nx = h_x.extent(0), ny = h_y.extent(0), nz = h_z.extent(0);
    View4D<double> u("u", DIM, nx, ny, nz);  // (u, v, w)
    m_uk    = View4D<Kokkos::complex<double>>("uk", DIM, nx, ny, nz / 2 + 1);
    m_dukdt = View4D<Kokkos::complex<double>>("dukdt", DIM, nx, ny, nz / 2 + 1);

    auto h_u = Kokkos::create_mirror_view(u);
    for (int iz = 0; iz < nz; iz++) {
      for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
          double x = h_x(ix), y = h_y(iy), z = h_z(iz);
          h_u(0, ix, iy, iz) =
              v0 * Kokkos::sin(x) * Kokkos::cos(y) * Kokkos::cos(z);
          h_u(1, ix, iy, iz) =
              -v0 * Kokkos::cos(x) * Kokkos::sin(y) * Kokkos::cos(z);
          h_u(2, ix, iy, iz) = 0.0;
        }
      }
    }
    Kokkos::deep_copy(u, h_u);

    using execution_space = Kokkos::DefaultExecutionSpace;
    execution_space exec;
    KokkosFFT::rfftn(exec, u, m_uk, KokkosFFT::axis_type<3>{-3, -2, -1});

    dealias(m_uk, grid.m_alias_mask);
    projection(grid, m_uk);
  }
};

// \brief A class to represent the 4th order Runge-Kutta method for solving ODE
// dy/dt = f(t, y) by
// y^{n+1} = y^{n} + (k1 + 2*k2 + 2*k3 + k4)/6
// t^{n+1} = t^{n} + h
// where h is a time step and
// k1 = f(t^{n}      , y^{n}     ) * h
// k2 = f(t^{n} + h/2, y^{n}+k1/2) * h
// k3 = f(t^{n} + h/2, y^{n}+k2/2) * h
// k4 = f(t^{n} + h  , y^{n}+k3  ) * h
//
// \tparam BufferType The type of the view
template <typename BufferType>
class RK4th {
  static_assert(BufferType::rank == 1, "RK4th: BufferType must have rank 1.");
  using value_type = typename BufferType::non_const_value_type;
  using float_type = KokkosFFT::Impl::base_floating_point_type<value_type>;

  //! Order of the Runge-Kutta method
  const int m_order = 4;

  //! Time step size
  const float_type m_h;

  //! Size of the input View after flattening
  std::size_t m_array_size;

  //! Buffer views for intermediate results
  BufferType m_y, m_k1, m_k2, m_k3;

 public:
  // \brief Constructor of a RK4th class
  // \param[in] y The variable to be solved
  // \param[in] h Time step
  RK4th(const BufferType& y, float_type h) : m_h(h) {
    m_array_size = y.size();
    m_y          = BufferType("y", m_array_size);
    m_k1         = BufferType("k1", m_array_size);
    m_k2         = BufferType("k2", m_array_size);
    m_k3         = BufferType("k3", m_array_size);
  }

  auto order() { return m_order; }

  // \brief Advances the solution by one step using the Runge-Kutta method.
  // \tparam ViewType The type of the view
  // \param[in] dydt The right-hand side of the ODE
  // \param[in,out] y The current solution.
  // \param[in] step The current step (0, 1, 2, or 3)
  template <typename ViewType>
  void advance(const ViewType& dydt, const ViewType& y, int step) {
    static_assert(ViewType::rank == 1, "RK4th: ViewType must have rank 1.");
    auto h      = m_h;
    auto y_copy = m_y;
    if (step == 0) {
      auto k1 = m_k1;
      Kokkos::parallel_for(
          "rk_step0",
          Kokkos::RangePolicy<execution_space, Kokkos::IndexType<std::size_t>>(
              execution_space(), 0, m_array_size),
          KOKKOS_LAMBDA(const std::size_t& i) {
            y_copy(i) = y(i);
            k1(i)     = dydt(i) * h;
            y(i)      = y_copy(i) + k1(i) / 2.0;
          });
    } else if (step == 1) {
      auto k2 = m_k2;
      Kokkos::parallel_for(
          "rk_step1",
          Kokkos::RangePolicy<execution_space, Kokkos::IndexType<std::size_t>>(
              execution_space(), 0, m_array_size),
          KOKKOS_LAMBDA(const std::size_t& i) {
            k2(i) = dydt(i) * h;
            y(i)  = y_copy(i) + k2(i) / 2.0;
          });
    } else if (step == 2) {
      auto k3 = m_k3;
      Kokkos::parallel_for(
          "rk_step2",
          Kokkos::RangePolicy<execution_space, Kokkos::IndexType<std::size_t>>(
              execution_space(), 0, m_array_size),
          KOKKOS_LAMBDA(const std::size_t& i) {
            k3(i) = dydt(i) * h;
            y(i)  = y_copy(i) + k3(i);
          });
    } else if (step == 3) {
      auto k1 = m_k1;
      auto k2 = m_k2;
      auto k3 = m_k3;
      Kokkos::parallel_for(
          "rk_step3",
          Kokkos::RangePolicy<execution_space, Kokkos::IndexType<std::size_t>>(
              execution_space(), 0, m_array_size),
          KOKKOS_LAMBDA(const std::size_t& i) {
            auto tmp_dy =
                (k1(i) + 2.0 * k2(i) + 2.0 * k3(i) + dydt(i) * h) / 6.0;
            y(i) = y_copy(i) + tmp_dy;
          });
    } else {
      throw std::runtime_error("step should be 0, 1, 2, or 3");
    }
  }
};

class NavierStokes {
  using OdeSolverType   = RK4th<View1D<Kokkos::complex<double>>>;
  using ForwardPlanType = KokkosFFT::Plan<execution_space, View4D<double>,
                                          View4D<Kokkos::complex<double>>, 3>;
  using BackwardPlanType =
      KokkosFFT::Plan<execution_space, View4D<Kokkos::complex<double>>,
                      View4D<double>, 3>;

  //! The ODE solver used in the simulation
  std::unique_ptr<OdeSolverType> m_ode;

  //! The forward fft plan used in the simulation
  std::unique_ptr<ForwardPlanType> m_forward_plan;

  //! The backward fft plan used in the simulation
  std::unique_ptr<BackwardPlanType> m_backward_plan;

  //! Buffer view to store the time derivative of u
  View4D<double> m_u, m_dudx, m_dudy, m_dudz;

  //! Buffer view to store the spatial derivatives of u
  View4D<Kokkos::complex<double>> m_dukdx, m_dukdy, m_dukdz;

  //! Radial bins of wavenumbers and energy
  View1D<double> m_k_bins, m_k_vals, m_E_spec;

  ///@{
  //! The number of grid points in each direction
  const int m_nx, m_ny, m_nz;
  ///@}

  //! The total number of iterations.
  const int m_nbiter;

  //! The parameters used in the simulation.
  double m_dt = 0.0;

  //! The directory to output diagnostic data.
  std::string m_out_dir;

  //! The grid used in the simulation
  Grid m_grid;

  //! The variables used in the simulation
  Variables m_variables;

  const double m_nu = 0.01;
  double m_coef;
  double m_time = 0.0;
  int m_diag_it = 0, m_diag_steps = 5000;
  int m_num_bins = 50;

 public:
  // \brief Constructor of a HasegawaWakatani class
  // \param[in] nx The number of grid points in each direction.
  // \param[in] lx The length of the domain in each direction.
  // \param[in] nbiter The total number of iterations.
  // \param[in] dt The time step size.
  // \param[in] out_dir The directory to output diagnostic data.
  NavierStokes(int nx, double lx, int nbiter, double dt,
               const std::string& out_dir)
      : m_nx(nx),
        m_ny(nx),
        m_nz(nx),
        m_nbiter(nbiter),
        m_dt(dt),
        m_out_dir(out_dir),
        m_grid(nx, nx, nx, lx, lx, lx),
        m_variables(m_grid) {
    View1D<Kokkos::complex<double>> uk_flatten(m_variables.m_uk.data(),
                                               m_variables.m_uk.size());
    m_ode = std::make_unique<OdeSolverType>(uk_flatten, dt);

    namespace fs = std::filesystem;
    IO::mkdir(m_out_dir, fs::perms::owner_all | fs::perms::group_read |
                             fs::perms::group_exec | fs::perms::others_read |
                             fs::perms::others_exec);

    const int ny = m_grid.m_y.extent(0), nz = m_grid.m_z.extent(0);
    const int nzh = m_grid.m_inv_ksq.extent(2);
    m_u           = View4D<double>("u", DIM, nx, ny, nz);
    m_dudx        = View4D<double>("dudx", DIM, nx, ny, nz);
    m_dudy        = View4D<double>("dudy", DIM, nx, ny, nz);
    m_dudz        = View4D<double>("dudz", DIM, nx, ny, nz);

    m_dukdx = View4D<Kokkos::complex<double>>("dukdx", DIM, nx, ny, nzh);
    m_dukdy = View4D<Kokkos::complex<double>>("dukdy", DIM, nx, ny, nzh);
    m_dukdz = View4D<Kokkos::complex<double>>("dukdz", DIM, nx, ny, nzh);

    // Create FFT plans
    m_forward_plan = std::make_unique<ForwardPlanType>(
        execution_space(), m_dudx, m_dukdx, KokkosFFT::Direction::forward,
        KokkosFFT::axis_type<3>({-3, -2, -1}));
    m_backward_plan = std::make_unique<BackwardPlanType>(
        execution_space(), m_dukdx, m_dudx, KokkosFFT::Direction::backward,
        KokkosFFT::axis_type<3>({-3, -2, -1}));

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

    m_k_bins = View1D<double>("k_bins", m_num_bins + 1);
    m_k_vals = View1D<double>("k_vals", m_num_bins);
    m_E_spec = View1D<double>("E_spec", m_num_bins);

    execution_space exec;
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
    m_coef = Kokkos::pow(lx, 3) / Kokkos::pow(static_cast<double>(nx), 6);
  }

  // \brief Runs the simulation for the specified number of iterations.
  void run() {
    m_time = 0.0;
    for (int iter = 0; iter < m_nbiter; iter++) {
      diag(iter);
      solve();
      m_time += m_dt;
    }
  }

  // \brief Advances the simulation by one time step.
  void solve() {
    for (int step = 0; step < m_ode->order(); step++) {
      rhs(m_variables.m_uk, m_variables.m_dukdt);

      // Flatten Views for time integral
      View1D<Kokkos::complex<double>> uk(m_variables.m_uk.data(),
                                         m_variables.m_uk.size()),
          dukdt(m_variables.m_dukdt.data(), m_variables.m_dukdt.size());
      m_ode->advance(dukdt, uk, step);
    }
  }

  // \brief Computes the RHS of Navier-Stokes equation
  //
  // \tparam ViewType The type of the velocity field.
  // \param[in] uk The velocity field.
  // \param[out] dukdt The RHS of the vorticity equation.
  template <typename ViewType>
  void rhs(const ViewType& uk, const ViewType& dukdt) {
    // Apply dealiasing filter before inverse transform
    dealias(uk, m_grid.m_alias_mask);

    // Calculate velocity derivatives in Fourier space
    derivative(uk, m_dukdx, m_dukdy, m_dukdz);

    // Inverse FFT to get velocity in real space
    KokkosFFT::execute(*m_backward_plan, uk, m_u);

    // Inverse FFT to get derivatives in real space
    KokkosFFT::execute(*m_backward_plan, m_dukdx, m_dudx);
    KokkosFFT::execute(*m_backward_plan, m_dukdy, m_dudy);
    KokkosFFT::execute(*m_backward_plan, m_dukdz, m_dudz);

    // Calculate nonlinear advection terms in real space
    advection(m_u, m_dudx, m_dudy, m_dudz);

    // FFT of nonlinear terms (stored in dukdt)
    KokkosFFT::execute(*m_forward_plan, m_u, dukdt);

    dealias(dukdt, m_grid.m_alias_mask);

    // Calculate viscous diffusion term in Fourier space
    // And combine terms (stored in dukdt)
    combine(dukdt, uk);

    // Project the RHS onto the divergence-free space
    // [NOTE] This is an external function, which is
    // unit-tested
    projection(m_grid, dukdt);
  }

  // \brief Computes the derivative in Fourier space
  //
  // \tparam ViewType The type of the first field.
  // \param[in] uk The first field.
  template <typename ViewType>
  void derivative(const ViewType& uk, const ViewType& dukdx,
                  const ViewType& dukdy, const ViewType& dukdz) {
    auto kx  = m_grid.m_kx;
    auto ky  = m_grid.m_ky;
    auto kzh = m_grid.m_kzh;

    const Kokkos::complex<double> z(0.0, 1.0);  // Imaginary unit
    const int n0 = uk.extent(1), n1 = uk.extent(2), n2 = uk.extent(3);

    range3D_type policy({0, 0, 0}, {n0, n1, n2});
    Kokkos::parallel_for(
        "derivative", policy, KOKKOS_LAMBDA(int i0, int i1, int i2) {
          Kokkos::complex<double> ikx = z * kx(i0), iky = z * ky(i1),
                                  ikz = z * kzh(i2);
          for (int d = 0; d < DIM; d++) {
            dukdx(d, i0, i1, i2) = ikx * uk(d, i0, i1, i2);
            dukdy(d, i0, i1, i2) = iky * uk(d, i0, i1, i2);
            dukdz(d, i0, i1, i2) = ikz * uk(d, i0, i1, i2);
          }
        });
  }

  // \brief Computes the nonlinear advection term in real space
  // u.grad(u) = u * du/dx + v * dv/dx + w * dw/dx
  //
  // \tparam ViewType The type of the velocity field.
  // \param[in,out] u On entrance, the velocity field. On exit, the
  // nonlinear advection term
  // \param[in] dudx The x derivative of velocity field.
  // \param[in] dudy The y derivative of velocity field.
  // \param[in] dudz The z derivative of velocity field.
  template <typename ViewType>
  void advection(const ViewType& u, const ViewType& dudx, const ViewType& dudy,
                 const ViewType& dudz) {
    const int n0 = u.extent(1), n1 = u.extent(2), n2 = u.extent(3);

    range3D_type policy({0, 0, 0}, {n0, n1, n2});
    Kokkos::parallel_for(
        "advection", policy, KOKKOS_LAMBDA(int i0, int i1, int i2) {
          double u_tmp = u(0, i0, i1, i2), v_tmp = u(1, i0, i1, i2),
                 w_tmp = u(2, i0, i1, i2);
          for (int d = 0; d < DIM; d++) {
            u(d, i0, i1, i2) = u_tmp * dudx(d, i0, i1, i2) +
                               v_tmp * dudy(d, i0, i1, i2) +
                               w_tmp * dudz(d, i0, i1, i2);
          }
        });
  }

  // \brief Computes the RHS excluding pressure in Fourier space
  // - advection_term (-u.grad(u))_hat + viscosity term (-nu * k^2 * u_hat)
  //
  // \tparam ViewType The type of the velocity field.
  // \param[in,out] dukdt On entrance, the advection term. On exit, the
  // the RHS excluding pressure
  // \param[in] uk The velocity field.
  template <typename ViewType>
  void combine(const ViewType& dukdt, const ViewType& uk) {
    const int n0 = uk.extent(1), n1 = uk.extent(2), n2 = uk.extent(3);
    const double nu = m_nu;
    auto ksq        = m_grid.m_ksq;

    range3D_type policy({0, 0, 0}, {n0, n1, n2});
    Kokkos::parallel_for(
        "rhs", policy, KOKKOS_LAMBDA(int i0, int i1, int i2) {
          double visc = -nu * ksq(i0, i1, i2);
          for (int d = 0; d < DIM; d++) {
            dukdt(d, i0, i1, i2) =
                -dukdt(d, i0, i1, i2) + visc * uk(d, i0, i1, i2);
          }
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
    KokkosFFT::execute(*m_backward_plan, m_variables.m_uk, m_u);

    auto u = Kokkos::subview(m_u, 0, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    auto v = Kokkos::subview(m_u, 1, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    auto w = Kokkos::subview(m_u, 2, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    to_binary_file("u", u, iter);
    to_binary_file("v", v, iter);
    to_binary_file("w", w, iter);
  }

  // \brief Saves the kinetic energy
  // \param[in] iter The current iteration number.
  void diag_energy(const int iter) {
    // Cleanup bins
    Kokkos::deep_copy(m_E_spec, 0.0);
    auto uk      = m_variables.m_uk;
    auto ksq     = m_grid.m_ksq;
    auto E_spec  = m_E_spec;
    auto k_bins  = m_k_bins;
    const int n0 = uk.extent(1), n1 = uk.extent(2), n2 = uk.extent(3);
    const int num_bins = m_E_spec.extent(0);

    range3D_type policy({0, 0, 0}, {n0, n1, n2});
    Kokkos::parallel_for(
        "spectral-density", policy, KOKKOS_LAMBDA(int i0, int i1, int i2) {
          auto k_mag = Kokkos::sqrt(ksq(i0, i1, i2));
          auto u_tmp = uk(0, i0, i1, i2), v_tmp = uk(1, i0, i1, i2),
               w_tmp = uk(2, i0, i1, i2);
          double factor =
              i2 == 0 ? 1.0
                      : 2.0;  // account for Hermitian symmetry in z direction
          auto Ek = 0.5 * factor *
                    (Kokkos::abs(u_tmp * u_tmp) + Kokkos::abs(v_tmp * v_tmp) +
                     Kokkos::abs(w_tmp * w_tmp));

          for (int i = 0; i < num_bins; i++) {
            bool mask = (k_mag >= k_bins(i)) && (k_mag < k_bins(i + 1));
            if (mask) {
              Kokkos::atomic_add(&E_spec(i), Ek);
            }
          }
        });

    double coef   = m_coef;
    double energy = 0.0;
    Kokkos::parallel_reduce(
        "total-energy", policy,
        KOKKOS_LAMBDA(int i0, int i1, int i2, double& l_energy) {
          auto u_tmp = uk(0, i0, i1, i2), v_tmp = uk(1, i0, i1, i2),
               w_tmp = uk(2, i0, i1, i2);
          double factor =
              i2 == 0 ? 1.0
                      : 2.0;  // account for Hermitian symmetry in z direction
          auto Ek = 0.5 * factor * coef *
                    (Kokkos::abs(u_tmp * u_tmp) + Kokkos::abs(v_tmp * v_tmp) +
                     Kokkos::abs(w_tmp * w_tmp));
          l_energy += Ek;
        },
        energy);

    std::cout << "Step: " << iter * m_diag_steps << "/" << m_nbiter << ", Time: " << m_time
              << ", Kinetic Energy: " << energy << std::endl;

    to_binary_file("E_spec", E_spec, iter);
    to_binary_file("k_vals", m_k_vals, iter);
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
        m_out_dir + "/" + label + "_" + IO::zfill(iter, 10) + ".dat";
    auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    IO::to_binary(file_name, h_out);
  }
};

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  auto kwargs         = IO::parse_args(argc, argv);
  std::string out_dir = IO::get_arg(kwargs, "out_dir", "data_kokkos");
  int nx = 32, nbiter = 500;
  double lx = 2.0, dt = 0.01;
  NavierStokes model(nx, lx, nbiter, dt, out_dir);
  Kokkos::Timer timer;
  model.run();
  Kokkos::fence();
  double seconds = timer.seconds();
  std::cout << "Elapsed time: " << seconds << " [s]" << std::endl;

  return 0;
}
