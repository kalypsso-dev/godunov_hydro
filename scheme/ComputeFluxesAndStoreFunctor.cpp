// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeFluxesAndStoreFunctor.cpp
 */
#include <godunov_hydro/scheme/ComputeFluxesAndStoreFunctor.h>

namespace kalypsso
{

namespace godunov_hydro
{

/*************************************************/
/*************************************************/
/*************************************************/
template <size_t dim, typename device_t>
ComputeFluxesAndStoreFunctor<dim, device_t>::ComputeFluxesAndStoreFunctor(
  orchard_key_view_t                orchard_keys,
  AMRMeshInfo                       amr_mesh_info,
  DataArrayBlock_t                  fluxes,
  DataArrayGhostedBlock_t           q_ghosted,
  DataArrayGhostedBlock_t           slopes_x,
  DataArrayGhostedBlock_t           slopes_y,
  DataArrayGhostedBlock_t           slopes_z,
  FieldMap<core::models::Hydro>     fm,
  int32_t                           iOct_flux_offset,
  int32_t                           num_quads,
  int                               direction,
  HydroSettings const &             hydro_settings,
  eos::EosWrapper<device_t> const & eos,
  real_t                            dt,
  real_t                            scaling_factor,
  bool                              gravity_enabled,
  UniformGravityField<dim>          gravity_field,
  ViscosityParams                   viscosity,
  TimeIntegrator                    time_integrator)
  : m_orchard_keys_device(orchard_keys)
  , m_amr_mesh_info(amr_mesh_info)
  , m_Fluxes(fluxes)
  , m_q(q_ghosted)
  , m_slopes_x(slopes_x)
  , m_slopes_y(slopes_y)
  , m_slopes_z(slopes_z)
  , m_fm(fm)
  , m_iOct_flux_offset(iOct_flux_offset)
  , m_num_quads(num_quads)
  , m_direction(direction)
  , m_block_sizes(slopes_x.block_size())
  , m_nbCellsPerLeaf(Kokkos::dim_prod(m_block_sizes))
  , m_hydro_settings(hydro_settings)
  , m_eos(eos)
  , m_dt(dt)
  , m_scaling_factor(scaling_factor)
  , m_gravity_enabled(gravity_enabled)
  , m_gravity_field(gravity_field)
  , m_viscosity(viscosity)
  , m_time_integrator(time_integrator)
{} // constructor

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
ComputeFluxesAndStoreFunctor<dim, device_t>::apply(ConfigMap const &             config_map,
                                                   orchard_key_view_t            orchard_keys,
                                                   AMRMeshInfo                   amr_mesh_info,
                                                   DataArrayBlock_t              fluxes,
                                                   DataArrayGhostedBlock_t       q_ghosted,
                                                   DataArrayGhostedBlock_t       slopes_x,
                                                   DataArrayGhostedBlock_t       slopes_y,
                                                   DataArrayGhostedBlock_t       slopes_z,
                                                   FieldMap<core::models::Hydro> fm,
                                                   int32_t                       iOct_flux_offset,
                                                   int32_t                       num_quads,
                                                   int                           direction,
                                                   HydroSettings const &         hydro_settings,
                                                   eos::EosWrapper<device_t> const & eos,
                                                   ViscosityParams                   viscosity,
                                                   real_t                            dt)
{
  // Important note: the caller is responsible for provide a flux array with right shape.
  {
    [[maybe_unused]] auto flux_block_sizes = q_ghosted.block_size();
    flux_block_sizes[static_cast<size_t>(direction)]++;
    assertm(flux_block_sizes == fluxes.shape(), "Flux array has incompatible shape.");
  }

  const auto gravity_enabled = config_map.getBool("gravity", "enabled", false);
  const auto gravity_field = get_uniform_gravity_vector<dim>(config_map);

  ComputeFluxesAndStoreFunctor<dim, device_t> functor(
    orchard_keys,
    amr_mesh_info,
    fluxes,
    q_ghosted,
    slopes_x,
    slopes_y,
    slopes_z,
    fm,
    iOct_flux_offset,
    num_quads,
    direction,
    hydro_settings,
    eos,
    dt,
    get_scaling_factor(config_map),
    gravity_enabled,
    gravity_field,
    viscosity,
    TimeIntegratorConfig::get_time_integrator(config_map));

  const auto nbIterations = num_quads * fluxes.num_cells();

#ifdef KOKKOS_ENABLE_CUDA
  using Property = Kokkos::Experimental::WorkItemProperty::HintHeavyWeight_t;
#else
  using Property = Kokkos::Experimental::WorkItemProperty::None_t;
#endif

  // launch computation
  Kokkos::parallel_for("kalypsso::godunov_hydro::ComputeFluxesAndStoreFunctor",
                       Kokkos::RangePolicy<exec_space, Property>(0, nbIterations),
                       functor);
} // apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION auto
ComputeFluxesAndStoreFunctor<dim, device_t>::reconstruct_state_2d(const HydroState<2> & q,
                                                                  int32_t               is,
                                                                  int32_t               js,
                                                                  int32_t               iOct_local,
                                                                  const offsets_t<2> &  offsets,
                                                                  real_t                dtdx,
                                                                  real_t                dtdy,
                                                                  real_t                dx,
                                                                  real_t                dy) const
{
  auto const & smallr = m_hydro_settings.smallr;
  auto const & smallp = m_hydro_settings.smallp;

  // retrieve primitive variables in current quadrant
  auto const & r = q[Hydro::ID];
  auto const & p = q[Hydro::IP];
  auto const & u = q[Hydro::IU];
  auto const & v = q[Hydro::IV];
  // auto const w = 0.0;

  const auto c = m_eos.sound_speed(p, r);

  auto const drx = m_slopes_x(is, js, m_fm[Hydro::ID], iOct_local);
  auto const dpx = m_slopes_x(is, js, m_fm[Hydro::IP], iOct_local);
  auto const dux = m_slopes_x(is, js, m_fm[Hydro::IU], iOct_local);
  auto const dvx = m_slopes_x(is, js, m_fm[Hydro::IV], iOct_local);
  // auto const dwx = 0.0;

  auto const dry = m_slopes_y(is, js, m_fm[Hydro::ID], iOct_local);
  auto const dpy = m_slopes_y(is, js, m_fm[Hydro::IP], iOct_local);
  auto const duy = m_slopes_y(is, js, m_fm[Hydro::IU], iOct_local);
  auto const dvy = m_slopes_y(is, js, m_fm[Hydro::IV], iOct_local);
  // auto const dwy = 0.0;

  // reconstruct state on interface
  HydroState<2> qr;

  if (m_time_integrator == +TimeIntegrator::HANCOCK)
  {
    // MUSCL-Hancock half time-step
    // clang-format off
    auto const sr0 = (-u * drx - dux * r) * dtdx + (-v * dry - dvy * r) * dtdy;
    auto const su0 = (-u * dux - dpx / r) * dtdx + (-v * duy          ) * dtdy;
    auto const sv0 = (-u * dvx          ) * dtdx + (-v * dvy - dpy / r) * dtdy;
    auto const sp0 = (-u * dpx - dux * r * c * c) * dtdx +
                     (-v * dpy - dvy * r * c * c) * dtdy;
    // clang-format on

    qr[Hydro::ID] = r + HALF_F * sr0 + offsets[IX] * drx + offsets[IY] * dry;
    qr[Hydro::IP] = p + HALF_F * sp0 + offsets[IX] * dpx + offsets[IY] * dpy;
    qr[Hydro::IU] = u + HALF_F * su0 + offsets[IX] * dux + offsets[IY] * duy;
    qr[Hydro::IV] = v + HALF_F * sv0 + offsets[IX] * dvx + offsets[IY] * dvy;

    // add gravity predictor step
    if (m_gravity_enabled)
    {
      qr[Hydro::IU] += m_gravity_field[IX] * HALF_F * m_dt;
      qr[Hydro::IV] += m_gravity_field[IY] * HALF_F * m_dt;
    }

    // add viscous force predictor
    if (m_viscosity.enabled and m_viscosity.hancock_predictor_enabled)
    {
      add_viscous_predictor_2d(qr, is, js, iOct_local, dtdx, dtdy, dx, dy);
    }
  }
  else
  {
    qr[Hydro::ID] = r + offsets[IX] * drx + offsets[IY] * dry;
    qr[Hydro::IP] = p + offsets[IX] * dpx + offsets[IY] * dpy;
    qr[Hydro::IU] = u + offsets[IX] * dux + offsets[IY] * duy;
    qr[Hydro::IV] = v + offsets[IX] * dvx + offsets[IY] * dvy;
  }

  qr[Hydro::ID] = fmax(smallr, qr[Hydro::ID]);
  qr[Hydro::IP] = fmax(smallp * qr[Hydro::ID], qr[Hydro::IP]);

  return qr;

} // reconstruct_state_2d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION auto
ComputeFluxesAndStoreFunctor<dim, device_t>::reconstruct_state_3d(const HydroState<3> & q,
                                                                  int32_t               is,
                                                                  int32_t               js,
                                                                  int32_t               ks,
                                                                  int32_t               iOct_local,
                                                                  const offsets_t<3> &  offsets,
                                                                  real_t                dtdx,
                                                                  real_t                dtdy,
                                                                  real_t                dtdz,
                                                                  real_t                dx,
                                                                  real_t                dy,
                                                                  real_t                dz) const
{
  auto const & smallr = m_hydro_settings.smallr;
  auto const & smallp = m_hydro_settings.smallp;

  // retrieve primitive variables in current quadrant
  const real_t & r = q[Hydro::ID];
  const real_t & p = q[Hydro::IP];
  const real_t & u = q[Hydro::IU];
  const real_t & v = q[Hydro::IV];
  const real_t & w = q[Hydro::IW];

  const auto c = m_eos.sound_speed(p, r);

  // retrieve variations = dx * slopes
  const real_t drx = m_slopes_x(is, js, ks, m_fm[Hydro::ID], iOct_local);
  const real_t dpx = m_slopes_x(is, js, ks, m_fm[Hydro::IP], iOct_local);
  const real_t dux = m_slopes_x(is, js, ks, m_fm[Hydro::IU], iOct_local);
  const real_t dvx = m_slopes_x(is, js, ks, m_fm[Hydro::IV], iOct_local);
  const real_t dwx = m_slopes_x(is, js, ks, m_fm[Hydro::IW], iOct_local);

  const real_t dry = m_slopes_y(is, js, ks, m_fm[Hydro::ID], iOct_local);
  const real_t dpy = m_slopes_y(is, js, ks, m_fm[Hydro::IP], iOct_local);
  const real_t duy = m_slopes_y(is, js, ks, m_fm[Hydro::IU], iOct_local);
  const real_t dvy = m_slopes_y(is, js, ks, m_fm[Hydro::IV], iOct_local);
  const real_t dwy = m_slopes_y(is, js, ks, m_fm[Hydro::IW], iOct_local);

  const real_t drz = m_slopes_z(is, js, ks, m_fm[Hydro::ID], iOct_local);
  const real_t dpz = m_slopes_z(is, js, ks, m_fm[Hydro::IP], iOct_local);
  const real_t duz = m_slopes_z(is, js, ks, m_fm[Hydro::IU], iOct_local);
  const real_t dvz = m_slopes_z(is, js, ks, m_fm[Hydro::IV], iOct_local);
  const real_t dwz = m_slopes_z(is, js, ks, m_fm[Hydro::IW], iOct_local);

  // reconstruct state on interface
  HydroState<3> qr;

  if (m_time_integrator == +TimeIntegrator::HANCOCK)
  {
    // MUSCL-Hancock half time-step
    // clang-format off
    const real_t sr0 = (-u * drx - dux * r) * dtdx +
                       (-v * dry - dvy * r) * dtdy +
                       (-w * drz - dwz * r) * dtdz;
    const real_t su0 = (-u * dux - dpx / r) * dtdx +
                       (-v * duy          ) * dtdy +
                       (-w * duz          ) * dtdz;
    const real_t sv0 = (-u * dvx          ) * dtdx +
                       (-v * dvy - dpy / r) * dtdy +
                       (-w * dvz          ) * dtdz;
    const real_t sw0 = (-u * dwx          ) * dtdx +
                       (-v * dwy          ) * dtdy +
                       (-w * dwz - dpz / r) * dtdz;
    const real_t sp0 = (-u * dpx - dux * r * c * c) * dtdx +
                       (-v * dpy - dvy * r * c * c) * dtdy +
                       (-w * dpz - dwz * r * c * c) * dtdz;
    // clang-format on

    qr[Hydro::ID] = r + HALF_F * sr0 + offsets[IX] * drx + offsets[IY] * dry + offsets[IZ] * drz;
    qr[Hydro::IP] = p + HALF_F * sp0 + offsets[IX] * dpx + offsets[IY] * dpy + offsets[IZ] * dpz;
    qr[Hydro::IU] = u + HALF_F * su0 + offsets[IX] * dux + offsets[IY] * duy + offsets[IZ] * duz;
    qr[Hydro::IV] = v + HALF_F * sv0 + offsets[IX] * dvx + offsets[IY] * dvy + offsets[IZ] * dvz;
    qr[Hydro::IW] = w + HALF_F * sw0 + offsets[IX] * dwx + offsets[IY] * dwy + offsets[IZ] * dwz;

    // add gravity predictor step
    if (m_gravity_enabled)
    {
      qr[Hydro::IU] += m_gravity_field[IX] * HALF_F * m_dt;
      qr[Hydro::IV] += m_gravity_field[IY] * HALF_F * m_dt;
      qr[Hydro::IW] += m_gravity_field[IZ] * HALF_F * m_dt;
    }

    // add viscous force predictor
    if (m_viscosity.enabled and m_viscosity.hancock_predictor_enabled)
    {
      add_viscous_predictor_3d(qr, is, js, ks, iOct_local, dtdx, dtdy, dtdz, dx, dy, dz);
    }
  }
  else
  {
    qr[Hydro::ID] = r + offsets[IX] * drx + offsets[IY] * dry + offsets[IZ] * drz;
    qr[Hydro::IP] = p + offsets[IX] * dpx + offsets[IY] * dpy + offsets[IZ] * dpz;
    qr[Hydro::IU] = u + offsets[IX] * dux + offsets[IY] * duy + offsets[IZ] * duz;
    qr[Hydro::IV] = v + offsets[IX] * dvx + offsets[IY] * dvy + offsets[IZ] * dvz;
    qr[Hydro::IW] = w + offsets[IX] * dwx + offsets[IY] * dwy + offsets[IZ] * dwz;
  }

  qr[Hydro::ID] = fmax(smallr, qr[Hydro::ID]);
  qr[Hydro::IP] = fmax(smallp * qr[Hydro::ID], qr[Hydro::IP]);

  return qr;

} // reconstruct_state_3d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION void
ComputeFluxesAndStoreFunctor<dim, device_t>::add_viscous_predictor_2d(HydroState<2> & qr,
                                                                      int32_t         is,
                                                                      int32_t         js,
                                                                      int32_t         iOct_local,
                                                                      real_t          dtdx,
                                                                      real_t          dtdy,
                                                                      real_t          dx,
                                                                      real_t          dy) const
{
  // kinematic viscosity
  const auto nu = m_viscosity.mu / m_q(is, js, m_fm[Hydro::ID], iOct_local);

  // IU
  const auto d2udx2 = m_q(is + 1, js, m_fm[Hydro::IU], iOct_local) +
                      m_q(is - 1, js, m_fm[Hydro::IU], iOct_local) -
                      2 * m_q(is, js, m_fm[Hydro::IU], iOct_local);
  qr[Hydro::IU] += HALF_F * nu * d2udx2 / dx * dtdx;

  const auto d2udy2 = m_q(is, js + 1, m_fm[Hydro::IU], iOct_local) +
                      m_q(is, js - 1, m_fm[Hydro::IU], iOct_local) -
                      2 * m_q(is, js, m_fm[Hydro::IU], iOct_local);
  qr[Hydro::IU] += HALF_F * nu * d2udy2 / dy * dtdy;

  // IV
  const auto d2vdx2 = m_q(is + 1, js, m_fm[Hydro::IV], iOct_local) +
                      m_q(is - 1, js, m_fm[Hydro::IV], iOct_local) -
                      2 * m_q(is, js, m_fm[Hydro::IV], iOct_local);
  qr[Hydro::IV] += HALF_F * nu * d2vdx2 / dx * dtdx;

  const auto d2vdy2 = m_q(is, js + 1, m_fm[Hydro::IV], iOct_local) +
                      m_q(is, js - 1, m_fm[Hydro::IV], iOct_local) -
                      2 * m_q(is, js, m_fm[Hydro::IV], iOct_local);
  qr[Hydro::IV] += HALF_F * nu * d2vdy2 / dy * dtdy;
} // add_viscous_predictor_2d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION void
ComputeFluxesAndStoreFunctor<dim, device_t>::add_viscous_predictor_3d(HydroState<3> & qr,
                                                                      int32_t         is,
                                                                      int32_t         js,
                                                                      int32_t         ks,
                                                                      int32_t         iOct_local,
                                                                      real_t          dtdx,
                                                                      real_t          dtdy,
                                                                      real_t          dtdz,
                                                                      real_t          dx,
                                                                      real_t          dy,
                                                                      real_t          dz) const
{
  // kinematic viscosity
  const auto nu = m_viscosity.mu / m_q(is, js, ks, m_fm[Hydro::ID], iOct_local);

  // IU
  const auto d2udx2 = m_q(is + 1, js, ks, m_fm[Hydro::IU], iOct_local) +
                      m_q(is - 1, js, ks, m_fm[Hydro::IU], iOct_local) -
                      2 * m_q(is, js, ks, m_fm[Hydro::IU], iOct_local);
  qr[Hydro::IU] += HALF_F * nu * d2udx2 / dx * dtdx;

  const auto d2udy2 = m_q(is, js + 1, ks, m_fm[Hydro::IU], iOct_local) +
                      m_q(is, js - 1, ks, m_fm[Hydro::IU], iOct_local) -
                      2 * m_q(is, js, ks, m_fm[Hydro::IU], iOct_local);
  qr[Hydro::IU] += HALF_F * nu * d2udy2 / dy * dtdy;

  const auto d2udz2 = m_q(is, js, ks + 1, m_fm[Hydro::IU], iOct_local) +
                      m_q(is, js, ks - 1, m_fm[Hydro::IU], iOct_local) -
                      2 * m_q(is, js, ks, m_fm[Hydro::IU], iOct_local);
  qr[Hydro::IU] += HALF_F * nu * d2udz2 / dz * dtdz;

  // IV
  const auto d2vdx2 = m_q(is + 1, js, ks, m_fm[Hydro::IV], iOct_local) +
                      m_q(is - 1, js, ks, m_fm[Hydro::IV], iOct_local) -
                      2 * m_q(is, js, ks, m_fm[Hydro::IV], iOct_local);
  qr[Hydro::IV] += HALF_F * nu * d2vdx2 / dx * dtdx;

  const auto d2vdy2 = m_q(is, js + 1, ks, m_fm[Hydro::IV], iOct_local) +
                      m_q(is, js - 1, ks, m_fm[Hydro::IV], iOct_local) -
                      2 * m_q(is, js, ks, m_fm[Hydro::IV], iOct_local);
  qr[Hydro::IV] += HALF_F * nu * d2vdy2 / dy * dtdy;

  const auto d2vdz2 = m_q(is, js, ks + 1, m_fm[Hydro::IV], iOct_local) +
                      m_q(is, js, ks - 1, m_fm[Hydro::IV], iOct_local) -
                      2 * m_q(is, js, ks, m_fm[Hydro::IV], iOct_local);
  qr[Hydro::IV] += HALF_F * nu * d2vdz2 / dz * dtdz;

  // IW
  const auto d2wdx2 = m_q(is + 1, js, ks, m_fm[Hydro::IW], iOct_local) +
                      m_q(is - 1, js, ks, m_fm[Hydro::IW], iOct_local) -
                      2 * m_q(is, js, ks, m_fm[Hydro::IW], iOct_local);
  qr[Hydro::IW] += HALF_F * nu * d2wdx2 / dx * dtdx;

  const auto d2wdy2 = m_q(is, js + 1, ks, m_fm[Hydro::IW], iOct_local) +
                      m_q(is, js - 1, ks, m_fm[Hydro::IW], iOct_local) -
                      2 * m_q(is, js, ks, m_fm[Hydro::IW], iOct_local);
  qr[Hydro::IW] += HALF_F * nu * d2wdy2 / dy * dtdy;

  const auto d2wdz2 = m_q(is, js, ks + 1, m_fm[Hydro::IW], iOct_local) +
                      m_q(is, js, ks - 1, m_fm[Hydro::IW], iOct_local) -
                      2 * m_q(is, js, ks, m_fm[Hydro::IW], iOct_local);
  qr[Hydro::IW] += HALF_F * nu * d2wdz2 / dz * dtdz;
} // add_viscous_predictor_3d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION void
ComputeFluxesAndStoreFunctor<dim, device_t>::compute_fluxes_and_store_2d(
  int32_t const & cell_index,
  int32_t const & iOct_local) const
{

  auto const coords = cell_index_unravel<2>(cell_index, m_Fluxes.shape());

  auto const & i = coords[IX];
  auto const & j = coords[IY];

  // index to address slope arrays
  auto const & is = i;
  auto const & js = j;

  // index to address primitive variables arrays
  auto const & iq = i;
  auto const & jq = j;

  // get AMR level
  auto const iOct_global = iOct_local + m_iOct_flux_offset;
  auto const level = orchard_key_t<2>::level(m_orchard_keys_device(iOct_global));

  // compute dS over dV in current cell and (larger) neighbor
  // a small cell will always update a large neighbor cell
  // Note: a larger neighbor has a volume 4 times larger than current cell volume
  auto const dx = compute_cell_length<2>(level, m_block_sizes[IX]) * m_scaling_factor;

  auto const dtdS_over_dV_cur = m_dt / dx;

  /*
   * reconstruct states on cells face and update
   */

  // get current location primitive variables state
  // note: primitive variables is a ghosted array with ghost width of 2
  auto qprim = get_prim_variables(iq, jq, iOct_local);

  /*
   * compute flux from left face along X dir and update both sides
   */
  if (m_direction == IX)
  {
    // get state in neighbor along X
    auto qprim_n = get_prim_variables(iq - 1, jq, iOct_local);

    // step 1 : reconstruct state in the left neighbor
    offsets_t<2> offsets{ 0.5, 0.0 };

    // reconstruct state in left neighbor (index relative to slopes array)
    auto qL = reconstruct_state_2d(
      qprim_n, is - 1, js, iOct_local, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur, dx, dx);

    // step 2 : reconstruct state in current cell
    offsets = { -0.5, 0.0 };

    // reconstruct state from current cell center to left interface
    auto qR = reconstruct_state_2d(
      qprim, is, js, iOct_local, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur, dx, dx);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_hydro<2>(qL, qR, m_hydro_settings, m_eos);

    // step 4 : accumulate flux in current cell
    const auto flux_cur = flux * dtdS_over_dV_cur;

    set_flux(i, j, iOct_local, flux_cur);
  }

  /*
   * compute flux from left face along Y dir and update both sides
   */
  if (m_direction == IY)
  {
    // get state in neighbor along Y
    auto qprim_n = get_prim_variables(iq, jq - 1, iOct_local);

    // step 1 : reconstruct state in the left neighbor
    offsets_t<2> offsets = { 0.0, 0.5 };

    // reconstruct "left" state
    auto qL = reconstruct_state_2d(
      qprim_n, is, js - 1, iOct_local, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur, dx, dx);

    // step 2 : reconstruct state in current cell
    offsets = { 0.0, -0.5 };

    // reconstruct state from current cell center to left interface
    auto qR = reconstruct_state_2d(
      qprim, is, js, iOct_local, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur, dx, dx);

    // swap IU / IV
    my_swap(qL[Hydro::IU], qL[Hydro::IV]);
    my_swap(qR[Hydro::IU], qR[Hydro::IV]);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_hydro<2>(qL, qR, m_hydro_settings, m_eos);

    my_swap(flux[Hydro::IU], flux[Hydro::IV]);

    // step 4 : accumulate flux in current cell
    const auto flux_cur = flux * dtdS_over_dV_cur;

    set_flux(i, j, iOct_local, flux_cur);
  }

} // compute_fluxes_and_store_2d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION void
ComputeFluxesAndStoreFunctor<dim, device_t>::compute_fluxes_and_store_3d(
  const int32_t & cell_index,
  const int32_t & iOct_local) const
{
  auto const coords = cell_index_unravel<3>(cell_index, m_Fluxes.shape());

  auto const & i = coords[IX];
  auto const & j = coords[IY];
  auto const & k = coords[IZ];

  // index to address slope arrays
  auto const & is = i;
  auto const & js = j;
  auto const & ks = k;

  // index to address primitive variables arrays
  auto const & iq = i;
  auto const & jq = j;
  auto const & kq = k;

  // get AMR level
  auto const iOct_global = iOct_local + m_iOct_flux_offset;
  auto const level = orchard_key_t<3>::level(m_orchard_keys_device(iOct_global));

  // compute dS over dV in current cell and (larger) neighbor
  // a small cell will always update a large neighbor cell
  // Note: a larger neighbor has a volume 8 times larger than current cell volume
  auto const dx = compute_cell_length<3>(level, m_block_sizes[IX]) * m_scaling_factor;

  auto const dtdS_over_dV_cur = m_dt / dx;

  /*
   * reconstruct states on cells face and update
   */

  // get current location primitive variables state
  // note: primitive variables is a ghosted array with ghost width of 2
  auto qprim = get_prim_variables(iq, jq, kq, iOct_local);

  /*
   * compute flux from left face along X dir
   */
  if (m_direction == IX)
  {
    // get state in neighbor along X
    auto qprim_n = get_prim_variables(iq - 1, jq, kq, iOct_local);

    // step 1 : reconstruct state in the left neighbor
    offsets_t<3> offsets{ 0.5, 0.0, 0.0 };

    // reconstruct state in left neighbor (index relative to slopes array)
    auto qL = reconstruct_state_3d(qprim_n,
                                   is - 1,
                                   js,
                                   ks,
                                   iOct_local,
                                   offsets,
                                   dtdS_over_dV_cur,
                                   dtdS_over_dV_cur,
                                   dtdS_over_dV_cur,
                                   dx,
                                   dx,
                                   dx);

    // step 2 : reconstruct state in current cell
    offsets = { -0.5, 0.0, 0.0 };

    // reconstruct state from current cell center to left interface
    auto qR = reconstruct_state_3d(qprim,
                                   is,
                                   js,
                                   ks,
                                   iOct_local,
                                   offsets,
                                   dtdS_over_dV_cur,
                                   dtdS_over_dV_cur,
                                   dtdS_over_dV_cur,
                                   dx,
                                   dx,
                                   dx);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_hydro<3>(qL, qR, m_hydro_settings, m_eos);

    // step 4 : accumulate flux in current cell
    const auto flux_cur = flux * dtdS_over_dV_cur;

    set_flux(i, j, k, iOct_local, flux_cur);
  } // end update along X

  /*
   * compute flux from left face along Y dir
   */
  if (m_direction == IY)
  {
    // get state in neighbor along Y
    auto qprim_n = get_prim_variables(iq, jq - 1, kq, iOct_local);

    // step 1 : reconstruct state in the left neighbor
    offsets_t<3> offsets = { 0.0, 0.5, 0.0 };

    // reconstruct "left" state
    auto qL = reconstruct_state_3d(qprim_n,
                                   is,
                                   js - 1,
                                   ks,
                                   iOct_local,
                                   offsets,
                                   dtdS_over_dV_cur,
                                   dtdS_over_dV_cur,
                                   dtdS_over_dV_cur,
                                   dx,
                                   dx,
                                   dx);

    offsets = { 0.0, -0.5, 0.0 };

    auto qR = reconstruct_state_3d(qprim,
                                   is,
                                   js,
                                   ks,
                                   iOct_local,
                                   offsets,
                                   dtdS_over_dV_cur,
                                   dtdS_over_dV_cur,
                                   dtdS_over_dV_cur,
                                   dx,
                                   dx,
                                   dx);

    // swap IU / IV
    my_swap(qL[Hydro::IU], qL[Hydro::IV]);
    my_swap(qR[Hydro::IU], qR[Hydro::IV]);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_hydro<3>(qL, qR, m_hydro_settings, m_eos);

    my_swap(flux[Hydro::IU], flux[Hydro::IV]);

    // step 4 : accumulate flux in current cell
    const auto flux_cur = flux * dtdS_over_dV_cur;

    set_flux(i, j, k, iOct_local, flux_cur);
  } // end update along Y

  /*
   * compute flux from left face along Z dir
   */
  if (m_direction == IZ)
  {
    // get state in neighbor along Z
    auto qprim_n = get_prim_variables(iq, jq, kq - 1, iOct_local);

    // step 1 : reconstruct state in the left neighbor
    offsets_t<3> offsets = { 0.0, 0.0, 0.5 };

    // reconstruct "left" state
    auto qL = reconstruct_state_3d(qprim_n,
                                   is,
                                   js,
                                   ks - 1,
                                   iOct_local,
                                   offsets,
                                   dtdS_over_dV_cur,
                                   dtdS_over_dV_cur,
                                   dtdS_over_dV_cur,
                                   dx,
                                   dx,
                                   dx);

    offsets = { 0.0, 0.0, -0.5 };

    auto qR = reconstruct_state_3d(qprim,
                                   is,
                                   js,
                                   ks,
                                   iOct_local,
                                   offsets,
                                   dtdS_over_dV_cur,
                                   dtdS_over_dV_cur,
                                   dtdS_over_dV_cur,
                                   dx,
                                   dx,
                                   dx);

    // swap IU / IW
    my_swap(qL[Hydro::IU], qL[Hydro::IW]);
    my_swap(qR[Hydro::IU], qR[Hydro::IW]);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_hydro<3>(qL, qR, m_hydro_settings, m_eos);

    my_swap(flux[Hydro::IU], flux[Hydro::IW]);

    // step 4 : accumulate flux in current cell
    const auto flux_cur = flux * dtdS_over_dV_cur;

    set_flux(i, j, k, iOct_local, flux_cur);
  } // end update along Z

} // compute_fluxes_and_store_3d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeFluxesAndStoreFunctor<dim, device_t>::operator()(const index_t & global_index) const
{

  // retrieve local octant index in range [0, num_quads_to_process [
  auto const iOct_local = static_cast<int32_t>(global_index / m_Fluxes.num_cells());
  auto const cell_index = static_cast<int32_t>(global_index - iOct_local * m_Fluxes.num_cells());

  if constexpr (dim == 2)
  {
    compute_fluxes_and_store_2d(cell_index, iOct_local);
  }
  else if constexpr (dim == 3)
  {
    compute_fluxes_and_store_3d(cell_index, iOct_local);
  }

} // operator ()

// explicit template instantiation
template class ComputeFluxesAndStoreFunctor<2, kalypsso::DefaultDevice>;
template class ComputeFluxesAndStoreFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso
