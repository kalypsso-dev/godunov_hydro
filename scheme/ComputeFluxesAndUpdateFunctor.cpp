// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeFluxesAndUpdateFunctor.cpp
 */
#include <godunov_hydro/scheme/ComputeFluxesAndUpdateFunctor.h>

#include <kalypsso/core/brick_utils.h>
#include <kalypsso/core/orchard_key_utils.h>

namespace kalypsso
{

namespace godunov_hydro
{

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
ComputeFluxesAndUpdateFunctor<dim, device_t>::ComputeFluxesAndUpdateFunctor(
  ConfigMap const &                   config_map,
  amr_hashmap_t const &               amr_hashmap,
  orchard_key_view_t const &          orchard_keys,
  DataArrayBlock_t const &            u,
  DataArrayGhostedBlock_t const &     q,
  DataArrayGhostedBlock_t const &     slopes_x,
  DataArrayGhostedBlock_t const &     slopes_y,
  DataArrayGhostedBlock_t const &     slopes_z,
  int32_t                             iOct_begin,
  int32_t                             num_octants,
  Kokkos::Array<uint8_t, dim> const & brick_sizes,
  Kokkos::Array<bool, dim> const &    is_brick_periodic,
  HydroSettings const &               hydro_settings,
  eos::EosWrapper<device_t> const &   eos,
  real_t                              dt,
  bool                                gravity_enabled,
  UniformGravityField<dim> const &    gravity_field,
  bool                                use_flux_oriented_computation)
  : m_amr_hashmap_device(amr_hashmap)
  , m_orchard_keys_device(orchard_keys)
  , m_mirror_orchard_keys_device()
  , m_U(u)
  , m_q(q)
  , m_slopes_x(slopes_x)
  , m_slopes_y(slopes_y)
  , m_slopes_z(slopes_z)
  , m_iOct_begin(iOct_begin)
  , m_num_octants(num_octants)
  , m_block_sizes(slopes_x.block_size())
  , m_block_sizes_fluxes(slopes_x.block_size() + 1)
  , m_nbCellsPerLeaf(Kokkos::dim_prod(m_block_sizes))
  , m_nbFluxesPerLeaf(Kokkos::dim_prod(m_block_sizes_fluxes))
  , m_brick_sizes(brick_sizes)
  , m_is_brick_periodic(is_brick_periodic)
  , m_hydro_settings(hydro_settings)
  , m_eos(eos)
  , m_dt(dt)
  , m_scaling_factor(get_scaling_factor(config_map))
  , m_gravity_enabled(gravity_enabled)
  , m_gravity_field(gravity_field)
  , m_use_flux_oriented_computation(use_flux_oriented_computation)
{} // constructor

// ==============================================================
// ==============================================================
//! static method which does it all: create and execute functor with range policy
//!
//! Use this member when computing primitive in a group of octant
template <size_t dim, typename device_t>
void
ComputeFluxesAndUpdateFunctor<dim, device_t>::apply_on_group(
  ConfigMap const &                   config_map,
  amr_hashmap_t const &               amr_hashmap,
  orchard_key_view_t const &          orchard_keys,
  DataArrayBlock_t const &            U,
  DataArrayGhostedBlock_t const &     q,
  DataArrayGhostedBlock_t const &     slopes_x,
  DataArrayGhostedBlock_t const &     slopes_y,
  DataArrayGhostedBlock_t const &     slopes_z,
  int32_t                             iOct_begin,
  int32_t                             num_octants,
  Kokkos::Array<uint8_t, dim> const & brick_sizes,
  Kokkos::Array<bool, dim> const &    is_brick_periodic,
  HydroSettings const &               hydro_settings,
  eos::EosWrapper<device_t> const &   eos,
  real_t                              dt,
  bool                                use_flux_oriented_computation)
{

  const auto gravity_enabled = config_map.getBool("gravity", "enabled", false);
  const auto gravity_field = get_uniform_gravity_vector<dim>(config_map);

  ComputeFluxesAndUpdateFunctor<dim, device_t> functor(config_map,
                                                       amr_hashmap,
                                                       orchard_keys,
                                                       U,
                                                       q,
                                                       slopes_x,
                                                       slopes_y,
                                                       slopes_z,
                                                       iOct_begin,  // first index to process
                                                       num_octants, // number of octants to process
                                                       brick_sizes,
                                                       is_brick_periodic,
                                                       hydro_settings,
                                                       eos,
                                                       dt,
                                                       gravity_enabled,
                                                       gravity_field,
                                                       use_flux_oriented_computation);

  // if we use atomic update, computations is flux-oriented
  // else computation is cell-oriented
  const auto nbIterations = use_flux_oriented_computation
                              ? num_octants * functor.nb_fluxes_per_leaf()
                              : num_octants * functor.nb_cells_per_leaf();

  // launch computation
  Kokkos::parallel_for("kalypsso::godunov_hydro::ComputeFluxesAndUpdateFunctor",
                       Kokkos::RangePolicy<exec_space, TagComputeAllQuadInGroup>(0, nbIterations),
                       functor);

} // apply_on_group

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION auto
ComputeFluxesAndUpdateFunctor<dim, device_t>::get_cons_variables(int32_t i,
                                                                 int32_t j,
                                                                 int32_t iOct_global) const
{

  HydroState<2> u;

  u[Hydro<dim>::ID] = m_U(i, j, Hydro<dim>::ID, iOct_global);
  u[Hydro<dim>::IP] = m_U(i, j, Hydro<dim>::IP, iOct_global);
  u[Hydro<dim>::IU] = m_U(i, j, Hydro<dim>::IU, iOct_global);
  u[Hydro<dim>::IV] = m_U(i, j, Hydro<dim>::IV, iOct_global);

  return u;

} // get_cons_variables

// ====================================================================
// ====================================================================
/**
 * Get conservative variables state vector.
 *
 * \param[in] index identifies location in the ghosted block
 * \param[in] iOct_global identifies octant (global)
 *
 */
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION auto
ComputeFluxesAndUpdateFunctor<dim, device_t>::get_cons_variables(int32_t i,
                                                                 int32_t j,
                                                                 int32_t k,
                                                                 int32_t iOct_global) const
{

  HydroState<3> u;

  u[Hydro<dim>::ID] = m_U(i, j, k, Hydro<dim>::ID, iOct_global);
  u[Hydro<dim>::IP] = m_U(i, j, k, Hydro<dim>::IP, iOct_global);
  u[Hydro<dim>::IU] = m_U(i, j, k, Hydro<dim>::IU, iOct_global);
  u[Hydro<dim>::IV] = m_U(i, j, k, Hydro<dim>::IV, iOct_global);
  u[Hydro<dim>::IW] = m_U(i, j, k, Hydro<dim>::IW, iOct_global);

  return u;

} // get_cons_variables

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION auto
ComputeFluxesAndUpdateFunctor<dim, device_t>::get_prim_variables(int32_t i,
                                                                 int32_t j,
                                                                 int32_t iOct_local) const
{

  HydroState<2> q;

  q[Hydro<dim>::ID] = m_q(i, j, Hydro<dim>::ID, iOct_local);
  q[Hydro<dim>::IP] = m_q(i, j, Hydro<dim>::IP, iOct_local);
  q[Hydro<dim>::IU] = m_q(i, j, Hydro<dim>::IU, iOct_local);
  q[Hydro<dim>::IV] = m_q(i, j, Hydro<dim>::IV, iOct_local);

  return q;

} // get_prim_variables

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION auto
ComputeFluxesAndUpdateFunctor<dim, device_t>::get_prim_variables(int32_t i,
                                                                 int32_t j,
                                                                 int32_t k,
                                                                 int32_t iOct_local) const
{

  HydroState<3> q;

  q[Hydro<dim>::ID] = m_q(i, j, k, Hydro<dim>::ID, iOct_local);
  q[Hydro<dim>::IP] = m_q(i, j, k, Hydro<dim>::IP, iOct_local);
  q[Hydro<dim>::IU] = m_q(i, j, k, Hydro<dim>::IU, iOct_local);
  q[Hydro<dim>::IV] = m_q(i, j, k, Hydro<dim>::IV, iOct_local);
  q[Hydro<dim>::IW] = m_q(i, j, k, Hydro<dim>::IW, iOct_local);

  return q;

} // get_prim_variables

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION auto
ComputeFluxesAndUpdateFunctor<dim, device_t>::reconstruct_state_2d(const HydroState<2> & q,
                                                                   int32_t               is,
                                                                   int32_t               js,
                                                                   int32_t               iOct_local,
                                                                   const offsets_t<2> &  offsets,
                                                                   real_t                dtdx,
                                                                   real_t                dtdy) const
{
  auto const & smallr = m_hydro_settings.smallr;
  auto const & smallp = m_hydro_settings.smallp;

  // retrieve primitive variables in current quadrant
  auto const & r = q[Hydro<dim>::ID];
  auto const & p = q[Hydro<dim>::IP];
  auto const & u = q[Hydro<dim>::IU];
  auto const & v = q[Hydro<dim>::IV];
  // auto const w = 0.0;

  const auto c = m_eos.sound_speed(p, r);

  auto const drx = m_slopes_x(is, js, Hydro<dim>::ID, iOct_local);
  auto const dpx = m_slopes_x(is, js, Hydro<dim>::IP, iOct_local);
  auto const dux = m_slopes_x(is, js, Hydro<dim>::IU, iOct_local);
  auto const dvx = m_slopes_x(is, js, Hydro<dim>::IV, iOct_local);
  // auto const dwx = 0.0;

  auto const dry = m_slopes_y(is, js, Hydro<dim>::ID, iOct_local);
  auto const dpy = m_slopes_y(is, js, Hydro<dim>::IP, iOct_local);
  auto const duy = m_slopes_y(is, js, Hydro<dim>::IU, iOct_local);
  auto const dvy = m_slopes_y(is, js, Hydro<dim>::IV, iOct_local);
  // auto const dwy = 0.0;

  // MUSCL-Hancock half time-step
  // clang-format off
  auto const sr0 = (-u * drx - dux * r) * dtdx + (-v * dry - dvy * r) * dtdy;
  auto const su0 = (-u * dux - dpx / r) * dtdx + (-v * duy          ) * dtdy;
  auto const sv0 = (-u * dvx          ) * dtdx + (-v * dvy - dpy / r) * dtdy;
  auto const sp0 = (-u * dpx - dux * r * c * c) * dtdx +
                   (-v * dpy - dvy * r * c * c) * dtdy;
  // clang-format on

  // reconstruct state on interface
  HydroState<2> qr;

  qr[Hydro<dim>::ID] = r + HALF_F * sr0 + offsets[IX] * drx + offsets[IY] * dry;
  qr[Hydro<dim>::IP] = p + HALF_F * sp0 + offsets[IX] * dpx + offsets[IY] * dpy;
  qr[Hydro<dim>::IU] = u + HALF_F * su0 + offsets[IX] * dux + offsets[IY] * duy;
  qr[Hydro<dim>::IV] = v + HALF_F * sv0 + offsets[IX] * dvx + offsets[IY] * dvy;

  qr[Hydro<dim>::ID] = fmax(smallr, qr[Hydro<dim>::ID]);
  qr[Hydro<dim>::IP] = fmax(smallp * qr[Hydro<dim>::ID], qr[Hydro<dim>::IP]);

  // add gravity predictor step
  if (m_gravity_enabled)
  {
    qr[Hydro<dim>::IU] += m_gravity_field[IX] * HALF_F * m_dt;
    qr[Hydro<dim>::IV] += m_gravity_field[IY] * HALF_F * m_dt;
  }

  return qr;

} // reconstruct_state_2d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION auto
ComputeFluxesAndUpdateFunctor<dim, device_t>::reconstruct_state_3d(const HydroState<3> & q,
                                                                   int32_t               is,
                                                                   int32_t               js,
                                                                   int32_t               ks,
                                                                   int32_t               iOct_local,
                                                                   const offsets_t<3> &  offsets,
                                                                   real_t                dtdx,
                                                                   real_t                dtdy,
                                                                   real_t                dtdz) const
{
  auto const & smallr = m_hydro_settings.smallr;
  auto const & smallp = m_hydro_settings.smallp;

  // retrieve primitive variables in current quadrant
  const real_t & r = q[Hydro<dim>::ID];
  const real_t & p = q[Hydro<dim>::IP];
  const real_t & u = q[Hydro<dim>::IU];
  const real_t & v = q[Hydro<dim>::IV];
  const real_t & w = q[Hydro<dim>::IW];

  const auto c = m_eos.sound_speed(p, r);

  // retrieve variations = dx * slopes
  const real_t drx = m_slopes_x(is, js, ks, Hydro<dim>::ID, iOct_local);
  const real_t dpx = m_slopes_x(is, js, ks, Hydro<dim>::IP, iOct_local);
  const real_t dux = m_slopes_x(is, js, ks, Hydro<dim>::IU, iOct_local);
  const real_t dvx = m_slopes_x(is, js, ks, Hydro<dim>::IV, iOct_local);
  const real_t dwx = m_slopes_x(is, js, ks, Hydro<dim>::IW, iOct_local);

  const real_t dry = m_slopes_y(is, js, ks, Hydro<dim>::ID, iOct_local);
  const real_t dpy = m_slopes_y(is, js, ks, Hydro<dim>::IP, iOct_local);
  const real_t duy = m_slopes_y(is, js, ks, Hydro<dim>::IU, iOct_local);
  const real_t dvy = m_slopes_y(is, js, ks, Hydro<dim>::IV, iOct_local);
  const real_t dwy = m_slopes_y(is, js, ks, Hydro<dim>::IW, iOct_local);

  const real_t drz = m_slopes_z(is, js, ks, Hydro<dim>::ID, iOct_local);
  const real_t dpz = m_slopes_z(is, js, ks, Hydro<dim>::IP, iOct_local);
  const real_t duz = m_slopes_z(is, js, ks, Hydro<dim>::IU, iOct_local);
  const real_t dvz = m_slopes_z(is, js, ks, Hydro<dim>::IV, iOct_local);
  const real_t dwz = m_slopes_z(is, js, ks, Hydro<dim>::IW, iOct_local);

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

  // reconstruct state on interface
  HydroState<3> qr;

  qr[Hydro<dim>::ID] = r + HALF_F * sr0 + offsets[IX] * drx + offsets[IY] * dry + offsets[IZ] * drz;
  qr[Hydro<dim>::IP] = p + HALF_F * sp0 + offsets[IX] * dpx + offsets[IY] * dpy + offsets[IZ] * dpz;
  qr[Hydro<dim>::IU] = u + HALF_F * su0 + offsets[IX] * dux + offsets[IY] * duy + offsets[IZ] * duz;
  qr[Hydro<dim>::IV] = v + HALF_F * sv0 + offsets[IX] * dvx + offsets[IY] * dvy + offsets[IZ] * dvz;
  qr[Hydro<dim>::IW] = w + HALF_F * sw0 + offsets[IX] * dwx + offsets[IY] * dwy + offsets[IZ] * dwz;

  qr[Hydro<dim>::ID] = fmax(smallr, qr[Hydro<dim>::ID]);
  qr[Hydro<dim>::IP] = fmax(smallp * qr[Hydro<dim>::ID], qr[Hydro<dim>::IP]);

  // add gravity predictor step
  if (m_gravity_enabled)
  {
    qr[Hydro<dim>::IU] += m_gravity_field[IX] * HALF_F * m_dt;
    qr[Hydro<dim>::IV] += m_gravity_field[IY] * HALF_F * m_dt;
    qr[Hydro<dim>::IW] += m_gravity_field[IZ] * HALF_F * m_dt;
  }

  return qr;

} // reconstruct_state_3d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION void
ComputeFluxesAndUpdateFunctor<dim, device_t>::compute_fluxes_and_update_2d(
  index_t const & cell_index,
  index_t const & iOct_local) const
{

  auto const iOct_global = m_iOct_begin + iOct_local;

  auto const   coords = cellindex_to_coord<2>(cell_index, m_block_sizes);
  auto const & i = coords[IX];
  auto const & j = coords[IY];

  // get AMR level
  auto const level = orchard_key_t<2>::level(m_orchard_keys_device(iOct_global));

  // compute dx / dy in real space units
  auto const dx = compute_cell_length<2>(level, m_block_sizes[IX]) * m_scaling_factor;
  auto const dy = compute_cell_length<2>(level, m_block_sizes[IY]) * m_scaling_factor;

  auto const dtdx = m_dt / dx;
  auto const dtdy = m_dt / dy;

  /*
   * reconstruct states on cells face and update
   */

  // get current location primitive variables state
  // note: primitive variables is a ghosted array with ghost width of 2
  auto qprim = get_prim_variables(i, j, iOct_local);

  // fluxes will be accumulated in qcons
  // note: conservative variables is a non-ghosted array
  auto qcons = get_cons_variables(i, j, iOct_global);

  /*
   * compute flux from left face along X dir
   */
  {
    // get state in neighbor along X
    auto qprim_n = get_prim_variables(i - 1, j, iOct_local);

    // step 1 : reconstruct state in the left neighbor
    offsets_t<2> offsets{ 1.0, 0.0 };

    // reconstruct state in left neighbor (index relative to slopes array)
    auto qL = reconstruct_state_2d(qprim_n, i - 1, j, iOct_local, offsets, dtdx, dtdy);

    // step 2 : reconstruct state in current cell
    offsets = { -1.0, 0.0 };

    // reconstruct state from current cell center to left interface
    auto qR = reconstruct_state_2d(qprim, i, j, iOct_local, offsets, dtdx, dtdy);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_hydro<2>(qL, qR, m_hydro_settings, m_eos);

    // step 4 : accumulate flux in current cell
    qcons += flux * dtdx;
  }

  /*
   * compute flux from right face along X dir
   */
  {
    // get state in neighbor along X
    auto qprim_n = get_prim_variables(i + 1, j, iOct_local);

    // step 1 : reconstruct state in the left neighbor
    offsets_t<2> offsets = { -1.0, 0.0 };

    // reconstruct state in right neighbor
    auto qR = reconstruct_state_2d(qprim_n, i + 1, j, iOct_local, offsets, dtdx, dtdy);

    // step 2 : reconstruct state in current cell
    offsets = { 1.0, 0.0 };

    auto qL = reconstruct_state_2d(qprim, i, j, iOct_local, offsets, dtdx, dtdy);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_hydro<2>(qL, qR, m_hydro_settings, m_eos);

    // step 4 : accumulate flux in current cell
    qcons -= flux * dtdx;
  }

  /*
   * compute flux from left face along Y dir
   */
  {
    // get state in neighbor along Y
    auto qprim_n = get_prim_variables(i, j - 1, iOct_local);

    // step 1 : reconstruct state in the left neighbor
    offsets_t<2> offsets = { 0.0, 1.0 };

    // reconstruct "left" state
    auto qL = reconstruct_state_2d(qprim_n, i, j - 1, iOct_local, offsets, dtdx, dtdy);

    // step 2 : reconstruct state in current cell
    offsets = { 0.0, -1.0 };

    auto qR = reconstruct_state_2d(qprim, i, j, iOct_local, offsets, dtdx, dtdy);

    // swap IU / IV
    my_swap(qL[Hydro<dim>::IU], qL[Hydro<dim>::IV]);
    my_swap(qR[Hydro<dim>::IU], qR[Hydro<dim>::IV]);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_hydro<2>(qL, qR, m_hydro_settings, m_eos);

    my_swap(flux[Hydro<dim>::IU], flux[Hydro<dim>::IV]);

    // step 4 : accumulate flux in current cell
    qcons += flux * dtdy;
  }

  /*
   * compute flux from right face along Y dir
   */
  {
    // get state in neighbor along Y
    auto qprim_n = get_prim_variables(i, j + 1, iOct_local);

    // step 1 : reconstruct state in the left neighbor
    offsets_t<2> offsets = { 0.0, -1.0 };

    // reconstruct "left" state
    auto qR = reconstruct_state_2d(qprim_n, i, j + 1, iOct_local, offsets, dtdx, dtdy);

    // step 2 : reconstruct state in current cell
    offsets = { 0.0, 1.0 };

    auto qL = reconstruct_state_2d(qprim, i, j, iOct_local, offsets, dtdx, dtdy);

    // swap IU / IV
    my_swap(qL[Hydro<dim>::IU], qL[Hydro<dim>::IV]);
    my_swap(qR[Hydro<dim>::IU], qR[Hydro<dim>::IV]);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_hydro<2>(qL, qR, m_hydro_settings, m_eos);

    my_swap(flux[Hydro<dim>::IU], flux[Hydro<dim>::IV]);

    // step 4 : accumulate flux in current cell
    qcons -= flux * dtdy;
  }

  m_U(i, j, Hydro<dim>::ID, iOct_global) = qcons[Hydro<dim>::ID];
  m_U(i, j, Hydro<dim>::IP, iOct_global) = qcons[Hydro<dim>::IP];
  m_U(i, j, Hydro<dim>::IU, iOct_global) = qcons[Hydro<dim>::IU];
  m_U(i, j, Hydro<dim>::IV, iOct_global) = qcons[Hydro<dim>::IV];

} // compute_fluxes_and_update_2d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION void
ComputeFluxesAndUpdateFunctor<dim, device_t>::compute_fluxes_and_update_2d_atomic(
  index_t const & cell_index,
  index_t const & iOct_local) const
{

  auto const iOct_global = m_iOct_begin + iOct_local;

  auto const   coords = cellindex_to_coord<2>(cell_index, m_block_sizes_fluxes);
  auto const & i = coords[IX];
  auto const & j = coords[IY];

  // get AMR level
  auto const level = orchard_key_t<2>::level(m_orchard_keys_device(iOct_global));

  // compute dx / dy in real space units
  auto const dx = compute_cell_length<2>(level, m_block_sizes[IX]) * m_scaling_factor;
  auto const dy = compute_cell_length<2>(level, m_block_sizes[IY]) * m_scaling_factor;

  auto const dtdx = m_dt / dx;
  auto const dtdy = m_dt / dy;

  /*
   * reconstruct states on cells face and update
   */

  // get current location primitive variables state
  // note: primitive variables is a ghosted array with ghost width of 2
  auto qprim = get_prim_variables(i, j, iOct_local);

  /*
   * compute flux from left face along X dir
   */
  {
    // get state in neighbor along X
    auto qprim_n = get_prim_variables(i - 1, j, iOct_local);

    // step 1 : reconstruct state in the left neighbor
    offsets_t<2> offsets{ 1.0, 0.0 };

    // reconstruct state in left neighbor (index relative to slopes array)
    auto qL = reconstruct_state_2d(qprim_n, i - 1, j, iOct_local, offsets, dtdx, dtdy);

    // step 2 : reconstruct state in current cell
    offsets = { -1.0, 0.0 };

    // reconstruct state from current cell center to left interface
    auto qR = reconstruct_state_2d(qprim, i, j, iOct_local, offsets, dtdx, dtdy);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_hydro<2>(qL, qR, m_hydro_settings, m_eos);

    // step 4 : accumulate flux in current cell
    const auto flux_accum = flux * dtdx;

    if (i < m_block_sizes[IX] and j < m_block_sizes[IY])
    {
      Kokkos::atomic_add(&m_U(i, j, Hydro<dim>::ID, iOct_global), flux_accum[Hydro<dim>::ID]);
      Kokkos::atomic_add(&m_U(i, j, Hydro<dim>::IP, iOct_global), flux_accum[Hydro<dim>::IP]);
      Kokkos::atomic_add(&m_U(i, j, Hydro<dim>::IU, iOct_global), flux_accum[Hydro<dim>::IU]);
      Kokkos::atomic_add(&m_U(i, j, Hydro<dim>::IV, iOct_global), flux_accum[Hydro<dim>::IV]);
    }
    if (i > 0 and j < m_block_sizes[IY])
    {
      Kokkos::atomic_sub(&m_U(i - 1, j, Hydro<dim>::ID, iOct_global), flux_accum[Hydro<dim>::ID]);
      Kokkos::atomic_sub(&m_U(i - 1, j, Hydro<dim>::IP, iOct_global), flux_accum[Hydro<dim>::IP]);
      Kokkos::atomic_sub(&m_U(i - 1, j, Hydro<dim>::IU, iOct_global), flux_accum[Hydro<dim>::IU]);
      Kokkos::atomic_sub(&m_U(i - 1, j, Hydro<dim>::IV, iOct_global), flux_accum[Hydro<dim>::IV]);
    }
  }

  /*
   * compute flux from left face along Y dir
   */
  {
    // get state in neighbor along Y
    auto qprim_n = get_prim_variables(i, j - 1, iOct_local);

    // step 1 : reconstruct state in the left neighbor
    offsets_t<2> offsets = { 0.0, 1.0 };

    // reconstruct "left" state
    auto qL = reconstruct_state_2d(qprim_n, i, j - 1, iOct_local, offsets, dtdx, dtdy);

    // step 2 : reconstruct state in current cell
    offsets = { 0.0, -1.0 };

    auto qR = reconstruct_state_2d(qprim, i, j, iOct_local, offsets, dtdx, dtdy);

    // swap IU / IV
    my_swap(qL[Hydro<dim>::IU], qL[Hydro<dim>::IV]);
    my_swap(qR[Hydro<dim>::IU], qR[Hydro<dim>::IV]);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_hydro<2>(qL, qR, m_hydro_settings, m_eos);

    my_swap(flux[Hydro<dim>::IU], flux[Hydro<dim>::IV]);

    // step 4 : accumulate flux in current cell
    const auto flux_accum = flux * dtdy;

    if (i < m_block_sizes[IX] and j < m_block_sizes[IY])
    {
      Kokkos::atomic_add(&m_U(i, j, Hydro<dim>::ID, iOct_global), flux_accum[Hydro<dim>::ID]);
      Kokkos::atomic_add(&m_U(i, j, Hydro<dim>::IP, iOct_global), flux_accum[Hydro<dim>::IP]);
      Kokkos::atomic_add(&m_U(i, j, Hydro<dim>::IU, iOct_global), flux_accum[Hydro<dim>::IU]);
      Kokkos::atomic_add(&m_U(i, j, Hydro<dim>::IV, iOct_global), flux_accum[Hydro<dim>::IV]);
    }
    if (i < m_block_sizes[IX] and j > 0)
    {
      Kokkos::atomic_sub(&m_U(i, j - 1, Hydro<dim>::ID, iOct_global), flux_accum[Hydro<dim>::ID]);
      Kokkos::atomic_sub(&m_U(i, j - 1, Hydro<dim>::IP, iOct_global), flux_accum[Hydro<dim>::IP]);
      Kokkos::atomic_sub(&m_U(i, j - 1, Hydro<dim>::IU, iOct_global), flux_accum[Hydro<dim>::IU]);
      Kokkos::atomic_sub(&m_U(i, j - 1, Hydro<dim>::IV, iOct_global), flux_accum[Hydro<dim>::IV]);
    }
  }

} // compute_fluxes_and_update_2d_atomic

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION void
ComputeFluxesAndUpdateFunctor<dim, device_t>::compute_fluxes_and_update_3d(
  const index_t & cell_index,
  const index_t & iOct_local) const
{
  auto const iOct_global = m_iOct_begin + iOct_local;

  auto const   coords = cellindex_to_coord<3>(cell_index, m_block_sizes);
  auto const & i = coords[IX];
  auto const & j = coords[IY];
  auto const & k = coords[IZ];

  // get AMR level
  auto const level = orchard_key_t<3>::level(m_orchard_keys_device(iOct_global));

  // compute dx / dy / dz in real space units
  auto const dx = compute_cell_length<3>(level, m_block_sizes[IX]) * m_scaling_factor;
  auto const dy = compute_cell_length<3>(level, m_block_sizes[IY]) * m_scaling_factor;
  auto const dz = compute_cell_length<3>(level, m_block_sizes[IZ]) * m_scaling_factor;

  auto const dtdx = m_dt / dx;
  auto const dtdy = m_dt / dy;
  auto const dtdz = m_dt / dz;

  /*
   * reconstruct states on cells face and update
   */

  // get current location primitive variables state
  // note: primitive variables is a ghosted array with ghost width of 2
  auto qprim = get_prim_variables(i, j, k, iOct_local);

  // fluxes will be accumulated in qcons
  // note: conservative variables is a non-ghosted array
  auto qcons = get_cons_variables(i, j, k, iOct_global);

  /*
   * compute flux from left face along X dir
   */
  {
    // get state in neighbor along X
    auto qprim_n = get_prim_variables(i - 1, j, k, iOct_local);

    // step 1 : reconstruct state in the left neighbor
    offsets_t<3> offsets{ 1.0, 0.0, 0.0 };

    // reconstruct state in left neighbor (index relative to slopes array)
    auto qL = reconstruct_state_3d(qprim_n, i - 1, j, k, iOct_local, offsets, dtdx, dtdy, dtdz);

    // step 2 : reconstruct state in current cell
    offsets = { -1.0, 0.0, 0.0 };

    // reconstruct state from current cell center to left interface
    auto qR = reconstruct_state_3d(qprim, i, j, k, iOct_local, offsets, dtdx, dtdy, dtdz);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_hydro<3>(qL, qR, m_hydro_settings, m_eos);

    // step 4 : accumulate flux in current cell
    qcons += flux * dtdx;
  }

  /*
   * compute flux from right face along X dir
   */
  {
    // get state in neighbor along X
    auto qprim_n = get_prim_variables(i + 1, j, k, iOct_local);

    // step 1 : reconstruct state in the left neighbor
    offsets_t<3> offsets = { -1.0, 0.0, 0.0 };

    // reconstruct state in right neighbor
    auto qR = reconstruct_state_3d(qprim_n, i + 1, j, k, iOct_local, offsets, dtdx, dtdy, dtdz);

    // step 2 : reconstruct state in current cell
    offsets = { 1.0, 0.0, 0.0 };

    auto qL = reconstruct_state_3d(qprim, i, j, k, iOct_local, offsets, dtdx, dtdy, dtdz);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_hydro<3>(qL, qR, m_hydro_settings, m_eos);

    // step 4 : accumulate flux in current cell
    qcons -= flux * dtdx;
  }

  /*
   * compute flux from left face along Y dir
   */
  {
    // get state in neighbor along Y
    auto qprim_n = get_prim_variables(i, j - 1, k, iOct_local);

    // step 1 : reconstruct state in the left neighbor
    offsets_t<3> offsets = { 0.0, 1.0, 0.0 };

    // reconstruct "left" state
    auto qL = reconstruct_state_3d(qprim_n, i, j - 1, k, iOct_local, offsets, dtdx, dtdy, dtdz);

    offsets = { 0.0, -1.0, 0.0 };

    auto qR = reconstruct_state_3d(qprim, i, j, k, iOct_local, offsets, dtdx, dtdy, dtdz);

    // swap IU / IV
    my_swap(qL[Hydro<dim>::IU], qL[Hydro<dim>::IV]);
    my_swap(qR[Hydro<dim>::IU], qR[Hydro<dim>::IV]);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_hydro<3>(qL, qR, m_hydro_settings, m_eos);

    my_swap(flux[Hydro<dim>::IU], flux[Hydro<dim>::IV]);

    // step 4 : accumulate flux in current cell
    qcons += flux * dtdy;
  }

  /*
   * compute flux from right face along Y dir
   */
  {
    // get state in neighbor along Y
    auto qprim_n = get_prim_variables(i, j + 1, k, iOct_local);

    // step 1 : reconstruct state in the left neighbor
    offsets_t<3> offsets = { 0.0, -1.0, 0.0 };

    // reconstruct "left" state
    auto qR = reconstruct_state_3d(qprim_n, i, j + 1, k, iOct_local, offsets, dtdx, dtdy, dtdz);

    // step 2 : reconstruct state in current cell
    offsets = { 0.0, 1.0, 0.0 };

    auto qL = reconstruct_state_3d(qprim, i, j, k, iOct_local, offsets, dtdx, dtdy, dtdz);

    // swap IU / IV
    my_swap(qL[Hydro<dim>::IU], qL[Hydro<dim>::IV]);
    my_swap(qR[Hydro<dim>::IU], qR[Hydro<dim>::IV]);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_hydro<3>(qL, qR, m_hydro_settings, m_eos);

    my_swap(flux[Hydro<dim>::IU], flux[Hydro<dim>::IV]);

    // step 4 : accumulate flux in current cell
    qcons -= flux * dtdy;
  }

  /*
   * compute flux from left face along Z dir
   */
  {
    // get state in neighbor along Z
    auto qprim_n = get_prim_variables(i, j, k - 1, iOct_local);

    // step 1 : reconstruct state in the left neighbor
    offsets_t<3> offsets = { 0.0, 0.0, 1.0 };

    // reconstruct "left" state
    auto qL = reconstruct_state_3d(qprim_n, i, j, k - 1, iOct_local, offsets, dtdx, dtdy, dtdz);

    offsets = { 0.0, 0.0, -1.0 };

    auto qR = reconstruct_state_3d(qprim, i, j, k, iOct_local, offsets, dtdx, dtdy, dtdz);

    // swap IU / IW
    my_swap(qL[Hydro<dim>::IU], qL[Hydro<dim>::IW]);
    my_swap(qR[Hydro<dim>::IU], qR[Hydro<dim>::IW]);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_hydro<3>(qL, qR, m_hydro_settings, m_eos);

    my_swap(flux[Hydro<dim>::IU], flux[Hydro<dim>::IW]);

    // step 4 : accumulate flux in current cell
    qcons += flux * dtdz;
  }

  /*
   * compute flux from right face along Z dir
   */
  {
    // get state in neighbor along Z
    auto qprim_n = get_prim_variables(i, j, k + 1, iOct_local);

    // step 1 : reconstruct state in the left neighbor
    offsets_t<3> offsets = { 0.0, 0.0, -1.0 };

    // reconstruct "left" state
    auto qR = reconstruct_state_3d(qprim_n, i, j, k + 1, iOct_local, offsets, dtdx, dtdy, dtdz);

    // step 2 : reconstruct state in current cell
    offsets = { 0.0, 0.0, 1.0 };

    auto qL = reconstruct_state_3d(qprim, i, j, k, iOct_local, offsets, dtdx, dtdy, dtdz);

    // swap IU / IW
    my_swap(qL[Hydro<dim>::IU], qL[Hydro<dim>::IW]);
    my_swap(qR[Hydro<dim>::IU], qR[Hydro<dim>::IW]);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_hydro<3>(qL, qR, m_hydro_settings, m_eos);

    my_swap(flux[Hydro<dim>::IU], flux[Hydro<dim>::IW]);

    // step 4 : accumulate flux in current cell
    qcons -= flux * dtdz;
  }

  m_U(i, j, k, Hydro<dim>::ID, iOct_global) = qcons[Hydro<dim>::ID];
  m_U(i, j, k, Hydro<dim>::IP, iOct_global) = qcons[Hydro<dim>::IP];
  m_U(i, j, k, Hydro<dim>::IU, iOct_global) = qcons[Hydro<dim>::IU];
  m_U(i, j, k, Hydro<dim>::IV, iOct_global) = qcons[Hydro<dim>::IV];
  m_U(i, j, k, Hydro<dim>::IW, iOct_global) = qcons[Hydro<dim>::IW];

} // compute_fluxes_and_update_3d

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION void
ComputeFluxesAndUpdateFunctor<dim, device_t>::compute_fluxes_and_update_3d_atomic(
  const index_t & cell_index,
  const index_t & iOct_local) const
{
  auto const iOct_global = m_iOct_begin + iOct_local;

  auto const   coords = cellindex_to_coord<3>(cell_index, m_block_sizes);
  auto const & i = coords[IX];
  auto const & j = coords[IY];
  auto const & k = coords[IZ];

  // get AMR level
  auto const level = orchard_key_t<3>::level(m_orchard_keys_device(iOct_local));

  // compute dx / dy / dz in real space units
  auto const dx = compute_cell_length<3>(level, m_block_sizes[IX]) * m_scaling_factor;
  auto const dy = compute_cell_length<3>(level, m_block_sizes[IY]) * m_scaling_factor;
  auto const dz = compute_cell_length<3>(level, m_block_sizes[IZ]) * m_scaling_factor;

  auto const dtdx = m_dt / dx;
  auto const dtdy = m_dt / dy;
  auto const dtdz = m_dt / dz;

  /*
   * reconstruct states on cells face and update
   */

  // get current location primitive variables state
  // note: primitive variables is a ghosted array with ghost width of 2
  auto qprim = get_prim_variables(i, j, k, iOct_local);

  /*
   * compute flux from left face along X dir
   */
  {
    // get state in neighbor along X
    auto qprim_n = get_prim_variables(i - 1, j, k, iOct_local);

    // step 1 : reconstruct state in the left neighbor
    offsets_t<3> offsets{ 1.0, 0.0, 0.0 };

    // reconstruct state in left neighbor (index relative to slopes array)
    auto qL = reconstruct_state_3d(qprim_n, i - 1, j, k, iOct_local, offsets, dtdx, dtdy, dtdz);

    // step 2 : reconstruct state in current cell
    offsets = { -1.0, 0.0, 0.0 };

    // reconstruct state from current cell center to left interface
    auto qR = reconstruct_state_3d(qprim, i, j, k, iOct_local, offsets, dtdx, dtdy, dtdz);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_hydro<3>(qL, qR, m_hydro_settings, m_eos);

    // step 4 : accumulate flux in current cell
    const auto flux_accum = flux * dtdx;

    if (i < m_block_sizes[IX] and j < m_block_sizes[IY] and k < m_block_sizes[IZ])
    {
      Kokkos::atomic_add(&m_U(i, j, k, Hydro<dim>::ID, iOct_global), flux_accum[Hydro<dim>::ID]);
      Kokkos::atomic_add(&m_U(i, j, k, Hydro<dim>::IP, iOct_global), flux_accum[Hydro<dim>::IP]);
      Kokkos::atomic_add(&m_U(i, j, k, Hydro<dim>::IU, iOct_global), flux_accum[Hydro<dim>::IU]);
      Kokkos::atomic_add(&m_U(i, j, k, Hydro<dim>::IV, iOct_global), flux_accum[Hydro<dim>::IV]);
      Kokkos::atomic_add(&m_U(i, j, k, Hydro<dim>::IW, iOct_global), flux_accum[Hydro<dim>::IW]);
    }
    if (i > 0 and j < m_block_sizes[IY] and k < m_block_sizes[IZ])
    {
      Kokkos::atomic_sub(&m_U(i - 1, j, k, Hydro<dim>::ID, iOct_global),
                         flux_accum[Hydro<dim>::ID]);
      Kokkos::atomic_sub(&m_U(i - 1, j, k, Hydro<dim>::IP, iOct_global),
                         flux_accum[Hydro<dim>::IP]);
      Kokkos::atomic_sub(&m_U(i - 1, j, k, Hydro<dim>::IU, iOct_global),
                         flux_accum[Hydro<dim>::IU]);
      Kokkos::atomic_sub(&m_U(i - 1, j, k, Hydro<dim>::IV, iOct_global),
                         flux_accum[Hydro<dim>::IV]);
      Kokkos::atomic_sub(&m_U(i - 1, j, k, Hydro<dim>::IW, iOct_global),
                         flux_accum[Hydro<dim>::IW]);
    }
  }

  /*
   * compute flux from left face along Y dir
   */
  {
    // get state in neighbor along Y
    auto qprim_n = get_prim_variables(i, j - 1, k, iOct_local);

    // step 1 : reconstruct state in the left neighbor
    offsets_t<3> offsets = { 0.0, 1.0, 0.0 };

    // reconstruct "left" state
    auto qL = reconstruct_state_3d(qprim_n, i, j - 1, k, iOct_local, offsets, dtdx, dtdy, dtdz);

    offsets = { 0.0, -1.0, 0.0 };

    auto qR = reconstruct_state_3d(qprim, i, j, k, iOct_local, offsets, dtdx, dtdy, dtdz);

    // swap IU / IV
    my_swap(qL[Hydro<dim>::IU], qL[Hydro<dim>::IV]);
    my_swap(qR[Hydro<dim>::IU], qR[Hydro<dim>::IV]);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_hydro<3>(qL, qR, m_hydro_settings, m_eos);

    my_swap(flux[Hydro<dim>::IU], flux[Hydro<dim>::IV]);

    // step 4 : accumulate flux in current cell
    const auto flux_accum = flux * dtdy;

    if (i < m_block_sizes[IX] and j < m_block_sizes[IY] and k < m_block_sizes[IZ])
    {
      Kokkos::atomic_add(&m_U(i, j, k, Hydro<dim>::ID, iOct_global), flux_accum[Hydro<dim>::ID]);
      Kokkos::atomic_add(&m_U(i, j, k, Hydro<dim>::IP, iOct_global), flux_accum[Hydro<dim>::IP]);
      Kokkos::atomic_add(&m_U(i, j, k, Hydro<dim>::IU, iOct_global), flux_accum[Hydro<dim>::IU]);
      Kokkos::atomic_add(&m_U(i, j, k, Hydro<dim>::IV, iOct_global), flux_accum[Hydro<dim>::IV]);
      Kokkos::atomic_add(&m_U(i, j, k, Hydro<dim>::IW, iOct_global), flux_accum[Hydro<dim>::IW]);
    }
    if (i < m_block_sizes[IX] and j > 0 and k < m_block_sizes[IZ])
    {
      Kokkos::atomic_sub(&m_U(i, j - 1, k, Hydro<dim>::ID, iOct_global),
                         flux_accum[Hydro<dim>::ID]);
      Kokkos::atomic_sub(&m_U(i, j - 1, k, Hydro<dim>::IP, iOct_global),
                         flux_accum[Hydro<dim>::IP]);
      Kokkos::atomic_sub(&m_U(i, j - 1, k, Hydro<dim>::IU, iOct_global),
                         flux_accum[Hydro<dim>::IU]);
      Kokkos::atomic_sub(&m_U(i, j - 1, k, Hydro<dim>::IV, iOct_global),
                         flux_accum[Hydro<dim>::IV]);
      Kokkos::atomic_sub(&m_U(i, j - 1, k, Hydro<dim>::IW, iOct_global),
                         flux_accum[Hydro<dim>::IW]);
    }
  }

  /*
   * compute flux from left face along Z dir
   */
  {
    // get state in neighbor along Z
    auto qprim_n = get_prim_variables(i, j, k - 1, iOct_local);

    // step 1 : reconstruct state in the left neighbor
    offsets_t<3> offsets = { 0.0, 0.0, 1.0 };

    // reconstruct "left" state
    auto qL = reconstruct_state_3d(qprim_n, i, j, k - 1, iOct_local, offsets, dtdx, dtdy, dtdz);

    offsets = { 0.0, 0.0, -1.0 };

    auto qR = reconstruct_state_3d(qprim, i, j, k, iOct_local, offsets, dtdx, dtdy, dtdz);

    // swap IU / IW
    my_swap(qL[Hydro<dim>::IU], qL[Hydro<dim>::IW]);
    my_swap(qR[Hydro<dim>::IU], qR[Hydro<dim>::IW]);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_hydro<3>(qL, qR, m_hydro_settings, m_eos);

    my_swap(flux[Hydro<dim>::IU], flux[Hydro<dim>::IW]);

    // step 4 : accumulate flux in current cell
    const auto flux_accum = flux * dtdz;

    if (i < m_block_sizes[IX] and j < m_block_sizes[IY] and k < m_block_sizes[IZ])
    {
      Kokkos::atomic_add(&m_U(i, j, k, Hydro<dim>::ID, iOct_global), flux_accum[Hydro<dim>::ID]);
      Kokkos::atomic_add(&m_U(i, j, k, Hydro<dim>::IP, iOct_global), flux_accum[Hydro<dim>::IP]);
      Kokkos::atomic_add(&m_U(i, j, k, Hydro<dim>::IU, iOct_global), flux_accum[Hydro<dim>::IU]);
      Kokkos::atomic_add(&m_U(i, j, k, Hydro<dim>::IV, iOct_global), flux_accum[Hydro<dim>::IV]);
      Kokkos::atomic_add(&m_U(i, j, k, Hydro<dim>::IW, iOct_global), flux_accum[Hydro<dim>::IW]);
    }
    if (i < m_block_sizes[IX] and j < m_block_sizes[IY] and k > 0)
    {
      Kokkos::atomic_sub(&m_U(i, j, k - 1, Hydro<dim>::ID, iOct_global),
                         flux_accum[Hydro<dim>::ID]);
      Kokkos::atomic_sub(&m_U(i, j, k - 1, Hydro<dim>::IP, iOct_global),
                         flux_accum[Hydro<dim>::IP]);
      Kokkos::atomic_sub(&m_U(i, j, k - 1, Hydro<dim>::IU, iOct_global),
                         flux_accum[Hydro<dim>::IU]);
      Kokkos::atomic_sub(&m_U(i, j, k - 1, Hydro<dim>::IV, iOct_global),
                         flux_accum[Hydro<dim>::IV]);
      Kokkos::atomic_sub(&m_U(i, j, k - 1, Hydro<dim>::IW, iOct_global),
                         flux_accum[Hydro<dim>::IW]);
    }
  }

} // compute_fluxes_and_update_3d_atomic

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeFluxesAndUpdateFunctor<dim, device_t>::operator()(TagComputeAllQuadInGroup const &,
                                                         const index_t & global_index) const
{

  if (m_use_flux_oriented_computation)
  {
    // retrieve local octant index (local to group)
    auto const iOct_local = global_index / m_nbFluxesPerLeaf;
    auto const cell_index = global_index - iOct_local * m_nbFluxesPerLeaf;

    if constexpr (dim == 2)
      compute_fluxes_and_update_2d_atomic(cell_index, iOct_local);
    else if constexpr (dim == 3)
      compute_fluxes_and_update_3d_atomic(cell_index, iOct_local);
  }
  else
  {

    // retrieve local octant index (local to group)
    auto const iOct_local = global_index / m_nbCellsPerLeaf;
    auto const cell_index = global_index - iOct_local * m_nbCellsPerLeaf;

    if constexpr (dim == 2)
      compute_fluxes_and_update_2d(cell_index, iOct_local);
    else if constexpr (dim == 3)
      compute_fluxes_and_update_3d(cell_index, iOct_local);
  }

} // operator () - TagComputeAllQuadInGroup

// explicit template instantiation
template class ComputeFluxesAndUpdateFunctor<2, kalypsso::DefaultDevice>;
template class ComputeFluxesAndUpdateFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso
