// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeFluxesAndConservativeUpdateFunctor.cpp
 */
#include <godunov_hydro/scheme/ComputeFluxesAndConservativeUpdateFunctor.h>

namespace kalypsso
{

namespace godunov_hydro
{

/*************************************************/
/*************************************************/
/*************************************************/
template <size_t dim, typename device_t>
ComputeFluxesAndConservativeUpdateFunctor<dim, device_t>::ComputeFluxesAndConservativeUpdateFunctor(
  ConfigMap const &                     config_map,
  StencilHelper_t const &               stencil_helper,
  orchard_key_view_t const &            orchard_keys,
  conformal_status_view_type const &    conformal_status,
  AMRMeshInfo const &                   amr_mesh_info,
  DataArrayBlock_t const &              u_in,
  DataArrayBlock_t const &              u_out,
  DataArrayGhostedBlock_t const &       q,
  DataArrayGhostedBlock_t const &       slopes_x,
  DataArrayGhostedBlock_t const &       slopes_y,
  DataArrayGhostedBlock_t const &       slopes_z,
  FieldMap<core::models::Hydro> const & fm,
  int32_t                               iOct_begin,
  int32_t                               num_octants,
  HydroSettings const &                 hydro_settings,
  eos::EosWrapper<device_t> const &     eos,
  real_t                                dt,
  bool                                  gravity_enabled,
  UniformGravityField<dim> const &      gravity_field,
  TimeIntegrator                        time_integrator)
  : m_stencil_helper(stencil_helper)
  , m_orchard_keys_device(orchard_keys)
  , m_conformal_status(conformal_status)
  , m_amr_mesh_info(amr_mesh_info)
  , m_Uin(u_in)
  , m_Uout(u_out)
  , m_q(q)
  , m_slopes_x(slopes_x)
  , m_slopes_y(slopes_y)
  , m_slopes_z(slopes_z)
  , m_fm(fm)
  , m_iOct_begin(iOct_begin)
  , m_num_octants(num_octants)
  , m_block_sizes(slopes_x.block_size())
  , m_block_sizes_fluxes(slopes_x.block_size() + 1)
  , m_nbCellsPerLeaf(Kokkos::dim_prod(m_block_sizes))
  , m_nbFluxesPerLeaf(Kokkos::dim_prod(m_block_sizes_fluxes))
  , m_hydro_settings(hydro_settings)
  , m_eos(eos)
  , m_dt(dt)
  , m_scaling_factor(get_scaling_factor(config_map))
  , m_gravity_enabled(gravity_enabled)
  , m_gravity_field(gravity_field)
  , m_time_integrator(time_integrator)
{} // constructor

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
ComputeFluxesAndConservativeUpdateFunctor<dim, device_t>::apply_on_group(
  ConfigMap const &                     config_map,
  amr_hashmap_t const &                 amr_hashmap,
  orchard_key_view_t const &            orchard_keys,
  conformal_status_view_type const &    conformal_status,
  AMRMeshInfo const &                   amr_mesh_info,
  DataArrayBlock_t const &              Uin,
  DataArrayBlock_t const &              Uout,
  DataArrayGhostedBlock_t const &       q,
  DataArrayGhostedBlock_t const &       slopes_x,
  DataArrayGhostedBlock_t const &       slopes_y,
  DataArrayGhostedBlock_t const &       slopes_z,
  FieldMap<core::models::Hydro> const & fm,
  int32_t                               iOct_begin,
  int32_t                               num_octants,
  brick_size_t<dim> const &             brick_sizes,
  Kokkos::Array<bool, dim> const &      is_brick_periodic,
  HydroSettings const &                 hydro_settings,
  eos::EosWrapper<device_t> const &     eos,
  real_t                                dt)
{

  auto stencil_helper = StencilHelper_t(
    amr_hashmap, orchard_keys, slopes_x.block_size(), brick_sizes, is_brick_periodic);

  const auto gravity_enabled = config_map.getBool("gravity", "enabled", false);
  const auto gravity_field = get_uniform_gravity_vector<dim>(config_map);

  ComputeFluxesAndConservativeUpdateFunctor<dim, device_t> functor(
    config_map,
    stencil_helper,
    orchard_keys,
    conformal_status,
    amr_mesh_info,
    Uin,
    Uout,
    q,
    slopes_x,
    slopes_y,
    slopes_z,
    fm,
    iOct_begin,  // first index to process
    num_octants, // number of octants to process
    hydro_settings,
    eos,
    dt,
    gravity_enabled,
    gravity_field,
    TimeIntegratorConfig::get_time_integrator(config_map));

  // if we use atomic update, computations is flux-oriented
  // else computation is cell-oriented
  const auto nbIterations = num_octants * functor.nb_fluxes_per_leaf();

  // launch computation
  Kokkos::parallel_for("kalypsso::godunov_hydro::ComputeFluxesAndConservativeUpdateFunctor - group",
                       Kokkos::RangePolicy<exec_space, TagComputeAllQuadInGroup>(0, nbIterations),
                       functor);

} // apply_on_group

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
ComputeFluxesAndConservativeUpdateFunctor<dim, device_t>::apply_on_ghosts(
  ConfigMap const &                     config_map,
  amr_hashmap_t const &                 amr_hashmap,
  orchard_key_view_t const &            orchard_keys,
  conformal_status_view_type const &    conformal_status,
  AMRMeshInfo const &                   amr_mesh_info,
  DataArrayBlock_t const &              Uin,
  DataArrayBlock_t const &              Uout,
  DataArrayGhostedBlock_t const &       q,
  DataArrayGhostedBlock_t const &       slopes_x,
  DataArrayGhostedBlock_t const &       slopes_y,
  DataArrayGhostedBlock_t const &       slopes_z,
  FieldMap<core::models::Hydro> const & fm,
  brick_size_t<dim> const &             brick_sizes,
  Kokkos::Array<bool, dim> const &      is_brick_periodic,
  HydroSettings const &                 hydro_settings,
  eos::EosWrapper<device_t> const &     eos,
  real_t                                dt)
{

  auto stencil_helper = StencilHelper_t(
    amr_hashmap, orchard_keys, slopes_x.block_size(), brick_sizes, is_brick_periodic);

  const auto gravity_enabled = config_map.getBool("gravity", "enabled", false);
  const auto gravity_field = get_uniform_gravity_vector<dim>(config_map);

  ComputeFluxesAndConservativeUpdateFunctor<dim, device_t> functor(
    config_map,
    stencil_helper,
    orchard_keys,
    conformal_status,
    amr_mesh_info,
    Uin,
    Uout,
    q,
    slopes_x,
    slopes_y,
    slopes_z,
    fm,
    amr_mesh_info.local_num_mirrors(), // first ghost after all mirror quads
    amr_mesh_info.local_num_ghosts(),  // number of octants to process
    hydro_settings,
    eos,
    dt,
    gravity_enabled,
    gravity_field,
    TimeIntegratorConfig::get_time_integrator(config_map));

  // if we use atomic update, computations is flux-oriented
  // else computation is cell-oriented
  const auto nbIterations = amr_mesh_info.local_num_ghosts() * functor.nb_fluxes_per_leaf();

  // launch computation
  Kokkos::parallel_for(
    "kalypsso::godunov_hydro::ComputeFluxesAndConservativeUpdateFunctor - ghosts",
    Kokkos::RangePolicy<exec_space, TagComputeGhostQuad>(0, nbIterations),
    functor);

} // apply_on_ghosts

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION auto
ComputeFluxesAndConservativeUpdateFunctor<dim, device_t>::reconstruct_state_2d(
  const HydroState<2> & q,
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
ComputeFluxesAndConservativeUpdateFunctor<dim, device_t>::reconstruct_state_3d(
  const HydroState<3> & q,
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
ComputeFluxesAndConservativeUpdateFunctor<dim, device_t>::compute_fluxes_and_update_2d_group(
  index_t const & cell_index,
  index_t const & iOct_local) const
{

  auto const iOct_global = m_iOct_begin + iOct_local;

  auto const   coords = cellindex_to_coord<2>(cell_index, m_block_sizes_fluxes);
  auto const & i = coords[IX];
  auto const & j = coords[IY];

  // index to address slope arrays
  auto const is = i;
  auto const js = j;

  // index to address primitive variables arrays
  auto const iq = i;
  auto const jq = j;

  // get AMR level
  auto const level = orchard_key_t<2>::level(m_orchard_keys_device(iOct_global));

  // compute dS over dV in current cell and (larger) neighbor
  // a small cell will always update a large neighbor cell
  // Note: a larger neighbor has a volume 4 times larger than current cell volume
  auto const dx = compute_cell_length<2>(level, m_block_sizes[IX]) * m_scaling_factor;

  auto const dtdS_over_dV_cur = m_dt / dx;
  auto const dtdS_over_dV_neigh = m_dt / dx / 4;

  /*
   * reconstruct states on cells face and update
   */

  // get current location primitive variables state
  // note: primitive variables is a ghosted array with ghost width of 2
  auto qprim = get_prim_variables(iq, jq, iOct_local);

  /*
   * compute flux from left face along X dir and update both sides
   */
  {
    // get state in neighbor along X
    auto qprim_n = get_prim_variables(iq - 1, jq, iOct_local);

    // step 1 : reconstruct state in the left neighbor
    offsets_t<2> offsets{ 0.5, 0.0 };

    // reconstruct state in left neighbor (index relative to slopes array)
    auto qL = reconstruct_state_2d(
      qprim_n, is - 1, js, iOct_local, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur);

    // step 2 : reconstruct state in current cell
    offsets = { -0.5, 0.0 };

    // reconstruct state from current cell center to left interface
    auto qR =
      reconstruct_state_2d(qprim, is, js, iOct_local, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_hydro<2>(qL, qR, m_hydro_settings, m_eos);

    // step 4 : accumulate flux in current cell
    const auto flux_cur = flux * dtdS_over_dV_cur;

    const auto face_xmin_neighbor_is_coarser =
      conformal_face_status_t<dim>::face_xmin(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_COARSER;

    const auto face_xmin_neighbor_is_finer =
      conformal_face_status_t<dim>::face_xmin(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    // update on the RIGHT hand side
    // check for special cases :
    // - if neighbor on the left is larger, then we need to update current and neighbor cell
    // - if neighbor on the left is smaller, then we don't update current cell
    if (i < m_block_sizes[IX] and j < m_block_sizes[IY])
    {
      if (i == 0 and face_xmin_neighbor_is_finer)
      {
        // don't update current cell, neighbor will update us
      }
      else
      {
        state_add(m_Uout, i, j, iOct_global, flux_cur);
      }

      if (i == 0 and face_xmin_neighbor_is_coarser)
      {
        // we need to update neighbor (only if neighbor is not a ghost)
        const auto             flux_neigh = flux * dtdS_over_dV_neigh;
        constexpr shift_t<dim> shift{ -1, 0 };
        const auto             key_cur = m_orchard_keys_device(iOct_global);
        const CellLocation_t   cell_loc{ coords, key_cur, iOct_global, false };
        const auto cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

        const auto ijk = cell_loc_neigh.ijk;
        const auto iOct_neigh = cell_loc_neigh.iOct;

        state_sub(m_Uout, ijk[IX], ijk[IY], iOct_neigh, flux_neigh);
      }
    }

    const auto face_xmax_neighbor_is_coarser =
      conformal_face_status_t<dim>::face_xmax(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_COARSER;

    const auto face_xmax_neighbor_is_finer =
      conformal_face_status_t<dim>::face_xmax(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    // update on the LEFT hand side
    // check for special cases :
    // - if neighbor on the right is larger, then we need to update current and neighbor cell
    // - if neighbor on the left is smaller, then we don't update current cell
    if (i > 0 and j < m_block_sizes[IY])
    {
      if (i == m_block_sizes[IX] and face_xmax_neighbor_is_finer)
      {
        // don't update current cell, neighbor will update us
      }
      else
      {
        state_sub(m_Uout, i - 1, j, iOct_global, flux_cur);
      }

      if (i == m_block_sizes[IX] and face_xmax_neighbor_is_coarser)
      {
        // we need to update neighbor (only if neighbor is not a ghost)
        const auto             flux_neigh = flux * dtdS_over_dV_neigh;
        constexpr shift_t<dim> shift{ 1, 0 };
        const auto             key_cur = m_orchard_keys_device(iOct_global);
        const coord_t<2>       coords2{ i - 1, j }; // last cell inside block
        const CellLocation_t   cell_loc{ coords2, key_cur, iOct_global, false };
        const auto cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

        const auto ijk = cell_loc_neigh.ijk;
        const auto iOct_neigh = cell_loc_neigh.iOct;

        state_add(m_Uout, ijk[IX], ijk[IY], iOct_neigh, flux_neigh);
      }
    }
  }

  /*
   * compute flux from left face along Y dir and update both sides
   */
  {
    // get state in neighbor along Y
    auto qprim_n = get_prim_variables(iq, jq - 1, iOct_local);

    // step 1 : reconstruct state in the left neighbor
    offsets_t<2> offsets = { 0.0, 0.5 };

    // reconstruct "left" state
    auto qL = reconstruct_state_2d(
      qprim_n, is, js - 1, iOct_local, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur);

    // step 2 : reconstruct state in current cell
    offsets = { 0.0, -0.5 };

    // reconstruct state from current cell center to left interface
    auto qR =
      reconstruct_state_2d(qprim, is, js, iOct_local, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur);

    // swap IU / IV
    my_swap(qL[Hydro::IU], qL[Hydro::IV]);
    my_swap(qR[Hydro::IU], qR[Hydro::IV]);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_hydro<2>(qL, qR, m_hydro_settings, m_eos);

    my_swap(flux[Hydro::IU], flux[Hydro::IV]);

    // step 4 : accumulate flux in current cell
    const auto flux_cur = flux * dtdS_over_dV_cur;

    // update
    const auto face_ymin_neighbor_is_coarser =
      conformal_face_status_t<dim>::face_ymin(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_COARSER;

    const auto face_ymin_neighbor_is_finer =
      conformal_face_status_t<dim>::face_ymin(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    // update on the RIGHT hand side
    // check for special cases :
    // - if neighbor on the left is larger, then we need to update current and neighbor cell
    // - if neighbor on the left is smaller, then we don't update current cell
    if (i < m_block_sizes[IX] and j < m_block_sizes[IY])
    {
      if (j == 0 and face_ymin_neighbor_is_finer)
      {
        // don't update current cell, neighbor will update us
      }
      else
      {
        state_add(m_Uout, i, j, iOct_global, flux_cur);
      }

      if (j == 0 and face_ymin_neighbor_is_coarser)
      {
        // we need to update neighbor (only if neighbor is not a ghost)
        const auto             flux_neigh = flux * dtdS_over_dV_neigh;
        constexpr shift_t<dim> shift{ 0, -1 };
        const auto             key_cur = m_orchard_keys_device(iOct_global);
        const CellLocation_t   cell_loc{ coords, key_cur, iOct_global, false };
        const auto cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

        const auto ijk = cell_loc_neigh.ijk;
        const auto iOct_neigh = cell_loc_neigh.iOct;

        state_sub(m_Uout, ijk[IX], ijk[IY], iOct_neigh, flux_neigh);
      }
    }

    const auto face_ymax_neighbor_is_coarser =
      conformal_face_status_t<dim>::face_ymax(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_COARSER;

    const auto face_ymax_neighbor_is_finer =
      conformal_face_status_t<dim>::face_ymax(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    // update on the LEFT hand side
    // check for special cases :
    // - if neighbor on the right is larger, then we need to update current and neighbor cell
    // - if neighbor on the left is smaller, then we don't update current cell
    if (j > 0 and i < m_block_sizes[IX])
    {
      if (j == m_block_sizes[IY] and face_ymax_neighbor_is_finer)
      {
        // don't update current cell, neighbor will update us
      }
      else
      {
        state_sub(m_Uout, i, j - 1, iOct_global, flux_cur);
      }

      if (j == m_block_sizes[IY] and face_ymax_neighbor_is_coarser)
      {
        // we need to update neighbor (only if neighbor is not a ghost)
        const auto             flux_neigh = flux * dtdS_over_dV_neigh;
        constexpr shift_t<dim> shift{ 0, 1 };
        const auto             key_cur = m_orchard_keys_device(iOct_global);
        const coord_t<2>       coords2{ i, j - 1 }; // last cell inside block
        const CellLocation_t   cell_loc{ coords2, key_cur, iOct_global, false };
        const auto cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

        const auto ijk = cell_loc_neigh.ijk;
        const auto iOct_neigh = cell_loc_neigh.iOct;

        state_add(m_Uout, ijk[IX], ijk[IY], iOct_neigh, flux_neigh);
      }
    }
  }

} // compute_fluxes_and_update_2d_group

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 2), bool>>
KOKKOS_INLINE_FUNCTION void
ComputeFluxesAndConservativeUpdateFunctor<dim, device_t>::compute_fluxes_and_update_2d_ghost(
  index_t const & cell_index,
  index_t const & first_ghost,
  index_t const & iGhost) const
{

  KOKKOS_ASSERT(first_ghost + iGhost < m_q.num_quadrants() && "Invalid access to m_q view.");

  // iOct local : to be used when accessing array sized upon num_mirrors + num_ghosts like m_q
  // (aka m_Qghosted_mg in SolverHydroMusclBlock)
  auto const iOct_local = first_ghost + iGhost;

  // iOct global : to be used when accessing global arrays like U, U2 (here u_in, u_out), orchard
  // keys, hashmap, ...
  auto const iOct_global = m_amr_mesh_info.local_num_quadrants() + iGhost;

  auto const   coords = cellindex_to_coord<2>(cell_index, m_block_sizes_fluxes);
  auto const & i = coords[IX];
  auto const & j = coords[IY];

  // index to address slope arrays
  auto const is = i;
  auto const js = j;

  // index to address primitive variables arrays
  auto const iq = i;
  auto const jq = j;

  // get AMR level
  auto const level = orchard_key_t<2>::level(m_orchard_keys_device(iOct_global));

  // compute dS over dV in current cell and (larger) neighbor
  // a small cell will always update a large neighbor cell
  // Note: a larger neighbor has a volume 4 times larger than current cell volume
  auto const dx = compute_cell_length<2>(level, m_block_sizes[IX]) * m_scaling_factor;

  auto const dtdS_over_dV_cur = m_dt / dx;
  auto const dtdS_over_dV_neigh = m_dt / dx / 4;

  // get current location primitive variables state
  // note: primitive variables is a ghosted array with ghost width of 2
  const auto qprim = get_prim_variables(iq, jq, iOct_local);

  /*
   * Face XMIN - we only need to update a coarser neighbor
   */
  const auto face_xmin_neighbor_is_coarser =
    conformal_face_status_t<dim>::face_xmin(m_conformal_status(iOct_global)) ==
    conformal_neighbor_status::NEIGHBOR_IS_COARSER;

  if (face_xmin_neighbor_is_coarser and i == 0 and j < m_block_sizes[IY])
  {
    // check that neighbor is a owned quadrant (not a ghost)
    constexpr shift_t<dim> shift{ -1, 0 };
    const auto             key_cur = m_orchard_keys_device(iOct_global);
    const CellLocation_t   cell_loc{ coords, key_cur, iOct_global, false };
    const auto             cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

    const auto ijk = cell_loc_neigh.ijk;
    const auto iOct_neigh = cell_loc_neigh.iOct;

    if (is_owned_quadrant(iOct_neigh))
    {
      // now we can genuinely consider that an update is needed

      // get state in neighbor along X
      auto qprim_n = get_prim_variables(iq - 1, jq, iOct_local);

      // step 1 : reconstruct state in the left neighbor
      offsets_t<2> offsets{ 0.5, 0.0 };

      // reconstruct state in left neighbor (index relative to slopes array)
      auto qL = reconstruct_state_2d(
        qprim_n, is - 1, js, iGhost, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur);

      // step 2 : reconstruct state in current cell
      offsets = { -0.5, 0.0 };

      // reconstruct state from current cell center to left interface
      auto qR =
        reconstruct_state_2d(qprim, is, js, iGhost, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur);

      // step 3 : compute flux (Riemann solver)
      const auto flux = riemann_hydro<2>(qL, qR, m_hydro_settings, m_eos);

      // we need to update neighbor (only if neighbor is not a ghost)
      const auto flux_neigh = flux * dtdS_over_dV_neigh;

      state_sub(m_Uout, ijk[IX], ijk[IY], iOct_neigh, flux_neigh);
    }
  }

  /*
   * Face XMAX - we only need to update a coarser neighbor
   */
  const auto face_xmax_neighbor_is_coarser =
    conformal_face_status_t<dim>::face_xmax(m_conformal_status(iOct_global)) ==
    conformal_neighbor_status::NEIGHBOR_IS_COARSER;

  if (face_xmax_neighbor_is_coarser and i == m_block_sizes[IX] and j < m_block_sizes[IY])
  {
    // check that neighbor is a owned quadrant (not a ghost)
    constexpr shift_t<dim> shift{ 1, 0 };
    const auto             key_cur = m_orchard_keys_device(iOct_global);
    const coord_t<2>       coords2{ i - 1, j }; // last cell inside block
    const CellLocation_t   cell_loc{ coords2, key_cur, iOct_global, false };
    const auto             cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

    const auto ijk = cell_loc_neigh.ijk;
    const auto iOct_neigh = cell_loc_neigh.iOct;

    if (is_owned_quadrant(iOct_neigh))
    {
      // now we can genuinely consider that an update is needed

      // get state in neighbor along X
      auto qprim_n = get_prim_variables(iq - 1, jq, iOct_local);

      // step 1 : reconstruct state in the left neighbor
      offsets_t<2> offsets{ 0.5, 0.0 };

      // reconstruct state in left neighbor (index relative to slopes array)
      auto qL = reconstruct_state_2d(
        qprim_n, is - 1, js, iGhost, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur);

      // step 2 : reconstruct state in current cell
      offsets = { -0.5, 0.0 };

      // reconstruct state from current cell center to left interface
      auto qR =
        reconstruct_state_2d(qprim, is, js, iGhost, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur);

      // step 3 : compute flux (Riemann solver)
      const auto flux = riemann_hydro<2>(qL, qR, m_hydro_settings, m_eos);

      // we need to update neighbor (only if neighbor is not a ghost)
      const auto flux_neigh = flux * dtdS_over_dV_neigh;

      state_add(m_Uout, ijk[IX], ijk[IY], iOct_neigh, flux_neigh);
    }
  }

  /*
   * Face YMIN - we only need to update a coarser neighbor
   */
  const auto face_ymin_neighbor_is_coarser =
    conformal_face_status_t<dim>::face_ymin(m_conformal_status(iOct_global)) ==
    conformal_neighbor_status::NEIGHBOR_IS_COARSER;

  if (face_ymin_neighbor_is_coarser and i < m_block_sizes[IX] and j == 0)
  {
    // check that neighbor is a owned quadrant (not a ghost)
    constexpr shift_t<dim> shift{ 0, -1 };
    const auto             key_cur = m_orchard_keys_device(iOct_global);
    const CellLocation_t   cell_loc{ coords, key_cur, iOct_global, false };
    const auto             cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

    const auto ijk = cell_loc_neigh.ijk;
    const auto iOct_neigh = cell_loc_neigh.iOct;

    if (is_owned_quadrant(iOct_neigh))
    {
      // now we can genuinely consider that an update is needed

      // get state in neighbor along Y
      auto qprim_n = get_prim_variables(iq, jq - 1, iOct_local);

      // step 1 : reconstruct state in the left neighbor
      offsets_t<2> offsets = { 0.0, 0.5 };

      // reconstruct "left" state
      auto qL = reconstruct_state_2d(
        qprim_n, is, js - 1, iGhost, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur);

      // step 2 : reconstruct state in current cell
      offsets = { 0.0, -0.5 };

      auto qR =
        reconstruct_state_2d(qprim, is, js, iGhost, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur);

      // swap IU / IV
      my_swap(qL[Hydro::IU], qL[Hydro::IV]);
      my_swap(qR[Hydro::IU], qR[Hydro::IV]);

      // step 3 : compute flux (Riemann solver)
      auto flux = riemann_hydro<2>(qL, qR, m_hydro_settings, m_eos);

      my_swap(flux[Hydro::IU], flux[Hydro::IV]);

      const auto flux_neigh = flux * dtdS_over_dV_neigh;

      state_sub(m_Uout, ijk[IX], ijk[IY], iOct_neigh, flux_neigh);
    }
  }

  /*
   * Face YMAX - we only need to update a coarser neighbor
   */
  const auto face_ymax_neighbor_is_coarser =
    conformal_face_status_t<dim>::face_ymax(m_conformal_status(iOct_global)) ==
    conformal_neighbor_status::NEIGHBOR_IS_COARSER;

  if (face_ymax_neighbor_is_coarser and i < m_block_sizes[IX] and j == m_block_sizes[IY])
  {
    // check that neighbor is a owned quadrant (not a ghost)
    constexpr shift_t<dim> shift{ 0, 1 };
    const auto             key_cur = m_orchard_keys_device(iOct_global);
    const coord_t<2>       coords2{ i, j - 1 }; // last cell inside block
    const CellLocation_t   cell_loc{ coords2, key_cur, iOct_global, false };
    const auto             cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

    const auto ijk = cell_loc_neigh.ijk;
    const auto iOct_neigh = cell_loc_neigh.iOct;

    if (is_owned_quadrant(iOct_neigh))
    {
      // now we can genuinely consider that an update is needed

      // get state in neighbor along Y
      auto qprim_n = get_prim_variables(iq, jq - 1, iOct_local);

      // step 1 : reconstruct state in the left neighbor
      offsets_t<2> offsets = { 0.0, 0.5 };

      // reconstruct "left" state
      auto qL = reconstruct_state_2d(
        qprim_n, is, js - 1, iGhost, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur);

      // step 2 : reconstruct state in current cell
      offsets = { 0.0, -0.5 };

      auto qR =
        reconstruct_state_2d(qprim, is, js, iGhost, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur);

      // swap IU / IV
      my_swap(qL[Hydro::IU], qL[Hydro::IV]);
      my_swap(qR[Hydro::IU], qR[Hydro::IV]);

      // step 3 : compute flux (Riemann solver)
      auto flux = riemann_hydro<2>(qL, qR, m_hydro_settings, m_eos);

      my_swap(flux[Hydro::IU], flux[Hydro::IV]);

      const auto flux_neigh = flux * dtdS_over_dV_neigh;

      state_add(m_Uout, ijk[IX], ijk[IY], iOct_neigh, flux_neigh);
    }
  }

} // compute_fluxes_and_update_2d_ghost

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION void
ComputeFluxesAndConservativeUpdateFunctor<dim, device_t>::compute_fluxes_and_update_3d_group(
  const index_t & cell_index,
  const index_t & iOct_local) const
{
  auto const iOct_global = m_iOct_begin + iOct_local;

  auto const   coords = cellindex_to_coord<3>(cell_index, m_block_sizes_fluxes);
  auto const & i = coords[IX];
  auto const & j = coords[IY];
  auto const & k = coords[IZ];

  // index to address slope arrays
  auto const is = i;
  auto const js = j;
  auto const ks = k;

  // index to address primitive variables arrays
  auto const iq = i;
  auto const jq = j;
  auto const kq = k;

  // get AMR level
  auto const level = orchard_key_t<3>::level(m_orchard_keys_device(iOct_global));

  // compute dS over dV in current cell and (larger) neighbor
  // a small cell will always update a large neighbor cell
  // Note: a larger neighbor has a volume 8 times larger than current cell volume
  auto const dx = compute_cell_length<3>(level, m_block_sizes[IX]) * m_scaling_factor;

  auto const dtdS_over_dV_cur = m_dt / dx;
  auto const dtdS_over_dV_neigh = m_dt / dx / 8;


  /*
   * reconstruct states on cells face and update
   */

  // get current location primitive variables state
  // note: primitive variables is a ghosted array with ghost width of 2
  auto qprim = get_prim_variables(iq, jq, kq, iOct_local);

  /*
   * compute flux from left face along X dir
   */
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
                                   dtdS_over_dV_cur);

    // step 2 : reconstruct state in current cell
    offsets = { -0.5, 0.0, 0.0 };

    // reconstruct state from current cell center to left interface
    auto qR = reconstruct_state_3d(
      qprim, is, js, ks, iOct_local, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur, dtdS_over_dV_cur);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_hydro<3>(qL, qR, m_hydro_settings, m_eos);

    // step 4 : accumulate flux in current cell
    const auto flux_cur = flux * dtdS_over_dV_cur;

    const auto face_xmin_neighbor_is_coarser =
      conformal_face_status_t<dim>::face_xmin(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_COARSER;

    const auto face_xmin_neighbor_is_finer =
      conformal_face_status_t<dim>::face_xmin(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    // update on the RIGHT hand side
    // check for special cases :
    // - if neighbor on the left is larger, then we need to update current and neighbor cell
    // - if neighbor on the left is smaller, then we don't update current cell
    if (i < m_block_sizes[IX] and j < m_block_sizes[IY] and k < m_block_sizes[IZ])
    {
      if (i == 0 and face_xmin_neighbor_is_finer)
      {
        // don't update current cell, neighbor will update us
      }
      else
      {
        state_add(m_Uout, i, j, k, iOct_global, flux_cur);
      }

      if (i == 0 and face_xmin_neighbor_is_coarser)
      {
        // we need to update neighbor (only if neighbor is not a ghost)
        const auto             flux_neigh = flux * dtdS_over_dV_neigh;
        constexpr shift_t<dim> shift{ -1, 0, 0 };
        const auto             key_cur = m_orchard_keys_device(iOct_global);
        const CellLocation_t   cell_loc{ coords, key_cur, iOct_global, false };
        const auto cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

        const auto ijk = cell_loc_neigh.ijk;
        const auto iOct_neigh = cell_loc_neigh.iOct;

        state_sub(m_Uout, ijk[IX], ijk[IY], ijk[IZ], iOct_neigh, flux_neigh);
      }
    }

    const auto face_xmax_neighbor_is_coarser =
      conformal_face_status_t<dim>::face_xmax(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_COARSER;

    const auto face_xmax_neighbor_is_finer =
      conformal_face_status_t<dim>::face_xmax(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    // update on the LEFT hand side
    // check for special cases :
    // - if neighbor on the right is larger, then we need to update current and neighbor cell
    // - if neighbor on the left is smaller, then we don't update current cell
    if (i > 0 and j < m_block_sizes[IY] and k < m_block_sizes[IZ])
    {
      if (i == m_block_sizes[IX] and face_xmax_neighbor_is_finer)
      {
        // don't update current cell, neighbor will update us
      }
      else
      {
        state_sub(m_Uout, i - 1, j, k, iOct_global, flux_cur);
      }

      if (i == m_block_sizes[IX] and face_xmax_neighbor_is_coarser)
      {
        // we need to update neighbor (only if neighbor is not a ghost)
        const auto             flux_neigh = flux * dtdS_over_dV_neigh;
        constexpr shift_t<dim> shift{ 1, 0, 0 };
        const auto             key_cur = m_orchard_keys_device(iOct_global);
        const coord_t<3>       coords2{ i - 1, j, k }; // last cell inside block
        const CellLocation_t   cell_loc{ coords2, key_cur, iOct_global, false };
        const auto cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

        const auto ijk = cell_loc_neigh.ijk;
        const auto iOct_neigh = cell_loc_neigh.iOct;

        state_add(m_Uout, ijk[IX], ijk[IY], ijk[IZ], iOct_neigh, flux_neigh);
      }
    }
  } // end update along X

  /*
   * compute flux from left face along Y dir
   */
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
                                   dtdS_over_dV_cur);

    offsets = { 0.0, -0.5, 0.0 };

    auto qR = reconstruct_state_3d(
      qprim, is, js, ks, iOct_local, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur, dtdS_over_dV_cur);

    // swap IU / IV
    my_swap(qL[Hydro::IU], qL[Hydro::IV]);
    my_swap(qR[Hydro::IU], qR[Hydro::IV]);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_hydro<3>(qL, qR, m_hydro_settings, m_eos);

    my_swap(flux[Hydro::IU], flux[Hydro::IV]);

    // step 4 : accumulate flux in current cell
    const auto flux_cur = flux * dtdS_over_dV_cur;

    // update
    const auto face_ymin_neighbor_is_coarser =
      conformal_face_status_t<dim>::face_ymin(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_COARSER;

    const auto face_ymin_neighbor_is_finer =
      conformal_face_status_t<dim>::face_ymin(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    // update on the RIGHT hand side
    // check for special cases :
    // - if neighbor on the left is larger, then we need to update current and neighbor cell
    // - if neighbor on the left is smaller, then we don't update current cell
    if (i < m_block_sizes[IX] and j < m_block_sizes[IY] and k < m_block_sizes[IZ])
    {
      if (j == 0 and face_ymin_neighbor_is_finer)
      {
        // don't update current cell, neighbor will update us
      }
      else
      {
        state_add(m_Uout, i, j, k, iOct_global, flux_cur);
      }

      if (j == 0 and face_ymin_neighbor_is_coarser)
      {
        // we need to update neighbor (only if neighbor is not a ghost)
        const auto             flux_neigh = flux * dtdS_over_dV_neigh;
        constexpr shift_t<dim> shift{ 0, -1, 0 };
        const auto             key_cur = m_orchard_keys_device(iOct_global);
        const CellLocation_t   cell_loc{ coords, key_cur, iOct_global, false };
        const auto cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

        const auto ijk = cell_loc_neigh.ijk;
        const auto iOct_neigh = cell_loc_neigh.iOct;

        state_sub(m_Uout, ijk[IX], ijk[IY], ijk[IZ], iOct_neigh, flux_neigh);
      }
    }


    const auto face_ymax_neighbor_is_coarser =
      conformal_face_status_t<dim>::face_ymax(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_COARSER;

    const auto face_ymax_neighbor_is_finer =
      conformal_face_status_t<dim>::face_ymax(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    // update on the LEFT hand side
    // check for special cases :
    // - if neighbor on the right is larger, then we need to update current and neighbor cell
    // - if neighbor on the left is smaller, then we don't update current cell
    if (i < m_block_sizes[IX] and j > 0 and k < m_block_sizes[IZ])
    {
      if (j == m_block_sizes[IY] and face_ymax_neighbor_is_finer)
      {
        // don't update current cell, neighbor will update us
      }
      else
      {
        state_sub(m_Uout, i, j - 1, k, iOct_global, flux_cur);
      }

      if (j == m_block_sizes[IY] and face_ymax_neighbor_is_coarser)
      {
        // we need to update neighbor (only if neighbor is not a ghost)
        const auto             flux_neigh = flux * dtdS_over_dV_neigh;
        constexpr shift_t<dim> shift{ 0, 1, 0 };
        const auto             key_cur = m_orchard_keys_device(iOct_global);
        const coord_t<3>       coords2{ i, j - 1, k }; // last cell inside block
        const CellLocation_t   cell_loc{ coords2, key_cur, iOct_global, false };
        const auto cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

        const auto ijk = cell_loc_neigh.ijk;
        const auto iOct_neigh = cell_loc_neigh.iOct;

        state_add(m_Uout, ijk[IX], ijk[IY], ijk[IZ], iOct_neigh, flux_neigh);
      }
    }
  } // end update along Y

  /*
   * compute flux from left face along Z dir
   */
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
                                   dtdS_over_dV_cur);

    offsets = { 0.0, 0.0, -0.5 };

    auto qR = reconstruct_state_3d(
      qprim, is, js, ks, iOct_local, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur, dtdS_over_dV_cur);

    // swap IU / IW
    my_swap(qL[Hydro::IU], qL[Hydro::IW]);
    my_swap(qR[Hydro::IU], qR[Hydro::IW]);

    // step 3 : compute flux (Riemann solver)
    auto flux = riemann_hydro<3>(qL, qR, m_hydro_settings, m_eos);

    my_swap(flux[Hydro::IU], flux[Hydro::IW]);

    // step 4 : accumulate flux in current cell
    const auto flux_cur = flux * dtdS_over_dV_cur;

    // update
    const auto face_zmin_neighbor_is_coarser =
      conformal_face_status_t<dim>::face_zmin(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_COARSER;

    const auto face_zmin_neighbor_is_finer =
      conformal_face_status_t<dim>::face_zmin(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    // update on the RIGHT hand side
    // check for special cases :
    // - if neighbor on the left is larger, then we need to update current and neighbor cell
    // - if neighbor on the left is smaller, then we don't update current cell
    if (i < m_block_sizes[IX] and j < m_block_sizes[IY] and k < m_block_sizes[IZ])
    {
      if (k == 0 and face_zmin_neighbor_is_finer)
      {
        // don't update current cell, neighbor will update us
      }
      else
      {
        state_add(m_Uout, i, j, k, iOct_global, flux_cur);
      }

      if (k == 0 and face_zmin_neighbor_is_coarser)
      {
        // we need to update neighbor (only if neighbor is not a ghost)
        const auto             flux_neigh = flux * dtdS_over_dV_neigh;
        constexpr shift_t<dim> shift{ 0, 0, -1 };
        const auto             key_cur = m_orchard_keys_device(iOct_global);
        const CellLocation_t   cell_loc{ coords, key_cur, iOct_global, false };
        const auto cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

        const auto ijk = cell_loc_neigh.ijk;
        const auto iOct_neigh = cell_loc_neigh.iOct;

        state_sub(m_Uout, ijk[IX], ijk[IY], ijk[IZ], iOct_neigh, flux_neigh);
      }
    }


    const auto face_zmax_neighbor_is_coarser =
      conformal_face_status_t<dim>::face_zmax(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_COARSER;

    const auto face_zmax_neighbor_is_finer =
      conformal_face_status_t<dim>::face_zmax(m_conformal_status(iOct_global)) ==
      conformal_neighbor_status::NEIGHBOR_IS_FINER;

    // update on the LEFT hand side
    // check for special cases :
    // - if neighbor on the right is larger, then we need to update current and neighbor cell
    // - if neighbor on the left is smaller, then we don't update current cell
    if (i < m_block_sizes[IX] and j < m_block_sizes[IY] and k > 0)
    {
      if (k == m_block_sizes[IZ] and face_zmax_neighbor_is_finer)
      {
        // don't update current cell, neighbor will update us
      }
      else
      {
        state_sub(m_Uout, i, j, k - 1, iOct_global, flux_cur);
      }

      if (k == m_block_sizes[IZ] and face_zmax_neighbor_is_coarser)
      {
        // we need to update neighbor (only if neighbor is not a ghost)
        const auto             flux_neigh = flux * dtdS_over_dV_neigh;
        constexpr shift_t<dim> shift{ 0, 0, 1 };
        const auto             key_cur = m_orchard_keys_device(iOct_global);
        const coord_t<3>       coords2{ i, j, k - 1 }; // last cell inside block
        const CellLocation_t   cell_loc{ coords2, key_cur, iOct_global, false };
        const auto cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

        const auto ijk = cell_loc_neigh.ijk;
        const auto iOct_neigh = cell_loc_neigh.iOct;

        state_add(m_Uout, ijk[IX], ijk[IY], ijk[IZ], iOct_neigh, flux_neigh);
      }
    }
  } // end update along Z

} // compute_fluxes_and_update_3d_group

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
template <size_t dim_, std::enable_if_t<(dim_ == 3), bool>>
KOKKOS_INLINE_FUNCTION void
ComputeFluxesAndConservativeUpdateFunctor<dim, device_t>::compute_fluxes_and_update_3d_ghost(
  index_t const & cell_index,
  index_t const & first_ghost,
  index_t const & iGhost) const
{

  KOKKOS_ASSERT(first_ghost + iGhost < m_q.num_quadrants() && "Invalid access to m_q view.");

  // iOct local : to be used when accessing array sized upon num_mirrors + num_ghosts like m_q
  // (aka m_Qghosted_mg in SolverHydroMusclBlock)
  auto const iOct_local = first_ghost + iGhost;

  // iOct global : to be used when accessing global arrays like U, U2 (here u_in, u_out), orchard
  // keys, hashmap, ...
  auto const iOct_global = m_amr_mesh_info.local_num_quadrants() + iGhost;

  auto const   coords = cellindex_to_coord<3>(cell_index, m_block_sizes_fluxes);
  auto const & i = coords[IX];
  auto const & j = coords[IY];
  auto const & k = coords[IZ];

  // index to address slope arrays
  auto const is = i;
  auto const js = j;
  auto const ks = k;

  // index to address primitive variables arrays
  auto const iq = i;
  auto const jq = j;
  auto const kq = k;

  // get AMR level
  auto const level = orchard_key_t<3>::level(m_orchard_keys_device(iOct_global));

  // compute dS over dV in current cell and (larger) neighbor
  // a small cell will always update a large neighbor cell
  // Note: a larger neighbor has a volume 4 times larger than current cell volume
  auto const dx = compute_cell_length<3>(level, m_block_sizes[IX]) * m_scaling_factor;

  auto const dtdS_over_dV_cur = m_dt / dx;
  auto const dtdS_over_dV_neigh = m_dt / dx / 8;

  // get current location primitive variables state
  // note: primitive variables is a ghosted array with ghost width of 2
  const auto qprim = get_prim_variables(iq, jq, kq, iOct_local);

  /*
   * Face XMIN - we only need to update a coarser neighbor
   */
  const auto face_xmin_neighbor_is_coarser =
    conformal_face_status_t<dim>::face_xmin(m_conformal_status(iOct_global)) ==
    conformal_neighbor_status::NEIGHBOR_IS_COARSER;

  if (face_xmin_neighbor_is_coarser and i == 0 and j < m_block_sizes[IY] and k < m_block_sizes[IZ])
  {
    // check that neighbor is a owned quadrant (not a ghost)
    constexpr shift_t<dim> shift{ -1, 0, 0 };
    const auto             key_cur = m_orchard_keys_device(iOct_global);
    const CellLocation_t   cell_loc{ coords, key_cur, iOct_global, false };
    const auto             cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

    const auto ijk = cell_loc_neigh.ijk;
    const auto iOct_neigh = cell_loc_neigh.iOct;

    if (is_owned_quadrant(iOct_neigh))
    {
      // now we can genuinely consider that an update is needed

      // get state in neighbor along X
      auto qprim_n = get_prim_variables(iq - 1, jq, kq, iOct_local);

      // step 1 : reconstruct state in the left neighbor
      offsets_t<3> offsets{ 0.5, 0.0, 0.0 };

      // reconstruct state in left neighbor (index relative to slopes array)
      auto qL = reconstruct_state_3d(qprim_n,
                                     is - 1,
                                     js,
                                     ks,
                                     iGhost,
                                     offsets,
                                     dtdS_over_dV_cur,
                                     dtdS_over_dV_cur,
                                     dtdS_over_dV_cur);

      // step 2 : reconstruct state in current cell
      offsets = { -0.5, 0.0, 0.0 };

      // reconstruct state from current cell center to left interface
      auto qR = reconstruct_state_3d(
        qprim, is, js, ks, iGhost, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur, dtdS_over_dV_cur);

      // step 3 : compute flux (Riemann solver)
      const auto flux = riemann_hydro<3>(qL, qR, m_hydro_settings, m_eos);

      // we need to update neighbor (only if neighbor is not a ghost)
      const auto flux_neigh = flux * dtdS_over_dV_neigh;

      state_sub(m_Uout, ijk[IX], ijk[IY], ijk[IZ], iOct_neigh, flux_neigh);
    }
  } // end face XMIN

  /*
   * Face XMAX - we only need to update a coarser neighbor
   */
  const auto face_xmax_neighbor_is_coarser =
    conformal_face_status_t<dim>::face_xmax(m_conformal_status(iOct_global)) ==
    conformal_neighbor_status::NEIGHBOR_IS_COARSER;

  if (face_xmax_neighbor_is_coarser and i == m_block_sizes[IX] and j < m_block_sizes[IY] and
      k < m_block_sizes[IZ])
  {
    // check that neighbor is a owned quadrant (not a ghost)
    constexpr shift_t<dim> shift{ 1, 0, 0 };
    const auto             key_cur = m_orchard_keys_device(iOct_global);
    const coord_t<3>       coords2{ i - 1, j, k }; // last cell inside block
    const CellLocation_t   cell_loc{ coords2, key_cur, iOct_global, false };
    const auto             cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

    const auto ijk = cell_loc_neigh.ijk;
    const auto iOct_neigh = cell_loc_neigh.iOct;

    if (is_owned_quadrant(iOct_neigh))
    {
      // now we can genuinely consider that an update is needed

      // get state in neighbor along X
      auto qprim_n = get_prim_variables(iq - 1, jq, kq, iOct_local);

      // step 1 : reconstruct state in the left neighbor
      offsets_t<3> offsets{ 0.5, 0.0, 0.0 };

      // reconstruct state in left neighbor (index relative to slopes array)
      auto qL = reconstruct_state_3d(qprim_n,
                                     is - 1,
                                     js,
                                     ks,
                                     iGhost,
                                     offsets,
                                     dtdS_over_dV_cur,
                                     dtdS_over_dV_cur,
                                     dtdS_over_dV_cur);

      // step 2 : reconstruct state in current cell
      offsets = { -0.5, 0.0, 0.0 };

      // reconstruct state from current cell center to left interface
      auto qR = reconstruct_state_3d(
        qprim, is, js, ks, iGhost, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur, dtdS_over_dV_cur);

      // step 3 : compute flux (Riemann solver)
      const auto flux = riemann_hydro<3>(qL, qR, m_hydro_settings, m_eos);

      // we need to update neighbor (only if neighbor is not a ghost)
      const auto flux_neigh = flux * dtdS_over_dV_neigh;

      state_add(m_Uout, ijk[IX], ijk[IY], ijk[IZ], iOct_neigh, flux_neigh);
    }
  } // end face XMAX

  /*
   * Face YMIN - we only need to update a coarser neighbor
   */
  const auto face_ymin_neighbor_is_coarser =
    conformal_face_status_t<dim>::face_ymin(m_conformal_status(iOct_global)) ==
    conformal_neighbor_status::NEIGHBOR_IS_COARSER;

  if (face_ymin_neighbor_is_coarser and i < m_block_sizes[IX] and j == 0 and k < m_block_sizes[IZ])
  {
    // check that neighbor is a owned quadrant (not a ghost)
    constexpr shift_t<dim> shift{ 0, -1, 0 };
    const auto             key_cur = m_orchard_keys_device(iOct_global);
    const CellLocation_t   cell_loc{ coords, key_cur, iOct_global, false };
    const auto             cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

    const auto ijk = cell_loc_neigh.ijk;
    const auto iOct_neigh = cell_loc_neigh.iOct;

    if (is_owned_quadrant(iOct_neigh))
    {
      // now we can genuinely consider that an update is needed

      // get state in neighbor along Y
      auto qprim_n = get_prim_variables(iq, jq - 1, kq, iOct_local);

      // step 1 : reconstruct state in the left neighbor
      offsets_t<3> offsets = { 0.0, 0.5, 0.0 };

      // reconstruct "left" state
      auto qL = reconstruct_state_3d(qprim_n,
                                     is,
                                     js - 1,
                                     ks,
                                     iGhost,
                                     offsets,
                                     dtdS_over_dV_cur,
                                     dtdS_over_dV_cur,
                                     dtdS_over_dV_cur);

      // step 2 : reconstruct state in current cell
      offsets = { 0.0, -0.5, 0.0 };

      auto qR = reconstruct_state_3d(
        qprim, is, js, ks, iGhost, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur, dtdS_over_dV_cur);

      // swap IU / IV
      my_swap(qL[Hydro::IU], qL[Hydro::IV]);
      my_swap(qR[Hydro::IU], qR[Hydro::IV]);

      // step 3 : compute flux (Riemann solver)
      auto flux = riemann_hydro<3>(qL, qR, m_hydro_settings, m_eos);

      my_swap(flux[Hydro::IU], flux[Hydro::IV]);

      const auto flux_neigh = flux * dtdS_over_dV_neigh;

      state_sub(m_Uout, ijk[IX], ijk[IY], ijk[IZ], iOct_neigh, flux_neigh);
    }
  } // end face YMIN

  /*
   * Face YMAX - we only need to update a coarser neighbor
   */
  const auto face_ymax_neighbor_is_coarser =
    conformal_face_status_t<dim>::face_ymax(m_conformal_status(iOct_global)) ==
    conformal_neighbor_status::NEIGHBOR_IS_COARSER;

  if (face_ymax_neighbor_is_coarser and i < m_block_sizes[IX] and j == m_block_sizes[IY] and
      k < m_block_sizes[IZ])
  {
    // check that neighbor is a owned quadrant (not a ghost)
    constexpr shift_t<dim> shift{ 0, 1, 0 };
    const auto             key_cur = m_orchard_keys_device(iOct_global);
    const coord_t<3>       coords2{ i, j - 1, k }; // last cell inside block
    const CellLocation_t   cell_loc{ coords2, key_cur, iOct_global, false };
    const auto             cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

    const auto ijk = cell_loc_neigh.ijk;
    const auto iOct_neigh = cell_loc_neigh.iOct;

    if (is_owned_quadrant(iOct_neigh))
    {
      // now we can genuinely consider that an update is needed

      // get state in neighbor along Y
      auto qprim_n = get_prim_variables(iq, jq - 1, kq, iOct_local);

      // step 1 : reconstruct state in the left neighbor
      offsets_t<3> offsets = { 0.0, 0.5, 0.0 };

      // reconstruct "left" state
      auto qL = reconstruct_state_3d(qprim_n,
                                     is,
                                     js - 1,
                                     ks,
                                     iGhost,
                                     offsets,
                                     dtdS_over_dV_cur,
                                     dtdS_over_dV_cur,
                                     dtdS_over_dV_cur);

      // step 2 : reconstruct state in current cell
      offsets = { 0.0, -0.5, 0.0 };

      auto qR = reconstruct_state_3d(qprim,
                                     is,
                                     js,
                                     ks,
                                     iGhost,
                                     offsets,
                                     dtdS_over_dV_cur,
                                     dtdS_over_dV_cur,
                                     dtdS_over_dV_cur); // swap IU / IV
      my_swap(qL[Hydro::IU], qL[Hydro::IV]);
      my_swap(qR[Hydro::IU], qR[Hydro::IV]);

      // step 3 : compute flux (Riemann solver)
      auto flux = riemann_hydro<3>(qL, qR, m_hydro_settings, m_eos);

      my_swap(flux[Hydro::IU], flux[Hydro::IV]);

      const auto flux_neigh = flux * dtdS_over_dV_neigh;

      state_add(m_Uout, ijk[IX], ijk[IY], ijk[IZ], iOct_neigh, flux_neigh);
    }
  } // end face YMAX

  /*
   * Face ZMIN - we only need to update a coarser neighbor
   */
  const auto face_zmin_neighbor_is_coarser =
    conformal_face_status_t<dim>::face_zmin(m_conformal_status(iOct_global)) ==
    conformal_neighbor_status::NEIGHBOR_IS_COARSER;

  if (face_zmin_neighbor_is_coarser and i < m_block_sizes[IX] and j < m_block_sizes[IY] and k == 0)
  {
    // check that neighbor is a owned quadrant (not a ghost)
    constexpr shift_t<dim> shift{ 0, 0, -1 };
    const auto             key_cur = m_orchard_keys_device(iOct_global);
    const CellLocation_t   cell_loc{ coords, key_cur, iOct_global, false };
    const auto             cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

    const auto ijk = cell_loc_neigh.ijk;
    const auto iOct_neigh = cell_loc_neigh.iOct;

    if (is_owned_quadrant(iOct_neigh))
    {
      // now we can genuinely consider that an update is needed

      // get state in neighbor along Y
      auto qprim_n = get_prim_variables(iq, jq, kq - 1, iOct_local);

      // step 1 : reconstruct state in the left neighbor
      offsets_t<3> offsets = { 0.0, 0.0, 0.5 };

      // reconstruct "left" state
      auto qL = reconstruct_state_3d(qprim_n,
                                     is,
                                     js,
                                     ks - 1,
                                     iGhost,
                                     offsets,
                                     dtdS_over_dV_cur,
                                     dtdS_over_dV_cur,
                                     dtdS_over_dV_cur);

      // step 2 : reconstruct state in current cell
      offsets = { 0.0, 0.0, -0.5 };

      auto qR = reconstruct_state_3d(
        qprim, is, js, ks, iGhost, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur, dtdS_over_dV_cur);

      // swap IU / IW
      my_swap(qL[Hydro::IU], qL[Hydro::IW]);
      my_swap(qR[Hydro::IU], qR[Hydro::IW]);

      // step 3 : compute flux (Riemann solver)
      auto flux = riemann_hydro<3>(qL, qR, m_hydro_settings, m_eos);

      my_swap(flux[Hydro::IU], flux[Hydro::IW]);

      const auto flux_neigh = flux * dtdS_over_dV_neigh;

      state_sub(m_Uout, ijk[IX], ijk[IY], ijk[IZ], iOct_neigh, flux_neigh);
    }
  } // end face ZMIN

  /*
   * Face ZMAX - we only need to update a coarser neighbor
   */
  const auto face_zmax_neighbor_is_coarser =
    conformal_face_status_t<dim>::face_zmax(m_conformal_status(iOct_global)) ==
    conformal_neighbor_status::NEIGHBOR_IS_COARSER;

  if (face_zmax_neighbor_is_coarser and i < m_block_sizes[IX] and j < m_block_sizes[IY] and
      k == m_block_sizes[IZ])
  {
    // check that neighbor is a owned quadrant (not a ghost)
    constexpr shift_t<dim> shift{ 0, 0, 1 };
    const auto             key_cur = m_orchard_keys_device(iOct_global);
    const coord_t<3>       coords2{ i, j, k - 1 }; // last cell inside block
    const CellLocation_t   cell_loc{ coords2, key_cur, iOct_global, false };
    const auto             cell_loc_neigh = m_stencil_helper.getNeighLocCoarser(cell_loc, shift);

    const auto ijk = cell_loc_neigh.ijk;
    const auto iOct_neigh = cell_loc_neigh.iOct;

    if (is_owned_quadrant(iOct_neigh))
    {
      // now we can genuinely consider that an update is needed

      // get state in neighbor along Y
      auto qprim_n = get_prim_variables(iq, jq, kq - 1, iOct_local);

      // step 1 : reconstruct state in the left neighbor
      offsets_t<3> offsets = { 0.0, 0.0, 0.5 };

      // reconstruct "left" state
      auto qL = reconstruct_state_3d(qprim_n,
                                     is,
                                     js,
                                     ks - 1,
                                     iGhost,
                                     offsets,
                                     dtdS_over_dV_cur,
                                     dtdS_over_dV_cur,
                                     dtdS_over_dV_cur);

      // step 2 : reconstruct state in current cell
      offsets = { 0.0, 0.0, -0.5 };

      auto qR = reconstruct_state_3d(
        qprim, is, js, ks, iGhost, offsets, dtdS_over_dV_cur, dtdS_over_dV_cur, dtdS_over_dV_cur);

      // swap IU / IW
      my_swap(qL[Hydro::IU], qL[Hydro::IW]);
      my_swap(qR[Hydro::IU], qR[Hydro::IW]);

      // step 3 : compute flux (Riemann solver)
      auto flux = riemann_hydro<3>(qL, qR, m_hydro_settings, m_eos);

      my_swap(flux[Hydro::IU], flux[Hydro::IW]);

      const auto flux_neigh = flux * dtdS_over_dV_neigh;

      state_add(m_Uout, ijk[IX], ijk[IY], ijk[IZ], iOct_neigh, flux_neigh);
    }
  } // end face ZMAX

} // compute_fluxes_and_update_3d_ghost

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeFluxesAndConservativeUpdateFunctor<dim, device_t>::operator()(
  TagComputeAllQuadInGroup const &,
  const int64_t & global_index) const
{

  // retrieve local octant index (local to group)
  auto const iOct_local = static_cast<int32_t>(global_index / m_nbFluxesPerLeaf);
  auto const cell_index = static_cast<int32_t>(global_index - iOct_local * m_nbFluxesPerLeaf);

  if constexpr (dim == 2)
    compute_fluxes_and_update_2d_group(cell_index, iOct_local);
  else if constexpr (dim == 3)
    compute_fluxes_and_update_3d_group(cell_index, iOct_local);

} // operator () - TagComputeAllQuadInGroup

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
ComputeFluxesAndConservativeUpdateFunctor<dim, device_t>::operator()(
  TagComputeGhostQuad const &,
  const int64_t & global_index) const
{

  // just creating an alias for code clarity
  // iOct_begin is actually the index of the first ghost quadrant
  auto const & first_ghost = m_iOct_begin;

  // retrieve ghost index - iGhost should take values between 0 and m_num_octants-1 and
  // in this context, m_num_octants must be equal to the number of ghosts quadrants
  int32_t       iGhost = static_cast<int32_t>(global_index / m_nbFluxesPerLeaf);
  const int32_t cell_index = static_cast<int32_t>(global_index - iGhost * m_nbFluxesPerLeaf);

  if constexpr (dim == 2)
    compute_fluxes_and_update_2d_ghost(cell_index, first_ghost, iGhost);
  else if constexpr (dim == 3)
    compute_fluxes_and_update_3d_ghost(cell_index, first_ghost, iGhost);

} // operator () - TagComputeAllQuadInGroup

// explicit template instantiation
template class ComputeFluxesAndConservativeUpdateFunctor<2, kalypsso::DefaultDevice>;
template class ComputeFluxesAndConservativeUpdateFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso
