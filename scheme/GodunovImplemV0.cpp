// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file GodunovImplemV0.cpp
 *
 * Godunov time integration implementation detail version v0.
 */
#include <godunov_hydro/scheme/GodunovImplemV0.h>

// Compute functors
#include <godunov_hydro/scheme/ConvertToPrimitivesVariablesFunctor.h>
#include <godunov_hydro/scheme/ComputeLimitedSlopesFunctor.h>
#include <godunov_hydro/scheme/ComputeFluxesAndStoreFunctor.h>
#include <godunov_hydro/scheme/ComputeViscousFluxesAndStoreFunctor.h>
#include <godunov_hydro/scheme/ReadFluxesAndConservativeUpdateFunctor.h>

// profiling colors
#include <godunov_hydro/profiling.h>

namespace kalypsso
{
namespace godunov_hydro
{

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV0<dim, device_t>::resize_auxiliary_data()
{
  const auto num_owned = this->m_mesh_map.get_amr_mesh_info().local_num_quadrants();
  const auto num_ghosts = this->m_amr_mesh.local_num_ghosts();
  const auto num_mirrors = this->m_amr_mesh.local_num_mirrors();

  m_Q_ghosted.resize(num_owned + num_ghosts);

  // note: m_Q_ghosted_mg is sized with the number of mirror + number of ghost quadrants,
  // because ghost quadrant will be populated with MeshGhostsExchanger
  m_Q_ghosted_mg.resize(num_mirrors + num_ghosts);

  m_Slopes_x.resize(num_owned + num_ghosts);
  m_Slopes_y.resize(num_owned + num_ghosts);
  if constexpr (dim == 3)
    m_Slopes_z.resize(num_owned + num_ghosts);

  // Flux is resized large enough to store fluxes for owned + ghost blocks
  // we don't need to compute in outside quad, since outside quad are at the same level of AMR than
  // the neighbor directly inside
  m_Fluxes.resize(num_owned + num_ghosts);

} // GodunovImplemV0<dim, device_t>::resize_auxiliary_data

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
uint64_t
GodunovImplemV0<dim, device_t>::total_mem_size_in_bytes()
{
  uint64_t total = 0;
  total += m_Q_ghosted.allocated_size_in_bytes();
  total += m_Q_ghosted_mg.allocated_size_in_bytes();
  total += m_Slopes_x.allocated_size_in_bytes();
  total += m_Slopes_y.allocated_size_in_bytes();
  total += m_Slopes_z.allocated_size_in_bytes();
  total += m_Fluxes.allocated_size_in_bytes();

  return total;
} // GodunovImplemV0<dim, device_t>::total_mem_size_in_bytes

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV0<dim, device_t>::do_time_step(DataArrayBlock_t U, DataArrayBlock_t U2, real_t dt)
{

  /*
   * Note about conservativity:
   *
   * To do a conservative update, we need to ensure that at the interface between two neighbors
   * quads living at different AMR levels, the coarse quad get updated using the fluxes computed by
   * the fine quad.
   *
   * Currently we compute fluxes in all quadrants (owned + ghosts).
   * Computing fluxes in ghost quadrants is only need in case a owned quadrant has a face neighbor
   * at finer AMR level that is ghost; in that case the flux in the ghost must be used to update
   * current owned quadrant.
   */

  // start main computation
  KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME);

  // convert conservative variable into primitives ones m_U ==> m_Q_ghosted_mg
  // also perform a half time step gravity predictor
  this->convert_to_primitives_in_mirror_quads(U);

  // do mpi comm to populate ghosts quads in m_Q_ghosted_mg (see MeshGhostExchanger)
  this->mpi_exchange_mirrors_and_ghosts(m_Q_ghosted_mg);

  // update primitive variables in owned quadrants + copy ghost quads
  this->convert_to_primitives(U);

  this->compute_limited_slopes_in_owned_and_ghosts();

  //
  // compute flux along X, store and update
  //
  this->compute_fluxes_and_store_in_owned_and_ghosts(dt, IX);
  this->read_fluxes_and_update_in_owned(U2, dt, IX);


  //
  // compute flux along Y, store and update
  //
  this->compute_fluxes_and_store_in_owned_and_ghosts(dt, IY);
  this->read_fluxes_and_update_in_owned(U2, dt, IY);


  //
  // compute flux along Z, store and update
  //
  if constexpr (dim == 3)
  {
    this->compute_fluxes_and_store_in_owned_and_ghosts(dt, IZ);
    this->read_fluxes_and_update_in_owned(U2, dt, IZ);
  }

  if (this->m_viscosity.enabled)
  {
    this->compute_viscous_fluxes_and_store_in_owned_and_ghosts(dt, IX);
    this->read_fluxes_and_update_in_owned(U2, dt, IX);

    this->compute_viscous_fluxes_and_store_in_owned_and_ghosts(dt, IY);
    this->read_fluxes_and_update_in_owned(U2, dt, IY);

    if constexpr (dim == 3)
    {
      this->compute_viscous_fluxes_and_store_in_owned_and_ghosts(dt, IZ);
      this->read_fluxes_and_update_in_owned(U2, dt, IZ);
    }
  }

  //
  // Add gravity source term when enabled
  //
  const auto gravity_enabled = this->m_config_map.getBool("gravity", "enabled", false);
  if (gravity_enabled)
    this->add_gravity_source_term(U, U2, dt);

} // GodunovImplemV0<dim, device_t>::do_time_step

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV0<dim, device_t>::convert_to_primitives_in_mirror_quads(DataArrayBlock_t U)
{

  KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME_CONV_PRIM);

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  const auto & fm = this->m_hydro.get_fieldmap();

  // compute primitive variables in owned mirror blocks (U must have MPI ghost up to date)
  ConvertToPrimitivesVariablesFunctor<dim, device_t>::apply_in_mirrors(
    this->m_config_map,
    this->m_mesh_map.hashmap(),
    this->m_mesh_map.orchard_keys(),
    this->m_mesh_map.mirror_orchard_keys(),
    this->m_mesh_map.get_amr_mesh_info(),
    U,
    m_Q_ghosted_mg,
    fm,
    this->m_brick_sizes,
    this->m_is_brick_periodic,
    this->m_hydro_settings,
    this->m_eos);

} // GodunovImplemV0<dim, device_t>::convert_to_primitives_in_mirror_quads

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV0<dim, device_t>::convert_to_primitives(DataArrayBlock_t U)
{

  KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME_CONV_PRIM);

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  const auto & fm = this->m_hydro.get_fieldmap();

  //
  // step 1: convert to primitive variables in owned quadrants
  //

  const auto num_quads_owned = this->m_mesh_map.get_amr_mesh_info().local_num_quadrants();
  const auto num_quads_mirrors = this->m_amr_mesh.local_num_mirrors();
  const auto num_quads_ghosts = this->m_amr_mesh.local_num_ghosts();

  // compute primitive variables in all owned blocks (U must have MPI ghost up to date)
  ConvertToPrimitivesVariablesFunctor<dim, device_t>::apply_on_group(
    this->m_config_map,
    this->m_mesh_map.hashmap(),
    this->m_mesh_map.orchard_keys(),
    this->m_mesh_map.get_amr_mesh_info(),
    0,
    num_quads_owned,
    U,
    m_Q_ghosted,
    fm,
    this->m_brick_sizes,
    this->m_is_brick_periodic,
    this->m_hydro_settings,
    this->m_eos);

  //
  // step 2: copy primitives variables in ghost
  //
  const auto num_ghosted_cells = m_Q_ghosted.num_cells();
  const auto num_vars = m_Q_ghosted.num_vars();

  auto range_in = std::pair<std::size_t, std::size_t>(
    num_ghosted_cells * num_vars * num_quads_mirrors,
    num_ghosted_cells * num_vars * (num_quads_mirrors + num_quads_ghosts));

  auto data_in = Kokkos::subview(m_Q_ghosted_mg.flat_view(), range_in);

  auto range_out = std::pair<std::size_t, std::size_t>(
    num_ghosted_cells * num_vars * num_quads_owned,
    num_ghosted_cells * num_vars * (num_quads_owned + num_quads_ghosts));

  auto data_out = Kokkos::subview(m_Q_ghosted.flat_view(), range_out);

  Kokkos::deep_copy(data_out, data_in);

} // GodunovImplemV0<dim, device_t>::convert_to_primitives

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV0<dim, device_t>::compute_limited_slopes_in_owned_and_ghosts()
{

  KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME_SLOPES);

  const auto num_quadrants_owned = this->m_mesh_map.get_amr_mesh_info().local_num_quadrants();
  const auto num_quadrants_ghost = this->m_mesh_map.get_amr_mesh_info().local_num_ghosts();

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  const auto & fm = this->m_hydro.get_fieldmap();

  // compute limited slopes in all quadrants (owned + ghost)
  ComputeLimitedSlopesFunctor<dim, device_t>::apply_on_group(m_Q_ghosted,
                                                             m_Slopes_x,
                                                             m_Slopes_y,
                                                             m_Slopes_z,
                                                             fm,
                                                             num_quadrants_owned +
                                                               num_quadrants_ghost,
                                                             this->m_hydro_settings);

} // GodunovImplemV0<dim, device_t>::compute_limited_slopes_in_owned_and_ghosts

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV0<dim, device_t>::compute_fluxes_and_store_in_owned_and_ghosts(real_t dt,
                                                                             int    direction)
{

  KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME_COMPUTE_FLUXES);

  const auto num_quadrants_owned = this->m_mesh_map.get_amr_mesh_info().local_num_quadrants();
  const auto num_quadrants_ghost = this->m_mesh_map.get_amr_mesh_info().local_num_ghosts();

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  const auto & fm = this->m_hydro.get_fieldmap();

  // reshape flux to be a flux array in given direction
  auto flux_block_sizes = m_Q_ghosted.block_size();
  flux_block_sizes[static_cast<size_t>(direction)]++;
  m_Fluxes.reshape(flux_block_sizes);

  // compute fluxes and update all quadrants in a group of quadrants
  ComputeFluxesAndStoreFunctor<dim, device_t>::apply(this->m_config_map,
                                                     this->m_mesh_map.orchard_keys(),
                                                     this->m_mesh_map.get_amr_mesh_info(),
                                                     m_Fluxes,
                                                     m_Q_ghosted,
                                                     m_Slopes_x,
                                                     m_Slopes_y,
                                                     m_Slopes_z,
                                                     fm,
                                                     0,
                                                     num_quadrants_owned + num_quadrants_ghost,
                                                     direction,
                                                     this->m_hydro_settings,
                                                     this->m_eos,
                                                     this->m_viscosity,
                                                     dt);

} // GodunovImplemV0<dim, device_t>::compute_fluxes_and_store_in_owned_and_ghosts

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV0<dim, device_t>::compute_viscous_fluxes_and_store_in_owned_and_ghosts(real_t dt,
                                                                                     int direction)
{
  KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME_COMPUTE_VISCOUS_FLUXES);

  const auto num_quadrants_owned = this->m_mesh_map.get_amr_mesh_info().local_num_quadrants();
  const auto num_quadrants_ghost = this->m_mesh_map.get_amr_mesh_info().local_num_ghosts();

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  const auto & fm = this->m_hydro.get_fieldmap();

  // reshape flux to be a flux array in given direction
  auto flux_block_sizes = m_Q_ghosted.block_size();
  flux_block_sizes[static_cast<size_t>(direction)]++;
  m_Fluxes.reshape(flux_block_sizes);

  // compute fluxes and update all quadrants in a group of quadrants
  ComputeViscousFluxesAndStoreFunctor<dim, device_t>::apply(this->m_config_map,
                                                            this->m_mesh_map.orchard_keys(),
                                                            this->m_mesh_map.get_amr_mesh_info(),
                                                            m_Fluxes,
                                                            m_Q_ghosted,
                                                            fm,
                                                            0,
                                                            num_quadrants_owned +
                                                              num_quadrants_ghost,
                                                            direction,
                                                            this->m_viscosity,
                                                            dt);

} // GodunovImplemV0<dim, device_t>::compute_viscous_fluxes_and_store_in_owned_and_ghosts

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV0<dim, device_t>::read_fluxes_and_update_in_owned(DataArrayBlock_t u_out,
                                                                real_t           dt,
                                                                int              direction)
{

  KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME_UPDATE);

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  const auto & fm = this->m_hydro.get_fieldmap();

  // check flux array sizes
  {
    [[maybe_unused]] auto flux_block_sizes = m_Q_ghosted.block_size();
    flux_block_sizes[static_cast<size_t>(direction)]++;
    assertm(flux_block_sizes == m_Fluxes.shape(), "Flux array has incompatible shape.");
  }

  // compute incoming fluxes in all ghost quadrants and perform conservative update only at
  // interface between ghost and owned quadrants (in case the owned quadrant is coarser than the
  // ghost quadrant)
  ReadFluxesAndConservativeUpdateFunctor<dim, device_t>::apply(this->m_config_map,
                                                               this->m_mesh_map.hashmap(),
                                                               this->m_mesh_map.orchard_keys(),
                                                               this->m_mesh_map.conformal_status(),
                                                               this->m_mesh_map.get_amr_mesh_info(),
                                                               u_out,
                                                               m_Fluxes,
                                                               fm,
                                                               direction,
                                                               this->m_brick_sizes,
                                                               this->m_is_brick_periodic,
                                                               this->m_hydro_settings,
                                                               dt);

} // GodunovImplemV0<dim, device_t>::read_fluxes_and_update_in_owned

// explicit template instantiation
template class GodunovImplemV0<2, kalypsso::DefaultDevice>;
template class GodunovImplemV0<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso
