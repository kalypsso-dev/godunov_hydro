// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file GodunovImplemV2.cpp
 *
 * Godunov time integration implementation detail version v2.
 */
#include <godunov_hydro/scheme/GodunovImplemV2.h>

// Compute functors
#include <godunov_hydro/scheme/ConvertToPrimitivesVariablesFunctor.h>
#include <godunov_hydro/scheme/ComputeLimitedSlopesFunctor.h>
#include <godunov_hydro/scheme/ComputeFluxesAndConservativeUpdateFunctor.h>

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
GodunovImplemV2<dim, device_t>::resize_auxiliary_data()
{
  // note: m_Q_ghosted_mg is sized with the number of mirror + number of ghost quadrants,
  // because ghost quadrant will be populated with MeshGhostsExchanger

  // clang-format off
  m_Q_ghosted_mg.resize(this->m_amr_mesh.local_num_mirrors() +
                        this->m_amr_mesh.local_num_ghosts());
  // clang-format on

  m_Slopes_x_g.resize(this->m_amr_mesh.local_num_ghosts());
  m_Slopes_y_g.resize(this->m_amr_mesh.local_num_ghosts());
  if constexpr (dim == 3)
    m_Slopes_z_g.resize(this->m_amr_mesh.local_num_ghosts());

} // GodunovImplemV2<dim, device_t>::resize_auxiliary_data

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
uint64_t
GodunovImplemV2<dim, device_t>::total_mem_size_in_bytes()
{
  uint64_t total = 0;
  total += m_Q_ghosted_group.allocated_size_in_bytes();
  total += m_Q_ghosted_mg.allocated_size_in_bytes();
  total += m_Slopes_x_group.allocated_size_in_bytes();
  total += m_Slopes_y_group.allocated_size_in_bytes();
  total += m_Slopes_z_group.allocated_size_in_bytes();
  total += m_Slopes_x_g.allocated_size_in_bytes();
  total += m_Slopes_y_g.allocated_size_in_bytes();
  total += m_Slopes_z_g.allocated_size_in_bytes();

  return total;
} // GodunovImplemV2<dim, device_t>::total_mem_size_in_bytes

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV2<dim, device_t>::do_time_step(DataArrayBlock_t U, DataArrayBlock_t U2, real_t dt)
{

  int32_t nbOcts = static_cast<int32_t>(this->m_amr_mesh.local_num_quadrants());

  /*
   * When doing conservative update, we need to ensure that at the interface between two quads at
   * AMR level transition, the coarse quad get updated using the fluxes compute by the fine quad.
   *
   * We need special care when MPI is active: when the fine quad are actually ghost quad, while the
   * coarse one is a proper owned quad. There are (at least) two possibilities to solve this
   * problem:
   * - one would be that each MPI compute their own fluxes, and then do MPI comm to populate fluxes
   * in ghost quadrants.
   * - another (implemented here): we first perform MPI comm to fill primitive variables (with block
   * ghost) in ghosts quad; and apply the full numerical schemes in those ghost quads, and perform
   * update.
   *
   * We are currently implementing the second possibility for no particular reason (one has to start
   * somewhere); but clearly at some point, we should refactor MeshGhostExchanger to allow
   * transferring directly fluxes.
   *
   */

  // convert conservative variable into primitives ones m_U ==> m_Q_ghosted_mg
  this->convert_to_primitives_in_mirror_quads(U);

  // do mpi comm to populate ghosts quads in m_Q_ghosted_mg (see MeshGhostExchanger)
  this->mpi_exchange_mirrors_and_ghosts(m_Q_ghosted_mg);

  this->compute_limited_slopes_in_ghosts();

  this->compute_fluxes_and_update_in_ghosts(U, U2, dt);

  // number of group of octants, rounding to upper value
  int32_t nbGroup = (nbOcts + m_nbOctsPerGroup - 1) / m_nbOctsPerGroup;

  //
  // Do spatial sub-cycling
  //
  for (int32_t iGroup = 0; iGroup < nbGroup; ++iGroup)
  {

    // at which index starts current group
    const auto iOct_begin = iGroup * m_nbOctsPerGroup;

    // last group might be smaller than the other (depending is integer division in exact)
    const auto num_octants_in_group =
      (iGroup == nbGroup - 1) ? nbOcts - iOct_begin : m_nbOctsPerGroup;

    // start main computation
    KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME);

    // update primitive variables in ghosted block for all octant in current group of octants
    this->convert_to_primitives_in_group_of_quads(U, iOct_begin, num_octants_in_group);


    // compute limited slopes
    this->compute_limited_slopes_in_group(num_octants_in_group);

    // perform time integration (compute flux and update) :
    this->compute_fluxes_and_update_in_group(U, U2, iOct_begin, num_octants_in_group, dt);

  } // end for iGroup

  //
  // Add gravity source term when enabled
  //
  const auto gravity_enabled = this->m_config_map.getBool("gravity", "enabled", false);
  if (gravity_enabled)
    this->add_gravity_source_term(U, U2, dt);

} // GodunovImplemV2<dim, device_t>::do_time_step

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV2<dim, device_t>::convert_to_primitives_in_mirror_quads(DataArrayBlock_t U)
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

} // GodunovImplemV2<dim, device_t>::convert_to_primitives_in_mirror_quads

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV2<dim, device_t>::convert_to_primitives_in_group_of_quads(DataArrayBlock_t U,
                                                                        int32_t          iOct_begin,
                                                                        int32_t num_octant_in_group)
{

  KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME_CONV_PRIM);

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  const auto & fm = this->m_hydro.get_fieldmap();

  // compute primitive variables in all owned blocks (U must have MPI ghost up to date)
  ConvertToPrimitivesVariablesFunctor<dim, device_t>::apply_on_group(
    this->m_config_map,
    this->m_mesh_map.hashmap(),
    this->m_mesh_map.orchard_keys(),
    this->m_mesh_map.get_amr_mesh_info(),
    iOct_begin,
    num_octant_in_group,
    U,
    m_Q_ghosted_group,
    fm,
    this->m_brick_sizes,
    this->m_is_brick_periodic,
    this->m_hydro_settings,
    this->m_eos);

} // GodunovImplemV2<dim, device_t>::convert_to_primitives_in_group_of_quads

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV2<dim, device_t>::compute_limited_slopes_in_group(int32_t num_octants_to_process)
{

  KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME_SLOPES);

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  const auto & fm = this->m_hydro.get_fieldmap();

  // compute limited slopes in all quadrants of q_group
  ComputeLimitedSlopesFunctor<dim, device_t>::apply_on_group(m_Q_ghosted_group,
                                                             m_Slopes_x_group,
                                                             m_Slopes_y_group,
                                                             m_Slopes_z_group,
                                                             fm,
                                                             num_octants_to_process,
                                                             this->m_hydro_settings);

} // GodunovImplemV2<dim, device_t>::compute_limited_slopes_in_group

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV2<dim, device_t>::compute_limited_slopes_in_ghosts()
{
  KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME_SLOPES);

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  const auto & fm = this->m_hydro.get_fieldmap();

  // compute limited slopes in all quadrants that are ghost quadrants
  ComputeLimitedSlopesFunctor<dim, device_t>::apply_on_ghosts(m_Q_ghosted_mg,
                                                              m_Slopes_x_g,
                                                              m_Slopes_y_g,
                                                              m_Slopes_z_g,
                                                              fm,
                                                              this->m_amr_mesh.local_num_mirrors(),
                                                              this->m_amr_mesh.local_num_ghosts(),
                                                              this->m_hydro_settings);

} // GodunovImplemV2<dim, device_t>::compute_limited_slopes_in_ghosts

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV2<dim, device_t>::compute_fluxes_and_update_in_group(DataArrayBlock_t u_in,
                                                                   DataArrayBlock_t u_out,
                                                                   int32_t          iOct_begin,
                                                                   int32_t num_octants_to_process,
                                                                   real_t  dt)
{

  KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME_UPDATE);

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  const auto & fm = this->m_hydro.get_fieldmap();

  // compute fluxes and update all quadrant in a group of quadrants
  ComputeFluxesAndConservativeUpdateFunctor<dim, device_t>::apply_on_group(
    this->m_config_map,
    this->m_mesh_map.hashmap(),
    this->m_mesh_map.orchard_keys(),
    this->m_mesh_map.conformal_status(),
    this->m_mesh_map.get_amr_mesh_info(),
    u_in,
    u_out,
    m_Q_ghosted_group,
    m_Slopes_x_group,
    m_Slopes_y_group,
    m_Slopes_z_group,
    fm,
    iOct_begin,
    num_octants_to_process,
    this->m_brick_sizes,
    this->m_is_brick_periodic,
    this->m_hydro_settings,
    this->m_eos,
    dt);

} // GodunovImplemV2<dim, device_t>::compute_fluxes_and_update_in_group

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV2<dim, device_t>::compute_fluxes_and_update_in_ghosts(DataArrayBlock_t u_in,
                                                                    DataArrayBlock_t u_out,
                                                                    real_t           dt)
{

  KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME_UPDATE);

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  const auto & fm = this->m_hydro.get_fieldmap();

  // compute incoming fluxes in all ghost quadrants and perform conservative update only at
  // interface between ghost and owned quadrants (in case the owned quadrant is coarser than the
  // ghost quadrant)
  ComputeFluxesAndConservativeUpdateFunctor<dim, device_t>::apply_on_ghosts(
    this->m_config_map,
    this->m_mesh_map.hashmap(),
    this->m_mesh_map.orchard_keys(),
    this->m_mesh_map.conformal_status(),
    this->m_mesh_map.get_amr_mesh_info(),
    u_in,
    u_out,
    m_Q_ghosted_mg,
    m_Slopes_x_g,
    m_Slopes_y_g,
    m_Slopes_z_g,
    fm,
    this->m_brick_sizes,
    this->m_is_brick_periodic,
    this->m_hydro_settings,
    this->m_eos,
    dt);

} // GodunovImplemV2<dim, device_t>::compute_fluxes_and_update_in_ghosts

// explicit template instantiation
template class GodunovImplemV2<2, kalypsso::DefaultDevice>;
template class GodunovImplemV2<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso
