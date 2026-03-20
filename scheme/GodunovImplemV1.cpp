// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file GodunovImplemV1.cpp
 *
 * Godunov time integration implementation detail version v1.
 */
#include <godunov_hydro/scheme/GodunovImplemV1.h>

// Compute functors
#include <godunov_hydro/scheme/ConvertToPrimitivesVariablesFunctor.h>
#include <godunov_hydro/scheme/ComputeLimitedSlopesFunctor.h>
#include <godunov_hydro/scheme/ComputeFluxesAndStoreFunctor.h>
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
GodunovImplemV1<dim, device_t>::resize_auxiliary_data()
{
  const auto num_owned = this->m_mesh_map.get_amr_mesh_info().local_num_quadrants();
  const auto num_ghosts = this->m_amr_mesh.local_num_ghosts();
  const auto num_mirrors = this->m_amr_mesh.local_num_mirrors();

  // note: m_Q_ghosted_mg is sized with the number of mirror + number of ghost quadrants,
  // because ghost quadrant will be populated with MeshGhostsExchanger
  m_Q_ghosted_mg.resize(num_mirrors + num_ghosts);

  // Flux is resized large enough to store fluxes for owned + ghost blocks
  // we don't need to compute in outside quad, since outside quad are at the same level of AMR than
  // the neighbor directly inside
  m_Fluxes_x.resize(num_owned + num_ghosts);
  m_Fluxes_y.resize(num_owned + num_ghosts);
  m_Fluxes_z.resize(num_owned + num_ghosts);

} // GodunovImplemV1<dim, device_t>::resize_auxiliary_data

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
uint64_t
GodunovImplemV1<dim, device_t>::total_mem_size_in_bytes()
{
  uint64_t total = 0;
  total += m_Q_ghosted_group.allocated_size_in_bytes();
  total += m_Q_ghosted_mg.allocated_size_in_bytes();
  total += m_Slopes_x.allocated_size_in_bytes();
  total += m_Slopes_y.allocated_size_in_bytes();
  total += m_Slopes_z.allocated_size_in_bytes();
  total += m_Fluxes_x.allocated_size_in_bytes();
  total += m_Fluxes_y.allocated_size_in_bytes();
  total += m_Fluxes_z.allocated_size_in_bytes();

  return total;
} // GodunovImplemV1<dim, device_t>::total_mem_size_in_bytes

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV1<dim, device_t>::do_time_step(DataArrayBlock_t U, DataArrayBlock_t U2, real_t dt)
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

  // number of quadrants used to do spatial subcycling: owned + ghosts
  int32_t nbQuads = static_cast<int32_t>(this->m_amr_mesh.local_num_quadrants() +
                                         this->m_amr_mesh.local_num_ghosts());

  // number of group of octants, rounding to upper value
  int32_t nbGroups = (nbQuads + m_nbQuadsPerGroup - 1) / m_nbQuadsPerGroup;

  //
  // Do spatial subcycling
  //
  for (int32_t iGroup = 0; iGroup < nbGroups; ++iGroup)
  {
    // at which index starts current group
    const int32_t iOct_begin = iGroup * m_nbQuadsPerGroup;

    // last group might be smaller than the other (depending is integer division in exact)
    const int32_t num_quads_in_group =
      (iGroup == nbGroups - 1) ? nbQuads - iOct_begin : m_nbQuadsPerGroup;

    // update primitive variables in owned + ghost quadrants
    this->convert_to_primitives_in_group(U, iOct_begin, num_quads_in_group);

    // compute limited slopes in group
    this->compute_limited_slopes_in_group(num_quads_in_group);

    // compute flux along X, Y and Z and store
    this->compute_fluxes_and_store_in_group(dt, iOct_begin, num_quads_in_group);

  } // end for group

  //
  // final update
  //
  this->read_fluxes_and_update_in_owned(U2, dt);

  //
  // Add gravity source term when enabled
  //
  const auto gravity_enabled = this->m_config_map.getBool("gravity", "enabled", false);
  if (gravity_enabled)
    this->add_gravity_source_term(U, U2, dt);

} // GodunovImplemV1<dim, device_t>::do_time_step

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV1<dim, device_t>::convert_to_primitives_in_mirror_quads(DataArrayBlock_t U)
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

} // GodunovImplemV1<dim, device_t>::convert_to_primitives_in_mirror_quads

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV1<dim, device_t>::convert_to_primitives_in_group(DataArrayBlock_t U,
                                                               int32_t          iOct_begin,
                                                               int32_t          num_quads_in_group)
{

  KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME_CONV_PRIM);

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  const auto & fm = this->m_hydro.get_fieldmap();

  const auto num_quads_owned_total =
    static_cast<int32_t>(this->m_mesh_map.get_amr_mesh_info().local_num_quadrants());
  const auto num_quads_mirrors_total = static_cast<int32_t>(this->m_amr_mesh.local_num_mirrors());
  const auto num_quads_ghosts_total = static_cast<int32_t>(this->m_amr_mesh.local_num_ghosts());
  const int32_t num_quads_total = (num_quads_owned_total + num_quads_ghosts_total);

  int32_t num_quads_owned = 0;
  int32_t num_quads_ghosts = 0;

  if (iOct_begin + num_quads_in_group <= num_quads_owned_total)
  {
    num_quads_owned = num_quads_in_group;
    num_quads_ghosts = 0;
  }
  else if (iOct_begin >= num_quads_owned_total)
  {
    num_quads_owned = 0;
    num_quads_ghosts = iOct_begin + num_quads_in_group <= num_quads_total
                         ? num_quads_in_group
                         : num_quads_total - iOct_begin;
  }
  else
  {
    num_quads_owned = num_quads_owned_total - iOct_begin;
    num_quads_ghosts = iOct_begin + num_quads_in_group <= num_quads_total
                         ? num_quads_in_group - num_quads_owned
                         : num_quads_total - iOct_begin - num_quads_owned;
  }

  //
  // step 1: if group of quadrants intersect range [0, num_quads_owned_total[ apply functor
  //

  if (num_quads_owned > 0)
  {
    // compute primitive variables in range of owned blocks
    ConvertToPrimitivesVariablesFunctor<dim, device_t>::apply_on_group(
      this->m_config_map,
      this->m_mesh_map.hashmap(),
      this->m_mesh_map.orchard_keys(),
      this->m_mesh_map.get_amr_mesh_info(),
      iOct_begin,
      num_quads_owned,
      U,
      m_Q_ghosted_group,
      fm,
      this->m_brick_sizes,
      this->m_is_brick_periodic,
      this->m_hydro_settings,
      this->m_eos);
  }

  //
  // step 2: if group of quadrants intersect range [num_quads_owned_total, num_quads_total[
  // do a memory copy
  //
  if (num_quads_ghosts > 0)
  {
    const auto num_ghosted_cells = m_Q_ghosted_group.num_cells();
    const auto num_vars = m_Q_ghosted_group.num_vars();

    const int32_t offset_mg = iOct_begin + num_quads_owned - num_quads_owned_total;

    auto range_in = std::pair<std::size_t, std::size_t>(
      num_ghosted_cells * num_vars * (num_quads_mirrors_total + offset_mg),
      num_ghosted_cells * num_vars * (num_quads_mirrors_total + offset_mg + num_quads_ghosts));

    auto data_in = Kokkos::subview(m_Q_ghosted_mg.flat_view(), range_in);

    auto range_out = std::pair<std::size_t, std::size_t>(
      num_ghosted_cells * num_vars * num_quads_owned,
      num_ghosted_cells * num_vars * (num_quads_owned + num_quads_ghosts));

    auto data_out = Kokkos::subview(m_Q_ghosted_group.flat_view(), range_out);

    Kokkos::deep_copy(data_out, data_in);
  }

} // GodunovImplemV1<dim, device_t>::convert_to_primitives_in_group

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV1<dim, device_t>::compute_limited_slopes_in_group(int32_t num_quads_to_process)
{

  KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME_SLOPES);

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  const auto & fm = this->m_hydro.get_fieldmap();

  // compute limited slopes in all quadrants of q_group
  ComputeLimitedSlopesFunctor<dim, device_t>::apply_on_group(m_Q_ghosted_group,
                                                             m_Slopes_x,
                                                             m_Slopes_y,
                                                             m_Slopes_z,
                                                             fm,
                                                             num_quads_to_process,
                                                             this->m_hydro_settings);

} // GodunovImplemV1<dim, device_t>::compute_limited_slopes_in_owned_and_ghosts

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV1<dim, device_t>::compute_fluxes_and_store_in_group(real_t  dt,
                                                                  int32_t iOct_begin,
                                                                  int32_t num_quadrants_in_group)
{

  KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME_COMPUTE_FLUXES);

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  const auto & fm = this->m_hydro.get_fieldmap();

  // compute fluxes and update all quadrant in a group of quadrants
  ComputeFluxesAndStoreFunctor<dim, device_t>::apply(this->m_config_map,
                                                     this->m_mesh_map.orchard_keys(),
                                                     this->m_mesh_map.get_amr_mesh_info(),
                                                     m_Fluxes_x,
                                                     m_Q_ghosted_group,
                                                     m_Slopes_x,
                                                     m_Slopes_y,
                                                     m_Slopes_z,
                                                     fm,
                                                     iOct_begin,
                                                     num_quadrants_in_group,
                                                     IX,
                                                     this->m_hydro_settings,
                                                     this->m_eos,
                                                     this->m_viscosity,
                                                     dt);

  ComputeFluxesAndStoreFunctor<dim, device_t>::apply(this->m_config_map,
                                                     this->m_mesh_map.orchard_keys(),
                                                     this->m_mesh_map.get_amr_mesh_info(),
                                                     m_Fluxes_y,
                                                     m_Q_ghosted_group,
                                                     m_Slopes_x,
                                                     m_Slopes_y,
                                                     m_Slopes_z,
                                                     fm,
                                                     iOct_begin,
                                                     num_quadrants_in_group,
                                                     IY,
                                                     this->m_hydro_settings,
                                                     this->m_eos,
                                                     this->m_viscosity,
                                                     dt);

  if constexpr (dim == 3)
  {
    ComputeFluxesAndStoreFunctor<dim, device_t>::apply(this->m_config_map,
                                                       this->m_mesh_map.orchard_keys(),
                                                       this->m_mesh_map.get_amr_mesh_info(),
                                                       m_Fluxes_z,
                                                       m_Q_ghosted_group,
                                                       m_Slopes_x,
                                                       m_Slopes_y,
                                                       m_Slopes_z,
                                                       fm,
                                                       iOct_begin,
                                                       num_quadrants_in_group,
                                                       IZ,
                                                       this->m_hydro_settings,
                                                       this->m_eos,
                                                       this->m_viscosity,
                                                       dt);
  }

} // GodunovImplemV1<dim, device_t>::compute_fluxes_and_store_in_owned_and_ghosts

// =====================================================================
// =====================================================================
template <size_t dim, typename device_t>
void
GodunovImplemV1<dim, device_t>::read_fluxes_and_update_in_owned(DataArrayBlock_t u_out, real_t dt)
{

  KALYPSSO_PROFILING_REGION(this->m_profiling_mgr, NUM_SCHEME_UPDATE);

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  const auto & fm = this->m_hydro.get_fieldmap();

  // compute incoming fluxes in all ghost quadrants and perform conservative update only at
  // interface between ghost and owned quadrants (in case the owned quadrant is coarser than the
  // ghost quadrant)
  ReadFluxesAndConservativeUpdateFunctor<dim, device_t>::apply(this->m_config_map,
                                                               this->m_mesh_map.hashmap(),
                                                               this->m_mesh_map.orchard_keys(),
                                                               this->m_mesh_map.conformal_status(),
                                                               this->m_mesh_map.get_amr_mesh_info(),
                                                               u_out,
                                                               m_Fluxes_x,
                                                               fm,
                                                               IX,
                                                               this->m_brick_sizes,
                                                               this->m_is_brick_periodic,
                                                               this->m_hydro_settings,
                                                               dt);

  ReadFluxesAndConservativeUpdateFunctor<dim, device_t>::apply(this->m_config_map,
                                                               this->m_mesh_map.hashmap(),
                                                               this->m_mesh_map.orchard_keys(),
                                                               this->m_mesh_map.conformal_status(),
                                                               this->m_mesh_map.get_amr_mesh_info(),
                                                               u_out,
                                                               m_Fluxes_y,
                                                               fm,
                                                               IY,
                                                               this->m_brick_sizes,
                                                               this->m_is_brick_periodic,
                                                               this->m_hydro_settings,
                                                               dt);

  if constexpr (dim == 3)
  {
    ReadFluxesAndConservativeUpdateFunctor<dim, device_t>::apply(
      this->m_config_map,
      this->m_mesh_map.hashmap(),
      this->m_mesh_map.orchard_keys(),
      this->m_mesh_map.conformal_status(),
      this->m_mesh_map.get_amr_mesh_info(),
      u_out,
      m_Fluxes_z,
      fm,
      IZ,
      this->m_brick_sizes,
      this->m_is_brick_periodic,
      this->m_hydro_settings,
      dt);
  }

} // GodunovImplemV1<dim, device_t>::read_fluxes_and_update_in_owned

// explicit template instantiation
template class GodunovImplemV1<2, kalypsso::DefaultDevice>;
template class GodunovImplemV1<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso
