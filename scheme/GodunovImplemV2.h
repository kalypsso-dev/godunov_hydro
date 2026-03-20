// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file GodunovImplemV2.h
 *
 * Godunov time integration implementation detail version v2.
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_GODUNOV_IMPLEM_V2_H_
#define KALYPSSO_GODUNOV_HYDRO_GODUNOV_IMPLEM_V2_H_

#include <godunov_hydro/scheme/GodunovImplemBase.h>

#include <kalypsso/core/utils_block.h>
#include <kalypsso/core/config_utils.h> // for get_block_sizes

namespace kalypsso
{
namespace godunov_hydro
{
/**
 * \class GodunovImplemV2
 *
 * version 0 features:
 *
 * - don't store fluxes;
 * - just compute flux and use them to perform atomic updates in a single kokkos functor;
 * - perform a sub-domain decomposition (which helps decreasing memory
 *   footprint due to ghosted block array).
 *
 * \sa GodunovImplemV2
 */
template <size_t dim, typename device_t>
class GodunovImplemV2 : public GodunovImplemBase<dim, device_t>
{
public:
  using GodunovImplemBase_t = GodunovImplemBase<dim, device_t>;
  using DataArrayBlock_t = typename GodunovImplemBase_t::DataArrayBlock_t;
  using DataArrayGhostedBlock_t = typename GodunovImplemBase_t::DataArrayGhostedBlock_t;


  GodunovImplemV2(ParallelEnv const &      par_env,
                  HydroParams const &      params,
                  ConfigMap const &        config_map,
                  ProfilingManager &       profiling_manager,
                  AMRmesh<dim> const &     amr_mesh,
                  MeshMap<dim, device_t> & mesh_map
#ifdef KALYPSSO_CORE_USE_MPI
                  ,
                  MeshGhostsExchanger<dim, real_t, device_t> & mesh_ghosts_exchanger
#endif // KALYPSSO_CORE_USE_MPI
                  )
    : GodunovImplemBase_t(par_env,
                          params,
                          config_map,
                          profiling_manager,
                          amr_mesh,
                          mesh_map
#ifdef KALYPSSO_CORE_USE_MPI
                          ,
                          mesh_ghosts_exchanger
#endif // KALYPSSO_CORE_USE_MPI
                          )
    , m_nbOctsPerGroup(config_map.getInteger("amr", "nbOctsPerGroup", 128))
    , m_Q_ghosted_group(this->m_block_sizes,
                        this->m_block_sizes + 2 * 2,
                        get_shift<dim>(-2),
                        "Q_ghosted_group",
                        nbvar_hydro<dim>(),
                        m_nbOctsPerGroup)
    , m_Q_ghosted_mg(this->m_block_sizes,
                     this->m_block_sizes + 2 * 2,
                     get_shift<dim>(-2),
                     "Q_ghosted_mg",
                     nbvar_hydro<dim>(),
                     0)
    , m_Slopes_x_group(this->m_block_sizes,
                       this->m_block_sizes + 2 * 1,
                       get_shift<dim>(-1),
                       "Slope_x_group",
                       nbvar_hydro<dim>(),
                       m_nbOctsPerGroup)
    , m_Slopes_y_group(this->m_block_sizes,
                       this->m_block_sizes + 2 * 1,
                       get_shift<dim>(-1),
                       "Slope_y_group",
                       nbvar_hydro<dim>(),
                       m_nbOctsPerGroup)
    , m_Slopes_z_group(this->m_block_sizes,
                       this->m_block_sizes + 2 * 1,
                       get_shift<dim>(-1),
                       "Slope_z_group",
                       nbvar_hydro<dim>(),
                       dim == 3 ? m_nbOctsPerGroup : 0)
    , m_Slopes_x_g(this->m_block_sizes,
                   this->m_block_sizes + 2 * 1,
                   get_shift<dim>(-1),
                   "Slope_x_g",
                   nbvar_hydro<dim>(),
                   0)
    , m_Slopes_y_g(this->m_block_sizes,
                   this->m_block_sizes + 2 * 1,
                   get_shift<dim>(-1),
                   "Slope_y_g",
                   nbvar_hydro<dim>(),
                   0)
    , m_Slopes_z_g(this->m_block_sizes,
                   this->m_block_sizes + 2 * 1,
                   get_shift<dim>(-1),
                   "Slope_z_g",
                   nbvar_hydro<dim>(),
                   0)
  {}

  // destructor
  ~GodunovImplemV2() = default;

  /**
   * Static creation method called by the solver factory.
   */
  static GodunovImplemBase_t *
  create(ParallelEnv const &      par_env,
         HydroParams const &      params,
         ConfigMap const &        config_map,
         ProfilingManager &       profiling_manager,
         AMRmesh<dim> const &     amr_mesh,
         MeshMap<dim, device_t> & mesh_map
#ifdef KALYPSSO_CORE_USE_MPI
         ,
         MeshGhostsExchanger<dim, real_t, device_t> & mesh_ghosts_exchanger
#endif // KALYPSSO_CORE_USE_MPI
  )
  {
#ifdef KALYPSSO_CORE_USE_MPI
    GodunovImplemV2<dim, device_t> * impl = new GodunovImplemV2<dim, device_t>(
      par_env, params, config_map, profiling_manager, amr_mesh, mesh_map, mesh_ghosts_exchanger);
#else
    GodunovImplemV2<dim, device_t> * impl = new GodunovImplemV2<dim, device_t>(
      par_env, params, config_map, profiling_manager, amr_mesh, mesh_map);
#endif // KALYPSSO_CORE_USE_MPI

    return impl;
  }

  /**
   * Public interface
   */

  //! resize only auxiliary data array (implementation dependent)
  void
  resize_auxiliary_data() override;

  //! memory footprint monitoring
  uint64_t
  total_mem_size_in_bytes() override;

  void
  do_time_step(DataArrayBlock_t U, DataArrayBlock_t U2, real_t dt) override;

private:
  //! number of octants (octree leaves) per group
  int32_t m_nbOctsPerGroup;

  /*
   * ghosted block arrays used for piece wise computation
   */
  //! hydrodynamics primitive - ghosted block - number of octants: m_nbOctsPerGroup
  //! ghostwidth of 2
  DataArrayGhostedBlock_t m_Q_ghosted_group;

  //! hydrodynamics primitive - ghosted block - number of octants: mirrors + MPI ghosts
  //! \note the suffix _mg is meant to tell the developer that this array "lives" on mirror (m) and
  //! ghosts(g)
  //! ghostwidth of 2
  DataArrayGhostedBlock_t m_Q_ghosted_mg;

  //! slopes along X dir - ghosted block array of octant's block data
  //! ghostwidth of 1
  DataArrayGhostedBlock_t m_Slopes_x_group;

  //! slopes along Y dir - ghosted block array of octant's block data
  //! ghostwidth of 1
  DataArrayGhostedBlock_t m_Slopes_y_group;

  //! slopes along Z dir - ghosted block array of octant's block data
  //! ghostwidth of 1
  DataArrayGhostedBlock_t m_Slopes_z_group;

  //! slopes along X dir - ghosted block in ghost quadrants only
  //! \note the suffix _g is meant to tell the developer that this array "lives" on ghosts
  //! ghostwidth of 1
  DataArrayGhostedBlock_t m_Slopes_x_g;

  //! slopes along Y dir - ghosted block in ghost quadrants only
  //! \note the suffix _g is meant to tell the developer that this array "lives" on ghosts
  //! ghostwidth of 1
  DataArrayGhostedBlock_t m_Slopes_y_g;

  //! slopes along Z dir - ghosted block in ghost quadrants only
  //! \note the suffix _g is meant to tell the developer that this array "lives" on ghosts
  //! ghostwidth of 1
  DataArrayGhostedBlock_t m_Slopes_z_g;

  //! convert conservative variables to primitive variables in mirror quadrants.
  //!
  //! \param[in] U conservative variables array (owned + MPI ghost + outside quadrants)
  //! \param[in] half_dt
  void
  convert_to_primitives_in_mirror_quads(DataArrayBlock_t U);

  //! convert conservative variables to primitive variables in a group of quadrants.
  //!
  //! \param[in] conservative variables (owned + MPI ghost + outside quadrants)
  //! \param[in] iOct_begin first quadrant id to process
  //! \param[in] num_quadrants_in_group is the number of quadrants to process
  //! \param[in] half_dt halt time step
  void
  convert_to_primitives_in_group_of_quads(DataArrayBlock_t U,
                                          int32_t          iOct_begin,
                                          int32_t          num_quadrants_in_group);

  //! compute limited slopes in a group of owned quadrants.
  //!
  //! \param[in] q primitive variables as a ghosted data array (ghost width = 2, size is
  //!            m_nbOctsPerGroup)
  //! \param[out] slopes_x is a a ghosted data array of slopes (ghost width = 1, size is
  //!            m_nbOctsPerGroup)
  void
  compute_limited_slopes_in_group(int32_t num_octants_to_process);

  //! compute limited slopes in ghost octants
  void
  compute_limited_slopes_in_ghosts();

  //! perform Godunov MUSCL-Hancock Update in bulk octant.
  //! this update can be either conservative on non-conservative (at interface between two octants
  //! of different levels)
  void
  compute_fluxes_and_update_in_group(DataArrayBlock_t u_in,
                                     DataArrayBlock_t u_out,
                                     int32_t          iOct_begin,
                                     int32_t          num_octants_to_process,
                                     real_t           dt);

  //! perform Godunov MUSCL Update in ghost octants.
  //! this function is only needed when performing a conservative update.
  //! the purpose of this function is that when one want to update a owned quadrant with flux coming
  //! from a finer level neighbor that is a ghost, flux must be computed in the ghost quadrant,
  //! before updating the owned quadrant.
  void
  compute_fluxes_and_update_in_ghosts(DataArrayBlock_t u_in, DataArrayBlock_t u_out, real_t dt);

}; // class GodunovImplemV2

extern template class GodunovImplemV2<2, kalypsso::DefaultDevice>;
extern template class GodunovImplemV2<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_GODUNOV_IMPLEM_V2_H_
