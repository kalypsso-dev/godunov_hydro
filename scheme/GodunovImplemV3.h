// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file GodunovImplemV3.h
 *
 * Godunov time integration implementation detail version v3.
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_GODUNOV_IMPLEM_V3_H_
#define KALYPSSO_GODUNOV_HYDRO_GODUNOV_IMPLEM_V3_H_

#include <godunov_hydro/scheme/GodunovImplemBase.h>

#include <kalypsso/core/utils_block.h>
#include <kalypsso/core/config_utils.h> // for get_block_sizes
#include <kalypsso/core/TimeIntegratorConfig.h>

namespace kalypsso
{
namespace godunov_hydro
{
/**
 * \class GodunovImplemV3
 *
 * version 3 features:
 * same sa version 0 but don't store slopes, store reconstructed states instead.
 *
 * \sa GodunovImplemV3
 */
template <size_t dim, typename device_t>
class GodunovImplemV3 : public GodunovImplemBase<dim, device_t>
{
public:
  using GodunovImplemBase_t = GodunovImplemBase<dim, device_t>;
  using DataArrayBlock_t = typename GodunovImplemBase_t::DataArrayBlock_t;
  using DataArrayGhostedBlock_t = typename GodunovImplemBase_t::DataArrayGhostedBlock_t;


  GodunovImplemV3(ParallelEnv const &      par_env,
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
    , m_Q_ghosted(this->m_block_sizes,
                  this->m_block_sizes + 2 * 2,
                  get_shift<dim>(-2),
                  "Q_ghosted (owned and ghosts)",
                  models::Hydro<dim>::nbvar(),
                  0)
    , m_Q_ghosted_mg(this->m_block_sizes,
                     this->m_block_sizes + 2 * 2,
                     get_shift<dim>(-2),
                     "Q_ghosted_mg",
                     models::Hydro<dim>::nbvar(),
                     0)
    , m_Slopes_x(this->m_block_sizes,
                 this->m_block_sizes + 2 * 1,
                 get_shift<dim>(-1),
                 "Slope_x_group",
                 models::Hydro<dim>::nbvar(),
                 0)
    , m_Slopes_y(this->m_block_sizes,
                 this->m_block_sizes + 2 * 1,
                 get_shift<dim>(-1),
                 "Slope_y_group",
                 models::Hydro<dim>::nbvar(),
                 0)
    , m_Slopes_z(this->m_block_sizes,
                 this->m_block_sizes + 2 * 1,
                 get_shift<dim>(-1),
                 "Slope_z_group",
                 models::Hydro<dim>::nbvar(),
                 0)
    , m_Q_left("Q_left",
               get_flux_block_sizes<dim>(this->m_block_sizes, IX),
               models::Hydro<dim>::nbvar(),
               0)
    , m_Q_right("Q_right",
                get_flux_block_sizes<dim>(this->m_block_sizes, IX),
                models::Hydro<dim>::nbvar(),
                0)
    , m_Fluxes("Fluxes",
               get_flux_block_sizes<dim>(this->m_block_sizes, IX),
               models::Hydro<dim>::nbvar(),
               0)
    , m_time_integrator(TimeIntegratorConfig::get_time_integrator(config_map))
  {} // GodunovImplemV3

  // destructor
  ~GodunovImplemV3() = default;

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
    GodunovImplemV3<dim, device_t> * impl = new GodunovImplemV3<dim, device_t>(
      par_env, params, config_map, profiling_manager, amr_mesh, mesh_map, mesh_ghosts_exchanger);
#else
    GodunovImplemV3<dim, device_t> * impl = new GodunovImplemV3<dim, device_t>(
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
  do_time_step(DataArrayBlock_t const & U, DataArrayBlock_t const & U2, real_t dt) override;

private:
  /*
   * ghosted block arrays used for piece wise computation
   */
  //! hydrodynamics primitive - ghosted block - number of octants : owned + ghost
  //! ghostwidth of 2
  DataArrayGhostedBlock_t m_Q_ghosted;

  //! hydrodynamics primitive - ghosted block - number of octants: MPI mirrors + MPI ghosts
  //! \note the suffix _mg is meant to tell the developer that this array "lives" on
  //! mirror (m) and ghosts(g)
  //! ghostwidth of 2
  DataArrayGhostedBlock_t m_Q_ghosted_mg;

  //! slopes along X dir - ghosted block array of octant's block data - owned + ghosts
  //! ghostwidth of 1
  DataArrayGhostedBlock_t m_Slopes_x;

  //! slopes along Y dir - ghosted block array of octant's block data - owned + ghosts
  //! ghostwidth of 1
  DataArrayGhostedBlock_t m_Slopes_y;

  //! slopes along Z dir - ghosted block array of octant's block data - owned + ghosts
  //! ghostwidth of 1
  DataArrayGhostedBlock_t m_Slopes_z;

  //! reconstruct state on the left cell interface along a given direction (X, Y or Z)
  DataArrayBlock_t m_Q_left;

  //! reconstruct state on the left cell interface along a given direction (X, Y or Z)
  DataArrayBlock_t m_Q_right;

  //! Temporary buffer used to stored hydrodynamics flux (owned + ghost + outside block).
  //! It will be reshaped as needed to store either fluxes along X, Y or Z direction.
  DataArrayBlock_t m_Fluxes;

  //! time integrator
  const TimeIntegrator m_time_integrator;

  //! Convert conservative variables to primitive variables in mirror quadrants.
  //! Fills m_Q_ghosted_mg
  //!
  //! \param[in] U conservative variables array (owned + MPI ghost + outside quadrants)
  //!
  void
  convert_to_primitives_in_mirror_quads(DataArrayBlock_t U);

  //! Convert conservative variables to primitive variables in owned octants + copy ghost octants
  //!
  //! \param[in] U conservative variables (owned + MPI ghost + outside quadrants)
  void
  convert_to_primitives(DataArrayBlock_t U);

  //! compute limited slopes in owned and ghosts quadrants
  void
  compute_limited_slopes_in_owned_and_ghosts();

  //! compute q_left / q_right (reconstructed stated at face centers)
  void
  compute_reconstructed_states_in_owned_and_ghosts(real_t dt, int direction);

  //! Compute hydro fluxes in all (owned and ghosts) quandrants.
  void
  compute_fluxes_and_store_in_owned_and_ghosts(real_t dt, int direction);

  //! Compute viscous fluxes in all (owned and ghosts) quandrants.
  void
  compute_viscous_fluxes_and_store_in_owned_and_ghosts(real_t dt, int direction);

  //! Update conservative variable in owned quadrants.
  void
  read_fluxes_and_update_in_owned(DataArrayBlock_t u_out, real_t dt, int direction);

}; // class GodunovImplemV3

extern template class GodunovImplemV3<2, kalypsso::DefaultDevice>;
extern template class GodunovImplemV3<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_GODUNOV_IMPLEM_V3_H_
