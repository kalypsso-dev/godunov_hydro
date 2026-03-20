// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file GodunovImplemV1.h
 *
 * Godunov time integration implementation detail version v1.
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_GODUNOV_IMPLEM_V1_H_
#define KALYPSSO_GODUNOV_HYDRO_GODUNOV_IMPLEM_V1_H_

#include <godunov_hydro/scheme/GodunovImplemBase.h>

#include <kalypsso/core/utils_block.h>
#include <kalypsso/core/config_utils.h> // for get_block_sizes

namespace kalypsso
{
namespace godunov_hydro
{
/**
 * \class GodunovImplemV1
 *
 * version 1 features:
 *
 * - same as version 0 (store flux) but computation of fluxes is done using spatial subcycling;
 *   subcycling is done along the Morton curve.
 *
 * \sa GodunovImplemV0
 */
template <size_t dim, typename device_t>
class GodunovImplemV1 : public GodunovImplemBase<dim, device_t>
{
public:
  using GodunovImplemBase_t = GodunovImplemBase<dim, device_t>;
  using DataArrayBlock_t = typename GodunovImplemBase_t::DataArrayBlock_t;
  using DataArrayGhostedBlock_t = typename GodunovImplemBase_t::DataArrayGhostedBlock_t;


  GodunovImplemV1(ParallelEnv const &      par_env,
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
    , m_nbQuadsPerGroup(config_map.getInteger("amr", "nbOctsPerGroup", 2048))
    , m_Q_ghosted_group(this->m_block_sizes,
                        this->m_block_sizes + 2 * 2,
                        get_shift<dim>(-2),
                        "Q_ghosted_group",
                        nbvar_hydro<dim>(),
                        m_nbQuadsPerGroup)
    , m_Q_ghosted_mg(this->m_block_sizes,
                     this->m_block_sizes + 2 * 2,
                     get_shift<dim>(-2),
                     "Q_ghosted_mg",
                     nbvar_hydro<dim>(),
                     0)
    , m_Slopes_x(this->m_block_sizes,
                 this->m_block_sizes + 2 * 1,
                 get_shift<dim>(-1),
                 "Slope_x_group",
                 nbvar_hydro<dim>(),
                 m_nbQuadsPerGroup)
    , m_Slopes_y(this->m_block_sizes,
                 this->m_block_sizes + 2 * 1,
                 get_shift<dim>(-1),
                 "Slope_y_group",
                 nbvar_hydro<dim>(),
                 m_nbQuadsPerGroup)
    , m_Slopes_z(this->m_block_sizes,
                 this->m_block_sizes + 2 * 1,
                 get_shift<dim>(-1),
                 "Slope_z_group",
                 nbvar_hydro<dim>(),
                 dim == 3 ? m_nbQuadsPerGroup : 0)
    , m_Fluxes_x("Fluxes_x",
                 get_flux_block_sizes<dim>(this->m_block_sizes, IX),
                 nbvar_hydro<dim>(),
                 0)
    , m_Fluxes_y("Fluxes_y",
                 get_flux_block_sizes<dim>(this->m_block_sizes, IY),
                 nbvar_hydro<dim>(),
                 0)
    , m_Fluxes_z("Fluxes_z",
                 get_flux_block_sizes<dim>(this->m_block_sizes, IZ),
                 nbvar_hydro<dim>(),
                 0)
  {} // GodunovImplemV1

  // destructor
  ~GodunovImplemV1() = default;

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
    GodunovImplemV1<dim, device_t> * impl = new GodunovImplemV1<dim, device_t>(
      par_env, params, config_map, profiling_manager, amr_mesh, mesh_map, mesh_ghosts_exchanger);
#else
    GodunovImplemV1<dim, device_t> * impl = new GodunovImplemV1<dim, device_t>(
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
  //! number of quadrants/octants (octree leaves) per group used for spatial subcycling.
  int32_t m_nbQuadsPerGroup;

  /*
   * ghosted block arrays used for piece wise computation
   */
  //! hydrodynamics primitive - ghosted block - number of octants per group
  DataArrayGhostedBlock_t m_Q_ghosted_group;

  //! hydrodynamics primitive - ghosted block - number of octants: MPI mirrors + MPI ghosts
  //! \note the suffix _mg is meant to tell the developer that this array "lives" on
  //! mirror (m) and ghosts(g)
  //! ghostwidth of 2
  DataArrayGhostedBlock_t m_Q_ghosted_mg;

  //! slopes along X dir - ghosted block array of octant's block data - number of octants per group
  //! ghostwidth of 1
  DataArrayGhostedBlock_t m_Slopes_x;

  //! slopes along Y dir - ghosted block array of octant's block data - number of octants per group
  //! ghostwidth of 1
  DataArrayGhostedBlock_t m_Slopes_y;

  //! slopes along Z dir - ghosted block array of octant's block data - number of octants per group
  //! ghostwidth of 1
  DataArrayGhostedBlock_t m_Slopes_z;

  //! Temporary buffer used to stored hydrodynamics flux (owned + ghost + outside block).
  //! It will be reshaped as needed to store either fluxes along X, Y or Z direction.
  DataArrayBlock_t m_Fluxes_x;
  DataArrayBlock_t m_Fluxes_y;
  DataArrayBlock_t m_Fluxes_z;

  //! Convert conservative variables to primitive variables in mirror quadrants.
  //! Fills m_Q_ghosted_mg
  //!
  //! \param[in] U conservative variables array (owned + MPI ghost + outside quadrants)
  //! \param[in] half_dt
  //!
  void
  convert_to_primitives_in_mirror_quads(DataArrayBlock_t U);

  //! convert conservative variables to primitive variables in a group of quadrants/octants.
  //!
  //! We want to fill primitive variables for a range of quadrants/octants starting at
  //! iOct_begin.
  //! - if the range overlaps owned quadrants then use ConvertToPrimitivesVariablesFunctor to
  //!   compute them from conservative variables.
  //! - if the range overlaps ghost quadrants then just perform a copy
  //! - if the range overlaps both, do both !
  //!
  //! \param[in] U conservative variables (owned + MPI ghost + outside quadrants)
  //! \param[in] iOct_begin
  //! \param[in] num_quadrants_in_group number of quadrant data to fill
  void
  convert_to_primitives_in_group(DataArrayBlock_t U,
                                 int32_t          iOct_begin,
                                 int32_t          num_quadrants_in_group);

  //! compute limited slopes in a group of owned octant
  void
  compute_limited_slopes_in_group(int32_t num_quads_to_process);

  //! Compute hydro fluxes in all (owned and ghosts) quandrants.
  void
  compute_fluxes_and_store_in_group(real_t dt, int32_t iOct_begin, int32_t num_quadrants_in_group);


  //! Update conservative variable in owned quadrants.
  void
  read_fluxes_and_update_in_owned(DataArrayBlock_t u_out, real_t dt);

}; // class GodunovImplemV1

extern template class GodunovImplemV1<2, kalypsso::DefaultDevice>;
extern template class GodunovImplemV1<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_GODUNOV_IMPLEM_V1_H_
