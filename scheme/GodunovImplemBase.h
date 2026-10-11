// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file GodunovImplemBase.h
 *
 * \brief Contains the interface of the Godunov time integration implementation details.
 *
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_GODUNOV_IMPLEM_BASE_H_
#define KALYPSSO_GODUNOV_HYDRO_GODUNOV_IMPLEM_BASE_H_

// shared
#include <kalypsso/core/kalypsso_core_config.h> // for KALYPSSO_CORE_USE_HDF5, ...
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h>
#include <kalypsso/core/HydroParams.h>
#include <kalypsso/core/AMRManager.h>
#include <kalypsso/core/config_utils.h> // for get_block_sizes

#include <kalypsso/core/ViscosityParams.h>

#include <kalypsso/utils/monitoring/ProfilingManager.h>

// profiling colors
#include <godunov_hydro/profiling.h>

// AMR services
#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <godunov_hydro/scheme/AddGravitySourceTerm.h>

namespace kalypsso
{
namespace godunov_hydro
{
template <size_t dim, typename device_t>
class GodunovImplemBase
{
public:
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using DataArrayBlockHost_t = DataArrayBlock<dim, real_t, HostDevice>;
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;

  GodunovImplemBase(ParallelEnv const &               par_env,
                    HydroParams const &               params,
                    ConfigMap const &                 config_map,
                    ProfilingManager &                profiling_manager,
                    core::AMRManager<dim, device_t> & amr_mgr)
    : m_par_env(par_env)
    , m_params(params)
    , m_hydro_settings(config_map)
    , m_eos(config_map)
    , m_config_map(config_map)
    , m_block_sizes(get_block_sizes<dim>(config_map))
    , m_brick_sizes(get_brick_sizes<dim>(config_map))
    , m_is_brick_periodic(get_brick_periodicity<dim>(config_map))
    , m_profiling_mgr(profiling_manager)
    , m_amr_mgr(amr_mgr)
    , m_viscosity(config_map)
  {}

  virtual ~GodunovImplemBase() = default;

  //! resize only auxiliary data array (implementation dependent)
  virtual void
  resize_auxiliary_data() = 0;

  //! memory footprint monitoring
  virtual uint64_t
  total_mem_size_in_bytes() = 0;

  virtual void
  do_time_step(DataArrayBlock_t const & U, DataArrayBlock_t const & U2, real_t dt) = 0;

  // =====================================================================
  // =====================================================================
  //! fills ghost octants with primitive variables from MPI exchange
  void
  mpi_exchange_mirrors_and_ghosts([[maybe_unused]] DataArrayGhostedBlock_t const & q_ghosted_mg)
  {

    // This fence ensure that buffer q_ghosted_mg (output of ConvertToPrimitivesVariablesFunctor
    // device kernel) is ready (ie. kernel has finished) before MPI exchange actually starts.
    // Note that without this fence, mesh_ghosts_echanger may start too early to send data.
    Kokkos::fence();

#ifdef KALYPSSO_CORE_USE_MPI
    KALYPSSO_PROFILING_REGION_DEVICE(m_profiling_mgr, NUM_SCHEME_EXCHANGE_Q_MIRROR_GHOST);
    this->m_amr_mgr.get_mesh_ghosts_exchanger().exchange_inplace(q_ghosted_mg);
#endif // KALYPSSO_CORE_USE_MPI

  } // mpi_exchange_mirrors_and_ghosts

  // =====================================================================
  // =====================================================================
  //! add gravity source term
  virtual void
  add_gravity_source_term(DataArrayBlock_t const & u_in, DataArrayBlock_t const & u_out, real_t dt)
  {

    KALYPSSO_PROFILING_REGION_DEVICE(m_profiling_mgr, NUM_SCHEME_GRAVITY);

    AddGravitySourceTerm<dim, device_t>::apply(
      this->m_config_map, u_in, u_out, this->m_amr_mgr.get_nb_owned(), dt);

  } // add_gravity_source_term

  /* useful data provided by solver */
  //! Parallel environment.
  ParallelEnv const & m_par_env;

  //! hydrodynamics parameters settings
  HydroParams const & m_params;

  //! Hydro settings
  const HydroSettings m_hydro_settings;

  //! EOS parameters
  EosWrapper<device_t> const m_eos;

  //! unordered map of parameters read from input ini file
  ConfigMap const & m_config_map;

  //! block sizes
  const block_size_t<dim> m_block_sizes;

  //! p4est brick connectivity sizes
  const brick_size_t<dim> m_brick_sizes;

  //! array of bool to tell if mesh is periodic or not
  const Kokkos::Array<bool, dim> m_is_brick_periodic;

  //! Profiling manager
  ProfilingManager & m_profiling_mgr;

  //! AMR manager
  core::AMRManager<dim, device_t> & m_amr_mgr;

  //! viscosity parameter
  ViscosityParams m_viscosity;

}; // class GodunovImplemBase

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_GODUNOV_IMPLEM_BASE_H_
