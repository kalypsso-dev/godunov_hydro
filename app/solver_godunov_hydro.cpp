// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file solver_godunov_hydro.cpp
 * \brief kalypsso hydro solver.
 */
#include <cstdlib>
#include <cstdio>
#include <string>

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>

#include <kalypsso/core/real_type.h>   // choose between single and double precision
#include <kalypsso/core/HydroParams.h> // read parameter file

#ifdef KALYPSSO_CORE_USE_MPI
#  include <mpi.h>
#endif // KALYPSSO_CORE_USE_MPI

#include <kalypsso/utils/mpi/ParallelEnv.h>

#include <kalypsso/core/SolverBase.h>
#include <godunov_hydro/SolverGodunovHydro.h>
// #include <godunov_hydro/init/InitSod.h>
#include <godunov_hydro/init/InitIsentropicVortex.h>
#include <godunov_hydro/init/InitBreakingWave.h>
#include <godunov_hydro/utils/ComputeRadialProfileFunctor.h>

#include <kalypsso/utils/config/ConfigMap.h>

#include <kalypsso/core/ComputeError.h>
#include <kalypsso/core/OutputParams.h>
#include <kalypsso/core/ComputeDataSliceAlongLine.h>

#include <kalypsso/core/cmdline_utils.h>
#include <kalypsso/utils/log/kalypsso_log.h>

// banner
#include "kalypsso_core_version.h"
#include <kalypsso/core/kalypsso_core_git_info.h>
#include <kalypsso/core/kalypsso_core_build_info.h>

#ifdef KALYPSSO_CORE_USE_CPPTRACE
#  include <kalypsso/core/cpptrace_utils.h>
#endif // KALYPSSO_CORE_USE_CPPTRACE

namespace kalypsso
{

/* ============================================================ */
/* ============================================================ */
/* ============================================================ */
template <size_t dim, typename device_t>
void
run_simulation(ParallelEnv const &       par_env,
               ConfigMap const &         config_map,
               [[maybe_unused]] int &    argc,
               [[maybe_unused]] char **& argv)
{

  // test: create a HydroParams object
  HydroParams params = HydroParams(config_map);

  // initialize workspace memory (U, U2, ...)
  auto solver =
    godunov_hydro::SolverGodunovHydro<dim, device_t>::create(par_env, params, config_map);

  // some monitoring prints
  {
    const auto eos_type = core::eos::get_eos_type(config_map);
    KALYPSSO_INFO("EOS type : {}", eos_type._to_string());
  }

  // diagnostics
  const auto conservativity_check_enabled =
    config_map.getBool("diagnostic", "conservativity_checks", true);

  // start computation
  if (par_env.rank() == 0)
  {
    KALYPSSO_INFO("Start computation....");
  }

  solver->profiling_mgr().get_whole_region().start();

  // register conservative integral values at initial time
  if (conservativity_check_enabled)
    solver->register_volume_integrals(true);

  // static AMR (froze AMR mesh computed in initial condition) ?
  const auto static_amr = config_map.getBool("amr", "static", false);
  if (static_amr)
  {
    solver->deactivate_amr_cycle();
  }

  // Hydrodynamics solver time loop
  solver->run();

  // register conservative integral values at final time
  if (conservativity_check_enabled)
  {
    solver->register_volume_integrals(false);
    solver->print_conservativity_check_report();
  }

  // save last time step as a regular checkpoint
  const auto output_params = OutputParams(config_map);
  if (output_params.nOutput != 0)
  {
    // save solution, this is not a pure checkpoint
    // just a regular output with all required fields for a checkpoint
    constexpr bool pure_checkpoint = false;
    solver->save_solution(pure_checkpoint);

    // save p4est mesh
    auto derived_solver = dynamic_cast<godunov_hydro::SolverGodunovHydro<dim, device_t> *>(solver);
    const std::string mesh_filename = derived_solver->output_basename() + ".p4est";
    derived_solver->template save_p4est_mesh<dim>(mesh_filename,
                                                  derived_solver->amr_mesh()->forest());
  }

  solver->profiling_mgr().get_whole_region().stop();

  // =================================================================
  // here we do something specific to a given test case
  // =================================================================

  // =================================================================================
  // when doing a shock tube problem, dump a 1D slice of data
  // =================================================================================
  if (!solver->problem_name().compare("sod"))
  {
    auto solver_hydro = dynamic_cast<godunov_hydro::SolverGodunovHydro<dim, device_t> *>(solver);

    // 1. save conservative variables
    {
      const auto cell_var_ids = std::vector<int32_t>{
        solver_hydro->hydro().get_fieldmap()[core::models::Hydro::ID],
        solver_hydro->hydro().get_fieldmap()[core::models::Hydro::IP],
        solver_hydro->hydro().get_fieldmap()[core::models::Hydro::IU],
        solver_hydro->hydro().get_fieldmap()[core::models::Hydro::IV],
        solver_hydro->hydro().get_fieldmap()[core::models::Hydro::IW],
      };

      const auto cell_var_names =
        std::vector<std::string>{ "rho", "e_tot", "rhou", "rhov", "rhow" };

      kalypsso::core::ComputeDataSliceAlongLine<dim, device_t>::apply(
        solver_hydro->U(),
        0,
        solver_hydro->mesh_map()->get_amr_mesh_info().local_num_quadrants(),
        IX,
        solver_hydro->mesh_map()->orchard_keys(),
        cell_var_ids,
        cell_var_names,
        "sod",
        par_env,
        config_map);
    }

    // 2. save other valuable quantities: pressure
    {
      const auto thermal_pressure =
        solver_hydro->get_derived_quantity(godunov_hydro::DERIVED_QUANTITY::THERMAL_PRESSURE);

      const auto cell_var_ids = std::vector<int32_t>{ 0 };
      const auto cell_var_names = std::vector<std::string>{ "pressure" };
      kalypsso::core::ComputeDataSliceAlongLine<dim, device_t>::apply(
        thermal_pressure,
        0,
        solver_hydro->mesh_map()->get_amr_mesh_info().local_num_quadrants(),
        IX,
        solver_hydro->mesh_map()->orchard_keys(),
        cell_var_ids,
        cell_var_names,
        "sod",
        par_env,
        config_map);
    }

    // 3. save other valuable quantities: speed of sound
    {
      const auto speed_of_sound =
        solver_hydro->get_derived_quantity(godunov_hydro::DERIVED_QUANTITY::SPEED_OF_SOUND);

      const auto cell_var_ids = std::vector<int32_t>{ 0 };
      const auto cell_var_names = std::vector<std::string>{ "speed_of_sound" };
      kalypsso::core::ComputeDataSliceAlongLine<dim, device_t>::apply(
        speed_of_sound,
        0,
        solver_hydro->mesh_map()->get_amr_mesh_info().local_num_quadrants(),
        IX,
        solver_hydro->mesh_map()->orchard_keys(),
        cell_var_ids,
        cell_var_names,
        "sod",
        par_env,
        config_map);
    }
  }
  // =================================================================================
  // when doing the isentropic vortex test case, compute error versus exact solution
  // =================================================================================
  else if (!solver->problem_name().compare("isentropic_vortex"))
  {

    if constexpr (dim == 2)
    {
      auto solver2d = dynamic_cast<godunov_hydro::SolverGodunovHydro<2, device_t> *>(solver);
      auto U = solver2d->U();
      auto U2 = solver2d->U2();

      // re-initialize U2 with exact solution
      godunov_hydro::InitIsentropicVortexDataFunctor<2, device_t>::apply(
        solver2d->U2(),
        solver2d->hydro().get_fieldmap(),
        solver2d->mesh_map()->orchard_keys(),
        solver2d->amr_mesh()->local_num_quadrants(),
        solver2d->hydro_settings(),
        config_map);

      const int  varId = core::models::Hydro::ID;
      const auto error_L1 =
        ComputeError<dim, device_t>::apply(par_env, U, U2, varId, NormType::L1, false);
      const auto error_L2 =
        ComputeError<dim, device_t>::apply(par_env, U, U2, varId, NormType::L2, false);
      if (par_env.rank() == 0)
      {
        const auto bsize = config_map.getInteger("amr", "bx", 1);
        const auto level_min = config_map.getInteger("amr", "level_min", -1);
        const auto level_max = config_map.getInteger("amr", "level_max", -1);

        KALYPSSO_INFO("test isentropic vortex for level_min={}, level_max={}, block_size={}, error "
                      "L1={:6.4e}, error L2={:6.4e}",
                      static_cast<int>(level_min),
                      static_cast<int>(level_max),
                      static_cast<int>(bsize),
                      error_L1,
                      error_L2);
      }
    } // dim == 2
  }
  // =================================================================================
  // when doing the breaking wave test case, compute error versus exact solution
  // =================================================================================
  else if (!solver->problem_name().compare("breaking_wave"))
  {
    if constexpr (dim == 2)
    {
      auto solver2d = dynamic_cast<godunov_hydro::SolverGodunovHydro<2, device_t> *>(solver);
      auto U = solver2d->U();
      auto U2 = solver2d->U2();

      // copy U into U2, because it will be used inside InitBreakingWaveDataFunctor to compute exact
      // solution from final time
      Kokkos::deep_copy(U2.logical_view(), U.logical_view());

      // re-initialize U2 with exact solution at time tEnd
      const auto tEnd = config_map.getReal("run", "tEnd", KALYPSSO_NUM(0.0));
      godunov_hydro::InitBreakingWaveDataFunctor<2, device_t>::apply(
        solver2d->U2(),
        solver2d->mesh_map()->orchard_keys(),
        solver2d->amr_mesh()->local_num_quadrants(),
        solver2d->hydro_settings(),
        solver2d->hydro().get_fieldmap(),
        tEnd,
        config_map);

      const int  varId = core::models::Hydro::IU;
      const auto error_L1 =
        ComputeError<dim, device_t>::apply(par_env, U, U2, varId, NormType::L1, true);
      const auto error_L2 =
        ComputeError<dim, device_t>::apply(par_env, U, U2, varId, NormType::L2, true);
      if (par_env.rank() == 0)
      {
        const auto bsize = config_map.getInteger("amr", "bx", 1);
        const auto level_min = config_map.getInteger("amr", "level_min", -1);
        const auto level_max = config_map.getInteger("amr", "level_max", -1);

        KALYPSSO_INFO("test breaking wave for level_min={}, level_max={}, block_size={}, error "
                      "L1={:6.4e}, error L2={:6.4e}\n",
                      static_cast<int>(level_min),
                      static_cast<int>(level_max),
                      static_cast<int>(bsize),
                      error_L1,
                      error_L2);
      }
    } // dim == 2
  }
  // =================================================================================
  // when doing sedov blast test case, dump radial profile
  // =================================================================================
  else if (!solver->problem_name().compare("blast") and
           config_map.getBool("blast", "compute_radial_profile", false))
  {
    // in order to compare numerical and analytical solutions of the sedov blast test, we need to
    // compute the radial profile
    auto solver_hydro = dynamic_cast<godunov_hydro::SolverGodunovHydro<dim, device_t> *>(solver);

    godunov_hydro::ComputeRadialProfileFunctor<dim, device_t>::apply(
      par_env,
      config_map,
      solver_hydro->mesh_map()->orchard_keys(),
      solver_hydro->mesh_map()->get_amr_mesh_info().local_num_quadrants(),
      solver_hydro->hydro().get_fieldmap(),
      solver_hydro->block_sizes(),
      solver_hydro->U());
  }
  // =================================================================

  KALYPSSO_INFO("final time is {:010.2f}", solver->current_time());
  KALYPSSO_INFO("");

  solver->print_monitoring_info_final();

  delete solver;

} // run_simulation

} // namespace kalypsso

// ===============================================================
// ===============================================================
// ===============================================================
int
main(int argc, char * argv[])
{

  {
    // create parallel environment (p4est, MPI, kokkos, ...)
    kalypsso::ParallelEnv par_env(argc, argv);

#ifdef KALYPSSO_CORE_USE_SPDLOG
    // logger setup
    kalypsso::kalypsso_spdlog_config(argc, argv, par_env.rank(), par_env.size());
#endif

#ifdef KALYPSSO_CORE_USE_CPPTRACE
    kalypsso::cpptrace_initialize();
#endif // KALYPSSO_CORE_USE_CPPTRACE

    // parse command line arguments
    if (kalypsso::cmdline_arg_exists(argv, argv + argc, "--version"))
    {
      if (par_env.rank() == 0)
      {
        kalypsso::GitRevisionInfo::print();
        kalypsso::BuildInfo::print();
      }
      return EXIT_SUCCESS;
    }
    else if (kalypsso::cmdline_arg_exists(argv, argv + argc, "--help"))
    {
      if (par_env.rank() == 0)
      {
        // clang-format off
        std::cout << "Example cmdline: \"mpirun -np 1 ./solver_godunov_hydro --ini test_blast_2D_block.ini\"\n";
        // clang-format on
      }
      return EXIT_SUCCESS;
    }

    // print kalypsso banner
    if (par_env.rank() == 0)
    {
      kalypsso::GitRevisionInfo::print();
      kalypsso::BuildInfo::print();
    }

    // check if user passed a custom ini filename
    // provide a default input filename if user did'nt set one
    std::string input_filename = kalypsso::cmdline_get_string(argv, argv + argc, "--ini");

    if (input_filename.size() == 0)
      input_filename = "test_blast_2D_block.ini";

    // only MPI rank 0 actually reads input file, and broadcast it to all other MPI processor
    kalypsso::ConfigMap config_map = kalypsso::broadcast_parameters(input_filename);

    const auto dim = kalypsso::get_dim(config_map);
    assertm(dim == 2 or dim == 3, "[solver_godunov_hydro] Wrong dimension");

    // run simulation
    if (dim == 2)
    {
      kalypsso::run_simulation<2, kalypsso::DefaultDevice>(par_env, config_map, argc, argv);
    }
    else if (dim == 3)
    {
      kalypsso::run_simulation<3, kalypsso::DefaultDevice>(par_env, config_map, argc, argv);
    }
  }

  return EXIT_SUCCESS;

} // end main
