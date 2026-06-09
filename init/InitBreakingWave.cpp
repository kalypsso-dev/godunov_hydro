// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitBreakingWave.cpp
 */

#include <godunov_hydro/init/InitBreakingWave.h>
#include <godunov_hydro/SolverGodunovHydro.h>

#include <kalypsso/core/models/utils_hydro.h>
#include <kalypsso/core/orchard_key_utils.h>

namespace kalypsso
{
namespace godunov_hydro
{

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitBreakingWaveDataFunctor<dim, device_t>::apply(DataArrayBlock_t const &             Udata,
                                                  orchard_key_view_t<device_t> const & orchard_keys,
                                                  int32_t               local_num_octants,
                                                  HydroSettings const & settings,
                                                  real_t                t_eval,
                                                  ConfigMap const &     config_map)
{
  BreakingWaveParams breakingWaveParams = BreakingWaveParams(config_map);

  // data init functor
  InitBreakingWaveDataFunctor functor(
    Udata, orchard_keys, local_num_octants, settings, breakingWaveParams, t_eval, config_map);

  // compute total number of cells
  const auto nbCellsPerLeaf = Udata.num_cells();
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  Kokkos::parallel_for("kalypsso::godunov_hydro::InitBreakingWaveDataFunctor",
                       Kokkos::RangePolicy<exec_space>(0, nbCellsTotal),
                       functor);

} // InitBreakingWaveDataFunctor<dim, device_t>::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitBreakingWaveDataFunctor<dim, device_t>::operator()(const int32_t & global_index) const
{

  // convert global index into
  // - octant id
  // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
  const auto iOct = global_index / m_Udata.num_cells();
  const auto cell_index = global_index - iOct * m_Udata.num_cells();

  const auto block_sizes = m_Udata.block_size();

  // breaking wave problem parameters
  const auto gamma = m_bwParams.gamma;

  // compute ix,iy,iz of local cell inside
  // block from index
  auto iCoord = cellindex_to_coord<dim>(cell_index, block_sizes);

  // get block orchard key
  const auto key = m_orchard_keys(iOct);

  // compute physical x,y,z for that cell (cell center)
  const auto xyz_vertex = orchard_key_to_cell_coord<dim>(key, iCoord, block_sizes[IX]);

  auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, m_scaling_factor, m_xyz_min);

  // initialize

  // if time is > 0 we compute preimage of x by advection at speed u-c
  // else just use x for a regular initialization at t=0
  auto x0 = xyz[IX];

  if (m_t_eval > 0)
  {
    // we want to compute exact solution at x=x0,t=0
    // we need first to evaluate u-c at x,t_eval
    // and to do that we need speed of sound and velocity
    HydroState<dim> hydro_state;
    hydro_state[Hydro<dim>::ID] = m_Udata(cell_index, Hydro<dim>::ID, iOct);
    hydro_state[Hydro<dim>::IP] = m_Udata(cell_index, Hydro<dim>::IP, iOct);
    hydro_state[Hydro<dim>::IU] = m_Udata(cell_index, Hydro<dim>::IU, iOct);
    hydro_state[Hydro<dim>::IV] = m_Udata(cell_index, Hydro<dim>::IV, iOct);
    if constexpr (dim == 3)
    {
      hydro_state[Hydro<dim>::IW] = m_Udata(cell_index, Hydro<dim>::IW, iOct);
    }
    real_t pf, cf;
    core::models::compute_Pressure_and_SpeedOfSound(hydro_state, pf, cf, m_settings);
    const real_t uf = hydro_state[Hydro<dim>::IU] / hydro_state[Hydro<dim>::ID];
    // get preimage by advection at speed u-c
    x0 = xyz[IX] - (uf - cf) * m_t_eval;
  }

  const auto rho = m_bwParams.rho(x0);
  const auto p = m_bwParams.p(rho);
  const auto c = m_bwParams.c(rho);
  const auto u = m_bwParams.u(c);
  const auto v = 0;
  const auto w = 0;

  // clang-format off
  const auto ekin = dim == 2 ?
    HALF_F * rho * (u * u + v * v) :
    HALF_F * rho * (u * u + v * v + w * w);
  // clang-format on

  m_Udata(cell_index, Hydro<dim>::ID, iOct) = rho;
  m_Udata(cell_index, Hydro<dim>::IP, iOct) = p / (gamma - ONE_F) + ekin;
  m_Udata(cell_index, Hydro<dim>::IU, iOct) = rho * u;
  m_Udata(cell_index, Hydro<dim>::IV, iOct) = rho * v;

  if constexpr (dim == 3)
    m_Udata(cell_index, Hydro<dim>::IW, iOct) = rho * w;

} // end InitBreakingWaveDataFunctor<dim, device_t>::operator ()

// ====================================================================
// ====================================================================
template class InitBreakingWaveDataFunctor<2, kalypsso::DefaultDevice>;
template class InitBreakingWaveDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitBreakingWave<dim, device_t>::apply(SolverGodunovHydro<dim, device_t> & solver)
{

  auto              amr_mesh = solver.amr_mesh();
  ConfigMap const & config_map = solver.config_map();
  const auto        settings = HydroSettings(solver.config_map());
  const int         level_min = solver.hydro_params().level_min;
  const int         level_max = solver.hydro_params().level_max;

  constexpr bool do_reset_ghosts = true;
  solver.update_mesh(do_reset_ghosts);

  // resize Udata
  solver.resize_solver_data();

  // initial time
  const auto t_init = ZERO_F;

  // first init of Udata
  InitBreakingWaveDataFunctor<dim, device_t>::apply(solver.U(),
                                                    solver.mesh_map()->orchard_keys(),
                                                    solver.amr_mesh()->local_num_quadrants(),
                                                    settings,
                                                    t_init,
                                                    config_map);

  const auto init_refine_type = core::get_init_indicator(config_map);

  if (init_refine_type == +core::InitConditionsIndicator::SAME_AS_REGULAR_DYNAMICS)
  {
    // iterate several refinements
    int level = level_min;
    while (level < level_max)
    {

      //
      // 1. apply amr cycle using regular refine criterion
      //
      solver.do_amr_cycle();

      //
      // 2. update Udata
      //
      InitBreakingWaveDataFunctor<dim, device_t>::apply(solver.U(),
                                                        solver.mesh_map()->orchard_keys(),
                                                        solver.amr_mesh()->local_num_quadrants(),
                                                        settings,
                                                        t_init,
                                                        config_map);

      // update level
      ++level;

    } // end while level<level_max
  }
  else
  {
    // iterate several refinements
    int level = level_min;
    while (level < level_max)
    {
      //
      // 1. create context data for AMR cycle
      //
      AMRContext<dim, device_t> amr_context(amr_mesh->forest()->local_num_quadrants);
      auto                      flags_d = amr_context.m_amrflags_d;
      auto                      flags_h = amr_context.m_amrflags_h;

      //
      // 2. compute refine/coarsen flags
      //
      InitBreakingWaveRefineFunctor<dim, device_t>::apply(
        flags_d, solver.amr_mesh()->local_num_quadrants(), solver.config_map());

      // amr context will adapt mesh on CPU, so we need flags on host up to date
      Kokkos::deep_copy(flags_h, flags_d);

      //
      // 3. apply AMR cycle on device : refine + coarsen + 2:1 balance
      //
      {
        Kokkos::Profiling::ScopedRegion prof("AMR_refinement_device");
        [[maybe_unused]] auto changed = amr_context.adapt_mesh(solver.amr_mesh()->forest());
        KALYPSSO_INFO_ALL("Mesh changed ? {}", static_cast<int>(changed));
      }

      //
      // 4. re-compute update orchard keys
      //
      solver.update_mesh(do_reset_ghosts);

      // 5. resize Udata
      // now we know the size of the mesh, we can allocate memory for
      // heavy data (U, U2, Uhost, ...)
      solver.resize_solver_data();

      //
      // 6. update Udata
      //
      InitBreakingWaveDataFunctor<dim, device_t>::apply(solver.U(),
                                                        solver.mesh_map()->orchard_keys(),
                                                        solver.amr_mesh()->local_num_quadrants(),
                                                        settings,
                                                        t_init,
                                                        config_map);

      // update level
      ++level;

    } // end while level<level_max
  }

#ifdef KALYPSSO_CORE_USE_MPI
  // load balancing (= repartitioning) the octree mesh + userdata over the MPI processes.
  // U and U2 will be resized
  solver.do_load_balancing();
#endif

} // apply

template class InitBreakingWave<2, kalypsso::DefaultDevice>;
template class InitBreakingWave<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso
