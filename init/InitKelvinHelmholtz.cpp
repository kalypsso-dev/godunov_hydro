// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitKelvinHelmholtz.cpp
 */

#include <godunov_hydro/init/InitKelvinHelmholtz.h>
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
InitKelvinHelmholtzDataFunctor<dim, device_t>::apply(
  DataArrayBlock_t const &             Udata,
  orchard_key_view_t<device_t> const & orchard_keys,
  int32_t                              local_num_octants,
  HydroSettings const &                settings,
  ConfigMap const &                    config_map)
{
  auto khParams = KHParams(config_map);

  // data init functor
  InitKelvinHelmholtzDataFunctor functor(
    Udata, orchard_keys, local_num_octants, settings, khParams, config_map);

  // compute total number of cells
  const auto nbCellsPerLeaf = Udata.num_cells();
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  Kokkos::parallel_for("kalypsso::godunov_hydro::InitKelvinHelmholtzDataFunctor",
                       Kokkos::RangePolicy<exec_space>(0, nbCellsTotal),
                       functor);

} // InitKelvinHelmholtzDataFunctor::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitKelvinHelmholtzDataFunctor<dim, device_t>::operator()(const int32_t & global_index) const
{

  // convert global index into
  // - octant id
  // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
  const auto iOct = global_index / m_Udata.num_cells();
  const auto cell_index = global_index - iOct * m_Udata.num_cells();

  const auto block_sizes = m_Udata.block_size();

  // Kelvin Helmholtz problem parameters
  const auto d_in = m_khParams.d_in;
  const auto d_out = m_khParams.d_out;
  const auto vflow_in = m_khParams.vflow_in;
  const auto vflow_out = m_khParams.vflow_out;
  const auto ampl = m_khParams.amplitude;
  const auto pressure = m_khParams.pressure;

  // compute ix,iy,iz of local cell inside
  // block from index
  auto iCoord = cellindex_to_coord<dim>(cell_index, block_sizes);

  // get block orchard key
  const auto key = m_orchard_keys(iOct);

  // get block level
  // const auto level = orchard_key_t<dim>::level(key);

  // compute cell size (assume dx=dy=dz, i.e. block_sizes are the same along all directions)
  // const auto dx = compute_cell_length<dim>(level, block_sizes[IX]);

  // compute physical x,y,z for that cell (cell center)
  const auto xyz_vertex = orchard_key_to_cell_coord<dim>(key, iCoord, block_sizes[IX]);
  const auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, m_scaling_factor, m_xyz_min);

  // normalized coordinates in [0,1]

  if constexpr (dim == 2)
  {
    [[maybe_unused]] auto const & xmin = m_xyz_min[IX];
    [[maybe_unused]] auto const & xmax = m_xyz_max[IX];
    auto const &                  ymin = m_xyz_min[IY];
    auto const &                  ymax = m_xyz_max[IY];

    [[maybe_unused]] const auto xn = (xyz[IX] - xmin) / (xmax - xmin);
    const auto                  yn = (xyz[IY] - ymin) / (ymax - ymin);

    if (m_khParams.p_rand)
    {

      // get random number state
      rng_state_t rand_gen = m_rand_pool.get_state();

      real_t d, u, v;

      if (yn < KALYPSSO_NUM(0.25) or yn > KALYPSSO_NUM(0.75))
      {

        d = d_out;
        u = vflow_out;
        v = KALYPSSO_NUM(0.0);
      }
      else
      {

        d = d_in;
        u = vflow_in;
        v = KALYPSSO_NUM(0.0);
      }

      u += ampl * (static_cast<real_t>(rand_gen.drand()) - HALF_F);
      v += ampl * (static_cast<real_t>(rand_gen.drand()) - HALF_F);

      m_Udata(cell_index, Hydro<dim>::ID, iOct) = d;
      m_Udata(cell_index, Hydro<dim>::IU, iOct) = d * u;
      m_Udata(cell_index, Hydro<dim>::IV, iOct) = d * v;
      m_Udata(cell_index, Hydro<dim>::IP, iOct) =
        m_eos_wrapper.volumic_eint_from_pressure(pressure, d) + HALF_F * d * (u * u + v * v);

      // free random number, so that it can be used by other threads later
      m_rand_pool.free_state(rand_gen);
    }
    else if (m_khParams.p_sine_rob)
    {

      const int    n = m_khParams.mode;
      const real_t w0 = m_khParams.w0;
      const real_t delta = m_khParams.delta;
      const real_t y1 = KALYPSSO_NUM(0.25);
      const real_t y2 = KALYPSSO_NUM(0.75);
      const real_t rho1 = d_in;
      const real_t rho2 = d_out;
      const real_t v1 = vflow_in;
      const real_t v2 = vflow_out;

      const real_t ramp = ONE_F / (ONE_F + exp(2 * (xyz[IY] - y1) / delta)) +
                          ONE_F / (ONE_F + exp(2 * (y2 - xyz[IY]) / delta));

      const real_t d = rho1 + ramp * (rho2 - rho1);
      const real_t u = v1 + ramp * (v2 - v1);
      const real_t v = w0 * sin(static_cast<real_t>(n) * PI_F * xyz[IX]);

      m_Udata(cell_index, Hydro<dim>::ID, iOct) = d;
      m_Udata(cell_index, Hydro<dim>::IU, iOct) = d * u;
      m_Udata(cell_index, Hydro<dim>::IV, iOct) = d * v;
      m_Udata(cell_index, Hydro<dim>::IP, iOct) =
        m_eos_wrapper.volumic_eint_from_pressure(pressure, d) + HALF_F * d * (u * u + v * v);
    }
    else if (m_khParams.p_sine_stone)
    {
      // remember that domain must [0.0, 1.0] x [0.0, 1.0]
      auto constexpr L = KALYPSSO_NUM(0.01);
      auto constexpr sigma = KALYPSSO_NUM(0.2);

      auto const ytilde = fabs(yn - HALF_F) - KALYPSSO_NUM(0.25);

      auto const d = KALYPSSO_NUM(1.5) - HALF_F * tanh(ytilde / L);
      auto const u = HALF_F * tanh(ytilde / L);
      auto const v = ampl * cos(4 * PI_F * (xn - HALF_F)) * exp(-ytilde * ytilde / sigma / sigma);

      m_Udata(cell_index, Hydro<dim>::ID, iOct) = d;
      m_Udata(cell_index, Hydro<dim>::IU, iOct) = d * u;
      m_Udata(cell_index, Hydro<dim>::IV, iOct) = d * v;
      m_Udata(cell_index, Hydro<dim>::IP, iOct) =
        m_eos_wrapper.volumic_eint_from_pressure(pressure, d) + HALF_F * d * (u * u + v * v);
    }
  }
  else if constexpr (dim == 3)
  {
    [[maybe_unused]] auto const & xmin = m_xyz_min[IX];
    [[maybe_unused]] auto const & xmax = m_xyz_max[IX];
    [[maybe_unused]] auto const & ymin = m_xyz_min[IY];
    [[maybe_unused]] auto const & ymax = m_xyz_max[IY];
    auto const &                  zmin = m_xyz_min[IZ];
    auto const &                  zmax = m_xyz_max[IZ];

    [[maybe_unused]] const auto xn = (xyz[IX] - xmin) / (xmax - xmin);
    [[maybe_unused]] const auto yn = (xyz[IY] - ymin) / (ymax - ymin);
    const auto                  zn = (xyz[IZ] - zmin) / (zmax - zmin);

    if (m_khParams.p_rand)
    {

      // get random number state
      rng_state_t rand_gen = m_rand_pool.get_state();

      real_t d, u, v, w;

      if (zn < KALYPSSO_NUM(0.25) or zn > KALYPSSO_NUM(0.75))
      {
        d = d_out;
        u = vflow_out;
        v = KALYPSSO_NUM(0.0);
        w = KALYPSSO_NUM(0.0);
      }
      else
      {
        d = d_in;
        u = vflow_in;
        v = KALYPSSO_NUM(0.0);
        w = KALYPSSO_NUM(0.0);
      }

      u += ampl * (static_cast<real_t>(rand_gen.drand()) - HALF_F);
      v += ampl * (static_cast<real_t>(rand_gen.drand()) - HALF_F);
      w += ampl * (static_cast<real_t>(rand_gen.drand()) - HALF_F);

      m_Udata(cell_index, Hydro<dim>::ID, iOct) = d;
      m_Udata(cell_index, Hydro<dim>::IU, iOct) = d * u;
      m_Udata(cell_index, Hydro<dim>::IV, iOct) = d * v;
      m_Udata(cell_index, Hydro<dim>::IW, iOct) = d * w;
      m_Udata(cell_index, Hydro<dim>::IP, iOct) =
        m_eos_wrapper.volumic_eint_from_pressure(pressure, d) +
        HALF_F * d * (u * u + v * v + w * w);

      // free random number, so that it can be used by other threads later
      m_rand_pool.free_state(rand_gen);
    }
    else if (m_khParams.p_sine_rob)
    {

      const int    n = m_khParams.mode;
      const real_t w0 = m_khParams.w0;
      const real_t delta = m_khParams.delta;

      const real_t z1 = KALYPSSO_NUM(0.25);
      const real_t z2 = KALYPSSO_NUM(0.75);

      const real_t rho1 = d_in;
      const real_t rho2 = d_out;

      const real_t v1x = vflow_in;
      const real_t v2x = vflow_out;

      const real_t v1y = vflow_in / 2;
      const real_t v2y = vflow_out / 2;

      const real_t ramp = ONE_F / (ONE_F + exp(2 * (xyz[IZ] - z1) / delta)) +
                          ONE_F / (ONE_F + exp(2 * (z2 - xyz[IZ]) / delta));

      const real_t d = rho1 + ramp * (rho2 - rho1);
      const real_t u = v1x + ramp * (v2x - v1x);
      const real_t v = v1y + ramp * (v2y - v1y);
      const real_t w = w0 * sin(static_cast<real_t>(n) * PI_F * xyz[IX]) *
                       sin(static_cast<real_t>(n) * PI_F * xyz[IY]);

      m_Udata(cell_index, Hydro<dim>::ID, iOct) = d;
      m_Udata(cell_index, Hydro<dim>::IU, iOct) = d * u;
      m_Udata(cell_index, Hydro<dim>::IV, iOct) = d * v;
      m_Udata(cell_index, Hydro<dim>::IW, iOct) = d * w;
      m_Udata(cell_index, Hydro<dim>::IP, iOct) =
        m_eos_wrapper.volumic_eint_from_pressure(pressure, d) +
        HALF_F * d * (u * u + v * v + w * w);
    }
    else if (m_khParams.p_sine_stone)
    {
      // remember that domain must [0.0, 1.0] x [0.0, 1.0]
      auto constexpr L = KALYPSSO_NUM(0.01);
      auto constexpr sigma = KALYPSSO_NUM(0.2);

      auto const ytilde = fabs(yn - HALF_F) - KALYPSSO_NUM(0.25);

      auto const d = KALYPSSO_NUM(1.5) - HALF_F * tanh(ytilde / L);
      auto const u = HALF_F * tanh(ytilde / L);
      auto const v = ampl * cos(4 * PI_F * (xn - HALF_F)) * exp(-ytilde * ytilde / sigma / sigma);
      auto const w = ZERO_F;

      m_Udata(cell_index, Hydro<dim>::ID, iOct) = d;
      m_Udata(cell_index, Hydro<dim>::IU, iOct) = d * u;
      m_Udata(cell_index, Hydro<dim>::IV, iOct) = d * v;
      m_Udata(cell_index, Hydro<dim>::IW, iOct) = d * w;
      m_Udata(cell_index, Hydro<dim>::IP, iOct) =
        m_eos_wrapper.volumic_eint_from_pressure(pressure, d) +
        HALF_F * d * (u * u + v * v + w * w);
    }
  }
} // end InitKelvinHelmholtzDataFunctor::operator ()

template class InitKelvinHelmholtzDataFunctor<2, kalypsso::DefaultDevice>;
template class InitKelvinHelmholtzDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitKelvinHelmholtzRefineFunctor<dim, device_t>::apply(
  DataArrayBlock_t const &             Udata,
  orchard_key_view_t<device_t> const & orchard_keys,
  amrflags_view_t const &              amrflags,
  int32_t                              local_num_octants,
  HydroSettings const &                settings,
  int                                  level_refine,
  ConfigMap const &                    config_map)
{
  auto khParams = KHParams(config_map);

  // iterate functor for refinement
  InitKelvinHelmholtzRefineFunctor functor(
    Udata, orchard_keys, amrflags, local_num_octants, settings, khParams, level_refine, config_map);

  const auto refine_type = core::get_init_indicator(config_map);

  if (refine_type == +core::InitConditionsIndicator::ALWAYS_REFINE)
  {
    Kokkos::parallel_for("kalypsso::godunov_hydro::InitKelvinHelmholtzRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineAlways>(0, local_num_octants),
                         functor);
  }
  else if (refine_type == +core::InitConditionsIndicator::GEOMETRIC)
  {
    Kokkos::parallel_for("kalypsso::godunov_hydro::InitKelvinHelmholtzRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineGeometric>(0, local_num_octants),
                         functor);
  }
  else
  {
    KALYPSSO_ERROR("Unknown value for refine indicator method.");
  }

} // InitKelvinHelmholtzRefineFunctor::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitKelvinHelmholtzRefineFunctor<dim, device_t>::operator()(TagRefineAlways const &,
                                                            const size_t & iOct) const
{
  m_amrflags(iOct) = AMRContextBase::KALYPSSO_DO_REFINE;
} // InitKelvinHelmholtzRefineFunctor::operator()

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitKelvinHelmholtzRefineFunctor<dim, device_t>::operator()(TagRefineGeometric const &,
                                                            const size_t & iOct) const
{

  // get block orchard key
  const auto key = m_orchard_keys(iOct);

  // get block level
  const auto level = orchard_key_t<dim>::level(key);

  // compute block length (in real space units)
  const auto block_length = compute_block_length<dim>(level) * m_scaling_factor;

  // default : do nothing, i.e. neither refine or coarsen
  auto flag = AMRContextBase::KALYPSSO_DO_NOTHING;

  const auto   y1 = KALYPSSO_NUM(0.25);
  const auto   y2 = KALYPSSO_NUM(0.75);
  auto const & ymin = m_xyz_min[IY];
  auto const & ymax = m_xyz_max[IY];

  // only look at level - 1
  if (level == m_level_refine)
  {

    // compute physical x,y,z for the block center
    constexpr auto centering = true;
    const auto     xyz_vertex = orchard_key_to_vertex_coord<dim>(key, centering);
    const auto     xyz = vertex_coord_to_real_space<dim>(xyz_vertex, m_scaling_factor, m_xyz_min);
    const auto     yn = (xyz[IY] - ymin) / (ymax - ymin);

    if constexpr (dim == 2)
    {
      auto d1 = fabs(yn - y1);
      auto d2 = fabs(yn - y2);

      if (d1 < (block_length * KALYPSSO_NUM(0.95)) or d2 < (block_length * KALYPSSO_NUM(0.95)))
        flag = AMRContextBase::KALYPSSO_DO_REFINE;
    }
    else if constexpr (dim == 3)
    {
      auto const & zmin = m_xyz_min[IZ];
      auto const & zmax = m_xyz_max[IZ];
      const auto   zn = (xyz[IZ] - zmin) / (zmax - zmin);
      auto         d1 = fabs(zn - y1);
      auto         d2 = fabs(zn - y2);

      if (d1 < (block_length * KALYPSSO_NUM(0.95)) or d2 < (block_length * KALYPSSO_NUM(0.95)))
        flag = AMRContextBase::KALYPSSO_DO_REFINE;
    }

  } // end if level == level_refine

  // perform max reduction
  // if all cell in current block agree on COARSEN => do coarsen
  // if a single cell in current block disagree on coarsening => do nothing or refine
  // if a single cell in current block needs to refine => do refine
  m_amrflags(iOct) = flag;

} // end InitKelvinHelmholtzRefineFunctor::operator ()

template class InitKelvinHelmholtzRefineFunctor<2, kalypsso::DefaultDevice>;
template class InitKelvinHelmholtzRefineFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitKelvinHelmholtz<dim, device_t>::apply(SolverGodunovHydro<dim, device_t> & solver)
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

  // first init of Udata
  InitKelvinHelmholtzDataFunctor<dim, device_t>::apply(solver.U(),
                                                       solver.mesh_map()->orchard_keys(),
                                                       solver.amr_mesh()->local_num_quadrants(),
                                                       settings,
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
      InitKelvinHelmholtzDataFunctor<dim, device_t>::apply(solver.U(),
                                                           solver.mesh_map()->orchard_keys(),
                                                           solver.amr_mesh()->local_num_quadrants(),
                                                           settings,
                                                           config_map);

      // update level
      ++level;

    } // end while level<level_max
  }
  else // use custom initial refine criterion (either GEOMETRIC or ALWAYS_REFINE)
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
      InitKelvinHelmholtzRefineFunctor<dim, device_t>::apply(
        solver.U(),
        solver.mesh_map()->orchard_keys(),
        flags_d,
        solver.amr_mesh()->local_num_quadrants(),
        settings,
        level,
        solver.config_map());

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
      InitKelvinHelmholtzDataFunctor<dim, device_t>::apply(solver.U(),
                                                           solver.mesh_map()->orchard_keys(),
                                                           solver.amr_mesh()->local_num_quadrants(),
                                                           settings,
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

} // InitKelvinHelmholtz::apply

template class InitKelvinHelmholtz<2, kalypsso::DefaultDevice>;
template class InitKelvinHelmholtz<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso
