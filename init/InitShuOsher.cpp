// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitShuOsher.cpp
 */

#include <godunov_hydro/init/InitShuOsher.h>
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
InitShuOsherDataFunctor<dim, device_t>::apply(DataArrayBlock_t const &             Udata,
                                              orchard_key_view_t<device_t> const & orchard_keys,
                                              int32_t               local_num_octants,
                                              HydroSettings const & settings,
                                              ConfigMap const &     config_map)
{
  ShuOsherParams shParams = ShuOsherParams(config_map);

  // data init functor
  InitShuOsherDataFunctor functor(
    Udata, orchard_keys, local_num_octants, settings, shParams, config_map);

  // compute total number of cells
  const auto nbCellsPerLeaf = Udata.num_cells();
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  Kokkos::parallel_for("kalypsso::godunov_hydro::InitShuOsherDataFunctor",
                       Kokkos::RangePolicy<exec_space>(0, nbCellsTotal),
                       functor);

} // InitShuOsherDataFunctor::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitShuOsherDataFunctor<dim, device_t>::operator()(const int32_t & global_index) const
{

  // convert global index into
  // - octant id
  // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
  const auto iOct = global_index / m_Udata.num_cells();
  const auto cell_index = global_index - iOct * m_Udata.num_cells();

  const auto block_sizes = m_Udata.block_size();

  // compute ix,iy,iz of local cell inside
  // block from index
  auto iCoord = cellindex_to_coord<dim>(cell_index, block_sizes);

  // get block orchard key
  const auto key = m_orchard_keys(iOct);

  // compute physical x,y,z for that cell (cell center)
  const auto xyz_vertex = orchard_key_to_cell_coord<dim>(key, iCoord, block_sizes[IX]);

  auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, m_scaling_factor, m_xyz_min);

  // initialize shock and density sine wave
  if (xyz[IX] < m_shParams.x0)
  {
    m_Udata(cell_index, Hydro<dim>::ID, iOct) = m_shParams.rhoL;
    m_Udata(cell_index, Hydro<dim>::IU, iOct) = m_shParams.rhoL * m_shParams.uL;
    m_Udata(cell_index, Hydro<dim>::IE, iOct) =
      m_eos_wrapper.volumic_eint_from_pressure(m_shParams.pL, m_shParams.rhoL) +
      HALF_F * m_shParams.rhoL * m_shParams.uL * m_shParams.uL;
  }
  else
  {
    const auto rho = m_shParams.rhoR + m_shParams.A * sin(m_shParams.k * xyz[IX]);
    m_Udata(cell_index, Hydro<dim>::ID, iOct) = rho;
    m_Udata(cell_index, Hydro<dim>::IU, iOct) = rho * m_shParams.uR;
    m_Udata(cell_index, Hydro<dim>::IE, iOct) =
      m_eos_wrapper.volumic_eint_from_pressure(m_shParams.pR, rho) +
      HALF_F * rho * m_shParams.uR * m_shParams.uR;
  }

  m_Udata(cell_index, Hydro<dim>::IV, iOct) = ZERO_F;

  if constexpr (dim == 3)
    m_Udata(cell_index, Hydro<dim>::IW, iOct) = ZERO_F;

} // end InitShuOsherDataFunctor::operator ()

template class InitShuOsherDataFunctor<2, kalypsso::DefaultDevice>;
template class InitShuOsherDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitShuOsherRefineFunctor<dim, device_t>::apply(DataArrayBlock_t const &             Udata,
                                                orchard_key_view_t<device_t> const & orchard_keys,
                                                amrflags_view_t const &              amrflags,
                                                int32_t               local_num_octants,
                                                HydroSettings const & settings,
                                                int                   level_refine,
                                                ConfigMap const &     config_map)
{
  auto shParams = ShuOsherParams(config_map);

  // iterate functor for refinement
  InitShuOsherRefineFunctor functor(
    Udata, orchard_keys, amrflags, local_num_octants, settings, shParams, level_refine, config_map);

  const auto refine_type = core::get_init_indicator(config_map);

  if (refine_type == +core::InitConditionsIndicator::ALWAYS_REFINE)
  {
    Kokkos::parallel_for("kalypsso::godunov_hydro::InitShuOsherRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineAlways>(0, local_num_octants),
                         functor);
  }
  else if (refine_type == +core::InitConditionsIndicator::GEOMETRIC)
  {
    Kokkos::parallel_for("kalypsso::godunov_hydro::InitShuOsherRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineGeometric>(0, local_num_octants),
                         functor);
  }
  else
  {
    KALYPSSO_ERROR("Unknown value for refine indicator method.");
  }

} // InitShuOsherRefineFunctor::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitShuOsherRefineFunctor<dim, device_t>::operator()(TagRefineAlways const &,
                                                     const size_t & iOct) const
{
  m_amrflags(iOct) = AMRContextBase::KALYPSSO_DO_REFINE;
}

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitShuOsherRefineFunctor<dim, device_t>::operator()(TagRefineGeometric const &,
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

  // only look at level - 1
  if (level == m_level_refine)
  {

    // compute physical x,y,z for the block center
    constexpr auto centering = true;
    const auto     xyz_vertex = orchard_key_to_vertex_coord<dim>(key, centering);
    const auto     xyz = vertex_coord_to_real_space<dim>(xyz_vertex, m_scaling_factor, m_xyz_min);

    // compute distance to interface
    const auto d = fabs(xyz[IX] - m_shParams.x0);

    if (d < (block_length * KALYPSSO_NUM(0.95)))
      flag = AMRContextBase::KALYPSSO_DO_REFINE;

  } // end if level == level_refine

  // perform max reduction
  // if all cell in current block agree on COARSEN => do coarsen
  // if a single cell in current block disagree on coarsening => do nothing or refine
  // if a single cell in current block needs to refine => do refine
  m_amrflags(iOct) = flag;

} // end InitShuOsherRefineFunctor::operator ()

template class InitShuOsherRefineFunctor<2, kalypsso::DefaultDevice>;
template class InitShuOsherRefineFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitShuOsher<dim, device_t>::apply(SolverGodunovHydro<dim, device_t> & solver)
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
  InitShuOsherDataFunctor<dim, device_t>::apply(solver.U(),
                                                solver.mesh_map()->orchard_keys(),
                                                solver.amr_mesh()->local_num_quadrants(),
                                                settings,
                                                config_map);

  const auto refine_type = core::get_init_indicator(config_map);

  if (refine_type == +core::InitConditionsIndicator::SAME_AS_REGULAR_DYNAMICS)
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
      InitShuOsherDataFunctor<dim, device_t>::apply(solver.U(),
                                                    solver.mesh_map()->orchard_keys(),
                                                    solver.amr_mesh()->local_num_quadrants(),
                                                    settings,
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
      InitShuOsherRefineFunctor<dim, device_t>::apply(solver.U(),
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
      InitShuOsherDataFunctor<dim, device_t>::apply(solver.U(),
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
} // InitShuOsher::apply

template class InitShuOsher<2, kalypsso::DefaultDevice>;
template class InitShuOsher<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso
