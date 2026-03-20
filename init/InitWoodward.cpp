// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitWoodward.cpp
 */

#include <godunov_hydro/init/InitWoodward.h>
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
InitWoodwardDataFunctor<dim, device_t>::apply(DataArrayBlock_t const &             Udata,
                                              FieldMap<core::models::Hydro>        fm,
                                              orchard_key_view_t<device_t> const & orchard_keys,
                                              int32_t               local_num_octants,
                                              HydroSettings const & settings,
                                              ConfigMap const &     config_map)
{
  WoodwardParams woodwardParams = WoodwardParams(config_map);

  // data init functor
  InitWoodwardDataFunctor functor(
    Udata, fm, orchard_keys, local_num_octants, settings, woodwardParams, config_map);

  // compute total number of cells
  const auto nbCellsPerLeaf = Udata.num_cells();
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  Kokkos::parallel_for("kalypsso::godunov_hydro::InitWoodwardDataFunctor",
                       Kokkos::RangePolicy<exec_space>(0, nbCellsTotal),
                       functor);

} // InitWoodwardDataFunctor::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitWoodwardDataFunctor<dim, device_t>::operator()(const int32_t & global_index) const
{

  // convert global index into
  // - octant id
  // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
  const auto iOct = global_index / m_Udata.num_cells();
  const auto cell_index = global_index - iOct * m_Udata.num_cells();

  const auto block_sizes = m_Udata.block_size();

  constexpr auto ID = core::models::Hydro::ID;
  constexpr auto IP = core::models::Hydro::IP;
  constexpr auto IU = core::models::Hydro::IU;
  constexpr auto IV = core::models::Hydro::IV;
  constexpr auto IW = core::models::Hydro::IW;

  // compute ix,iy,iz of local cell inside
  // block from index
  auto iCoord = cellindex_to_coord<dim>(cell_index, block_sizes);

  // get block orchard key
  const auto key = m_orchard_keys(iOct);

  // compute physical x,y,z for that cell (cell center)
  const auto xyz_vertex = orchard_key_to_cell_coord<dim>(key, iCoord, block_sizes[IX]);

  auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, m_scaling_factor, m_xyz_min);

  // initialize
  if (xyz[IX] < m_woodwardParams.xdL)
  {
    m_Udata(cell_index, m_fm[ID], iOct) = m_woodwardParams.rhoL;
    m_Udata(cell_index, m_fm[IU], iOct) = m_woodwardParams.uL;
    m_Udata(cell_index, m_fm[IV], iOct) = ZERO_F;
    if constexpr (dim == 3)
      m_Udata(cell_index, m_fm[IW], iOct) = ZERO_F;

    const auto ekin = HALF_F *
                      (m_Udata(cell_index, m_fm[IU], iOct) * m_Udata(cell_index, m_fm[IU], iOct)) /
                      m_Udata(cell_index, m_fm[ID], iOct);

    m_Udata(cell_index, m_fm[IP], iOct) =
      m_eos_wrapper.volumic_eint_from_pressure(m_woodwardParams.pL, m_woodwardParams.rhoL) + ekin;
  }
  else if (xyz[IX] > m_woodwardParams.xdR)
  {
    m_Udata(cell_index, m_fm[ID], iOct) = m_woodwardParams.rhoR;
    m_Udata(cell_index, m_fm[IU], iOct) = m_woodwardParams.uR;
    m_Udata(cell_index, m_fm[IV], iOct) = ZERO_F;
    if constexpr (dim == 3)
      m_Udata(cell_index, m_fm[IW], iOct) = ZERO_F;

    const auto ekin = HALF_F *
                      (m_Udata(cell_index, m_fm[IU], iOct) * m_Udata(cell_index, m_fm[IU], iOct)) /
                      m_Udata(cell_index, m_fm[ID], iOct);

    m_Udata(cell_index, m_fm[IP], iOct) =
      m_eos_wrapper.volumic_eint_from_pressure(m_woodwardParams.pR, m_woodwardParams.rhoR) + ekin;
  }
  else
  {
    m_Udata(cell_index, m_fm[ID], iOct) = m_woodwardParams.rhoC;
    m_Udata(cell_index, m_fm[IU], iOct) = m_woodwardParams.uC;
    m_Udata(cell_index, m_fm[IV], iOct) = ZERO_F;
    if constexpr (dim == 3)
      m_Udata(cell_index, m_fm[IW], iOct) = ZERO_F;

    const auto ekin = HALF_F *
                      (m_Udata(cell_index, m_fm[IU], iOct) * m_Udata(cell_index, m_fm[IU], iOct)) /
                      m_Udata(cell_index, m_fm[ID], iOct);

    m_Udata(cell_index, m_fm[IP], iOct) =
      m_eos_wrapper.volumic_eint_from_pressure(m_woodwardParams.pC, m_woodwardParams.rhoC) + ekin;
  }

} // end InitWoodwardDataFunctor::operator ()

template class InitWoodwardDataFunctor<2, kalypsso::DefaultDevice>;
template class InitWoodwardDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitWoodwardRefineFunctor<dim, device_t>::apply(DataArrayBlock_t const &             Udata,
                                                FieldMap<core::models::Hydro>        fm,
                                                orchard_key_view_t<device_t> const & orchard_keys,
                                                amrflags_view_t const &              amrflags,
                                                int32_t               local_num_octants,
                                                HydroSettings const & settings,
                                                int                   level_refine,
                                                ConfigMap const &     config_map)
{
  auto woodwardParams = WoodwardParams(config_map);

  // iterate functor for refinement
  InitWoodwardRefineFunctor functor(Udata,
                                    fm,
                                    orchard_keys,
                                    amrflags,
                                    local_num_octants,
                                    settings,
                                    woodwardParams,
                                    level_refine,
                                    config_map);

  const auto refine_type = core::get_init_indicator(config_map);

  if (refine_type == +core::InitConditionsIndicator::ALWAYS_REFINE)
  {
    Kokkos::parallel_for("kalypsso::godunov_hydro::InitWoodwardRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineAlways>(0, local_num_octants),
                         functor);
  }
  else if (refine_type == +core::InitConditionsIndicator::GEOMETRIC)
  {
    Kokkos::parallel_for("kalypsso::godunov_hydro::InitWoodwardRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineGeometric>(0, local_num_octants),
                         functor);
  }
  else
  {
    KALYPSSO_ERROR("Unknown value for refine indicator method.");
  }

} // InitWoodwardRefineFunctor::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitWoodwardRefineFunctor<dim, device_t>::operator()(TagRefineAlways const &,
                                                     const size_t & iOct) const
{
  m_amrflags(iOct) = AMRContextBase::KALYPSSO_DO_REFINE;
}

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitWoodwardRefineFunctor<dim, device_t>::operator()(TagRefineGeometric const &,
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
    const auto dL = fabs(xyz[IX] - m_woodwardParams.xdL);
    const auto dR = fabs(xyz[IX] - m_woodwardParams.xdR);

    if (dL < (block_length * KALYPSSO_NUM(0.95)) or dR < (block_length * KALYPSSO_NUM(0.95)))
      flag = AMRContextBase::KALYPSSO_DO_REFINE;

  } // end if level == level_refine

  // perform max reduction
  // if all cell in current block agree on COARSEN => do coarsen
  // if a single cell in current block disagree on coarsening => do nothing or refine
  // if a single cell in current block needs to refine => do refine
  m_amrflags(iOct) = flag;

} // end InitWoodwardRefineFunctor::operator ()

template class InitWoodwardRefineFunctor<2, kalypsso::DefaultDevice>;
template class InitWoodwardRefineFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitWoodward<dim, device_t>::apply(SolverGodunovHydro<dim, device_t> & solver)
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
  InitWoodwardDataFunctor<dim, device_t>::apply(solver.U(),
                                                solver.hydro().get_fieldmap(),
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
      InitWoodwardDataFunctor<dim, device_t>::apply(solver.U(),
                                                    solver.hydro().get_fieldmap(),
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
      InitWoodwardRefineFunctor<dim, device_t>::apply(solver.U(),
                                                      solver.hydro().get_fieldmap(),
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
      InitWoodwardDataFunctor<dim, device_t>::apply(solver.U(),
                                                    solver.hydro().get_fieldmap(),
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
} // InitWoodward::apply

template class InitWoodward<2, kalypsso::DefaultDevice>;
template class InitWoodward<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso
