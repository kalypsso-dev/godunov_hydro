// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitFourQuadrant.cpp
 */

#include <godunov_hydro/init/InitFourQuadrant.h>
#include <godunov_hydro/SolverGodunovHydro.h>
#include <godunov_hydro/models/utils_hydro.h>

namespace kalypsso
{
namespace godunov_hydro
{
// ====================================================================
// ====================================================================

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitFourQuadrantDataFunctor<dim, device_t>::apply(DataArrayBlock_t const &             Udata,
                                                  FieldMap<core::models::Hydro>        fm,
                                                  orchard_key_view_t<device_t> const & orchard_keys,
                                                  int32_t               local_num_octants,
                                                  HydroSettings const & settings,
                                                  ConfigMap const &     config_map)
{

  // load problem specific parameters
  const int configNumber = config_map.getInteger("four_quadrant", "config_number", 0);
  Kokkos::Array<real_t, dim> pos;
  pos[IX] = config_map.getReal("four_quadrant", "x", KALYPSSO_NUM(0.8));
  pos[IY] = config_map.getReal("four_quadrant", "y", KALYPSSO_NUM(0.8));
  if constexpr (dim == 3)
  {
    pos[IZ] = config_map.getReal("four_quadrant", "z", KALYPSSO_NUM(0.8));
  }

  auto Us = getRiemannConfig<dim>(configNumber);

  /*
   * convert primitive to conservative variables
   */

  // Equation of state wrapper
  const auto            eos = eos::EosWrapper<HostDevice>(config_map);
  [[maybe_unused]] bool valid = true;

  for (int i = 0; i < 4; ++i)
    Us[i] = models::compute_conservative_variables<dim, HostDevice>(Us[i], settings, eos, valid);
  if constexpr (dim == 3)
  {
    for (int i = 4; i < 8; ++i)
      Us[i] = models::compute_conservative_variables<dim, HostDevice>(Us[i], settings, eos, valid);
  }

  InitFourQuadrantDataFunctor functor(
    Udata, fm, orchard_keys, local_num_octants, settings, Us, pos, config_map);

  // compute total number of cells
  const auto nbCellsPerLeaf = Udata.num_cells();
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  Kokkos::parallel_for("kalypsso::godunov_hydro::InitFourQuadrantDataFunctor",
                       Kokkos::RangePolicy<exec_space>(0, nbCellsTotal),
                       functor);

} // InitFourQuadrantDataFunctor<dim, device_t>::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitFourQuadrantDataFunctor<dim, device_t>::operator()(const int32_t & global_index) const
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
  const auto center_vertex = orchard_key_to_cell_coord<dim>(key, iCoord, block_sizes[IX]);
  const auto center = vertex_coord_to_real_space<dim>(center_vertex, m_scaling_factor, m_xyz_min);

  const real_t & x = center[IX];
  const real_t & y = center[IY];

  const auto & U0 = m_Us[0];
  const auto & U1 = m_Us[1];
  const auto & U2 = m_Us[2];
  const auto & U3 = m_Us[3];

  if constexpr (dim == 2)
  {
    if (x < m_pos[IX])
    {
      if (y < m_pos[IY])
      {
        // region 2
        m_Udata(cell_index, m_fm[ID], iOct) = U2[ID];
        m_Udata(cell_index, m_fm[IP], iOct) = U2[IP];
        m_Udata(cell_index, m_fm[IU], iOct) = U2[IU];
        m_Udata(cell_index, m_fm[IV], iOct) = U2[IV];
      }
      else
      {
        // region 1
        m_Udata(cell_index, m_fm[ID], iOct) = U1[ID];
        m_Udata(cell_index, m_fm[IP], iOct) = U1[IP];
        m_Udata(cell_index, m_fm[IU], iOct) = U1[IU];
        m_Udata(cell_index, m_fm[IV], iOct) = U1[IV];
      }
    }
    else
    {
      if (y < m_pos[IY])
      {
        // region 3
        m_Udata(cell_index, m_fm[ID], iOct) = U3[ID];
        m_Udata(cell_index, m_fm[IP], iOct) = U3[IP];
        m_Udata(cell_index, m_fm[IU], iOct) = U3[IU];
        m_Udata(cell_index, m_fm[IV], iOct) = U3[IV];
      }
      else
      {
        // region 0
        m_Udata(cell_index, m_fm[ID], iOct) = U0[ID];
        m_Udata(cell_index, m_fm[IP], iOct) = U0[IP];
        m_Udata(cell_index, m_fm[IU], iOct) = U0[IU];
        m_Udata(cell_index, m_fm[IV], iOct) = U0[IV];
      }
    }
  }
  else if constexpr (dim == 3)
  {
    const real_t & z = center[IZ];

    const auto & U4 = m_Us[4];
    const auto & U5 = m_Us[5];
    const auto & U6 = m_Us[6];
    const auto & U7 = m_Us[7];

    if (x < m_pos[IX])
    {
      if (y < m_pos[IY])
      {
        if (z < m_pos[IZ])
        {
          // region 2
          m_Udata(cell_index, m_fm[ID], iOct) = U2[ID];
          m_Udata(cell_index, m_fm[IP], iOct) = U2[IP];
          m_Udata(cell_index, m_fm[IU], iOct) = U2[IU];
          m_Udata(cell_index, m_fm[IV], iOct) = U2[IV];
          m_Udata(cell_index, m_fm[IW], iOct) = U2[IW];
        }
        else
        {
          // region 6
          m_Udata(cell_index, m_fm[ID], iOct) = U6[ID];
          m_Udata(cell_index, m_fm[IP], iOct) = U6[IP];
          m_Udata(cell_index, m_fm[IU], iOct) = U6[IU];
          m_Udata(cell_index, m_fm[IV], iOct) = U6[IV];
          m_Udata(cell_index, m_fm[IW], iOct) = U6[IW];
        }
      }
      else
      {
        if (z < m_pos[IZ])
        {
          // region 1
          m_Udata(cell_index, m_fm[ID], iOct) = U1[ID];
          m_Udata(cell_index, m_fm[IP], iOct) = U1[IP];
          m_Udata(cell_index, m_fm[IU], iOct) = U1[IU];
          m_Udata(cell_index, m_fm[IV], iOct) = U1[IV];
          m_Udata(cell_index, m_fm[IW], iOct) = U1[IW];
        }
        else
        {
          // region 5
          m_Udata(cell_index, m_fm[ID], iOct) = U5[ID];
          m_Udata(cell_index, m_fm[IP], iOct) = U5[IP];
          m_Udata(cell_index, m_fm[IU], iOct) = U5[IU];
          m_Udata(cell_index, m_fm[IV], iOct) = U5[IV];
          m_Udata(cell_index, m_fm[IW], iOct) = U5[IW];
        }
      }
    }
    else
    {
      if (y < m_pos[IY])
      {
        if (z < m_pos[IZ])
        {
          // region 3
          m_Udata(cell_index, m_fm[ID], iOct) = U3[ID];
          m_Udata(cell_index, m_fm[IP], iOct) = U3[IP];
          m_Udata(cell_index, m_fm[IU], iOct) = U3[IU];
          m_Udata(cell_index, m_fm[IV], iOct) = U3[IV];
          m_Udata(cell_index, m_fm[IW], iOct) = U3[IW];
        }
        else
        {
          // region 7
          m_Udata(cell_index, m_fm[ID], iOct) = U7[ID];
          m_Udata(cell_index, m_fm[IP], iOct) = U7[IP];
          m_Udata(cell_index, m_fm[IU], iOct) = U7[IU];
          m_Udata(cell_index, m_fm[IV], iOct) = U7[IV];
          m_Udata(cell_index, m_fm[IW], iOct) = U7[IW];
        }
      }
      else
      {
        if (z < m_pos[IZ])
        {
          // region 0
          m_Udata(cell_index, m_fm[ID], iOct) = U0[ID];
          m_Udata(cell_index, m_fm[IP], iOct) = U0[IP];
          m_Udata(cell_index, m_fm[IU], iOct) = U0[IU];
          m_Udata(cell_index, m_fm[IV], iOct) = U0[IV];
          m_Udata(cell_index, m_fm[IW], iOct) = U0[IW];
        }
        else
        {
          // region 4
          m_Udata(cell_index, m_fm[ID], iOct) = U4[ID];
          m_Udata(cell_index, m_fm[IP], iOct) = U4[IP];
          m_Udata(cell_index, m_fm[IU], iOct) = U4[IU];
          m_Udata(cell_index, m_fm[IV], iOct) = U4[IV];
          m_Udata(cell_index, m_fm[IW], iOct) = U4[IW];
        }
      }
    }
  } // end dim == 3

} // end InitFourQuadrantDataFunctor<dim, device_t>::operator ()

template class InitFourQuadrantDataFunctor<2, kalypsso::DefaultDevice>;
template class InitFourQuadrantDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitFourQuadrantRefineFunctor<dim, device_t>::apply(
  DataArrayBlock_t const &             Udata,
  FieldMap<core::models::Hydro>        fm,
  orchard_key_view_t<device_t> const & orchard_keys,
  amrflags_view_t const &              amrflags,
  int32_t                              local_num_octants,
  HydroSettings const &                settings,
  int                                  level_refine,
  ConfigMap const &                    config_map)
{

  real_t xt = config_map.getReal("four_quadrant", "x", KALYPSSO_NUM(0.8));
  real_t yt = config_map.getReal("four_quadrant", "y", KALYPSSO_NUM(0.8));
  real_t zt = config_map.getReal("four_quadrant", "z", KALYPSSO_NUM(0.8));

  // iterate functor for refinement
  InitFourQuadrantRefineFunctor functor(Udata,
                                        fm,
                                        orchard_keys,
                                        amrflags,
                                        local_num_octants,
                                        settings,
                                        level_refine,
                                        xt,
                                        yt,
                                        zt,
                                        config_map);

  const auto refine_type = core::get_init_indicator(config_map);

  if (refine_type == +core::InitConditionsIndicator::ALWAYS_REFINE)
  {

    Kokkos::parallel_for("kalypsso::godunov_hydro::InitFourQuadrantRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineAlways>(0, local_num_octants),
                         functor);
  }
  else if (refine_type == +core::InitConditionsIndicator::GEOMETRIC)
  {
    Kokkos::parallel_for("kalypsso::godunov_hydro::InitFourQuadrantRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineGeometric>(0, local_num_octants),
                         functor);
  }
  else
  {
    KALYPSSO_ERROR("Unknown value for refine indicator method.");
  }

} // InitFourQuadrantRefineFunctor<dim, device_t>::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitFourQuadrantRefineFunctor<dim, device_t>::operator()(TagRefineAlways const &,
                                                         const size_t & iOct) const
{
  m_amrflags(iOct) = AMRContextBase::KALYPSSO_DO_REFINE;
} // InitFourQuadrantRefineFunctor<dim, device_t>::operator()

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitFourQuadrantRefineFunctor<dim, device_t>::operator()(TagRefineGeometric const &,
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


    real_t block_length_th = block_length * KALYPSSO_NUM(0.75);

    bool should_refine = (xyz[IX] + block_length_th >= m_xt and xyz[IX] - block_length_th < m_xt) or
                         (xyz[IX] - block_length_th <= m_xt and xyz[IX] + block_length_th > m_xt) or
                         (xyz[IY] + block_length_th >= m_yt and xyz[IY] - block_length_th < m_yt) or
                         (xyz[IY] - block_length_th <= m_yt and xyz[IY] + block_length_th > m_yt);

    if constexpr (dim == 3)
    {
      should_refine = should_refine or
                      (xyz[IZ] + block_length_th >= m_zt and xyz[IZ] - block_length_th < m_zt) or
                      (xyz[IZ] - block_length_th <= m_zt and xyz[IZ] + block_length_th > m_zt);
    }

    if (should_refine)
      flag = AMRContextBase::KALYPSSO_DO_REFINE;

  } // end if level == level_refine

  m_amrflags(iOct) = flag;

} // end InitFourQuadrantRefineFunctor<dim, device_t>::operator()

template class InitFourQuadrantRefineFunctor<2, kalypsso::DefaultDevice>;
template class InitFourQuadrantRefineFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitFourQuadrant<dim, device_t>::apply(SolverGodunovHydro<dim, device_t> & solver)
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
  InitFourQuadrantDataFunctor<dim, device_t>::apply(solver.U(),
                                                    solver.hydro().get_fieldmap(),
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
      InitFourQuadrantDataFunctor<dim, device_t>::apply(solver.U(),
                                                        solver.hydro().get_fieldmap(),
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
      InitFourQuadrantRefineFunctor<dim, device_t>::apply(solver.U(),
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
      InitFourQuadrantDataFunctor<dim, device_t>::apply(solver.U(),
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

} // InitFourQuadrant::apply

template class InitFourQuadrant<2, kalypsso::DefaultDevice>;
template class InitFourQuadrant<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso
