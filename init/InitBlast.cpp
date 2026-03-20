// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitBlast.cpp
 */

#include <godunov_hydro/init/InitBlast.h>
#include <godunov_hydro/SolverGodunovHydro.h>

#include <kalypsso/core/models/utils_hydro.h>
#include <kalypsso/core/orchard_key_utils.h>
#include <kalypsso/core/problems/init_cond_utils.h>
#include <kalypsso/core/region_utils.h>
#include <kalypsso/core/vof/interface_tracking_utils.h>
#include <kalypsso/core/geometry_utils.h>

namespace kalypsso
{
namespace godunov_hydro
{

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
InitBlastDataFunctor<dim, device_t>::InitBlastDataFunctor(
  DataArrayBlock_t const &             Udata,
  FieldMap<core::models::Hydro>        fm,
  orchard_key_view_t<device_t> const & orchard_keys,
  int32_t                              local_num_octants,
  HydroSettings const &                settings,
  BlastParams const &                  bParams,
  InitialStates<dim, device_t> const & initial_states,
  bool                                 replicated_init_cond,
  real_t                               total_volume_inside,
  ConfigMap const &                    config_map)
  : m_Udata(Udata)
  , m_fm(fm)
  , m_orchard_keys(orchard_keys)
  , m_local_num_octants(local_num_octants)
  , m_settings(settings)
  , m_bParams(bParams)
  , m_initial_states(initial_states)
  , m_scaling_factor(get_scaling_factor(config_map))
  , m_xyz_min(get_xyz_min<dim>(config_map))
  , m_replicated_init_cond(replicated_init_cond)
  , m_total_volume_inside(total_volume_inside){};

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
auto
InitBlastDataFunctor<dim, device_t>::apply(DataArrayBlock_t const &             Udata,
                                           FieldMap<core::models::Hydro>        fm,
                                           orchard_key_view_t<device_t> const & orchard_keys,
                                           int32_t                              local_num_octants,
                                           HydroSettings const &                settings,
                                           InitialStates<dim, device_t> const & initial_states,
                                           bool              replicated_init_cond,
                                           ConfigMap const & config_map,
                                           [[maybe_unused]] ParallelEnv const & par_env)
{
  BlastParams blastParams = BlastParams(config_map);

  // data init functor
  InitBlastDataFunctor functor(Udata,
                               fm,
                               orchard_keys,
                               local_num_octants,
                               settings,
                               blastParams,
                               initial_states,
                               replicated_init_cond,
                               ZERO_F,
                               config_map);

  // compute volume inside ball
  real_t              volume_inside, total_volume_inside = ZERO_F;
  Kokkos::Sum<real_t> reducer(volume_inside);

  // compute total number of cells
  const auto nbCellsPerLeaf = Udata.num_cells();
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  Kokkos::parallel_reduce("kalypsso::godunov_hydro::InitBlastDataFunctor",
                          Kokkos::RangePolicy<exec_space, TagInitData>(0, nbCellsTotal),
                          functor,
                          reducer);

#ifdef KALYPSSO_CORE_USE_MPI
  par_env.comm().MPI_Allreduce<MpiComm::SUM>(&volume_inside, &total_volume_inside, 1);
#else
  total_volume_inside = volume_inside;
#endif // KALYPSSO_CORE_USE_MPI

  return total_volume_inside;

} // InitBlastDataFunctor::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitBlastDataFunctor<dim, device_t>::rescale_energy(
  DataArrayBlock_t const &             Udata,
  FieldMap<core::models::Hydro>        fm,
  orchard_key_view_t<device_t> const & orchard_keys,
  int32_t                              local_num_octants,
  HydroSettings const &                settings,
  InitialStates<dim, device_t> const & initial_states,
  bool                                 replicated_init_cond,
  ConfigMap const &                    config_map,
  [[maybe_unused]] ParallelEnv const & par_env,
  real_t                               total_volume_inside)
{
  BlastParams blastParams = BlastParams(config_map);

  // data init functor
  InitBlastDataFunctor functor(Udata,
                               fm,
                               orchard_keys,
                               local_num_octants,
                               settings,
                               blastParams,
                               initial_states,
                               replicated_init_cond,
                               total_volume_inside,
                               config_map);

  // compute total number of cells
  const auto nbCellsPerLeaf = Udata.num_cells();
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  Kokkos::parallel_for("kalypsso::godunov_hydro::InitBlastDataFunctor",
                       Kokkos::RangePolicy<exec_space, TagRescaleEnergy>(0, nbCellsTotal),
                       functor);

} // InitBlastDataFunctor::rescale_energy

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitBlastDataFunctor<dim, device_t>::operator()(TagInitData const &,
                                                const int32_t & global_index,
                                                real_t &        volume) const
{

  // convert global index into
  // - octant id
  // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
  const auto iOct = global_index / m_Udata.num_cells();
  const auto cell_index = global_index - iOct * m_Udata.num_cells();

  auto const & block_sizes = m_Udata.block_size();

  // compute ix,iy,iz of local cell inside
  // block from index
  auto iCoord = cellindex_to_coord<dim>(cell_index, block_sizes);

  // get block orchard key
  const auto key = m_orchard_keys(iOct);

  // compute physical x,y,z for that cell (cell center)
  const auto xyz_vertex = orchard_key_to_cell_coord<dim>(key, iCoord, block_sizes[IX]);

  auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, m_scaling_factor, m_xyz_min);

  // compute cell size
  const auto level = orchard_key_t<dim>::level(key);
  const auto dx = compute_cell_length<dim>(level, block_sizes[IX]) * m_scaling_factor;

  // compute volume of current cell assuming
  // dx=dy=dz, i.e. block_sizes are the same along all directions
  const auto dvol = (dim == 3) ? dx * dx * dx : dx * dx;

  // if using replicated initial conditions, shift coordinates into tree id 0
  // so that all trees get initialized the same
  if (m_replicated_init_cond)
  {
    const auto tree_coords = orchard_key_t<dim>::get_tree_coords(key);
    xyz[IX] -= tree_coords[IX] * m_scaling_factor;
    xyz[IY] -= tree_coords[IY] * m_scaling_factor;
    if constexpr (dim == 3)
    {
      xyz[IZ] -= tree_coords[IZ] * m_scaling_factor;
    }
  }

  // array of region id (one per corner)
  Kokkos::Array<int, Corner::num_corners<dim>()> corner_regions;
  compute_corner_to_region(iCoord, key, block_sizes, m_replicated_init_cond, corner_regions);

  const bool init_as_pure_cell = is_cell_fully_inside_region<dim>(corner_regions);

  if (init_as_pure_cell)
  {
    const auto region = corner_regions[0];

    m_Udata(cell_index, m_fm[Hydro::ID], iOct) = m_initial_states(region)[Hydro::ID];
    m_Udata(cell_index, m_fm[Hydro::IE], iOct) = m_initial_states(region)[Hydro::IE];
    m_Udata(cell_index, m_fm[Hydro::IU], iOct) = m_initial_states(region)[Hydro::IU];
    m_Udata(cell_index, m_fm[Hydro::IV], iOct) = m_initial_states(region)[Hydro::IV];
    if constexpr (dim == 3)
    {
      m_Udata(cell_index, m_fm[Hydro::IW], iOct) = m_initial_states(region)[Hydro::IW];
    }

    // if cell is fully inside, reduce its volume
    if (region == 0)
    {
      volume += dvol;
    }
  }
  else
  {
    // we have a "mixed" cell

    // blast center
    Kokkos::Array<real_t, dim> blast_center;
    blast_center[IX] = m_bParams.blast_center_x;
    blast_center[IY] = m_bParams.blast_center_y;
    if constexpr (dim == 3)
      blast_center[IZ] = m_bParams.blast_center_z;

    // blast problem parameters
    const real_t blast_radius = m_bParams.blast_radius;

    // compute volume inside
    Kokkos::Array<real_t, dim> normal;
    real_t                     alpha;
    get_tangent_to_sphere(blast_center, blast_radius, xyz, normal, alpha);

    // compute volume fraction of material inside bubble
    const auto blast_vol_frac =
      vof::compute_volume_fraction_of_rect_below_plane(normal, alpha, xyz - dx / 2, xyz + dx / 2);

    const auto u_mixed = compute_mixed_state<dim, device_t>(
      m_initial_states, 0, blast_vol_frac, 1, ONE_F - blast_vol_frac);

    m_Udata(cell_index, m_fm[Hydro::ID], iOct) = u_mixed[Hydro::ID];
    m_Udata(cell_index, m_fm[Hydro::IE], iOct) = u_mixed[Hydro::IE];
    m_Udata(cell_index, m_fm[Hydro::IU], iOct) = u_mixed[Hydro::IU];
    m_Udata(cell_index, m_fm[Hydro::IV], iOct) = u_mixed[Hydro::IV];
    if constexpr (dim == 3)
    {
      m_Udata(cell_index, m_fm[Hydro::IW], iOct) = u_mixed[Hydro::IW];
    }

    volume += dvol * blast_vol_frac;
  }

} // end InitBlastDataFunctor::operator () - TagInitData

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitBlastDataFunctor<dim, device_t>::operator()(TagRescaleEnergy const &,
                                                const int32_t & global_index) const
{

  const auto iOct = global_index / m_Udata.num_cells();
  const auto cell_index = global_index - iOct * m_Udata.num_cells();

  auto const & block_sizes = m_Udata.block_size();
  auto const & blast_radius = m_bParams.blast_radius;

  // compute ix,iy,iz of local cell inside
  // block from index
  auto iCoord = cellindex_to_coord<dim>(static_cast<int32_t>(cell_index), block_sizes);

  // get block orchard key
  const auto key = m_orchard_keys(iOct);

  // compute physical x,y,z for that cell (cell center)
  const auto xyz_vertex = orchard_key_to_cell_coord<dim>(key, iCoord, block_sizes[IX]);

  auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, m_scaling_factor, m_xyz_min);

  // if using replicated initial conditions, shift coordinates into tree id 0
  // so that all trees get initialized the same
  if (m_replicated_init_cond)
  {
    const auto tree_coords = orchard_key_t<dim>::get_tree_coords(key);
    xyz[IX] -= tree_coords[IX] * m_scaling_factor;
    xyz[IY] -= tree_coords[IY] * m_scaling_factor;
    if constexpr (dim == 3)
    {
      xyz[IZ] -= tree_coords[IZ] * m_scaling_factor;
    }
  }

  // compute cell size
  const auto level = orchard_key_t<dim>::level(key);
  const auto dx = compute_cell_length<dim>(level, block_sizes[IX]) * m_scaling_factor;

  // blast center
  Kokkos::Array<real_t, dim> blast_center;
  blast_center[IX] = m_bParams.blast_center_x;
  blast_center[IY] = m_bParams.blast_center_y;
  if constexpr (dim == 3)
    blast_center[IZ] = m_bParams.blast_center_z;

  // compute volume inside
  Kokkos::Array<real_t, dim> normal;
  real_t                     alpha;
  get_tangent_to_sphere(blast_center, blast_radius, xyz, normal, alpha);

  // compute volume fraction of material inside bubble
  const auto blast_vol_frac =
    vof::compute_volume_fraction_of_rect_below_plane(normal, alpha, xyz - dx / 2, xyz + dx / 2);

  // array of region id (one per corner)
  Kokkos::Array<int, Corner::num_corners<dim>()> corner_regions;
  compute_corner_to_region(iCoord, key, block_sizes, m_replicated_init_cond, corner_regions);

  const bool cell_fully_inside = is_cell_fully_inside_region<dim>(corner_regions);

  if (blast_vol_frac > 0)
  {
    if (cell_fully_inside)
    {
      m_Udata(cell_index, m_fm[Hydro::IE], iOct) =
        m_bParams.total_energy_inside / m_total_volume_inside;
    }
    else
    {
      auto U_in = m_initial_states(0);
      U_in[Hydro::IE] = m_bParams.total_energy_inside / m_total_volume_inside;

      const auto U_out = m_initial_states(1);

      const auto U_mix =
        compute_mixed_state<dim, device_t>(U_in, blast_vol_frac, U_out, ONE_F - blast_vol_frac);

      m_Udata(cell_index, m_fm[Hydro::IE], iOct) = U_mix[Hydro::IE];
    }
  }

} // end InitBlastDataFunctor::operator () - TagRescaleEnergy

template class InitBlastDataFunctor<2, kalypsso::DefaultDevice>;
template class InitBlastDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
InitBlastRefineFunctor<dim, device_t>::InitBlastRefineFunctor(
  DataArrayBlock_t const &             Udata,
  FieldMap<core::models::Hydro>        fm,
  orchard_key_view_t<device_t> const & orchard_keys,
  amrflags_view_t const &              amrflags,
  int32_t                              local_num_octants,
  HydroSettings const &                settings,
  BlastParams const &                  bParams,
  int                                  level_refine,
  bool                                 replicated_init_cond,
  ConfigMap const &                    config_map)
  : m_Udata(Udata)
  , m_fm(fm)
  , m_orchard_keys(orchard_keys)
  , m_amrflags(amrflags)
  , m_local_num_octants(local_num_octants)
  , m_settings(settings)
  , m_bParams(bParams)
  , m_level_refine(level_refine)
  , m_scaling_factor(get_scaling_factor(config_map))
  , m_xyz_min(get_xyz_min<dim>(config_map))
  , m_replicated_init_cond(replicated_init_cond){};

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitBlastRefineFunctor<dim, device_t>::apply(DataArrayBlock_t const &             Udata,
                                             FieldMap<core::models::Hydro>        fm,
                                             orchard_key_view_t<device_t> const & orchard_keys,
                                             amrflags_view_t const &              amrflags,
                                             int32_t                              local_num_octants,
                                             HydroSettings const &                settings,
                                             int                                  level_refine,
                                             bool              replicated_init_cond,
                                             ConfigMap const & config_map)
{
  BlastParams blastParams = BlastParams(config_map);

  // iterate functor for refinement
  InitBlastRefineFunctor functor(Udata,
                                 fm,
                                 orchard_keys,
                                 amrflags,
                                 local_num_octants,
                                 settings,
                                 blastParams,
                                 level_refine,
                                 replicated_init_cond,
                                 config_map);


  const auto refine_type = core::get_init_indicator(config_map);

  if (refine_type == +core::InitConditionsIndicator::ALWAYS_REFINE)
  {
    Kokkos::parallel_for("kalypsso::godunov_hydro::InitBlastRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineAlways>(0, local_num_octants),
                         functor);
  }
  else if (refine_type == +core::InitConditionsIndicator::GEOMETRIC)
  {
    Kokkos::parallel_for("kalypsso::godunov_hydro::InitBlastRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineGeometric>(0, local_num_octants),
                         functor);
  }
  else
  {
    KALYPSSO_ERROR("Unknown value for refine indicator method.");
  }
} // IniBlastRefineFunctor::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitBlastRefineFunctor<dim, device_t>::operator()(TagRefineAlways const &,
                                                  const size_t & iOct) const
{
  m_amrflags(iOct) = AMRContextBase::KALYPSSO_DO_REFINE;
}

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitBlastRefineFunctor<dim, device_t>::operator()(TagRefineGeometric const &,
                                                  const size_t & iOct) const
{

  // blast problem parameters
  const auto radius = m_bParams.blast_radius;
  const auto blast_center_x = m_bParams.blast_center_x;
  const auto blast_center_y = m_bParams.blast_center_y;
  const auto blast_center_z = m_bParams.blast_center_z;

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
    auto           xyz = vertex_coord_to_real_space<dim>(xyz_vertex, m_scaling_factor, m_xyz_min);

    // if using replicated initial conditions, shift coordinates into tree id 0
    // so that all trees get initialized the same
    if (m_replicated_init_cond)
    {
      const auto tree_coords = orchard_key_t<dim>::get_tree_coords(key);
      xyz[IX] -= tree_coords[IX] * m_scaling_factor;
      xyz[IY] -= tree_coords[IY] * m_scaling_factor;
      if constexpr (dim == 3)
      {
        xyz[IZ] -= tree_coords[IZ] * m_scaling_factor;
      }
    }

    auto d2 = (xyz[IX] - blast_center_x) * (xyz[IX] - blast_center_x) +
              (xyz[IY] - blast_center_y) * (xyz[IY] - blast_center_y);

    if constexpr (dim == 3)
      d2 += (xyz[IZ] - blast_center_z) * (xyz[IZ] - blast_center_z);

    if (fabs(sqrt(d2) - radius) < (block_length * KALYPSSO_NUM(1.25)))
      flag = AMRContextBase::KALYPSSO_DO_REFINE;

  } // end if level == level_refine

  // perform max reduction
  // if all cell in current block agree on COARSEN => do coarsen
  // if a single cell in current block disagree on coarsening => do nothing or refine
  // if a single cell in current block needs to refine => do refine
  m_amrflags(iOct) = flag;

} // InitBlastRefineFunctor::operator ()

template class InitBlastRefineFunctor<2, kalypsso::DefaultDevice>;
template class InitBlastRefineFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitBlast<dim, device_t>::apply(SolverGodunovHydro<dim, device_t> & solver)
{
  auto              amr_mesh = solver.amr_mesh();
  ConfigMap const & config_map = solver.config_map();
  const auto        settings = HydroSettings(solver.config_map());
  const int         level_min = solver.hydro_params().level_min;
  const int         level_max = solver.hydro_params().level_max;
  const bool        replicated_init_cond = solver.hydro_params().replicated_init_cond;

  const auto initial_states = get_initial_states<dim, device_t>(config_map, 2);

  constexpr bool do_reset_ghosts = true;
  solver.update_mesh(do_reset_ghosts);

  // resize Udata
  solver.resize_solver_data();

  // initialize total volume inside
  auto total_volume_inside = ZERO_F;

  // first init of Udata
  total_volume_inside =
    InitBlastDataFunctor<dim, device_t>::apply(solver.U(),
                                               solver.hydro().get_fieldmap(),
                                               solver.mesh_map()->orchard_keys(),
                                               solver.amr_mesh()->local_num_quadrants(),
                                               settings,
                                               initial_states,
                                               replicated_init_cond,
                                               config_map,
                                               solver.par_env());

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
      total_volume_inside =
        InitBlastDataFunctor<dim, device_t>::apply(solver.U(),
                                                   solver.hydro().get_fieldmap(),
                                                   solver.mesh_map()->orchard_keys(),
                                                   solver.amr_mesh()->local_num_quadrants(),
                                                   settings,
                                                   initial_states,
                                                   replicated_init_cond,
                                                   config_map,
                                                   solver.par_env());

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
      InitBlastRefineFunctor<dim, device_t>::apply(solver.U(),
                                                   solver.hydro().get_fieldmap(),
                                                   solver.mesh_map()->orchard_keys(),
                                                   flags_d,
                                                   solver.amr_mesh()->local_num_quadrants(),
                                                   settings,
                                                   level,
                                                   replicated_init_cond,
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
      total_volume_inside =
        InitBlastDataFunctor<dim, device_t>::apply(solver.U(),
                                                   solver.hydro().get_fieldmap(),
                                                   solver.mesh_map()->orchard_keys(),
                                                   solver.amr_mesh()->local_num_quadrants(),
                                                   settings,
                                                   initial_states,
                                                   replicated_init_cond,
                                                   config_map,
                                                   solver.par_env());

      // update level
      ++level;

    } // end while level<level_max
  } // end init_refine_type

#ifdef KALYPSSO_CORE_USE_MPI
  // load balancing (= repartitioning) the octree mesh + userdata over the MPI processes.
  // U and U2 will be resized
  solver.do_load_balancing();
#endif

  BlastParams blastParams = BlastParams(config_map);

  // checking if we need rescaling pressure/total_energy inside the ball
  if (blastParams.total_energy_inside > 0)
  {
    InitBlastDataFunctor<dim, device_t>::rescale_energy(solver.U(),
                                                        solver.hydro().get_fieldmap(),
                                                        solver.mesh_map()->orchard_keys(),
                                                        solver.amr_mesh()->local_num_quadrants(),
                                                        settings,
                                                        initial_states,
                                                        replicated_init_cond,
                                                        config_map,
                                                        solver.par_env(),
                                                        total_volume_inside);
  }

} // InitBlast<dim, device_t>::apply

template class InitBlast<2, kalypsso::DefaultDevice>;
template class InitBlast<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso
