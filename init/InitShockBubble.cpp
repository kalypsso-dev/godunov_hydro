// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitShockBubble.cpp
 */

#include <godunov_hydro/init/InitShockBubble.h>
#include <godunov_hydro/SolverGodunovHydro.h>

#include <kalypsso/core/models/utils_hydro.h>
#include <kalypsso/core/orchard_key_utils.h>
#include <kalypsso/core/vof/interface_tracking_utils.h>
#include <kalypsso/core/geometry_utils.h>

namespace kalypsso
{
namespace godunov_hydro
{

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitShockBubbleDataFunctor<dim, device_t>::apply(
  DataArrayBlock_t const &             Udata,
  FieldMap<core::models::Hydro>        fm,
  orchard_key_view_t<device_t> const & orchard_keys,
  int32_t                              local_num_octants,
  InitialStates<dim, device_t> const & initial_states,
  ConfigMap const &                    config_map)
{
  // data init functor
  InitShockBubbleDataFunctor functor(
    Udata, fm, orchard_keys, local_num_octants, initial_states, config_map);

  // compute total number of cells
  const auto nbCellsPerLeaf = Udata.num_cells();
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  Kokkos::parallel_for("kalypsso::godunov_hydro::InitShockBubbleDataFunctor",
                       Kokkos::RangePolicy<exec_space>(0, nbCellsTotal),
                       functor);

} // InitShockBubbleDataFunctor::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitShockBubbleDataFunctor<dim, device_t>::operator()(const int32_t & global_index) const
{

  // convert global index into
  // - octant id
  // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
  const auto iOct = global_index / m_Udata.num_cells();
  const auto cell_index = global_index - iOct * m_Udata.num_cells();

  const auto block_sizes = m_Udata.block_size();

  // makes enum Hydro::VarId available
  // using Hydro = core::models::Hydro;

  // compute ix,iy,iz of local cell inside
  // block from index
  auto iCoord = cellindex_to_coord<dim>(cell_index, block_sizes);

  // get block orchard key
  const auto key = m_orchard_keys(iOct);

  // compute physical x,y,z for that cell (cell center)
  const auto xyz_vertex = orchard_key_to_cell_coord<dim>(key, iCoord, block_sizes[IX]);

  const auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, m_scaling_factor, m_xyz_min);

  // number of bubbles
  auto const & num_bubbles = m_sb_params.num_bubbles;

  auto const & bubble_radius = m_sb_params.bubble_radius;
  auto const & bubble_x = m_sb_params.bubble_x;
  auto const & bubble_y = m_sb_params.bubble_y;
  auto const & bubble_z = m_sb_params.bubble_z;

  // Fill as if the bubbles do not exist
  if (xyz[IX] <= m_sb_params.x_front)
  {
    // use post-shock hydro state
    m_Udata(cell_index, m_fm[Hydro::ID], iOct) = m_initial_states(0)[Hydro::ID];
    m_Udata(cell_index, m_fm[Hydro::IU], iOct) = m_initial_states(0)[Hydro::IU];
    m_Udata(cell_index, m_fm[Hydro::IV], iOct) = m_initial_states(0)[Hydro::IV];
    if constexpr (dim == 3)
    {
      m_Udata(cell_index, m_fm[Hydro::IW], iOct) = m_initial_states(0)[Hydro::IW];
    }
    m_Udata(cell_index, m_fm[Hydro::IE], iOct) = m_initial_states(0)[Hydro::IE];
  }
  else
  {
    // use pore-shock hydro state
    m_Udata(cell_index, m_fm[Hydro::ID], iOct) = m_initial_states(1)[Hydro::ID];
    m_Udata(cell_index, m_fm[Hydro::IU], iOct) = m_initial_states(1)[Hydro::IU];
    m_Udata(cell_index, m_fm[Hydro::IV], iOct) = m_initial_states(1)[Hydro::IV];
    if constexpr (dim == 3)
    {
      m_Udata(cell_index, m_fm[Hydro::IW], iOct) = m_initial_states(1)[Hydro::IW];
    }
    m_Udata(cell_index, m_fm[Hydro::IE], iOct) = m_initial_states(1)[Hydro::IE];
  }

  // get cell size
  const auto level = orchard_key_t<dim>::level(key);
  const auto dx = compute_cell_length<dim>(level, m_Udata.block_size()[IX]) * m_scaling_factor;

  for (int ib = 0; ib < num_bubbles; ++ib)
  {

    // we will adjust volume fractions near the bubble surface
    Kokkos::Array<real_t, dim> bubble_center;
    bubble_center[IX] = bubble_x(ib);
    bubble_center[IY] = bubble_y(ib);
    if constexpr (dim == 3)
    {
      bubble_center[IZ] = bubble_z(ib);
    }

    auto const &               cell_center = xyz;
    Kokkos::Array<real_t, dim> normal;
    real_t                     alpha;
    get_tangent_to_sphere(bubble_center, bubble_radius(ib), cell_center, normal, alpha);

    // compute volume fraction of material inside bubble
    const auto bubble_vol_frac = vof::compute_volume_fraction_of_rect_below_plane(
      normal, alpha, cell_center - dx / 2, cell_center + dx / 2);

    if (bubble_vol_frac > ZERO_F)
    {
      const auto phi0 = ONE_F - bubble_vol_frac;

      m_Udata(cell_index, m_fm[Hydro::ID], iOct) =
        phi0 * m_initial_states(1)[Hydro::ID] +
        (ONE_F - phi0) * m_initial_states(2 + ib)[Hydro::ID];

      m_Udata(cell_index, m_fm[Hydro::IU], iOct) =
        phi0 * m_initial_states(1)[Hydro::IU] +
        (ONE_F - phi0) * m_initial_states(2 + ib)[Hydro::IU];

      m_Udata(cell_index, m_fm[Hydro::IV], iOct) =
        phi0 * m_initial_states(1)[Hydro::IV] +
        (ONE_F - phi0) * m_initial_states(2 + ib)[Hydro::IV];

      if constexpr (dim == 3)
      {
        m_Udata(cell_index, m_fm[Hydro::IW], iOct) =
          phi0 * m_initial_states(1)[Hydro::IW] +
          (ONE_F - phi0) * m_initial_states(2 + ib)[Hydro::IW];
      }

      // compute pure state internal energy
      real_t eint_pre_shock = ZERO_F;
      {
        auto ekin = m_initial_states(1)[Hydro::IU] * m_initial_states(1)[Hydro::IU] +
                    m_initial_states(1)[Hydro::IV] * m_initial_states(1)[Hydro::IV];
        if constexpr (dim == 3)
        {
          ekin += m_initial_states(1)[Hydro::IW] * m_initial_states(1)[Hydro::IW];
        }
        ekin /= (TWO_F * m_initial_states(1)[Hydro::ID]);

        eint_pre_shock = m_initial_states(1)[Hydro::IE] - ekin;
      }

      real_t eint_bubble = ZERO_F;
      {
        auto ekin = m_initial_states(2 + ib)[Hydro::IU] * m_initial_states(2 + ib)[Hydro::IU] +
                    m_initial_states(2 + ib)[Hydro::IV] * m_initial_states(2 + ib)[Hydro::IV];
        if constexpr (dim == 3)
        {
          ekin += m_initial_states(2 + ib)[Hydro::IW] * m_initial_states(2 + ib)[Hydro::IW];
        }
        ekin /= (TWO_F * m_initial_states(2 + ib)[Hydro::ID]);

        eint_bubble = m_initial_states(2 + ib)[Hydro::IE] - ekin;
      }

      const auto eint_mixed = phi0 * eint_pre_shock + (ONE_F - phi0) * eint_bubble;

      real_t rho_mixed = m_Udata(cell_index, m_fm[Hydro::ID], iOct);
      auto   ekin_mixed =
        m_Udata(cell_index, m_fm[Hydro::IU], iOct) * m_Udata(cell_index, m_fm[Hydro::IU], iOct) +
        m_Udata(cell_index, m_fm[Hydro::IV], iOct) * m_Udata(cell_index, m_fm[Hydro::IV], iOct);
      if constexpr (dim == 3)
        ekin_mixed +=
          m_Udata(cell_index, m_fm[Hydro::IW], iOct) * m_Udata(cell_index, m_fm[Hydro::IW], iOct);
      ekin_mixed /= (TWO_F * rho_mixed);

      m_Udata(cell_index, m_fm[Hydro::IE], iOct) = eint_mixed + ekin_mixed;
    }
  } // end for ib

} // end InitShockBubbleDataFunctor::operator ()

template class InitShockBubbleDataFunctor<2, kalypsso::DefaultDevice>;
template class InitShockBubbleDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitShockBubbleRefineFunctor<dim, device_t>::apply(
  DataArrayBlock_t const &             Udata,
  FieldMap<core::models::Hydro>        fm,
  orchard_key_view_t<device_t> const & orchard_keys,
  amrflags_view_t const &              amrflags,
  int32_t                              local_num_octants,
  int                                  level_refine,
  ConfigMap const &                    config_map)
{
  // iterate functor for refinement
  InitShockBubbleRefineFunctor functor(
    Udata, fm, orchard_keys, amrflags, local_num_octants, level_refine, config_map);

  const auto refine_type = core::get_init_indicator(config_map);

  if (refine_type == +core::InitConditionsIndicator::ALWAYS_REFINE)
  {
    Kokkos::parallel_for("kalypsso::godunov_hydro::InitShockBubbleRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineAlways>(0, local_num_octants),
                         functor);
  }
  else
  {
    KALYPSSO_ERROR("Unknown value for refine indicator method.");
  }

} // InitShockBubbleRefineFunctor::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitShockBubbleRefineFunctor<dim, device_t>::operator()(TagRefineAlways const &,
                                                        const size_t & iOct) const
{
  m_amrflags(iOct) = AMRContextBase::KALYPSSO_DO_REFINE;
} // InitShockBubbleRefineFunctor::operator()

template class InitShockBubbleRefineFunctor<2, kalypsso::DefaultDevice>;
template class InitShockBubbleRefineFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitShockBubble<dim, device_t>::apply(SolverGodunovHydro<dim, device_t> & solver)
{

  auto              amr_mesh = solver.amr_mesh();
  ConfigMap const & config_map = solver.config_map();
  const int         level_min = solver.hydro_params().level_min;
  const int         level_max = solver.hydro_params().level_max;

  auto         shock_bubble_params = ShockBubbleParams<device_t>(config_map);
  auto const & nb_regions = shock_bubble_params.num_bubbles + 2;
  auto         initial_states = get_initial_states<dim, device_t>(config_map, nb_regions);

  constexpr bool do_reset_ghosts = true;
  solver.update_mesh(do_reset_ghosts);

  // resize Udata
  solver.resize_solver_data();

  // first init of Udata
  InitShockBubbleDataFunctor<dim, device_t>::apply(solver.U(),
                                                   solver.hydro().get_fieldmap(),
                                                   solver.mesh_map()->orchard_keys(),
                                                   solver.amr_mesh()->local_num_quadrants(),
                                                   initial_states,
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
      InitShockBubbleDataFunctor<dim, device_t>::apply(solver.U(),
                                                       solver.hydro().get_fieldmap(),
                                                       solver.mesh_map()->orchard_keys(),
                                                       solver.amr_mesh()->local_num_quadrants(),
                                                       initial_states,
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
      InitShockBubbleRefineFunctor<dim, device_t>::apply(solver.U(),
                                                         solver.hydro().get_fieldmap(),
                                                         solver.mesh_map()->orchard_keys(),
                                                         flags_d,
                                                         solver.amr_mesh()->local_num_quadrants(),
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
      InitShockBubbleDataFunctor<dim, device_t>::apply(solver.U(),
                                                       solver.hydro().get_fieldmap(),
                                                       solver.mesh_map()->orchard_keys(),
                                                       solver.amr_mesh()->local_num_quadrants(),
                                                       initial_states,
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

} // InitShockBubble::apply

template class InitShockBubble<2, kalypsso::DefaultDevice>;
template class InitShockBubble<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso
