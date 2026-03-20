// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitImplode.cpp
 */

#include <godunov_hydro/init/InitImplode.h>
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
InitImplodeDataFunctor<dim, device_t>::apply(DataArrayBlock_t const &             Udata,
                                             FieldMap<core::models::Hydro>        fm,
                                             orchard_key_view_t<device_t> const & orchard_keys,
                                             int32_t                              local_num_octants,
                                             HydroSettings const &                settings,
                                             ConfigMap const &                    config_map)
{
  ImplodeParams implode_params = ImplodeParams(config_map);

  // data init functor
  InitImplodeDataFunctor functor(
    Udata, fm, orchard_keys, local_num_octants, settings, implode_params, config_map);

  // compute total number of cells
  const auto nbCellsPerLeaf = Udata.num_cells();
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  Kokkos::parallel_for("kalypsso::godunov_hydro::InitImplodeDataFunctor",
                       Kokkos::RangePolicy<exec_space>(0, nbCellsTotal),
                       functor);

} // InitImplodeDataFunctor::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitImplodeDataFunctor<dim, device_t>::operator()(const int32_t & global_index) const
{

  // convert global index into
  // - octant id
  // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
  const auto iOct = global_index / m_Udata.num_cells();
  const auto cell_index = global_index - iOct * m_Udata.num_cells();

  const auto block_sizes = m_Udata.block_size();

  constexpr auto                  ID = core::models::Hydro::ID;
  constexpr auto                  IP = core::models::Hydro::IP;
  constexpr auto                  IU = core::models::Hydro::IU;
  constexpr auto                  IV = core::models::Hydro::IV;
  [[maybe_unused]] constexpr auto IW = core::models::Hydro::IW;

  // compute ix,iy,iz of local cell inside
  // block from index
  auto iCoord = cellindex_to_coord<dim>(cell_index, block_sizes);

  // get block orchard key
  const auto key = m_orchard_keys(iOct);

  // compute physical x,y,z for that cell (cell center)
  const auto xyz_vertex = orchard_key_to_cell_coord<dim>(key, iCoord, block_sizes[IX]);

  const auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, m_scaling_factor, m_xyz_min);

  // outer parameters
  auto const &                  rho_out = m_implode_params.rho_out;
  auto const &                  p_out = m_implode_params.p_out;
  auto const &                  u_out = m_implode_params.u_out;
  auto const &                  v_out = m_implode_params.v_out;
  [[maybe_unused]] auto const & w_out = m_implode_params.w_out;

  // inner parameters
  auto const &                  rho_in = m_implode_params.rho_in;
  auto const &                  p_in = m_implode_params.p_in;
  auto const &                  u_in = m_implode_params.u_in;
  auto const &                  v_in = m_implode_params.v_in;
  [[maybe_unused]] auto const & w_in = m_implode_params.w_in;

  auto const & shape = m_implode_params.shape;

  if constexpr (dim == 2)
  {
    auto const & x = xyz[IX];
    auto const & y = xyz[IY];

    auto const & xmin = m_xyz_min[IX];
    auto const & ymin = m_xyz_min[IY];
    auto const & xmax = m_xyz_max[IX];

    // initialize
    bool is_in = true;

    const auto threshold = xmin / 4 + 3 * xmax / 4 + ymin + KALYPSSO_NUM(1e-10);

    if (shape == +ImplodeShape::DIAGONAL)
    {
      is_in = (x + y) < threshold;
    }
    else if (shape == +ImplodeShape::DIAMOND)
    {
      is_in = fabs(x + y) < fabs(threshold) and fabs(x - y) < fabs(threshold);
    }

    if (is_in)
    {
      m_Udata(cell_index, m_fm[ID], iOct) = rho_in;
      m_Udata(cell_index, m_fm[IP], iOct) = m_eos_wrapper.volumic_eint_from_pressure(p_in, rho_in) +
                                            HALF_F * rho_in * (u_in * u_in + v_in * v_in);
      m_Udata(cell_index, m_fm[IU], iOct) = rho_in * u_in;
      m_Udata(cell_index, m_fm[IV], iOct) = rho_in * v_in;
    }
    else
    {
      m_Udata(cell_index, m_fm[ID], iOct) = rho_out;
      m_Udata(cell_index, m_fm[IP], iOct) =
        m_eos_wrapper.volumic_eint_from_pressure(p_out, rho_out) +
        HALF_F * rho_out * (u_out * u_out + v_out * v_out);
      m_Udata(cell_index, m_fm[IU], iOct) = rho_out * u_out;
      m_Udata(cell_index, m_fm[IV], iOct) = rho_out * v_out;
    }
  }
  else if constexpr (dim == 3)
  {
    auto const & x = xyz[IX];
    auto const & y = xyz[IY];
    auto const & z = xyz[IZ];

    auto const & xmin = m_xyz_min[IX];
    auto const & ymin = m_xyz_min[IY];
    auto const & zmin = m_xyz_min[IZ];
    auto const & xmax = m_xyz_max[IX];

    // initialize
    bool is_in = true;

    const auto threshold = xmin / 4 + 3 * xmax / 4 + ymin + zmin + KALYPSSO_NUM(1e-10);

    if (shape == +ImplodeShape::DIAGONAL)
    {
      is_in = (x + y + z) < threshold;
    }
    else if (shape == +ImplodeShape::DIAMOND)
    {
      is_in = fabs(x + y + z) < fabs(threshold) and fabs(x - y + z) < fabs(threshold) and
              fabs(x + y - z) < fabs(threshold) and fabs(-x + y + z) < fabs(threshold);
    }

    if (is_in)
    {
      m_Udata(cell_index, m_fm[ID], iOct) = rho_in;
      m_Udata(cell_index, m_fm[IP], iOct) =
        m_eos_wrapper.volumic_eint_from_pressure(p_in, rho_in) +
        HALF_F * rho_in * (u_in * u_in + v_in * v_in + w_in * w_in);
      m_Udata(cell_index, m_fm[IU], iOct) = rho_in * u_in;
      m_Udata(cell_index, m_fm[IV], iOct) = rho_in * v_in;
      m_Udata(cell_index, m_fm[IW], iOct) = rho_in * w_in;
    }
    else
    {
      m_Udata(cell_index, m_fm[ID], iOct) = rho_out;
      m_Udata(cell_index, m_fm[IP], iOct) =
        m_eos_wrapper.volumic_eint_from_pressure(p_out, rho_out) +
        HALF_F * rho_out * (u_out * u_out + v_out * v_out + w_out * w_out);
      m_Udata(cell_index, m_fm[IU], iOct) = rho_out * u_out;
      m_Udata(cell_index, m_fm[IV], iOct) = rho_out * v_out;
      m_Udata(cell_index, m_fm[IW], iOct) = rho_out * w_out;
    }
  }
} // end InitImplodeDataFunctor::operator ()

template class InitImplodeDataFunctor<2, kalypsso::DefaultDevice>;
template class InitImplodeDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================

template <size_t dim, typename device_t>
void
InitImplodeRefineFunctor<dim, device_t>::apply(DataArrayBlock_t const &             Udata,
                                               FieldMap<core::models::Hydro>        fm,
                                               orchard_key_view_t<device_t> const & orchard_keys,
                                               amrflags_view_t const &              amrflags,
                                               int32_t               local_num_octants,
                                               HydroSettings const & settings,
                                               int                   level_refine,
                                               ConfigMap const &     config_map)
{
  auto implode_params = ImplodeParams(config_map);

  // iterate functor for refinement
  InitImplodeRefineFunctor functor(Udata,
                                   fm,
                                   orchard_keys,
                                   amrflags,
                                   local_num_octants,
                                   settings,
                                   implode_params,
                                   level_refine,
                                   config_map);

  const auto refine_type = core::get_init_indicator(config_map);

  if (refine_type == +core::InitConditionsIndicator::ALWAYS_REFINE)
  {
    Kokkos::parallel_for("kalypsso::godunov_hydro::InitImplodeRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineAlways>(0, local_num_octants),
                         functor);
  }
  else if (refine_type == +core::InitConditionsIndicator::GEOMETRIC)
  {
    Kokkos::parallel_for("kalypsso::godunov_hydro::InitImplodeRefineFunctor",
                         Kokkos::RangePolicy<exec_space, TagRefineGeometric>(0, local_num_octants),
                         functor);
  }
  else
  {
    KALYPSSO_ERROR("Unknown value for refine indicator method.");
  }

} // InitImplodeRefineFunctor::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitImplodeRefineFunctor<dim, device_t>::operator()(TagRefineAlways const &,
                                                    const size_t & iOct) const
{
  m_amrflags(iOct) = AMRContextBase::KALYPSSO_DO_REFINE;
} // InitImplodeRefineFunctor::operator()

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitImplodeRefineFunctor<dim, device_t>::operator()(TagRefineGeometric const &,
                                                    const size_t & iOct) const
{

  // get block orchard key
  const auto key = m_orchard_keys(iOct);

  // get block level
  const auto level = orchard_key_t<dim>::level(key);

  // compute block length (in real space units)
  const auto block_length = compute_block_length<dim>(level) * m_scaling_factor;

  // distance threshold
  const auto d_th = block_length * KALYPSSO_NUM(0.95);

  // default : do nothing, i.e. neither refine or coarsen
  auto flag = AMRContextBase::KALYPSSO_DO_NOTHING;

  // only look at level - 1
  if (level == m_level_refine)
  {

    // compute physical x,y,z for the block center
    constexpr auto centering = true;
    const auto     xyz_vertex = orchard_key_to_vertex_coord<dim>(key, centering);
    const auto     xyz = vertex_coord_to_real_space<dim>(xyz_vertex, m_scaling_factor, m_xyz_min);

    if constexpr (dim == 2)
    {
      auto const & x = xyz[IX];
      auto const & y = xyz[IY];

      auto const & xmin = m_xyz_min[IX];
      auto const & ymin = m_xyz_min[IY];
      auto const & xmax = m_xyz_max[IX];

      if (m_implode_params.shape == +ImplodeShape::DIAGONAL)
      {
        const auto c = xmin / 4 + 3 * xmax / 4 + ymin;

        // compute distance to discontinuous interface
        const auto d = fabs(x + y - c);

        if (d < d_th)
          flag = AMRContextBase::KALYPSSO_DO_REFINE;
      }
      else if (m_implode_params.shape == +ImplodeShape::DIAMOND)
      {
        const auto c = xmin / 4 + 3 * xmax / 4 + ymin + KALYPSSO_NUM(1e-10);

        const auto d0 = fabs(x + y - c);
        const auto d1 = fabs(x + y + c);
        const auto d2 = fabs(x - y - c);
        const auto d3 = fabs(x - y + c);
        if (d0 < d_th or d1 < d_th or d2 < d_th or d3 < d_th)
          flag = AMRContextBase::KALYPSSO_DO_REFINE;
      }
    }
    else if constexpr (dim == 3)
    {
      auto const & x = xyz[IX];
      auto const & y = xyz[IY];
      auto const & z = xyz[IZ];

      auto const & xmin = m_xyz_min[IX];
      auto const & ymin = m_xyz_min[IY];
      auto const & zmin = m_xyz_min[IZ];
      auto const & xmax = m_xyz_max[IX];

      if (m_implode_params.shape == +ImplodeShape::DIAGONAL)
      {
        const auto c = xmin / 4 + 3 * xmax / 4 + ymin + zmin;

        // compute distance to interface
        const auto d = fabs(x + y + z - c);

        if (d < d_th)
          flag = AMRContextBase::KALYPSSO_DO_REFINE;
      }
      else if (m_implode_params.shape == +ImplodeShape::DIAMOND)
      {
        const auto c = xmin / 4 + 3 * xmax / 4 + ymin + zmin + KALYPSSO_NUM(1e-10);

        const auto d0 = fabs(x + y + z - c);
        const auto d1 = fabs(x + y + z + c);
        const auto d2 = fabs(x - y + z - c);
        const auto d3 = fabs(x - y + z + c);
        const auto d4 = fabs(x + y - z - c);
        const auto d5 = fabs(x + y - z + c);
        const auto d6 = fabs(x - y - z - c);
        const auto d7 = fabs(x - y - z + c);
        if (d0 < d_th or d1 < d_th or d2 < d_th or d3 < d_th or d4 < d_th or d5 < d_th or
            d6 < d_th or d7 < d_th)
          flag = AMRContextBase::KALYPSSO_DO_REFINE;
      }
    }
  } // end if level == level_refine

  // perform max reduction
  // if all cell in current block agree on COARSEN => do coarsen
  // if a single cell in current block disagree on coarsening => do nothing or refine
  // if a single cell in current block needs to refine => do refine
  m_amrflags(iOct) = flag;

} // end InitImplodeRefineFunctor::operator ()

template class InitImplodeRefineFunctor<2, kalypsso::DefaultDevice>;
template class InitImplodeRefineFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitImplode<dim, device_t>::apply(SolverGodunovHydro<dim, device_t> & solver)
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
  InitImplodeDataFunctor<dim, device_t>::apply(solver.U(),
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
      InitImplodeDataFunctor<dim, device_t>::apply(solver.U(),
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
      InitImplodeRefineFunctor<dim, device_t>::apply(solver.U(),
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
      InitImplodeDataFunctor<dim, device_t>::apply(solver.U(),
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

} // InitImplode::apply

template class InitImplode<2, kalypsso::DefaultDevice>;
template class InitImplode<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso
