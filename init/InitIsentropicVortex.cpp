// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitIsentropicVortex.cpp
 */

#include <godunov_hydro/init/InitIsentropicVortex.h>
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
InitIsentropicVortexDataFunctor<dim, device_t>::apply(
  DataArrayBlock_t const &             Udata,
  FieldMap<core::models::Hydro>        fm,
  orchard_key_view_t<device_t> const & orchard_keys,
  int32_t                              local_num_octants,
  HydroSettings const &                settings,
  ConfigMap const &                    config_map)
{
  IsentropicVortexParams ivParams = IsentropicVortexParams(config_map);

  // data init functor
  InitIsentropicVortexDataFunctor functor(
    Udata, fm, orchard_keys, local_num_octants, settings, ivParams, config_map);

  // compute total number of cells
  const auto nbCellsPerLeaf = Udata.num_cells();
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  Kokkos::parallel_for("kalypsso::godunov_hydro::InitIsentropicVortexDataFunctor",
                       Kokkos::RangePolicy<exec_space>(0, nbCellsTotal),
                       functor);

} // InitIsentropicVortexDataFunctor::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitIsentropicVortexDataFunctor<dim, device_t>::operator()(const int32_t & global_index) const
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

  //
  // isentropic vortex problem parameters
  //

  // ambient flow
  auto const & rho_a = m_iParams.rho_a;
  auto const & T_a = m_iParams.T_a;
  auto const & u_a = m_iParams.u_a;
  auto const & v_a = m_iParams.v_a;
  auto const & beta = m_iParams.beta;

  // vortex center
  auto const & vortex_x = m_iParams.vortex_x;
  auto const & vortex_y = m_iParams.vortex_y;

  // compute ix,iy,iz of local cell inside
  // block from index
  auto iCoord = cellindex_to_coord<dim>(cell_index, block_sizes);

  // get block orchard key
  const auto key = m_orchard_keys(iOct);

  // compute physical x,y,z for that cell (cell center)
  const auto xyz_vertex = orchard_key_to_cell_coord<dim>(key, iCoord, block_sizes[IX]);

  auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, m_scaling_factor, m_xyz_min);

  // compute relative coordinates versus vortex center
  const auto xp = xyz[IX] - vortex_x;
  const auto yp = xyz[IY] - vortex_y;
  const auto r = sqrt(xp * xp + yp * yp);

  const auto du = -yp * beta / (2 * PI_F) * exp(HALF_F * (ONE_F - r * r));
  const auto dv = xp * beta / (2 * PI_F) * exp(HALF_F * (ONE_F - r * r));

  const auto T =
    T_a - (m_gamma - ONE_F) * beta * beta / (8 * m_gamma * PI_F * PI_F) * exp(ONE_F - r * r);
  const auto rho = rho_a * pow(T / T_a, ONE_F / (m_gamma - 1));

  m_Udata(cell_index, m_fm[ID], iOct) = rho;
  m_Udata(cell_index, m_fm[IU], iOct) = rho * (u_a + du);
  m_Udata(cell_index, m_fm[IV], iOct) = rho * (v_a + dv);
  m_Udata(cell_index, m_fm[IP], iOct) = rho * T / (m_gamma - ONE_F) +
                                        HALF_F * rho * (u_a + du) * (u_a + du) +
                                        HALF_F * rho * (v_a + dv) * (v_a + dv);

  if constexpr (dim == 3)
    m_Udata(cell_index, m_fm[IW], iOct) = 0.0;

} // end InitIsentropicVortexDataFunctor::operator ()

template class InitIsentropicVortexDataFunctor<2, kalypsso::DefaultDevice>;
template class InitIsentropicVortexDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitIsentropicVortex<dim, device_t>::apply(SolverGodunovHydro<dim, device_t> & solver)
{

  auto              amr_mesh = solver.amr_mesh();
  ConfigMap const & config_map = solver.config_map();
  const auto        settings = HydroSettings(solver.config_map());
  const int         level_min = solver.hydro_params().level_min;
  const int         level_max = solver.hydro_params().level_max;

  // IsentropicVortexParams blastParams = IsentropicVortexParams(config_map);
  //  field manager index array
  //  auto fm = solver.hydro().get_fieldmap();

  constexpr bool do_reset_ghosts = true;
  solver.update_mesh(do_reset_ghosts);

  // resize Udata
  solver.resize_solver_data();

  // first init of Udata
  InitIsentropicVortexDataFunctor<dim, device_t>::apply(solver.U(),
                                                        solver.hydro().get_fieldmap(),
                                                        solver.mesh_map()->orchard_keys(),
                                                        solver.amr_mesh()->local_num_quadrants(),
                                                        settings,
                                                        config_map);

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
    InitIsentropicVortexDataFunctor<dim, device_t>::apply(solver.U(),
                                                          solver.hydro().get_fieldmap(),
                                                          solver.mesh_map()->orchard_keys(),
                                                          solver.amr_mesh()->local_num_quadrants(),
                                                          settings,
                                                          config_map);

    // update level
    ++level;

  } // end while level<level_max

#ifdef KALYPSSO_CORE_USE_MPI
  // load balancing (= repartitioning) the octree mesh + userdata over the MPI processes.
  // U and U2 will be resized
  solver.do_load_balancing();
#endif

} // InitIsentropicVortex::apply

template class InitIsentropicVortex<2, kalypsso::DefaultDevice>;
template class InitIsentropicVortex<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso
