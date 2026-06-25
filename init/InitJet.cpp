// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitJet.cpp
 */

#include <godunov_hydro/init/InitJet.h>
#include <godunov_hydro/SolverGodunovHydro.h>
#include <godunov_hydro/models/utils_hydro.h>

#include <kalypsso/core/orchard_key_utils.h>

namespace kalypsso
{
namespace godunov_hydro
{
// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitJetDataFunctor<dim, device_t>::apply(DataArrayBlock_t const &             Udata,
                                         orchard_key_view_t<device_t> const & orchard_keys,
                                         int32_t                              local_num_octants,
                                         HydroSettings const &                settings,
                                         ConfigMap const &                    config_map)
{
  // read primitive variables
  HydroState<dim> q;
  q[Hydro<dim>::ID] = config_map.getReal("jet", "rho_inside", KALYPSSO_NUM(1.0));
  q[Hydro<dim>::IP] = config_map.getReal("jet", "pressure_inside", KALYPSSO_NUM(1.0));
  q[Hydro<dim>::IU] = config_map.getReal("jet", "u_inside", KALYPSSO_NUM(0.0));
  q[Hydro<dim>::IV] = config_map.getReal("jet", "v_inside", KALYPSSO_NUM(0.0));
  if constexpr (dim == 3)
    q[Hydro<dim>::IW] = config_map.getReal("jet", "w_inside", KALYPSSO_NUM(0.0));

  // Equation of state wrapper
  const auto            eos = EosWrapper<HostDevice>(config_map);
  [[maybe_unused]] bool valid = true;

  const auto U_jet =
    models::compute_conservative_variables<dim, HostDevice>(q, settings, eos, valid);

  // data init functor
  InitJetDataFunctor functor(Udata, orchard_keys, local_num_octants, settings, U_jet, config_map);

  // compute total number of cells
  const auto nbCellsPerLeaf = Udata.num_cells();
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  Kokkos::parallel_for("kalypsso::godunov_hydro::InitJetDataFunctor",
                       Kokkos::RangePolicy<exec_space>(0, nbCellsTotal),
                       functor);

} // InitJetDataFunctor::apply

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
InitJetDataFunctor<dim, device_t>::operator()(const int32_t & global_index) const
{

  // convert global index into
  // - octant id
  // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
  const auto iOct = global_index / m_Udata.num_cells();
  const auto cell_index = global_index - iOct * m_Udata.num_cells();

  // initialize
  m_Udata(cell_index, Hydro<dim>::ID, iOct) = m_U_jet[Hydro<dim>::ID];
  m_Udata(cell_index, Hydro<dim>::IE, iOct) = m_U_jet[Hydro<dim>::IE];
  m_Udata(cell_index, Hydro<dim>::IU, iOct) = m_U_jet[Hydro<dim>::IU];
  m_Udata(cell_index, Hydro<dim>::IV, iOct) = m_U_jet[Hydro<dim>::IV];

  if constexpr (dim == 3)
    m_Udata(cell_index, Hydro<dim>::IW, iOct) = m_U_jet[Hydro<dim>::IW];

} // end InitJetDataFunctor::operator ()

template class InitJetDataFunctor<2, kalypsso::DefaultDevice>;
template class InitJetDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
InitJet<dim, device_t>::apply(SolverGodunovHydro<dim, device_t> & solver)
{

  auto              amr_mesh = solver.amr_mesh();
  ConfigMap const & config_map = solver.config_map();
  const auto        settings = HydroSettings(solver.config_map());

  constexpr bool do_reset_ghosts = true;
  solver.update_mesh(do_reset_ghosts);

  // resize Udata
  solver.resize_solver_data();

  // first init of Udata
  InitJetDataFunctor<dim, device_t>::apply(solver.U(),
                                           solver.mesh_map()->orchard_keys(),
                                           solver.amr_mesh()->local_num_quadrants(),
                                           settings,
                                           config_map);

#ifdef KALYPSSO_CORE_USE_MPI
  // load balancing (= repartitioning) the octree mesh + userdata over the MPI processes.
  // U and U2 will be resized
  solver.do_load_balancing();
#endif

} // InitJet::apply

template class InitJet<2, kalypsso::DefaultDevice>;
template class InitJet<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso
