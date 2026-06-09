// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file AddGravitySourceTerm.cpp
 * \brief \copybrief AddGravitySourceTerm.h
 */

#include "AddGravitySourceTerm.h"

namespace kalypsso
{
namespace godunov_hydro
{

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
void
AddGravitySourceTerm<dim, device_t>::apply(ConfigMap const &        config_map,
                                           DataArrayBlock_t const & Uold,
                                           DataArrayBlock_t const & Unew,
                                           int32_t                  local_num_octants,
                                           real_t                   dt)
{

  const auto gravity_field = get_uniform_gravity_vector<dim>(config_map);

  AddGravitySourceTerm functor(gravity_field, Uold, Unew, dt);

  // compute total number of cells
  const auto nbCellsPerLeaf = Uold.num_cells();
  const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

  Kokkos::parallel_for("kalypsso::godunov_hydro::AddGravitySourceTerm",
                       Kokkos::RangePolicy<exec_space>(0, nbCellsTotal),
                       functor);
} // apply

// // ====================================================================
// // ====================================================================
// template <size_t dim, typename device_t>
// KOKKOS_INLINE_FUNCTION
// void
// AddGravitySourceTerm<dim, device_t>::operator()(const index_t & global_index) const
// {
//   // convert global index into
//   // - octant id
//   // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
//   const auto i_oct = global_index / m_nbCellsPerLeaf;
//   const auto cell_index = global_index - i_oct * m_nbCellsPerLeaf;

//   const auto rho_old = m_Uold(cell_index, Hydro<dim>::ID, i_oct);
//   const auto rho_new = m_Unew(cell_index, Hydro<dim>::ID, i_oct);

//   real_t u = ZERO_F, v = ZERO_F, w = ZERO_F;
//   u = m_Unew(cell_index, Hydro<dim>::IU, i_oct) / rho_new;
//   v = m_Unew(cell_index, Hydro<dim>::IV, i_oct) / rho_new;
//   if constexpr (dim == 3)
//   {
//     w = m_Unew(cell_index, Hydro<dim>::IW, i_oct) / rho_new;
//   }

//   // compute kinetic energy
//   auto ekin = [&]() {
//     if constexpr (dim == 2)
//     {
//       return HALF_F * rho_new * (u * u + v * v);
//     }
//     if constexpr (dim == 3)
//     {
//       return HALF_F * rho_new * (u * u + v * v + w * w);
//     }
//   }();

//   // retrieve internal energy, before correcting ekin
//   auto eint = m_Unew(cell_index, Hydro<dim>::IP], i_oct) - ekin;

//   // update velocity
//   u += rho_old / rho_new * m_grav[IX] * m_dt;
//   v += rho_old / rho_new * m_grav[IY] * m_dt;
//   if constexpr (dim == 3)
//   {
//     w += rho_old / rho_new * m_grav[IZ] * m_dt;
//   }
//   // u += m_grav[IX] * m_dt;
//   // v += m_grav[IY] * m_dt;
//   // if constexpr (dim == 3)
//   // {
//   //   w += m_grav[IZ] * m_dt;
//   // }

//   // update momentum
//   m_Unew(cell_index, Hydro<dim>::IU, i_oct) = rho_new * u;
//   m_Unew(cell_index, Hydro<dim>::IV, i_oct) = rho_new * v;
//   if constexpr (dim == 3)
//   {
//     m_Unew(cell_index, Hydro<dim>::IW, i_oct) = rho_new * w;
//   }

//   // update kinetic energy
//   ekin = [&]() {
//     if constexpr (dim == 2)
//     {
//       return HALF_F * rho_new * (u * u + v * v);
//     }
//     if constexpr (dim == 3)
//     {
//       return HALF_F * rho_new * (u * u + v * v + w * w);
//     }
//   }();

//   // update total energy
//   m_Unew(cell_index, Hydro<dim>::IP, i_oct) = eint + ekin;
// } // operator()

// ====================================================================
// ====================================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
AddGravitySourceTerm<dim, device_t>::operator()(const index_t & global_index) const
{
  // convert global index into
  // - octant id
  // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
  const auto i_oct = global_index / m_nbCellsPerLeaf;
  const auto cell_index = global_index - i_oct * m_nbCellsPerLeaf;

  const auto rho_old = m_Uold(cell_index, Hydro<dim>::ID, i_oct);
  const auto rho_new = m_Unew(cell_index, Hydro<dim>::ID, i_oct);

  // read momentum before gravity update
  auto rhou = m_Unew(cell_index, Hydro<dim>::IU, i_oct);
  auto rhov = m_Unew(cell_index, Hydro<dim>::IV, i_oct);
  auto rhow = ZERO_F;
  if constexpr (dim == 3)
  {
    rhow = m_Unew(cell_index, Hydro<dim>::IW, i_oct);
  }

  // compute kinetic energy before updating momentum
  auto ekin_old = HALF_F * (rhou * rhou + rhov * rhov) / rho_new;
  if constexpr (dim == 3)
  {
    ekin_old += HALF_F * (rhow * rhow) / rho_new;
  }

  // update momentum
  rhou += m_dt * m_grav[IX] * (rho_old + rho_new) / 2;
  rhov += m_dt * m_grav[IY] * (rho_old + rho_new) / 2;
  if constexpr (dim == 3)
  {
    rhow += m_dt * m_grav[IZ] * (rho_old + rho_new) / 2;
  }

  m_Unew(cell_index, Hydro<dim>::IU, i_oct) = rhou;
  m_Unew(cell_index, Hydro<dim>::IV, i_oct) = rhov;
  if constexpr (dim == 3)
  {
    m_Unew(cell_index, Hydro<dim>::IW, i_oct) = rhow;
  }

  // compute kinetic energy after updating momentum
  auto ekin_new = HALF_F * (rhou * rhou + rhov * rhov) / rho_new;
  if constexpr (dim == 3)
  {
    ekin_new += HALF_F * (rhow * rhow) / rho_new;
  }

  // update total energy
  m_Unew(cell_index, Hydro<dim>::IP, i_oct) += (ekin_new - ekin_old);

} // operator()

template class AddGravitySourceTerm<2, kalypsso::DefaultDevice>;
template class AddGravitySourceTerm<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso
