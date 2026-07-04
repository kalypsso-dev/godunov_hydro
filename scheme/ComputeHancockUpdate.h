// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeHancockUpdate.h
 *
 * Muscl-Hancock is a second order in time and space Godunov method.
 * Here we isolate the Hancock half time step in order to decouple space reconstruction from time
 * integration. This we allow to have either MUSCL-Hancock or THINC-Hancock half time step.
 *
 * Just as a reminder, here are the main steps of the MUSCL-Hancock algorithm:
 *
 * 1. Compute primitive variables from conservative ones
 * (with the help of an equation of state): \f${\bf Q}_{i}^n({\bf U}_{i}^n)\f$
 *
 * 2. Space reconstruction of primitive variables from cell-center to left and right
 * face center at time \f$t_n\f$:
 * - \f${\bf Q}_{i,L}^n={\bf Q}_{i}^n-\frac{1}{2}\overline{\delta}_i\f$,
 * - \f${\bf Q}_{i,R}^n={\bf Q}_{i}^n+\frac{1}{2}\overline{\delta}_i\f$
 *
 * 3. Half time step evolution of face reconstructed states:
 *
 * \f{align*}{
   {\bf U}_{i,L}^{n+1/2}& ={\bf U}_{i,L}^n-\frac{\Delta t}{2\Delta x}\left[ {\bf F}_x({\bf
 Q}_{i,R}^n) -{\bf F}_x({\bf Q}_{i,L}^n)\right]\\
   {\bf U}_{i,R}^{n+1/2}& ={\bf U}_{i,R}^n-\frac{\Delta t}{2\Delta x}\left[ {\bf F}_x({\bf
 Q}_{i,R}^n) -{\bf F}_x({\bf Q}_{i,L}^n)\right]
  \f}
 *
 * 4. Compute flux using a Riemann solver (e.g. HLLC):
 *  \f${\bf F}_{x,i-1/2}^{n+1/2}=RS({\bf U}_{i-1,R}^{n+1/2},{\bf U}_{i,L}^{n+1/2})\f$
 *
 * 5.Time update from \f$t_n\f$ to \f$t_{n+1}\f$:
 *
 *  \f${\bf U}_{i}^{n+1}={\bf U}_{i}^{n}-\frac{\Delta}{\Delta x}\left[ {\bf
 F}_{x,i+1/2}^{n+1/2}-{\bf F}_{x,i-1/2}^{n+1/2}\right]\f$
 *
 *
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_COMPUTEHANCOCKUPDATE_H_
#define KALYPSSO_GODUNOV_HYDRO_COMPUTEHANCOCKUPDATE_H_

#include <godunov_hydro/models/HydroState.h>

namespace kalypsso
{

namespace godunov_hydro
{

/**
 * Compute hydrodynamics fluxes.
 *
 * \param[in] q primitive variables: rho, u, v, w, P
 * \param[in] eos Equation of state wrapper
 *
 * \return hydrodynamics flux in direction dir
 */
template <size_t dim, typename device_t, int dir>
KOKKOS_FUNCTION HydroState<dim>
                compute_hydro_flux(const HydroState<dim> & q, EosWrapper<device_t> const & eos)
{
  using Hydro = kalypsso::godunov_hydro::models::Hydro<dim>;

  if constexpr (dim == 2)
  {
    auto const & rho = q[Hydro::ID];
    auto const & u = q[Hydro::IU];
    auto const & v = q[Hydro::IV];
    auto const & P = q[Hydro::IP];

    // total energy
    const real_t E = eos.specific_eint_from(P, rho) + HALF_F * (u * u + v * v);

    HydroState<dim> flux;

    if constexpr (dir == IX)
    {
      flux[Hydro::ID] = rho * u;
      flux[Hydro::IU] = rho * u * u + P;
      flux[Hydro::IV] = rho * u * v;
      flux[Hydro::IP] = u * (E + P);
      return flux;
    }
    else if constexpr (dir == IY)
    {
      flux[Hydro::ID] = rho * v;
      flux[Hydro::IU] = rho * v * u;
      flux[Hydro::IV] = rho * v * v + P;
      flux[Hydro::IP] = v * (E + P);
      return flux;
    }
  } // end dim==2
  else if constexpr (dim == 3)
  {
    auto const & rho = q[Hydro::ID];
    auto const & u = q[Hydro::IU];
    auto const & v = q[Hydro::IV];
    auto const & w = q[Hydro::IW];
    auto const & P = q[Hydro::IP];

    // total energy
    const real_t E = eos.specific_eint_from(P, rho) + HALF_F * (u * u + v * v + w * w);

    HydroState<dim> flux;

    if constexpr (dir == IX)
    {
      flux[Hydro::ID] = rho * u;
      flux[Hydro::IU] = rho * u * u + P;
      flux[Hydro::IV] = rho * u * v;
      flux[Hydro::IW] = rho * u * w;
      flux[Hydro::IP] = u * (E + P);
      return flux;
    }
    else if constexpr (dir == IY)
    {
      flux[Hydro::ID] = rho * v;
      flux[Hydro::IU] = rho * v * u;
      flux[Hydro::IV] = rho * v * v + P;
      flux[Hydro::IW] = rho * v * w;
      flux[Hydro::IP] = v * (E + P);
      return flux;
    }
    else if constexpr (dir == IZ)
    {
      flux[Hydro::ID] = rho * w;
      flux[Hydro::IU] = rho * w * u;
      flux[Hydro::IV] = rho * w * v;
      flux[Hydro::IW] = rho * w * w + P;
      flux[Hydro::IP] = w * (E + P);
      return flux;
    }

  } // end dim==3

} // compute_hydro_flux

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_COMPUTEHANCOCKUPDATE_H_
