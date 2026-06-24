// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file utils_hydro.h
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_MODELS_UTILS_HYDRO_H_
#define KALYPSSO_GODUNOV_HYDRO_MODELS_UTILS_HYDRO_H_

#include <type_traits>

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/HydroParams.h>
#include <kalypsso/core/models/HydroSettings.h>

#include <godunov_hydro/models/Hydro.h>
#include <godunov_hydro/models/HydroState.h>
#include <godunov_hydro/eos/EosWrapper.h>

namespace kalypsso
{

namespace godunov_hydro
{

namespace models
{

// ================================================================================
// ================================================================================
/**
 * Compute specific kinetic energy from primitive variables.
 *
 * \param[in] q  primitive variables vector
 *
 * \return specific kinetic energy
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION real_t
compute_ekin_from_primitives(const HydroState<dim> & q)
{
  using Hydro = models::Hydro<dim>;

  real_t ekin = q[Hydro::IU] * q[Hydro::IU] + q[Hydro::IV] * q[Hydro::IV];
  if constexpr (dim == 3)
    ekin += q[Hydro::IW] * q[Hydro::IW];

  return HALF_F * ekin;
}

// ================================================================================
// ================================================================================
/**
 * Convert conservative variables (rho, rho*u, rho*v, rho*e_tot) to
 * primitive variables (rho,u,v,p)
 *
 * \param[in]  u  conservative variables array
 * \param[in]  settings
 * \param[out] valid will be true if pressure and internal energy are positive
 * \return     q  primitive    variables array
 */
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION HydroState<dim>
                       compute_primitives(const HydroState<dim> &           u,
                                          const HydroSettings &             settings,
                                          eos::EosWrapper<device_t> const & eos,
                                          bool &                            valid)
{
  using Hydro = models::Hydro<dim>;

  real_t smallr = settings.smallr;

  const auto                 d = fmax(u[Hydro::ID], smallr);
  Kokkos::Array<real_t, dim> uvw;
  uvw[IX] = u[Hydro::IU] / d;
  uvw[IY] = u[Hydro::IV] / d;
  if constexpr (dim == 3)
  {
    uvw[IZ] = u[Hydro::IW] / d;
  }

  // specific kinetic energy
  auto eken = HALF_F * (uvw[IX] * uvw[IX] + uvw[IY] * uvw[IY]);
  if constexpr (dim == 3)
  {
    eken += HALF_F * (uvw[IZ] * uvw[IZ]);
  }

  // specific internal energy
  const auto specific_eint = u[Hydro::IE] / d - eken;

  // compute pressure
  const auto p = eos.pressure_from_specific_eint(specific_eint, d);

  valid = p >= 0;

  if (settings.abort_when_negative_pressure and !valid)
  {
    Kokkos::abort("Negative pressure detected - can't proceed further");
  }

  HydroState<dim> q;
  q[Hydro::ID] = d;
  q[Hydro::IP] = p;
  q[Hydro::IU] = uvw[IX];
  q[Hydro::IV] = uvw[IY];

  if constexpr (dim == 3)
  {
    q[Hydro::IW] = uvw[IZ];
  }

  return q;

} // compute_primitives

// ================================================================================
// ================================================================================
/**
 * Convert conservative variables (rho, rho*u, rho*v, e) to
 * primitive variables (rho,u,v,p)
 * \param[in]  u  conservative variables array
 * \param[in]  settings
 * \return     q  primitive    variables array
 *
 * \note discard argument "valid"
 */
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION HydroState<dim>
                       compute_primitives(HydroState<dim> const &           u,
                                          HydroSettings const &             settings,
                                          eos::EosWrapper<device_t> const & eos)
{

  [[maybe_unused]] bool valid = true;
  return compute_primitives<dim, device_t>(u, settings, eos, valid);

} // compute_primitives

// ================================================================================
// ================================================================================
/**
 * Convert primitive variables (rho,u,v,p) to
 * conservative variables (rho, rho*u, rho*v, rho*e_tot)
 *
 * \param[in]  q  primitive    variables array
 * \param[in]  settings
 * \param[out] valid will be true if pressure and internal energy are positive
 * \return     u  conservative variables array
 */
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION HydroState<dim>
                       compute_conservative_variables(const HydroState<dim> &           q,
                                                      const HydroSettings &             settings,
                                                      eos::EosWrapper<device_t> const & eos,
                                                      bool &                            valid)
{
  using Hydro = models::Hydro<dim>;

  valid = q[Hydro::IP] >= 0;

  if (settings.abort_when_negative_pressure and !valid)
  {
    Kokkos::abort("Negative pressure detected - can't proceed further");
  }

  real_t smallr = settings.smallr;

  const auto d = fmax(q[Hydro::ID], smallr);

  HydroState<dim> u;
  u[Hydro::ID] = d;
  u[Hydro::IU] = d * q[Hydro::IU];
  u[Hydro::IV] = d * q[Hydro::IV];

  if constexpr (dim == 3)
  {
    u[Hydro::IW] = d * q[Hydro::IW];
  }

  // volumic kinetic energy
  auto volumic_ekin = HALF_F * d * (q[Hydro::IU] * q[Hydro::IU] + q[Hydro::IV] * q[Hydro::IV]);
  if constexpr (dim == 3)
  {
    volumic_ekin += HALF_F * d * (q[Hydro::IW] * q[Hydro::IW]);
  }

  // volumic internal energy
  const auto volumic_eint = eos.volumic_eint_from_pressure(q[Hydro::IP], d);

  u[Hydro::IE] = volumic_eint + volumic_ekin;

  return u;

} // compute_conservative_variables

} // namespace models

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_MODELS_UTILS_HYDRO_H_
