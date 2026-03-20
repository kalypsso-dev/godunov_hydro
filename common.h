// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file common.h
 *
 * \brief Contains a shared set of values and types.
 */

#ifndef KALYPSSO_GODUNOV_HYDRO_COMMON_H_
#define KALYPSSO_GODUNOV_HYDRO_COMMON_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/DataArrayBlock.h>
#include <kalypsso/core/models/Hydro.h>
#include <kalypsso/core/models/HydroSettings.h>
#include <kalypsso/core/models/HydroState.h>
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/AMRContext.h>
#include <kalypsso/core/brick_utils.h>

#include <godunov_hydro/eos/EosWrapper.h>

#include <../better-enums/enum.h>

namespace kalypsso
{

namespace godunov_hydro
{

template <size_t dim, typename device_t>
class SolverGodunovHydro;

//! Array of Orchard keys
template <typename device_t>
using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

//! Shorthand for the hydrodynamic model
using Hydro = core::models::Hydro;

//! Hashmap type from Orchard keys to octant index
template <typename device_t>
using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;

//! AMR flags array type
template <size_t dim, typename device_t>
using amrflags_view_t = typename AMRContext<dim, device_t>::amrflags_view_t;

//! Region initial states array
template <size_t dim, typename device_t>
using InitialStates = Kokkos::View<HydroState<dim> *, device_t>;

// ===============================================================================
// ===============================================================================
/**
 * Read initial hydrodynamics state.
 *
 * This is useful when doing an initialization that is constant (i.e. space uniform)
 * inside spatial regions.
 *
 * Currently region shapes are defined by each test case.
 * Soon we will also be able to define region shapes through geometric modelling.
 *
 * \return a vector of conservative variables
 *
 * \param[in] i_region region id
 * \param[in] config_map application's configuration parameters
 */
template <size_t dim>
HydroState<dim>
get_region_init_state(const int32_t                       i_region,
                      eos::EosWrapper<HostDevice> const & eos_wrapper,
                      const ConfigMap &                   config_map)
{

  // variables specific (per mass unit)
  HydroState<dim> q;

  const auto section = "region" + std::to_string(i_region);
  const auto rho = config_map.getReal(section, "rho", KALYPSSO_NUM(1.0));

  q[Hydro::ID] = rho;
  q[Hydro::IU] = config_map.getReal(section, "u", KALYPSSO_NUM(0.0));
  q[Hydro::IV] = config_map.getReal(section, "v", KALYPSSO_NUM(0.0));
  if constexpr (dim == 3)
    q[Hydro::IW] = config_map.getReal(section, "w", KALYPSSO_NUM(0.0));

  const auto p = config_map.getReal(section, "p", KALYPSSO_NUM(1.0));
  assertm(rho >= 0, "Invalid value for density");
  assertm(p >= 0, "Invalid value for pressure");

  const auto eint_specific = eos_wrapper.specific_eint_from_pressure(p, rho);

  const real_t ekin_specific = [](HydroState<dim> & qq) {
    if constexpr (dim == 2)
      return HALF_F * (qq[Hydro::IU] * qq[Hydro::IU] + qq[Hydro::IV] * qq[Hydro::IV]);
    else if constexpr (dim == 3)
      return HALF_F * (qq[Hydro::IU] * qq[Hydro::IU] + qq[Hydro::IV] * qq[Hydro::IV] +
                       qq[Hydro::IW] * qq[Hydro::IW]);
  }(q);

  // compute conservative variables
  HydroState<dim> Ucons;

  Ucons[Hydro::ID] = rho;
  Ucons[Hydro::IE] = (eint_specific + ekin_specific) * rho;
  Ucons[Hydro::IU] = q[Hydro::IU] * rho;
  Ucons[Hydro::IV] = q[Hydro::IV] * rho;
  if constexpr (dim == 3)
    Ucons[Hydro::IW] = q[Hydro::IW] * rho;

  return Ucons;

} // get_region_init_state

// ===============================================================================
// ===============================================================================
/**
 * Get an array of initiaal states (one per region).
 *
 * This is useful for all simple test case that are initialized with a uniform state per region.
 */
template <size_t dim, typename device_t>
auto
get_initial_states(ConfigMap const & config_map, int nb_regions) -> InitialStates<dim, device_t>
{
  InitialStates<dim, device_t> initial_states("Initial states", static_cast<uint>(nb_regions));
  auto                         initial_states_host = Kokkos::create_mirror_view(initial_states);

  const auto eos_wrapper = eos::EosWrapper<HostDevice>(config_map);

  for (int i_region = 0; i_region < nb_regions; i_region++)
  {
    initial_states_host(i_region) = get_region_init_state<dim>(i_region, eos_wrapper, config_map);
  }
  Kokkos::deep_copy(initial_states, initial_states_host);

  return initial_states;
}

// ===============================================================================
// ===============================================================================
/**
 * Compute volumic kinetic energy of a pure state.
 *
 * \param[in] initial_states array of conservative variables hydrodynamics states
 * \param[in] i_state state index to initial_states
 */
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION real_t
compute_volumic_ekin(InitialStates<dim, device_t> const & initial_states, int i_state)
{
  auto ekin = initial_states(i_state)[Hydro::IU] * initial_states(i_state)[Hydro::IU] +
              initial_states(i_state)[Hydro::IV] * initial_states(i_state)[Hydro::IV];
  if constexpr (dim == 3)
  {
    ekin += initial_states(i_state)[Hydro::IW] * initial_states(i_state)[Hydro::IW];
  }
  ekin /= (TWO_F * initial_states(i_state)[Hydro::ID]);

  return ekin;
}

// ===============================================================================
// ===============================================================================
/**
 * Compute volumic kinetic energy of a pure state (conservative variables).
 */
template <size_t dim>
KOKKOS_INLINE_FUNCTION real_t
compute_volumic_ekin(HydroState<dim> const & U)
{
  auto ekin = U[Hydro::IU] * U[Hydro::IU] + U[Hydro::IV] * U[Hydro::IV];

  if constexpr (dim == 3)
    ekin += U[Hydro::IW] * U[Hydro::IW];

  ekin /= (TWO_F * U[Hydro::ID]);

  return ekin;
}

// ===============================================================================
// ===============================================================================
/**
 * Compute a mixed state between two pure states.
 *
 * \param[in] initial_states array of conservative variables hydrodynamics states
 * \param[in] state0 first state index to initial_states
 * \param[in] vf0 first state volume fraction
 * \param[in] state1 second state index to initial_states
 * \param[in] vf1 second state volume fraction
 */
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION HydroState<dim>
                       compute_mixed_state(HydroState<dim> const & state0,
                                           real_t                  vf0,
                                           HydroState<dim> const & state1,
                                           real_t                  vf1)
{
  HydroState<dim> U;

  U[Hydro::ID] = vf0 * state0[Hydro::ID] + vf1 * state1[Hydro::ID];
  U[Hydro::IU] = vf0 * state0[Hydro::IU] + vf1 * state1[Hydro::IU];
  U[Hydro::IV] = vf0 * state0[Hydro::IV] + vf1 * state1[Hydro::IV];
  if constexpr (dim == 3)
    U[Hydro::IW] = vf0 * state0[Hydro::IW] + vf1 * state1[Hydro::IW];

  // compute pure states internal energy
  const auto eint0 = state0[Hydro::IE] - compute_volumic_ekin<dim>(state0);
  const auto eint1 = state1[Hydro::IE] - compute_volumic_ekin<dim>(state1);

  // compute mixed internal energy
  const auto eint_mixed = vf0 * eint0 + vf1 * eint1;

  // compute mixed kinetic energy
  const auto ekin_mixed = compute_volumic_ekin<dim>(U);

  // set mixed total energy
  U[Hydro::IE] = eint_mixed + ekin_mixed;

  return U;
}

// ===============================================================================
// ===============================================================================
/**
 * Compute a mixed state between two pure states.
 *
 * \param[in] initial_states array of conservative variables hydrodynamics states
 * \param[in] state0 first state index to initial_states
 * \param[in] vf0 first state volume fraction
 * \param[in] state1 second state index to initial_states
 * \param[in] vf1 second state volume fraction
 */
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION HydroState<dim>
                       compute_mixed_state(InitialStates<dim, device_t> const & initial_states,
                                           int                                  state0,
                                           real_t                               vf0,
                                           int                                  state1,
                                           real_t                               vf1)
{
  return compute_mixed_state<dim, device_t>(
    initial_states(state0), vf0, initial_states(state1), vf1);
}

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_COMMON_H_
