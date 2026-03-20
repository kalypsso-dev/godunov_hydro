// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitIsentropicVortex.h
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_INIT_ISENTROPIC_VORTEX_H_
#define KALYPSSO_GODUNOV_HYDRO_INIT_ISENTROPIC_VORTEX_H_

#include <godunov_hydro/common.h>
#include <kalypsso/core/problems/IsentropicVortexParams.h>
#include <kalypsso/core/problems/init_cond_utils.h>

namespace kalypsso
{

namespace godunov_hydro
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement user data initialization to solve isentropic vortex problem.
 */
template <size_t dim, typename device_t>
class InitIsentropicVortexDataFunctor
{

public:
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

private:
  //! heavy data
  DataArrayBlock_t m_Udata;

  //! field manager
  FieldMap<core::models::Hydro> m_fm;

  //! list of orchard key of the mesh
  orchard_key_view_t<device_t> m_orchard_keys;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! general parameters (used on device)
  HydroSettings m_settings;

  //! IsentropicVortex problem specific parameters (used on device)
  IsentropicVortexParams m_iParams;

  //! specific heat ratio (equation of state is ideal gas)
  real_t m_gamma;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  InitIsentropicVortexDataFunctor(DataArrayBlock_t const &             Udata,
                                  FieldMap<core::models::Hydro>        fm,
                                  orchard_key_view_t<device_t> const & orchard_keys,
                                  int32_t                              local_num_octants,
                                  HydroSettings const &                settings,
                                  IsentropicVortexParams const &       iParams,
                                  ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_fm(fm)
    , m_orchard_keys(orchard_keys)
    , m_local_num_octants(local_num_octants)
    , m_settings(settings)
    , m_iParams(iParams)
    , m_gamma(config_map.getReal("material0", "gamma", KALYPSSO_NUM(1.4)))
    , m_scaling_factor(get_scaling_factor(config_map))
    , m_xyz_min(get_xyz_min<dim>(config_map)){};

public:
  //! static method which does it all: create and execute functor
  static void
  apply(DataArrayBlock_t const &             Udata,
        FieldMap<core::models::Hydro>        fm,
        orchard_key_view_t<device_t> const & orchard_keys,
        int32_t                              local_num_octants,
        HydroSettings const &                settings,
        ConfigMap const &                    config_map);

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int32_t & global_index) const;

}; // InitIsentropicVortexDataFunctor

extern template class InitIsentropicVortexDataFunctor<2, kalypsso::DefaultDevice>;
extern template class InitIsentropicVortexDataFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
/**
 * \class InitIsentropicVortex
 *
 * This is a stationary solution of Euler equations, and as such useful for testing
 * how the numerical scheme is able to maintain this solution, and testing space convergence.
 *
 * References:
 * - https://www.cfd-online.com/Wiki/2-D_vortex_in_isentropic_flow
 * - https://hal.archives-ouvertes.fr/hal-01485587/document
 */
template <size_t dim, typename device_t>
class InitIsentropicVortex
{
public:
  static void
  apply(SolverGodunovHydro<dim, device_t> & solver);
}; // class InitIsentropicVortex

extern template class InitIsentropicVortex<2, kalypsso::DefaultDevice>;
extern template class InitIsentropicVortex<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_INIT_ISENTROPIC_VORTEX_H_
