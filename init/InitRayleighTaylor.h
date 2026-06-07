// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitRayleighTaylor.h
 *
 * Reference:
 * - http://www.astro.princeton.edu/~jstone/Athena/tests/rt/rt.html
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_INIT_RAYLEIGH_TAYLOR_H_
#define KALYPSSO_GODUNOV_HYDRO_INIT_RAYLEIGH_TAYLOR_H_

#include <godunov_hydro/common.h>
#include <kalypsso/core/problems/init_cond_utils.h>
#include <kalypsso/core/problems/RayleighTaylorParams.h>

#include <Kokkos_Random.hpp>

namespace kalypsso
{

namespace godunov_hydro
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement user data initialization to solve Rayleigh-Taylor problem.
 *
 * This functor takes as input a mesh, already refined and initializes user data
 * on device.
 *
 * Initial conditions is refined near strong density gradients.
 *
 * Reference:
 * - http://www.astro.princeton.edu/~jstone/Athena/tests/rt/rt.html
 */
template <size_t dim, typename device_t>
class InitRayleighTaylorDataFunctor
{

public:
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

  using RGPool_t = typename Kokkos::Random_XorShift64_Pool<exec_space>;
  using rng_state_t = typename RGPool_t::generator_type;

private:
  //! heavy data
  DataArrayBlock_t m_Udata;

  //! list of orchard key of the mesh
  orchard_key_view_t<device_t> m_orchard_keys;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! general parameters (used on device)
  HydroSettings m_settings;

  //! RayleighTaylor problem specific parameters (used on device)
  RayleighTaylorParams m_rt_params;

  //! Equation of state wrapper
  eos::EosWrapper<device_t> m_eos_wrapper;

  //! gravity field
  Kokkos::Array<real_t, dim> m_grav;

  // random number generator pool
  RGPool_t m_rand_pool;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  //! get domain upper right corner
  const Kokkos::Array<real_t, dim> m_xyz_max;

  InitRayleighTaylorDataFunctor(DataArrayBlock_t const &             Udata,
                                orchard_key_view_t<device_t> const & orchard_keys,
                                int32_t                              local_num_octants,
                                HydroSettings const &                settings,
                                RayleighTaylorParams const &         rt_params,
                                Kokkos::Array<real_t, dim>           gravity_field,
                                ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_local_num_octants(local_num_octants)
    , m_settings(settings)
    , m_rt_params(rt_params)
    , m_eos_wrapper(config_map)
    , m_grav(gravity_field)
    , m_rand_pool(rt_params.seed)
    , m_scaling_factor(get_scaling_factor(config_map))
    , m_xyz_min(get_xyz_min<dim>(config_map))
    , m_xyz_max(get_xyz_max<dim>(config_map)){};

public:
  //! static method which does it all: create and execute functor
  static void
  apply(DataArrayBlock_t const &             Udata,
        orchard_key_view_t<device_t> const & orchard_keys,
        int32_t                              local_num_octants,
        HydroSettings const &                settings,
        ConfigMap const &                    config_map);

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int32_t & global_index) const;

}; // InitRayleighTaylorDataFunctor

extern template class InitRayleighTaylorDataFunctor<2, kalypsso::DefaultDevice>;
extern template class InitRayleighTaylorDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Implement initial refinement to solve Rayleigh-Taylor problem.
 *
 * Use distance to interface as refine criterion.
 *
 * \sa InitRayleighTaylorDataFunctor
 *
 */
template <size_t dim, typename device_t>
class InitRayleighTaylorRefineFunctor
{
public:
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using DataArrayBlockHost_t = DataArrayBlock<dim, real_t, HostDevice>;

  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

  //! type alias for a (device) Kokkos view of refinement flags
  using amrflags_view_t = typename AMRContext<dim, device_t>::amrflags_view_t;

  struct TagRefineAlways
  {};
  struct TagRefineGeometric
  {};

private:
  //! heavy hydrodynamics data
  DataArrayBlock_t m_Udata;

  //! list of orchard key of the mesh
  orchard_key_view_t<device_t> m_orchard_keys;

  //! refinement flags (to be filled)
  amrflags_view_t m_amrflags;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! general parameters (used on device)
  HydroSettings m_settings;

  //! RayleighTaylor problem specific parameters (used on device)
  RayleighTaylorParams m_rt_params;

  //! which level should we look at
  int m_level_refine;

  // get geometrical scaling factor
  const real_t m_scaling_factor;

  // get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  //! get domain upper right corner
  const Kokkos::Array<real_t, dim> m_xyz_max;

  InitRayleighTaylorRefineFunctor(DataArrayBlock_t const &             Udata,
                                  orchard_key_view_t<device_t> const & orchard_keys,
                                  amrflags_view_t const &              amrflags,
                                  int32_t                              local_num_octants,
                                  HydroSettings const &                settings,
                                  RayleighTaylorParams const &         rt_params,
                                  int                                  level_refine,
                                  ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_amrflags(amrflags)
    , m_local_num_octants(local_num_octants)
    , m_settings(settings)
    , m_rt_params(rt_params)
    , m_level_refine(level_refine)
    , m_scaling_factor(get_scaling_factor(config_map))
    , m_xyz_min(get_xyz_min<dim>(config_map))
    , m_xyz_max(get_xyz_max<dim>(config_map)){};

public:
  //! static method which does it all: create and execute functor
  static void
  apply(DataArrayBlock_t const &             Udata,
        orchard_key_view_t<device_t> const & orchard_keys,
        amrflags_view_t const &              amrflags,
        int32_t                              local_num_octants,
        HydroSettings const &                settings,
        int                                  level_refine,
        ConfigMap const &                    config_map);

  // ===========================================================
  // ===========================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagRefineAlways const &, const size_t & iOct) const;

  // ===========================================================
  // ===========================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagRefineGeometric const &, const size_t & iOct) const;

}; // InitRayleighTaylorRefineFunctor

extern template class InitRayleighTaylorRefineFunctor<2, kalypsso::DefaultDevice>;
extern template class InitRayleighTaylorRefineFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
/**
 *
 * Initial condition is mostly done on device.
 *
 */
template <size_t dim, typename device_t>
class InitRayleighTaylor
{
public:
  static void
  apply(SolverGodunovHydro<dim, device_t> & solver);
}; // class InitRayleighTaylor

extern template class InitRayleighTaylor<2, kalypsso::DefaultDevice>;
extern template class InitRayleighTaylor<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_INIT_RAYLEIGH_TAYLOR_H_
