// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitKelvinHelmholtz.h
 *
 * Reference:
 * - http://www.astro.princeton.edu/~jstone/Athena/tests/kh/kh.html
 * - article by Robertson et al:
 *   "Computational Eulerian hydrodynamics and Galilean invariance",
 *   B.E. Robertson et al, Mon. Not. R. Astron. Soc., 401, 2463-2476, (2010).
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_INIT_KELVIN_HELMHOLTZ_H_
#define KALYPSSO_GODUNOV_HYDRO_INIT_KELVIN_HELMHOLTZ_H_

#include <godunov_hydro/common.h>
#include <kalypsso/core/problems/init_cond_utils.h>
#include <kalypsso/core/problems/KHParams.h>

#include <Kokkos_Random.hpp> // for random number drawing on device

namespace kalypsso
{

namespace godunov_hydro
{

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Implement user data initialization to solve Kelvin-Helmholtz problem.
 *
 * This functor takes as input a mesh, already refined and initializes user data
 * on device.
 *
 * Initial conditions is refined near strong density gradients.
 *
 * Reference:
 * - http://www.astro.princeton.edu/~jstone/Athena/tests/kh/kh.html
 * - article by Robertson et al:
 *   "Computational Eulerian hydrodynamics and Galilean invariance",
 *   B.E. Robertson et al, Mon. Not. R. Astron. Soc., 401, 2463-2476, (2010).

 */
template <size_t dim, typename device_t>
class InitKelvinHelmholtzDataFunctor
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

  //! KelvinHelmholtz problem specific parameters (used on device)
  KHParams m_khParams;

  //! Equation of state wrapper
  EosWrapper<device_t> m_eos_wrapper;

  //! random number generator pool
  RGPool_t m_rand_pool;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  //! get domain upper right corner
  const Kokkos::Array<real_t, dim> m_xyz_max;

  InitKelvinHelmholtzDataFunctor(DataArrayBlock_t const &             Udata,
                                 orchard_key_view_t<device_t> const & orchard_keys,
                                 int32_t                              local_num_octants,
                                 HydroSettings const &                settings,
                                 KHParams const &                     khParams,
                                 ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_local_num_octants(local_num_octants)
    , m_settings(settings)
    , m_khParams(khParams)
    , m_eos_wrapper(config_map)
    , m_rand_pool(khParams.seed)
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

}; // InitKelvinHelmholtzDataFunctor

extern template class InitKelvinHelmholtzDataFunctor<2, kalypsso::DefaultDevice>;
extern template class InitKelvinHelmholtzDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Implement initial refinement to solve Kelvin-Helmholtz problem.
 *
 * Use distance to interface as refine criterion.
 *
 * \sa InitKelvinHelmholtzDataFunctor
 *
 */
template <size_t dim, typename device_t>
class InitKelvinHelmholtzRefineFunctor
{
public:
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

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

  //! KelvinHelmholtz problem specific parameters (used on device)
  KHParams m_khParams;

  //! which level should we look at
  int m_level_refine;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  //! get domain upper right corner
  const Kokkos::Array<real_t, dim> m_xyz_max;

  InitKelvinHelmholtzRefineFunctor(DataArrayBlock_t const &             Udata,
                                   orchard_key_view_t<device_t> const & orchard_keys,
                                   amrflags_view_t const &              amrflags,
                                   int32_t                              local_num_octants,
                                   HydroSettings const &                settings,
                                   KHParams const &                     khParams,
                                   int                                  level_refine,
                                   ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_amrflags(amrflags)
    , m_local_num_octants(local_num_octants)
    , m_settings(settings)
    , m_khParams(khParams)
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

}; // InitKelvinHelmholtzRefineFunctor

extern template class InitKelvinHelmholtzRefineFunctor<2, kalypsso::DefaultDevice>;
extern template class InitKelvinHelmholtzRefineFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
/**
 *
 * Initial condition is mostly done on device.
 *
 */
template <size_t dim, typename device_t>
class InitKelvinHelmholtz
{
public:
  static void
  apply(SolverGodunovHydro<dim, device_t> & solver);
}; // class InitKelvinHelmholtz

extern template class InitKelvinHelmholtz<2, kalypsso::DefaultDevice>;
extern template class InitKelvinHelmholtz<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_INIT_KELVIN_HELMHOLTZ_H_
