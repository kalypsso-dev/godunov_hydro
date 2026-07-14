// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitWoodward.h
 *
 * - Woodward, P. and Colella, P., "The Numerical Simulation of Two-Dimensional
 * Fluid Flow with Strong Shocks", J. Computational Physics, 54, 115-173 (1984).
 * https://doi.org/10.1016/0021-9991(84)90142-6
 * - https://www.astro.princeton.edu/~jstone/Athena/tests/twoibw/TwoIBW.html
 * - https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node191.html
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_INIT_WOODWARD_H_
#define KALYPSSO_GODUNOV_HYDRO_INIT_WOODWARD_H_

#include <godunov_hydro/common.h>
#include <kalypsso/core/problems/init_cond_utils.h>
#include <kalypsso/core/problems/WoodwardParams.h>

namespace kalypsso
{

namespace godunov_hydro
{

// ====================================================================
// ====================================================================
/**
 * Implement user data initialization to solve Woodward problem: two interact blast waves.
 *
 * \sa references
 *
 * - Woodward, P. and Colella, P., "The Numerical Simulation of Two-Dimensional
 * Fluid Flow with Strong Shocks", J. Computational Physics, 54, 115-173 (1984).
 * https://doi.org/10.1016/0021-9991(84)90142-6
 * - https://www.astro.princeton.edu/~jstone/Athena/tests/twoibw/TwoIBW.html
 * - https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node191.html
 *
 * Initial conditions is refined near strong density gradients.
 */
template <size_t dim, typename device_t>
class InitWoodwardDataFunctor
{

public:
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

private:
  //! heavy data
  DataArrayBlock_t m_Udata;

  //! list of orchard key of the mesh
  orchard_key_view_t<device_t> m_orchard_keys;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! general parameters (used on device)
  HydroSettings m_settings;

  //! Woodward problem specific parameters (used on device)
  WoodwardParams m_woodwardParams;

  //! Equation of state wrapper
  EosWrapper<device_t> m_eos_wrapper;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  InitWoodwardDataFunctor(DataArrayBlock_t const &             Udata,
                          orchard_key_view_t<device_t> const & orchard_keys,
                          int32_t                              local_num_octants,
                          HydroSettings const &                settings,
                          WoodwardParams const &               woodwardParams,
                          ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_local_num_octants(local_num_octants)
    , m_settings(settings)
    , m_woodwardParams(woodwardParams)
    , m_eos_wrapper(config_map)
    , m_scaling_factor(get_scaling_factor(config_map))
    , m_xyz_min(get_xyz_min<dim>(config_map)){};

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

}; // InitWoodwardDataFunctor

extern template class InitWoodwardDataFunctor<2, kalypsso::DefaultDevice>;
extern template class InitWoodwardDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Implement initial refinement to solve Woodward problem.
 *
 * Use distance to interface as refine criterion.
 *
 * \sa InitWoodwardDataFunctor
 *
 */
template <size_t dim, typename device_t>
class InitWoodwardRefineFunctor
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

  //! Woodward problem specific parameters (used on device)
  WoodwardParams m_woodwardParams;

  //! which level should we look at
  int m_level_refine;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  InitWoodwardRefineFunctor(DataArrayBlock_t const &             Udata,
                            orchard_key_view_t<device_t> const & orchard_keys,
                            amrflags_view_t const &              amrflags,
                            int32_t                              local_num_octants,
                            HydroSettings const &                settings,
                            WoodwardParams const &               woodwardParams,
                            int                                  level_refine,
                            ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_amrflags(amrflags)
    , m_local_num_octants(local_num_octants)
    , m_settings(settings)
    , m_woodwardParams(woodwardParams)
    , m_level_refine(level_refine)
    , m_scaling_factor(get_scaling_factor(config_map))
    , m_xyz_min(get_xyz_min<dim>(config_map)){};

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

}; // InitWoodwardRefineFunctor

extern template class InitWoodwardRefineFunctor<2, kalypsso::DefaultDevice>;
extern template class InitWoodwardRefineFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
/**
 * Hydrodynamical Woodward shock tube init.
 */
template <size_t dim, typename device_t>
class InitWoodward
{
public:
  static void
  apply(SolverGodunovHydro<dim, device_t> & solver);
}; // class InitWoodward

extern template class InitWoodward<2, kalypsso::DefaultDevice>;
extern template class InitWoodward<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_INIT_WOODWARD_H_
