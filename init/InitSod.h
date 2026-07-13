// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitSod.h
 *
 * - https://github.com/ibackus/sod-shocktube
 * - https://github.com/pmocz/riemann-solver
 * - https://en.wikipedia.org/wiki/Sod_shock_tube
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_INIT_SOD_H_
#define KALYPSSO_GODUNOV_HYDRO_INIT_SOD_H_

#include <godunov_hydro/common.h>
#include <kalypsso/core/problems/init_cond_utils.h>
#include <kalypsso/core/problems/SodParams.h>

namespace kalypsso
{

namespace godunov_hydro
{

// ====================================================================
// ====================================================================
/**
 * Implement user data initialization to solve sod shock tube problem.
 *
 * \sa references
 * - https://en.wikipedia.org/wiki/Sod_shock_tube
 * - exact solution python script :
 *   https://github.com/pkestene/sod-shocktube/blob/test_ppkMHD_sdm/exactRiemann.py
 *
 * Initial conditions is refined near strong density gradients.
 */
template <size_t dim, typename device_t>
class InitSodDataFunctor
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

  //! Sod problem specific parameters (used on device)
  SodParams m_sodParams;

  //! Equation of state wrapper
  EosWrapper<device_t> m_eos_wrapper;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  InitSodDataFunctor(DataArrayBlock_t const &             Udata,
                     orchard_key_view_t<device_t> const & orchard_keys,
                     int32_t                              local_num_octants,
                     HydroSettings const &                settings,
                     SodParams const &                    sodParams,
                     ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_local_num_octants(local_num_octants)
    , m_settings(settings)
    , m_sodParams(sodParams)
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

}; // InitSodDataFunctor

extern template class InitSodDataFunctor<2, kalypsso::DefaultDevice>;
extern template class InitSodDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Implement initial refinement to solve Sod problem.
 *
 * Use distance to interface as refine criterion.
 *
 * \sa InitSodDataFunctor
 *
 */
template <size_t dim, typename device_t>
class InitSodRefineFunctor
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

  //! Sod problem specific parameters (used on device)
  SodParams m_sodParams;

  //! which level should we look at
  int m_level_refine;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  InitSodRefineFunctor(DataArrayBlock_t const &             Udata,
                       orchard_key_view_t<device_t> const & orchard_keys,
                       amrflags_view_t const &              amrflags,
                       int32_t                              local_num_octants,
                       HydroSettings const &                settings,
                       SodParams const &                    sodParams,
                       int                                  level_refine,
                       ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_amrflags(amrflags)
    , m_local_num_octants(local_num_octants)
    , m_settings(settings)
    , m_sodParams(sodParams)
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

}; // InitSodRefineFunctor

extern template class InitSodRefineFunctor<2, kalypsso::DefaultDevice>;
extern template class InitSodRefineFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
/**
 * Hydrodynamical sod shock tube init.
 */
template <size_t dim, typename device_t>
class InitSod
{
public:
  static void
  apply(SolverGodunovHydro<dim, device_t> & solver);
}; // class InitSod

extern template class InitSod<2, kalypsso::DefaultDevice>;
extern template class InitSod<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_INIT_SOD_H_
