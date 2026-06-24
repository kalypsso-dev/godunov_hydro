// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitImplode.h
 *
 * Implosion test references:
 * - https://www.astro.princeton.edu/~jstone/Athena/tests/implode/Implode.html
 * - https://www.sciencedirect.com/science/article/pii/S0021999199962952
 * - http://www-troja.fjfi.cvut.cz/~liska/CompareEuler/compare8/
 * - https://www.sciencedirect.com/science/article/pii/S0045793021003364#b46
 *
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_INIT_IMPLODE_H_
#define KALYPSSO_GODUNOV_HYDRO_INIT_IMPLODE_H_

#include <godunov_hydro/common.h>
#include <kalypsso/core/problems/ImplodeParams.h>
#include <kalypsso/core/problems/init_cond_utils.h>

namespace kalypsso
{

namespace godunov_hydro
{

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Implement user data initialization to solve implosion test.
 */
template <size_t dim, typename device_t>
class InitImplodeDataFunctor
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

  //! Implode problem specific parameters (used on device)
  ImplodeParams m_implode_params;

  //! Equation of state wrapper
  EosWrapper<device_t> m_eos_wrapper;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  //! get domain upper right corner
  const Kokkos::Array<real_t, dim> m_xyz_max;

  InitImplodeDataFunctor(DataArrayBlock_t const &             Udata,
                         orchard_key_view_t<device_t> const & orchard_keys,
                         int32_t                              local_num_octants,
                         HydroSettings const &                settings,
                         ImplodeParams const &                implode_params,
                         ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_local_num_octants(local_num_octants)
    , m_settings(settings)
    , m_implode_params(implode_params)
    , m_eos_wrapper(config_map)
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

}; // InitImplodeDataFunctor

extern template class InitImplodeDataFunctor<2, kalypsso::DefaultDevice>;
extern template class InitImplodeDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Implement initial refinement to solve Implode problem.
 *
 * Use distance to interface as refine criterion.
 *
 * \sa InitImplodeDataFunctor
 *
 */
template <size_t dim, typename device_t>
class InitImplodeRefineFunctor
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

  //! Implode problem specific parameters (used on device)
  ImplodeParams m_implode_params;

  //! which level should we look at
  int m_level_refine;

  // get geometrical scaling factor
  const real_t m_scaling_factor;

  // get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  //! get domain upper right corner
  const Kokkos::Array<real_t, dim> m_xyz_max;

  InitImplodeRefineFunctor(DataArrayBlock_t const &             Udata,
                           orchard_key_view_t<device_t> const & orchard_keys,
                           amrflags_view_t const &              amrflags,
                           int32_t                              local_num_octants,
                           HydroSettings const &                settings,
                           ImplodeParams const &                implode_params,
                           int                                  level_refine,
                           ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_amrflags(amrflags)
    , m_local_num_octants(local_num_octants)
    , m_settings(settings)
    , m_implode_params(implode_params)
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
  KOKKOS_INLINE_FUNCTION void
  operator()(TagRefineAlways const &, const size_t & iOct) const;

  // ===========================================================
  // ===========================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagRefineGeometric const &, const size_t & iOct) const;

}; // InitImplodeRefineFunctor

extern template class InitImplodeRefineFunctor<2, kalypsso::DefaultDevice>;
extern template class InitImplodeRefineFunctor<3, kalypsso::DefaultDevice>;

// ============================================================
// ============================================================
/**
 * \class InitImplode
 *
 * Hydrodynamical implode shock tube init.
 *
 * \sa references
 * - https://www.astro.princeton.edu/~jstone/Athena/tests/implode/Implode.html
 * - https://www.sciencedirect.com/science/article/pii/S0021999199962952
 * - http://www-troja.fjfi.cvut.cz/~liska/CompareEuler/compare8/
 * - https://www.sciencedirect.com/science/article/pii/S0045793021003364#b46
 *
 * Initial conditions is refined near strong density gradients.
 */
template <size_t dim, typename device_t>
class InitImplode
{
public:
  static void
  apply(SolverGodunovHydro<dim, device_t> & solver);
};

extern template class InitImplode<2, kalypsso::DefaultDevice>;
extern template class InitImplode<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_INIT_IMPLODE_H_
