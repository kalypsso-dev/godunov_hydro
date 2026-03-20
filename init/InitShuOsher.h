// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitShuOsher.h
 *
 * A moving shock interacting with a density sine wave.
 *
 * Reference:
 * Efficient implementation of essentially non-oscillatory shock-capturing schemes, II,
 * C.-W. Shu and S. Osher, JCP vol. 83, pp 32-78 (1989).
 * https://doi.org/10.1016/0021-9991(89)90222-2
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_INIT_SHUOSHER_H_
#define KALYPSSO_GODUNOV_HYDRO_INIT_SHUOSHER_H_

#include <godunov_hydro/common.h>
#include <kalypsso/core/problems/init_cond_utils.h>
#include <kalypsso/core/problems/ShuOsherParams.h>

namespace kalypsso
{

namespace godunov_hydro
{

// ====================================================================
// ====================================================================
/**
 * Implement user data initialization to solve Shu-Osher problem.
 *
 */
template <size_t dim, typename device_t>
class InitShuOsherDataFunctor
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

  //! ShuOsher problem specific parameters (used on device)
  ShuOsherParams m_shParams;

  //! Equation of state wrapper
  eos::EosWrapper<device_t> m_eos_wrapper;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  InitShuOsherDataFunctor(DataArrayBlock_t const &             Udata,
                          FieldMap<core::models::Hydro>        fm,
                          orchard_key_view_t<device_t> const & orchard_keys,
                          int32_t                              local_num_octants,
                          HydroSettings const &                settings,
                          ShuOsherParams const &               shParams,
                          ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_fm(fm)
    , m_orchard_keys(orchard_keys)
    , m_local_num_octants(local_num_octants)
    , m_settings(settings)
    , m_shParams(shParams)
    , m_eos_wrapper(config_map)
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

}; // InitShuOsherDataFunctor

extern template class InitShuOsherDataFunctor<2, kalypsso::DefaultDevice>;
extern template class InitShuOsherDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Implement initial refinement to solve ShuOsher problem.
 *
 * Use distance to interface as refine criterion.
 *
 * \sa InitShuOsherDataFunctor
 *
 */
template <size_t dim, typename device_t>
class InitShuOsherRefineFunctor
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

  //! field manager
  FieldMap<core::models::Hydro> m_fm;

  //! list of orchard key of the mesh
  orchard_key_view_t<device_t> m_orchard_keys;

  //! refinement flags (to be filled)
  amrflags_view_t m_amrflags;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! general parameters (used on device)
  HydroSettings m_settings;

  //! ShuOsher problem specific parameters (used on device)
  ShuOsherParams m_shParams;

  //! which level should we look at
  int m_level_refine;

  // get geometrical scaling factor
  const real_t m_scaling_factor;

  // get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  InitShuOsherRefineFunctor(DataArrayBlock_t const &             Udata,
                            FieldMap<core::models::Hydro>        fm,
                            orchard_key_view_t<device_t> const & orchard_keys,
                            amrflags_view_t const &              amrflags,
                            int32_t                              local_num_octants,
                            HydroSettings const &                settings,
                            ShuOsherParams const &               shParams,
                            int                                  level_refine,
                            ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_fm(fm)
    , m_orchard_keys(orchard_keys)
    , m_amrflags(amrflags)
    , m_local_num_octants(local_num_octants)
    , m_settings(settings)
    , m_shParams(shParams)
    , m_level_refine(level_refine)
    , m_scaling_factor(get_scaling_factor(config_map))
    , m_xyz_min(get_xyz_min<dim>(config_map)){};

public:
  //! static method which does it all: create and execute functor
  static void
  apply(DataArrayBlock_t const &             Udata,
        FieldMap<core::models::Hydro>        fm,
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

}; // InitShuOsherRefineFunctor

extern template class InitShuOsherRefineFunctor<2, kalypsso::DefaultDevice>;
extern template class InitShuOsherRefineFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
/**
 * Shu-Osher problem initialization.
 */
template <size_t dim, typename device_t>
class InitShuOsher
{
public:
  static void
  apply(SolverGodunovHydro<dim, device_t> & solver);
}; // class InitShuOsher

extern template class InitShuOsher<2, kalypsso::DefaultDevice>;
extern template class InitShuOsher<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_INIT_SHUOSHER_H_
