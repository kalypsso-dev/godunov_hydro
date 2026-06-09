// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitTriplePoint.h
 *
 * Triple point problem.
 *
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_INIT_TRIPLEPOINT_H_
#define KALYPSSO_GODUNOV_HYDRO_INIT_TRIPLEPOINT_H_

#include <godunov_hydro/common.h>
#include <kalypsso/core/problems/init_cond_utils.h>
#include <kalypsso/core/problems/TriplePointParams.h>

namespace kalypsso
{

namespace godunov_hydro
{

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Implement user data initialization to solve triple point problem.
 *
 * Initial conditions is refined near strong density gradients.
 */
template <size_t dim, typename device_t>
class InitTriplePointDataFunctor
{

public:
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

private:
  //! conservative variables to be initialized
  DataArrayBlock_t m_Udata;

  //! list of orchard key of the mesh
  orchard_key_view_t<device_t> m_orchard_keys;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! general parameters (used on device)
  HydroSettings m_settings;

  //! TriplePoint problem specific parameters (used on device)
  TriplePointParams m_triple_point_params;

  //! Equation of state wrapper
  eos::EosWrapper<device_t> m_eos_wrapper;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  InitTriplePointDataFunctor(DataArrayBlock_t const &             Udata,
                             orchard_key_view_t<device_t> const & orchard_keys,
                             int32_t                              local_num_octants,
                             HydroSettings const &                settings,
                             ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_local_num_octants(local_num_octants)
    , m_settings(settings)
    , m_triple_point_params(config_map)
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
  KOKKOS_INLINE_FUNCTION void
  operator()(const int32_t & global_index) const;

}; // InitTriplePointDataFunctor

extern template class InitTriplePointDataFunctor<2, kalypsso::DefaultDevice>;
extern template class InitTriplePointDataFunctor<3, kalypsso::DefaultDevice>;

// ==================================================================================
// ==================================================================================
// ==================================================================================
/**
 * Implement initial refinement to solve TriplePoint problem.
 *
 * Use distance to interface as refine criterion.
 *
 * \sa InitTriplePointDataFunctor
 *
 */
template <size_t dim, typename device_t>
class InitTriplePointRefineFunctor
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
  //! conservative variables
  DataArrayBlock_t m_Udata;

  //! list of orchard key of the mesh
  orchard_key_view_t<device_t> m_orchard_keys;

  //! refinement flags (to be filled)
  amrflags_view_t m_amrflags;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! general parameters (used on device)
  HydroSettings m_settings;

  //! TriplePoint problem specific parameters (used on device)
  TriplePointParams m_triple_point_params;

  //! which level should we look at
  int m_level_refine;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  InitTriplePointRefineFunctor(DataArrayBlock_t const &             Udata,
                               orchard_key_view_t<device_t> const & orchard_keys,
                               amrflags_view_t const &              amrflags,
                               int32_t                              local_num_octants,
                               HydroSettings const &                settings,
                               int                                  level_refine,
                               ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_amrflags(amrflags)
    , m_local_num_octants(local_num_octants)
    , m_settings(settings)
    , m_triple_point_params(config_map)
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

}; // InitTriplePointRefineFunctor

extern template class InitTriplePointRefineFunctor<2, kalypsso::DefaultDevice>;
extern template class InitTriplePointRefineFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
/**
 * Hydrodynamical triple point problem initialization.
 */
template <size_t dim, typename device_t>
class InitTriplePoint
{
public:
  static void
  apply(SolverGodunovHydro<dim, device_t> & solver);
};

extern template class InitTriplePoint<2, kalypsso::DefaultDevice>;
extern template class InitTriplePoint<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_INIT_TRIPLEPOINT_H_
