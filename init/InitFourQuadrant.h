// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitFourQuadrant.h
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_INIT_FOUR_QUANDRANT_H_
#define KALYPSSO_GODUNOV_HYDRO_INIT_FOUR_QUANDRANT_H_

#include <godunov_hydro/common.h>

#include <kalypsso/core/problems/FourQuadrantParams.h>
#include <kalypsso/core/problems/initRiemannConfig2d.h>
#include <kalypsso/core/models/HydroState.h>

#include <kalypsso/core/problems/init_cond_utils.h>

namespace kalypsso
{

namespace godunov_hydro
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement initialization functor to solve a four quadrant 2D Riemann
 * problem.
 *
 * This functor takes as input a mesh, already refined, and initializes
 * user data.
 *
 * In the 2D case, there are 19 different possible configurations (see
 * article by Lax and Liu, "Solution of two-dimensional riemann
 * problems of gas dynamics by positive schemes",SIAM journal on
 * scientific computing, 1998, vol. 19, no2, pp. 319-340).
 *
 * Initial conditions is refined near interface separating the 4
 * four quadrants.
 *
 */
template <size_t dim, typename device_t>
class InitFourQuadrantDataFunctor
{

public:
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  using exec_space = typename device_t::execution_space;

  using HydroState_t = typename RiemannConfig<dim>::HydroState_t;
  using HydroStates_t = typename RiemannConfig<dim>::HydroStates_t;

private:
  //! conservative variables to be initialized
  DataArrayBlock_t m_Udata;

  //! list of orchard key of the mesh
  orchard_key_view_t<device_t> m_orchard_keys;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! general parameters (used on device)
  HydroSettings m_settings;

  //! Four quadrant states
  HydroStates_t m_Us;

  //! discontinuity location
  Kokkos::Array<real_t, dim> m_pos;

  // get geometrical scaling factor
  const real_t m_scaling_factor;

  // get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  InitFourQuadrantDataFunctor(DataArrayBlock_t const &             Udata,
                              orchard_key_view_t<device_t> const & orchard_keys,
                              int32_t                              local_num_octants,
                              HydroSettings const &                settings,
                              HydroStates_t const &                Us,
                              Kokkos::Array<real_t, dim> const &   pos,
                              ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_local_num_octants(local_num_octants)
    , m_settings(settings)
    , m_Us(Us)
    , m_pos(pos)
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

}; // InitFourQuadrantDataFunctor

extern template class InitFourQuadrantDataFunctor<2, kalypsso::DefaultDevice>;
extern template class InitFourQuadrantDataFunctor<3, kalypsso::DefaultDevice>;

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement initialization functor to solve a four quadrant 2D Riemann
 * problem.
 *
 * This functor only performs mesh refinement, no user data init.
 *
 * In the 2D case, there are 19 different possible configurations (see
 * article by Lax and Liu, "Solution of two-dimensional riemann
 * problems of gas dynamics by positive schemes",SIAM journal on
 * scientific computing, 1998, vol. 19, no2, pp. 319-340).
 *
 * Initial conditions is refined near interface separating the 4
 * four quadrants.
 *
 * \sa InitFourQuadrantDataFunctor
 *
 */
template <size_t dim, typename device_t>
class InitFourQuadrantRefineFunctor
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

  //! which level should we look at
  int m_level_refine;

  //! discontinuity location
  real_t m_xt, m_yt, m_zt;

  // get geometrical scaling factor
  const real_t m_scaling_factor;

  // get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  InitFourQuadrantRefineFunctor(DataArrayBlock_t const &             Udata,
                                orchard_key_view_t<device_t> const & orchard_keys,
                                amrflags_view_t const &              amrflags,
                                int32_t                              local_num_octants,
                                HydroSettings const &                settings,
                                int                                  level_refine,
                                real_t                               xt,
                                real_t                               yt,
                                real_t                               zt,
                                ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_amrflags(amrflags)
    , m_local_num_octants(local_num_octants)
    , m_settings(settings)
    , m_level_refine(level_refine)
    , m_xt(xt)
    , m_yt(yt)
    , m_zt(zt)
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

}; // InitFourQuadrantRefineFunctor

extern template class InitFourQuadrantRefineFunctor<2, kalypsso::DefaultDevice>;
extern template class InitFourQuadrantRefineFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
/**
 * \class InitFourQuadrant
 * \brief Implement initialization to solve a four quadrant 2D Riemann
 * problem.
 *
 * In the 2D case, there are 19 different possible configurations (see
 * article by Lax and Liu, "Solution of two-dimensional riemann
 * problems of gas dynamics by positive schemes",SIAM journal on
 * scientific computing, 1998, vol. 19, no2, pp. 319-340).
 *
 * Initial conditions is refined near interface separating the 4
 * four quadrants.
 */
template <size_t dim, typename device_t>
class InitFourQuadrant
{
public:
  static void
  apply(SolverGodunovHydro<dim, device_t> & solver);
}; // class InitFourQuadrant

extern template class InitFourQuadrant<2, kalypsso::DefaultDevice>;
extern template class InitFourQuadrant<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_INIT_FOUR_QUANDRANT_H_
