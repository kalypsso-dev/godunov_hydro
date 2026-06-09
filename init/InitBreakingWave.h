// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitBreakingWave.h
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_INIT_BREAKINGWAVE_H_
#define KALYPSSO_GODUNOV_HYDRO_INIT_BREAKINGWAVE_H_

#include <godunov_hydro/common.h>

#include <kalypsso/core/problems/BreakingWaveParams.h>
#include <kalypsso/core/problems/init_cond_utils.h>

namespace kalypsso
{

namespace godunov_hydro
{

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Implement user data initialization for breaking wave test case.
 *
 * see reference:
 * A. W. Cook and W. H. Cabot. A high-wavenumber viscosity for high-resolution numerical
methods. In: Journal of Computational Physics 195.2 (2004), pp. 594–601.
* https://doi.org/10.1016/j.jcp.2003.10.012
*
* this test case has an analytical solution: the initial velocity profile is advected with velocity
* u-c; in other words, at time t, solution moved to x(t) = x(t=0) + (u-c)*t
* note that is valid only for time t <= ts where ts is the time where a shock is formed.
*
* As an analytical solution is know, this can be used to perform convergence studies.
*
* So this functor can evaluate solution either at t=0 (direct evaluation) or at t>0 by searching for
* x0 such that u(x0, t=0) = u(x,t)
*/
template <size_t dim, typename device_t>
class InitBreakingWaveDataFunctor
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

  //! BreakingWave problem specific parameters (used on device)
  BreakingWaveParams m_bwParams;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  //! time at which breaking wave analytical solution is evaluated
  //! usually it will be 0.0 when evaluating initial condition
  //! but it can also be tEnd, when evaluating analytical solution at the end of simulation
  //! to compare numerical and analytical solutions and assess numerical scheme convergence
  const real_t m_t_eval;

  InitBreakingWaveDataFunctor(DataArrayBlock_t const &             Udata,
                              orchard_key_view_t<device_t> const & orchard_keys,
                              int32_t                              local_num_octants,
                              HydroSettings const &                settings,
                              BreakingWaveParams const &           bwParams,
                              real_t                               t_eval,
                              ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_local_num_octants(local_num_octants)
    , m_settings(settings)
    , m_bwParams(bwParams)
    , m_scaling_factor(get_scaling_factor(config_map))
    , m_xyz_min(get_xyz_min<dim>(config_map))
    , m_t_eval(t_eval){};

public:
  // static method which does it all: create and execute functor
  static void
  apply(DataArrayBlock_t const &             Udata,
        orchard_key_view_t<device_t> const & orchard_keys,
        int32_t                              local_num_octants,
        HydroSettings const &                settings,
        real_t                               t_eval,
        ConfigMap const &                    config_map);

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int32_t & global_index) const;

}; // InitBreakingWaveDataFunctor

extern template class InitBreakingWaveDataFunctor<2, kalypsso::DefaultDevice>;
extern template class InitBreakingWaveDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Implement initial refinement for breaking wave testcase.
 *
 * \sa InitBreakingWaveDataFunctor
 *
 */
template <size_t dim, typename device_t>
class InitBreakingWaveRefineFunctor
{
public:
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

  //! type alias for a (device) Kokkos view of refinement flags
  using amrflags_view_t = typename AMRContext<dim, device_t>::amrflags_view_t;

  struct TagRefineAlways
  {};

private:
  //! refinement flags (to be filled)
  amrflags_view_t m_amrflags;

public:
  InitBreakingWaveRefineFunctor(amrflags_view_t amrflags)
    : m_amrflags(amrflags){};

  // static method which does it all: create and execute functor
  static void
  apply(amrflags_view_t const & amrflags, int32_t local_num_octants, ConfigMap const & config_map)
  {
    // iterate functor for refinement
    InitBreakingWaveRefineFunctor functor(amrflags);

    const auto refine_type = core::get_init_indicator(config_map);

    if (refine_type == +core::InitConditionsIndicator::ALWAYS_REFINE)
    {
      Kokkos::parallel_for("kalypsso::godunov_hydro::InitBreakingWaveRefineFunctor",
                           Kokkos::RangePolicy<exec_space, TagRefineAlways>(0, local_num_octants),
                           functor);
    }
    else
    {
      KALYPSSO_ERROR("Unknown value for refine indicator method.");
    }

  } // apply

  // ===========================================================
  // ===========================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagRefineAlways const &, const size_t & iOct) const
  {
    m_amrflags(iOct) = AMRContextBase::KALYPSSO_DO_REFINE;
  }

}; // InitBreakingWaveRefineFunctor

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Cook-Cabot breaking wave test case.
 *
 */
template <size_t dim, typename device_t>
class InitBreakingWave
{

public:
  static void
  apply(SolverGodunovHydro<dim, device_t> & solver);

}; // class InitBreakingWave

extern template class InitBreakingWave<2, kalypsso::DefaultDevice>;
extern template class InitBreakingWave<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_INIT_BREAKINGWAVE_H
