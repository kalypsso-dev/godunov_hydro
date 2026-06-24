// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitShockBubble.h
 *
 * Shock-bubble interaction.
 *
 * Reference: http://amroc.sourceforge.net/examples/euler/2d/html/shbubble_n.htm
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_INIT_SHOCK_BUBBLE_H_
#define KALYPSSO_GODUNOV_HYDRO_INIT_SHOCK_BUBBLE_H_

#include <godunov_hydro/common.h>
#include <kalypsso/core/problems/init_cond_utils.h>
#include <kalypsso/core/problems/ShockBubbleParams.h>

namespace kalypsso
{

namespace godunov_hydro
{

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Implement user data initialization to solve a shock-bubble interaction problem.
 *
 * \sa references
 * - http://amroc.sourceforge.net/examples/euler/2d/html/shbubble_n.htm
 *
 * Initial conditions is refined near strong density gradients.
 */
template <size_t dim, typename device_t>
class InitShockBubbleDataFunctor
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

  //! ShockBubble problem specific parameters (used on device)
  ShockBubbleParams<device_t> m_sb_params;

  //! Initial states (one per region, conservative variables)
  InitialStates<dim, device_t> m_initial_states;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  InitShockBubbleDataFunctor(DataArrayBlock_t const &             Udata,
                             orchard_key_view_t<device_t> const & orchard_keys,
                             int32_t                              local_num_octants,
                             InitialStates<dim, device_t> const & initial_states,
                             ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_local_num_octants(local_num_octants)
    , m_sb_params(config_map)
    , m_initial_states(initial_states)
    , m_scaling_factor(get_scaling_factor(config_map))
    , m_xyz_min(get_xyz_min<dim>(config_map)){};

public:
  //! static method which does it all: create and execute functor
  static void
  apply(DataArrayBlock_t const &             Udata,
        orchard_key_view_t<device_t> const & orchard_keys,
        int32_t                              local_num_octants,
        InitialStates<dim, device_t> const & initial_states,
        ConfigMap const &                    config_map);

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const int32_t & global_index) const;

}; // class InitShockBubbleDataFunctor

extern template class InitShockBubbleDataFunctor<2, kalypsso::DefaultDevice>;
extern template class InitShockBubbleDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Implement initial refinement to solve ShockBubble problem.
 *
 * Use distance to interface as refine criterion.
 *
 * \sa InitShockBubbleDataFunctor
 *
 */
template <size_t dim, typename device_t>
class InitShockBubbleRefineFunctor
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
  //! heavy hydrodynamics data
  DataArrayBlock_t m_Udata;

  //! list of orchard key of the mesh
  orchard_key_view_t<device_t> m_orchard_keys;

  //! refinement flags (to be filled)
  amrflags_view_t m_amrflags;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! ShockBubble problem specific parameters (used on device)
  ShockBubbleParams<device_t> m_sb_params;

  //! which level should we look at
  int m_level_refine;

  // get geometrical scaling factor
  const real_t m_scaling_factor;

  // get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

public:
  InitShockBubbleRefineFunctor(DataArrayBlock_t             Udata,
                               orchard_key_view_t<device_t> orchard_keys,
                               amrflags_view_t              amrflags,
                               int32_t                      local_num_octants,
                               int                          level_refine,
                               ConfigMap const &            config_map)
    : m_Udata(Udata)
    , m_orchard_keys(orchard_keys)
    , m_amrflags(amrflags)
    , m_local_num_octants(local_num_octants)
    , m_sb_params(config_map)
    , m_level_refine(level_refine)
    , m_scaling_factor(get_scaling_factor(config_map))
    , m_xyz_min(get_xyz_min<dim>(config_map)){};

  // static method which does it all: create and execute functor
  static void
  apply(DataArrayBlock_t const &             Udata,
        orchard_key_view_t<device_t> const & orchard_keys,
        amrflags_view_t const &              amrflags,
        int32_t                              local_num_octants,
        int                                  level_refine,
        ConfigMap const &                    config_map);

  // ===========================================================
  // ===========================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagRefineAlways const &, const size_t & iOct) const;

}; // class InitShockBubbleRefineFunctor

extern template class InitShockBubbleRefineFunctor<2, kalypsso::DefaultDevice>;
extern template class InitShockBubbleRefineFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
/**
 * \class InitShockBubble
 *
 * Shock-bubble init.
 */
template <size_t dim, typename device_t>
class InitShockBubble
{
public:
  static void
  apply(SolverGodunovHydro<dim, device_t> & solver);

}; // class InitShockBubble

extern template class InitShockBubble<2, kalypsso::DefaultDevice>;
extern template class InitShockBubble<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_INIT_SHOCK_BUBBLE_H_
