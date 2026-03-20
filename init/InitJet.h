// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitJet.h
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_INIT_JET_H_
#define KALYPSSO_GODUNOV_HYDRO_INIT_JET_H_

#include <godunov_hydro/common.h>
#include <kalypsso/core/problems/init_cond_utils.h>

namespace kalypsso
{

namespace godunov_hydro
{

// =======================================================
// =======================================================
// =======================================================
/**
 * Implement user data initialization to solve jet testcase.
 *
 * Init: fluid at rest in a mesh uniformly refined at level min.
 */
template <size_t dim, typename device_t>
class InitJetDataFunctor
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

  //! Jet hydro state (conservative variables)
  HydroState<dim> m_U_jet;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  InitJetDataFunctor(DataArrayBlock_t const &             Udata,
                     FieldMap<core::models::Hydro>        fm,
                     orchard_key_view_t<device_t> const & orchard_keys,
                     int32_t                              local_num_octants,
                     HydroSettings const &                settings,
                     HydroState<dim> const &              U_jet,
                     ConfigMap const &                    config_map)
    : m_Udata(Udata)
    , m_fm(fm)
    , m_orchard_keys(orchard_keys)
    , m_local_num_octants(local_num_octants)
    , m_settings(settings)
    , m_U_jet(U_jet)
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

}; // InitJetDataFunctor

extern template class InitJetDataFunctor<2, kalypsso::DefaultDevice>;
extern template class InitJetDataFunctor<3, kalypsso::DefaultDevice>;

// =======================================================
// =======================================================
// =======================================================
/**
 * Hydrodynamical jet shock tube init.
 */
template <size_t dim, typename device_t>
class InitJet
{
public:
  static void
  apply(SolverGodunovHydro<dim, device_t> & solver);
}; // class InitJet

extern template class InitJet<2, kalypsso::DefaultDevice>;
extern template class InitJet<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_INIT_JET_H_
