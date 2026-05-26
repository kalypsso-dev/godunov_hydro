// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeDtHydroFunctor.h
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_COMPUTE_DT_HYDRO_FUNCTOR_H_
#define KALYPSSO_GODUNOV_HYDRO_COMPUTE_DT_HYDRO_FUNCTOR_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock
#include <kalypsso/core/FieldMap.h>
#include <kalypsso/core/orchard_key_base.h>

// hydro utils (conservative versus primitive variable, equation of state, ...)
#include <kalypsso/core/models/HydroState.h>
#include <kalypsso/core/models/utils_hydro.h>
#include <kalypsso/core/utils_block.h>
#include <kalypsso/core/GravityField.h>
#include <kalypsso/core/ViscosityParams.h>

#include <godunov_hydro/eos/EosWrapper.h>

namespace kalypsso
{
namespace godunov_hydro
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Simplest CFL computational functor for compressible mono fluid hydrodynamics.
 *
 * All cell, whatever level, contribute equally to the CFL condition.
 *
 * We actually compute inverse of cfl, the user is responsible to convert it to actual CFL.
 *
 * \tparam dim is space dimension (2 or 3)
 * \tparam device_t is the Kokkos device use for computation (CPU, GPU, ...)
 */
template <size_t dim, typename device_t>
class ComputeDtHydroFunctor
{

public:
  //! type alias for a data array at block level (see kalypsso_data_container.h)
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  //! type alias for a (device) Kokkos view of orchard keys
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

  // makes enum Hydro::VarId available
  using Hydro = kalypsso::core::models::Hydro;

  //! global cell index
  using index_t = int32_t;

private:
  //! list of orchard key of the mesh
  orchard_key_view_t m_orchard_keys;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! hydro parameters
  HydroSettings m_hydro_settings;

  //! Viscosity parameters
  ViscosityParams m_viscosity_params;

  //! field manager
  FieldMap<core::models::Hydro> m_fm;

  //! block sizes
  block_size_t<dim> m_block_sizes;

  //! number of cells per leaf
  const int32_t m_nbCellsPerLeaf;

  // get geometrical scaling factor
  const real_t m_scaling_factor;

  //! heavy data - conservative variables
  DataArrayBlock_t m_Udata;

  //! Stiffened gas eos parameters
  eos::EosWrapper<device_t> m_eos;

  //! gravity source term enabled ?
  const bool m_gravity_enabled;

  //! uniform gravity field
  const UniformGravityField<dim> m_gravity_field;

  ComputeDtHydroFunctor(ConfigMap const &                     config_map,
                        orchard_key_view_t const &            orchard_keys,
                        int32_t                               local_num_octants,
                        HydroSettings const &                 hydro_settings,
                        FieldMap<core::models::Hydro> const & fm,
                        block_size_t<dim> const &             block_sizes,
                        DataArrayBlock_t const &              Udata,
                        eos::EosWrapper<device_t> const &     eos,
                        bool                                  gravity_enabled,
                        UniformGravityField<dim>              gravity_field);

public:
  // ====================================================================
  // ====================================================================
  //! static method which does it all: create and execute functor using range policy
  //!
  //! \param[in] orchard_keys is a vector of all local (owned+ghost) octant orchard/morton keys
  //! \param[in] local_num_octants is the number of octants owned by current MPI process (ghost
  //!            excluded)
  //! \param[in] hydro_settings contains hydrodynamics parameter used to perform conservative to
  //! primitive
  //!            variable conversion (equation of state)
  //! \param[in] fm is the field map (TODO refactor this)
  //! \param[in] block_sizes is an array the cartesian block sizes
  //! \param[in,out] invDt is the inverse of time step, the output of this functor
  //!
  static void
  apply(ConfigMap const &                     config_map,
        orchard_key_view_t const &            orchard_keys,
        int32_t                               local_num_octants,
        HydroSettings const &                 hydro_settings,
        FieldMap<core::models::Hydro> const & fm,
        block_size_t<dim> const &             block_sizes,
        DataArrayBlock_t const &              Udata,
        eos::EosWrapper<device_t> const &     eos,
        real_t &                              invDt);

  // ====================================================================
  // ====================================================================
  /**
   * Update reduced variable when visiting a cell.
   *
   * \param[in] iOct is the visited octant id
   * \param[in] cell_index is the visited local cell index (local to block)
   * \param[in,out] invDt is the reduced variable to update
   *
   */
  KOKKOS_INLINE_FUNCTION void
  compute_cfl(int32_t const & iOct, int32_t const & cell_index, real_t & invDt) const;

  KOKKOS_INLINE_FUNCTION void
  compute_cfl_with_gravity(int32_t const & iOct, int32_t const & cell_index, real_t & invDt) const;

  /**
   * range policy functor for computing CFL condition.
   *
   * \param[in] global_index spans range from 0 to nbCellsPerLeaf * local_num_octants-1
   *            (i.e. total number of cells in current MPI process)
   * \param[in,out] invDt is the reduced variable to update
   */
  KOKKOS_INLINE_FUNCTION void
  operator()(const index_t & global_index, real_t & invDt) const;

}; // class ComputeDtHydroFunctor

// explicit template instantiation
extern template class ComputeDtHydroFunctor<2, kalypsso::DefaultDevice>;
extern template class ComputeDtHydroFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_COMPUTE_DT_HYDRO_FUNCTOR_H_
