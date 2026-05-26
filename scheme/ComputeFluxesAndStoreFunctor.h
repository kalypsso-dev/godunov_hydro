// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeFluxesAndStoreFunctor.h
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_COMPUTE_FLUXES_AND_STORE_FUNCTOR_H_
#define KALYPSSO_GODUNOV_HYDRO_COMPUTE_FLUXES_AND_STORE_FUNCTOR_H_

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/FieldMap.h>
#include <kalypsso/core/models/HydroState.h>
#include <kalypsso/core/models/RiemannSolvers.h>
#include <kalypsso/core/ConformalFaceStatus.h>
#include <kalypsso/core/StencilHelper.h>
#include <kalypsso/core/AMRMeshInfo.h>
#include <kalypsso/core/GravityField.h>
#include <kalypsso/core/ViscosityParams.h>
#include <kalypsso/core/TimeIntegratorConfig.h>

// utils hydro
#include <kalypsso/core/models/utils_hydro.h>

// equation of state wrapper
#include <godunov_hydro/eos/EosWrapper.h>

#include <type_traits>

namespace kalypsso
{

namespace godunov_hydro
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Compute fluxes (on conservative variables) and store. Actual update of conservative
 * variables (i.e. perform time integration using Godunov (e.g. MUSCL-Hancock) scheme) will be
 * done in a separate functor. This functor is designed to be used when performing a non-piecewise
 * godunov update (all block at once).
 *
 * Input data is Ugroup (containing ghosted block data)
 *
 * We compute fluxes (using Riemann solver) and store.
 * Loop through all cell (sub-)faces.
 *
 * \note This functor actually assumes the slopes array to be ghosted array with ghost width of 1.
 * Conservative variable array is assume to be a block array (no ghost).
 *
 * \todo routines like reconstruct_state_2d/3d could probably be
 * moved outside to alleviate this class.
 *
 */
template <size_t dim, typename device_t>
class ComputeFluxesAndStoreFunctor
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int64_t;

  // data array related type aliases
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;

  // makes enum Hydro::VarId available
  using Hydro = kalypsso::core::models::Hydro;

  // access quadrant <-> orchard key (hence AMR level and quadrant size)
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  template <size_t _dim>
  using offsets_t = coord_t<_dim, real_t>;

private:
  //! list of orchard key of the mesh
  orchard_key_view_t m_orchard_keys_device;

  //! AMR mesh info (number of owned, MPI ghost, outside quadrants)
  AMRMeshInfo m_amr_mesh_info;

  //! fluxes (output)
  DataArrayBlock_t m_Fluxes;

  //! a ghosted block array of primitive variables (ghost width is 2)
  //! size :
  //! if implem version 0 : owned + ghost quadrants
  //! if implem version 1 : size of group of quadrants
  DataArrayGhostedBlock_t m_q;

  //! ghosted block data arrays (ghost width is 1) - slopes along X
  DataArrayGhostedBlock_t m_slopes_x;

  //! ghosted block data arrays (ghost width is 1) - slopes along Y
  DataArrayGhostedBlock_t m_slopes_y;

  //! ghosted block data arrays (ghost width is 1) - slopes along Z - only used when dim=3
  DataArrayGhostedBlock_t m_slopes_z;

  //! field manager
  FieldMap<core::models::Hydro> m_fm;

  //! offset to first octant in flux array where to write
  const int32_t m_iOct_flux_offset;

  //! number of quadrants to process
  const int32_t m_num_quads;

  //! flux direction (IX, IY or IZ)
  int m_direction;

  //! block sizes (no ghost)
  const block_size_t<dim> m_block_sizes;

  //! number of cells per leaf block
  const int32_t m_nbCellsPerLeaf;

  //! hydro settings
  HydroSettings m_hydro_settings;

  //! EOS parameters
  eos::EosWrapper<device_t> m_eos;

  //! time step
  real_t m_dt;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! gravity source term enabled ?
  const bool m_gravity_enabled;

  //! uniform gravity field
  const UniformGravityField<dim> m_gravity_field;

  //! viscosity parameters (needed for the Muscl-Hancock predictor)
  const ViscosityParams m_viscosity;

  //! time integrator id
  const TimeIntegrator m_time_integrator;

public:
  /**
   * Compute Godunov fluxes along a given direction.
   *
   * \param[in]  time step (as computed by CFL condition)
   *
   */
  ComputeFluxesAndStoreFunctor(orchard_key_view_t const &            orchard_keys,
                               AMRMeshInfo const &                   amr_mesh_info,
                               DataArrayBlock_t const &              fluxes,
                               DataArrayGhostedBlock_t const &       q_ghosted,
                               DataArrayGhostedBlock_t const &       slopes_x,
                               DataArrayGhostedBlock_t const &       slopes_y,
                               DataArrayGhostedBlock_t const &       slopes_z,
                               FieldMap<core::models::Hydro> const & fm,
                               int32_t                               iOct_flux_offset,
                               int32_t                               num_quads,
                               int                                   direction,
                               HydroSettings const &                 hydro_settings,
                               eos::EosWrapper<device_t> const &     eos,
                               real_t                                dt,
                               real_t                                scaling_factor,
                               bool                                  gravity_enabled,
                               UniformGravityField<dim> const &      gravity_field,
                               ViscosityParams const &               viscosity,
                               TimeIntegrator                        time_integrator);

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  //!
  static void
  apply(ConfigMap const &                     config_map,
        orchard_key_view_t const &            orchard_keys,
        AMRMeshInfo const &                   amr_mesh_info,
        DataArrayBlock_t const &              fluxes,
        DataArrayGhostedBlock_t const &       q_ghosted,
        DataArrayGhostedBlock_t const &       slopes_x,
        DataArrayGhostedBlock_t const &       slopes_y,
        DataArrayGhostedBlock_t const &       slopes_z,
        FieldMap<core::models::Hydro> const & fm,
        int32_t                               iOct_flux_offset,
        int32_t                               num_quads,
        int                                   direction,
        HydroSettings const &                 hydro_settings,
        eos::EosWrapper<device_t> const &     eos,
        ViscosityParams const &               viscosity,
        real_t                                dt);

  // ====================================================================
  // ====================================================================
  /**
   * Get primitive variables state vector.
   *
   * \param[in] index identifies location in the ghosted block
   * \param[in] iOct_local identifies octant (from 0 to owned+ghost-1)
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  get_prim_variables(int32_t i, int32_t j, int32_t iOct_local) const
  {

    HydroState<2> q;

    q[Hydro::ID] = m_q(i, j, m_fm[Hydro::ID], iOct_local);
    q[Hydro::IP] = m_q(i, j, m_fm[Hydro::IP], iOct_local);
    q[Hydro::IU] = m_q(i, j, m_fm[Hydro::IU], iOct_local);
    q[Hydro::IV] = m_q(i, j, m_fm[Hydro::IV], iOct_local);

    return q;

  } // get_prim_variables

  // ====================================================================
  // ====================================================================
  /**
   * Get primitive variables state vector.
   *
   * \param[in] index identifies location in the ghosted block
   * \param[in] iOct_local identifies octant (from 0 to owned+ghost-1)
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  get_prim_variables(int32_t i, int32_t j, int32_t k, int32_t iOct_local) const
  {

    HydroState<3> q;

    q[Hydro::ID] = m_q(i, j, k, m_fm[Hydro::ID], iOct_local);
    q[Hydro::IP] = m_q(i, j, k, m_fm[Hydro::IP], iOct_local);
    q[Hydro::IU] = m_q(i, j, k, m_fm[Hydro::IU], iOct_local);
    q[Hydro::IV] = m_q(i, j, k, m_fm[Hydro::IV], iOct_local);
    q[Hydro::IW] = m_q(i, j, k, m_fm[Hydro::IW], iOct_local);

    return q;

  } // get_prim_variables

  // ====================================================================
  // ====================================================================
  /**
   * Set flux (hydro variables only).
   *
   * \param[in] i identifies location in the flux in block
   * \param[in] j identifies location in the flux in block
   * \param[in] iOct identifies octant (local index relative to
   *            a group of octant)
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  set_flux(int32_t i, int32_t j, int32_t iOct, HydroState<2> const & flux) const
  {
    iOct += m_iOct_flux_offset;

    m_Fluxes(i, j, m_fm[Hydro::ID], iOct) = flux[Hydro::ID];
    m_Fluxes(i, j, m_fm[Hydro::IP], iOct) = flux[Hydro::IP];
    m_Fluxes(i, j, m_fm[Hydro::IU], iOct) = flux[Hydro::IU];
    m_Fluxes(i, j, m_fm[Hydro::IV], iOct) = flux[Hydro::IV];

  } // set_flux - 2d

  // ====================================================================
  // ====================================================================
  /**
   * Set flux (hydro variables only).
   *
   * \param[in] i identifies location in the flux in block
   * \param[in] j identifies location in the flux in block
   * \param[in] k identifies location in the flux in block
   * \param[in] iOct identifies octant (local index relative to
   *            a group of octant)
   *
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  set_flux(int32_t i, int32_t j, int32_t k, int32_t iOct, HydroState<3> const & flux) const
  {
    iOct += m_iOct_flux_offset;

    m_Fluxes(i, j, k, m_fm[Hydro::ID], iOct) = flux[Hydro::ID];
    m_Fluxes(i, j, k, m_fm[Hydro::IP], iOct) = flux[Hydro::IP];
    m_Fluxes(i, j, k, m_fm[Hydro::IU], iOct) = flux[Hydro::IU];
    m_Fluxes(i, j, k, m_fm[Hydro::IV], iOct) = flux[Hydro::IV];
    m_Fluxes(i, j, k, m_fm[Hydro::IW], iOct) = flux[Hydro::IW];

  } // set_flux - 3d

  // ====================================================================
  // ====================================================================
  /**
   * Reconstruct an hydro state at a cell border location specified by offsets.
   *
   * This is equivalent to trace operation in Ramses.
   * We just extrapolate primitive variables (at cell center) to border
   * using limited slopes.
   *
   * \note offsets are given in units dx/2, i.e. a vector containing only 1.0 or -1.0
   *
   * \param[in] q primitive variables at cell center
   * \param[in] i_s X coordinate to access slope array
   * \param[in] j_s Y coordinate to access slope array
   * \param[in] iOct_local index to octant in local array
   * \param[in] offsets identifies where to reconstruct
   * \param[in] dtdx dt divided by dx
   * \param[in] dtdy dt divided by dy
   * \param[in] dx (only used when adding viscous force predictor)
   * \param[in] dy (only used when adding viscous force predictor)
   *
   * \return qr reconstructed state (primitive variables)
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  reconstruct_state_2d(const HydroState<2> &   q,
                       int32_t                 i_s,
                       int32_t                 j_s,
                       int32_t                 iOct_local,
                       const offsets_t<2> &    offsets,
                       real_t                  dtdx,
                       real_t                  dtdy,
                       [[maybe_unused]] real_t dx,
                       [[maybe_unused]] real_t dy) const;

  // ====================================================================
  // ====================================================================
  /**
   * Reconstruct an hydro state at a cell border location specified by offsets (3d version).
   *
   * This is equivalent to trace operation in Ramses.
   * We just extrapolate primitive variables (at cell center) to border
   * using limited slopes.
   *
   * \note offsets are given in units dx/2, i.e. a vector containing only 1.0 or -1.0
   *
   * \param[in] q primitive variables at cell center
   * \param[in] is X coordinate to access slope array
   * \param[in] js Y coordinate to access slope array
   * \param[in] ks Y coordinate to access slope array
   * \param[in] iOct_local index to octant in local array
   * \param[in] offsets identifies where to reconstruct
   * \param[in] dtdx dt divided by dx
   * \param[in] dtdy dt divided by dy
   * \param[in] dtdz dt divided by dz
   * \param[in] dx (only used when adding viscous force predictor)
   * \param[in] dy (only used when adding viscous force predictor)
   * \param[in] dz (only used when adding viscous force predictor)
   *
   * \return qr reconstructed state (primitive variables)
   *
   * \sa reconstruct_state_2d
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  reconstruct_state_3d(const HydroState<3> &   q,
                       int32_t                 is,
                       int32_t                 js,
                       int32_t                 ks,
                       int32_t                 iOct_local,
                       const offsets_t<3> &    offsets,
                       real_t                  dtdx,
                       real_t                  dtdy,
                       real_t                  dtdz,
                       [[maybe_unused]] real_t dx,
                       [[maybe_unused]] real_t dy,
                       [[maybe_unused]] real_t dz) const;

  // ====================================================================
  // ====================================================================
  /**
   * Add viscous force predictor to reconstructed primitive variables state.
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  add_viscous_predictor_2d(HydroState<2> & qr,
                           int32_t         i_s,
                           int32_t         j_s,
                           int32_t         iOct_local,
                           real_t          dtdx,
                           real_t          dtdy,
                           real_t          dx,
                           real_t          dy) const;

  // ====================================================================
  // ====================================================================
  /**
   * Add viscous force predictor to reconstructed primitive variables state.
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  add_viscous_predictor_3d(HydroState<3> & qr,
                           int32_t         i_s,
                           int32_t         j_s,
                           int32_t         k_s,
                           int32_t         iOct_local,
                           real_t          dtdx,
                           real_t          dtdy,
                           real_t          dtdz,
                           real_t          dx,
                           real_t          dy,
                           real_t          dz) const;

  // ====================================================================
  // ====================================================================
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  compute_fluxes_and_store_2d(int32_t const & cell_index, int32_t const & iOct_local) const;

  // ====================================================================
  // ====================================================================
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  compute_fluxes_and_store_3d(const int32_t & cell_index, const int32_t & iOct_local) const;

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const index_t & global_index) const;

}; // ComputeFluxesAndStoreFunctor

// explicit template instantiation
extern template class ComputeFluxesAndStoreFunctor<2, kalypsso::DefaultDevice>;
extern template class ComputeFluxesAndStoreFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_COMPUTE_FLUXES_AND_STORE_FUNCTOR_H_
