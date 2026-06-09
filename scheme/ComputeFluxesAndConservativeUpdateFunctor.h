// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeFluxesAndConservativeUpdateFunctor.h
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_COMPUTE_FLUXES_AND_CONSERVATIVE_UPDATE_FUNCTOR_H_
#define KALYPSSO_GODUNOV_HYDRO_COMPUTE_FLUXES_AND_CONSERVATIVE_UPDATE_FUNCTOR_H_

#include <godunov_hydro/common.h>

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/ConformalFaceStatus.h>
#include <kalypsso/core/StencilHelper.h>
#include <kalypsso/core/AMRMeshInfo.h>
#include <kalypsso/core/GravityField.h>
#include <kalypsso/core/TimeIntegratorConfig.h>

// equation of state wrapper
#include <godunov_hydro/eos/EosWrapper.h>
#include <godunov_hydro/models/RiemannSolvers.h>

#include <type_traits>

namespace kalypsso
{

namespace godunov_hydro
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Compute fluxes (on conservative variables) and perform a CONSERVATIVE update of conservative
 * variables (i.e. perform time integration using Godunov (e.g. MUSCL-Hancock) scheme).
 *
 * Input data is Ugroup (containing ghosted block data)
 *
 * We compute fluxes (using Riemann solver) and perform
 * update directly in external array U.
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
class ComputeFluxesAndConservativeUpdateFunctor
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;

  // hashmap related type aliases
  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  using conformal_status_view_type = conformal_status_view_t<dim, device_t>;

  // data array related type aliases
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;

  template <size_t _dim>
  using offsets_t = coord_t<_dim, real_t>;

  using CellLocation_t = CellLocation<dim>;
  using StencilHelper_t = StencilHelper<dim, device_t>;

private:
  //! helper to compute neighbor cell location
  StencilHelper_t m_stencil_helper;

  //! list of orchard key of the mesh
  orchard_key_view_t m_orchard_keys_device;

  //! conformal status view
  conformal_status_view_type m_conformal_status;

  //! AMR mesh info (number of owned, MPI ghost, outside quadrants)
  AMRMeshInfo m_amr_mesh_info;

  //! user data - entire mesh - input
  DataArrayBlock_t m_Uin;

  //! user data - entire mesh - output
  DataArrayBlock_t m_Uout;

  //! a ghosted block array of primitive variables (ghost width is 2)
  DataArrayGhostedBlock_t m_q;

  //! ghosted block data arrays (ghost width is 1) - slopes along X
  DataArrayGhostedBlock_t m_slopes_x;

  //! ghosted block data arrays (ghost width is 1) - slopes along Y
  DataArrayGhostedBlock_t m_slopes_y;

  //! ghosted block data arrays (ghost width is 1) - slopes along Z - only used when dim=3
  DataArrayGhostedBlock_t m_slopes_z;

  //! starting octant id
  const int32_t m_iOct_begin;

  //! number of octant to process, starting at m_iOct_begin
  const int32_t m_num_octants;

  //! block sizes (no ghost)
  const block_size_t<dim> m_block_sizes;

  //! block sizes with 1 ghost on the right (to be able to compute fluxes on the last right cell)
  const block_size_t<dim> m_block_sizes_fluxes;

  //! number of cells per leaf block
  const int32_t m_nbCellsPerLeaf;

  //! number of fluxes per leaf block (just 1 more than the number of cells in all direction)
  const int32_t m_nbFluxesPerLeaf;

  //! hydro settings
  HydroSettings m_hydro_settings;

  //! EOS parameters
  eos::EosWrapper<device_t> m_eos;

  //! time step
  real_t m_dt;

  //! geometrical scaling factor
  const real_t m_scaling_factor;

  //! gravity source term enabled ?
  const bool m_gravity_enabled;

  //! uniform gravity field
  const UniformGravityField<dim> m_gravity_field;

  //! time integrator id
  const TimeIntegrator m_time_integrator;

public:
  struct TagComputeGhostQuad
  {};
  struct TagComputeAllQuadInGroup
  {};

  auto
  nb_fluxes_per_leaf() const
  {
    return m_nbFluxesPerLeaf;
  }

  /**
   * Perform full or partial time integration (either MUSCL-Hancock or one Runge-Kutta step).
   *
   * \param[in]  time step (as computed by CFL condition)
   *
   */
  ComputeFluxesAndConservativeUpdateFunctor(ConfigMap const &                  config_map,
                                            StencilHelper_t const &            stencil_helper,
                                            orchard_key_view_t const &         orchard_keys,
                                            conformal_status_view_type const & conformal_status,
                                            AMRMeshInfo const &                amr_mesh_info,
                                            DataArrayBlock_t const &           u_in,
                                            DataArrayBlock_t const &           u_out,
                                            DataArrayGhostedBlock_t const &    q,
                                            DataArrayGhostedBlock_t const &    slopes_x,
                                            DataArrayGhostedBlock_t const &    slopes_y,
                                            DataArrayGhostedBlock_t const &    slopes_z,
                                            int32_t                            iOct_begin,
                                            int32_t                            num_octants,
                                            HydroSettings const &              hydro_settings,
                                            eos::EosWrapper<device_t> const &  eos,
                                            real_t                             dt,
                                            bool                               gravity_enabled,
                                            UniformGravityField<dim> const &   gravity_field,
                                            TimeIntegrator                     time_integrator);

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  //!
  //! Use this member when computing primitive in a group of octant
  static void
  apply_on_group(ConfigMap const &                  config_map,
                 amr_hashmap_t const &              amr_hashmap,
                 orchard_key_view_t const &         orchard_keys,
                 conformal_status_view_type const & conformal_status,
                 AMRMeshInfo const &                amr_mesh_info,
                 DataArrayBlock_t const &           Uin,
                 DataArrayBlock_t const &           Uout,
                 DataArrayGhostedBlock_t const &    q,
                 DataArrayGhostedBlock_t const &    slopes_x,
                 DataArrayGhostedBlock_t const &    slopes_y,
                 DataArrayGhostedBlock_t const &    slopes_z,
                 int32_t                            iOct_begin,
                 int32_t                            num_octants,
                 brick_size_t<dim> const &          brick_sizes,
                 Kokkos::Array<bool, dim> const &   is_brick_periodic,
                 HydroSettings const &              hydro_settings,
                 eos::EosWrapper<device_t> const &  eos,
                 real_t                             dt);

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  //!
  //! Use this member when computing primitive in ghosts octant
  static void
  apply_on_ghosts(ConfigMap const &                  config_map,
                  amr_hashmap_t const &              amr_hashmap,
                  orchard_key_view_t const &         orchard_keys,
                  conformal_status_view_type const & conformal_status,
                  AMRMeshInfo const &                amr_mesh_info,
                  DataArrayBlock_t const &           Uin,
                  DataArrayBlock_t const &           Uout,
                  DataArrayGhostedBlock_t const &    q,
                  DataArrayGhostedBlock_t const &    slopes_x,
                  DataArrayGhostedBlock_t const &    slopes_y,
                  DataArrayGhostedBlock_t const &    slopes_z,
                  brick_size_t<dim> const &          brick_sizes,
                  Kokkos::Array<bool, dim> const &   is_brick_periodic,
                  HydroSettings const &              hydro_settings,
                  eos::EosWrapper<device_t> const &  eos,
                  real_t                             dt);

  // ====================================================================
  // ====================================================================
  /**
   * return true when octant is a "owned" octant (not a ghost or outside)
   */
  KOKKOS_INLINE_FUNCTION bool
  is_owned_quadrant(iOct_t const & iOct_global) const
  {
    return iOct_global < m_amr_mesh_info.local_num_quadrants();
  }

  // ====================================================================
  // ====================================================================
  /**
   * Get conservative variables state vector.
   *
   * \param[in] index identifies location in the ghosted block
   * \param[in] iOct_global identifies octant (global)
   *
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  get_cons_variables(int32_t i, int32_t j, int32_t iOct_global) const
  {

    HydroState<2> u;

    u[Hydro<dim>::ID] = m_Uin(i, j, Hydro<dim>::ID, iOct_global);
    u[Hydro<dim>::IP] = m_Uin(i, j, Hydro<dim>::IP, iOct_global);
    u[Hydro<dim>::IU] = m_Uin(i, j, Hydro<dim>::IU, iOct_global);
    u[Hydro<dim>::IV] = m_Uin(i, j, Hydro<dim>::IV, iOct_global);

    return u;

  } // get_cons_variables

  // ====================================================================
  // ====================================================================
  /**
   * Get conservative variables state vector.
   *
   * \param[in] index identifies location in the ghosted block
   * \param[in] iOct_global identifies octant (global)
   *
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  get_cons_variables(int32_t i, int32_t j, int32_t k, int32_t iOct_global) const
  {

    HydroState<3> u;

    u[Hydro<dim>::ID] = m_Uin(i, j, k, Hydro<dim>::ID, iOct_global);
    u[Hydro<dim>::IP] = m_Uin(i, j, k, Hydro<dim>::IP, iOct_global);
    u[Hydro<dim>::IU] = m_Uin(i, j, k, Hydro<dim>::IU, iOct_global);
    u[Hydro<dim>::IV] = m_Uin(i, j, k, Hydro<dim>::IV, iOct_global);
    u[Hydro<dim>::IW] = m_Uin(i, j, k, Hydro<dim>::IW, iOct_global);

    return u;

  } // get_cons_variables

  // ====================================================================
  // ====================================================================
  /**
   * Get primitive variables state vector.
   *
   * \param[in] index identifies location in the ghosted block
   * \param[in] iOct_local identifies octant (local index relative to
   *            a group of octant)
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  get_prim_variables(int32_t i, int32_t j, int32_t iOct_local) const
  {

    HydroState<2> q;

    q[Hydro<dim>::ID] = m_q(i, j, Hydro<dim>::ID, iOct_local);
    q[Hydro<dim>::IP] = m_q(i, j, Hydro<dim>::IP, iOct_local);
    q[Hydro<dim>::IU] = m_q(i, j, Hydro<dim>::IU, iOct_local);
    q[Hydro<dim>::IV] = m_q(i, j, Hydro<dim>::IV, iOct_local);

    return q;

  } // get_prim_variables

  // ====================================================================
  // ====================================================================
  /**
   * Get primitive variables state vector.
   *
   * \param[in] index identifies location in the ghosted block
   * \param[in] iOct_local identifies octant (local index relative to
   *            a group of octant)
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  get_prim_variables(int32_t i, int32_t j, int32_t k, int32_t iOct_local) const
  {

    HydroState<3> q;

    q[Hydro<dim>::ID] = m_q(i, j, k, Hydro<dim>::ID, iOct_local);
    q[Hydro<dim>::IP] = m_q(i, j, k, Hydro<dim>::IP, iOct_local);
    q[Hydro<dim>::IU] = m_q(i, j, k, Hydro<dim>::IU, iOct_local);
    q[Hydro<dim>::IV] = m_q(i, j, k, Hydro<dim>::IV, iOct_local);
    q[Hydro<dim>::IW] = m_q(i, j, k, Hydro<dim>::IW, iOct_local);

    return q;

  } // get_prim_variables

  // ====================================================================
  // ====================================================================
  /**
   * Update state vector (hydro variables only) using Kokkos::atomic_add.
   *
   * \param[in] data a  block data array 2d view (Uout)
   * \param[in] i identifies location in the ghosted block
   * \param[in] j identifies location in the ghosted block
   * \param[in] iOct identifies octant (local index relative to
   *            a group of octant)
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  state_add(DataArrayBlock_t const & data,
            int32_t                  i,
            int32_t                  j,
            int64_t                  iOct,
            HydroState<2> const &    q) const
  {

    Kokkos::atomic_add(&data(i, j, Hydro<dim>::ID, iOct), q[Hydro<dim>::ID]);
    Kokkos::atomic_add(&data(i, j, Hydro<dim>::IP, iOct), q[Hydro<dim>::IP]);
    Kokkos::atomic_add(&data(i, j, Hydro<dim>::IU, iOct), q[Hydro<dim>::IU]);
    Kokkos::atomic_add(&data(i, j, Hydro<dim>::IV, iOct), q[Hydro<dim>::IV]);

  } // state_add - 2d

  // ====================================================================
  // ====================================================================
  /**
   * Update state vector (hydro variables only) using Kokkos::atomic_add.
   *
   * \param[in] data a block data array 3d view (Uout)
   * \param[in] i identifies location in the ghosted block
   * \param[in] j identifies location in the ghosted block
   * \param[in] k identifies location in the ghosted block
   * \param[in] iOct identifies octant (local index relative to
   *            a group of octant)
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  state_add(DataArrayBlock_t const & data,
            int32_t                  i,
            int32_t                  j,
            int32_t                  k,
            int64_t                  iOct,
            HydroState<3> const &    q) const
  {

    Kokkos::atomic_add(&data(i, j, k, Hydro<dim>::ID, iOct), q[Hydro<dim>::ID]);
    Kokkos::atomic_add(&data(i, j, k, Hydro<dim>::IP, iOct), q[Hydro<dim>::IP]);
    Kokkos::atomic_add(&data(i, j, k, Hydro<dim>::IU, iOct), q[Hydro<dim>::IU]);
    Kokkos::atomic_add(&data(i, j, k, Hydro<dim>::IV, iOct), q[Hydro<dim>::IV]);
    Kokkos::atomic_add(&data(i, j, k, Hydro<dim>::IW, iOct), q[Hydro<dim>::IW]);

  } // state_add - 3d

  // ====================================================================
  // ====================================================================
  /**
   * Update state vector (hydro variables only) using Kokkos::atomic_sub.
   *
   * \param[in] data a block data array 2d view (Uout)
   * \param[in] i identifies location in the ghosted block
   * \param[in] j identifies location in the ghosted block
   * \param[in] iOct identifies octant (local index relative to
   *            a group of octant)
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  state_sub(DataArrayBlock_t const & data,
            int32_t                  i,
            int32_t                  j,
            int64_t                  iOct,
            HydroState<2> const &    q) const
  {

    Kokkos::atomic_sub(&data(i, j, Hydro<dim>::ID, iOct), q[Hydro<dim>::ID]);
    Kokkos::atomic_sub(&data(i, j, Hydro<dim>::IP, iOct), q[Hydro<dim>::IP]);
    Kokkos::atomic_sub(&data(i, j, Hydro<dim>::IU, iOct), q[Hydro<dim>::IU]);
    Kokkos::atomic_sub(&data(i, j, Hydro<dim>::IV, iOct), q[Hydro<dim>::IV]);

  } // state_sub

  // ====================================================================
  // ====================================================================
  /**
   * Update state vector (hydro variables only) using Kokkos::atomic_sub.
   *
   * \param[in] data a block data array 3d view (Uout)
   * \param[in] i identifies location in the ghosted block
   * \param[in] j identifies location in the ghosted block
   * \param[in] k identifies location in the ghosted block
   * \param[in] iOct identifies octant (local index relative to
   *            a group of octant)
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  state_sub(DataArrayBlock_t const & data,
            int32_t                  i,
            int32_t                  j,
            int32_t                  k,
            int64_t                  iOct,
            HydroState<3> const &    q) const
  {

    Kokkos::atomic_sub(&data(i, j, k, Hydro<dim>::ID, iOct), q[Hydro<dim>::ID]);
    Kokkos::atomic_sub(&data(i, j, k, Hydro<dim>::IP, iOct), q[Hydro<dim>::IP]);
    Kokkos::atomic_sub(&data(i, j, k, Hydro<dim>::IU, iOct), q[Hydro<dim>::IU]);
    Kokkos::atomic_sub(&data(i, j, k, Hydro<dim>::IV, iOct), q[Hydro<dim>::IV]);
    Kokkos::atomic_sub(&data(i, j, k, Hydro<dim>::IW, iOct), q[Hydro<dim>::IW]);

  } // state_sub

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
   *
   * \return qr reconstructed state (primitive variables)
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  reconstruct_state_2d(const HydroState<2> & q,
                       int32_t               i_s,
                       int32_t               j_s,
                       int32_t               iOct_local,
                       const offsets_t<2> &  offsets,
                       real_t                dtdx,
                       real_t                dtdy) const;

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
   *
   * \return qr reconstructed state (primitive variables)
   *
   * \sa reconstruct_state_2d
   */
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION auto
  reconstruct_state_3d(const HydroState<3> & q,
                       int32_t               is,
                       int32_t               js,
                       int32_t               ks,
                       int32_t               iOct_local,
                       const offsets_t<3> &  offsets,
                       real_t                dtdx,
                       real_t                dtdy,
                       real_t                dtdz) const;

  // ====================================================================
  // ====================================================================
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  compute_fluxes_and_update_2d_group(index_t const & cell_index, index_t const & iOct_local) const;

  // ====================================================================
  // ====================================================================
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  compute_fluxes_and_update_2d_ghost(index_t const & cell_index,
                                     index_t const & first_ghost,
                                     index_t const & iGhost) const;

  // ====================================================================
  // ====================================================================
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  compute_fluxes_and_update_3d_group(const index_t & cell_index, const index_t & iOct_local) const;

  // ====================================================================
  // ====================================================================
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  compute_fluxes_and_update_3d_ghost(index_t const & cell_index,
                                     index_t const & first_ghost,
                                     index_t const & iGhost) const;

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagComputeAllQuadInGroup const &, const int64_t & global_index) const;

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagComputeGhostQuad const &, const int64_t & global_index) const;

}; // ComputeFluxesAndConservativeUpdateFunctor

// explicit template instantiation
extern template class ComputeFluxesAndConservativeUpdateFunctor<2, kalypsso::DefaultDevice>;
extern template class ComputeFluxesAndConservativeUpdateFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_COMPUTE_FLUXES_AND_CONSERVATIVE_UPDATE_FUNCTOR_H_
