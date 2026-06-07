// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeFluxesAndUpdateFunctor.h
 *
 * This is a legacy functor (don't use it). It is still here for reference.
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_COMPUTE_FLUXES_AND_UPDATE_FUNCTOR_H_
#define KALYPSSO_GODUNOV_HYDRO_COMPUTE_FLUXES_AND_UPDATE_FUNCTOR_H_

#include <godunov_hydro/common.h>

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/amr_hashmap.h>
#include <kalypsso/core/Kokkos_Array_extensions.h>
#include <kalypsso/core/models/RiemannSolvers.h>
#include <kalypsso/core/GravityField.h>

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
 * Compute fluxes (on conservative variables) and perform a NON CONSERVATIVE update of conservative
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
class ComputeFluxesAndUpdateFunctor
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;

  // hashmap related type aliases
  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  // data array related type aliases
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;

  template <size_t _dim>
  using offsets_t = coord_t<_dim, real_t>;

private:
  //! AMR unordered map which maps orchard keys to quadrant number for all key in the mesh
  //! (owned quadrants and ghost quadrants)
  amr_hashmap_t m_amr_hashmap_device;

  //! list of orchard key of the mesh
  orchard_key_view_t m_orchard_keys_device;

  //! list of orchard keys that are "mirrors" (in the p4est sense).
  //! only used when we want to solely computed mirror quadrants.
  orchard_key_view_t m_mirror_orchard_keys_device;

  //! user data - entire mesh - input/output
  DataArrayBlock_t m_U;

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

  //! p4est brick connectivity sizes
  const Kokkos::Array<uint8_t, dim> m_brick_sizes;

  //! is p4est connectivity periodic ?
  const Kokkos::Array<bool, dim> m_is_brick_periodic;

  //! hydro settings (EOS parameters)
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

  //! bool use flux oriented computation
  bool m_use_flux_oriented_computation;

public:
  struct TagComputeGhostQuad
  {};
  struct TagComputeAllQuadInGroup
  {};

  auto
  nb_cells_per_leaf() const
  {
    return m_nbCellsPerLeaf;
  }

  auto
  nb_fluxes_per_leaf() const
  {
    return m_nbFluxesPerLeaf;
  }

  /**
   * Perform time integration (MUSCL Godunov).
   *
   * \param[in]  time step (as cmputed by CFL condition)
   *
   */
  ComputeFluxesAndUpdateFunctor(ConfigMap const &                   config_map,
                                amr_hashmap_t const &               amr_hashmap,
                                orchard_key_view_t const &          orchard_keys,
                                DataArrayBlock_t const &            u,
                                DataArrayGhostedBlock_t const &     q,
                                DataArrayGhostedBlock_t const &     slopes_x,
                                DataArrayGhostedBlock_t const &     slopes_y,
                                DataArrayGhostedBlock_t const &     slopes_z,
                                int32_t                             iOct_begin,
                                int32_t                             num_octants,
                                Kokkos::Array<uint8_t, dim> const & brick_sizes,
                                Kokkos::Array<bool, dim> const &    is_brick_periodic,
                                HydroSettings const &               hydro_settings,
                                eos::EosWrapper<device_t> const &   eos,
                                real_t                              dt,
                                bool                                gravity_enabled,
                                UniformGravityField<dim> const &    gravity_field,
                                bool                                use_flux_oriented_computation);

  // ==============================================================
  // ==============================================================
  //! static method which does it all: create and execute functor with range policy
  //!
  //! Use this member when computing primitive in a group of octant
  static void
  apply_on_group(ConfigMap const &                   config_map,
                 amr_hashmap_t const &               amr_hashmap,
                 orchard_key_view_t const &          orchard_keys,
                 DataArrayBlock_t const &            U,
                 DataArrayGhostedBlock_t const &     q,
                 DataArrayGhostedBlock_t const &     slopes_x,
                 DataArrayGhostedBlock_t const &     slopes_y,
                 DataArrayGhostedBlock_t const &     slopes_z,
                 int32_t                             iOct_begin,
                 int32_t                             num_octants,
                 Kokkos::Array<uint8_t, dim> const & brick_sizes,
                 Kokkos::Array<bool, dim> const &    is_brick_periodic,
                 HydroSettings const &               hydro_settings,
                 eos::EosWrapper<device_t> const &   eos,
                 real_t                              dt,
                 bool                                use_flux_oriented_computation);

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
  get_cons_variables(int32_t i, int32_t j, int32_t iOct_global) const;

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
  get_cons_variables(int32_t i, int32_t j, int32_t k, int32_t iOct_global) const;

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
  get_prim_variables(int32_t i, int32_t j, int32_t iOct_local) const;

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
  get_prim_variables(int32_t i, int32_t j, int32_t k, int32_t iOct_local) const;

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
  compute_fluxes_and_update_2d(index_t const & cell_index, index_t const & iOct_local) const;

  // ====================================================================
  // ====================================================================
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 2), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  compute_fluxes_and_update_2d_atomic(index_t const & cell_index, index_t const & iOct_local) const;

  // ====================================================================
  // ====================================================================
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  compute_fluxes_and_update_3d(const index_t & cell_index, const index_t & iOct_local) const;

  // ====================================================================
  // ====================================================================
  template <size_t dim_ = dim, std::enable_if_t<(dim_ == 3), bool> = true>
  KOKKOS_INLINE_FUNCTION void
  compute_fluxes_and_update_3d_atomic(const index_t & cell_index, const index_t & iOct_local) const;

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagComputeAllQuadInGroup const &, const index_t & global_index) const;

}; // ComputeFluxesAndUpdateFunctor

// explicit template instantiation
extern template class ComputeFluxesAndUpdateFunctor<2, kalypsso::DefaultDevice>;
extern template class ComputeFluxesAndUpdateFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_COMPUTE_FLUXES_AND_UPDATE_FUNCTOR_H_
