// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeLimitedSlopesFunctor.h
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_COMPUTE_LIMITED_SLOPES_H_
#define KALYPSSO_GODUNOV_HYDRO_COMPUTE_LIMITED_SLOPES_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock
#include <kalypsso/core/FieldMap.h>

// hydro utils (conservative versus primitive variable, equation of state, ...)
#include <kalypsso/core/models/Hydro.h>
#include <kalypsso/core/models/HydroState.h>
#include <kalypsso/core/models/utils_hydro.h>
#include <kalypsso/core/utils_block.h>

namespace kalypsso
{
namespace godunov_hydro
{

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Compute limited slopes in a range of octants.
 *
 * \note
 * This functor has two modes of operation:
 * - either we compute slopes in a group of owned octants (batch mode, see TagComputeAllQuad)
 * - either we compute slopes in ghost octants (see TagComputeGhostQuad)
 * In batch mode, primitive variables array and slopes arrays must have the same sizes.
 * When computing in a ghost octant, the slopes array are sized upon the number of ghost octants.
 *
 * \note
 * For each octant containing a (bx,by,bz) cells grid, slopes are computed in a (bx+1,by+1,bz+1)
 * block of cells (original block equipped with one ghost cell all around); the primitive variables
 * ghosted array has block size (bx+2, by+2, bz+2).
 *
 * \note
 * The slopes are stored in external storage, a ghosted block array, to be reused later by the
 * Godunov functor (to compute hydro fluxes).
 *
 */
template <size_t dim, typename device_t>
class ComputeLimitedSlopesFunctor
{

public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;

  // data array related type aliases
  using DataArrayGhostedBlock_t = DataArrayGhostedBlock<dim, real_t, device_t>;

  // makes enum Hydro::VarId available
  using Hydro = kalypsso::core::models::Hydro;

private:
  //! a ghosted block array of primitive variables (ghost width is 2)
  DataArrayGhostedBlock_t m_q;

  //! ghosted block data arrays (ghost width is 1) - slopes along X
  DataArrayGhostedBlock_t m_slopes_x;

  //! ghosted block data arrays (ghost width is 1) - slopes along Y
  DataArrayGhostedBlock_t m_slopes_y;

  //! ghosted block data arrays (ghost width is 1) - slopes along Z - only used when dim=3
  DataArrayGhostedBlock_t m_slopes_z;

  //! field manager
  FieldMap<core::models::Hydro> m_fm;

  //! starting octant id
  const int32_t m_iOct_begin;

  //! number of octant to process, starting at m_iOct_begin
  const int32_t m_num_octants;

  //! hydro settings (EOS parameters)
  HydroSettings m_hydro_settings;

public:
  //! Tag used for computing only ghost quadrants (in q_mg)
  struct TagComputeGhostQuad
  {};

  //! Tag used for computing only a subset (group) of quadrants
  struct TagComputeAllQuadInGroup
  {};

  // ====================================================================
  // ====================================================================
  /**
   * Compute limited slopes functor.
   *
   * \param[in] prim_var : primitives variables (ghosted block, width=2)
   * \param[in,out] slopes_x : limited slopes along x (ghosted block width=1)
   * \param[in,out] slopes_y : limited slopes along y (ghosted block width=1)
   * \param[in,out] slopes_z : limited slopes along z (ghosted block width=1) - not used in 2d
   * \param[in] iOct_begin is the first octant index to process wrt the primitives variables array
   *            - when computing slopes in a group of owned octant, iOct_begin must be 0
   *               (because primitive var and slopes array are the same size)
   *            - when computing slopes in ghost octants, we don't compute slopes in mirror octant,
   *              only in ghosts, so we need to skip mirror octants
   *
   * \param[in] num_octants is the number of octant to process
   * \param[in] hydro_settings contains specific useful parameters like slope type
   *
   *
   */
  ComputeLimitedSlopesFunctor(DataArrayGhostedBlock_t       prim_var,
                              DataArrayGhostedBlock_t       slopes_x,
                              DataArrayGhostedBlock_t       slopes_y,
                              DataArrayGhostedBlock_t       slopes_z,
                              FieldMap<core::models::Hydro> fm,
                              int32_t                       iOct_begin,
                              int32_t                       num_octants,
                              HydroSettings                 hydro_settings);

  // ====================================================================
  // ====================================================================
  //! static method which does it all: create and execute functor with range policy
  //!
  //! Use this member when computing slopes in a group of octant
  static void
  apply_on_group(DataArrayGhostedBlock_t       primitive_vars,
                 DataArrayGhostedBlock_t       slopes_x,
                 DataArrayGhostedBlock_t       slopes_y,
                 DataArrayGhostedBlock_t       slopes_z,
                 FieldMap<core::models::Hydro> fm,
                 int32_t                       num_octants,
                 HydroSettings                 hydro_settings);

  // ====================================================================
  // ====================================================================
  //! static method which does it all: create and execute functor with range policy
  //!
  //! \param[in] primitive_vars_mg is a ghosted data array of primitive variables in mirror and
  //! ghost quads quadrants.
  //!
  //! \param[out] slopes_x primitive variable slopes along X axis in ghost quads
  //! \param[out] slopes_y primitive variable slopes along Y axis in ghost quads
  //! \param[out] slopes_z primitive variable slopes along Z axis in ghost quads
  //!
  //! \note primitive_vars_mg is sized upon the number of mirrors and ghosts quad (not just ghosts
  //! quadrants) because it is the output of MeshGhostExchanger.
  //!
  //! \note slopes arrays are sized upon the number of owned+ghosts quadrants.
  //!
  //!
  //! Use this member when computing slopes in ghost quadrants.
  static void
  apply_on_ghosts(DataArrayGhostedBlock_t       primitive_vars_mg,
                  DataArrayGhostedBlock_t       slopes_x,
                  DataArrayGhostedBlock_t       slopes_y,
                  DataArrayGhostedBlock_t       slopes_z,
                  FieldMap<core::models::Hydro> fm,
                  int32_t                       num_mirrors,
                  int32_t                       num_ghosts,
                  HydroSettings                 hydro_settings);

  // ====================================================================
  // ====================================================================
  /**
   * Compute primitive variables slopes (dq) for one component from q and its neighbors.
   *
   * Only slope_type 1 and 2 are supported.
   *
   * \param[in] q scalar value in current cell
   * \param[in] qPlus scalar value in right neighbor
   * \param[in] qMinus scalar value in left neighbor
   *
   * \return dq limited slope (scalar)
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  slope_unsplit_scalar(real_t q, real_t qPlus, real_t qMinus) const;

  // ====================================================================
  // ====================================================================
  /**
   * Compute limited slopes in all cells of ghosted block array (slopes_x, ...)
   *
   * \param[in] cell index integer used to map a cell inside a block
   * \param[in] iOct_in index identify an octant in input array
   * \param[in] iOct_out index identify an octant in output array
   *
   * This routine can be used to compute slopes either in own quadrant or in ghost quadrant, but it
   * must be done in two separate parallel_for (see TagComputeAllQuadInGroup and
   * TagComputeGhostQuad).
   *
   * \note watchout index shift m_q has a ghost width of 2, while slopes_x (and other) have a ghost
   * width of 1.
   */
  KOKKOS_INLINE_FUNCTION
  void
  compute_limited_slopes(index_t const & cell_index,
                         index_t const   iOct_in,
                         index_t const   iOct_out) const;

  // ====================================================================
  // ====================================================================
  /**
   * range policy functor for computing limited slopes in all quadrants of a group of quadrants.
   */
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagComputeAllQuadInGroup const &, const index_t & global_index) const;

  // ====================================================================
  // ====================================================================
  /**
   * range policy functor when computing only ghosts quadrants.
   */
  KOKKOS_INLINE_FUNCTION void
  operator()(TagComputeGhostQuad const &, const index_t & global_index) const;

}; // class ComputeLimitedSlopesFunctor

// explicit template instantiation
extern template class ComputeLimitedSlopesFunctor<2, kalypsso::DefaultDevice>;
extern template class ComputeLimitedSlopesFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_COMPUTE_LIMITED_SLOPES_H_
