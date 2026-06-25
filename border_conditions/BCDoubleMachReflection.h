// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file BCDoubleMachReflection.h
 *
 * Define a border condition functor that can be used to simulate an inclined shock wave entering
 * the computational domain from the left, and lower y-border is a wall.
 *
 * This functor is aimed to be called after FillOutside, to override values at some location.
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_BC_DOUBLE_MACH_REFLECTION_H_
#define KALYPSSO_GODUNOV_HYDRO_BC_DOUBLE_MACH_REFLECTION_H_

#include <kalypsso/core/FillOutside_utils.h>

#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost, DataArrayGhostedBlock<dim, device_t>

#include <kalypsso/core/orchard_key.h>
#include <kalypsso/core/orchard_key_utils.h>
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/amr_hashmap.h>

#include <kalypsso/core/utils_block.h> // for definition of function cellindex_to_coord and coord_to_cellindex
#include <kalypsso/core/AMRMeshInfo.h>
#include <kalypsso/core/problems/DoubleMachReflectionParams.h>

#include <godunov_hydro/common.h>
#include <godunov_hydro/models/utils_hydro.h>

namespace kalypsso
{

namespace godunov_hydro
{

// =============================================================================
// =============================================================================
//! \struct BCDoubleMachReflection
//!
template <size_t dim>
struct BCDoubleMachReflection
{

  DoubleMachReflectionParams dmr_params;
  // conservative variables
  HydroState<dim> uL, uR;

  BCDoubleMachReflection(ConfigMap const & config_map)
    : dmr_params(config_map)
  {
    [[maybe_unused]] bool valid = true;
    const auto            eos_wrapper = EosWrapper<HostDevice>(config_map);
    const auto            settings = HydroSettings(config_map);

    // get primitive variables
    HydroState<dim> qL, qR;
    qL[Hydro<dim>::ID] = dmr_params.rhoL;
    qL[Hydro<dim>::IU] = dmr_params.uL;
    qL[Hydro<dim>::IV] = dmr_params.vL;
    if constexpr (dim == 3)
      qL[Hydro<dim>::IW] = ZERO_F;
    qL[Hydro<dim>::IP] = dmr_params.pL;

    qR[Hydro<dim>::ID] = dmr_params.rhoR;
    qR[Hydro<dim>::IU] = dmr_params.uR;
    qR[Hydro<dim>::IV] = dmr_params.vR;
    if constexpr (dim == 3)
      qR[Hydro<dim>::IW] = ZERO_F;
    qR[Hydro<dim>::IP] = dmr_params.pR;

    // get conservative variables
    uL = models::compute_conservative_variables<dim, HostDevice>(qL, settings, eos_wrapper, valid);
    uR = models::compute_conservative_variables<dim, HostDevice>(qR, settings, eos_wrapper, valid);
  }

  KOKKOS_FUNCTION bool
  needs_override(real_t x, real_t y) const
  {
    return (y >= 0) or (x < dmr_params.x0);
  }

  KOKKOS_FUNCTION bool
  needs_override(real_t x, real_t y, [[maybe_unused]] real_t z) const
  {
    return (y >= 0) or (x < dmr_params.x0);
  }

  //! return conservative variables - 2d case
  KOKKOS_FUNCTION real_t
  bc_state([[maybe_unused]] real_t x, [[maybe_unused]] real_t y, int var, bool is_left) const
  {
    if (var == Hydro<dim>::ID)
      return is_left ? uL[Hydro<dim>::ID] : uR[Hydro<dim>::ID];
    else if (var == Hydro<dim>::IE)
      return is_left ? uL[Hydro<dim>::IE] : uR[Hydro<dim>::IE];
    else if (var == Hydro<dim>::IU)
      return is_left ? uL[Hydro<dim>::IU] : uR[Hydro<dim>::IU];
    else if (var == Hydro<dim>::IV)
      return is_left ? uL[Hydro<dim>::IV] : uR[Hydro<dim>::IV];
    return ZERO_F;
  } // bc_state - 2d

  //! return conservative variables - 3d case
  KOKKOS_FUNCTION real_t
  bc_state([[maybe_unused]] real_t x,
           [[maybe_unused]] real_t y,
           [[maybe_unused]] real_t z,
           int                     var,
           bool                    is_left) const
  {
    if (var == Hydro<dim>::ID)
      return is_left ? uL[Hydro<dim>::ID] : uR[Hydro<dim>::ID];
    else if (var == Hydro<dim>::IE)
      return is_left ? uL[Hydro<dim>::IE] : uR[Hydro<dim>::IE];
    else if (var == Hydro<dim>::IU)
      return is_left ? uL[Hydro<dim>::IU] : uR[Hydro<dim>::IU];
    else if (var == Hydro<dim>::IV)
      return is_left ? uL[Hydro<dim>::IV] : uR[Hydro<dim>::IV];
    else if (var == Hydro<dim>::IW)
    {
      if constexpr (dim == 3)
      {
        return is_left ? uL[Hydro<dim>::IW] : uR[Hydro<dim>::IW];
      }
      else
      {
        return ZERO_F;
      }
    }
    return ZERO_F;
  } // operator() - 3d

}; // struct BCDoubleMachReflection

// ==============================================================================
// ==============================================================================
/**
 * \class FillOutsideDoubleMachReflection
 *
 * Override values in some outside cell.
 *
 *
 * --------------------------------
 * |               /               |
 * |    __________/____________    |
 * |    |        /            |    |
 * |    | shock /             |    |
 * | -> | left /     right    |    |
 * | -> |     /               |    |
 * |    |    /                |    |
 * |    |___/_________________|    |
 * |       /      wall        |    |
 * |______/___________________|____|
 *
 */
template <size_t dim, typename device_t>
class FillOutsideDoubleMachReflection
{
public:
  using exec_space = typename device_t::execution_space;
  using index_t = int32_t;

  using amr_hashmap_t = typename hashmap_base_t<device_t>::map_t;
  using orchard_key_view_t = typename orchard_key_base_t<device_t>::view_t;

  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  // ==============================================================
  // ==============================================================
  /** Fill outside octant cells.
   *
   * \param[in,out] userdata_out data array which we want to fill the block ghosts cells
   * \param[in] amr_info gives the number of owned, ghost, outside, ghost_outside quads
   * \param[in] orchard_keys array of orchard key ordered by Morton order
   * \param[in] amr_hashmap unordered map from orchard key to memory index for owned and ghost
   *            quadrants
   * \param[in] config_map application parameter map
   */
  static void
  apply(DataArrayBlock_t const &   userdata,
        AMRMeshInfo const &        amr_mesh_info,
        orchard_key_view_t const & orchard_keys,
        amr_hashmap_t const &      amr_hashmap,
        ConfigMap const &          config_map,
        real_t                     time);

  // ==============================================================
  // ==============================================================
  /**
   * range policy functor to sweep outside leaves.
   *
   * \see FillOutsideCellFunctor
   */
  KOKKOS_INLINE_FUNCTION void
  operator()(const index_t & i_global) const;

private:
  /**
   *
   * \param[in,out] userdata data array which we want to fill the block ghosts cells
   * \param[in] amr_meshinfo gives the number of owned, ghost, outside, ghost_outside quads
   * \param[in] orchard_keys array of orchard key ordered by Morton order
   * \param[in] amr_hashmap unordered map from orchard key to memory index for owned and ghost
   *            quadrants
   * \param[in] config_map application parameter map
   * \param[in] time
   *
   */
  FillOutsideDoubleMachReflection(DataArrayBlock_t const &   userdata,
                                  AMRMeshInfo const &        amr_mesh_info,
                                  orchard_key_view_t const & orchard_keys,
                                  amr_hashmap_t const &      amr_hashmap,
                                  ConfigMap const &          config_map,
                                  real_t                     time);

  //! a block data array (no ghosts, sizes= bx,by,bz)
  DataArrayBlock_t m_userdata;

  //! AMR mesh information (number of owned quadrants, number of ghosts quadrants, number of
  //! outside quads, number of outside ghosts, etc...)
  AMRMeshInfo m_amr_mesh_info;

  //! list of orchard key of the mesh
  orchard_key_view_t m_orchard_keys_device;

  //! AMR unordered map which maps orchard keys to quadrant number for all key in the mesh
  //! (owned quadrants and ghost quadrants)
  amr_hashmap_t m_amr_hashmap_device;

  //! p4est brick connectivity sizes
  const brick_size_t<dim> m_brick_size;

  //! foer each direction, state if mesh is periodic
  const Kokkos::Array<bool, dim> m_brick_periodicity;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  //! DoubleMachReflection parameters
  BCDoubleMachReflection<dim> m_bc_dmr;

  //! time
  real_t m_time;

}; // class FillOutsideCellFunctor

// explicit template instantiation
extern template class FillOutsideDoubleMachReflection<2, kalypsso::DefaultDevice>;
extern template class FillOutsideDoubleMachReflection<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_BC_DOUBLE_MACH_REFLECTION_H_
