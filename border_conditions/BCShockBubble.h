// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file BCShockBubble.h
 *
 * Define a border condition functor that can be used to simulate a high-speed inlet flow inside
 * the computational domain.
 *
 * This functor is aimed to be called after FillOutside, to override values at some location.
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_BC_SHOCK_BUBBLE_H_
#define KALYPSSO_GODUNOV_HYDRO_BC_SHOCK_BUBBLE_H_

#include <kalypsso/core/FillOutside_utils.h>

#include <kalypsso/core/kalypsso_data_container.h> // for DataArray, DataArrayHost, DataArrayGhostedBlock<dim, device_t>

#include <kalypsso/core/orchard_key.h>
#include <kalypsso/core/orchard_key_utils.h>
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/core/amr_hashmap.h>

#include <kalypsso/core/utils_block.h> // for definition of function cellindex_to_coord and coord_to_cellindex
#include <kalypsso/core/AMRMeshInfo.h>

#include <kalypsso/core/problems/ShockBubbleParams.h>

#include <kalypsso/core/models/Hydro.h>
#include <godunov_hydro/eos/EosWrapper.h>
#include <godunov_hydro/common.h>

namespace kalypsso
{

namespace godunov_hydro
{

// =============================================================================
// =============================================================================
//! \struct BCShockBubble
//!
//! Define a circular inlet on the left x face.
template <size_t dim>
struct BCShockBubble
{

  HydroState<dim> m_inflow;

  real_t m_xmin;

  BCShockBubble(ConfigMap const & config_map)
    : m_xmin(config_map.getReal("mesh", "xmin", KALYPSSO_NUM(0.0)))
  {
    auto eos_wrapper = eos::EosWrapper<HostDevice>(config_map);

    // inflow from the left (same as region 0)
    m_inflow = get_region_init_state<dim>(0, eos_wrapper, config_map);
  }

  KOKKOS_FUNCTION bool
  needs_override(real_t x, [[maybe_unused]] real_t y) const
  {
    return (x < m_xmin);
  }

  KOKKOS_FUNCTION bool
  needs_override(real_t x, [[maybe_unused]] real_t y, [[maybe_unused]] real_t z) const
  {
    return (x < m_xmin);
  }

  //! return conservative variables - 2d
  KOKKOS_FUNCTION real_t
  operator()([[maybe_unused]] real_t x, [[maybe_unused]] real_t y, int var) const
  {
    return m_inflow[var];
  } // operator() - 2d

  //! return conservative variables - 3d
  KOKKOS_FUNCTION real_t
  operator()([[maybe_unused]] real_t x,
             [[maybe_unused]] real_t y,
             [[maybe_unused]] real_t z,
             int                     var) const
  {
    return m_inflow[var];
  } // operator() - 3d

}; // struct BCShockBubble

// ==============================================================================
// ==============================================================================
/**
 * \class FillOutsideShockBubble
 *
 * Override values in some outside cell to have in inlet shock_bubble.
 *
 * In the drawing below, the arrow locates the inlet shock_bubble.
 *
 * --------------------------------
 * |              3               |
 * |______________________________|
 * |    |                    |    |
 * |    |                    |    |
 * | -> |                    | 2  |
 * | -> |                    |    |
 * |    |                    |    |
 * |____|____________________|____|
 * |              4               |
 * |______________________________|
 *
 */
template <size_t dim, typename device_t>
class FillOutsideShockBubble
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
        ConfigMap const &          config_map);

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
   * \param[in,out] userdata_out data array which we want to fill the block ghosts cells
   * \param[in] amr_info gives the number of owned, ghost, outside, ghost_outside quads
   * \param[in] orchard_keys array of orchard key ordered by Morton order
   * \param[in] amr_hashmap unordered map from orchard key to memory index for owned and ghost
   *            quadrants
   * \param[in] config_map application parameter map
   *
   */
  FillOutsideShockBubble(DataArrayBlock_t const &   userdata,
                         AMRMeshInfo const &        amr_mesh_info,
                         orchard_key_view_t const & orchard_keys,
                         amr_hashmap_t const &      amr_hashmap,
                         ConfigMap const &          config_map);

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

  //! for each direction, state if mesh is periodic
  const Kokkos::Array<bool, dim> m_brick_periodicity;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  //! ShockBubble parameters
  BCShockBubble<dim> m_bc_shock_bubble;

}; // class FillOutsideCellFunctor

// explicit template instantiation
extern template class FillOutsideShockBubble<2, kalypsso::DefaultDevice>;
extern template class FillOutsideShockBubble<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_BC_SHOCK_BUBBLE_H_
