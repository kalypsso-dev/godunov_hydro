// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file BCJet.cpp
 *
 * Implement inlet jet border conditions.
 */

#include <godunov_hydro/border_conditions/BCJet.h>

#include <kalypsso/core/MeshMap.h> // key_to_value

namespace kalypsso
{

namespace godunov_hydro
{

// ==============================================================================
// ==============================================================================
template <size_t dim, typename device_t>
FillOutsideJet<dim, device_t>::FillOutsideJet(DataArrayBlock_t const &   userdata,
                                              AMRMeshInfo const &        amr_mesh_info,
                                              orchard_key_view_t const & orchard_keys,
                                              amr_hashmap_t const &      amr_hashmap,
                                              ConfigMap const &          config_map)
  : m_userdata(userdata)
  , m_amr_mesh_info(amr_mesh_info)
  , m_orchard_keys_device(orchard_keys)
  , m_amr_hashmap_device(amr_hashmap)
  , m_brick_size(get_brick_sizes<dim>(config_map))
  , m_brick_periodicity(get_brick_periodicity<dim>(config_map))
  , m_scaling_factor(get_scaling_factor(config_map))
  , m_xyz_min(get_xyz_min<dim>(config_map))
  , m_bc_jet(config_map)
{}

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
FillOutsideJet<dim, device_t>::apply(DataArrayBlock_t const &   userdata,
                                     AMRMeshInfo const &        amr_mesh_info,
                                     orchard_key_view_t const & orchard_keys,
                                     amr_hashmap_t const &      amr_hashmap,
                                     ConfigMap const &          config_map)
{

  // create compute functor
  FillOutsideJet<dim, device_t> functor(
    userdata, amr_mesh_info, orchard_keys, amr_hashmap, config_map);

  const int32_t start_octant = amr_mesh_info.first_outside_quad_local_id();
  const int32_t end_octant = start_octant + amr_mesh_info.total_local_number_of_outside_quads();

  const int32_t nb_cells_per_leaf = userdata.num_cells();
  const int32_t start = start_octant * nb_cells_per_leaf;
  const int32_t end = end_octant * nb_cells_per_leaf;

  Kokkos::parallel_for("FillOutsideJet", Kokkos::RangePolicy<exec_space>(start, end), functor);

} // apply

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
FillOutsideJet<dim, device_t>::operator()(const index_t & i_global) const
{

  const auto block_size = m_userdata.block_size();
  const auto num_cells = m_userdata.num_cells();

  // iOct_local, by design, is associated to an outside quadrant
  const auto i_oct_outside = i_global / num_cells;

  const auto cellindex_out = i_global - num_cells * i_oct_outside;
  const auto coord_out = cellindex_to_coord<dim>(cellindex_out, block_size);

  // get orchard key corresponding to visited outside quadrants
  const auto key_outside = m_orchard_keys_device(i_oct_outside);

  // when computing inside quadrant key, we always use a "virtual" key computed as the
  // periodic image of the outside quadrant; so here we need is_periodic to be array of
  // "true"
  constexpr auto is_periodic = get_bool_array<dim>(true);

  // quadrant is at domain border, compute outside normal at the periodic image of key_outside
  auto outside_normal =
    orchard_key_t<dim>::get_outside_normal(key_outside, m_brick_size, is_periodic);

  outside_normal[IX] = outside_normal[IX] * orchard_key_t<dim>::is_touching_face_X(key_outside);
  outside_normal[IY] = outside_normal[IY] * orchard_key_t<dim>::is_touching_face_Y(key_outside);
  if constexpr (dim == 3)
  {
    outside_normal[IZ] = outside_normal[IZ] * orchard_key_t<dim>::is_touching_face_Z(key_outside);
  }

  // compute corresponding inside quadrant (just across external border, i.e. along outside
  // normal, but opposite direction)
  auto key_inside = orchard_key_t<dim>::get_neighbor_key_same_level(
    key_outside, outside_normal, m_brick_size, m_brick_periodicity);
  orchard_key_t<dim>::reset_outside_bits(key_inside);

  // get i_oct_inside from the unordered map
  const auto key_status = key_to_value(key_inside, m_amr_hashmap_device);

  [[maybe_unused]] auto const & is_valid_key = key_status.first;

  // make sure key_inside actually exist in map
  // if it doesn't we have a serious logical problem
  KOKKOS_ASSERT(is_valid_key && "[FillOutsideJet] key_inside does not exist in hashmap !?");

  [[maybe_unused]] const auto i_oct_inside = key_status.second;

  // make sure i_oct_inside is actually inside
  KOKKOS_ASSERT(i_oct_inside < m_amr_mesh_info.first_outside_quad_local_id() &&
                "[FillOutsideJet] i_oct_inside does not identify an octant inside domain ?!");

  //
  // now we can fill outside userdata according to border condition
  //

  if (orchard_key_t<dim>::is_at_any_domain_border(key_inside, m_brick_size))
  {
    const auto xyz_corner = outside_key_to_vertex_coord<dim>(key_outside, false, m_brick_size);

    const auto xyz_cell_vertex = compute_cell_coordinates<dim>(
      orchard_key_t<dim>::level(key_outside), xyz_corner, coord_out, block_size);

    const auto xyz_cell =
      vertex_coord_to_real_space<dim>(xyz_cell_vertex, m_scaling_factor, m_xyz_min);


    if constexpr (dim == 2)
    {
      auto const & x = xyz_cell[IX];
      auto const & y = xyz_cell[IY];

      if (m_bc_jet.needs_override(x, y))
      {
        for (int32_t ivar = 0; ivar < m_userdata.num_vars(); ++ivar)
        {
          m_userdata(cellindex_out, ivar, i_oct_outside) =
            m_bc_jet(xyz_cell[IX], xyz_cell[IY], ivar);
        }
      }
    }
    else if constexpr (dim == 3)
    {
      auto const & x = xyz_cell[IX];
      auto const & y = xyz_cell[IY];
      auto const & z = xyz_cell[IZ];

      if (m_bc_jet.needs_override(x, y, z))
      {
        for (int32_t ivar = 0; ivar < m_userdata.num_vars(); ++ivar)
        {
          m_userdata(cellindex_out, ivar, i_oct_outside) =
            m_bc_jet(xyz_cell[IX], xyz_cell[IY], xyz_cell[IZ], ivar);
        }
      }
    }
  } // end if is_at_domain_border

} // operator()

// explicit template instantiation
template class FillOutsideJet<2, kalypsso::DefaultDevice>;
template class FillOutsideJet<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso
