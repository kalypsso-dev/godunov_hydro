// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file FillOutside.cpp
 *
 * Implement border conditions (other than periodic) for Hydrodynamics.
 */

#include <godunov_hydro/border_conditions/FillOutside.h>

#include <kalypsso/core/MeshMap.h> // key_to_value

namespace kalypsso
{

namespace godunov_hydro
{

// ==============================================================================
// ==============================================================================
template <size_t dim, typename device_t>
FillOutsideCellFunctor<dim, device_t>::FillOutsideCellFunctor(
  DataArrayBlock_t const &   userdata,
  AMRMeshInfo const &        amr_mesh_info,
  orchard_key_view_t const & orchard_keys,
  amr_hashmap_t const &      amr_hashmap,
  ConfigMap const &          config_map,
  ParallelEnv const &        par_env)
  : m_userdata(userdata)
  , m_amr_mesh_info(amr_mesh_info)
  , m_orchard_keys_device(orchard_keys)
  , m_amr_hashmap_device(amr_hashmap)
  , m_brick_size(get_brick_sizes<dim>(config_map))
  , m_brick_periodicity(get_brick_periodicity<dim>(config_map))
  , m_bc_types(BorderConditionsConfig<BC_HYDRO>::read_border_condition<dim>(config_map, par_env))
{}

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
void
FillOutsideCellFunctor<dim, device_t>::apply(DataArrayBlock_t const &   userdata,
                                             AMRMeshInfo const &        amr_mesh_info,
                                             orchard_key_view_t const & orchard_keys,
                                             amr_hashmap_t const &      amr_hashmap,
                                             ConfigMap const &          config_map,
                                             ParallelEnv const &        par_env)
{

  // create compute functor
  FillOutsideCellFunctor<dim, device_t> functor(
    userdata, amr_mesh_info, orchard_keys, amr_hashmap, config_map, par_env);

  const int32_t start_octant = amr_mesh_info.first_outside_quad_local_id();
  const int32_t end_octant = start_octant + amr_mesh_info.total_local_number_of_outside_quads();

  const int32_t nb_cells_per_leaf = userdata.num_cells();
  const int32_t start = start_octant * nb_cells_per_leaf;
  const int32_t end = end_octant * nb_cells_per_leaf;

  Kokkos::parallel_for(
    "FillOutsideCellFunctor", Kokkos::RangePolicy<exec_space>(start, end), functor);

} // apply

// ==============================================================
// ==============================================================
template <size_t dim, typename device_t>
KOKKOS_INLINE_FUNCTION void
FillOutsideCellFunctor<dim, device_t>::operator()(const index_t & i_global) const
{

  const auto block_size = m_userdata.block_size();
  const auto num_cells = m_userdata.num_cells();

  // i_oct_outside, by design, is associated to an outside quadrant
  const auto i_oct_outside = i_global / num_cells;

  const auto i_cell_outside = i_global - num_cells * i_oct_outside;
  const auto coord_out = cellindex_to_coord<dim>(i_cell_outside, block_size);

  // get orchard key corresponding to visited outside quadrants
  const auto key_outside = m_orchard_keys_device(i_oct_outside);

  // when computing inside quadrant key, we always use a "virtual" key computed as the
  // periodic image of the outside quadrant; so here we need is_periodic to be array of
  // "true"
  constexpr auto is_periodic = get_bool_array<dim>(true);

  // quadrant is at domain border, compute outside normal at the periodic image of key_outside
  auto outside_normal =
    orchard_key_t<dim>::get_outside_normal(key_outside, m_brick_size, is_periodic);

  outside_normal[IX] *= orchard_key_t<dim>::is_touching_face_X(key_outside);
  outside_normal[IY] *= orchard_key_t<dim>::is_touching_face_Y(key_outside);
  if constexpr (dim == 3)
  {
    outside_normal[IZ] *= orchard_key_t<dim>::is_touching_face_Z(key_outside);
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
  KOKKOS_ASSERT(is_valid_key && "[FillOutsideCellFunctor] key_inside does not exist in hashmap !?");

  const auto i_oct_inside = key_status.second;

  // make sure i_oct_inside is actually inside
  KOKKOS_ASSERT(
    i_oct_inside < m_amr_mesh_info.first_outside_quad_local_id() &&
    "[FillOutsideCellFunctor] i_oct_inside does not identify an octant inside domain?!");

  //
  // now we can fill outside userdata according to border condition
  //

  // get list of faces
  const auto faces = Face::get_all_faces<dim>();

  for (const auto face : faces)
  {
    // normal direction
    const Dir::dir_t dir = face / 2;

    if (orchard_key_t<dim>::is_at_domain_border(key_inside, face, m_brick_size) and
        orchard_key_t<dim>::is_outside_dir(key_outside, dir))
    {
      if (m_bc_types[face]._to_integral() == +BC_HYDRO::ZERO_GRADIENT)
      {
        auto coord_in = coord_out;
        coord_in[dir] = Face::is_left_face(face) ? 0 : block_size[dir] - 1;
        auto i_cell_inside = coord_to_cellindex<dim>(coord_in, block_size);

        for (int32_t ivar = 0; ivar < m_userdata.num_vars(); ++ivar)
        {
          m_userdata(i_cell_outside, ivar, i_oct_outside) =
            m_userdata(i_cell_inside, ivar, i_oct_inside);
        }
      }
      else if (m_bc_types[face]._to_integral() == +BC_HYDRO::WALL)
      {
        auto coord_in = coord_out;
        coord_in[dir] = block_size[dir] - 1 - coord_out[dir];
        auto i_cell_inside = coord_to_cellindex<dim>(coord_in, block_size);

        // copy data and negate normal velocity
        for (int32_t ivar = 0; ivar < m_userdata.num_vars(); ++ivar)
        {
          m_userdata(i_cell_outside, ivar, i_oct_outside) =
            m_userdata(i_cell_inside, ivar, i_oct_inside);
        }

        m_userdata(i_cell_outside, Hydro::IU + dir, i_oct_outside) *= -1;

      } // end BC_HYDRO::WALL
    } // end if is_at_domain_border
  } // for faces

} // operator()

// explicit template instantiation
template class FillOutsideCellFunctor<2, kalypsso::DefaultDevice>;
template class FillOutsideCellFunctor<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso
