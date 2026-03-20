// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file InitBlast.h
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_INIT_BLAST_H_
#define KALYPSSO_GODUNOV_HYDRO_INIT_BLAST_H_

#include <godunov_hydro/common.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>
#include <kalypsso/core/problems/BlastParams.h>
#include <kalypsso/core/orchard_key_utils.h>

namespace kalypsso
{

namespace godunov_hydro
{

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * \class InitBlastDataFunctor
 * \brief Implement user data initialization to solve blast problem.
 *
 * See http://www.astro.princeton.edu/~jstone/Athena/tests/blast/blast.html
 *
 * This functor takes as input a mesh, already refined (after companion functor
 * InitBlastRefineFunctor), and initializes user data on host.
 * Copying data from host to device, should be done outside.
 *
 * Initial conditions is refined near strong density gradients.
 *
 * \sa InitBlastRefineFunctor
 */
template <size_t dim, typename device_t>
class InitBlastDataFunctor
{

public:
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

  struct TagInitData
  {};
  struct TagRescaleEnergy
  {};

private:
  //! conservative variables to be initialized
  DataArrayBlock_t m_Udata;

  //! field manager
  FieldMap<core::models::Hydro> m_fm;

  //! list of orchard key of the mesh
  orchard_key_view_t<device_t> m_orchard_keys;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! general parameters (used on device)
  HydroSettings m_settings;

  //! Blast problem specific parameters (used on device)
  BlastParams m_bParams;

  //! Initial states (one per region, conservative variables)
  InitialStates<dim, device_t> m_initial_states;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  //! should we replicate initial condition in each p4est tree
  bool m_replicated_init_cond;

  //! total volume inside initial blast
  real_t m_total_volume_inside;

  InitBlastDataFunctor(DataArrayBlock_t const &             Udata,
                       FieldMap<core::models::Hydro>        fm,
                       orchard_key_view_t<device_t> const & orchard_keys,
                       int32_t                              local_num_octants,
                       HydroSettings const &                settings,
                       BlastParams const &                  bParams,
                       InitialStates<dim, device_t> const & initial_states,
                       bool                                 replicated_init_cond,
                       real_t                               total_volume_inside,
                       ConfigMap const &                    config_map);

public:
  //! static method which does it all: create and execute functor
  static auto
  apply(DataArrayBlock_t const &             Udata,
        FieldMap<core::models::Hydro>        fm,
        orchard_key_view_t<device_t> const & orchard_keys,
        int32_t                              local_num_octants,
        HydroSettings const &                settings,
        InitialStates<dim, device_t> const & initial_states,
        bool                                 replicated_init_cond,
        ConfigMap const &                    config_map,
        ParallelEnv const &                  par_env);

  static void
  rescale_energy(DataArrayBlock_t const &             Udata,
                 FieldMap<core::models::Hydro>        fm,
                 orchard_key_view_t<device_t> const & orchard_keys,
                 int32_t                              local_num_octants,
                 HydroSettings const &                settings,
                 InitialStates<dim, device_t> const & initial_states,
                 bool                                 replicated_init_cond,
                 ConfigMap const &                    config_map,
                 ParallelEnv const &                  par_env,
                 real_t                               total_volume_inside);

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagInitData const &, const int32_t & global_index, real_t & volume) const;

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagRescaleEnergy const &, const int32_t & global_index) const;

private:
  // ==========================================================================
  // ==========================================================================
  /**
   * Utility to determine in which region a given point is.
   */
  KOKKOS_INLINE_FUNCTION int
  point_to_region(Kokkos::Array<real_t, dim> const & xyz) const
  {
    auto const & blast_radius = m_bParams.blast_radius;
    auto const & blast_center_x = m_bParams.blast_center_x;
    auto const & blast_center_y = m_bParams.blast_center_y;
    auto const & blast_center_z = m_bParams.blast_center_z;

    // see if point is inside of the blast region
    auto r2 = (blast_center_x - xyz[IX]) * (blast_center_x - xyz[IX]) +
              (blast_center_y - xyz[IY]) * (blast_center_y - xyz[IY]);
    if constexpr (dim == 3)
      r2 += (blast_center_z - xyz[IZ]) * (blast_center_z - xyz[IZ]);

    if (r2 < blast_radius * blast_radius)
      return 0;

    // point can only be in pre-shock region
    return 1;

  } // point_to_region

  // ==========================================================================
  // ==========================================================================
  /**
   * For each corner of a given cell, determine in which region this corner is.
   *
   * \param[in] ijk i-j-k indexes of a cell inside a block of cells
   * \param[in] key orchard key of current octant
   * \param[in] block_sizes number of cells per direction
   * \param[in] replicated_init_cond
   * \param[out] regions array of regions (one per corner)
   */
  KOKKOS_INLINE_FUNCTION void
  compute_corner_to_region(coord_t<dim> const &                             ijk,
                           key_t const &                                    key,
                           block_size_t<dim> const &                        block_sizes,
                           bool const &                                     replicated_init_cond,
                           Kokkos::Array<int, Corner::num_corners<dim>()> & regions) const
  {

    for (uint8_t i_corner = 0; i_corner < Corner::num_corners<dim>(); i_corner++)
    {
      const auto xyz_vertex_corner =
        orchard_key_to_corner_coord<dim>(key, ijk, block_sizes[IX], i_corner);
      auto xyz_corner =
        vertex_coord_to_real_space<dim>(xyz_vertex_corner, m_scaling_factor, m_xyz_min);

      if (replicated_init_cond)
      {
        const auto tree_coords = orchard_key_t<dim>::get_tree_coords(key);
        xyz_corner[IX] -= tree_coords[IX] * m_scaling_factor;
        xyz_corner[IY] -= tree_coords[IY] * m_scaling_factor;
        if constexpr (dim == 3)
        {
          xyz_corner[IZ] -= tree_coords[IZ] * m_scaling_factor;
        }
      }

      regions[i_corner] = point_to_region(xyz_corner);
    }
  }

}; // InitBlastDataFunctor

extern template class InitBlastDataFunctor<2, kalypsso::DefaultDevice>;
extern template class InitBlastDataFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * Implement initial refinement to solve blast problem.
 *
 * See http://www.astro.princeton.edu/~jstone/Athena/tests/blast/blast.html
 *
 * This functor only performs mesh refinement, no user data init.
 * User data init is actually done in InitBlastDataFunctor
 *
 * Initial conditions is refined near initial density gradients.
 *
 * \sa InitBlastDataFunctor
 *
 */
template <size_t dim, typename device_t>
class InitBlastRefineFunctor
{
public:
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  //! our kokkos execution space
  using exec_space = typename device_t::execution_space;

  //! type alias for a (device) Kokkos view of refinement flags
  using amrflags_view_t = typename AMRContext<dim, device_t>::amrflags_view_t;

  struct TagRefineAlways
  {};
  struct TagRefineGeometric
  {};

private:
  //! conservative variables
  DataArrayBlock_t m_Udata;

  //! field manager
  FieldMap<core::models::Hydro> m_fm;

  //! list of orchard key of the mesh
  orchard_key_view_t<device_t> m_orchard_keys;

  //! refinement flags (to be filled)
  amrflags_view_t m_amrflags;

  //! number of octants in the new mesh
  const int32_t m_local_num_octants;

  //! general parameters (used on device)
  HydroSettings m_settings;

  //! Blast problem specific parameters (used on device)
  BlastParams m_bParams;

  //! which level should we look at
  int m_level_refine;

  //! get geometrical scaling factor
  const real_t m_scaling_factor;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  //! should we replicate initial condition in each p4est tree
  bool m_replicated_init_cond;

  // ===========================================================
  // ===========================================================
  InitBlastRefineFunctor(DataArrayBlock_t const &             Udata,
                         FieldMap<core::models::Hydro>        fm,
                         orchard_key_view_t<device_t> const & orchard_keys,
                         amrflags_view_t const &              amrflags,
                         int32_t                              local_num_octants,
                         HydroSettings const &                settings,
                         BlastParams const &                  bParams,
                         int                                  level_refine,
                         bool                                 replicated_init_cond,
                         ConfigMap const &                    config_map);

public:
  // ===========================================================
  // ===========================================================
  // static method which does it all: create and execute functor
  static void
  apply(DataArrayBlock_t const &             Udata,
        FieldMap<core::models::Hydro>        fm,
        orchard_key_view_t<device_t> const & orchard_keys,
        amrflags_view_t const &              amrflags,
        int32_t                              local_num_octants,
        HydroSettings const &                settings,
        int                                  level_refine,
        bool                                 replicated_init_cond,
        ConfigMap const &                    config_map);

  // ===========================================================
  // ===========================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagRefineAlways const &, const size_t & iOct) const;

  // ===========================================================
  // ===========================================================
  KOKKOS_INLINE_FUNCTION
  void
  operator()(TagRefineGeometric const &, const size_t & iOct) const;

}; // InitBlastRefineFunctor

extern template class InitBlastRefineFunctor<2, kalypsso::DefaultDevice>;
extern template class InitBlastRefineFunctor<3, kalypsso::DefaultDevice>;

// ====================================================================
// ====================================================================
// ====================================================================
/**
 * \class InitBlast
 * \brief Hydrodynamical blast Test.
 *
 * Reference: http://www.astro.princeton.edu/~jstone/Athena/tests/blast/blast.html
 *
 * Initial condition is mostly done on host, the final refined initial
 * condition data are uploaded to kokkos device.
 *
 * Different initial refinement strategies are possible:
 * - no refinement at all
 * - geometric refinement, i.e. refine near interface
 * - regular gradient based refinement
 *
 * This is controlled input parameter in ini file : "amr"/"init_condition_refine_criterion"
 */
template <size_t dim, typename device_t>
class InitBlast
{
public:
  static void
  apply(SolverGodunovHydro<dim, device_t> & solver);
};

extern template class InitBlast<2, kalypsso::DefaultDevice>;
extern template class InitBlast<3, kalypsso::DefaultDevice>;

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_INIT_BLAST_H_
