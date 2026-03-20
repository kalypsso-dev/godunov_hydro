// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeRadialProfile.h
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_COMPUTE_RADIAL_PROFILE_FUNCTOR_H_
#define KALYPSSO_GODUNOV_HYDRO_COMPUTE_RADIAL_PROFILE_FUNCTOR_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock
#include <kalypsso/core/FieldMap.h>
#include <kalypsso/core/orchard_key_base.h>
#include <kalypsso/utils/mpi/ParallelEnv.h>

// hydro utils (conservative versus primitive variable, equation of state, ...)
#include <kalypsso/core/models/HydroState.h>
#include <kalypsso/core/models/utils_hydro.h>
#include <kalypsso/core/utils_block.h>

// other utilities
#ifdef KALYPSSO_CORE_USE_CNPY
#  include <kalypsso/core/cnpy_io.h>
#endif

namespace kalypsso
{
namespace godunov_hydro
{

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Compute radial profile by averaging all direction.
 *
 * This functor is mainly aimed at checking numerical and analytical solution of the radial Sedov
 * blast.
 *
 * \tparam dim is space dimension (2 or 3)
 * \tparam device_t is the Kokkos device use for computation (CPU, GPU, ...)
 */
template <size_t dim, typename device_t>
class ComputeRadialProfileFunctor
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

  //! field manager
  const FieldMap<core::models::Hydro> m_fm;

  //! block sizes
  const block_size_t<dim> m_block_sizes;

  //! number of cells per leaf
  const int32_t m_nbCellsPerLeaf;

  //! get domain lower left corner
  const Kokkos::Array<real_t, dim> m_xyz_min;

  // get geometrical scaling factor
  const real_t m_scaling_factor;

  //! heavy data - conservative variables
  DataArrayBlock_t m_Udata;

  //! number of bins in radial direction
  const int m_nbins;

  //! max radial distance
  const real_t m_max_radial_distance;

  //! center of the box
  Kokkos::Array<real_t, dim> m_box_center;

public:
  //! radial distance histogram
  Kokkos::View<int *, device_t, Kokkos::MemoryTraits<Kokkos::Atomic>> m_radial_distance_histo;

  //! averaged density radial profile
  Kokkos::View<real_t *, device_t, Kokkos::MemoryTraits<Kokkos::Atomic>> m_density_profile;

public:
  ComputeRadialProfileFunctor(ConfigMap const &             config_map,
                              orchard_key_view_t            orchard_keys,
                              int32_t                       local_num_octants,
                              FieldMap<core::models::Hydro> fm,
                              block_size_t<dim>             block_sizes,
                              DataArrayBlock_t              Udata,
                              real_t                        max_radial_distance,
                              Kokkos::Array<real_t, dim>    box_center)
    : m_orchard_keys(orchard_keys)
    , m_local_num_octants(local_num_octants)
    , m_fm(fm)
    , m_block_sizes(block_sizes)
    , m_nbCellsPerLeaf(Udata.num_cells())
    , m_xyz_min(get_xyz_min<dim>(config_map))
    , m_scaling_factor(get_scaling_factor(config_map))
    , m_Udata(Udata)
    , m_nbins(config_map.getInteger("blast", "nbins", 1000))
    , m_max_radial_distance(max_radial_distance)
    , m_box_center(box_center)
    , m_radial_distance_histo("radial_distance_histo", static_cast<size_t>(m_nbins))
    , m_density_profile("density_profile", static_cast<size_t>(m_nbins))
  {
    Kokkos::deep_copy(m_radial_distance_histo, 0);
    Kokkos::deep_copy(m_density_profile, ZERO_F);
  };

  // ====================================================================
  // ====================================================================
  //! static method which does it all: create and execute functor using range policy
  //!
  //! \param[in] orchard_keys is a vector of all local (owned+ghost) octant orchard/morton keys
  //! \param[in] local_num_octants is the number of octants owned by current MPI process (ghost
  //!            excluded)
  //! \param[in] fm is the field map (TODO refactor this)
  //! \param[in] block_sizes is an array the cartesian block sizes
  //! \param[in,out] invDt is the inverse of time step, the output of this functor
  //!
  static void
  apply([[maybe_unused]] ParallelEnv const & par_env,
        ConfigMap const &                    config_map,
        orchard_key_view_t                   orchard_keys,
        int32_t                              local_num_octants,
        FieldMap<core::models::Hydro>        fm,
        block_size_t<dim>                    block_sizes,
        DataArrayBlock_t                     Udata)
  {

    // get center of the box coordinates
    Kokkos::Array<real_t, dim> box_center;

    const auto xmin = config_map.getReal("mesh", "xmin", KALYPSSO_NUM(0.0));
    const auto ymin = config_map.getReal("mesh", "ymin", KALYPSSO_NUM(0.0));
    const auto zmin = config_map.getReal("mesh", "zmin", KALYPSSO_NUM(0.0));

    const auto nbrick_x = config_map.getInteger("p4est_connectivity", "nbrick_x", 1);
    const auto nbrick_y = config_map.getInteger("p4est_connectivity", "nbrick_y", 1);
    const auto nbrick_z = config_map.getInteger("p4est_connectivity", "nbrick_z", 1);

    const auto scaling_factor = config_map.getReal("mesh", "scaling_factor", KALYPSSO_NUM(1.0));

    const auto xmax = xmin + static_cast<real_t>(nbrick_x) * scaling_factor;
    const auto ymax = ymin + static_cast<real_t>(nbrick_y) * scaling_factor;
    const auto zmax = zmin + static_cast<real_t>(nbrick_z) * scaling_factor;

    box_center[IX] = (xmin + xmax) / 2;
    box_center[IY] = (ymin + ymax) / 2;
    if constexpr (dim == 3)
    {
      box_center[IZ] = (zmin + zmax) / 2;
    }

    // compute maximum radial distance
    const auto dx = (xmax - xmin) / 2;
    const auto dy = (ymax - ymin) / 2;
    const auto dz = (zmax - zmin) / 2;
    const auto max_radial_distance =
      dim == 2 ? sqrt(dx * dx + dy * dy) : sqrt(dx * dx + dy * dy + dz * dz);

    ComputeRadialProfileFunctor functor(config_map,
                                        orchard_keys,
                                        local_num_octants,
                                        fm,
                                        block_sizes,
                                        Udata,
                                        max_radial_distance,
                                        box_center);

    const auto nbCellsPerLeaf = Udata.num_cells();

    // compute total number of cells
    const auto nbCellsTotal = local_num_octants * nbCellsPerLeaf;

    Kokkos::parallel_for("kalypsso::godunov_hydro::ComputeRadialProfileFunctor",
                         Kokkos::RangePolicy<exec_space>(0, nbCellsTotal),
                         functor);

    // once the radial profile is computed, gather all the MPI pieces
    const auto nbins = config_map.getInteger("blast", "nbins", 1000);
    auto       total_radial_distance_histo =
      Kokkos::View<int *, device_t>("total_radial_distance_histo", static_cast<size_t>(nbins));

    const auto total_density_profile =
      Kokkos::View<real_t *, device_t>("total_density_profile", static_cast<size_t>(nbins));

#ifdef KALYPSSO_CORE_USE_MPI
    par_env.comm().MPI_Reduce<MpiComm::SUM>(
      functor.m_radial_distance_histo.data(), total_radial_distance_histo.data(), nbins, 0);

    par_env.comm().MPI_Reduce<MpiComm::SUM>(
      functor.m_density_profile.data(), total_density_profile.data(), nbins, 0);
#else
    Kokkos::deep_copy(total_radial_distance_histo, functor.m_radial_distance_histo);
    Kokkos::deep_copy(total_density_profile, functor.m_density_profile);
#endif // KALYPSSO_CORE_USE_MPI

    // then we compute the profile and output in numpy format
    auto rad_dist =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, total_radial_distance_histo);
    auto dens_prof =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, total_density_profile);

    auto distances =
      Kokkos::View<real_t *, Kokkos::HostSpace>("distances", static_cast<size_t>(nbins));
    const auto dr = max_radial_distance / static_cast<real_t>(nbins);

    for (int i = 0; i < nbins; ++i)
    {
      distances(i) = (static_cast<real_t>(i) + KALYPSSO_NUM(0.5)) * dr;
      dens_prof(i) /= static_cast<real_t>(rad_dist(i));
    }

    // now we can output results
#ifdef KALYPSSO_CORE_USE_CNPY
    save_cnpy(distances, "sedov_blast_radial_distances");
    save_cnpy(dens_prof, "sedov_blast_density_profile");
#else
    if (par_env.rank() == 0)
    {
      KALYPSSO_ERROR(
        "Please rebuild kalypsso-core by enabling cnpy library: KALYPSSO_CORE_BUILD_CNPY=ON.\n");
    }
#endif

  } // apply

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  real_t
  compute_distance(index_t iOct, index_t cell_index) const
  { // compute ix,iy,iz of local cell inside
    // block from index
    const auto iCoord = cellindex_to_coord<dim>(cell_index, m_block_sizes);

    // get block orchard key
    const auto key = m_orchard_keys(iOct);

    // compute physical x,y,z for that cell (cell center)
    const auto xyz_vertex = orchard_key_to_cell_coord<dim>(key, iCoord, m_block_sizes[IX]);
    const auto xyz = vertex_coord_to_real_space<dim>(xyz_vertex, m_scaling_factor, m_xyz_min);

    const auto distance = dim == 2
                            ? sqrt((xyz[IX] - m_box_center[IX]) * (xyz[IX] - m_box_center[IX]) +
                                   (xyz[IY] - m_box_center[IY]) * (xyz[IY] - m_box_center[IY]))
                            : sqrt((xyz[IX] - m_box_center[IX]) * (xyz[IX] - m_box_center[IX]) +
                                   (xyz[IY] - m_box_center[IY]) * (xyz[IY] - m_box_center[IY]) +
                                   (xyz[IZ] - m_box_center[IZ]) * (xyz[IZ] - m_box_center[IZ]));

    return distance;
  }

  // ====================================================================
  // ====================================================================
  /**
   * range policy functor for computing CFL condition.
   *
   * \param[in] global_index spans range from 0 to nbCellsPerLeaf * local_num_octants-1
   *            (i.e. total number of cells in current MPI process)
   * \param[in,out] invDt is the reduced variable to update
   */
  KOKKOS_INLINE_FUNCTION
  void
  operator()(const index_t & global_index) const
  {
    // convert global index into
    // - octant id
    // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
    const auto iOct = global_index / m_nbCellsPerLeaf;
    const auto cell_index = global_index - iOct * m_nbCellsPerLeaf;

    const auto distance = compute_distance(iOct, cell_index);

    // compute bin number from distance to center of the box
    const auto bin =
      static_cast<int>(distance / m_max_radial_distance * static_cast<real_t>(m_nbins));

    // update density profile
    m_radial_distance_histo(bin) += 1;

    // update density profile
    constexpr auto ID = core::models::Hydro::ID;

    m_density_profile(bin) += m_Udata(cell_index, m_fm[ID], iOct);
  } // operator()

}; // ComputeRadialProfileFunctor

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_COMPUTE_RADIAL_PROFILE_FUNCTOR_H_
