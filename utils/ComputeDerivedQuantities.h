// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file ComputeDerivedQuantities.h
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_COMPUTE_DERIVED_QUANTITIES_H_
#define KALYPSSO_GODUNOV_HYDRO_COMPUTE_DERIVED_QUANTITIES_H_

#include <kalypsso/core/kalypsso_core_base.h> // for assertm
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/kalypsso_data_container.h> // for DataArrayBlock

#include <kalypsso/core/FieldMap.h>
#include <kalypsso/core/models/HydroState.h>
#include <godunov_hydro/models/utils_hydro.h> // for primitive/conservative variables conversion

#include <kalypsso/utils/mpi/ParallelEnv.h>

namespace kalypsso
{

namespace godunov_hydro
{

/**
 * Define a better enum to list supported derived quantities.
 *
 * Derived quantities are quantities that are not solved in the PDE systems but can be
 * deduced from conservative variables.
 *
 * Derived quantities can be scalar or vector valued.
 *
 * - thermal pressure
 * - specific kinetic energy
 * - local Mach number : M =|u|/c where c is local speed of sound
 */
// clang-format off
BETTER_ENUM(DERIVED_QUANTITY, uint32_t,
            THERMAL_PRESSURE,
            SPECIFIC_EKIN,
            LOCAL_MACH_NUMBER
  )
// clang-format on

// ========================================================
// ========================================================
// ========================================================
/**
 * A simple helper structure to compute a derived quantity (thermal pressure, specific kinetic
 * energy, ...)
 */
template <size_t dim, typename device_t>
struct ComputeDerivedQuantities
{

  //! type alias for cell-centered data array at block level (see kalypsso_data_container.h)
  using DataArrayBlock_t = DataArrayBlock<dim, real_t, device_t>;

  //! our kokkos execution space
  using ExecutionSpace = typename device_t::execution_space;

  //! makes enum Hydro::VarId available
  using Hydro = kalypsso::core::models::Hydro;

  // ==========================================================================
  // ==========================================================================
  static void
  check_args_validity(DataArrayBlock_t const & Udata,
                      int64_t const &          iOct_begin,
                      int64_t const &          num_octs)
  {

    if (iOct_begin < 0 or iOct_begin >= Udata.num_quadrants())
    {
      Kokkos::abort("[ComputeDerivedQuantities::run] : invalid argument for iOct_begin.");
    }
    if (iOct_begin + num_octs < 0 or (iOct_begin + num_octs) > Udata.num_quadrants())
    {
      Kokkos::abort("[ComputeDerivedQuantities::run] : invalid argument for num_octs.");
    }

  } // check_args_validity

  // ==========================================================================
  // ==========================================================================
  static DataArrayBlock_t
  run(DataArrayBlock_t              Udata,
      FieldMap<core::models::Hydro> fm,
      DERIVED_QUANTITY              quantity,
      HydroSettings                 hydro_settings,
      eos::EosWrapper<device_t>     eos,
      int64_t                       iOct_begin,
      int64_t                       num_octs,
      ParallelEnv const &           par_env)
  {
    check_args_validity(Udata, iOct_begin, num_octs);

    const auto label = std::string("compute derived quantity ") + quantity._to_string();

    auto res = DataArrayBlock_t(label, Udata.block_size(), 1, Udata.num_quadrants());

    const auto    nbCellsPerLeaf = Udata.num_cells();
    const int64_t total_num_cells = nbCellsPerLeaf * num_octs;

    const auto rank = par_env.rank();

    Kokkos::parallel_for(
      label,
      Kokkos::RangePolicy<ExecutionSpace>(0, total_num_cells),
      KOKKOS_LAMBDA(const int64_t & global_index) {
        /// convert global index into
        // - octant id
        // - cell_index inside block (from 0 to nbCellsPerLeaf-1)
        const auto iOct = iOct_begin + global_index / nbCellsPerLeaf;
        const auto cell_index =
          static_cast<int32_t>(global_index - (iOct - iOct_begin) * nbCellsPerLeaf);

        HydroState<dim> uLoc; // cell-centered conservative variables in current cell

        // get conservative variable in current cell
        uLoc[Hydro::ID] = Udata(cell_index, fm[Hydro::ID], iOct);
        uLoc[Hydro::IP] = Udata(cell_index, fm[Hydro::IP], iOct);
        uLoc[Hydro::IU] = Udata(cell_index, fm[Hydro::IU], iOct);
        uLoc[Hydro::IV] = Udata(cell_index, fm[Hydro::IV], iOct);
        if constexpr (dim == 3)
          uLoc[Hydro::IW] = Udata(cell_index, fm[Hydro::IW], iOct);

        // compute primitive variables and speed of sound in current cell
        bool       valid = true;
        const auto qLoc =
          models::compute_primitives<dim, device_t>(uLoc, hydro_settings, eos, valid);
        if (!valid)
        {
          printf("[mpi rank=%d] Invalid primitive variable at cell index %d in octant %ld / %ld\n",
                 rank,
                 cell_index,
                 iOct,
                 num_octs);
        }

        if (quantity._to_integral() == +DERIVED_QUANTITY::THERMAL_PRESSURE)
        {
          res(cell_index, 0, iOct) = qLoc[Hydro::IP];
        }
        else if (quantity._to_integral() == +DERIVED_QUANTITY::SPECIFIC_EKIN)
        {
          res(cell_index, 0, iOct) = models::compute_ekin_from_primitives<dim>(qLoc);
        }
        else if (quantity._to_integral() == +DERIVED_QUANTITY::LOCAL_MACH_NUMBER)
        {
          auto u_norm = qLoc[Hydro::IU] * qLoc[Hydro::IU] + qLoc[Hydro::IV] * qLoc[Hydro::IV];
          if constexpr (dim == 3)
            u_norm += qLoc[Hydro::IW] * qLoc[Hydro::IW];
          u_norm = sqrt(u_norm);

          // speed of sound
          const auto cs = eos.sound_speed(qLoc[Hydro::IP], qLoc[Hydro::ID]);

          res(cell_index, 0, iOct) = u_norm / cs;
        }
      });

    return res;

  } // run

  // ==========================================================================
  // ==========================================================================
  static DataArrayBlock_t
  run(DataArrayBlock_t              Udata,
      FieldMap<core::models::Hydro> fm,
      std::string                   quantity,
      HydroSettings                 hydro_settings,
      eos::EosWrapper<device_t>     eos,
      int64_t                       iOct_begin,
      int64_t                       num_octs,
      ParallelEnv const &           par_env)
  {
    if (quantity == "thermal_pressure")
    {
      return run(Udata,
                 fm,
                 DERIVED_QUANTITY::THERMAL_PRESSURE,
                 hydro_settings,
                 eos,
                 iOct_begin,
                 num_octs,
                 par_env);
    }
    else if (quantity == "specific_ekin")
    {
      // clang-format off
      return run(Udata,
                 fm,
                 DERIVED_QUANTITY::SPECIFIC_EKIN,
                 hydro_settings,eos,
                 iOct_begin,
                 num_octs,
                 par_env);
      // clang-format on
    }
    else if (quantity == "local_mach_number")
    {
      return run(Udata,
                 fm,
                 DERIVED_QUANTITY::LOCAL_MACH_NUMBER,
                 hydro_settings,
                 eos,
                 iOct_begin,
                 num_octs,
                 par_env);
    }
    else
    {
      KALYPSSO_ERROR(
        "ComputeDerivedQuantity: unknow quantity (check your input parameter file) - use "
        "thermal pressure instead.");
      return run(Udata,
                 fm,
                 DERIVED_QUANTITY::THERMAL_PRESSURE,
                 hydro_settings,
                 eos,
                 iOct_begin,
                 num_octs,
                 par_env);
    }
  } // run

}; // struct ComputeDerivedQuantities

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_COMPUTE_DERIVED_QUANTITIES_H_
