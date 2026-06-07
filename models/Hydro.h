// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_five_eq authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file Hydro.h
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_MODELS_FIVE_EQ_H_
#define KALYPSSO_GODUNOV_HYDRO_MODELS_FIVE_EQ_H_

#include <kalypsso/core/kokkos_shared.h>

#include <cstdint>
#include <string>

namespace kalypsso
{

namespace godunov_hydro
{

namespace models
{

// =============================================================
// =============================================================
template <size_t dim>
class Hydro
{

public:
  using Id_t = int32_t;

  /**
   * Five equations models field ids
   */
  enum VarId : Id_t
  {
    ID = 0,                         /*!< ID mixture density */
    IP = 1,                         /*!< IP Pressure/Energy field index */
    IE = 1,                         /*!< IE Energy/Pressure field index */
    IU = 2,                         /*!< X velocity / momentum index */
    IV = 3,                         /*!< Y velocity / momentum index */
    IW = (dim == 2) ? IV : IV + 1,  /*!< Z velocity / momentum index */
    HYDRO_VARID_COUNT = IW + 1,     /*!< number of hydrodynamics variables */
    VARID_COUNT = HYDRO_VARID_COUNT /*!< invalid index, just counting number of fields */
  };

  enum GradTensorId : Id_t
  {
    IUX = 0, /*!< du/dx */
    IUY = 1, /*!< du/dy */
    IUZ = 2, /*!< du/dz */
    IVX = 3, /*!< dv/dx */
    IVY = 4, /*!< dv/dy */
    IVZ = 5, /*!< dv/dz */
    IWX = 6, /*!< dw/dx */
    IWY = 7, /*!< dw/dy */
    IWZ = 8, /*!< dw/dz */
  };

  using var_names_t = std::array<std::string, static_cast<size_t>(VARID_COUNT)>;
  static const var_names_t ID_TO_NAMES;

  KOKKOS_INLINE_FUNCTION static constexpr size_t
  nbvar()
  {
    return static_cast<size_t>(VARID_COUNT);
  }

  static std::string
  name(Id_t var)
  {
    if (var < HYDRO_VARID_COUNT)
      return ID_TO_NAMES[var];

    return std::string("invalid_variable");
  }

  static std::string
  name(size_t var)
  {
    return name(static_cast<Id_t>(var));
  }

  static var_names_t
  get_var_names()
  {
    var_names_t arr;

    arr[ID] = "rho";
    arr[IE] = "e_tot";
    arr[IU] = "rho_vx";
    arr[IV] = "rho_vy";
    if constexpr (dim == 3)
      arr[IW] = "rho_vz";

    return arr;
  }

}; // class Hydro

template <size_t dim>
const typename Hydro<dim>::var_names_t Hydro<dim>::ID_TO_NAMES = Hydro<dim>::get_var_names();

} // namespace models

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_MODELS_FIVE_EQ_H_
