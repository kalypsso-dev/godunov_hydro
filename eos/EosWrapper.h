// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file EosWrapper.h
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_MODELS_EOS_WRAPPER_H_
#define KALYPSSO_GODUNOV_HYDRO_MODELS_EOS_WRAPPER_H_

#include <kalypsso/core/kalypsso_core_config.h>
#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h>
#include <kalypsso/core/eos/IdealGasEos.h>
#include <kalypsso/core/eos/StiffenedGasEos.h>
#include <kalypsso/core/eos/VanDerWaalsGasEos.h>
#include <kalypsso/core/eos/MieGruneisenEosSW.h>
#include <kalypsso/core/eos/eos_utils.h>

namespace kalypsso
{

namespace godunov_hydro
{

namespace eos
{

// ==========================================================================
// ==========================================================================
/**
 * This a front-end helper to access equation of states data.
 *
 * \todo See if we could refactor using static polymorphism.
 */
template <typename device_t>
class EosWrapper
{
public:
  EosWrapper() = default;

  //! Initialized Eos for a given material by reading property from ini file.
  EosWrapper(ConfigMap const & config_map)
    : m_eos_type(core::eos::get_eos_type(config_map))
    , m_ig_eos(config_map)
    , m_sg_eos(config_map)
    , m_vdw_eos(config_map)
    , m_sw_eos(config_map)
  {}

  /**
   * Compute pressure from specific internal energy.
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  pressure_from_specific_eint(real_t specific_eint, real_t rho) const
  {
    if (m_eos_type == +core::eos::EOS_TYPE::IDEAL_GAS)
    {
      return m_ig_eos.pressure_from_specific_eint(specific_eint, rho);
    }
    else if (m_eos_type == +core::eos::EOS_TYPE::STIFFENED_GAS)
    {
      return m_sg_eos.pressure_from_specific_eint(specific_eint, rho);
    }
    else if (m_eos_type == +core::eos::EOS_TYPE::VANDERWAALS_GAS)
    {
      return m_vdw_eos.pressure_from_specific_eint(specific_eint, rho);
    }
    else if (m_eos_type == +core::eos::EOS_TYPE::MIE_GRUNEISEN_SW)
    {
      return m_sw_eos.pressure_from_specific_eint(specific_eint, rho);
    }
    return ZERO_F;
  }

  /**
   * Compute pressure from volumic internal energy.
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  pressure_from_volumic_eint(real_t volumic_eint, real_t rho) const
  {
    if (m_eos_type == +core::eos::EOS_TYPE::IDEAL_GAS)
    {
      return m_ig_eos.pressure_from_volumic_eint(volumic_eint, rho);
    }
    else if (m_eos_type == +core::eos::EOS_TYPE::STIFFENED_GAS)
    {
      return m_sg_eos.pressure_from_volumic_eint(volumic_eint, rho);
    }
    else if (m_eos_type == +core::eos::EOS_TYPE::VANDERWAALS_GAS)
    {
      return m_vdw_eos.pressure_from_volumic_eint(volumic_eint, rho);
    }
    else if (m_eos_type == +core::eos::EOS_TYPE::MIE_GRUNEISEN_SW)
    {
      return m_sw_eos.pressure_from_volumic_eint(volumic_eint, rho);
    }
    return ZERO_F;
  }

  /**
   * Compute specific internal energy.
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  specific_eint_from_pressure(real_t pressure, real_t rho) const
  {
    if (m_eos_type == +core::eos::EOS_TYPE::IDEAL_GAS)
    {
      return m_ig_eos.specific_eint_from_pressure(pressure, rho);
    }
    else if (m_eos_type == +core::eos::EOS_TYPE::STIFFENED_GAS)
    {
      return m_sg_eos.specific_eint_from_pressure(pressure, rho);
    }
    else if (m_eos_type == +core::eos::EOS_TYPE::VANDERWAALS_GAS)
    {
      return m_vdw_eos.specific_eint_from_pressure(pressure, rho);
    }
    else if (m_eos_type == +core::eos::EOS_TYPE::MIE_GRUNEISEN_SW)
    {
      return m_sw_eos.specific_eint_from_pressure(pressure, rho);
    }
    return ZERO_F;
  }

  /**
   * Compute volumic internal energy.
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  volumic_eint_from_pressure(real_t pressure, [[maybe_unused]] real_t rho) const
  {
    if (m_eos_type == +core::eos::EOS_TYPE::IDEAL_GAS)
    {
      return m_ig_eos.volumic_eint_from_pressure(pressure);
    }
    else if (m_eos_type == +core::eos::EOS_TYPE::STIFFENED_GAS)
    {
      return m_sg_eos.volumic_eint_from_pressure(pressure);
    }
    else if (m_eos_type == +core::eos::EOS_TYPE::VANDERWAALS_GAS)
    {
      return m_vdw_eos.volumic_eint_from_pressure(pressure, rho);
    }
    else if (m_eos_type == +core::eos::EOS_TYPE::MIE_GRUNEISEN_SW)
    {
      return m_sw_eos.volumic_eint_from_pressure(pressure, rho);
    }
    return ZERO_F;
  }

  /**
   * Compute speed of sound.
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  sound_speed(real_t pressure, real_t rho) const
  {
    if (m_eos_type == +core::eos::EOS_TYPE::IDEAL_GAS)
    {
      return m_ig_eos.sound_speed(pressure, rho);
    }
    else if (m_eos_type == +core::eos::EOS_TYPE::STIFFENED_GAS)
    {
      return m_sg_eos.sound_speed(pressure, rho);
    }
    else if (m_eos_type == +core::eos::EOS_TYPE::VANDERWAALS_GAS)
    {
      return m_vdw_eos.sound_speed(pressure, rho);
    }
    else if (m_eos_type == +core::eos::EOS_TYPE::MIE_GRUNEISEN_SW)
    {
      return m_sw_eos.sound_speed(pressure, rho);
    }
    return ZERO_F;
  }

  /**
   * specific heat ratio (when available and meaningful)
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  gamma() const
  {
    if (m_eos_type == +core::eos::EOS_TYPE::IDEAL_GAS)
    {
      return m_ig_eos.gamma();
    }
    else if (m_eos_type == +core::eos::EOS_TYPE::STIFFENED_GAS)
    {
      return m_sg_eos.gamma();
    }
    else if (m_eos_type == +core::eos::EOS_TYPE::VANDERWAALS_GAS)
    {
      return m_vdw_eos.gamma();
    }

    // return an invalid value
    return ZERO_F;
  }

private:
  //! type of equation of state
  core::eos::EOS_TYPE m_eos_type;

  //! Ideal gas
  core::eos::IdealGasEos m_ig_eos;

  //! Stiffened gas
  core::eos::StiffenedGasEos m_sg_eos;

  //! Van der Waals gas
  core::eos::VanDerWaalsGasEos m_vdw_eos;

  //! Mie-Gruneisen Shock wave EOS
  core::eos::MieGruneisenEosSW m_sw_eos;

}; // class EosWrapper

} // namespace eos

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_MODELS_EOS_WRAPPER_H_
