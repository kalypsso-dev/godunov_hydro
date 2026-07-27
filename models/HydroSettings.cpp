// SPDX-FileCopyrightText: 2025 kalypsso authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file HydroSettings.cpp
 */
#include <godunov_hydro/models/HydroSettings.h>

namespace kalypsso
{

namespace godunov_hydro
{

// =======================================================
// =======================================================
HydroSettings::HydroSettings(ConfigMap const & config_map)
  : cfl(config_map.getReal("hydro", "cfl", KALYPSSO_NUM(0.5)))
  , slope_type(config_map.getReal("hydro", "slope_type", KALYPSSO_NUM(1.0)))
  , smallr(config_map.getReal("hydro", "smallr", KALYPSSO_NUM(1e-10)))
  , smallc(config_map.getReal("hydro", "smallc", KALYPSSO_NUM(1e-10)))
  , smallp(smallc * smallc)
  , smallpp(smallr * smallp)
  , niter_riemann(config_map.getInteger("hydro", "niter_riemann", 10))
  , riemannSolverType(get_riemann_solver_type(config_map))
  , abort_when_negative_eint(config_map.getBool("hydro", "abort_when_negative_eint", false))
  , abort_when_negative_pressure(config_map.getBool("hydro", "abort_when_negative_pressure", true))
{} // HydroSettings::HydroSettings

// =======================================================
// =======================================================
void
HydroSettings::print() const
{

  KALYPSSO_INFO("##########################");
  KALYPSSO_INFO("Hydrodynamics parameters: ");
  KALYPSSO_INFO("##########################");
  KALYPSSO_INFO("cfl           : {}", cfl);
  KALYPSSO_INFO("smallr        : {:12.10f}", smallr);
  KALYPSSO_INFO("smallc        : {:12.10f}", smallc);
  KALYPSSO_INFO("smallp        : {:12.10f}", smallp);
  KALYPSSO_INFO("smallpp       : {:12.10f}", smallpp);
  KALYPSSO_INFO("niter_riemann : {}", niter_riemann);
  KALYPSSO_INFO("slope_type    : {}", slope_type);
  KALYPSSO_INFO("riemann       : {}", riemannSolverType._to_string());
  KALYPSSO_INFO("##########################");

} // HydroSettings::print

} // namespace godunov_hydro

} // namespace kalypsso
