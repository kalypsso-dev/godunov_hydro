// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file HydroState.h
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_MODELS_HYDRO_STATE_H_
#define KALYPSSO_GODUNOV_HYDRO_MODELS_HYDRO_STATE_H_

#include <kalypsso/core/kokkos_shared.h>
#include <kalypsso/core/real_type.h>
#include <godunov_hydro/models/Hydro.h>

namespace kalypsso
{

namespace godunov_hydro
{

template <size_t nbvar>
using StateNd = Kokkos::Array<real_t, nbvar>;

/**
 * HydroState is a small array used for storing either primitive or conservative variables.
 *
 * \note the size is the number of independent degrees of freedom plus one for mixture density.
 */
template <size_t dim>
using HydroState = StateNd<models::Hydro<dim>::nbvar()>;

template <size_t dim>
using GradTensor = Kokkos::Array<real_t, dim * dim>;

using HydroState2d = HydroState<2>;
using HydroState3d = HydroState<3>;
using GravityState = Kokkos::Array<real_t, 3>;

} // namespace godunov_hydro

} // namespace kalypsso

#endif // KALYPSSO_GODUNOV_HYDRO_MODELS_HYDRO_STATE_H_
