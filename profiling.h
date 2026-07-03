// SPDX-FileCopyrightText: 2025 kalypsso-dev/godunov_hydro authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file profiling.h
 */
#ifndef KALYPSSO_GODUNOV_HYDRO_PROFILING_DATA_H_
#define KALYPSSO_GODUNOV_HYDRO_PROFILING_DATA_H_

#include <kalypsso/utils/monitoring/ProfilingManager.h>
#include <../better-enums/enum.h>

#define ADD_CASE(enum, name, color) \
  case ProfilingZone::enum:         \
    return                          \
    {                               \
      name, Color_t::color()        \
    }

#ifndef KALYPSSO_PROFILING_REGION
#  define KALYPSSO_PROFILING_REGION_HOST(mgr, value) \
    ProfilingTimer region_##value(mgr, ProfilingZone::value, ProfilingRegion::TIMER_HOST)
#  define KALYPSSO_PROFILING_REGION_DEVICE(mgr, value) \
    ProfilingTimer region_##value(mgr, ProfilingZone::value, ProfilingRegion::TIMER_DEVICE)
#endif

namespace kalypsso
{
namespace godunov_hydro
{

// clang-format off
BETTER_ENUM(ProfilingZone, uint32_t,
  AMR_CYCLE,
    AMR_CYCLE_RESIZE_AUX,
    AMR_CYCLE_COPY_HASHMAP,
    AMR_CYCLE_SYNC_MPI_GHOSTS,
    AMR_CYCLE_MARK_CELLS,
    AMR_CYCLE_ADAPT_MESH,
    AMR_CYCLE_USERDATA_REMAP,
  LOAD_BALANCING,
    LOAD_BALANCING_PARTITION_MESH,
    LOAD_BALANCING_PARTITION_USERDATA,
    LOAD_BALANCING_UPDATE_MESH,
    LOAD_BALANCING_RESIZE,
  NUM,
    NUM_CFL,
    NUM_CFL_LOCAL,
    NUM_SCHEME,
      NUM_SCHEME_CONV_PRIM,
      NUM_SCHEME_EXCHANGE_Q_MIRROR_GHOST,
      NUM_SCHEME_SLOPES,
      NUM_SCHEME_COMPUTE_FLUXES,
      NUM_SCHEME_COMPUTE_VISCOUS_FLUXES,
      NUM_SCHEME_UPDATE,
      NUM_SCHEME_GRAVITY,
  IO
)
// clang-format on

class ProfilingTimer
{
private:
  static constexpr std::pair<const char *, Color_t>
  get_name_color(const ProfilingZone zone)
  {
    switch (zone)
    {
      // clang-format off
      ADD_CASE(AMR_CYCLE,                         "AMR_CYCLE                              ", FullBlue);
      ADD_CASE(AMR_CYCLE_RESIZE_AUX,              "AMR_CYCLE::RESIZE_AUX                  ", RoyalBlue);
      ADD_CASE(AMR_CYCLE_COPY_HASHMAP,            "AMR_CYCLE::COPY_HASHMAP                ", DodgerBlue);
      ADD_CASE(AMR_CYCLE_SYNC_MPI_GHOSTS,         "AMR_CYCLE::SYNC_MPI_GHOSTS             ", SteelBlue);
      ADD_CASE(AMR_CYCLE_MARK_CELLS,              "AMR_CYCLE::MARK_CELLS                  ", SteelBlue);
      ADD_CASE(AMR_CYCLE_ADAPT_MESH,              "AMR_CYCLE::ADAPT_MESH                  ", Lavender);
      ADD_CASE(AMR_CYCLE_USERDATA_REMAP,          "AMR_CYCLE::USERDATA_REMAP              ", DeepSkyBlue);
      ADD_CASE(LOAD_BALANCING,                    "LOAD_BALANCING                         ", LightBlue);
      ADD_CASE(LOAD_BALANCING_PARTITION_MESH,     "LOAD_BALANCING::PARTITION_MESH         ", LightBlue);
      ADD_CASE(LOAD_BALANCING_PARTITION_USERDATA, "LOAD_BALANCING::PARTITION_USERDATA     ", LightBlue);
      ADD_CASE(LOAD_BALANCING_UPDATE_MESH,        "LOAD_BALANCING::UPDATE_MESH            ", LightBlue);
      ADD_CASE(LOAD_BALANCING_RESIZE,             "LOAD_BALANCING::RESIZE                 ", LightBlue);
      ADD_CASE(NUM_CFL,                           "NUM::CFL                               ", DarkRed);
      ADD_CASE(NUM_CFL_LOCAL,                     "NUM::CFL_LOCAL                         ", DarkRed);
      ADD_CASE(NUM_SCHEME,                        "NUM::SCHEME                            ", FullRed);
      ADD_CASE(NUM_SCHEME_CONV_PRIM,              "NUM::SCHEME::CONV_PRIM                 ", LightRed);
      ADD_CASE(NUM_SCHEME_EXCHANGE_Q_MIRROR_GHOST,"NUM::SCHEME::EXCHANGE_Q_MIRROR_GHOST   ", LightRed);
      ADD_CASE(NUM_SCHEME_SLOPES,                 "NUM::SCHEME::SLOPES                    ", Salmon);
      ADD_CASE(NUM_SCHEME_COMPUTE_FLUXES,         "NUM::SCHEME::COMPUTE_FLUXES            ", Coral);
      ADD_CASE(NUM_SCHEME_COMPUTE_VISCOUS_FLUXES, "NUM::SCHEME::COMPUTE_VISCOUS_FLUXES    ", Coral);
      ADD_CASE(NUM_SCHEME_UPDATE,                 "NUM::SCHEME::UPDATE                    ", Pink);
      ADD_CASE(NUM_SCHEME_GRAVITY,                "NUM::SCHEME::GRAVITY                   ", PaleGold);
      ADD_CASE(IO,                                "IO                                     ", FullBlue);
      // clang-format on
      default:
        return { zone._to_string(), Color_t::Default() };
    }
  }

public:
  static auto &
  get_profiling_region(ProfilingManager &                profiling_mgr,
                       const ProfilingZone               zone,
                       ProfilingRegion::timer_location_t timer_location)
  {
    const auto [name, color] = get_name_color(zone);
    return profiling_mgr.get_region(name, timer_location, color);
  }

  ProfilingTimer(ProfilingManager & profiling_mgr, const ProfilingZone zone)
    : m_timer(get_profiling_region(profiling_mgr, zone, ProfilingRegion::TIMER_HOST))
  {
    m_timer.start();
  }

  ProfilingTimer(ProfilingManager &                profiling_mgr,
                 const ProfilingZone               zone,
                 ProfilingRegion::timer_location_t timer_location)
    : m_timer(get_profiling_region(profiling_mgr, zone, timer_location))
  {
    m_timer.start();
  }

  ~ProfilingTimer() { m_timer.stop(); }

private:
  ProfilingRegion & m_timer;
};

} // namespace godunov_hydro

} // namespace kalypsso

#undef ADD_CASE

#endif // KALYPSSO_GODUNOV_HYDRO_PROFILING_DATA_H_
