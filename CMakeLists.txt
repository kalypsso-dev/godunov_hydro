set(GODUNOV_HYDRO_SRCS
    border_conditions/BCDoubleMachReflection.cpp
    border_conditions/BCDoubleMachReflection.h
    border_conditions/BCJet.cpp
    border_conditions/BCJet.h
    border_conditions/BCShockBubble.cpp
    border_conditions/BCShockBubble.h
    border_conditions/FillOutside.cpp
    border_conditions/FillOutside.h
    common.h
    eos/EosWrapper.h
    init/init.cpp
    init/init.h
    init/InitBlast.cpp
    init/InitBlast.h
    init/InitBreakingWave.cpp
    init/InitBreakingWave.h
    init/InitDoubleMachReflection.cpp
    init/InitDoubleMachReflection.h
    init/InitFourQuadrant.cpp
    init/InitFourQuadrant.h
    init/InitImplode.cpp
    init/InitImplode.h
    init/InitIsentropicVortex.cpp
    init/InitIsentropicVortex.h
    init/InitJet.cpp
    init/InitJet.h
    init/InitKelvinHelmholtz.cpp
    init/InitKelvinHelmholtz.h
    init/InitRayleighTaylor.cpp
    init/InitRayleighTaylor.h
    init/InitShockBubble.cpp
    init/InitShockBubble.h
    init/InitShuOsher.cpp
    init/InitShuOsher.h
    init/InitSod.cpp
    init/InitSod.h
    init/InitTriplePoint.cpp
    init/InitTriplePoint.h
    init/InitWoodward.cpp
    init/InitWoodward.h
    models/Hydro.h
    models/HydroState.h
    models/utils_hydro.h
    profiling.h
    scheme/AddGravitySourceTerm.cpp
    scheme/AddGravitySourceTerm.h
    scheme/ComputeDtHydroFunctor.cpp
    scheme/ComputeDtHydroFunctor.h
    scheme/ComputeFluxesAndStoreFunctor.cpp
    scheme/ComputeFluxesAndStoreFunctor.h
    scheme/ComputeFluxesAndUpdateFunctor.cpp
    scheme/ComputeFluxesAndUpdateFunctor.h
    scheme/ComputeFluxesAndConservativeUpdateFunctor.cpp
    scheme/ComputeFluxesAndConservativeUpdateFunctor.h
    scheme/ComputeHancockUpdate.h
    scheme/ComputeLimitedSlopesFunctor.cpp
    scheme/ComputeLimitedSlopesFunctor.h
    scheme/ComputeViscousFluxesAndStoreFunctor.h
    scheme/ComputeViscousFluxesAndStoreFunctor.cpp
    scheme/ConvertToPrimitivesVariablesFunctor.cpp
    scheme/ConvertToPrimitivesVariablesFunctor.h
    scheme/GodunovImplemBase.h
    scheme/GodunovImplemV0.cpp
    scheme/GodunovImplemV0.h
    scheme/GodunovImplemV1.cpp
    scheme/GodunovImplemV1.h
    scheme/GodunovImplemV2.cpp
    scheme/GodunovImplemV2.h
    scheme/ReadFluxesAndConservativeUpdateFunctor.cpp
    scheme/ReadFluxesAndConservativeUpdateFunctor.h
    SolverGodunovHydro.cpp
    SolverGodunovHydro.h
    utils/ComputeDerivedQuantities.h
    utils/ComputeRadialProfileFunctor.h)

add_library(godunov_hydro ${GODUNOV_HYDRO_SRCS})
# target_compile_options(godunov_hydro PUBLIC "-G")
target_include_directories(godunov_hydro PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR}/src
                                                ${CMAKE_SOURCE_DIR}/external/better-enums)

target_link_libraries(godunov_hydro PRIVATE kalypsso_warnings)

target_link_libraries(godunov_hydro PUBLIC Kokkos::kokkos kalypsso::log kalypsso::core
                                           kalypsso::mpiUtils kalypsso::p4est_wrap)

# declare library alias
add_library(kalypsso::godunov_hydro ALIAS godunov_hydro)

# application main
add_subdirectory(app)
