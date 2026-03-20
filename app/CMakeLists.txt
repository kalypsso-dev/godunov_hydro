#
# Kalypsso hydrodynamics solver
#
set(EXE_NAME "solver_godunov_hydro")
add_executable(${EXE_NAME} solver_godunov_hydro.cpp)

target_include_directories(${EXE_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src
                                              ${CMAKE_BINARY_DIR}/src)

target_link_libraries(${EXE_NAME} PRIVATE kalypsso_warnings)

target_link_libraries(
  ${EXE_NAME}
  PRIVATE kalypsso::core
          kalypsso::log
          kalypsso::p4est_wrap
          kalypsso::config
          kalypsso::godunov_hydro
          Kokkos::kokkos
          kalypsso::monitoring)

if(KALYPSSO_CORE_USE_CNPY)
  target_link_libraries(${EXE_NAME} PRIVATE cnpy::cnpy)
endif(KALYPSSO_CORE_USE_CNPY)

if(KALYPSSO_CORE_USE_CPPTRACE)
  target_link_libraries(${EXE_NAME} PRIVATE cpptrace::cpptrace)
endif(KALYPSSO_CORE_USE_CPPTRACE)

# if(KALYPSSO_CORE_USE_MPI) set(TEST_MPI_PREFIX_CMD ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG}
# ${KALYPSSO_TEST_MPIEXEC_NUMPROCS}) endif()

# add_test(NAME solver COMMAND ${TEST_MPI_PREFIX_CMD} test_solver) set_property(TEST solver PROPERTY
# LABELS solver) if(Kokkos_ENABLE_OPENMP) set_property(TEST solver PROPERTY ENVIRONMENT
# OMP_NUM_THREADS=${KALYPSSO_TEST_OMP_NUM_THREADS} OMP_PROC_BIND=spread OMP_PLACES=threads) endif()

configure_file(test_implode_2d.ini test_implode_2d.ini COPYONLY)
configure_file(test_implode_3d.ini test_implode_3d.ini COPYONLY)
configure_file(test_blast_2d.ini test_blast_2d.ini COPYONLY)
configure_file(test_blast_2d_restart.ini test_blast_2d_restart.ini COPYONLY)
configure_file(test_blast_3d.ini test_blast_3d.ini COPYONLY)
configure_file(test_breaking_wave_2d.ini test_breaking_wave_2d.ini COPYONLY)
configure_file(test_double_mach_reflection_2d.ini test_double_mach_reflection_2d.ini COPYONLY)
configure_file(test_four_quadrant_2d.ini test_four_quadrant_2d.ini COPYONLY)
configure_file(test_four_quadrant_3d.ini test_four_quadrant_3d.ini COPYONLY)
configure_file(test_kelvin_helmholtz_2d.ini test_kelvin_helmholtz_2d.ini COPYONLY)
configure_file(test_kelvin_helmholtz_3d.ini test_kelvin_helmholtz_3d.ini COPYONLY)
configure_file(test_kelvin_helmholtz_2d_stone_small.ini test_kelvin_helmholtz_2d_stone_small.ini
               COPYONLY)
configure_file(test_kelvin_helmholtz_2d_stone_large.ini test_kelvin_helmholtz_2d_stone_large.ini
               COPYONLY)
configure_file(test_isentropic_vortex_2d.ini test_isentropic_vortex_2d.ini COPYONLY)
configure_file(test_jet_2d.ini test_jet_2d.ini COPYONLY)
configure_file(test_jet_3d.ini test_jet_3d.ini COPYONLY)
configure_file(test_rayleigh_taylor_2d.ini test_rayleigh_taylor_2d.ini COPYONLY)
configure_file(test_rayleigh_taylor_3d.ini test_rayleigh_taylor_3d.ini COPYONLY)
configure_file(test_shock_bubble_2d.ini test_shock_bubble_2d.ini COPYONLY)
configure_file(test_shock_bubble_3d.ini test_shock_bubble_3d.ini COPYONLY)
configure_file(test_shock_bubble_2d_multi.ini test_shock_bubble_2d_multi.ini COPYONLY)
configure_file(test_shock_bubble_3d_multi.ini test_shock_bubble_3d_multi.ini COPYONLY)
configure_file(test_triple_point_2d.ini test_triple_point_2d.ini COPYONLY)
configure_file(test_shock_diffraction_2d.ini test_shock_diffraction_2d.ini COPYONLY)
configure_file(test_shu_osher_2d.ini test_shu_osher_2d.ini COPYONLY)
configure_file(test_woodward_2d.ini test_woodward_2d.ini COPYONLY)

# input files used for benchmark / weak scaling studies these files are templates. Some variables
# (e.g. NBRICK_X, ...) will be substituted by script submit_job.sh
configure_file(bench/weak_scaling_2d/test_blast_2d_replicated.ini.tmpl
               bench/weak_scaling_2d/test_blast_2d_replicated.ini.tmpl COPYONLY)
configure_file(bench/weak_scaling_2d/job_topaze.sh.tmpl bench/weak_scaling_2d/job_topaze.sh.tmpl
               @ONLY)
configure_file(bench/weak_scaling_2d/test_blast_2d_replicated_cea_he.ini.tmpl
               bench/weak_scaling_2d/test_blast_2d_replicated_cea_he.ini.tmpl COPYONLY)
configure_file(bench/weak_scaling_2d/job_cea_he.sh.tmpl bench/weak_scaling_2d/job_cea_he.sh.tmpl
               @ONLY)
configure_file(bench/weak_scaling_2d/job_cea_inti.sh.tmpl
               bench/weak_scaling_2d/job_cea_inti.sh.tmpl @ONLY)

configure_file(bench/weak_scaling_3d/test_blast_3d_replicated.ini.tmpl
               bench/weak_scaling_3d/test_blast_3d_replicated.ini.tmpl COPYONLY)
configure_file(bench/weak_scaling_3d/job_topaze.sh.tmpl bench/weak_scaling_3d/job_topaze.sh.tmpl
               @ONLY)
configure_file(bench/weak_scaling_3d/test_blast_3d_replicated_cea_he.ini.tmpl
               bench/weak_scaling_3d/test_blast_3d_replicated_cea_he.ini.tmpl COPYONLY)
configure_file(bench/weak_scaling_3d/job_cea_he.sh.tmpl bench/weak_scaling_3d/job_cea_he.sh.tmpl
               @ONLY)
configure_file(bench/weak_scaling_3d/job_cea_inti.sh.tmpl
               bench/weak_scaling_3d/job_cea_inti.sh.tmpl @ONLY)

configure_file(bench/set_visible_device.sh bench/set_visible_device.sh COPYONLY)
configure_file(bench/submit_job_topaze.sh bench/submit_job_topaze.sh COPYONLY)
configure_file(bench/submit_job_cea_he.sh bench/submit_job_cea_he.sh COPYONLY)

# input files used for convergence study
configure_file(convergence/test_isentropic_vortex_2d.ini convergence/test_isentropic_vortex_2d.ini
               COPYONLY)
configure_file(convergence/test_isentropic_vortex_2d.ini.tmpl
               convergence/test_isentropic_vortex_2d.ini.tmpl COPYONLY)
configure_file(convergence/test_breaking_wave_2d.ini.tmpl
               convergence/test_breaking_wave_2d.ini.tmpl COPYONLY)
configure_file(convergence/job.sh.tmpl convergence/job.sh.tmpl @ONLY)
configure_file(convergence/submit_job.sh convergence/submit_job.sh COPYONLY)
configure_file(convergence/run.py convergence/run.py @ONLY)

# Sedov-Taylor blast test case
add_subdirectory(sedov_blast)

# Sod shock tube test case
add_subdirectory(sod)

# block size study
add_subdirectory(block_size_study)
