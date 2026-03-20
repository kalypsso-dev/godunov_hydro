if(NOT CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")
  add_executable(sedov3_qp sedov3_qp.f90)

  # TODO: refactor sedov3_qp.f90 to avoid all the warnings
  remove_definitions(-Wall)
else()
  message(
    WARNING
      "nvfortran doesn't support quad precision natively; it is only available through third party library. Just use gfortran to build sedov3_qp."
  )
endif()
