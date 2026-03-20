add_subdirectory(sedov_fortran)

configure_file(sedov.py sedov.py COPYONLY)
configure_file(sedov_plot.py sedov_plot.py COPYONLY)
configure_file(test_sedov_blast_2d.ini test_sedov_blast_2d.ini COPYONLY)
configure_file(test_sedov_blast_3d.ini test_sedov_blast_3d.ini COPYONLY)
