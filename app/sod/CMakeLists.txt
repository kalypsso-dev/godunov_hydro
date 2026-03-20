configure_file(README.md README.md COPYONLY)

configure_file(test_sod_2d.ini test_sod_2d.ini COPYONLY)

# this script computes exact solution
configure_file(exact_solution/sod.py sod.py COPYONLY)

# this python script uses paraview to extract numerical solution from XDMF data
configure_file(extract_sod_numerical_solution.py extract_sod_numerical_solution.py @ONLY)

# plot analytical and numerical solution
configure_file(sod_plot.py sod_plot.py COPYONLY)
