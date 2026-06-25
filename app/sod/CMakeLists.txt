configure_file(README.md README.md COPYONLY)

configure_file(test_sod_2d.ini test_sod_2d.ini COPYONLY)
configure_file(test_sod_2d_large.ini test_sod_2d_large.ini COPYONLY)

# this script computes exact solution
configure_file(exact_solution/sod.py sod.py COPYONLY)

# this python script uses paraview to extract numerical solution from XDMF data
configure_file(extract_sod_numerical_solution.py extract_sod_numerical_solution.py @ONLY)

# plot analytical and numerical solution
configure_file(sod_plot.py sod_plot.py COPYONLY)

#
# additional shock tube tests
#

# Mie-Gruneisen with shockwave EOS: Alu
configure_file(test_sod_2d_alu.ini test_sod_2d_alu.ini COPYONLY)
configure_file(sod_alu_plot.py sod_alu_plot.py COPYONLY)

# Mie-Gruneisen with Cochran-Chan EOS: nitromethane
configure_file(test_sod_2d_nitromethane.ini test_sod_2d_nitromethane.ini COPYONLY)
configure_file(sod_nitromethane_plot.py sod_nitromethane_plot.py COPYONLY)

# Mie-Gruneisen with JWL EOS: TNT
configure_file(test_sod_2d_tnt.ini test_sod_2d_tnt.ini COPYONLY)
configure_file(sod_tnt_plot.py sod_tnt_plot.py COPYONLY)

# Mie-Gruneisen with JWL EOS: LX17
configure_file(test_sod_2d_lx17_jwl.ini test_sod_2d_lx17_jwl.ini COPYONLY)
configure_file(sod_lx17_plot.py sod_lx17_plot.py COPYONLY)
