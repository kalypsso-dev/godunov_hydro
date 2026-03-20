Steps to compare numerical and analytical solution for the Sod shock tube:

1. Run kalypsso to get numerical solution
```shell
../solver_godunov_hydro --ini ./test_sod_2d.ini
```

2. Run python script to extract a 1D slice (paraview required here):
```shell
pvbatch extract_sod_numerical_solution.py
```
this should create file `sod_numerical_solution.csv`

3. Plot analytical and numerical solutions:
```shell
./sod_plot.py
```

Note that the current test parameter file `test_sod_2d.ini` was used to produce figure 6 (a) of kalypsso-core paper.
