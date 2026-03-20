# How to perform convergence study ?

## No AMR

Just use the same value for `level_min` and `level_max`.

```bash
# example
./submit_job.sh -j ./job.sh.tmpl -i ./test_isentropic_vortex_2d.ini.tmpl -b 16 -k 5 -l 5 -m 1
./submit_job.sh -j ./job.sh.tmpl -i ./test_isentropic_vortex_2d.ini.tmpl -b 16 -k 6 -l 6 -m 1
./submit_job.sh -j ./job.sh.tmpl -i ./test_isentropic_vortex_2d.ini.tmpl -b 16 -k 7 -l 7 -m 1
./submit_job.sh -j ./job.sh.tmpl -i ./test_isentropic_vortex_2d.ini.tmpl -b 16 -k 8 -l 8 -m 1
```

## AMR activated

Just use different values for `level_min` and `level_max`.

```bash
# examples with 3 levels
./submit_job.sh -j ./job.sh.tmpl -i ./test_isentropic_vortex_2d.ini.tmpl -b 16 -k 4 -l 6 -m 1
./submit_job.sh -j ./job.sh.tmpl -i ./test_isentropic_vortex_2d.ini.tmpl -b 16 -k 5 -l 7 -m 1
./submit_job.sh -j ./job.sh.tmpl -i ./test_isentropic_vortex_2d.ini.tmpl -b 16 -k 6 -l 8 -m 1

# examples with 4 levels
./submit_job.sh -j ./job.sh.tmpl -i ./test_isentropic_vortex_2d.ini.tmpl -b 16 -k 4 -l 7 -m 1
./submit_job.sh -j ./job.sh.tmpl -i ./test_isentropic_vortex_2d.ini.tmpl -b 16 -k 5 -l 8 -m 1
./submit_job.sh -j ./job.sh.tmpl -i ./test_isentropic_vortex_2d.ini.tmpl -b 16 -k 6 -l 9 -m 1
```
