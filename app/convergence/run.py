#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import sys, getopt, os
import argparse
import configparser
import subprocess

def run(ini_filename, level_min, level_max, block_size):
    print("Run test with\n  template={}\n  level_min={}\n  level_max={}\n  block_size={}".format(ini_filename, level_min, level_max, block_size))
    ini = configparser.RawConfigParser()
    ini.optionxform = lambda option : option
    ini.read(ini_filename)
    ini['amr']['level_min'] = "{}".format(level_min)
    ini['amr']['level_max'] = "{}".format(level_max)
    ini['amr']['bx'] = "{}".format(block_size)
    ini['amr']['by'] = "{}".format(block_size)

    prefix = os.path.splitext(ini_filename)[0]+'_{}_{}_{}'.format(level_min, level_max, block_size)

    ini['output']['outputPrefix'] = prefix

    ini_filename_out = prefix+'.ini'

    with open(ini_filename_out, 'w') as ini_out:
        ini.write(ini_out, space_around_delimiters=False)

    with open(prefix+'.log', 'w') as log_out:
        print("Actually running test...")
        res = subprocess.run(["../solver_godunov_hydro", "--ini", ini_filename_out], stdout=log_out, stderr=subprocess.STDOUT)
        if res.returncode != 0:
            print("Error running kalypsso, returned {}".format(res.returncode))

def convergence_run(ini_filename):

    run(ini_filename, 2, 2, 16)
    run(ini_filename, 3, 3, 16)
    run(ini_filename, 4, 4, 16)
    run(ini_filename, 5, 5, 16)
    run(ini_filename, 6, 6, 16)
    run(ini_filename, 7, 7, 16)
    run(ini_filename, 8, 8, 16)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convergence study.')
    parser.add_argument('--ini', type=str, default='test_isentropic_vortex_2d.ini', help='kalypsso template ini parameter file')
    args = parser.parse_args()

    convergence_run(args.ini)
