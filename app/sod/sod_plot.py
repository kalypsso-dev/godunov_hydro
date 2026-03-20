#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import sys, getopt
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

import configparser
import subprocess

# exact solution
import sod

def sod_plot(ini_filename):

    config = configparser.ConfigParser()
    config.read(ini_filename)

    tEnd=config.getfloat('run','tEnd', fallback=0.2)
    gamma=config.getfloat('material0','gamma', fallback=1.4)

    if (config['run']['dimension'] == '2'):
        # doing 2d
        output_filename='sod_analytical_2d.dat'
        dim=2
    else:
        print("Error: only 2D supported here")
        return

    rhoL = config.getfloat('sod',' rhoL', fallback=1.0)
    pL = config.getfloat('sod', 'pL', fallback=1.0)
    uL = config.getfloat('sod', 'uL', fallback=0.0)
    rhoR = config.getfloat('sod', 'rhoR', fallback=0.125)
    pR = config.getfloat('sod', 'pR', fallback=0.1)
    uR = config.getfloat('sod', 'uR', fallback=0.0)
    stateL = (pL, rhoL, uL)
    stateR = (pR, rhoR, uR)

    npts = 500
    dustFrac = 0.0

    print('sod args : dim={} gamma={} tEnd={} output_filename={}'.format(dim,gamma,tEnd,output_filename))
    print('sod left state: rhoL={} pL={} uL={}'.format(rhoL,pL,uL))
    print('sod right state: rhoR={} pR={} uR={}'.format(rhoR,pR,uR))

    # compute analytical solution
    positions, regions, values = sod.solve(left_state=stateL,
                                           right_state=stateR,
                                           geometry=(0., 1., 0.5),
                                           t=tEnd,
                                           gamma=gamma,
                                           npts=npts,
                                           dustFrac=dustFrac)

    sod_ana_x = values['x']
    sod_ana_rho = values['rho']

    # load numerical solution
    num_data = sod_num=np.loadtxt('sod_numerical_solution.csv', skiprows=1, delimiter=',',usecols=(1,2,6))
    sod_num_x = sod_num[:,2]
    sod_num_rho = sod_num[:,1]
    sod_num_level = sod_num[:,0]

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(11,8))
    #ax1 = plt.subplot(2,1,1)
    #ax2 = plt.subplot(2,1,2, sharex=ax1)
    ax1.plot(sod_ana_x, sod_ana_rho, 'go-', label='analytical')
    ax1.plot(sod_num_x, sod_num_rho, 'xb-', label='numerical')
    ax2.plot(sod_num_x, sod_num_level, 'xb-', label='AMR levels')
    ax1.legend()
    ax2.legend()
    plt.suptitle('Sod at tEnd={} with gamma={}'.format(tEnd,gamma), fontsize=20)
    plt.show()

###############################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Display sod density plots.')
    parser.add_argument('--ini', type=str, default='test_sod_2d.ini', help='kalypsso ini parameter file')
    args = parser.parse_args()

    sod_plot(args.ini)
