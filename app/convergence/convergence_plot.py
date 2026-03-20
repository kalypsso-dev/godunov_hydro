#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import sys, getopt
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

font = {'family': 'normal',
        'size': 20,
        }

rc('font', **font)

def convergence_plot():
    dx_no_amr = np.array([4, 5, 6, 7, 8])
    dx_no_amr += 4
    dx_no_amr = 1.0/2**(dx_no_amr)
    L1_no_amr = np.array([4.21e-5, 1.03e-5, 2.58e-6, 6.46e-7, 1.61e-07])
    L2_no_amr = np.array([1.20e-4, 2.90e-5, 7.15e-6, 1.77e-6, 4.44e-07])

    dx_amr_levels3 = np.array([6, 7, 8])
    dx_amr_levels3 += 4
    dx_amr_levels3 = 1.0/2**(dx_amr_levels3)
    L1_amr_levels3 = np.array([4.16e-4, 3.06e-4, 9.6e-5])
    L2_amr_levels3 = np.array([5.90e-4, 4.50e-4, 1.50e-4])

    dx_amr_levels4 = np.array([7, 8, 9])
    dx_amr_levels4 += 4
    dx_amr_levels4 = 1.0/2**(dx_amr_levels4)
    L1_amr_levels4 = np.array([4.72e-4, 3.13e-4, 9.64e-5])
    L2_amr_levels4 = np.array([6.37e-4, 4.60e-4, 1.50e-4])

    fig, (ax1, ax2) = plt.subplots(1,2)

    fig.suptitle(r'kalypsso - convergence plot for Godunov hydro (Muscl, order 2)\\test case: advected isentropic vortex at t=10.0', fontsize=30, fontweight='bold')

    ax1.loglog(dx_no_amr, L1_no_amr, color='r', marker="+", label='relative L1 error - no amr')
    ax1.loglog(dx_amr_levels3, L1_amr_levels3, color='g', marker="+", label='relative L1 error - amr - 3 levels')
    ax1.loglog(dx_amr_levels4, L1_amr_levels4, color='b', marker="+", label='relative L1 error - amr - 4 levels')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=2)
    ax1.set_ylabel(r'L1 error')
    ax1.yaxis.label.set_color('black')
    ax1.legend()

    ax2.loglog(dx_no_amr, L2_no_amr, color='r', marker="+", label='relative L2 error - no amr')
    ax2.loglog(dx_amr_levels3, L2_amr_levels3, color='g', marker="+", label='relative L2 error - amr - 3 levels')
    ax2.loglog(dx_amr_levels4, L2_amr_levels4, color='b', marker="+", label='relative L2 error - amr - 4 levels')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log', base=2)
    ax2.set_ylabel(r'L2 error')
    ax2.yaxis.label.set_color('black')
    ax2.legend()

    #plt.savefig('convergence_godunov_hydro_muscl.png')

    plt.show()

if __name__ == "__main__":
    convergence_plot()
