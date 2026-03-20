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

import matplotlib.ticker as ticker

def weak_scaling_plot(machine):

    if machine == 'cea_he':
        nb_gpu = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
        perf_2d = np.array([1566.5, 2452.8, 3922.7, 7777.0, 15215.7, 29936.5, 54421.7, 108579.7, 198631.3, 364872.8])/1000
        perf_3d = np.array([784.8, 1257.9, 1958.2, 3873.8, 7361.7, 14773.5, 29545.3, 56957.9, 109947.2, 200998.0])/1000
    else:
        print("Arguments error: data for machine={} is not available".format(machine))
        return

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(25,10))
    #fig, ax1= plt.subplots()
    fig.suptitle('Kalypsso - Weak scaling - machine:{}'.format(machine), fontsize=40, fontweight='bold')
    plt.subplots_adjust(hspace=20)
    #fig.tight_layout(pad=1)

    ax1.plot(nb_gpu, perf_2d, color='r', marker="+", label='GPU - 2d - {}'.format(machine))
    ax1p = ax1.twinx()
    ax1p.plot(nb_gpu, 100*perf_2d/perf_2d[2]/(nb_gpu/nb_gpu[2]), color='g', label='2d weak scaling efficiency')
    ax1p.set_ylabel(r'2d weak scaling efficiency (\%)', fontsize=25)
    ax1p.yaxis.label.set_color('green')
    ax1p.set_ylim([0.0, 110.0])
    ax1p.grid(color = 'green', linestyle = '--', linewidth = 0.7, axis='both')
    ax1p.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax1p.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax1.set_xlabel('\#GPU', fontsize=25)
    ax1.set_ylabel(r'GCell-updates/s', fontsize=25)
    ax1.yaxis.label.set_color('red')
    ax1.set_ylim([0.0, 400.0])
    ax1.grid(color = 'green', linestyle = '--', linewidth = 0.7, axis='x')


    ax2.plot(nb_gpu, perf_3d, color='r', marker="+", label='GPU - 3d - {}'.format(machine))
    ax2p = ax2.twinx()
    ax2p.plot(nb_gpu, 100*perf_3d/perf_3d[2]/(nb_gpu/nb_gpu[2]), color='g', label='3d weak scaling efficiency')
    ax2p.set_ylabel(r'3d weak scaling efficiency (\%)', fontsize=25)
    ax2p.yaxis.label.set_color('green')
    ax2p.set_ylim([0.0, 110.0])
    ax2p.grid(color = 'green', linestyle = '--', linewidth = 0.7, axis='both')
    ax2p.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax2p.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax2.set_xlabel('\#GPU', fontsize=25)
    ax2.set_ylabel(r'GCell-updates/s', fontsize=25)
    ax2.yaxis.label.set_color('red')
    ax2.set_ylim([0.0, 300.0])
    ax2.grid(color = 'green', linestyle = '--', linewidth = 0.7, axis='x')

    plt.savefig('weak_scaling_{}.png'.format(machine))

    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Display performance plots.')
    parser.add_argument('--machine', type=str, default='cea_he', help='machine name (cea_he, ...)')
    args = parser.parse_args()

    weak_scaling_plot(args.machine)
