# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 15:12:44 2014

@author: oscar
"""

from __future__ import division, absolute_import, print_function
import matplotlib.pyplot as plt
import numpy as np
from dmft.twosite import out_plot, dmft_loop

def plot_feature(res, name, feature):
    for U, zet, sim in res:
        fig = plt.figure()
        out_plot(sim, feature, '')

        plt.legend()
        plt.title('U={:.4f}, hyb={:.4f}'.format(U, np.sqrt(zet*sim.m2)))
        if sim.freq_axis == 'real':
            plt.xlabel('$\\omega$')
        else:
            plt.xlabel('$i\\omega_n$')

        fig.savefig('{}_{}_U{:.2f}.png'.format(name, feature, U), format='png',
                    transparent=False, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)

def run_halffill(axis = 'matsubara'):
    fig = plt.figure()
    u_int = np.arange(0, 3.2, 0.05)
    for beta in [6, 10, 20, 30, 50, 100, 1e5]:
        out_file = axis+'_halffill_b'+str(beta)
        try:
            res = np.load(out_file+'.npy')
        except IOError:
            res = dmft_loop(u_int, axis=axis, beta=beta, hop=0.5)
            np.save(out_file, res)

        plot_feature(res, out_file, 'A')
        plot_feature(res, out_file, 'sigma')


        plt.plot(res[:, 0], res[:, 1], '+-', label='$\\beta = {}$'.format(beta))
    #    plt.plot(u_int, 1-u_int.clip(0, 3)**2/9, '--', label='$1-U^2/U_c^2')
    plt.legend()
    plt.title('Quasiparticle weigth')
    plt.ylabel('Z')
    plt.xlabel('U/D')
    fig.savefig(out_file+'_Z.png', format='png',
                transparent=False, bbox_inches='tight', pad_inches=0.05)
#    plt.close(fig)

if __name__ == "__main__":
    fig = plt.figure()
    axis = 'real'
    u_int = np.arange(0, 3.2, 0.05)
    beta = 1e5
    for fill in np.arange(1, 0.9, -0.025):
        out_file = 'real_{}fill_b{}'.format(fill, beta)
        try:
            res = np.load(out_file+'.npy')
        except IOError:
            res = dmft_loop(u_int, axis=axis, beta=beta, hop=0.5, filling=fill)
            np.save(out_file, res)

        plot_feature(res, out_file, 'A')
        plot_feature(res, out_file, 'sigma')


        plt.plot(res[:, 0], res[:, 1], '+-', label='$\\beta = {}$'.format(beta))
    #    plt.plot(u_int, 1-u_int.clip(0, 3)**2/9, '--', label='$1-U^2/U_c^2')
    plt.legend()
    plt.title('Quasiparticle weigth')
    plt.ylabel('Z')
    plt.xlabel('U/D')
    fig.savefig(out_file+'_Z.png', format='png',
                transparent=False, bbox_inches='tight', pad_inches=0.05)
#    plt.close(fig)
