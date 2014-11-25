# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 15:12:44 2014

@author: oscar
"""

from __future__ import division, absolute_import, print_function
import matplotlib.pyplot as plt
import numpy as np
from dmft.twosite import lattice_gf, out_plot, metallic_loop


def plot_spectalfunc(res, name):
    for U, zet, sim in res:
        fig = plt.figure()
        sim.GF['Lat G'] = lattice_gf(sim, U/2)
        out_plot(sim, 'A', '$(\\omega)$')
        plt.plot(sim.omega, sim.interacting_dos(U/2), '--',
                 lw=2, label='$\\rho(\\omega)$')

        plt.legend()
        plt.title('U={:.4f}, hyb={:.4f}'.format(U, np.sqrt(zet*sim.m2)))
        plt.xlabel('$\\omega$')
        plt.ylim([0, 0.7])
        fig.savefig(name+'rhow_U{:.2f}.png'.format(U), format='png',
                    transparent=False, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)


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
        plt.ylim([0, 0.7])
        fig.savefig('{}_{}_U{:.2f}.png'.format(name, feature, U), format='png',
                    transparent=False, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)

if __name__ == "__main__":
    axis = 'matsubara'
    beta = 10
    out_file = axis+'_halffill_b'+str(beta)
    u_int = np.arange(0, 3.2, 0.05)
    try:
        res = np.load(out_file+'.npy')
    except IOError:
        res = metallic_loop(u_int, axis=axis, beta=beta, hop=0.5)
        np.save(out_file, res)

#    plot_spectalfunc(res, out_file)
    plot_feature(res, out_file, 'sigma')


    fig = plt.figure()
    zet = res[:, 1]
    plt.plot(u_int, zet, '+-', label='2 site DMFT')
    plt.plot(u_int, 1-u_int.clip(0, 3)**2/9, '--', label='$1-U^2/U_c^2')
    plt.legend()
    plt.title('Quasiparticle weigth of the impurity')
    plt.ylabel('Z')
    plt.xlabel('U/D')
    fig.savefig('Quasiparticle_ins.png', format='png',
                transparent=False, bbox_inches='tight', pad_inches=0.05)
#    plt.close(fig)
