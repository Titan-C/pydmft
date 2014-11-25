# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 15:12:44 2014

@author: oscar
"""

from __future__ import division, absolute_import, print_function
import matplotlib.pyplot as plt
import numpy as np
from dmft.twosite import lattice_gf, out_plot, metallic_loop

if __name__ == "__main__":
    out_file = 'realax_halffill.npy'
    u_int = np.arange(0, 3.2, 0.05)
    try:
        res = np.load(out_file)
    except IOError:
        res = metallic_loop(u_int, axis='real', beta=1e5, hop=0.5)
        np.save(out_file, res)

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
        fig.savefig('rhow_halffill_ins{:.2f}.png'.format(U), format='png',
                    transparent=False, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)

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
    plt.close(fig)
