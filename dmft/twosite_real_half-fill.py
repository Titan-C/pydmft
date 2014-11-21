# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 15:12:44 2014

@author: oscar
"""

from __future__ import division, absolute_import, print_function
import matplotlib.pyplot as plt
import numpy as np
from twosite import twosite, lattice_gf, out_plot

if __name__ == "__main__":
    res = []
    u_int = np.arange(0, 3.2, 0.05)
    hyb = 0.4
    for U in u_int:
        sim = twosite(1e5, 0.5, 'real')
        for i in range(80):
            old = hyb
            hyb = sim.solve(U/2, U/2, U, old)
            if 2.5 < U < 3:
                hyb = (hyb*0.7 + .3*old)
            if np.abs(old - hyb) < 1e-4:
                break

        hyb = sim.solve(U/2, U/2, U, hyb)
        res.append((U, sim))
    np.save('realax_halffill_ins', res)

    for U, sim in res:
        fig = plt.figure()
        sim.GF['Lat G'] = lattice_gf(sim, U/2)
        out_plot(sim, 'A', '')

        plt.legend()
        plt.title('U={}, hyb={:.4f}'.format(U, np.sqrt(sim.imp_z()*sim.m2)))
        plt.ylabel('A($\omega$)')
        plt.xlabel('$\omega$')
        fig.savefig('Aw_halffill_end{:.2f}.png'.format(U), format='png',
                    transparent=False, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)

    fig = plt.figure()
    zet = []
    for U, sim in res:
        zet.append(sim.imp_z())
    plt.plot(u_int[:len(zet)], zet, '+-', label='2 site DMFT')
    plt.plot(u_int, 1-u_int.clip(0,3)**2/9, '--', label='$1-U^2/U_c^2')
    plt.legend()
    plt.title('Quasiparticle weigth of the impurity')
    plt.ylabel('Z')
    plt.xlabel('U/D')
    fig.savefig('Quasiparticle.png', format='png',
                transparent=False, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
