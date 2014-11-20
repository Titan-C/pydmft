# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 15:12:44 2014

@author: oscar
"""

from __future__ import division, absolute_import, print_function
import matplotlib.pyplot as plt
import numpy as np
from twosite import twosite, out_plot

if __name__ == "__main__":
    res = []
    u_int = np.arange(0, 4, 0.2)
    for U in u_int:

        sim = twosite(80, 0.5, 'real')
        hyb = 0.05  # np.sqrt(sim.imp_z()*sim.m2)
        fig = plt.figure()
        for i in range(80):
            old = hyb
            hyb = sim.solve(U/2, U/2, U, old)
            out_plot(sim, 'A', 'loop {} hyb {}'.format(i, hyb))
            if np.abs(old - hyb) < 1e-6:
                break

        plt.legend()
        plt.title('U={}, hyb={}'.format(U, hyb))
        plt.ylabel('A($\omega$)')
        plt.xlabel('$\omega$')
        fig.savefig('Aw_halffill_Ins{:.2f}.png'.format(U), format='png',
                    transparent=False, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        res.append((U, sim))
    np.save('realax_halffill_ins', res)

    for U, sim in res:
        fig = plt.figure()
        out_plot(sim, 'A', '')

        plt.legend()
        plt.title('U={}, hyb={}'.format(U, np.sqrt(sim.imp_z()*sim.m2)))
        plt.ylabel('A($\omega$)')
        plt.xlabel('$\omega$')
        fig.savefig('Aw_halffill_end_Ins{:.2f}.png'.format(U), format='png',
                    transparent=False, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)

    fig = plt.figure()
    zet = []
    for U, sim in res:
        zet.append(sim.imp_z())
    plt.plot(u_int, zet,'+-', label='2 site DMFT')
    plt.plot(u_int, 1-u_int.clip(0,3)**2/9, '--', label='Gutwiller $1-U^2/U_c^2)
    plt.legend()
    plt.title('Quasiparticle weigth of the impurity')
    plt.ylabel('Z')
    plt.xlabel('U/D')
    fig.savefig('Quasiparticle.png', format='png',
                transparent=False, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
