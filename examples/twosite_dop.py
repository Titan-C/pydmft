# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 15:12:44 2014

@author: oscar
"""

from __future__ import division, absolute_import, print_function
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from dmft.twosite_dop import twosite_real_dop
import copy

from slaveparticles.quantum import dos

def movie_feature_real(res, name):
    """Outputs an animate movie of the evolution of an specific feature"""
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    line, = ax1.plot([], [], '--')
    ax1.set_xlim([-6, 6])
    ax1.set_ylim([0, 0.66])

    line2, = ax2.plot([], [], '-')
    ax2.set_ylim([-6, 6])

    line3, = ax3.plot([], [], '-')
    ax3.set_ylim([-6, 6])

    beta = res[0, 2].beta
    ax1.set_xlabel('$\\omega$')
    ax1.set_ylabel(r'$A(\omega)$')
    ax2.set_ylabel(r'$\Sigma(\omega)$')
    ax3.set_ylabel(r'$G_{imp}(\omega)$')
    ax2.set_title('Evolution under doping at $\\beta=${}'.format(beta))

    f.subplots_adjust(hspace=0)

    def init():
        line.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        return line, line2, line3,

    def run(i):
        u_int = res[i, 0]
        w = res[i, 2].omega
        s = res[i, 2].GF[r'$\Sigma$']
        g = res[i, 2].GF['Imp G']
        ra = w+u_int/2.-s
        rho = dos.bethe_lattice(ra, res[i, 2].t)

        line.set_data(w, rho)
        plt.legend([line], ['U={:.2f}'.format(u_int)])
        line2.set_data(w, s)
        line3.set_data(w, g)
        return line, line2, line3

    ani = anim.FuncAnimation(f, run, blit=True, interval=150, init_func=init,
                             frames=res.shape[0])
    ani.save(name+'.mp4')
    plt.close(f)


def run_halffill(axis='matsubara'):
    fig = plt.figure()
    du = 0.05
    u_int = np.arange(0, 6.2, du)
    for beta in [6, 10, 20, 30, 50, 100, 1e3]:
        out_file = axis+'_halffill_b{}_dU{}'.format(beta, du)
        try:
            res = np.load(out_file+'.npy')
        except IOError:
            res = dmft_loop(u_int, axis, beta=beta, hop=1)
            np.save(out_file, res)

        if axis == 'real':
            movie_feature_real(res, out_file)
        movie_feature(res, out_file)
        plt.plot(res[:, 0]/2, res[:, 1], '+-', label='$\\beta = {}$'.format(beta))
    #    plt.plot(u_int, 1-u_int.clip(0, 3)**2/9, '--', label='$1-U^2/U_c^2')
    plt.legend(loc=0)

    plt.title('Quasiparticle weigth, estimated in real freq')
    plt.ylabel('Z')
    plt.xlabel('U/D')
    fig.savefig(out_file+'_Z.png', format='png',
                transparent=False, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

if __name__ == "__main__":
    res = np.load('real_halffill_b1000.0_dU0.05.npy')
    filling = np.arange(1, 0.9, -0.02)
    log = []
    for U, z, sim in res:
        doper = twosite_real_dop()
        doper.solve(sim.e_c, U, sim.hyb_V())
        for n in filling:
            doper.selfconsistency(doper.e_c, doper.hyb_V(), n, U)
            log.append([U, n, copy.deepcopy(doper)])

    log = np.asarray(log)

