# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 15:12:44 2014

@author: oscar
"""

from __future__ import division, absolute_import, print_function
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from dmft.twosite_dop import dmft_loop_dop
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

    beta = res[0, 1].beta
    ax1.set_xlabel('$\\omega$')
    ax1.set_ylabel(r'$A(\omega)$')
    ax2.set_ylabel(r'$\Sigma(\omega)$')
    ax3.set_ylabel(r'$G_{imp}(\omega)$')

    f.subplots_adjust(hspace=0)

    def init():
        line.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        return line, line2, line3,

    def run(i):
        w = res[i, 1].omega
        s = res[i, 1].GF[r'$\Sigma$']
        g = res[i, 1].GF['Imp G']
        ra = w + res[i, 1].mu - s
        rho = dos.bethe_lattice(ra, res[i, 1].t)

        line.set_data(w, rho)
        ax1.set_title('Evolution under doping n={} at $\\beta=${}'.format(res[i, 0],beta))
        line2.set_data(w, s)
        line3.set_data(w, g)
        return line, line2, line3

    ani = anim.FuncAnimation(f, run, blit=True, interval=150, init_func=init,
                             frames=res.shape[0])
    ani.save(name+'.mp4')
    plt.close(f)


def doping_config(res, name):
    fig, axes = plt.subplots(3, sharex=True)
    axes[-1].set_xlabel('$<N>_{imp}$')
    fill = res[:, 0]
    axes[0].set_xlim([0, 1])
    e_c = [sim.e_c for sim in res[:, 1]]
    V = [sim.hyb_V() for sim in res[:, 1]]
    mu = [sim.mu for sim in res[:, 1]]
    for feat, ax, lab in zip([e_c, V, mu], axes, ['$\\epsilon_c$', 'V', '$\\mu$']):
        ax.plot(fill, feat, label=lab)
        ax.set_ylabel(lab)

    fig.savefig(name+'_bathparam.png', format='png',
                transparent=False, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

def run_dop(axis='real', beta=1e3, u_int=[1., 2., 4.0]):#, 4, 6, 8, 10, 100]):
    fig = plt.figure()
    for u in u_int:
        out_file = axis+'_dop_b{}_U{}'.format(beta, u)
        try:
            res = np.load(out_file+'.npy')
        except IOError:
            res = dmft_loop_dop(u, u/2, 0.8, np.arange(u/2, np.min([-2,u/2]), -0.1))
            np.save(out_file, res)

        doping_config(res, out_file)
#        movie_feature_real(res, out_file)
        zet = [sim.imp_z() for sim in res[:, 1]]
        plt.plot(res[:, 0], zet, '+-', label='$U/t= {}$'.format(u))

    plt.legend(loc=0)
    plt.title('Quasiparticle weigth, estimated in real freq at $\\beta={}$'.format(beta))
    plt.ylabel('Z')
    plt.xlabel('n')
    fig.savefig(out_file+'_Z.png', format='png',
                transparent=False, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

if __name__ == "__main__":
    run_dop()
#    movie_feature_real(res, 'dopU4')