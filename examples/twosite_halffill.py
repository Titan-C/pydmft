# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 15:12:44 2014

@author: oscar
"""

from __future__ import division, absolute_import, print_function
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from dmft.twosite import dmft_loop, twosite_matsubara

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

def movie_feature(res, name, feature):
    """Outputs an animate movie of the evolution of an specific feature"""
    figi, ax = plt.subplots()
    line, = ax.plot([], [], '*-')
    ax.set_ylim(-1e-8, 0)
    beta = res[0,2].beta
    sim = twosite_matsubara(beta, res[0, 2].t,  30*beta)

    iwn = np.arange(1, 30*beta, 2) / beta
    ax.set_xlim(0, iwn.max())
    ax.set_xlabel('$i\\omega_n$')
    ax.set_ylabel(r'$Im \Sigma$')
    ax.set_title('Evolution of the Self Energy at $\\beta=${}'.format(beta))


    def run(i):
        # update the data
        u_int = res[i, 0]
        sim.mu = u_int / 2.
        sim.solve(u_int/2., u_int, res[i, 2].hyb_V())
        s = sim.GF[r'$\Sigma$'].imag
        ymin, ymax = ax.get_ylim()

        if s.min() <= ymin and s.min() >= -12:
            ax.set_ylim(np.max([2*s.min(), -12]), 0)
            ax.figure.canvas.draw()
        line.set_data(iwn, s)
        plt.legend([line], ['U={:.2f}'.format(u_int)])
        return line,

    ani = anim.FuncAnimation(figi, run, blit=True, interval=150,
                             frames=res.shape[0])
    ani.save(name+'.mp4')
    plt.close(figi)

from slaveparticles.quantum import dos

def movie_feature_real(res, name, feature):
    """Outputs an animate movie of the evolution of an specific feature"""
    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    line, = ax1.plot([], [], '--')
    ax1.set_xlim([-6, 6])
    ax1.set_ylim([-6, 6])
    line2, = ax2.plot([], [], '-')
    ax2.set_ylim([0, 0.66])

    beta = res[0, 2].beta
    ax1.set_xlabel('$\\omega$')
    ax1.set_ylabel(r'$\Sigma(\omega)$')
    ax2.set_ylabel(r'$A(\omega)$')
    ax1.set_title('Evolution of the Self Energy at $\\beta=${}'.format(beta))
    f.subplots_adjust(hspace=0)
    def init():
        line.set_data([], [])
        line2.set_data([], [])
        return line, line2,

    def run(i):
        # update the data
        u_int = res[i, 0]
        w = res[i, 2].omega
        s = res[i, 2].GF[r'$\Sigma$']
        ra = w+u_int/2.-s
        rho = dos.bethe_lattice(ra, res[i, 2].t)

        line.set_data(w, s)
        line2.set_data(w, rho)
        plt.legend([line], ['U={:.2f}'.format(u_int)])
        return line, line2,

    ani = anim.FuncAnimation(f, run, blit=True, interval=150, init_func=init,
                             frames=res.shape[0])
    ani.save(name+'.mp4')
    plt.close(f)

def run_halffill(axis = 'matsubara'):
    fig = plt.figure()
    u_int = np.arange(0, 6.2, 0.01)
    for beta in [6, 10, 20, 30, 50, 100, 1e3]:
        out_file = axis+'_halffill_b{}_dU{}'.format(beta, 0.01)
        try:
            res = np.load(out_file+'.npy')
        except IOError:
            res = dmft_loop(u_int, axis, beta=beta, hop=1)
            np.save(out_file, res)

#        plot_feature(res, out_file, 'A')
#        plot_feature(res, out_file, 'sigma')
#        ste=118
#        w=res[ste,2].omega.imag
#        s = res[ste, 2].GF[r'$\Sigma$'].imag
#        plt.plot(w,s,'+--', label=r'U={}, $\beta$={}'.format(res[ste,0],res[ste,2].beta))
        movie_feature_real(res, out_file, 'sigma')
        plt.plot(res[:, 0]/2, res[:, 1], '+-', label='$\\beta = {}$'.format(beta))
    #    plt.plot(u_int, 1-u_int.clip(0, 3)**2/9, '--', label='$1-U^2/U_c^2')
    plt.legend(loc=0)
#    plt.xlim([0,30])
#    plt.ylim([-12,0])

    plt.title('Quasiparticle weigth, estimated in real freq')
    plt.ylabel('Z')
    plt.xlabel('U/D')
    fig.savefig(out_file+'_Z.png', format='png',
                transparent=False, bbox_inches='tight', pad_inches=0.05)
#    plt.close(fig)

if __name__ == "__main__":
#    run_halffill()
    run_halffill('real')
#    fig = plt.figure()
#    axis = 'real'
#    u_int = np.arange(0, 3.2, 0.05)
#    beta = 1e5
#    for fill in np.arange(1, 0.9, -0.025):
#        out_file = 'real_{}fill_b{}'.format(fill, beta)
#        try:
#            res = np.load(out_file+'.npy')
#        except IOError:
#            res = dmft_loop(u_int, axis=axis, beta=beta, hop=0.5, filling=fill)
#            np.save(out_file, res)
#
#        plot_feature(res, out_file, 'A')
#        plot_feature(res, out_file, 'sigma')
#
#
#        plt.plot(res[:, 0], res[:, 1], '+-', label='$\\beta = {}$'.format(beta))
#    #    plt.plot(u_int, 1-u_int.clip(0, 3)**2/9, '--', label='$1-U^2/U_c^2')
#    plt.legend()
#    plt.title('Quasiparticle weigth')
#    plt.ylabel('Z')
#    plt.xlabel('U/D')
#    fig.savefig(out_file+'_Z.png', format='png',
#                transparent=False, bbox_inches='tight', pad_inches=0.05)
#    plt.close(fig)
