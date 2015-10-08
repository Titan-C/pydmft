# -*- coding: utf-8 -*-
"""
Anti-Ferromagnetism in with IPT
===============================

Abusing the use limits of IPT to get a magnetic solution

"""
from __future__ import division, absolute_import, print_function

from pytriqs.gf.local import GfImFreq, GfImTime, GfReFreq, \
    inverse, Omega, iOmega_n, Wilson, SemiCircular, InverseFourier, Fourier

from pytriqs.plot.mpl_interface import oplot
from pytriqs.plot.mpl_interface import subplots
import matplotlib.pyplot as plt
import numpy as np


class IPTSolver(object):

    def __init__(self, **params):
        self.beta = params['beta']

        # Matsubara frequency
        self.gup = GfImFreq(indices=[0], beta=self.beta)
        self.gdw = self.gup.copy()
        self.g0up = self.gup.copy()
        self.g0dw = self.gup.copy()
        self.sigmaup = self.gup.copy()
        self.sigmadw = self.gup.copy()

        # Imaginary time
        self.g0tup = GfImTime(indices=[0], beta=self.beta)
        self.g0tdw = self.g0tup.copy()
        self.sigmatup = self.g0tup.copy()
        self.sigmatdw = self.g0tup.copy()

    def solve(self, U):

        self.g0tup <<= InverseFourier(self.g0up)
        self.g0tdw <<= InverseFourier(self.g0dw)
        self.sigmatup <<= (U**2) * self.g0tup * self.g0tdw * self.g0tdw
        self.sigmatdw <<= (U**2) * self.g0tdw * self.g0tup * self.g0tup
        self.sigmaup <<= Fourier(self.sigmatup)
        self.sigmadw <<= Fourier(self.sigmatdw)

        # Dyson equation to get G
        self.gup <<= self.g0up * inverse(1.0 - self.sigmaup * self.g0up)
        self.gdw <<= self.g0dw * inverse(1.0 - self.sigmadw * self.g0dw)

    def dmft_loop(self, n_loops, t, U):
        for i in range(n_loops):
            self.g0up <<= inverse(iOmega_n - t**2 * self.gdw)
            self.g0dw <<= inverse(iOmega_n - t**2 * self.gup)
            self.solve(U)


def ev_on_loops(n_loops, U, beta, t):
    """Studies change of Green's function depending on DMFT loop count"""
# open 2 panels top (t) and bottom (b)

    plt_env = subplots(3, 1)

    for loop in n_loops:
        S = IPTSolver(beta=beta)
        S.gup <<= inverse(iOmega_n + 0.1 + 0.5j)

        S.gdw <<= inverse(iOmega_n - 0.1 + 0.5j)

        S.dmft_loop(loop, t, U)
        plot_GSG(S, loop, plt_env)

    plt_env[1][0].set_ylim(-2, 0)
    plt_env[0].savefig('gf_{:.2f}.png'.format(U), format='png',
                       transparent=False, bbox_inches='tight', pad_inches=0.05)
    plt.close(plt_env[0])


def plot_GSG(S, loop, plt_env):
    """Takes an IPT Solver and plots 3 subplots showing the selfenergy
       , the greensfunction in matsubara frequencies and the greens function
       int the real axis"""
    nl = ""#"  nl={}".format(loop)

    for i, ax in enumerate(plt_env[1]):
        if i == 0:
            grealup = GfReFreq(indices=[1], window=(-4, 4), n_points=400)
            grealdw = grealup.copy()
            grealup.set_from_pade(S.gup, 201, 0.0)
            grealdw.set_from_pade(S.gdw, 201, 0.0)
            ax.oplot(grealup, RI='S', label="$\uparrow$"+nl)
            ax.oplot(grealdw, '--', RI='S',
                     label="$\downarrow$"+nl)
            ax.set_ylabel('$A(\omega)$')
        if i == 1:
            ax.oplot(S.gup, '-o', RI='I', x_window=(0, 5),
                     label="$\uparrow$"+nl)
            ax.oplot(S.gdw, '-+', RI='I', x_window=(0, 5),
                     label="$\downarrow$"+nl)
        if i == 2:
            ax.oplot(S.sigmaup, '-o', RI='I', x_window=(0, 5),
                     label="$\uparrow$"+nl)
            ax.oplot(S.sigmadw, '-+', RI='I', x_window=(0, 5),
                     label="$\downarrow$"+nl)
            ax.set_ylabel('$\Sigma(i\omega_n)$')
        if i == 3:
            ax.oplot(S.g0tup, '-', label="$\uparrow$"+nl)
            ax.oplot(S.g0tdw, '--', label="$\downarrow$"+nl)

    plt_env[1][0].set_title("Green's functions, IPT, Bethe lattice,"
                            "$\\beta={:.2f}$, $U={:.2f}$".format(beta, U))


def AFM_follow(S, t, U, loops):
    """Follows the Metal to insulator transition promoting keeping old GF"""

    S.gup <<= inverse(iOmega_n + 0.1 + 0.5j)
    S.gdw <<= inverse(iOmega_n - 0.1 + 0.5j)
    S.dmft_loop(loops, t, U)

    plot_GSG(S, loops, subplots(4, 1))


if __name__ == "__main__":
    AFM_follow(S=IPTSolver(beta=40), t=0.5, U=2.7, loops=55)
