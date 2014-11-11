# -*- coding: utf-8 -*-
"""
@author: Óscar Nájera
Created on Thu Nov 06 15:11:14 2014
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

    f, (ax1, ax2, ax3) = subplots(3, 1)

    for loop in n_loops:
        S = IPTSolver(beta=beta)
        S.gup <<= inverse(iOmega_n + 0.1 + 0.5j)

        S.gdw <<= inverse(iOmega_n - 0.1 + 0.5j)

        S.dmft_loop(loop, t, U)
        ax1.oplot(S.sigmaup, '-o', RI='I', x_window=(0, 2),
                  label="$\uparrow$ nl={}".format(loop))
        ax1.oplot(S.sigmadw, '-o', RI='I', x_window=(0, 2),
                  label="$\downarrow$ nl={}".format(loop))
        ax1.set_ylabel('$\Sigma(i\omega_n)$')

        ax2.oplot(S.g0tup, label="$\uparrow$ nl={}".format(loop))
        ax2.oplot(S.g0tdw, label="$\downarrow$ nl={}".format(loop))

        grealup = GfReFreq(indices=[1], window=(-4, 4), n_points=400)
        grealdw = grealup.copy()
        grealup.set_from_pade(S.gup, 201, 0.0)
        grealdw.set_from_pade(S.gdw, 201, 0.0)
        ax3.oplot(grealup, RI='I', label="$\uparrow$  nl={}".format(loop))
        ax3.oplot(grealdw, '--', RI='I',
                  label="$\downarrow$ nl={}".format(loop))
        ax3.set_ylim(-2, 0)

    ax1.set_title("Green's functions, IPT, Bethe lattice,"
                  "$\\beta={:.2f}$, $U={:.2f}$".format(beta, U))
    f.savefig('gf_{:.2f}.png'.format(U), format='png',
              transparent=False, bbox_inches='tight', pad_inches=0.05)
    plt.close(f)


if __name__ == "__main__":
    for u_int in np.linspace(0, 4, 40):
        ev_on_loops([20], u_int, 35, 0.5)

# Get the real-axis with Pade approximants
#        greal = GfReFreq(indices = [1], window = (-4.0,4.0), n_points = 400)
#        greal.set_from_pade(S.g, 201, 0.0)

        # Generate the plot


#    plt.xlim(-4,4)
#    plt.ylim(0,0.7)
#    plt.ylabel("$A(\omega)$")
#    plt.title("Local DOS, IPT, Bethe lattice, $\\beta=%.2f$, $U=%.2f$"%(beta,U))

#    # Save the plot in a file
#    fig.savefig("dos_%s"%U, format="png", transparent=False)
#    dos_files.append("dos_%s"%U)
#    plt.close(fig)

