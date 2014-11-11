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


class IPTSolver(object):

    def __init__(self, **params):

        self.U = params['U']
        self.beta = params['beta']

        # Matsubara frequency
        self.gup = GfImFreq(indices=[0], beta=self.beta)
        self.gdw = self.gup.copy()
        self.g0up = self.gup.copy()
        self.g0dw = self.gup.copy()
        self.sigma = self.gup.copy()

        # Imaginary time
        self.g0t = GfImTime(indices=[0], beta=self.beta)
        self.sigmat = self.g0t.copy()

    def solve(self):

        self.g0t <<= InverseFourier(self.g0up)
        self.sigmat <<= (self.U**2) * self.g0t * self.g0t * self.g0t
        self.sigma <<= Fourier(self.sigmat)

        # Dyson equation to get G
        self.gup <<= self.g0up * inverse(1.0 - self.sigma * self.g0up)
        self.gdw <<= self.g0dw * inverse(1.0 - self.sigma * self.g0dw)

    def dmft_loop(self, n_loops, t):
        for i in range(n_loops):
            self.g0up <<= inverse(iOmega_n - t**2 * self.gdw)
            self.g0dw <<= inverse(iOmega_n - t**2 * self.gup)
            self.solve()


def ev_on_loops(n_loops, U, beta, t):
    """Studies change of Green's function depending on DMFT loop count"""
# open 2 panels top (t) and bottom (b)

    f, (ax1, ax2, ax3) = subplots(3, 1)

    for loop in n_loops:
        S = IPTSolver(U=U, beta=beta)
        S.gup <<= inverse(iOmega_n +0.1+ 0.5j)

        S.gdw <<= inverse(iOmega_n + 0.4j)

        S.dmft_loop(loop, t)
        ax1.oplot(S.sigma, '-o', RI='I', x_window=(0, 2),
                  label="nloop={}".format(loop))
        ax1.set_ylabel('$\Sigma(i\omega_n)$')

        ax2.oplot(S.g0t, label="nloop={}".format(loop))

        grealup = GfReFreq(indices=[1], window=(-4, 4), n_points=400)
        grealdw = grealup.copy()
        grealup.set_from_pade(S.gup, 201, 0.0)
        grealdw.set_from_pade(S.gdw, 201, 0.0)
        ax3.oplot(grealup, RI='I', label="$\uparrow$  nloop={}".format(loop))
        ax3.oplot(grealdw, '--', RI='I',
                  label="$\downarrow$ nloop={}".format(loop))

    ax1.set_title("Matsubara Green's functions, IPT, Bethe lattice, $\\beta=%.2f$, $U=%.2f$" % (beta, U))


def ev_on_interaction(n_loops, U, beta, t):
    """Studies change of Green's function depending on DMFT interaction"""


    for u_int in U:
        S = IPTSolver(U = u_int, beta = beta)
        S.g <<= SemiCircular(2*t)
        S.dmft_loop(n_loops, t)
        oplot(S.g, label = "U={}".format(u_int))

    plt.title("Matsubara Green's functions, IPT, Bethe lattice, $\\beta=%.2f$, $loop=%.2f$".format(beta,n_loops))

if __name__ == "__main__":
    ev_on_loops([15], 2., 55, 0.5)


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

