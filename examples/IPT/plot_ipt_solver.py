# -*- coding: utf-8 -*-
"""
Simple IPT solver
=================

Using triqs
"""
from __future__ import division, absolute_import, print_function
import matplotlib.pyplot as plt
from pytriqs.gf.local import GfImFreq, GfImTime, GfReFreq, \
    inverse, Omega, iOmega_n, Wilson, SemiCircular,InverseFourier, Fourier

from pytriqs.plot.mpl_interface import oplot
from pytriqs.plot.mpl_interface import subplots
import numpy as np
from dmft.twosite import matsubara_Z
import matplotlib.pyplot as plt

class IPTSolver(object):

    def __init__(self, **params):

        self.U = params['U']
        self.beta = params['beta']

        # Matsubara frequency
        self.g = GfImFreq(indices=[0], beta=self.beta)
        self.g0 = self.g.copy()
        self.sigma = self.g.copy()

        # Imaginary time
        self.g0t = GfImTime(indices=[0], beta=self.beta)
        self.sigmat = self.g0t.copy()

    def solve(self):

        self.g0t << InverseFourier(self.g0)
        self.sigmat << (self.U**2) * self.g0t * self.g0t * self.g0t
        self.sigma << Fourier(self.sigmat)

        # Dyson equation to get G
        self.g << self.g0 * inverse(1.0 - self.sigma * self.g0)

    def dmft_loop(self, t, max_loops):
        converged = False
        loops = 0
        while not converged:
            oldg = self.g.data.copy()
            self.g0 << inverse( iOmega_n - t**2 * self.g)
            self.solve()
            converged = np.allclose(self.g.data, oldg, atol=1e-3)
            loops += 1
            if loops > max_loops:
                converged = True

        return loops


def ev_on_loops(max_loops, U, beta, t):
    """Studies change of Green's function depending on DMFT loop count"""
# open 2 panels top (t) and bottom (b)

    f, (ax1, ax2, ax3) = subplots( 3,1)

    for loop in max_loops:
        S = IPTSolver(U = U, beta = beta)
        S.g << SemiCircular(2*t)
        nloop = S.dmft_loop(t, loop)
        ax1.oplot(S.sigma, RI='I', x_window  = (0,2), label = "nloop={}".format(nloop))
        ax1.set_ylabel('$\Sigma(i\omega_n)$')
        ax2.oplot(S.g, RI='I', x_window  = (0,2), label = "nloop={}".format(nloop))
        greal = GfReFreq(indices=[1], window=(-4,4), n_points=400)
        greal.set_from_pade(S.g,201,0.0)
        ax3.oplot(greal,RI='I', label = "nloop={}".format(nloop))

    ax1.set_title("Matsubara Green's functions, IPT, Bethe lattice, $\\beta=%.2f$, $U=%.2f$"%(beta,U))


def ev_on_interaction(U, beta, t, max_loops=300):
    """Studies change of Green's function depending on DMFT interaction"""

    u_zet = []
    for u_int in U:
#        S.U = u_int

        S = IPTSolver(U = u_int, beta = beta)
        S.g << SemiCircular(2*t)
        nloop = S.dmft_loop(t, max_loops)
        u_zet.append(matsubara_Z(S.sigma.data[:,0, 0].imag, beta))
#        oplot(S.g, label = "U={}".format(u_int))

#    plt.title("Matsubara Green's functions, IPT, Bethe lattice, $\\beta=%.2f$".format(beta))
    return u_zet


if __name__ == "__main__":
#    ev_on_loops([1,2,5,10, 20], 3, 200, 0.5)
    U = np.linspace(0, 4, 50)
#    U = np.concatenate((U, U[-2:11:-1]))

    for beta in [150, 300.]:
        zet=ev_on_interaction(U, beta, 0.5)
        plt.plot(U, zet, label='$\\beta={}$'.format(beta))
    plt.title('Hysteresis loop of the quasiparticle weigth')
    plt.legend()
    plt.ylabel('Z')
    plt.xlabel('U/D')
