# -*- coding: utf-8 -*-
"""
@author: Óscar Nájera
Created on Tue Oct 28 16:33:14 2014
"""
from __future__ import division, absolute_import, print_function
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/oscar/libs/lib/python2.7/site-packages')
from pytriqs.gf.local import GfImFreq, GfImTime, GfReFreq, \
    inverse, Omega, iOmega_n, Wilson, SemiCircular,InverseFourier, Fourier

from pytriqs.plot.mpl_interface import oplot
from pytriqs.plot.mpl_interface import subplots

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

    def dmft_loop(self, n_loops, t):
        for i in range(n_loops):
            self.g0 << inverse( iOmega_n - t**2 * self.g)
            self.solve()

def ev_on_loops(n_loops, U, beta, t):
    """Studies change of Green's function depending on DMFT loop count"""
# open 2 panels top (t) and bottom (b)

    f, (ax1, ax2, ax3) = subplots( 3,1)

    for loop in n_loops:
        S = IPTSolver(U = U, beta = beta)
        S.g << SemiCircular(2*t)
        S.dmft_loop(loop, t)
        ax1.oplot(S.sigma, RI='I', x_window  = (0,2), label = "nloop={}".format(loop))
        ax1.set_ylabel('$\Sigma(i\omega_n)$')
        ax2.oplot(S.g, RI='I', x_window  = (0,2), label = "nloop={}".format(loop))
        greal = GfReFreq(indices=[1], window=(-4,4), n_points=400)
        greal.set_from_pade(S.g,201,0.0)
        ax3.oplot(greal,RI='I', label = "nloop={}".format(loop))

    ax1.set_title("Matsubara Green's functions, IPT, Bethe lattice, $\\beta=%.2f$, $U=%.2f$"%(beta,U))


def ev_on_interaction(n_loops, U, beta, t):
    """Studies change of Green's function depending on DMFT interaction"""


    for u_int in U:
        S = IPTSolver(U = u_int, beta = beta)
        S.g << SemiCircular(2*t)
        S.dmft_loop(n_loops, t)
        oplot(S.g, label = "U={}".format(u_int))

    plt.title("Matsubara Green's functions, IPT, Bethe lattice, $\\beta=%.2f$, $loop=%.2f$".format(beta,n_loops))

if __name__ == "__main__":
    ev_on_loops([1,2,5,10, 20], 3, 20, 0.5)


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

