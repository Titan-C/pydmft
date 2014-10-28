# -*- coding: utf-8 -*-
"""
@author: Óscar Nájera
Created on Tue Oct 28 16:33:14 2014
"""
from __future__ import division, absolute_import, print_function

from pytriqs.gf.local import GfImFreq, GfImTime, GfReFreq, \
    inverse, Omega, iOmega_n, Wilson, SemiCircular,InverseFourier, Fourier

from pytriqs.plot.mpl_interface import oplot
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

        self.g0t <<= InverseFourier(self.g0)
        self.sigmat <<= (self.U**2) * self.g0t * self.g0t * self.g0t
        self.sigma <<= Fourier(self.sigmat)

        # Dyson equation to get G
        self.g <<= self.g0 * inverse(1.0 - self.sigma * self.g0)

    def dmft_loop(self, n_loops, t):
        for i in range(n_loops):
            self.g0 <<= inverse( iOmega_n - t**2 * self.g)
            self.solve()


if __name__ == "__main__":
    fig = plt.figure(figsize=(6,6))
    t = 0.5
    beta = 50
    U = 3.0
    n_loops = 40
    for loop in [10, 20, 25, 30, 40, 50]:
        S = IPTSolver(U = U, beta = beta)
        S.g <<= SemiCircular(2*t)
        S.dmft_loop(loop, t)

            # Get the real-axis with Pade approximants
#        greal = GfReFreq(indices = [1], window = (-4.0,4.0), n_points = 400)
#        greal.set_from_pade(S.g, 201, 0.0)

        # Generate the plot

        oplot(S.g, figure = fig, label = "nloop={}".format(loop))

    plt.title("Matsubara Green's functions, IPT, Bethe lattice, $\\beta=%.2f$, $U=%.2f$"%(beta,U))
#    plt.xlim(-4,4)
#    plt.ylim(0,0.7)
#    plt.ylabel("$A(\omega)$")
#    plt.title("Local DOS, IPT, Bethe lattice, $\\beta=%.2f$, $U=%.2f$"%(beta,U))

#    # Save the plot in a file
#    fig.savefig("dos_%s"%U, format="png", transparent=False)
#    dos_files.append("dos_%s"%U)
#    plt.close(fig)
