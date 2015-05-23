# -*- coding: utf-8 -*-
"""
Dimer Bethe lattice
===================

Non interacting dimer of a Bethe lattice
"""

from pytriqs.gf.local import GfImFreq, GfImTime, InverseFourier, \
Fourier, iOmega_n, inverse, SemiCircular
#GfReFreq, Omega, Wilson, inverse
import numpy

class IPT_dimer_Solver:

    def __init__(self, **params):

        self.U = params['U']
        self.beta = params['beta']

        # Matsubara frequency(indices = ['s','d'], window = (-2, 2), n_points = 1000, name = "$G_\mathrm{s+d}$")
        self.g_iw = GfImFreq(indices = ['A','B'], beta=self.beta)
        self.g0_iw = self.g_iw.copy()
        self.sigma_iw = self.g_iw.copy()

        # Imaginary time
        self.g0_tau = GfImTime(indices = ['A','B'], beta = self.beta)
        self.sigma_tau = self.g0_tau.copy()

    def solve(self):

        self.g0_tau << InverseFourier(self.g0_iw)
        self.sigma_tau << (self.U**2) * self.g0_tau * self.g0_tau * self.g0_tau
        self.sigma_iw << Fourier(self.sigma_tau)

        # Dyson equation to get G
        self.g_iw << inverse(inverse(self.g0_iw) - self.sigma_iw)

mu, t = 0.1, 0.5
t2 = t**2
tab = 0.1
beta = 50.

g_iw = GfImFreq(indices=['A', 'B'], beta=beta)
g_iw['A', 'A'] = iOmega_n + mu
g_iw['A', 'B'] = -tab
g_iw['B', 'A'] = -tab
g_iw['B', 'B'] = iOmega_n + mu
g_iw << g_iw + t2 * SemiCircular(2*t)
g_iw.invert()
