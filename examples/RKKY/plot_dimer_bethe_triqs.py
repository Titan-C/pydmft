# -*- coding: utf-8 -*-
"""
Dimer Bethe lattice
===================

Non interacting dimer of a Bethe lattice
"""

from pytriqs.gf.local import GfImFreq, GfImTime, InverseFourier, \
    Fourier, iOmega_n, inverse, SemiCircular
from pytriqs.gf.local import GfReFreq, Omega
from pytriqs.plot.mpl_interface import oplot
import numpy as np
import matplotlib.pyplot as plt

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

mu, t = 0.0, 0.5
t2 = t**2
tab = 0.2
beta = 100.

g_iw = GfImFreq(indices=['A', 'B'], beta=beta)
gmix = g_iw.copy()
gmix['A', 'A'] = iOmega_n + mu
gmix['A', 'B'] = -tab
gmix['B', 'A'] = -tab
gmix['B', 'B'] = iOmega_n + mu
for i in xrange(3):

    g_iw << gmix - t2 * g_iw
    g_iw.invert()
    oplot(g_iw['A', 'A'])
#    oplot(g_iw['B','A'])
plt.xlim([0, .8])

g_iw = GfReFreq(indices=['A', 'B'], window=(-3,3), n_points = 2**9)
gmix = g_iw.copy()

plt.figure()
g_iw << 0.
for gam in [0.1, 0.01, 0.001]:
    gmix['A', 'A'] = Omega + mu + 1j * gam
    gmix['A', 'B'] = -0.2
    gmix['B', 'A'] = -0.2
    gmix['B', 'B'] = Omega + mu + 1j * gam


#g_iw << SemiCircular(2*t)
#g_iw << 0.
    for i in xrange(200):
        g_iw << gmix - t2 * g_iw
        #
        g_iw.invert()

    oplot(g_iw['A', 'A'])
oplot(g_iw['B', 'B'])
