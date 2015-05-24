# -*- coding: utf-8 -*-
"""
Dimer Bethe lattice
===================

Non interacting dimer of a Bethe lattice
"""
from __future__ import division, print_function, absolute_import
from pytriqs.gf.local import GfImFreq, GfImTime, InverseFourier, \
    Fourier, iOmega_n, inverse
from pytriqs.gf.local import GfReFreq, Omega
from pytriqs.plot.mpl_interface import oplot
import dmft.common as gf
import numpy as np
import matplotlib.pyplot as plt


class IPT_dimer_Solver:

    def __init__(self, **params):

        self.U = params['U']
        self.beta = params['beta']

        self.g_iw = GfImFreq(indices=['A', 'B'], beta=self.beta)
        self.g0_iw = self.g_iw.copy()
        self.sigma_iw = self.g_iw.copy()

        # Imaginary time
        self.g0_tau = GfImTime(indices=['A', 'B'], beta=self.beta)
        self.sigma_tau = self.g0_tau.copy()

    def solve(self):

        self.g0_tau << InverseFourier(self.g0_iw)
        self.sigma_tau << (self.U**2) * self.g0_tau * self.g0_tau * self.g0_tau
        self.sigma_iw << Fourier(self.sigma_tau)

        # Dyson equation to get G
        self.g_iw << inverse(inverse(self.g0_iw) - self.sigma_iw)


def mix_gf_dimer(gmix, omega, mu, tab):
    gmix['A', 'A'] = omega + mu
    gmix['A', 'B'] = -tab
    gmix['B', 'A'] = -tab
    gmix['B', 'B'] = omega + mu
    return gmix


def init_gf(g_iw, omega, mu, tab, t):
    G1 = gf.greenF(omega, mu=mu-tab, D=2*t)
    G2 = gf.greenF(omega, mu=mu+tab, D=2*t)

    Gd = .5*(G1 + G2)
    Gc = .5*(G1 - G2)

    g_iw['A', 'A'].data[:, 0, 0] = Gd
    g_iw['A', 'B'].data[:, 0, 0] = Gc
    g_iw['B', 'A'] << g_iw['A', 'B']
    g_iw['B', 'B'] << g_iw['A', 'A']


mu, t = 0.0, 0.5
t2 = t**2
tab = 0.3
beta = 100.

# Matsubara frequency Green's function
g_iw = GfImFreq(indices=['A', 'B'], beta=beta, n_points=2**9)
gmix = mix_gf_dimer(g_iw.copy(), iOmega_n, mu, tab)

w_n = gf.matsubara_freq(beta, 512)
init_gf(g_iw, w_n, mu, tab, t)

for i in xrange(10):

    g_iw << gmix - t2 * g_iw
    g_iw.invert()
    oplot(g_iw['A', 'A'], RI='I')
#    oplot(g_iw['B','A'])
plt.xlim([0, .8])

# Real frequency spectral function
g_iw = GfReFreq(indices=['A', 'B'], window=(-3,3), n_points = 2**9)
gmix = g_iw.copy()


w = 1e-3j+np.linspace(-3, 3, 2**9)

plt.figure()
for gam in [0.1, 0.01, 0.001]:
    gmix = mix_gf_dimer(g_iw.copy(), Omega + 1j * gam, mu, tab)
    init_gf(g_iw, -1j*w, mu, tab, t)

    for i in xrange(5):
        g_iw << gmix - t2 * g_iw
        g_iw.invert()

    oplot(g_iw['A', 'A'], RI='S')
oplot(g_iw['B', 'B'])
