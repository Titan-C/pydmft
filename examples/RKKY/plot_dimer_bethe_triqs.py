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

        self.g_iw = GfImFreq(indices=['A', 'B'], beta=self.beta,
                             n_points=params['n_points'])
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


# Real frequency spectral function
plt.figure(2)
w = 1e-3j+np.linspace(-3, 3, 2**9)

for tab in [0, 0.25, 0.5, 0.75, 1.1]:
    g_re = GfReFreq(indices=['A', 'B'], window=(-3, 3), n_points=len(w))
    gmix_re = mix_gf_dimer(g_re.copy(), Omega + 1e-3j, mu, tab)

    init_gf(g_re, -1j*w, mu, tab, t)
    g_re << gmix_re - t2 * g_re
    g_re.invert()

    oplot(2, g_re['A', 'A'], RI='S', label=r'$t_c={}$'.format(tab))


# Matsubara frequency Green's function
plt.figure(1)
w_n = gf.matsubara_freq(beta, 512)
for tab in [0, 0.25, 0.5, 0.75, 1.1]:
    g_iw = GfImFreq(indices=['A', 'B'], beta=beta, n_points=len(w_n))
    gmix = mix_gf_dimer(g_iw.copy(), iOmega_n, mu, tab)

    init_gf(g_iw, w_n, mu, tab, t)
    g_iw << gmix - t2 * g_iw
    g_iw.invert()
    oplot(1, g_iw['A', 'A'], RI='I', label=r'$t_c={}$'.format(tab))
plt.xlim([0, 6.5])

# Matsubara interacting self-consistency
tab = 1.1
S = IPT_dimer_Solver(U=0, beta=beta, n_points=len(w_n))
gmix = mix_gf_dimer(S.g_iw.copy(), iOmega_n, mu, tab)

init_gf(S.g_iw, w_n, mu, tab, t)

for i in xrange(10):
    # Bethe lattice bath
    S.g0_iw << gmix - t2 * S.g_iw
    S.g0_iw.invert()

    S.solve()
#    oplot(1, S.g_iw['A', 'A'], RI='I', label=r'$iter {}$'.format(i))

greal = GfReFreq(indices=[1], window=(-4.0, 4.0), n_points=400)
greal.set_from_pade(S.g_iw['A', 'A'], 200, 0.0)
oplot(2, greal, RI='S', label=r'$t_c={}$'.format(tab))
