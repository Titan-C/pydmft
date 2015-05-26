# -*- coding: utf-8 -*-
"""
Dimer Bethe lattice
===================

Non interacting dimer of a Bethe lattice
"""
#from __future__ import division, print_function, absolute_import
from pytriqs.gf.local import GfImFreq, GfImTime, InverseFourier, \
    Fourier, iOmega_n, inverse
from pytriqs.gf.local import GfReFreq, Omega
from pytriqs.plot.mpl_interface import oplot
import dmft.common as gf
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from pytriqs.archive import HDFArchive


class IPT_dimer_Solver:

    def __init__(self, **params):

        self.U = params['U']
        self.beta = params['beta']
        self.setup = {}

        self.g_iw = GfImFreq(indices=['A', 'B'], beta=self.beta,
                             n_points=params['n_points'])
        self.g0_iw = self.g_iw.copy()
        self.sigma_iw = self.g_iw.copy()

        # Imaginary time
        self.g0_tau = GfImTime(indices=['A', 'B'], beta=self.beta)
        self.sigma_tau = self.g0_tau.copy()

    def solve(self):

        self.g0_tau << InverseFourier(self.g0_iw)
        for name in [('A', 'A'), ('A', 'B'), ('B', 'A'), ('B', 'B')]:
            self.sigma_tau[name] << (self.U**2) * self.g0_tau[name] * self.g0_tau[name] * self.g0_tau[name]
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
beta = 300.


# Real frequency spectral function
w = 1e-3j+np.linspace(-3, 3, 2**9)

for tab in [0, 0.25, 0.5, 0.75, 1.1]:
    g_re = GfReFreq(indices=['A', 'B'], window=(-3, 3), n_points=len(w))
    gmix_re = mix_gf_dimer(g_re.copy(), Omega + 1e-3j, mu, tab)

    init_gf(g_re, -1j*w, mu, tab, t)
    g_re << gmix_re - t2 * g_re
    g_re.invert()

    oplot(g_re['A', 'A'], RI='S', label=r'$t_c={}$'.format(tab), num=1)


# Matsubara frequency Green's function
w_n = gf.matsubara_freq(beta, 512)
for tab in [0, 0.25, 0.5, 0.75, 1.1]:
    g_iw = GfImFreq(indices=['A', 'B'], beta=beta, n_points=len(w_n))
    gmix = mix_gf_dimer(g_iw.copy(), iOmega_n, mu, tab)

    init_gf(g_iw, w_n, mu, tab, t)
    g_iw << gmix - t2 * g_iw
    g_iw.invert()
    oplot(g_iw['A', 'A'], RI='I', label=r'$t_c={}$'.format(tab), num=2)
plt.xlim([0, 6.5])
plt.ylabel(r'$A(\omega)$')
plt.title(u'Spectral functions of dimer Bethe lattice at $\\beta/D=100$ and $U/D=0$.\n Analitical continuation Padé approximant')

# Matsubara interacting self-consistency

def dimer(tab, t, beta):
    w_n = gf.matsubara_freq(beta, 1025)
    S = IPT_dimer_Solver(U=0, beta=beta, n_points=len(w_n))
    gmix = mix_gf_dimer(S.g_iw.copy(), iOmega_n, 0, tab)

    init_gf(S.g_iw, w_n, 0, tab, t)

    converged = False
    loops = 0
    while not converged:
        oldg = S.g_iw.data.copy()
        # Bethe lattice bath
        S.g0_iw << gmix - t2 * S.g_iw
        S.g0_iw.invert()
        S.solve()
        converged = np.allclose(S.g_iw.data, oldg, atol=1e-5)
        loops += 1

    # Store
    S.setup.update({'U': U, 't': t, 'tab': tab, 'beta': beta, 'loops': loops})

    groupname = 'U{U}_t{t}_tab{tab}_B{beta}_c{loops}.h5'.format(**S.setup)
    R = HDFArchive(groupname, 'w')
    R['g0_tau'] = S.g0_tau
    R['G_iw'] = S.g_iw
    R['S_iw'] = S.sigma_iw
    R['setup'] = S.setup
    del R

    return True

def dimhelp(args): return dimer(*args)

p=Pool(4)
conf=[(u,0.5,0.5,150) for u in [0., 0.05, 0.1, 0.15, 0.2, 0.25]]

conf=[(1.5, tab,0.5,150) for tab in [0., 0.25, 0.5, 0.75, 1.]]
ou = p.map(dimhelp, conf)


def plot_re(filen):
    R = HDFArchive(filen, 'r')
#
    greal = GfReFreq(indices=[1], window=(-4.0, 4.0), n_points=256)
    greal.set_from_pade(R['G_iw']['A', 'A'], 100, 0.0)
    oplot(-1*greal, RI='I', label=r'$t_c={}$'.format(tab), num=4)
