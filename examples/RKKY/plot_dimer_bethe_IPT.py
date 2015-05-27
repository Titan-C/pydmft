# -*- coding: utf-8 -*-
"""
Dimer Bethe lattice
===================

Non interacting dimer of a Bethe lattice
"""
#from __future__ import division, print_function, absolute_import
import sys
sys.path.append('/home/oscar/libs/lib/python2.7/site-packages')
from pytriqs.gf.local import GfImFreq, GfImTime, InverseFourier, \
    Fourier, iOmega_n, inverse
from pytriqs.gf.local import GfReFreq, Omega
from pytriqs.plot.mpl_interface import oplot
import dmft.common as gf
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from pytriqs.archive import HDFArchive
from plot_dimer_bethe_triqs import mix_gf_dimer, init_gf

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


mu, t = 0.0, 0.5
t2 = t**2
tab = 0.3
beta = 300.

# Matsubara interacting self-consistency
def loop_u(urange, tab, t, beta):
    w_n = gf.matsubara_freq(beta, 1025)
    S = IPT_dimer_Solver(U=0, beta=beta, n_points=len(w_n))
    gmix = mix_gf_dimer(S.g_iw.copy(), iOmega_n, 0, tab)
    init_gf(S.g_iw, w_n, 0, tab, t)
    S.setup.update({'t': t, 'tab': tab, 'beta': beta})

    for u_int in urange:
        S.U = u_int
        dimer(S, gmix)

    return True


def dimer(S, gmix):

    converged = False
    loops = 0
    while not converged:
        oldg = S.g_iw.data.copy()
        # Bethe lattice bath
        S.g0_iw << gmix - t2 * S.g_iw
        S.g0_iw.invert()
        S.solve()
        converged = np.allclose(S.g_iw.data, oldg, atol=1e-3)
        loops += 1

    S.setup.update({'U':S.U, 'loops': loops})

    store_sim(S)


def store_sim(S):
    u_step = '/U{U}/'.format(**S.setup)
    file_name = 'uloop_t{t}_tab{tab}_B{beta}.h5'.format(**S.setup)
    R = HDFArchive(file_name, 'a')
    R[u_step+'setup'] = S.setup
    R[u_step+'G_iw'] = S.g_iw
    R[u_step+'g0_tau'] = S.g0_tau
    R[u_step+'S_iw'] = S.sigma_iw
    del R

ur=np.arange(0, 4, 0.025)

def dimhelp(tab): return loop_u(ur, tab, 0.5, 150)

p = Pool(12)
tabra = np.arange(0, 1.3, 0.1)
ou = p.map(dimhelp, tabra.tolist())


def plot_re(filen, U):
    R = HDFArchive(filen, 'r')
#
    greal = GfReFreq(indices=[1], window=(-4.0, 4.0), n_points=256)
    greal.set_from_pade(R['U{}'.format(U)]['G_iw']['A', 'A'], 100, 0.0)
    oplot(-1*greal, RI='I', label=r'$t_c={}$'.format(tab), num=4)

    del R
