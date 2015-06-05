# -*- coding: utf-8 -*-
"""
Dimer Bethe lattice
===================

Non interacting dimer of a Bethe lattice
Based on the work G. Moeller et all PRB 59, 10, 6846 (1999)
"""
#from __future__ import division, print_function, absolute_import
from pytriqs.gf.local import GfImFreq, GfImTime, InverseFourier, \
    Fourier, inverse, TailGf
import numpy as np
from pytriqs.archive import HDFArchive
import dmft.common as gf


def mix_gf_dimer(gmix, omega, mu, tab):
    gmix['A', 'A'] = omega + mu
    gmix['A', 'B'] = -tab
    gmix['B', 'A'] = -tab
    gmix['B', 'B'] = omega + mu
    return gmix


def init_gf_met(g_iw, omega, mu, tab, t):
    G1 = gf.greenF(omega, mu=mu-tab, D=2*t)
    G2 = gf.greenF(omega, mu=mu+tab, D=2*t)

    Gd = .5*(G1 + G2)
    Gc = .5*(G1 - G2)

    g_iw['A', 'A'].data[:, 0, 0] = Gd
    g_iw['A', 'B'].data[:, 0, 0] = Gc
    g_iw['B', 'A'] << g_iw['A', 'B']
    g_iw['B', 'B'] << g_iw['A', 'A']

    fixed_co = TailGf(2, 2, 4, -1)
    fixed_co[1] = np.array([[1, 0], [0, 1]])
    fixed_co[2] = tab*np.array([[0, 1], [1, 0]])
    g_iw.fit_tail(fixed_co, 6, int(0.6*len(omega)), int(0.8*len(omega)))


def init_gf_ins(g_iw, omega, mu, tab, U):
    G1 = 1./(1j*omega - tab + U / 2.)
    G2 = 1./(1j*omega + tab - U / 2.)

    Gd = .5*(G1 + G2)
    Gc = .5*(G1 - G2)

    g_iw['A', 'A'].data[:, 0, 0] = Gd
    g_iw['A', 'B'].data[:, 0, 0] = Gc
    g_iw['B', 'A'] << g_iw['A', 'B']
    g_iw['B', 'B'] << g_iw['A', 'A']



class Dimer_Solver:

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
        for name in [('A', 'A'), ('B', 'B')]:
            self.sigma_tau[name] << (self.U**2) * self.g0_tau[name] * self.g0_tau[name] * self.g0_tau[name]
        for name in [('A', 'B'), ('B', 'A')]:
            self.sigma_tau[name] << -(self.U**2) * self.g0_tau[name] * self.g0_tau[name] * self.g0_tau[name]

        self.sigma_iw << Fourier(self.sigma_tau)

        # Dyson equation to get G
        self.g_iw << inverse(inverse(self.g0_iw) - self.sigma_iw)


def dimer(S, gmix, filename, step):

    converged = False
    loops = 0
    t2 = S.setup['t']**2
    while not converged:
        # Enforce DMFT Paramagnetic, IPT conditions
        # Pure imaginary GF in diagonals
        S.g_iw.data[:, [0, 1], [0, 1]] = 1j*S.g_iw.data[:, [0, 1], [0, 1]].imag
        # Pure real GF in off-diagonals
        S.g_iw.data[:, [0, 1], [1, 0]] = S.g_iw.data[:, [0, 1], [1, 0]].real

        oldg = S.g_iw.data.copy()
        # Bethe lattice bath
        S.g0_iw << gmix - t2 * S.g_iw
        S.g0_iw.invert()
        S.solve()

        converged = np.allclose(S.g_iw.data, oldg, atol=1e-3)
        loops += 1
        if loops > 600:
            converged = True

#        #Finer loop of complicated region
#        if S.setup['tab'] > 0.5 and S.U > 1.:
        S.g_iw.data[:] = 0.9*S.g_iw.data + 0.1*oldg

    S.setup.update({'U': S.U, 'loops': loops})

    store_sim(S, filename, step)


def store_sim(S, file_str, step_str):
    file_name = file_str.format(**S.setup)
    step = step_str.format(**S.setup)
    R = HDFArchive(file_name, 'a')
    R[step+'setup'] = S.setup
    R[step+'G_iw'] = S.g_iw
    R[step+'g0_tau'] = S.g0_tau
    R[step+'S_iw'] = S.sigma_iw
    del R
