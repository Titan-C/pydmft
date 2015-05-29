# -*- coding: utf-8 -*-
"""
Dimer Bethe lattice
===================

Non interacting dimer of a Bethe lattice
"""
#from __future__ import division, print_function, absolute_import
from pytriqs.gf.local import GfImFreq, GfImTime, InverseFourier, \
    Fourier, inverse
import numpy as np
from pytriqs.archive import HDFArchive

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
        for name in [('A', 'A'), ('A', 'B'), ('B', 'A'), ('B', 'B')]:
            self.sigma_tau[name] << (self.U**2) * self.g0_tau[name] * self.g0_tau[name] * self.g0_tau[name]
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
        if loops > 300:
            converged = True

    S.setup.update({'U': S.U, 'loops': loops})

    store_sim(S, filename, step)


def store_sim(S, filename, step):
    R = HDFArchive(filename, 'a')
    R[step+'setup'] = S.setup
    R[step+'G_iw'] = S.g_iw
    R[step+'g0_tau'] = S.g0_tau
    R[step+'S_iw'] = S.sigma_iw
    del R
