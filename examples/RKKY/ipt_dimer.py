# -*- coding: utf-8 -*-
r"""
======================
IPT Solver for a dimer
======================

Simple IPT solver for a dimer
"""
# Created Thu Nov 12 17:55:48 2015
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function
from joblib import Parallel, delayed
from dmft.ipt_imag import dimer_solver
import argparse
import dmft.RKKY_dimer as rt
import dmft.common as gf
import dmft.h5archive as h5
import numpy as np


def dimer_dmft_loop(BETA, u_int, tp, giw_d, giw_o, conv=1e-3):
    tau, w_n = gf.tau_wn_setup(dict(BETA=BETA, N_MATSUBARA=5*BETA))

    converged = False
    loops = 0
    iw_n = 1j*w_n

    while not converged:
        # Half-filling, particle-hole cleaning
        giw_d.real = 0.
        giw_o.imag = 0.

        giw_d_old = giw_d.copy()
        giw_o_old = giw_o.copy()

        g0iw_d, g0iw_o = rt.self_consistency(iw_n, giw_d, giw_o, 0., tp, 0.25)

        giw_d, giw_o = dimer_solver(u_int, tp, g0iw_d, g0iw_o, w_n, tau)

        converged = np.allclose(giw_d_old, giw_d, conv)
        converged *= np.allclose(giw_o_old, giw_o, conv)

        loops += 1

    return giw_d, giw_o, loops


def loop_u(urange, tp, BETA, filestr):
    tau, w_n = gf.tau_wn_setup(dict(BETA=BETA, N_MATSUBARA=5*BETA))
    giw_d, giw_o = rt.gf_met(w_n, 0., tp, 0.5, 0.)
    for u_int in urange:
        giw_d, giw_o, loops = dimer_dmft_loop(BETA, u_int, tp, giw_d, giw_o)

        with h5.File(filestr.format(tp=tp, BETA=BETA), 'a') as store:
            u_group = '/U{}/'.format(u_int)
            store[u_group+'giw_d'] = giw_d
            store[u_group+'giw_o'] = giw_o
            store[u_group+'loops'] = loops

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DMFT loop for a dimer Bethe'
                                     'lattice solved by IPT')
    parser.add_argument('beta', metavar='B', type=float,
                        default=150., help='The inverse temperature')

    tabra = np.hstack((np.arange(0, 0.5, 0.02), np.arange(0.5, 1.1, 0.05)))
    args = parser.parse_args()
    BETA = args.beta

    ur = np.arange(0, 4.5, 0.1)

    #print(BETA)
    Parallel(n_jobs=-1, verbose=5)(delayed(loop_u)(ur, tp, 0.5, 0., BETA,
             'disk/Dimer_ipt_tp{tp}_B{BETA}.h5') for tp in tabra)
    Parallel(n_jobs=-1, verbose=5)(delayed(loop_u)(ur[::-1], tp, 0.5, 0., BETA,
             'disk/Dimer_ipt_tp{tp}_B{BETA}.h5') for tp in tabra)
