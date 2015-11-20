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
import argparse
import dmft.RKKY_dimer as rt
import dmft.common as gf
import dmft.h5archive as h5
import dmft.ipt_imag as ipt
import numpy as np


def loop_tp_u(urange, tprange, BETA, filestr, metal=True):
    tau, w_n = gf.tau_wn_setup(dict(BETA=BETA, N_MATSUBARA=5*BETA))
    for tp in tprange:
        giw_d, giw_o = rt.gf_met(w_n, 0., tp, 0.5, 0.)
        if not metal:
            giw_d = 1/(1j*w_n - 4j/w_n)
            giw_o[:] = 0.

        for u_int in urange:
            giw_d, giw_o, loops = rt.ipt_dmft_loop(BETA, u_int, tp, giw_d, giw_o)

            with h5.File(filestr.format(BETA), 'a') as store:
                u_group = '/tp{}/U{}/'.format(tp, u_int)
                store[u_group+'giw_d'] = giw_d.imag
                store[u_group+'giw_o'] = giw_o.real
                store[u_group+'loops'] = loops

if __name__ == "__main__":

    tpr = np.hstack((np.arange(0, 0.5, 0.02), np.arange(0.5, 1.1, 0.05)))
    ur = np.arange(0, 4.5, 0.1)
    BETARANGE = np.round(np.hstack(([768, 512], np.logspace(8, -4.5, 41, base=2))), 2)

    jobs = [(ur, tpr, BETA, 'disk/Dimer_ipt_B{}.h5') for BETA in BETARANGE]
    jobs+= [(ur[::-1], tpr, BETA, 'disk/Dimer_ins_ipt_B{}.h5', False) for BETA in BETARANGE]

    Parallel(n_jobs=-1, verbose=5)(delayed(loop_tp_u)(*job) for job in jobs)
