# -*- coding: utf-8 -*-
r"""
======================
IPT Solver for a dimer
======================

Simple IPT solver for a dimer

For symmetry reason only the imaginary diagonal and the real
off-diagonal parts of the block green function are stored in
disk. Keep that in mind when dealing with files
"""
# Created Thu Nov 12 17:55:48 2015
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function
from math import ceil, log
import itertools
import json
import os
import numpy as np
from joblib import Parallel, delayed
import dmft.RKKY_dimer as rt
import dmft.common as gf


def loop_tp_u(tprange, Drange, beta, filestr, seed='mott gap'):

    save_dir = filestr.format(beta)
    if np.allclose(tprange, tprange[0]) and 'tp' not in save_dir:
        save_dir = os.path.join(save_dir, 'tp' + str(tprange[0]))
    elif np.allclose(Drange, Drange[0]):
        save_dir = os.path.join(save_dir, 'D' + str(Drange[0]))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    setup = {'beta': beta, 'tprange': tprange.tolist(),
             'Drange': Drange.tolist()}
    with open(save_dir + '/setup', 'w') as conf:
        json.dump(setup, conf, indent=2)

###############################################################################

    tau, w_n = gf.tau_wn_setup(
        dict(BETA=beta, N_MATSUBARA=max(2**ceil(log(6 * beta) / log(2)), 256)))
    giw_d, giw_o = rt.gf_met(w_n, 0., tprange[0], 0.5, 0.)
    if seed == 'mott gap':
        giw_d, giw_o = 1 / (1j * w_n - 4j / w_n), np.zeros_like(w_n) + 0j

    giw_s = []
    for tp, D in zip(tprange, Drange):
        giw_d, giw_o, loops = rt.ipt_dmft_loop(
            beta, 1, tp, giw_d, giw_o, tau, w_n, 1 / 5 / beta, t=D / 2)
        giw_s.append((giw_d.imag, giw_o.real))
    np.save(save_dir + '/giw', np.array(giw_s))


if __name__ == "__main__":

    Drange = np.linspace(0.05, .85, 61)
    tpr = np.arange(0, 1.1, 0.02)
    BETARANGE = [100., 30.]
    jobs = [(job.T[0], job.T[1], BETA, 'disk/phase_Dimer_ipt_D_met_B{:.5}', 'metal')
            for BETA in BETARANGE
            for job in np.array(list(itertools.product(tpr, Drange))).reshape(len(tpr), len(Drange), 2)]
    jobs += [(job.T[0], job.T[1][::-1], BETA, 'disk/phase_Dimer_ipt_D_ins_B{:.5}')
             for BETA in BETARANGE
             for job in np.array(list(itertools.product(tpr, Drange))).reshape(len(tpr), len(Drange), 2)]

    Parallel(n_jobs=-1, verbose=5)(delayed(loop_tp_u)(*job) for job in jobs)
