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
import dmft.ipt_imag as ipt


def loop_tp_u(tprange, u_range, beta, filestr, seed='mott gap'):

    save_dir = filestr.format(beta)
    if np.allclose(tprange, tprange[0]) and 'tp' not in save_dir:
        save_dir = os.path.join(save_dir, 'tp' + str(tprange[0]))
    elif np.allclose(u_range, u_range[0]):
        save_dir = os.path.join(save_dir, 'U' + str(u_range[0]))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    setup = {'beta': beta, 'tprange': tprange.tolist(),
             'u_range': u_range.tolist()}
    with open(save_dir + '/setup', 'w') as conf:
        json.dump(setup, conf, indent=2)

###############################################################################

    tau, w_n = gf.tau_wn_setup(
        dict(BETA=beta, N_MATSUBARA=max(2**ceil(log(6 * beta) / log(2)), 256)))
    giw_d, giw_o = rt.gf_met(w_n, 0., tprange[0], 0.5, 0.)
    if seed == 'mott gap':
        giw_d, giw_o = 1 / (1j * w_n - 4j / w_n), np.zeros_like(w_n) + 0j

    giw_s = []
    for tp, u_int in zip(tprange, u_range):
        giw_d, giw_o, loops = rt.ipt_dmft_loop(
            beta, u_int, tp, giw_d, giw_o, tau, w_n, 1 / 5 / beta)
        giw_s.append((giw_d.imag, giw_o.real))
    np.save(save_dir + '/giw', np.array(giw_s))


if __name__ == "__main__":

    tpr = np.arange(0, 1.1, 0.02)
    ur = np.arange(0, 4.5, 0.1)
    BETARANGE = [1000., 100., 30.]
    jobs = [(job.T[0], job.T[1], BETA, 'disk/phase_Dimer_ipt_met_B{:.5}', 'metal')
            for BETA in BETARANGE
            for job in np.array(list(itertools.product(tpr, ur))).reshape(len(tpr), len(ur), 2)]
    jobs += [(job.T[0], job.T[1][::-1], BETA, 'disk/phase_Dimer_ipt_ins_B{:.5}')
             for BETA in BETARANGE
             for job in np.array(list(itertools.product(tpr, ur))).reshape(len(tpr), len(ur), 2)]

    Parallel(n_jobs=-1, verbose=5)(delayed(loop_tp_u)(*job) for job in jobs)
    tpr = [0, .15, .3, .5]
    ur = np.arange(0, 4.5, 0.1)
    BETARANGE = 1 / np.arange(1 / 500., .14, 1 / 400)
    jobs = [(job.T[0], job.T[1], BETA, 'disk/phase_Dimer_ipt_met_tp{}/B{{:.5}}'.format(job[0][0]), 'metal')
            for BETA in BETARANGE
            for job in np.array(list(itertools.product(tpr, ur))).reshape(len(tpr), len(ur), 2)]
    jobs += [(job.T[0], job.T[1][::-1], BETA, 'disk/phase_Dimer_ipt_ins_tp{}/B{{:.5}}'.format(job[0][0]))
             for BETA in BETARANGE
             for job in np.array(list(itertools.product(tpr, ur))).reshape(len(tpr), len(ur), 2)]

    Parallel(n_jobs=-1, verbose=5)(delayed(loop_tp_u)(*job) for job in jobs)
