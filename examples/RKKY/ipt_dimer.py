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
    if np.allclose(tprange, tprange[0]):
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

    tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=max(4 * beta, 256)))
    giw_d, giw_o = rt.gf_met(w_n, 0., tprange[0], 0.5, 0.)
    if seed == 'mott gap':
        giw_d, giw_o = 1 / (1j * w_n - 4j / w_n), np.zeros_like(w_n) + 0j

    giw_s = []
    sigma_iw = []
    ekin, epot = [], []
    iterations = []
    for tp, u_int in zip(tprange, u_range):
        giw_d, giw_o, loops = rt.ipt_dmft_loop(
            beta, u_int, tp, giw_d, giw_o, tau, w_n, 1e-3)
        giw_s.append((giw_d, giw_o))
        iterations.append(loops)
        g0iw_d, g0iw_o = rt.self_consistency(
            1j * w_n, 1j * giw_d.imag, giw_o.real, 0., tp, 0.25)
        siw_d, siw_o = ipt.dimer_sigma(u_int, tp, g0iw_d, g0iw_o, tau, w_n)
        sigma_iw.append((siw_d.copy(), siw_o.copy()))

        ekin.append(rt.ekin(giw_d, giw_o, w_n, tp, beta))

        epot.append(rt.epot(giw_d, giw_o, siw_d, siw_o, w_n, tp, u_int, beta))

    np.save(save_dir + '/giw', np.array(giw_s))
    np.save(save_dir + '/sigmaiw', np.array(sigma_iw))
    np.save(save_dir + '/ekin', np.array(ekin))
    np.save(save_dir + '/epot', np.array(epot))
    np.save(save_dir + '/complexity', np.array(iterations))


if __name__ == "__main__":

    tpr = np.hstack((np.arange(0, 0.5, 0.02), np.arange(0.5, 1.1, 0.05)))
    ur = np.arange(0, 4.5, 0.1)
    BETARANGE = np.round(
        np.hstack(([756., 512.], np.logspace(8, -4.5, 45, base=2), [.03, .02])), 2)
    jobs = [(job.T[0], job.T[1], BETA, 'disk/Dimer_ipt_B{}', 'metal')
            for BETA in BETARANGE
            for job in np.array(list(itertools.product(tpr, ur))).reshape(len(tpr), len(ur), 2)]
    jobs += [(job.T[0], job.T[1][::-1], BETA, 'disk/Dimer_ins_ipt_B{}')
             for BETA in BETARANGE
             for job in np.array(list(itertools.product(tpr, ur))).reshape(len(tpr), len(ur), 2)]

    Parallel(n_jobs=-1, verbose=5)(delayed(loop_tp_u)(*job) for job in jobs)
