#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
============================================
QMC Hirsch - Fye Impurity solver for a Dimer
============================================

To treat the dimer in a Bethe lattice and solve it using the Hirsch - Fye
Quantum Monte Carlo algorithm
"""

from __future__ import division, absolute_import, print_function
import ctypes
import sys
import os
import json

import numpy as np
from mpi4py import MPI
import dmft.common as gf
import dmft.hirschfye as hf
import dmft.RKKY_dimer as dimer
import dmft.plot.hf_dimer as pd
comm = MPI.COMM_WORLD


def num_threads(cores):
    try:
        cdll = ctypes.CDLL('libmkl_rt.so')
        cdll.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))
        return
    except OSError:
        pass

    try:
        cdll = ctypes.CDLL('libopenblas.so')
        cdll.openblas_set_num_threads(int(cores))
        return
    except OSError:
        pass

    raise OSError('Unknow blas library')


def dmft_loop_pm(simulation, U, g_iw_start=None):
    """Implementation of the solver"""
    setup = {'t':         0.5,
             'BANDS':     1,
             'SITES':     2,
             }

    setup.update(simulation)
    setup['dtau_mc'] = setup['BETA'] / 2. / setup['N_MATSUBARA']
    current_u = 'U' + str(U)
    setup['U'] = U
    setup['simt'] = 'PM'  # simulation type ParaMagnetic
    if setup['AFM']:
        setup['simt'] = 'AFM'  # simulation type AntiFerroMagnetic

    tau, w_n = gf.tau_wn_setup(setup)
    intm = hf.interaction_matrix(setup['BANDS'])
    setup['n_tau_mc'] = len(tau)
    mu, tp = setup['MU'], setup['tp']
    giw_d, giw_o = dimer.gf_met(w_n, mu, tp, 0.5, 0.)

    gmix = np.array([[1j * w_n + mu, -tp * np.ones_like(w_n)],
                     [-tp * np.ones_like(w_n), 1j * w_n + mu]])

    giw = np.array([[giw_d, giw_o], [giw_o, giw_d]])
    g0tau0 = -0.5 * np.eye(2).reshape(2, 2, 1)
    gtu = gf.gw_invfouriertrans(giw, tau, w_n, pd.gf_tail(g0tau0, 0., mu, tp))
    gtd = np.copy(gtu)

    if g_iw_start is not None:
        giw_up = g_iw_start[0]
        giw_dw = g_iw_start[1]

    save_dir = os.path.join(setup['ofile'].format(**setup), current_u)
    try:  # try reloading data from disk
        with open(save_dir + '/setup', 'r') as conf:
            last_loop = json.load(conf)['last_loop']
        gtu = np.load(os.path.join(save_dir,
                                   'it{:03}'.format(last_loop),
                                   'gtau_up.npy')).reshape(2, 2, -1)
        gtd = np.load(os.path.join(save_dir,
                                   'it{:03}'.format(last_loop),
                                   'gtau_dw.npy')).reshape(2, 2, -1)

        last_loop += 1
    except (IOError, KeyError, ValueError):  # if no data clean start
        last_loop = 0

    V_field = hf.ising_v(setup['dtau_mc'], U,
                         L=setup['SITES'] * setup['n_tau_mc'],
                         polar=setup['spin_polarization'])

    for iter_count in range(last_loop, last_loop + setup['Niter']):
        work_dir = os.path.join(save_dir, 'it{:03}'.format(iter_count))
        setup['work_dir'] = work_dir

        if comm.rank == 0:
            print('On loop', iter_count, 'beta', setup['BETA'],
                  'U', U, 'tp', tp)
        # paramagnetic cleaning
        gtu = 0.5 * (gtu + gtd)
        gtd = gtu

        giw_up = gf.gt_fouriertrans(gtu, tau, w_n, pd.gf_tail(gtu, U, mu, tp))
        giw_dw = gf.gt_fouriertrans(gtd, tau, w_n, pd.gf_tail(gtd, U, mu, tp))

        # Bethe lattice bath
        g0iw_up = dimer.mat_2_inv(gmix - 0.25 * giw_up)
        g0iw_dw = dimer.mat_2_inv(gmix - 0.25 * giw_dw)

        g0tau_up = gf.gw_invfouriertrans(
            g0iw_up, tau, w_n, pd.gf_tail(g0tau0, 0., mu, tp))
        g0tau_dw = gf.gw_invfouriertrans(
            g0iw_dw, tau, w_n, pd.gf_tail(g0tau0, 0., mu, tp))

        # Impurity solver

        gtu, gtd = hf.imp_solver([g0tau_dw, g0tau_up], V_field, intm, setup)

        # Save output
        if comm.rank == 0:
            np.save(work_dir + '/gtau_up', gtu.reshape(4, -1))
            np.save(work_dir + '/gtau_dw', gtd.reshape(4, -1))
            with open(save_dir + '/setup', 'w') as conf:
                setup['last_loop'] = iter_count
                json.dump(setup, conf, indent=2)
        sys.stdout.flush()


if __name__ == "__main__":
    parser = hf.do_input('DMFT loop for Hirsh-Fye dimer lattice')
    parser.add_argument('-tp', type=float, default=0.3,
                        help='Dimerization strength')
    parser.add_argument('-df', '--double_flip_prob', type=float, default=0.,
                        help='Probability for double spin flip on equal sites')
    parser.add_argument('-afm', '--AFM', action='store_true',
                        help='Use the self-consistency for Antiferromagnetism')
    parser.set_defaults(ofile='DIMER_{simt}_B{BETA}_tp{tp}_MU{MU}')

    SETUP = vars(parser.parse_args())
    num_threads(1)
    for U in SETUP['urange']:
        dmft_loop_pm(SETUP, U)
