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
import numpy as np
import sys
import os
import json
from mpi4py import MPI
import dmft.common as gf
import dmft.hirschfye as hf
import dmft.RKKY_dimer as dimer
import dmft.plot.hf_dimer as pd
comm = MPI.COMM_WORLD


def mat_2_inv(A):
    det = A[0, 0]*A[1, 1]-A[1, 0]*A[0, 1]
    return np.asarray([[A[1, 1], -A[0, 1]],  [-A[1, 0],  A[0, 0]]])/det


def gf_tail(gtau, U, mu, tp):

    g_t0 = gtau[:, :, 0]

    gtail = [np.eye(2).reshape(2, 2, 1),
             (-mu  + tp*np.array([[0, 1], [1, 0]])).reshape(2, 2, 1),
             (0.25 + U**2/4 + tp**2)*np.eye(2).reshape(2, 2, 1)]
    return gtail

def dmft_loop_pm(simulation, U, g_iw_start=None):
    """Implementation of the solver"""
    setup = {'t':         0.5,
             'BANDS':     1,
             'SITES':     2,
             }

    if simulation['new_seed']:
        if comm.rank == 0:
            hf.set_new_seed(simulation, ['gtau_d', 'gtau_o'])
        simulation['U'] = simulation['new_seed'][1]
        return

    setup.update(simulation)
    setup['dtau_mc'] = setup['BETA']/2./setup['N_MATSUBARA']
    current_u = 'U'+str(U)
    setup['simt'] = 'PM' # simulation type ParaMagnetic

    tau, w_n = gf.tau_wn_setup(setup)
    intm = hf.interaction_matrix(setup['BANDS'])
    setup['n_tau_mc'] = len(tau)
    mu, tp = setup['MU'], setup['tp']
    giw_d, giw_o = dimer.gf_met(w_n, mu, tp, 0.5, 0.)

    gmix = np.array([[1j*w_n, -tp*np.ones_like(w_n)],
                       [-tp*np.ones_like(w_n), 1j*w_n]])

    giw_up = np.array([[giw_d, giw_o], [giw_o, giw_d]])
    giw_dw = np.array([[giw_d, giw_o], [giw_o, giw_d]])

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

        giw_up = gf.gt_fouriertrans(-gtu, tau, w_n, gf_tail(-gtu, U, mu, tp))
        giw_dw = gf.gt_fouriertrans(-gtd, tau, w_n, gf_tail(-gtd, U, mu, tp))

        last_loop += 1
    except (IOError, KeyError, ValueError):  # if no data clean start
        last_loop = 0

    V_field = hf.ising_v(setup['dtau_mc'], U,
                         L=setup['SITES']*setup['n_tau_mc'],
                         polar=setup['spin_polarization'])


    for iter_count in range(last_loop, last_loop + setup['Niter']):
        work_dir = os.path.join(save_dir, 'it{:03}'.format(iter_count))
        setup['work_dir'] = work_dir

        if comm.rank == 0:
            print('On loop', iter_count, 'beta', setup['BETA'],
                  'U', U, 'tp', tp)

        # Bethe lattice bath
        g0iw_up = mat_2_inv(gmix - 0.25*giw_up)
        g0iw_dw = mat_2_inv(gmix - 0.25*giw_dw)

        g0tau0 = -0.5*np.eye(2).reshape(2, 2, 1)
        g0tau_up = gf.gw_invfouriertrans(g0iw_up, tau, w_n, gf_tail(g0tau0, 0., mu, tp))
        g0tau_dw = gf.gw_invfouriertrans(g0iw_dw, tau, w_n, gf_tail(g0tau0, 0., mu, tp))

        # Impurity solver

        gtu, gtd = hf.imp_solver([g0tau_up, g0tau_dw], V_field, intm, setup)

        giw_up = gf.gt_fouriertrans(-gtu, tau, w_n, gf_tail(-gtu, U, mu, tp))
        giw_dw = gf.gt_fouriertrans(-gtd, tau, w_n, gf_tail(-gtd, U, mu, tp))


        # Save output
        if comm.rank == 0:
            np.save(work_dir+'/gtau_up', gtu.reshape(4, -1))
            np.save(work_dir+'/gtau_dw', gtd.reshape(4, -1))
            with open(save_dir + '/setup', 'w') as conf:
                setup['last_loop'] = iter_count
                json.dump(setup, conf, indent=2)
        sys.stdout.flush()


if __name__ == "__main__":
    parser = hf.do_input('DMFT loop for Hirsh-Fye dimer lattice')
    parser.add_argument('-tp', type=float, default=.25,
                        help='Dimerization strength')
    parser.add_argument('-df', '--double_flip_prob', type=float, default=0.,
                        help='Probability for double spin flip on equal sites')

    SETUP = vars(parser.parse_args())
    for U in SETUP['U']:
        dmft_loop_pm(SETUP, U)
