#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
================================================
QMC Hirsch - Fye Impurity solver for a Dimer AFM
================================================

To treat the dimer in a Bethe lattice and solve it using the Hirsch - Fye
Quantum Monte Carlo algorithm
"""

from __future__ import division, absolute_import, print_function
from mpi4py import MPI
import dmft.h5archive as h5
import dmft.common as gf
import dmft.hirschfye as hf
import dmft.dimer as dimer
import dmft.plot.hf_dimer as pd
import numpy as np
import sys

comm = MPI.COMM_WORLD


def mat_2_inv(A):
    det = A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1]
    return np.asarray([[A[1, 1], -A[0, 1]],  [-A[1, 0],  A[0, 0]]]) / det


def dmft_loop_pm(simulation):
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
    setup['dtau_mc'] = setup['BETA'] / 2. / setup['N_MATSUBARA']
    current_u = 'U' + str(setup['U'])

    tau, w_n = gf.tau_wn_setup(setup)
    intm = hf.interaction_matrix(setup['BANDS'])
    setup['n_tau_mc'] = len(tau)
    mu, tp, U = setup['MU'], setup['tp'], setup['U']

    giw_d_up, giw_o_up = dimer.gf_met(w_n, 1e-3, tp, 0.5, 0.)
    giw_d_dw, giw_o_dw = dimer.gf_met(w_n, -1e-3, tp, 0.5, 0.)
    gmix = np.array([[1j * w_n, -tp * np.ones_like(w_n)],
                     [-tp * np.ones_like(w_n), 1j * w_n]])
    g0tail = [np.eye(2).reshape(2, 2, 1),
              tp * np.array([[0, 1], [1, 0]]).reshape(2, 2, 1),
              np.array([[tp**2, 0], [0, tp**2]]).reshape(2, 2, 1)]
    gtail = [np.eye(2).reshape(2, 2, 1),
             tp * np.array([[0, 1], [1, 0]]).reshape(2, 2, 1),
             np.array([[tp**2 + U**2 / 4, 0], [0, tp**2 + U**2 / 4]]).reshape(2, 2, 1)]

    giw_up = np.array([[giw_d_up, giw_o_up], [giw_o_dw, giw_d_dw]])
    giw_dw = np.array([[giw_d_dw, giw_o_dw], [giw_o_up, giw_d_up]])

    try:  # try reloading data from disk
        with h5.File(setup['ofile'].format(**setup), 'r') as last_run:
            last_loop = len(last_run[current_u].keys())
            last_it = 'it{:03}'.format(last_loop - 1)
            giw_d, giw_o = pd.get_giw(last_run[current_u], last_it,
                                      tau, w_n)
    except (IOError, KeyError):  # if no data clean start
        last_loop = 0

    V_field = hf.ising_v(setup['dtau_mc'], setup['U'],
                         L=setup['SITES'] * setup['n_tau_mc'],
                         polar=setup['spin_polarization'])

    for loop_count in range(last_loop, last_loop + setup['Niter']):
        # For saving in the h5 file
        dest_group = current_u + '/it{:03}/'.format(loop_count)
        setup['group'] = dest_group

        if comm.rank == 0:
            print('On loop', loop_count, 'beta', setup['BETA'],
                  'U', U, 'tp', tp)

        # Bethe lattice bath
        g0iw_up = mat_2_inv(gmix - 0.25 * giw_up)
        g0iw_dw = mat_2_inv(gmix - 0.25 * giw_dw)

        g0tau_up = gf.gw_invfouriertrans(g0iw_up, tau, w_n, g0tail)
        g0tau_dw = gf.gw_invfouriertrans(g0iw_dw, tau, w_n, g0tail)

        # Impurity solver

        gtu, gtd = hf.imp_solver([g0tau_up, g0tau_dw], V_field, intm, setup)

        giw_up = gf.gt_fouriertrans(-gtu, tau, w_n, gtail)
        giw_dw = gf.gt_fouriertrans(-gtd, tau, w_n, gtail)

        # Save output
        if comm.rank == 0:
            with h5.File(setup['ofile'].format(**setup), 'a') as store:
                store[dest_group + 'gtau_u'] = gtu
                store[dest_group + 'gtau_d'] = gtd
                h5.add_attributes(store[dest_group], setup)
        sys.stdout.flush()


if __name__ == "__main__":
    parser = hf.do_input('DMFT loop for Hirsh-Fye dimer lattice')
    parser.add_argument('-tp', type=float, default=.25,
                        help='Dimerization strength')
    parser.add_argument('-df', '--double_flip_prob', type=float, default=0.,
                        help='Probability for double spin flip on equal sites')

    SETUP = vars(parser.parse_args())
    dmft_loop_pm(SETUP)
