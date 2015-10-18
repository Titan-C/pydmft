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
from mpi4py import MPI
import dmft.h5archive as h5
import dmft.common as gf
import dmft.hirschfye as hf
import dmft.plot.hf_dimer as pd
import numpy as np
import sys

comm = MPI.COMM_WORLD


def init_gf_met(omega, mu, tab, tn, t):
    """Gives a metalic seed of a non-interacting system

    """
    G1 = gf.greenF(omega, mu=mu-tab, D=2*(t+tn))
    G2 = gf.greenF(omega, mu=mu+tab, D=2*abs(t-tn))
    return .5*(G1 + G2), .5*(G1 - G2)


def dmft_loop_pm(simulation):
    """Implementation of the solver"""
    setup = {'t':         0.5,
             'BANDS':     1,
             'SITES':     2,
             'save_logs': False,
             'global_flip': True,
             }

    if simulation['new_seed']:
        if comm.rank == 0:
            hf.set_new_seed(simulation, ['gtau_d', 'gtau_o'])
        simulation['U'] = simulation['new_seed'][1]
        return

    setup.update(simulation)
    setup['dtau_mc'] = setup['BETA']/2./setup['N_MATSUBARA']
    current_u = 'U'+str(setup['U'])

    tau, w_n = gf.tau_wn_setup(setup)
    intm = hf.interaction_matrix(setup['BANDS'])
    setup['n_tau_mc'] = len(tau)
    giw_D, giw_N = init_gf_met(w_n, setup['MU'], setup['tp'], 0., 0.5)

    try:  # try reloading data from disk
        with h5.File(setup['ofile'].format(**setup), 'r') as last_run:
            last_loop = len(last_run[current_u].keys())
            last_it = 'it{:03}'.format(last_loop-1)
            giw_D, giw_N = pd.get_giw(last_run[current_u], last_it,
                                      tau, w_n, setup['tp'])
    except (IOError, KeyError):  # if no data clean start
        last_loop = 0

    V_field = hf.ising_v(setup['dtau_mc'], setup['U'],
                           L=setup['SITES']*setup['n_tau_mc'])

    for loop_count in range(last_loop, last_loop + setup['Niter']):
        # For saving in the h5 file
        dest_group = current_u+'/it{:03}/'.format(loop_count)
        setup['group'] = dest_group

        if comm.rank == 0:
            print('On loop', loop_count, 'beta', setup['BETA'],
                  'U', setup['U'], 'tp', setup['tp'])


        # Bethe lattice bath
        g0iw_D = 1.j*w_n - 0.25 * giw_D
        g0iw_N = -setup['tp'] - 0.25 * giw_N

        det = g0iw_D**2 - g0iw_N**2
        g0iw_D /= det
        g0iw_N /= -det

        g0tau_D = gf.gw_invfouriertrans(g0iw_D, tau, w_n)
        g0tau_N = gf.gw_invfouriertrans(g0iw_N, tau, w_n,
                                        [0., setup['tp'], 0.])

        g0t = np.array([[g0tau_D, g0tau_N], [g0tau_N, g0tau_D]])

        gtu, gtd = hf.imp_solver([g0t]*2, V_field, intm, setup)
        gtau_d = -0.25 * (gtu[0, 0] + gtu[1, 1] + gtd[0, 0] + gtd[1, 1])
        gtau_o = -0.25 * (gtu[1, 0] + gtu[0, 1] + gtd[1, 0] + gtd[0, 1])

        giw_D = gf.gt_fouriertrans(gtau_d, tau, w_n)
        giw_N = gf.gt_fouriertrans(gtau_o, tau, w_n,
                                   [0., setup['tp'], 0.])


        if comm.rank == 0:
            with h5.File(setup['ofile'].format(**setup), 'a') as store:
                store[dest_group + 'gtau_d'] = gtau_d
                store[dest_group + 'gtau_o'] = gtau_o
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
