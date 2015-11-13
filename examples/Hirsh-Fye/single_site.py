#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
================================
QMC Hirsch - Fye Impurity solver
================================

To treat the Anderson impurity model and solve it using the Hirsch - Fye
Quantum Monte Carlo algorithm for a paramagnetic impurity
"""

from __future__ import division, absolute_import, print_function

from mpi4py import MPI
import dmft.common as gf
import dmft.h5archive as h5
import dmft.hirschfye as hf
import numpy as np
import sys
comm = MPI.COMM_WORLD


def dmft_loop_pm(simulation):
    """Implementation of the solver"""
    setup = {'t':           .5,
             'SITES':       1,
             }

    if simulation['new_seed']:
        if comm.rank == 0:
            hf.set_new_seed(simulation, ['gtau'])
        simulation['U'] = simulation['new_seed'][1]
        return

    current_u = 'U'+str(simulation['U'])
    setup.update(simulation)

    tau, w_n, _, giw, v_aux, intm = hf.setup_PM_sim(setup)
    if setup['AFM']:
        giw = giw + np.array([[-1], [1]])*1e-2/w_n**2

    try:
        with h5.File(setup['ofile'].format(**setup), 'a') as store:
            last_loop = len(store[current_u].keys())
            gtau = store[current_u]['it{:03}'.format(last_loop-1)]['gtau'][:]
            giw = gf.gt_fouriertrans(gtau, tau, w_n)
    except (IOError, KeyError):
        last_loop = 0

    for iter_count in range(last_loop, last_loop + setup['Niter']):
        # For saving in the h5 file
        dest_group = current_u+'/it{:03}/'.format(iter_count)
        setup['group'] = dest_group

        if comm.rank == 0:
            print('On loop', iter_count, 'beta', setup['BETA'], 'U', setup['U'])


        if setup['AFM']:
            g0iw = 1/(1j*w_n + setup['MU'] - setup['t']**2 * giw[[1, 0]])
            g0tau = gf.gw_invfouriertrans(g0iw, tau, w_n)
            gtu, gtd = hf.imp_solver([g0tau[0], g0tau[1]], v_aux, intm, setup)
            gtau = -np.squeeze([gtu, gtd])

            giw = gf.gt_fouriertrans(gtau, tau, w_n,
                                    [1., -setup['U']*(0.5+gtau[:, 0]).reshape(2, 1),
                                                    (setup['U']/2)**2])
        else:
            # enforce Half-fill, particle-hole symmetry
            giw.real = 0.

            g0iw = 1/(1j*w_n + setup['MU'] - setup['t']**2 * giw)
            g0tau = gf.gw_invfouriertrans(g0iw, tau, w_n)
            gtu, gtd = hf.imp_solver([g0tau]*2, v_aux, intm, setup)
            gtau = -np.squeeze(0.5 * (gtu+gtd))

            giw = gf.gt_fouriertrans(gtau, tau, w_n, [1., 0., setup['U']**2/4])

        if comm.rank == 0:
            with h5.File(setup['ofile'].format(**setup), 'a') as store:
                store[dest_group + 'gtau'] = gtau
                h5.add_attributes(store[dest_group], setup)
        sys.stdout.flush()


if __name__ == "__main__":

    SETUP = hf.do_input('DMFT Loop For the single band para-magnetic case')
    SETUP.add_argument('-afm', '--AFM', action='store_true',
                        help='Use the self-consistency for Antiferromagnetism')
    SETUP = vars(SETUP.parse_args())

    dmft_loop_pm(SETUP)
