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
import dmft.plot.hf_single_site as pss
import numpy as np
import sys
comm = MPI.COMM_WORLD


def set_new_seed(setup):
    """Generates a new starting Green's function for the DMFT loop
    based on the finishing state of the system at a diffent parameter set"""

    src_U = 'U' + str(setup['new_seed'][0])
    dest_U = 'U' + str(setup['new_seed'][1])
    avg_over = int(setup['new_seed'][2])

    with h5.File(setup['ofile'].format(**SETUP), 'a') as outp:
        last_iterations = outp[src_U].keys()[-avg_over:]
        gtau = pss.averager(outp[src_U], last_iterations)
        # This is a particular cleaning for the half-filled single band
        try:
            dest_count = len(outp[dest_U].keys())
        except KeyError:
            dest_count = 0
        dest_group = '/{}/it{:03}/'.format(dest_U, dest_count)

        outp[dest_group + 'gtau'] = gtau
        outp.flush()
        h5.add_attributes(outp[dest_group],
                          h5.get_attribites(outp[src_U][last_iterations[-1]]))

    print(setup['new_seed'])


def dmft_loop_pm(simulation):
    """Implementation of the solver"""
    setup = {'t':           .5,
             'SITES':       1,
             }



    if simulation['new_seed']:
        if comm.rank == 0:
            set_new_seed(simulation)
        simulation['U'] = simulation['new_seed'][1]
        return

    current_u = 'U'+str(simulation['U'])
    setup.update(simulation)

    tau, w_n, _, giw, v_aux, intm = hf.setup_PM_sim(setup)

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

        # patch tail on
        giw.real = 0.

        g0iw = 1/(1j*w_n + setup['MU'] - setup['t']**2 * giw)
        g0tau = gf.gw_invfouriertrans(g0iw, tau, w_n)
        gtu, gtd = hf.imp_solver([g0tau]*2, v_aux, intm, setup)
        gtau = -np.squeeze(0.5 * (gtu+gtd))

        giw = gf.gt_fouriertrans(gtau, tau, w_n)

        if comm.rank == 0:
            with h5.File(setup['ofile'].format(**setup), 'a') as store:
                store[dest_group + 'gtau'] = gtau
                h5.add_attributes(store[dest_group], setup)
        sys.stdout.flush()


if __name__ == "__main__":

    SETUP = hf.do_input('DMFT Loop For the single band para-magnetic case')
    SETUP = vars(SETUP.parse_args())

    dmft_loop_pm(SETUP)
