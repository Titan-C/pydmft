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
import json
import sys
import os
from mpi4py import MPI
import numpy as np
import dmft.common as gf
import dmft.hirschfye as hf
import dmft.plot.hf_single_site as pss
COMM = MPI.COMM_WORLD


def dmft_loop_pm(simulation, U, g_iw_start=None):
    """Implementation of the solver"""
    setup = {'t':           .5,
             'SITES':       1,
            }

    current_u = 'U'+str(U)
    setup.update(simulation)
    setup['U'] = U
    setup['simt'] = 'PM' # simulation type ParaMagnetic

    tau, w_n, _, giw, v_aux, intm = hf.setup_PM_sim(setup)
    if setup['AFM']:
        giw = giw + np.array([[-1], [1]])*1e-3*giw.imag
        setup['simt'] = 'AFM' # simulation type Anti-Ferro-Magnetic

    if g_iw_start is not None:
        giw = g_iw_start

    gtau = gf.gw_invfouriertrans(giw, tau, w_n, [1., 0., .25])
    save_dir = os.path.join(setup['ofile'].format(**setup), current_u)
    try:
        with open(save_dir + '/setup', 'r') as conf:
            last_loop = json.load(conf)['last_loop']
        gtau = np.load(os.path.join(save_dir,
                                    'it{:03}'.format(last_loop),
                                    'gtau.npy'))
        last_loop += 1
    except (IOError, OSError):
        last_loop = 0

    for iter_count in range(last_loop, last_loop + setup['Niter']):
        # For saving in the h5 file
        work_dir = os.path.join(save_dir, 'it{:03}'.format(iter_count))
        setup['work_dir'] = work_dir

        if COMM.rank == 0:
            print('On loop', iter_count, 'beta', setup['BETA'], 'U', setup['U'])

        giw = gf.gt_fouriertrans(gtau, tau, w_n,
                                 pss.gf_tail(gtau, U, setup['MU']))

        if setup['AFM']:
            g0iw = 1/(1j*w_n + setup['MU'] - setup['t']**2 * giw[[1, 0]])
            g0tau = gf.gw_invfouriertrans(g0iw, tau, w_n, [1., 0., .25])
            gtu, gtd = hf.imp_solver([g0tau[0], g0tau[1]], v_aux, intm, setup)
            gtau = -np.squeeze([gtu, gtd])

        else:
            # enforce Half-fill, particle-hole symmetry
            giw.real = 0.

            g0iw = 1/(1j*w_n + setup['MU'] - setup['t']**2 * giw)
            g0tau = gf.gw_invfouriertrans(g0iw, tau, w_n, [1., 0., .25])
            gtu, gtd = hf.imp_solver([g0tau]*2, v_aux, intm, setup)
            gtau = -np.squeeze(0.5 * (gtu+gtd))


        if COMM.rank == 0:
            np.save(work_dir + '/gtau', gtau)
            with open(save_dir + '/setup', 'w') as conf:
                setup['last_loop'] = iter_count
                json.dump(setup, conf, indent=2)
        sys.stdout.flush()

    return giw


if __name__ == "__main__":

    SETUP = hf.do_input('DMFT Loop For the single band para-magnetic case')
    SETUP.add_argument('-afm', '--AFM', action='store_true',
                       help='Use the self-consistency for Antiferromagnetism')
    SETUP = vars(SETUP.parse_args())

    G_iw = None
    for u in SETUP['urange']:
        G_iw = dmft_loop_pm(SETUP, u, G_iw)
