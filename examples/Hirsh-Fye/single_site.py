# -*- coding: utf-8 -*-
r"""
================================
QMC Hirsch - Fye Impurity solver
================================

To treat the Anderson impurity model and solve it using the Hirsch - Fye
Quantum Monte Carlo algorithm for a paramagnetic impurity
"""

from __future__ import division, absolute_import, print_function

import dmft.common as gf
import dmft.hirschfye as hf
import numpy as np
import shelve
import argparse
from mpi4py import MPI
comm = MPI.COMM_WORLD


def do_input():
    """Prepares the input for the simulation at hand"""

    parser = argparse.ArgumentParser(description='DMFT loop for Hirsh-Fye single band')
    parser.add_argument('-BETA', metavar='B', type=float,
                        default=32., help='The inverse temperature')
    parser.add_argument('-n_tau_mc', metavar='B', type=int,
                        default=64, help='Number of time slices')
    parser.add_argument('-sweeps', metavar='MCS', type=int, default=int(1e5),
                        help='Number Monte Carlo Measurement')
    parser.add_argument('-therm', type=int, default=int(1e4),
                        help='Monte Carlo sweeps of thermalization')
    parser.add_argument('-N_meas', type=int, default=3,
                        help='Number of Updates before measurements')
    parser.add_argument('-Niter', metavar='N', type=int,
                        default=20, help='Number of iterations')
    parser.add_argument('-U', type=float, nargs='+',
                        default=[2.5], help='Local interaction strenght')
    parser.add_argument('pref', default='st', help='fileprefix')

    parser.add_argument('-M', '--Heat_bath', action='store_false',
                        help='Use Metropolis importance sampling')
    parser.add_argument('-r', '--resume', action='store_true',
                        help='Resume DMFT loops from inside folder. Do not'
                        'copy a seed file from the main directory')
    parser.add_argument('-liter', metavar='N', type=int, default=5,
                        help='On resume, average over liter[ations]')
    return vars(parser.parse_args())


def dmft_loop_pm(simulation, **kwarg):
    """Implementation of the solver"""
    setup = {'N_TAU':    2**11,
             'N_MATSUBARA': 512,
             't':           .5,
             'MU':          0,
             'SITES':       1,
             'save_logs':   False,
             'updater':     'discrete'}

    setup.update(simulation.pop('setup', {}))
    setup.update(kwarg)
    tau, w_n, _, giw, v_aux, intm = hf.setup_PM_sim(setup)

    simulation.update({'setup': setup})
    simulation['U'] = kwarg['U']

    current_u = 'U'+str(setup['U'])
    try:
        last_loop = len(simulation[current_u])
        giw = simulation[current_u]['it{:0>2}'.format(last_loop-1)]['giw'].copy()
    except Exception:
        last_loop = 0
        simulation.update({current_u: {}})

    for iter_count in range(last_loop, last_loop + setup['Niter']):
        if comm.rank == 0:
            print('On loop', iter_count, 'beta', setup['BETA'], 'U', setup['U'])

        # patch tail on
        giw.real = 0.
        giw[setup['n_tau_mc']//2:] = -1j/w_n[setup['n_tau_mc']//2:]

        g0iw = 1/(1j*w_n + setup['MU'] - setup['t']**2 * giw)
        g0tau = gf.gw_invfouriertrans(g0iw, tau, w_n)
        g0t = hf.interpol(g0tau, setup['n_tau_mc'])[:-1].reshape(-1, 1, 1)
        gtu, gtd = hf.imp_solver([g0t, g0t], v_aux, intm, setup)
        gt = -np.squeeze(0.5 * (gtu+gtd))

        gtau = hf.interpol(gt, setup['N_TAU'])
        giw = gf.gt_fouriertrans(gtau, tau, w_n)

        if comm.rank == 0:
            simulation[current_u]['it{:0>2}'.format(iter_count)] = {
                                'g0iw': g0iw.copy(),
                                'setup': setup.copy(),
                                'giw':  giw.copy(),
                                'gtau': gt.copy(),
                                }
            simulation.sync()


if __name__ == "__main__":

    SETUP = do_input()
    U_rang = SETUP.pop('U')
    sim = shelve.open(SETUP['pref'] + 'stb{BETA}_met'.format(**SETUP),
                      writeback=True)
    sim['setup'] = SETUP
    for u_int in U_rang:
        dmft_loop_pm(sim, U=u_int)
    sim.close()
