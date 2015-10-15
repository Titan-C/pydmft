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
from pytriqs.archive import HDFArchive
from pytriqs.gf.local import iOmega_n
import argparse
import dmft.RKKY_dimer as rt
import dmft.common as gf
import dmft.hirschfye as hf
import numpy as np
import sys

comm = MPI.COMM_WORLD


def do_input():
    """Prepares the input for the simulation at hand"""

    parser = argparse.ArgumentParser(description='DMFT loop for Hirsh-Fye dimer lattice',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-BETA', metavar='B', type=float,
                        default=32., help='The inverse temperature')
    parser.add_argument('-dtau_mc', metavar='dt', type=float,
                        default=0.5, help='Trotter time step')
    parser.add_argument('-sweeps', metavar='MCS', type=int, default=int(5e4),
                        help='Number Monte Carlo Measurement')
    parser.add_argument('-therm', type=int, default=int(1e4),
                        help='Monte Carlo sweeps of thermalization')
    parser.add_argument('-N_meas', type=int, default=3,
                        help='Number of Updates before measurements')
    parser.add_argument('-Niter', metavar='N', type=int,
                        default=20, help='Number of iterations')
    parser.add_argument('-U', type=float, default=2.,
                        help='Local interaction strength')
    parser.add_argument('-tp', type=float, default=.25,
                        help='Dimerization strength')
    parser.add_argument('-df', '--double_flip_prob', type=float, default=0.,
                        help='Probability for double spin flip on equal sites')
    parser.add_argument('-ofile', default='disk/dim_PM_B{BETA}_t{tp}.h5',
                        help='Output file')

    parser.add_argument('-M', '--Heat_bath', action='store_false',
                        help='Use Metropolis importance sampling')
    parser.add_argument('-new_seed', type=float, nargs=3, default=False,
                        metavar=('U_src', 'U_target', 'avg_over'),
                        help='Resume DMFT loops from on disk data files')
    return vars(parser.parse_args())


def dmft_loop_pm(params):
    """Implementation of the solver"""
    n_freq = int(15.*params['BETA']/np.pi)
    setup = {'N_TAU':     2**10,
             'n_points':  n_freq,
             't':         0.5,
             'MU':        0.,
             'BANDS':     1,
             'SITES':     2,
             'save_logs': False,
             'global_flip': True,
             'updater':   'discrete'}

    setup.update(params)
    current_u = 'U'+str(setup['U'])

    S = rt.Dimer_Solver_hf(**setup)
    w_n = gf.matsubara_freq(setup['BETA'], setup['n_points'])
    rt.init_gf_met(S.g_iw, w_n, setup['MU'], setup['tp'], 0., setup['t'])

    try:  # try reloading data from disk
        with HDFArchive(setup['ofile'].format(**setup), 'r') as last_run:
            last_loop = len(last_run[current_u].keys())
            last_it = 'it{:03}'.format(last_loop-1)
            rt.load_gf(S.g_iw, last_run[current_u][last_it]['G_iwd'],
                       last_run[current_u][last_it]['G_iwo'])
    except (IOError, KeyError):  # if no data clean start
        last_loop = 0

    tau = np.arange(0, S.setup['BETA'], S.setup['dtau_mc'])
    S.setup['n_tau_mc'] = len(tau)

    gmix = rt.mix_gf_dimer(S.g_iw.copy(), iOmega_n, setup['MU'], setup['tp'])

    S.V_field = hf.ising_v(S.setup['dtau_mc'], S.U,
                           L=S.setup['SITES']*S.setup['n_tau_mc'])

    for loop_count in range(last_loop, last_loop + setup['Niter']):
        if comm.rank == 0:
            print('B', S.beta, 'tp', S.setup['tp'], 'U:', S.U, 'l:', loop_count)

        rt.gf_symetrizer(S.g_iw)

        # Bethe lattice bath
        S.g0_iw << gmix - 0.25 * S.g_iw
        S.g0_iw.invert()
        S.solve(tau)

        if comm.rank == 0:
            with HDFArchive(setup['ofile'].format(**setup), 'a') as simulation:
                simulation[current_u+'/it{:03}'.format(loop_count)] = {
                            'setup':  S.setup.copy(),
                            'G_iwd':  S.g_iw['A', 'A'],
                            'G_iwo':  S.g_iw['A', 'B'],
                            }
        sys.stdout.flush()

if __name__ == "__main__":
    SETUP = do_input()
    dmft_loop_pm(SETUP)
