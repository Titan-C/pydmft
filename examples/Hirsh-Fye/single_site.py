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
import matplotlib.pyplot as plt
import numpy as np
import shelve
import argparse

def do_input():

    parser = argparse.ArgumentParser(description='DMFT loop for Hirsh-Fye single band')
    parser.add_argument('-beta', metavar='B', type=float,
                        default=32., help='The inverse temperature')
    parser.add_argument('-n_tau', metavar='B', type=float,
                        default=64., help='Number of time slices')
    parser.add_argument('-Niter', metavar='N', type=int,
                        default=20, help='Number of iterations')
    parser.add_argument('-U', metavar='U', nargs='+', type=float,
                        default=[2.7], help='Local interaction strenght')
    parser.add_argument('-r', '--resume', action='store_true',
                        help='Resume DMFT loops from inside folder. Do not copy'
                        'a seed file from the main directory')
    parser.add_argument('-liter', metavar='N', type=int,
                        default=5, help='On resume, average over liter[ations]')
    return vars(parser.parse_args())


def dmft_loop_pm(simulation={}, **kwarg):
    """Implementation of the solver"""
    setup = {
             'n_tau_mc':    64,
             'BETA':        64,
             'N_TAU':    2**11,
             'N_MATSUBARA': 512,
             'U':           2.2,
             't':           .5,
             'MU':          0,
             'SITES':       1,
             'loops':       10,
             'sweeps':      100000,
             'therm':       10000,
             'N_meas':      3,
             'save_logs':   False,
             'updater':     'discrete'
            }

    setup.update(simulation.pop('setup', {}))
    setup.update(kwarg)
    tau, w_n, __, Giw, v_aux, intm = hf.setup_PM_sim(setup)

    simulation.update({'setup': setup})

    current_u = 'U'+str(setup['U'])
    try:
        last_loop = len(simulation[current_u])
        Giw = simulation[current_u]['it{:0>2}'.format(last_loop-1)]['Giw'].copy()
    except Exception:
        last_loop = 0
        simulation.update({current_u:{}})

    for iter_count in range(setup['loops']):
        #patch tail on
        print('On loop', iter_count, 'beta', setup['BETA'])
        Giw.real = 0.
        Giw[setup['n_tau_mc']//2:] = -1j/w_n[setup['n_tau_mc']//2:]

        G0iw = 1/(1j*w_n + setup['MU'] - setup['t']**2 * Giw)
        G0t = gf.gw_invfouriertrans(G0iw, tau, w_n)
        g0t = hf.interpol(G0t, setup['n_tau_mc'])[:-1].reshape(-1, 1, 1)
        gtu, gtd = hf.imp_solver([g0t, g0t], v_aux, intm, setup)
        gt = -np.squeeze(0.5 * (gtu+gtd))

        Gt = hf.interpol(gt, setup['N_TAU'])
        Giw = gf.gt_fouriertrans(Gt, tau, w_n)
        simulation[current_u]['it{:0>2}'.format(last_loop + iter_count)] = {
                            'G0iw': G0iw.copy(),
                            'setup': setup.copy(),
                            'Giw':  Giw.copy(),
                            'gtau': gt.copy(),
                            }
        simulation.sync()

if __name__ == "__main__":


    sim = shelve.open('HF_stb64_met', writeback=True)
    dmft_loop_pm(sim, n_tau_mc=128)
    dmft_loop_pm(sim, n_tau_mc=256)

    sim.update({'U2.7': {'it00': {'Giw': sim['U2.2']['it19']['Giw']}}})
    dmft_loop_pm(sim, U=2.7, n_tau_mc=128,
                  Heat_bath=False)
    dmft_loop_pm(sim, n_tau_mc=256)


    sim.update({'U3.2': {'it00': {'Giw': sim['U2.7']['it19']['Giw']}}})
    dmft_loop_pm(sim, U=3.2, n_tau_mc=128,
                 Heat_bath=False)
    dmft_loop_pm(sim, n_tau_mc=256)

    shelve.close()
