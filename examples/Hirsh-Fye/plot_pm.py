# -*- coding: utf-8 -*-
r"""
================================
QMC Hirsch - Fye Impurity solver
================================

To treat the Anderson impurity model and solve it using the Hirsch - Fye
Quantum Monte Carlo algorithm for a paramagnetic impurity
"""

from __future__ import division, absolute_import, print_function

import matplotlib.pyplot as plt
import dmft.hirschfye as hf
import numpy as np
from dmft.common import gt_fouriertrans, gw_invfouriertrans


def dmft_loop_pm(gw=None, **kwargs):
    """Implementation of the solver"""
    parameters = {
                   'n_tau_mc':    64,
                   'BETA':        32,
                   'N_TAU':    2**11,
                   'N_MATSUBARA': 500,
                   'U':           3.2,
                   't':           .5,
                   'MU':          0,
                   'SITES':       1,
                   'loops':       2,
                   'sweeps':      8000,
                   'therm':       8000,
                   'N_meas':      5,
                   'save_logs':   False,
                   'updater':     'discrete'
                  }

    tau, w_n, __, Giw, v_aux, intm = hf.setup_PM_sim(parameters)

    simulation = {'parameters': parameters}

    if gw is not None:
        Giw = gw

    for iter_count in range(parameters['loops']):
        G0iw = 1/(1j*w_n + parameters['MU'] - parameters['t']**2 * Giw)
        G0t = gw_invfouriertrans(G0iw, tau, w_n)
        g0t = hf.interpol(G0t, parameters['n_tau_mc'])[:-1].reshape(-1, 1, 1)
        gtu, gtd = hf.imp_solver([g0t, g0t], v_aux, intm, parameters)
        gt = -np.squeeze(0.5 * (gtu+gtd))

        Gt = hf.interpol(gt, parameters['N_TAU'])
        Giw = gt_fouriertrans(Gt, tau, w_n)
        simulation['it{:0>2}'.format(iter_count)] = {
                            'G0iw': G0iw.copy(),
                            'Giw':  Giw.copy(),
                            'gtau': gt.copy(),
                            }
        #patch tail on
        Giw.real = 0.
        Giw[parameters['n_tau_mc']:] = -1j/w_n[parameters['n_tau_mc']:]
    return simulation

if __name__ == "__main__":
    gw=np.load('fgiws500.npy')
    sim1 = dmft_loop_pm(gw)
    plt.figure()
    tau = np.linspace(0, sim1['parameters']['BETA'], sim1['parameters']['n_tau_mc']+1)
    for it in sorted(sim1):
        if 'it' in it:
            plt.plot(tau,-sim1[it]['gtau'], 'o', label=it)
    plt.legend()
    plt.figure()
    for it in sorted(sim1):
        if 'it' in it:
            plt.plot(sim1[it]['Giw'].imag, 'o-', label=it)
    plt.legend()
#    print(np.polyfit(tau[:10], np.log(-sim1['it09']['gtau'][:10]), 1))
    plt.show()