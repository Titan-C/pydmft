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


def dmft_loop_pm(simulation={}):
    """Implementation of the solver"""
    setup = {
             'n_tau_mc':    64,
             'BETA':        32,
             'N_TAU':    2**11,
             'N_MATSUBARA': 500,
             'U':           2.2,
             't':           .5,
             'MU':          0,
             'SITES':       1,
             'loops':       2,
             'sweeps':      2000,
             'therm':       5000,
             'N_meas':      2,
             'save_logs':   False,
             'updater':     'discrete'
            }

    setup.update(simulation.pop('setup', {}))
    tau, w_n, __, Giw, v_aux, intm = hf.setup_PM_sim(setup)

    simulation.update({'setup': setup})

    last_loop = len(simulation) - 1

    for iter_count in range(setup['loops']):
        G0iw = 1/(1j*w_n + setup['MU'] - setup['t']**2 * Giw)
        G0t = gw_invfouriertrans(G0iw, tau, w_n)
        g0t = hf.interpol(G0t, setup['n_tau_mc'])[:-1].reshape(-1, 1, 1)
        gtu, gtd = hf.imp_solver([g0t, g0t], v_aux, intm, setup)
        gt = -np.squeeze(0.5 * (gtu+gtd))

        Gt = hf.interpol(gt, setup['N_TAU'])
        Giw = gt_fouriertrans(Gt, tau, w_n)
        simulation['it{:0>2}'.format(last_loop + iter_count)] = {
                            'G0iw': G0iw.copy(),
                            'Giw':  Giw.copy(),
                            'gtau': gt.copy(),
                            }
        #patch tail on
        Giw.real = 0.
        Giw[setup['n_tau_mc']:] = -1j/w_n[setup['n_tau_mc']:]
    return simulation

if __name__ == "__main__":

    sim=dmft_loop_pm()
