# -*- coding: utf-8 -*-
r"""
================================
QMC Hirsch - Fye Impurity solver
================================

To treat the Anderson impurity model and solve it using the Hirsch - Fye
Quantum Monte Carlo algorithm for a paramagnetic impurity
"""

from __future__ import division, absolute_import, print_function

import numpy as np

import matplotlib.pyplot as plt
import dmft.hirschfye as hf
from dmft.common import gt_fouriertrans, gw_invfouriertrans, greenF,  tau_wn_setup


def dmft_loop_pm(gw=None, **kwargs):
    """Implementation of the solver"""
    parameters = {
                   'dtau_mc':     0.5,
                   'n_tau_mc':    32,
                   'BETA':        16,
                   'N_TAU':    2**11,
                   'N_MATSUBARA': 64,
                   'U':           2,
                   'mu':          0,
                   'loops':       8,
                   'sweeps':      15000,
                  }
    parameters['BETA'] = parameters['dtau_mc'] * parameters['n_tau_mc']

    simulation = {'parameters': parameters}
    v_aux = hf.ising_v(parameters['dtau_mc'], parameters['U'], parameters['n_tau_mc'])

    tau, w_n = tau_wn_setup(parameters)
    if gw is None:
        Giw = greenF(w_n, mu=parameters['mu'])
    else:
        Giw = gw

    for iter_count in range(parameters['loops']):
        G0iw = 1/(1j*w_n + parameters['mu'] - .25*Giw)
        G0t = gw_invfouriertrans(G0iw, tau, w_n)
        g0t = hf.extract_g0t(G0t, parameters['n_tau_mc'])

        gtu, gtd = hf.imp_solver(-g0t, v_aux, parameters['sweeps'])
        gt = -0.5 * (gtu+gtd)

        Gt = hf.interpol(gt, parameters['N_TAU'])
        Giw = gt_fouriertrans(Gt, tau, w_n)
        simulation['it{:0>2}'.format(iter_count)] = {
                            'G0iw': G0iw,
                            'Giw':  Giw,
                            'gtau': gt,
                            }
    return simulation

if __name__ == "__main__":
    sim1 = dmft_loop_pm()
    for it in sorted(sim2):
        if 'it' in it:
#            plt.plot(s['Giw'].real.T, label=it)
            plt.plot(sim2[it]['Giw'].imag,'o-', label=it)
    plt.legend()
#    plt.figure()
#    for it in sorted(sim2):
#        if 'it' in it:
##            plt.plot(s['Giw'].real.T, label=it)
#            plt.plot(sim2[it]['Giw'].mean(axis=0).T.imag, 's-', label=it)
#    plt.legend()
#    sim2=hf.dmft_loop(3.9, gw=sim[-1]['Giw'])
#    plt.figure()
#    sim3=hf.dmft_loop(2.8, gw=sim2[-1]['Giw'])
#    plt.figure()
#    for i,s in enumerate(sim3):plt.plot(s['Giw'].imag, label=str(i))
#    plt.figure()
#    for i,s in enumerate(sim3):plt.plot(s['gtau'], label=str(i))