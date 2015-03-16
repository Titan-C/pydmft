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
from dmft.common import gt_fouriertrans, gw_invfouriertrans, greenF,  matsubara_freq


def dmft_loop_pm(gw=None, **kwargs):
    """Implementation of the solver"""
    parameters = {
                   'dtau_mc':     0.5,
                   'n_tau_mc':    32,
                   'beta':        16,
                   'tau_rang':    1024,
                   'n_matsubara': 64,
                   'U':           2,
                   'mu':          0.0,
                   'loops':       8,
                   'sweeps':      5000,
                  }
    parameters['beta'] = parameters['dtau_mc'] * parameters['n_tau_mc']

    i_omega = matsubara_freq(parameters['beta'],
                             parameters['n_matsubara'])
    fine_tau = np.linspace(0, parameters['beta'],
                              parameters['tau_rang'] + 1)
    if gw is None:
        Giw = greenF(i_omega, parameters['mu'])
    else:
        Giw = gw
    v_aux = hf.ising_v(parameters['dtau_mc'], parameters['U'], parameters['n_tau_mc'])
    simulation = {'parameters': parameters}

    for iter_count in range(parameters['loops']):
        G0iw = 1/(i_omega - .25*Giw)
        G0t = gw_invfouriertrans(G0iw, fine_tau, i_omega)
        g0t = hf.extract_g0t(G0t, parameters['n_tau_mc'])

        gtu, gtd = hf.imp_solver(-g0t, v_aux, parameters['sweeps'])
        gt = (gtu + gtd) / 2

        Gt = hf.interpol(-gt, parameters['tau_rang'])
        Giw = gt_fouriertrans(Gt, fine_tau, i_omega)
        simulation['it{:0>2}'.format(iter_count)] = {
                            'G0iw': G0iw,
                            'Giw':  Giw,
                            'gtau': gt,
                            }
    return simulation

if __name__ == "__main__":
    sim = dmft_loop_pm(3.5)
    for it, s in sim.iteritems():
        if 'it' in it:
            plt.plot(s['Giw'].real, label=it)
            plt.plot(s['Giw'].imag, label=it)

#    sim2=hf.dmft_loop(3.9, gw=sim[-1]['Giw'])
#    plt.figure()
#    sim3=hf.dmft_loop(2.8, gw=sim2[-1]['Giw'])
#    plt.figure()
#    for i,s in enumerate(sim3):plt.plot(s['Giw'].imag, label=str(i))
#    plt.figure()
#    for i,s in enumerate(sim3):plt.plot(s['gtau'], label=str(i))