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
import dmft.common as gf
from pytriqs.gf.local import GfImFreq, GfImTime, InverseFourier, \
    Fourier, inverse, TailGf, iOmega_n
import dmft.RKKY_dimer_IPT as rt
from pytriqs.plot.mpl_interface import oplot

def dmft_loop_pm(gw=None, **kwargs):
    """Implementation of the solver"""
    parameters = {
                   'n_tau_mc':    128,
                   'BETA':        16,
                   'N_TAU':    2**11,
                   'N_MATSUBARA': 128,
                   'U':           1.5,
                   't':           0.5,
                   'tp':          0.1,
                   'MU':          0,
                   'BANDS': 1,
                   'SITES': 2,
                   'loops':       1,
                   'sweeps':      50000,
                   'therm':       2000,
                   'N_meas':      4,
                   'save_logs':   False,
                   'updater':     'discrete'
                  }

    tau = np.arange(0, parameters['BETA'], parameters['BETA']/parameters['n_tau_mc'])
    w_n = gf.matsubara_freq(parameters['BETA'], parameters['N_MATSUBARA'])

    parameters['dtau_mc'] = parameters['BETA']/parameters['n_tau_mc']
    v = hf.ising_v(parameters['dtau_mc'], parameters['U'], L=parameters['SITES']*parameters['n_tau_mc'])
    g_iw = GfImFreq(indices=['A', 'B'], beta=parameters['BETA'],
                             n_points=len(w_n))
    g0_iw = g_iw.copy()
    g0_tau = GfImTime(indices=['A', 'B'], beta=parameters['BETA'], n_points=parameters['N_TAU'])
    g_tau = g0_tau.copy()
    gmix = rt.mix_gf_dimer(g_iw.copy(), iOmega_n, parameters['MU'], parameters['tp'])
    rt.init_gf_met(g_iw, w_n, 0, parameters['tp'], 0., 0.5)
    simulation = {'parameters': parameters}

    if gw is not None:
        Giw = gw

    converged = False
    loops = 0
    while not converged:
        # Enforce DMFT Paramagnetic, IPT conditions
        # Pure imaginary GF in diagonals
        g_iw.data[:, 0, 0] = 1j*g_iw.data[:, 0, 0].imag
        g_iw['B', 'B']<<g_iw['A', 'A']
        # Pure real GF in off-diagonals
#        S.g_iw.data[:, 0, 1] = S.g_iw.data[:, 1, 0].real
        g_iw['B', 'A']<< 0.5*(g_iw['A', 'B'] + g_iw['B', 'A'])
        g_iw['A', 'B']<<g_iw['B', 'A']

        oldg = g_iw.data.copy()
        # Bethe lattice bath
        g0_iw << gmix - 0.25 * g_iw
        g0_iw.invert()

        g0_tau << InverseFourier(g0_iw)

        g0t = np.asarray([g0_tau(t).real for t in tau])

        gtu, gtd = hf.imp_solver(g0t, g0t, v, parameters)
        gt_D = -0.25 * (gtu[0 ,0] + gtu[1, 1] + gtd[0, 0] + gtd[1, 1])
        gt_N = -0.25 * (gtu[1, 0] + gtu[0, 1] + gtd[1, 0] + gtd[0, 1])

        g_tau.data[:, 0, 0] = hf.interpol(gt_D, parameters['N_TAU']-1)
        g_tau['B', 'B'] << g_tau['A', 'A']
        g_tau.data[:, 0, 1] = hf.interpol(gt_N, parameters['N_TAU']-1)
        g_tau['B', 'A'] << g_tau['A', 'B']

        fixed_co = TailGf(2, 2, 4, -1)
        fixed_co[1] = np.array([[1, 0], [0, 1]])
        fixed_co[2] = parameters['tp']*np.array([[0, 1], [1, 0]])
        g_tau.tail = fixed_co

        g_iw << Fourier(g_tau)

        converged = np.allclose(g_iw.data, oldg, atol=1e-2)
        loops += 1
#        mix = mixer(loops)
        if loops > 8:
            converged = True

    print(loops)

    return simulation

if __name__ == "__main__":
    sim = dmft_loop_pm()
#    plt.figure(2)
#    tau = np.linspace(0, sim1['parameters']['BETA'], sim1['parameters']['n_tau_mc']+1)
    for it in sorted(sim):
        if 'it' in it:
            oplot(sim[it]['Giw']['A','B'], lw=2, num=2)
#            plt.semilogy(tau,-sim1[it]['gtaud'], 'o', label=it)
#    plt.legend()
#    print(np.polyfit(tau[:10], np.log(-sim1['it00']['gtaud'][:10]), 1))
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