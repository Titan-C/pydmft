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
                   'sweeps':      10000,
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


        converged = np.allclose(g_iw.data, oldg, atol=1e-2)
        loops += 1
        if loops > 8:
            converged = True

    print(loops)

    return simulation

if __name__ == "__main__":
    sim = dmft_loop_pm()
    for it in sorted(sim):
        if 'it' in it:
            oplot(sim[it]['Giw']['A','B'], lw=2, num=2)