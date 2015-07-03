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
    setup = {
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

    setup['dtau_mc'] = setup['BETA']/setup['n_tau_mc']
    tau = np.arange(0, setup['BETA'], )
    w_n = gf.matsubara_freq(parameters['BETA'], parameters['N_MATSUBARA'])

    gmix = rt.mix_gf_dimer(g_iw.copy(), iOmega_n, parameters['MU'], parameters['tp'])
    rt.init_gf_met(g_iw, w_n, setup['MU'], setu['tp'], 0., 0.5)
    simulation = {'parameters': setup}


    converged = False
    loops = 0
    while not converged:
        # Enforce DMFT Paramagnetic
        rt.symetrizer(g_iw)

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