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
import argparse
from joblib import Parallel, delayed


def dmft_loop_pm(urange, tab, t, tn, beta, file_str):
    """Implementation of the solver"""
    n_freq = int(15.*beta/np.pi)
    setup = {
               'n_tau_mc':    32,
               'BETA':        beta,
               'N_TAU':    2**13,
               'n_points': n_freq,
               'U':           0.,
               't':           t,
               'tp':          tab,
               'MU':          0.,
               'BANDS': 1,
               'SITES': 2,
               'loops':       1,
               'sweeps':      25000,
               'therm':       2500,
               'N_meas':      3,
               'save_logs':   False,
               'updater':     'discrete'
              }


    w_n = gf.matsubara_freq(setup['BETA'], setup['n_points'])
    S = rt.Dimer_Solver_hf(**setup)

    gmix = rt.mix_gf_dimer(S.g_iw.copy(), iOmega_n, setup['MU'], setup['tp'])
    rt.init_gf_met(S.g_iw, w_n, setup['MU'], setup['tp'], 0., 0.5)
    for u_int in urange:
        S.U = u_int
        S.setup['dtau_mc'] = min(0.5, 0.3/S.U)
        tau = np.arange(0, S.setup['BETA'], S.setup['dtau_mc'])
        S.setup['n_tau_mc'] = len(tau)
        S.V_field = hf.ising_v(S.setup['dtau_mc'], S.U, L=S.setup['SITES']*S.setup['n_tau_mc'])
        dimer_loop(S, gmix, tau, file_str, '/U{U}/')

def dimer_loop(S, gmix, tau, filename, step):
    converged = False
    loops = 0
    while not converged:
        # Enforce DMFT Paramagnetic
        rt.gf_symetrizer(S.g_iw)

        oldg = S.g_iw.data.copy()
        # Bethe lattice bath
        S.g0_iw << gmix - 0.25 * S.g_iw
        S.g0_iw.invert()
        S.solve(tau)

        converged = np.allclose(S.g_iw.data, oldg, atol=5e-3)
        loops += 1
#        simulation['it{:0>2}'.format(loops)] = {
#                            'Giw':  S.g_iw.copy(),
#                            }
        print(loops, converged, np.max(abs(S.g_iw.data- oldg)))
        oplot(S.g_iw['A', 'A'], RI='I', label='d'+str(loops))
        oplot(S.g_iw['A', 'B'], RI='R', label='o'+str(loops))
        if loops > 5:
            converged = True

    S.setup.update({'U': S.U, 'loops': loops})
#    rt.store_sim(S, filename, step)


if __name__ == "__main__":
    sim = dmft_loop_pm([1.5], 0.2, 0.5, 0., 12., 'tes.h5')
#    plt.figure()
#    for it in sorted(sim):
#        if 'it' in it:
#            oplot(sim[it]['Giw']['B','B'], lw=2)
#    parser = argparse.ArgumentParser(description='DMFT loop for a dimer bethe lattice solved by IPT')
#    parser.add_argument('beta', metavar='B', type=float,
#                        default=16., help='The inverse temperature')
#
#
#    tabra = np.hstack((np.arange(0, 0.5, 0.1), np.arange(0.5, 1.1, 0.2)))
#    args = parser.parse_args()
#    BETA = args.beta
#
#    ur = np.arange(0, 4.5, 0.3)
#    Parallel(n_jobs=-1, verbose=5)(delayed(dmft_loop_pm)(ur,
#         tab, 0.5, 0., BETA, 'disk/met_HF_Ul_t{t}_tp{tp}_B{beta}.h5')
#         for tab in tabra)
