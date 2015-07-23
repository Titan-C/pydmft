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
    Fourier, inverse, TailGf, iOmega_n, inverse
import dmft.RKKY_dimer_IPT as rt
from pytriqs.plot.mpl_interface import oplot
import argparse
from joblib import Parallel, delayed


def dmft_loop_pm(urange, tab, t, tn, beta, file_str):
    """Implementation of the solver"""
    n_freq = int(15.*beta/np.pi)
    setup = {
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
               'sweeps':      500000,
               'therm':       80000,
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
#        S.setup['dtau_mc'] = min(0.5, 0.3/S.U)
        S.setup['dtau_mc'] = 0.5
        tau = np.arange(0, S.setup['BETA'], S.setup['dtau_mc'])
        S.setup['n_tau_mc'] = len(tau)
        S.V_field = hf.ising_v(S.setup['dtau_mc'], S.U, L=S.setup['SITES']*S.setup['n_tau_mc'])
        dimer_loop(S, gmix, tau, file_str, '/U{U}/')

def restartt_dmft(file_str, urange, **params):
    last_run = rt.HDFArchive(file_str, 'r')
    lastU = last_run.keys()[-1]
    lastit = last_run[lastU].keys()[-1]
    setup = last_run[lastU][lastit]['setup']
    setup.update(params)

    S = rt.Dimer_Solver_hf(**setup)
    gmix = rt.mix_gf_dimer(S.g_iw.copy(), iOmega_n, setup['MU'], setup['tp'])
    rt.load_gf(S.g_iw, last_run[lastU][lastit]['G_iwd'],
               last_run[lastU][lastit]['G_iwo'])
    del last_run
    nurange = [u for u in urange if u>=float(lastU[1:])]
    for u_int in nurange:
        S.U = u_int
#        S.setup['dtau_mc'] = min(0.5, 0.3/S.U)
        tau = np.arange(0, S.setup['BETA'], S.setup['dtau_mc'])
        S.setup['n_tau_mc'] = len(tau)
        S.V_field = hf.ising_v(S.setup['dtau_mc'], S.U, L=S.setup['SITES']*S.setup['n_tau_mc'])
        dimer_loop(S, gmix, tau, file_str, '/U{U}/', setup['loops'])

def get_selfE(G_iwd, G_iwo):
    nf = len(G_iwd.mesh)
    beta = G_iwd.beta
    g_iw = GfImFreq(indices=['A', 'B'], beta=beta, n_points=nf)
    rt.load_gf(g_iw, G_iwd, G_iwo)
    gmix = rt.mix_gf_dimer(g_iw.copy(), iOmega_n, 0, 0.2)
    sigma = g_iw.copy()
    sigma << gmix  - 0.25*g_iw - inverse(g_iw)
    return sigma

def dimer_loop(S, gmix, tau, filename, step, loop_count=0):
    converged = False
    while not converged:
#    for i in range(12):
        # Enforce DMFT Paramagnetic
        rt.gf_symetrizer(S.g_iw)

        oldg = S.g_iw.data.copy()
        # Bethe lattice bath
        S.g0_iw << gmix - 0.25 * S.g_iw
        S.g0_iw.invert()
        S.solve(tau)

        converged = np.allclose(S.g_iw.data, oldg, atol=4e-3)
        loop_count += 1
        S.setup.update({'U': S.U, 'loops': loop_count})
        rt.store_sim(S, filename, step+'it{:02}/'.format(loop_count))

        max_dist = np.max(abs(S.g_iw.data - oldg))
        print('B', S.beta, 'tp', S.setup['tp'], 'U:', S.U, 'l:', loop_count,
              converged, max_dist)

#        ct = '-' if loops<8 else '+--'
#        oplot(S.g_iw['A', 'A'], ct, RI='I', label='d'+str(loops), num=6)
#        oplot(S.g_iw['A', 'B'], ct, RI='R', label='o'+str(loops), num=7)

        if loop_count > 20:
            converged = True

    rt.store_sim(S, filename, step)

if __name__ == "__main__":
#    restartt_dmft('metf_HF_Ul_dt0.3_t0.5_tp0.2_B36.0.h5', [2.], sweeps=5000, dtau_mc=1. )

    parser = argparse.ArgumentParser(description='DMFT loop for a dimer bethe lattice solved by IPT')
    parser.add_argument('beta', metavar='B', type=float,
                        default=16., help='The inverse temperature')
    parser.add_argument('-r','--restart', action='store_true',
                        help='Restart the job using data from disk')
#
#
    tabra = [.18, 0.22, 0.25, 0.27]
    args = parser.parse_args()
    BETA = args.beta

    ur = np.arange(2, 3, 0.1)
    Parallel(n_jobs=4, verbose=5)(delayed(dmft_loop_pm)(ur,
         tab, 0.5, 0., BETA, 'disk/metf_HF_Ul_t{t}_tp{tp}_B{BETA}.h5')
         for tab in tabra)