# -*- coding: utf-8 -*-
r"""
================================
QMC Hirsch - Fye Impurity solver
================================

To treat the Anderson impurity model and solve it using the Hirsch - Fye
Quantum Monte Carlo algorithm for a paramagnetic impurity
"""

from __future__ import division, absolute_import, print_function

import dmft.hirschfye as hf
import numpy as np
import dmft.common as gf
import dmft.RKKY_dimer as rt
import sys
from pytriqs.gf.local import GfImFreq, iOmega_n

def dmft_loop_pm(u_int, tab, t, tn, beta, file_str, **params):
    """Implementation of the solver"""
    n_freq = int(15.*beta/np.pi)
    setup = {
               'BETA':        beta,
               'N_TAU':    2**13,
               'n_points': n_freq,
               'dtau_mc': 0.5,
               'U':           u_int,
               't':           t,
               'tp':          tab,
               'MU':          0.,
               'BANDS': 1,
               'SITES': 2,
               'loops':       0,  # starting loop count
               'max_loops':   20,
               'sweeps':      int(2e4),
               'therm':       int(5e3),
               'N_meas':      3,
               'save_logs':   False,
               'updater':     'discrete',
               'convegence_tol': 4e-3,
              }
    setup.update(params)

    try:  # try reloading data from disk
        with rt.HDFArchive(file_str.format(**setup), 'r') as last_run:
            lastU = 'U'+str(u_int)
            lastit = last_run[lastU].keys()[-1]
            setup = last_run[lastU][lastit]['setup']
            setup.update(params)
    except (IOError, KeyError):  # if no data clean start
        pass
    finally:
        S = rt.Dimer_Solver_hf(**setup)
        w_n = gf.matsubara_freq(setup['BETA'], setup['n_points'])
        rt.init_gf_met(S.g_iw, w_n, setup['MU'], setup['tp'], 0., t)

    tau = np.arange(0, S.setup['BETA'], S.setup['dtau_mc'])
    S.setup['n_tau_mc'] = len(tau)

    gmix = rt.mix_gf_dimer(S.g_iw.copy(), iOmega_n, setup['MU'], setup['tp'])

    S.V_field = hf.ising_v(S.setup['dtau_mc'], S.U,
                           L=S.setup['SITES']*S.setup['n_tau_mc'])
    dimer_loop(S, gmix, tau, file_str, '/U{U}/')


def dimer_loop(S, gmix, tau, filename, step):
    converged = False
    rt.recover_lastit(S, filename)
    while not converged:
        rt.gf_symetrizer(S.g_iw)

        oldg = S.g_iw.data.copy()
        # Bethe lattice bath
        S.g0_iw << gmix - 0.25 * S.g_iw
        S.g0_iw.invert()
        S.solve(tau)

        converged = np.allclose(S.g_iw.data, oldg,
                                atol=S.setup['convegence_tol'])

        loop_count = S.setup['loops'] + 1
        max_dist = np.max(abs(S.g_iw.data - oldg))
        print('B', S.beta, 'tp', S.setup['tp'], 'U:', S.U, 'l:', loop_count,
              converged, max_dist)
        sys.stdout.flush()

        S.setup.update({'U': S.U, 'loops': loop_count})
        rt.store_sim(S, filename, step+'it{:02}/'.format(loop_count))


#        ct = '-' if loops<8 else '+--'
#        oplot(S.g_iw['A', 'A'], ct, RI='I', label='d'+str(loops), num=6)
#        oplot(S.g_iw['A', 'B'], ct, RI='R', label='o'+str(loops), num=7)

        if loop_count >= S.setup['max_loops']:
            converged = True


if __name__ == "__main__":
    dmft_loop_pm([2.5], 0.23, 0.5, 0., 40., 'disk/metf_HF_Ul_tp{tp}_B{BETA}.h5',
                 dtau_mc=0.5, sweeps=20000, max_loops=1)

    from dimer_plots import plot_gf_loopU, plot_gf_iter
    plot_gf_loopU(40., 0.23, 2.5, 'disk/metf_HF_Ul_tp{}_B{}.h5', 5)
