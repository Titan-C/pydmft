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



def dmft_loop_pm(urange, tab, t, tn, beta, file_str, **params):
    """Implementation of the solver"""
    n_freq = int(15.*beta/np.pi)
    setup = {
               'BETA':        beta,
               'N_TAU':    2**13,
               'n_points': n_freq,
               'dtau_mc': 0.5,
               'U':           0.,
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
        last_run = rt.HDFArchive(file_str.format(**setup), 'r')
        lastU = 'U'+str(urange[0])
        lastit = last_run[lastU].keys()[-1]
        setup = last_run[lastU][lastit]['setup']
        setup.update(params)
        S = rt.Dimer_Solver_hf(**setup)
        rt.load_gf(S.g_iw, last_run[lastU][lastit]['G_iwd'],
                   last_run[lastU][lastit]['G_iwo'])
        del last_run
    except Exception:  # if no data clean start
        S = rt.Dimer_Solver_hf(**setup)
        w_n = gf.matsubara_freq(setup['BETA'], setup['n_points'])
        rt.init_gf_met(S.g_iw, w_n, setup['MU'], setup['tp'], 0., t)

    tau = np.arange(0, S.setup['BETA'], S.setup['dtau_mc'])
    S.setup['n_tau_mc'] = len(tau)

    gmix = rt.mix_gf_dimer(S.g_iw.copy(), iOmega_n, setup['MU'], setup['tp'])

    for u_int in urange:
        S.U = u_int
        S.setup['max_loops'] = setup['loops'] + setup['max_loops']
        S.V_field = hf.ising_v(S.setup['dtau_mc'], S.U,
                               L=S.setup['SITES']*S.setup['n_tau_mc'])
        dimer_loop(S, gmix, tau, file_str, '/U{U}/')


def get_selfE(G_iwd, G_iwo):
    nf = len(G_iwd.mesh)
    beta = G_iwd.beta
    g_iw = GfImFreq(indices=['A', 'B'], beta=beta, n_points=nf)
    rt.load_gf(g_iw, G_iwd, G_iwo)
    gmix = rt.mix_gf_dimer(g_iw.copy(), iOmega_n, 0, 0.2)
    sigma = g_iw.copy()
    sigma << gmix - 0.25*g_iw - inverse(g_iw)
    return sigma


def dimer_loop(S, gmix, tau, filename, step):
    converged = False
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
        S.setup.update({'U': S.U, 'loops': loop_count})
        print(S.setup, loop_count)
        rt.store_sim(S, filename, step+'it{:02}/'.format(loop_count))

        max_dist = np.max(abs(S.g_iw.data - oldg))
        print('B', S.beta, 'tp', S.setup['tp'], 'U:', S.U, 'l:', loop_count,
              converged, max_dist)

#        ct = '-' if loops<8 else '+--'
#        oplot(S.g_iw['A', 'A'], ct, RI='I', label='d'+str(loops), num=6)
#        oplot(S.g_iw['A', 'B'], ct, RI='R', label='o'+str(loops), num=7)

        if loop_count > S.setup['max_loops']:
            converged = True


def plot_gf_loops(tab, beta, filestr, nf):
    R = rt.HDFArchive(filestr.format(tab, beta), 'r')
    w_n = gf.matsubara_freq(beta, 2*nf)
    for ru in R.keys():
        diag_f = []
        offdiag_f = []
        f, ax = plt.subplots(1, 2, figsize=(18, 8), sharex=True)
        f.subplots_adjust(hspace=0.2)
        gfin = f.add_axes([0.16, 0.17, 0.20, 0.25])
        for u_iter in R[ru].keys():
            if 'it' in u_iter:
                diag_f.append(R[ru][u_iter]['G_iwd'].data[:nf, 0, 0].imag)
                offdiag_f.append(R[ru][u_iter]['G_iwo'].data[:nf, 0, 0].real)
                gfin.plot(w_n, R[ru][u_iter]['G_iwd'].data[:2*nf, 0, 0].imag, 'bs:')
                gfin.plot(w_n, R[ru][u_iter]['G_iwo'].data[:2*nf, 0, 0].real, 'gs:')

        diag_f = np.asarray(diag_f).T
        offdiag_f = np.asarray(offdiag_f).T
        gfin.set_xticks(gfin.get_xlim())
        gfin.set_yticks(gfin.get_ylim())
        plt.axhline()
        for freq, (hd, ho) in enumerate(zip(diag_f, offdiag_f)):
            ax[0].plot(hd,'o-.', label='n='+str(freq+1))
            ax[1].plot(ho,'o-.', label='n='+str(freq+1))
        ax[1].legend(loc=3,prop={'size':18})
        plt.suptitle('First frequencies of the Matsubara GF, at iteration @ U/D={} $t_{{ab}}/D={}$ $\\beta D={}$'.format(ru, tab, beta))
        #show()
        #close()
    del R
if __name__ == "__main__":
    dmft_loop_pm([2.7], 0.2, 0.5, 0., 36., '1metf_HF_Ul_dt0.3_t{t}_tp{tp}_B{BETA}.h5',
                 dtau_mc=.4, sweeps=10000, max_loops=10)

    plot_gf_loops(0.2, 36., '1metf_HF_Ul_dt0.3_t0.5_tp{}_B{}.h5', 5 )

