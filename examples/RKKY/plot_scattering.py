# -*- coding: utf-8 -*-
r"""
================
Scattering rates
================

"""
# Created Mon Mar  7 01:14:02 2016
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function

from math import log, ceil
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import dmft.RKKY_dimer as rt
import dmft.common as gf
import dmft.ipt_imag as ipt


def loop_beta(u_int, tp, betarange, seed='mott gap'):
    """Solves IPT dimer and return Im Sigma_AA, Re Simga_AB

    returns list len(betarange) x 2 Sigma arrays
"""

    sigma_iw = []
    iterations = []
    for beta in betarange:
        tau, w_n = gf.tau_wn_setup(
            dict(BETA=beta, N_MATSUBARA=max(2**ceil(log(4 * beta) / log(2)), 256)))

        giw_d, giw_o = rt.gf_met(w_n, 0., 0., 0.5, 0.)
        if seed == 'I':
            giw_d, giw_o = 1 / (1j * w_n - 4j / w_n), np.zeros_like(w_n) + 0j
        giw_d, giw_o, loops = rt.ipt_dmft_loop(
            beta, u_int, tp, giw_d, giw_o, tau, w_n, 1e-8)
        iterations.append(loops)
        g0iw_d, g0iw_o = rt.self_consistency(
            1j * w_n, 1j * giw_d.imag, giw_o.real, 0., tp, 0.25)
        siw_d, siw_o = ipt.dimer_sigma(u_int, tp, g0iw_d, g0iw_o, tau, w_n)
        sigma_iw.append((siw_d.imag, siw_o.real))

    print(np.array(iterations))

    return sigma_iw

temp = np.arange(1 / 500., .061, 1 / 250)
BETARANGE = 1 / temp

###############################################################################
# Calculate in the METAL dome the metal and insulator solution fix tp

U_int = [2.5, 3., 3.5]
sigmasM_U = Parallel(n_jobs=-1)(delayed(loop_beta)(u_int, .3, BETARANGE, 'M')
                                for u_int in U_int)

sigmasI_U = Parallel(n_jobs=-1)(delayed(loop_beta)(u_int, .3, BETARANGE, 'I')
                                for u_int in U_int)

# Plot a 2x2 frame following sigma and d_sigma Changing U


def plot_zero_w(function_array, iter_range, betarange, entry, label_head, ax, dx):
    """Plot the zero frequency extrapolation of a function
    Parameters
    ----------
    function_array: real ndarray
      contains the function (G, Sigma) to linearly fit over 2 first frequencies
    iter_range: list floats
      values of changing variable U or tp
    berarange: real ndarray 1D, values of beta
    entry: 0, 1 corresponds to diagonal or off-diagonal entry of function
    label_head: string for label
    ax, dx: matplotlib axis to plot in
    """

    for j, u in enumerate(iter_range):
        dat = []
        for i, beta in list(enumerate(betarange)):
            tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=20))
            dat.append(np.polyfit(
                w_n[:2], function_array[j][i][entry][:2], 1))
        ax.plot(1 / BETARANGE, -np.array(dat)[:, 1], label=label_head + str(u))
        dx.plot(1 / BETARANGE, -np.array(dat)[:, 0], label=label_head + str(u))

f, (si, ds) = plt.subplots(2, 2, sharex=True)
plot_zero_w(sigmasI_U, U_int, BETARANGE, 0, 'INS U=', si[0], ds[0])
plot_zero_w(sigmasM_U, U_int, BETARANGE, 0, 'MET U=', si[0], ds[0])

plot_zero_w(sigmasI_U, U_int, BETARANGE, 1, 'INS U=', si[1], ds[1])
plot_zero_w(sigmasM_U, U_int, BETARANGE, 1, 'MET U=', si[1], ds[1])

si[0].legend(loc=0, prop={'size': 8})

si[0].set_ylabel(r'-$\Im \Sigma_{AA}(w=0)$')
ds[0].set_ylabel(r'-$\Im d\Sigma_{AA}(w=0)$')
si[1].set_ylabel(r'-$\Im \Sigma_{AB}(w=0)$')
ds[1].set_ylabel(r'-$\Im d\Sigma_{AB}(w=0)$')
ds[0].set_xlim([0, .06])

ds[0].set_xlabel('$T/D$')
ds[1].set_xlabel('$T/D$')

###############################################################################
# Changing tp in the METAL dome for metal and insulator Fix U

tp_r = [0.2, 0.3, 0.4]
sigmasM_tp = Parallel(n_jobs=-1)(delayed(loop_beta)(2.7, tp, BETARANGE, 'M')
                                 for tp in tp_r)
sigmasI_tp = Parallel(n_jobs=-1)(delayed(loop_beta)(2.7, tp, BETARANGE, 'I')
                                 for tp in tp_r)

f, (si, ds) = plt.subplots(2, 2, sharex=True)
plot_zero_w(sigmasI_tp, tp_r, BETARANGE, 0, 'INS tp=', si[0], ds[0])
plot_zero_w(sigmasM_tp, tp_r, BETARANGE, 0, 'MET tp=', si[0], ds[0])

plot_zero_w(sigmasI_tp, tp_r, BETARANGE, 1, 'INS tp=', si[1], ds[1])
plot_zero_w(sigmasM_tp, tp_r, BETARANGE, 1, 'MET tp=', si[1], ds[1])

si[0].set_ylabel(r'-$\Im \Sigma_{AA}(w=0)$')
ds[0].set_ylabel(r'-$\Im d\Sigma_{AA}(w=0)$')
si[1].set_ylabel(r'-$\Im \Sigma_{AB}(w=0)$')
ds[1].set_ylabel(r'-$\Im d\Sigma_{AB}(w=0)$')
ds[0].set_xlim([0, .06])

si[0].legend(loc=0, prop={'size': 8})
ds[0].set_xlabel('$T/D$')
ds[1].set_xlabel('$T/D$')
