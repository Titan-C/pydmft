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
plt.matplotlib.rcParams.update({'axes.labelsize': 22,
                                'xtick.labelsize': 14, 'ytick.labelsize': 14,
                                'axes.titlesize': 22})


def loop_beta(u_int, tp, betarange, seed='ins'):
    """Solves IPT dimer and return Im Sigma_AA, Re Simga_AB

    returns list len(betarange) x 2 Sigma arrays
"""

    sigma_iw = []
    iterations = []
    for beta in betarange:
        tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=2**12))

        giw_d, giw_o = rt.gf_met(w_n, 0., 0., 0.5, 0.)
        if seed == 'ins':
            giw_d, giw_o = 1 / (1j * w_n - 4j / w_n), np.zeros_like(w_n) + 0j

        giw_d, giw_o, loops = rt.ipt_dmft_loop(
            beta, u_int, tp, giw_d, giw_o, tau, w_n, 1e-5)
        iterations.append(loops)
        g0iw_d, g0iw_o = rt.self_consistency(
            1j * w_n, 1j * giw_d.imag, giw_o.real, 0., tp, 0.25)
        siw_d, siw_o = ipt.dimer_sigma(u_int, tp, g0iw_d, g0iw_o, tau, w_n)
        sigma_iw.append((siw_d.imag, siw_o.real))

    print(np.array(iterations))

    return sigma_iw

temp = np.arange(1 / 500., .05, .001)
BETARANGE = 1 / temp

###############################################################################
# Calculate in the METAL dome the metal and insulator solution fix tp


# Plot a 2x2 frame following sigma and d_sigma Changing U


def plot_zero_w(function_array, iter_range, tp, betarange, ax, color):
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

    sig_11_0 = []
    rtp = []
    for j, u in enumerate(iter_range):
        sig_11_0.append([])
        rtp.append([])
        for i, beta in list(enumerate(betarange)):
            tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=2))
            sig_11_0[j].append(np.polyfit(
                w_n, function_array[j][i][0][:2], 1)[1])
            rtp[j].append(np.polyfit(w_n, function_array[j][i][1][:2], 1)[1])

        ax[0].plot(1 / BETARANGE, -np.array(sig_11_0[j]), color, label=str(u))
        ax[1].plot(1 / BETARANGE, tp + np.array(rtp[j]), color, label=str(u))
    ax[0].set_ylabel(r'$-\Im m \Sigma_{11}(w=0)$')
    ax[1].set_ylabel(r'$t_\perp + \Re e\Sigma_{12}(w=0)$')
    ax[1].set_xlabel('$T/D$')
    ax[1].set_xlim([min(temp), max(temp)])
    return np.array(sig_11_0)

U_inti = [2.5, 3.]
sigmasI_U = Parallel(n_jobs=-1)(delayed(loop_beta)(u_int, .3, BETARANGE, 'ins')
                                for u_int in U_inti)
U_intm = np.linspace(0, 3, 13)
sigmasM_U = Parallel(n_jobs=-1)(delayed(loop_beta)
                                (u_int, .3, BETARANGE, 'met') for u_int in U_intm)

fig, si = plt.subplots(2, 1, sharex=True)
sig_11_0i = plot_zero_w(sigmasI_U, U_inti, .3, BETARANGE, si, 'r--')
#fig, si = plt.subplots(2, 1, sharex=True)
sig_11_0m = plot_zero_w(sigmasM_U, U_intm, .3, BETARANGE, si, 'b')
si[0].set_ylim([0, 0.5])

ax = fig.add_axes([.2, .65, .2, .25])
ax.plot(U_intm, -sig_11_0m[:, -7])
ax.set_xticks(np.arange(0, 3.1, 1))
si[0].axvline(temp[-7], color='k')


# si[0].legend(loc=0)


###############################################################################


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
si[1].set_ylabel(r'-$\Re \Sigma_{AB}(w=0)$')
ds[1].set_ylabel(r'-$\Re d\Sigma_{AB}(w=0)$')
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
si[1].set_ylabel(r'-$\Re \Sigma_{AB}(w=0)$')
ds[1].set_ylabel(r'-$\Re d\Sigma_{AB}(w=0)$')
ds[0].set_xlim([0, .06])

si[0].legend(loc=0, prop={'size': 8})
ds[0].set_xlabel('$T/D$')
ds[1].set_xlabel('$T/D$')


###############################################################################
# Fine graphics
f, (si, ds) = plt.subplots(2, 1, sharex=True)
plot_zero_w([sigmasI_U[1]], [3.], BETARANGE, 0, 'INS', si, ds)
plot_zero_w([sigmasM_U[1]], [3.], BETARANGE, 0, 'MET', si, ds)
si.set_title(r'$\Im m\Sigma_{Aa}$ cut and slope tp=0.3, U=3')
si.set_ylabel(r'-$\Im \Sigma_{AA}(w=0)$')
ds.set_ylabel(r'-$\Im d\Sigma_{AA}(w=0)$')
ds.set_xlim([0, .06])
si.legend(loc=0, prop={'size': 10})
ds.set_xlabel('$T/D$')

f, (si, ds) = plt.subplots(2, 1, sharex=True)
plot_zero_w([sigmasI_U[1]], [3.], BETARANGE, 1, 'INS', si, ds)
plot_zero_w([sigmasM_U[1]], [3.], BETARANGE, 1, 'MET', si, ds)
si.set_title(r'$\Re e\Sigma_{AB}$ cut and slope tp=0.3, U=3')
si.set_ylabel(r'-$\Re e \Sigma_{AB}(w=0)$')
ds.set_ylabel(r'-$\Re e d\Sigma_{AB}(w=0)$')
ds.set_xlim([0, .06])

si.legend(loc=0, prop={'size': 10})
ds.set_xlabel('$T/D$')


###############################################################################
# Study a fix Beta

def loop_u_tp(u_range, tp_range, beta, seed='mott gap'):
    """Solves IPT dimer and return Im Sigma_AA, Re Simga_AB

    returns list len(betarange) x 2 Sigma arrays
"""
    tau, w_n = gf.tau_wn_setup(
        dict(BETA=beta, N_MATSUBARA=max(2**ceil(log(4 * beta) / log(2)), 256)))
    giw_d, giw_o = rt.gf_met(w_n, 0., 0., 0.5, 0.)
    if seed == 'I':
        giw_d, giw_o = 1 / (1j * w_n - 4j / w_n), np.zeros_like(w_n) + 0j

    sigma_iw = []
    iterations = []
    for tp, u_int in zip(tp_range, u_range):
        giw_d, giw_o, loops = rt.ipt_dmft_loop(
            beta, u_int, tp, giw_d, giw_o, tau, w_n, 1e-8)
        iterations.append(loops)
        g0iw_d, g0iw_o = rt.self_consistency(
            1j * w_n, 1j * giw_d.imag, giw_o.real, 0., tp, 0.25)
        siw_d, siw_o = ipt.dimer_sigma(u_int, tp, g0iw_d, g0iw_o, tau, w_n)
        sigma_iw.append((siw_d.imag, siw_o.real))

    print(np.array(iterations))

    return sigma_iw

###############################################################################


def plot_zero_w(function_array, iter_range, beta, entry, label_head, ax, dx):
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
    tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=20))

    dat = []
    for j, u in enumerate(iter_range):
        dat.append(np.polyfit(w_n[:2], function_array[j][entry][:2], 1))
    ax.plot(iter_range, -np.array(dat)[:, 1], label=label_head)
    dx.plot(iter_range, -np.array(dat)[:, 0], label=label_head)

U_int = np.arange(.5, 4.5, 0.1)
sigmasM_Ur = loop_u_tp(U_int, .3 * np.ones_like(U_int), 200., 'M')
sigmasI_Ur = loop_u_tp(U_int[::-1], .3 * np.ones_like(U_int), 200., 'I')[::-1]

f, (si, ds) = plt.subplots(2, 2, sharex=True)
plot_zero_w(sigmasI_Ur, U_int, 100., 0, 'INS', si[0], ds[0])
plot_zero_w(sigmasM_Ur, U_int, 100., 0, 'MET', si[0], ds[0])
plot_zero_w(sigmasI_Ur, U_int, 100., 1, 'INS', si[1], ds[1])
plot_zero_w(sigmasM_Ur, U_int, 100., 1, 'MET', si[1], ds[1])

si[0].legend(loc=0, prop={'size': 8})

si[0].set_ylabel(r'-$\Im \Sigma_{AA}(w=0)$')
ds[0].set_ylabel(r'-$\Im d\Sigma_{AA}(w=0)$')
si[1].set_ylabel(r'-$\Re \Sigma_{AB}(w=0)$')
ds[1].set_ylabel(r'-$\Re d\Sigma_{AB}(w=0)$')

ds[0].set_xlabel('$U/D$')
ds[1].set_xlabel('$U/D$')
