# -*- coding: utf-8 -*-
r"""
======================================
Scattering rates change in Temperature
======================================

Explore the low energy expansion of the Matsubara self-energy. The
zero frequency value being the scattering rate.
Figure is discussed in reference [Najera2017]_


.. [Najera2017] O. Nájera, Civelli, M., V. Dobrosavljevic, & Rozenberg,
  M. J. (2017). Resolving the VO_2 controversy: Mott mechanism dominates
  the insulator-to-metal transition. Physical Review B, 95(3),
  035113. http://dx.doi.org/10.1103/physrevb.95.035113

"""

# Created Mon Mar  7 01:14:02 2016
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function

from math import log, ceil
import numpy as np
import matplotlib.pyplot as plt
import dmft.dimer as dimer
import dmft.common as gf
import dmft.ipt_imag as ipt


def loop_beta(u_int, tp, betarange, seed='ins'):
    """Solves IPT dimer and return Im Sigma_AA, Re Simga_AB

    returns list len(betarange) x 2 Sigma arrays """

    sigma_iw = []
    iterations = []
    for beta in betarange:
        tau, w_n = gf.tau_wn_setup(
            dict(BETA=beta, N_MATSUBARA=max(2**ceil(log(4 * beta) / log(2)), 256)))
        giw_d, giw_o = dimer.gf_met(w_n, 0., 0., 0.5, 0.)
        if seed == 'ins':
            giw_d, giw_o = 1 / (1j * w_n - 4j / w_n), np.zeros_like(w_n) + 0j

        giw_d, giw_o, loops = dimer.ipt_dmft_loop(
            beta, u_int, tp, giw_d, giw_o, tau, w_n, 1e-3)
        iterations.append(loops)
        g0iw_d, g0iw_o = dimer.self_consistency(
            1j * w_n, 1j * giw_d.imag, giw_o.real, 0., tp, 0.25)
        siw_d, siw_o = ipt.dimer_sigma(u_int, tp, g0iw_d, g0iw_o, tau, w_n)
        sigma_iw.append((siw_d.imag, siw_o.real))

    print(np.array(iterations))

    return sigma_iw

###############################################################################
# Calculate the dimer solution for Metals up to Uc2 in the in the given
# temperature range. Include the behavior of 2 insulators inside the
# coexistence region.


temp = np.arange(1 / 500., .05, .001)
BETARANGE = 1 / temp

U_intm = np.linspace(0, 3, 13)
sigmasM_U = [loop_beta(u_int, .3, BETARANGE, 'met') for u_int in U_intm]

U_inti = [2.5, 3.]
sigmasI_U = [loop_beta(u_int, .3, BETARANGE, 'ins') for u_int in U_inti]

###############################################################################
# Plot the zero frequency extrapolation of the self-energy. Top panel
# shows for all metals how upon cooling they become more coherent.
# Interestingly enough is the insulators is red showning even higher
# coherence(lower value of the scattering rate) than the metal. This is
# how the dimer Mott insulator fundamentally differs from the Single Site
# DMFT Mott insulator. When performing a correct analytical continuation
# of the Self-energy on the insulator it becomes evident that it is gaped
# at zero frequency.
#
# The lower panel shows the effective hybridization of the dimers. It is
# barely changed in the metal but is strongly boosted in the insulator.


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


fig, si = plt.subplots(2, 1, sharex=True)
sig_11_0i = plot_zero_w(sigmasI_U, U_inti, .3, BETARANGE, si, 'r--')
# fig, si = plt.subplots(2, 1, sharex=True)
sig_11_0m = plot_zero_w(sigmasM_U, U_intm, .3, BETARANGE, si, 'b')
si[0].set_ylim([0, 0.5])

ax = fig.add_axes([.2, .65, .2, .25])
ax.plot(U_intm, -sig_11_0m[:, -7])
ax.set_xticks(np.arange(0, 3.1, 1))
si[0].axvline(temp[-7], color='k')

###############################################################################
# Plot the derivative at zero frequency of the self-energy


def plot_der_zero_w(function_array, iter_range, betarange, entry, label_head, ax, dx):
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
        ax.plot(1 / BETARANGE, -np.array(dat)
                [:, 1], label=label_head + str(u))
        dx.plot(1 / BETARANGE, -np.array(dat)
                [:, 0], label=label_head + str(u))


f, (si, ds) = plt.subplots(2, 2, sharex=True)
plot_der_zero_w(sigmasI_U, U_inti, BETARANGE, 0, 'INS U=', si[0], ds[0])
plot_der_zero_w(sigmasM_U, U_intm, BETARANGE, 0, 'MET U=', si[0], ds[0])

plot_der_zero_w(sigmasI_U, U_inti, BETARANGE, 1, 'INS U=', si[1], ds[1])
plot_der_zero_w(sigmasM_U, U_intm, BETARANGE, 1, 'MET U=', si[1], ds[1])

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
sigmasM_tp = [loop_beta(2.7, tp, BETARANGE, 'M') for tp in tp_r]
sigmasI_tp = [loop_beta(2.7, tp, BETARANGE, 'I') for tp in tp_r]

f, (si, ds) = plt.subplots(2, 2, sharex=True)
plot_der_zero_w(sigmasI_tp, tp_r, BETARANGE, 0, 'INS tp=', si[0], ds[0])
plot_der_zero_w(sigmasM_tp, tp_r, BETARANGE, 0, 'MET tp=', si[0], ds[0])

plot_der_zero_w(sigmasI_tp, tp_r, BETARANGE, 1, 'INS tp=', si[1], ds[1])
plot_der_zero_w(sigmasM_tp, tp_r, BETARANGE, 1, 'MET tp=', si[1], ds[1])

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
plot_der_zero_w([sigmasI_U[1]], [3.], BETARANGE, 0, 'INS', si, ds)
plot_der_zero_w([sigmasM_U[-1]], [3.], BETARANGE, 0, 'MET', si, ds)
si.set_title(r'$\Im m\Sigma_{Aa}$ cut and slope tp=0.3, U=3')
si.set_ylabel(r'-$\Im \Sigma_{AA}(w=0)$')
ds.set_ylabel(r'-$\Im d\Sigma_{AA}(w=0)$')
ds.set_xlim([0, .06])
si.legend(loc=0, prop={'size': 10})
ds.set_xlabel('$T/D$')

f, (si, ds) = plt.subplots(2, 1, sharex=True)
plot_der_zero_w([sigmasI_U[1]], [3.], BETARANGE, 1, 'INS', si, ds)
plot_der_zero_w([sigmasM_U[-1]], [3.], BETARANGE, 1, 'MET', si, ds)
si.set_title(r'$\Re e\Sigma_{AB}$ cut and slope tp=0.3, U=3')
si.set_ylabel(r'-$\Re e \Sigma_{AB}(w=0)$')
ds.set_ylabel(r'-$\Re e d\Sigma_{AB}(w=0)$')
ds.set_xlim([0, .06])

si.legend(loc=0, prop={'size': 10})
ds.set_xlabel('$T/D$')
