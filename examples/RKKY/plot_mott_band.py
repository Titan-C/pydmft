# -*- coding: utf-8 -*-
r"""
The Mott and Band insulator Character
=====================================

Adding the order parameter plus the half-gap
"""
from __future__ import division, absolute_import, print_function

from math import ceil, log
import os

import numpy as np
import matplotlib.pyplot as plt

import dmft.common as gf
import dmft.dimer as dimer

plt.matplotlib.rcParams.update({'axes.labelsize': 22,
                                'xtick.labelsize': 14, 'ytick.labelsize': 14,
                                'axes.titlesize': 22,
                                'mathtext.fontset': 'cm'})


def measure_gap(gloc, rw):
    gapped = gloc.imag[rw] > -0.015
    try:
        lb = w[rw][gapped][0]
        ub = w[rw][gapped][-1]
    except IndexError:
        return 0

    if gloc.imag[int(len(w) / 2)] < -0.3:
        return 0

    if lb is not None and ub is not None:
        gap = ub - lb
    else:
        gap = 0
    return gap


def estimate_gap_U_vs_tp(tpr, u_range, beta, phase):
    w_n = gf.matsubara_freq(beta, max(2**ceil(log(6 * beta) / log(2)), 256))

    gaps = []
    for tp in tpr:
        filestr = '/home/oscar/dev/dmft-learn/examples/RKKY/disk/phase_Dimer_ipt_{}_B{}/tp{:.3}/giw.npy'.format(
            phase, beta, tp)
        gfs = np.load(filestr)

        for i, u_int in enumerate(u_range):
            gf_aa, gf_ab = 1j * gfs[i][0], gfs[i][1]
            gr_ss, gr_sa = dimer.pade_diag(
                gf_aa, gf_ab, w_n, np.arange(0, beta + 100, 9, dtype=np.int), w)
            gloc = (gr_ss + gr_sa) / 2

            gaps.append(measure_gap(gloc, rw))
            #plt.plot(w, -gloc.imag + i * 0.1)
            #plt.plot(gaps[-1] / 2, i * 0.1, 'o')

    gaps = np.array(gaps).reshape(len(tpr), len(u_range)).T

    return gaps


def estimate_zero_w_sigma_U_vs_tp(tpr, u_range, beta, phase):
    sd_zew, so_zew = [], []
    tau, w_n = gf.tau_wn_setup(
        dict(BETA=beta, N_MATSUBARA=max(2**ceil(log(8 * beta) / log(2)), 256)))

    for tp in tpr:
        filestr = '/home/oscar/dev/dmft-learn/examples/RKKY/disk/phase_Dimer_ipt_{}_B{}/tp{:.3}/giw.npy'.format(
            phase, beta, tp)
        gfs = np.load(filestr)

        for i, u_int in enumerate(u_range):
            giw_d, giw_o = 1j * gfs[i][0], gfs[i][1]
            g0iw_d, g0iw_o = dimer.self_consistency(
                1j * w_n, giw_d, giw_o, 0., tp, 0.25)
            siw_d, siw_o = ipt.dimer_sigma(u_int, tp, g0iw_d, g0iw_o, tau, w_n)
            sd_zew.append(np.polyfit(w_n[:2], siw_d[:2].imag, 1))
            so_zew.append(np.polyfit(w_n[:2], siw_o[:2].real, 1))

    sd_zew = np.array(sd_zew).reshape(len(tpr), len(u_range), -1)
    so_zew = np.array(so_zew).reshape(len(tpr), len(u_range), -1)

    return sd_zew, so_zew

TPR = np.arange(0, 1.11, 0.05)
#TPR = [1.05]
UR = np.arange(0, 4.5, 0.1)[::-1]
x, y = np.meshgrid(TPR, UR)

sd_zew, so_zew = estimate_zero_w_sigma_U_vs_tp(TPR, UR, 1000., 'ins')
dw_sig11 = np.ma.masked_array(sd_zew[:, :, 0], sd_zew[:, :, 1] < -0.1)
zet = 1 / (1 - dw_sig11.T)
sig11_0 = np.ma.masked_array(so_zew[:, :, 1], sd_zew[:, :, 1] < -0.1)
tpp = (TPR + so_zew[:, :, 1].T)

order = zet - tpp * zet
#order = np.abs(zet - tpp * zet)
order = np.ma.masked_array(order, order > 0.01)

w = np.linspace(-5, 5, 2**10)
rw = np.abs(w) < 1.5
dw = w[1] - w[0]

#gaps = estimate_gap_U_vs_tp(TPR, UR, 1000., 'ins') / 2

plt.figure()
#gaps = np.ma.masked_array(gaps, gaps <= 0)
eta_plus_gap = order + gaps
plt.pcolormesh(x, y, eta_plus_gap)
cs = plt.contour(x, y, eta_plus_gap, 15, colors='k')
plt.clabel(cs, inline=1, fontsize=10, colors='k')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r'$U/D$')
plt.grid()
plt.xlim(0, 1.0801)
plt.savefig('IPT_eta_plus_half_gap.png')
