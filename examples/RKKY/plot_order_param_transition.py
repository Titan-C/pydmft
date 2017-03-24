# -*- coding: utf-8 -*-
r"""
The insulator transition order parameter
========================================

The low energy expansion is used to define an order parameter for the
Dimer-Mott transition.  Using the renormalized quantities for the
Half-filling Particle-hole Symmetric case.

.. math:: Z \equiv \left(1-\frac{\partial\Re e \Sigma_{11} (\omega)}{\partial \omega} \bigg |_{\omega=0}\right)^{-1} \\

.. math:: \tilde{t}_\perp \equiv  Z\left[t_\perp + \Re e \Sigma_{12}(0) \right]

The order parameter is defined as

.. math:: \eta = Z - \tilde{t}_\perp

"""
# Created Mon Mar  7 19:19:04 2016
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function


import os
from math import log, ceil
import numpy as np
import matplotlib.pyplot as plt
import dmft.common as gf
import dmft.dimer as dimer
import dmft.ipt_imag as ipt

plt.matplotlib.rcParams.update({'axes.labelsize': 22,
                                'xtick.labelsize': 14, 'ytick.labelsize': 14,
                                'axes.titlesize': 22, 'legend.fontsize': 14,
                                'mathtext.fontset': 'cm'})


def estimate_zero_w_sigma_U_vs_tp(tpr, u_range, beta, phase):
    sd_zew, so_zew = [], []
    tau, w_n = gf.tau_wn_setup(
        dict(BETA=beta, N_MATSUBARA=max(2**ceil(log(8 * beta) / log(2)), 256)))
    u_range = u_range if phase == 'met' else u_range[::-1]

    save_file = 'dimer_ipt_{}_Z_B{}.npy'.format(phase, beta)
    if os.path.exists(save_file):
        return np.load(save_file)

    for tp in tpr:
        filestr = 'disk/phase_Dimer_ipt_{}_B{}/tp{:.3}/giw.npy'.format(
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
    np.save(save_file, (sd_zew, so_zew))

    return sd_zew, so_zew


TPR = np.arange(0, 1.1, 0.02)
UR = np.arange(0, 4.5, 0.1)
f, ax = plt.subplots()
x, y = np.meshgrid(TPR, UR)

sd_zew, so_zew = estimate_zero_w_sigma_U_vs_tp(TPR, UR, 1000., 'met')
dw_sig11 = np.ma.masked_array(sd_zew[:, :, 0], sd_zew[:, :, 1] < -0.1)
zet = 1 / (1 - dw_sig11.T)
sig11_0 = np.ma.masked_array(so_zew[:, :, 1], sd_zew[:, :, 1] < -0.1)
tpp = (TPR + so_zew[:, :, 1].T)

order = zet - tpp * zet
#order = np.abs(zet - tpp * zet)
#order = np.ma.masked_array(order, order < -0.01)
cs = plt.contourf(x, y, order, 31)
plt.colorbar()
cs = plt.contour(x, y, order, 3, colors='k')
plt.clabel(cs, inline=1, fontsize=10, colors='k')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r'$U/D$')
plt.savefig('IPT_Uc2_orderparameter.png')

###############################################################################
# Insulator
# ---------

sd_zew, so_zew = estimate_zero_w_sigma_U_vs_tp(TPR, UR, 1000., 'ins')
dw_sig11 = np.ma.masked_array(sd_zew[:, :, 0], sd_zew[:, :, 1] < -0.1)
zet = 1 / (1 - dw_sig11.T)
sig11_0 = np.ma.masked_array(so_zew[:, :, 1], sd_zew[:, :, 1] < -0.1)
tpp = (TPR + so_zew[:, :, 1].T)

order = zet - tpp * zet
#order = np.abs(zet - tpp * zet)
#order = np.ma.masked_array(order, order < -0.01)
order = order[::-1]
plt.figure()
cs = plt.contourf(x, y, order, 31)
plt.colorbar()
cs = plt.contour(x, y, order, 7, colors='k')
plt.clabel(cs, inline=1, fontsize=10, colors='k')
plt.xlabel(r'$t_\perp/D$')
plt.ylabel(r'$U/D$')
plt.savefig('IPT_Uc2_orderparameter_ins.png')
