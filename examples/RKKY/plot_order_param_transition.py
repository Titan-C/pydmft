# -*- coding: utf-8 -*-
r"""
The insulator transition order parameter
========================================

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


def plot_z_diagram_U_vs_tp(tpr, ur, beta, ax):

    x, y = np.meshgrid(tpr, ur)

    sd_zew, so_zew = estimate_zero_w_sigma_U_vs_tp(tpr, ur, beta, 'met')
    z = np.ma.masked_array(sd_zew[:, :, 0], sd_zew[:, :, 1] < -0.1)
    ax.pcolormesh(x, y, 1 / (1 - z.T), cmap=plt.get_cmap(r'viridis'))

    sd_zew, so_zew = estimate_zero_w_sigma_U_vs_tp(tpr, ur, beta, 'ins')
    z = np.ma.masked_array(sd_zew[:, ::-1, 0], sd_zew[:, ::-1, 1] < -0.1)
    ax.pcolormesh(x, y, 1 / (1 - z.T),
                  cmap=plt.get_cmap(r'viridis'), alpha=0.2)
    ax.axis([x.min(), x.max(), y.min(), y.max()])

TPR = np.arange(0, 1.1, 0.02)
UR = np.arange(0, 4.5, 0.1)
f, ax = plt.subplots()
plot_z_diagram_U_vs_tp(TPR, UR, 1000., ax)
