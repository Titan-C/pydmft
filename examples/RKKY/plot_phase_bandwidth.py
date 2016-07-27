# -*- coding: utf-8 -*-
r"""
===========================
Phase diagrams on Bandwidth
===========================

"""
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function
import os
from math import log, ceil
import numpy as np
import matplotlib.pyplot as plt
import dmft.common as gf
import dmft.RKKY_dimer as rt
import dmft.ipt_imag as ipt


def estimate_dos_at_fermi_level_U_vs_tp(tpr, dlist, beta, phase):
    dos_fl = []
    w_n = gf.matsubara_freq(beta, 3)
    save_file = 'dimer_ipt_{}_B{}.npy'.format(phase, beta)
    if os.path.exists(save_file):
        return -np.load(save_file).T

    for tp in tpr:
        filestr = 'disk/phase_Dimer_ipt_D_{}_B{}/tp{:.3}/giw.npy'.format(
            phase, beta, tp)
        gfs = np.load(filestr)
        dos_fl.append(np.array([gf.fit_gf(w_n, gfs[i][0][:3])(0.)
                                for i in dlist]))

    np.save(save_file, dos_fl)

    return -np.array(dos_fl).T


def plot_phase_diagram_U_vs_tp(tpr, ur, beta, ax):

    x, y = np.meshgrid(tpr, ur)
    ax.pcolormesh(x, y, estimate_dos_at_fermi_level_U_vs_tp(
        tpr, range(len(ur)), beta, 'met'), cmap=plt.get_cmap(r'viridis'), vmin=0, vmax=2)
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    ax.pcolormesh(x, y, estimate_dos_at_fermi_level_U_vs_tp(
        tpr, range(len(ur))[::-1], beta, 'ins'), alpha=0.2, cmap=plt.get_cmap(r'viridis'), vmin=0, vmax=2)


TPR = np.arange(0, 1.1, 0.02)
Drange = np.linspace(0.05, .85, 61)

for beta in [100.]:
    f, ax = plt.subplots()
    plot_phase_diagram_U_vs_tp(TPR, Drange, beta, ax)
    ax.set_xlabel(r'$t_\perp/U$')
    ax.set_ylabel(r'$D/U$')
    ax.set_title(
        'Phase diagram $\\beta={}$,\n color represents $-\\Im G_{{AA}}(0)$'.format(beta))
