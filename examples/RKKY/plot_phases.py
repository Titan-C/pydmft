# -*- coding: utf-8 -*-
r"""
==============
Phase diagrams
==============

"""
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import dmft.common as gf


def estimate_dos_at_fermi_level_U_vs_tp(tpr, ulist, beta, phase):
    dos_fl = []
    w_n = gf.matsubara_freq(beta, 3)
    save_file = 'dimer_ipt_{}_B{}.npy'.format(phase, beta)
    if os.path.exists(save_file):
        return -np.load(save_file).T

    for tp in tpr:
        filestr = 'disk/phase_Dimer_ipt_{}_B{}/tp{:.3}/giw.npy'.format(
            phase, beta, tp)
        gfs = np.load(filestr)
        dos_fl.append(np.array([gf.fit_gf(w_n, gfs[i][0][:3])(0.)
                                for i in ulist]))

    np.save(save_file, dos_fl)

    return -np.array(dos_fl).T


def plot_phase_diagram_U_vs_tp(tpr, ur, beta):

    x, y = np.meshgrid(tpr, ur)
    plt.pcolormesh(x, y, estimate_dos_at_fermi_level_U_vs_tp(
        tpr, range(len(ur)), beta, 'met'), cmap=plt.get_cmap(r'viridis'), vmin=0, vmax=2)
    plt.axis([x.min(), x.max(), y.min(), y.max()])
    plt.colorbar()
    plt.pcolormesh(x, y, estimate_dos_at_fermi_level_U_vs_tp(
        tpr, range(len(ur))[::-1], beta, 'ins'), alpha=0.2, cmap=plt.get_cmap(r'viridis'), vmin=0, vmax=2)

    plt.xlabel(r'$t_\perp/D$')
    plt.ylabel(r'$U/D$')
    plt.title(
        'Phase diagram $\\beta={}$,\n color represents $-\\Im G_{{AA}}(0)$'.format(beta))


TPR = np.arange(0, 1.1, 0.02)
UR = np.arange(0, 4.5, 0.1)
for beta in [100., 1000.]:
    plt.figure()
    plot_phase_diagram_U_vs_tp(TPR, UR, beta)

###############################################################################


def estimate_dos_at_fermi_level_T_vs_U(tp, ulist, temp, phase):
    dos_fl = []
    save_file = 'dimer_ipt_{}_tp{:.2}.npy'.format(phase, tp)
    if os.path.exists(save_file):
        return -np.load(save_file)

    for T in temp:
        w_n = gf.matsubara_freq(100., 3)
        filestr = 'disk/phase_Dimer_ipt_{}_tp{}/B{:.5}/giw.npy'.format(
            phase, tp, 1 / T)
        gfs = np.load(filestr)
        dos_fl.append(np.array([gf.fit_gf(w_n, gfs[i][0][:3])(0.)
                                for i in ulist]))

    np.save(save_file, dos_fl)

    return -np.array(dos_fl)


def plot_phase_diagram_T_vs_U(tp, ur, temp):

    x, y = np.meshgrid(ur, temp)
    plt.pcolormesh(x, y, estimate_dos_at_fermi_level_T_vs_U(
        tp, range(len(ur)), temp, 'met'), cmap=plt.get_cmap(r'viridis'), vmin=0, vmax=2)
    plt.axis([x.min(), x.max(), y.min(), y.max()])
    plt.colorbar()
    plt.pcolormesh(x, y, estimate_dos_at_fermi_level_T_vs_U(
        tp, range(len(ur))[::-1], temp, 'ins'), alpha=0.2, cmap=plt.get_cmap(r'viridis'), vmin=0, vmax=2)

    plt.xlabel(r'$U/D$')
    plt.ylabel(r'$T/D$')
    plt.title(
        'Phase diagram $t_\\perp={}$,\n color represents $-\\Im G_{{AA}}(0)$'.format(tp))

TEMP = np.arange(1 / 500., .14, 1 / 400)
for tp in [0., .15, .3, .5]:
    plt.figure()
    plot_phase_diagram_T_vs_U(tp, UR, TEMP)
plt.show()
