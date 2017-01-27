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
import dmft.dimer as dimer
import dmft.ipt_imag as ipt


def dos_at_fermi_level_fixbeta(xlist, ylist, beta, phasename, datapath):
    dos_fl = []
    w_n = gf.matsubara_freq(beta, 3)
    save_file = 'phase_dimer_ipt_{}.npy'.format(phasename)
    if os.path.exists(save_file):
        dos_fl = np.load(save_file)
        return dos_fl

    for xdat in xlist:
        filestr = datapath.format(phasename, xdat)
        gfs = np.load(filestr)
        dos_fl.append(np.array([gf.fit_gf(w_n, gfs[i][0][:3])(0.)
                                for i in ylist]))

    dos_fl = -np.array(dos_fl).T
    np.save(save_file, dos_fl)

    return dos_fl


def plot_phase_diagram_fixbeta(xlist, ylist, phasename, beta, ax, alpha=1):

    x, y = np.meshgrid(xlist, ylist)
    yrange = list(range(len(ylist)))
    if alpha < 1:
        yrange = yrange[::-1]

    ax.pcolormesh(x, y,
                  dos_at_fermi_level_fixbeta(xlist, yrange, beta, phasename,
                                             'disk/phase_Dimer_ipt_{}/tp{:.3}/giw.npy'),
                  cmap=plt.get_cmap(r'viridis'), alpha=alpha, vmin=0, vmax=3)
    ax.axis([x.min(), x.max(), y.min(), y.max()])


TPR = np.arange(0, 1.1, 0.02)
Drange = np.linspace(0.05, 1.1, 81)

for beta in [1000., 100., 30.]:
    f, ax = plt.subplots()
    plot_phase_diagram_fixbeta(
        TPR, Drange, 'D_met_B{}'.format(beta), beta, ax, 1)
    plot_phase_diagram_fixbeta(
        TPR, Drange, 'D_ins_B{}'.format(beta), beta, ax, .21)
    ax.set_xlabel(r'$t_\perp/U$')
    ax.set_ylabel(r'$D/U$')
    ax.set_title(
        'Phase diagram $\\beta={}$,\n color represents $-\\Im G_{{AA}}(0)$'.format(beta))
plt.show()

###############################################################################


def dos_at_fermi_level_temp(xlist, temp, phasename, datapath):
    dos_fl = []
    save_file = 'phase_dimer_ipt_{}.npy'.format(phasename)
    if os.path.exists(save_file):
        dos_fl = np.load(save_file)
        return dos_fl

    for T in temp:
        w_n = gf.matsubara_freq(1 / T, 3)
        filestr = datapath.format(phasename, 1 / T)
        gfs = np.load(filestr)
        dos_fl.append(np.array([gf.fit_gf(w_n, gfs[i][0][:3])(0.)
                                for i in xlist]))

    dos_fl = -np.array(dos_fl)
    np.save(save_file, dos_fl)

    return dos_fl


def plot_phase_diagram_temp(xlist, temp, phasename, ax, alpha=1):

    x, y = np.meshgrid(xlist, temp)
    x_range = list(range(len(xlist)))
    if alpha < 1:
        x_range = x_range[::-1]

    ax.pcolormesh(x, y, dos_at_fermi_level_temp(x_range, temp, phasename,
                                                'disk/phase_Dimer_ipt_{}/B{:.5}/giw.npy'),
                  cmap=plt.get_cmap(r'viridis'), alpha=alpha, vmin=0, vmax=3)
    ax.axis([x.min(), x.max(), y.min(), y.max()])

TEMP = np.arange(1 / 512., .08, 1 / 400)
for tp in [.15, .3, .5]:
    f, ax = plt.subplots()
    plot_phase_diagram_temp(Drange, TEMP, 'D_met_tp{}'.format(tp), ax, 1)
    plot_phase_diagram_temp(Drange, TEMP, 'D_ins_tp{}'.format(tp), ax, 0.2)
    ax.set_ylabel(r'$T/U$')
    ax.set_xlabel(r'$D/U$')
    ax.set_title(
        'Phase diagram $t_\\perp={}$,\n color represents $-\\Im G_{{AA}}(0)$'.format(tp))

plt.show()
