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


def estimate_dos_at_fermi_level(tpr, ulist, beta, phase):
    dos_fl = []
    w_n = gf.matsubara_freq(100., 3)
    save_file = 'dimer_ipt_{}_B{}.npy'.format(phase, beta)
    if os.path.exists(save_file):
        return -np.load(save_file).T

    for tp in tpr:
        filestr = 'disk/phase_Dimer_ipt_{}_B{}/tp{:.3}/giw.npy'.format(
            phase, beta, tp)
        gfs = np.load(filestr)
        dos_fl.append(np.array([gf.fit_gf(w_n, gfs[i][0][:3].imag)(0.)
                                for i in ulist]))

    np.save(save_file, dos_fl)

    return -np.array(dos_fl).T


def plot_phase_diagram(tpr, ur, beta):

    x, y = np.meshgrid(tpr, ur)
    plt.pcolormesh(x, y, estimate_dos_at_fermi_level(
        tpr, range(len(ur)), beta, 'met'), cmap=plt.get_cmap(r'viridis'))
    plt.axis([x.min(), x.max(), y.min(), y.max()])
    plt.colorbar()
    plt.pcolormesh(x, y, estimate_dos_at_fermi_level(
        tpr, range(len(ur))[::-1], beta, 'ins'), alpha=0.2, cmap=plt.get_cmap(r'viridis'))

    plt.xlabel(r'$t_\perp$')
    plt.ylabel(r'$U/D$')
    plt.title(
        'Phase diagram $\\beta={}$,\n color represents $-\\Im G_{{AA}}(0)$'.format(beta))


TPR = np.arange(0, 1.1, 0.02)
UR = np.arange(0, 4.5, 0.1)

plt.figure()
plot_phase_diagram(TPR, UR, 1000.)
plt.figure()
plot_phase_diagram(TPR, UR, 100.)
plt.show()
