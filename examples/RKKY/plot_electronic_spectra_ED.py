# -*- coding: utf-8 -*-
r"""
===================================
Dispersion of the spectral function
===================================

Extracting data from ED

"""
# Author: Óscar Nájera

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import matplotlib.pyplot as plt
import numpy as np
plt.matplotlib.rcParams.update({'axes.labelsize': 22,
                                'xtick.labelsize': 14, 'ytick.labelsize': 14,
                                'axes.titlesize': 22})

import dmft.common as gf
import dmft.RKKY_dimer as rt
import dmft.ipt_imag as ipt


def extract_data(dirname):
    sig11 = np.loadtxt(dirname + '/fort.511').T / 2
    sig12 = np.loadtxt(dirname + '/fort.512').T / 2
    sig_s = (sig11[1] + sig12[1]) + 1j * (sig11[2] + sig12[2])
    sig_a = (sig11[1] - sig12[1]) + 1j * (sig11[2] - sig12[2])
    w = sig11[0]
    return w, sig_s, sig_a


def calculate_Aw(u_int, eps_k, tp):
    w, ss, sa = extract_data('extract_U' + str(u_int))

    lat_gfs = 1 / np.add.outer(-eps_k, w + u_int / 4 + tp + 4e-5j - ss)
    lat_gfa = 1 / np.add.outer(-eps_k, w + u_int / 4 - tp + 4e-5j - sa)
    Aw = -.5 * (lat_gfa + lat_gfs).imag / np.pi

    return Aw, ss, sa, w


def hiltrans(zeta):
    sqr = np.sqrt(zeta**2 - 1)
    sqr = np.sign(sqr.imag) * sqr
    return 2 * (zeta - sqr)


def plot_spectra(tp, eps_k, axes):
    pdm, pam, pdi, pai = axes
    # metal
    u_int = 3.6
    Aw, ss, sa, w = calculate_Aw(u_int, eps_k, 0.6)
    x, y = np.meshgrid(eps_k, w)
    Aw = np.clip(Aw, 0, 1,)
    pdm.pcolormesh(x, y, Aw.T, cmap=plt.get_cmap(r'inferno'))
    gsts = hiltrans(w + u_int / 4 + tp + 4e-5j - ss)
    gsta = hiltrans(w + u_int / 4 - tp + 4e-5j - sa)
    gloc = 0.5 * (gsts + gsta)
    pam.plot(-gloc.imag / np.pi, w)

    # insulator
    u_int = 6.
    Aw, ss, sa, w = calculate_Aw(u_int, eps_k, 0.6)
    Aw = np.clip(Aw, 0, 1,)
    pdi.pcolormesh(x, y, Aw.T, cmap=plt.get_cmap(r'inferno'))
    gsts = hiltrans(w + u_int / 4 + tp + 4e-5j - ss)
    gsta = hiltrans(w + u_int / 4 - tp + 4e-5j - sa)
    gloc = 0.5 * (gsts + gsta)
    pai.plot(-gloc.imag / np.pi, w)


w = np.linspace(-3, 3, 800)
eps_k = np.linspace(-1., 1., 61)
w_set = np.arange(200)
fig, ax = plt.subplots(2, 2, gridspec_kw=dict(
    wspace=0.05, hspace=0.1, width_ratios=[3, 1]))
axes = ax.flatten()
plot_spectra(.3, eps_k, axes)
axes[0].set_ylim([-3, 3])
axes[0].set_yticks(np.linspace(-2.5, 2.5, 5))
axes[1].set_ylim([-3, 3])
axes[1].set_yticks(np.linspace(-2.5, 2.5, 5))
axes[1].set_yticklabels([])
axes[1].set_xticklabels([])
axes[2].set_ylim([-3, 3])
axes[2].set_yticks(np.linspace(-2.5, 2.5, 5))
axes[3].set_ylim([-3, 3])
axes[3].set_yticks(np.linspace(-2.5, 2.5, 5))
axes[3].set_yticklabels([])
axes[3].set_xticklabels([])
axes[0].set_ylabel(r'$\omega$')
axes[2].set_ylabel(r'$\omega$')
axes[2].set_xlabel(r'$\epsilon$')
axes[3].set_xlabel(r'$A(\omega)$')

#plt.savefig('arpes_coexistence.pdf', dpi=96, format='pdf', transparent=False, bbox_inches='tight', pad_inches=0.05)
