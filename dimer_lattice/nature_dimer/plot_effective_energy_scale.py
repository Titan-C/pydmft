# -*- coding: utf-8 -*-
r"""
Effective energy scale
======================

This describes the renormalized energy scales of the metallic dimer
"""
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev

import dmft.common as gf
from dmft import ipt_real
from dmft import ipt_imag
import dmft.dimer as dimer


def low_en_qp(ss):
    glp = np.array([0.])
    sigtck = splrep(w, ss.real, s=0)
    sig_0 = splev(glp, sigtck, der=0)[0]
    dw_sig0 = splev(glp, sigtck, der=1)[0]
    quas_z = 1 / (1 - dw_sig0)
    return quas_z, sig_0, dw_sig0


w = np.linspace(-4, 4, 2**12)
dw = w[1] - w[0]

BETA = 512.
nfp = gf.fermi_dist(w, BETA)


def plot_low_energy(u_int, tp):
    gss = gf.semi_circle_hiltrans(w + 5e-3j)
    gsa = gf.semi_circle_hiltrans(w + 5e-3j)

    (gss, _), (ss, _) = ipt_real.dimer_dmft(
        u_int, tp, nfp, w, dw, gss, gsa, conv=1e-3)

    quas_z, sig_0, dw_sig0 = low_en_qp(ss)
    tpp = (tp + sig_0) * quas_z
    ax = plt.subplot(111)
    ax.plot(w, -gss.imag, 'C0')
    llg = gf.semi_circle_hiltrans(w + 1e-8j - tpp, quas_z) * quas_z
    ax.plot(w, -llg.imag, "C3--", lw=2)
    plt.title(r'$U={}$; $t_\perp={}$'.format(u_int, tp), fontsize=14)

    plt.ylabel(r'$-\Im m G_{AB}(\omega)$')
    plt.xlim(-2.3, 2.3)
    plt.ylim(0, 2.5)
    plt.xlabel(r'$\omega$')
    plt.arrow(0, 2, tpp, 0, shape='right',
              width=0.05, length_includes_head=True,
              head_width=0.1, head_length=0.1)
    plt.text(0, 2.1, r"$\tilde{t}_\perp$", color="C0", size=26)
    plt.arrow(tpp, 0, -quas_z, 0, shape='left',
              width=0.05, length_includes_head=True,
              head_width=0.1, head_length=0.1)
    plt.arrow(tpp, 0, quas_z, 0, shape='right',
              width=0.05, length_includes_head=True,
              head_width=0.1, head_length=0.1)
    plt.text(tpp, 0.1, r"$2\tilde{D}$", color="C0", size=26)


###############################################################################
plot_low_energy(2, 0.3)
# plt.savefig("low_en_qp_U2_tp0.3.pdf")
###############################################################################
plot_low_energy(2, 0.5)
###############################################################################
plot_low_energy(1.4, 0.8)
