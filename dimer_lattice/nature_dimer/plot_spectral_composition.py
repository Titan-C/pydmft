# -*- coding: utf-8 -*-
r"""
Spectral composition
====================

The energy resolved spectral function is plotted to view the quality of
the Fermi Liquid approximation

"""

from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev

import dmft.common as gf
import dmft.ipt_imag as ipt
import dmft.dimer as dimer


def ipt_u_tp(u_int, tp, beta, seed='ins'):

    tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=2**11))
    giw_d, giw_o = dimer.gf_met(w_n, 0., 0., 0.5, 0.)

    if 'ins' in seed:
        giw_d, giw_o = 1 / (1j * w_n + 4j / w_n), np.zeros_like(w_n) + 0j

    giw_d, giw_o, loops = dimer.ipt_dmft_loop(
        beta, u_int, tp, giw_d, giw_o, tau, w_n, 1e-12)
    g0iw_d, g0iw_o = dimer.self_consistency(
        1j * w_n, 1j * giw_d.imag, giw_o.real, 0., tp, 0.25)
    siw_d, siw_o = ipt.dimer_sigma(u_int, tp, g0iw_d, g0iw_o, tau, w_n)

    return giw_d, giw_o, siw_d, siw_o, w_n


def ipt_g_s(u_int, tp, BETA, seed, w):
    giw_d, giw_o, siw_d, siw_o, w_n = ipt_u_tp(u_int, tp, BETA, seed)

    w_set = np.arange(0, 541, 4)
    ss = gf.pade_continuation(
        1j * siw_d.imag + siw_o.real, w_n, w + 0.0005j, w_set)  # A-bond

    gst = gf.semi_circle_hiltrans(w - tp - (ss.real - 1j * np.abs(ss.imag)))
    return gst, ss, w


def low_en_qp(ss):
    glp = np.array([0.])
    sigtck = splrep(w, ss.real, s=0)
    sig_0 = splev(glp, sigtck, der=0)[0]
    dw_sig0 = splev(glp, sigtck, der=1)[0]
    quas_z = 1 / (1 - dw_sig0)
    return quas_z, sig_0, dw_sig0


def plot_spectral(omega, tp, gss, ss, axes, xlim):

    (axsg, axgf, axsd) = axes
    quas_z, sig_0, dw_sig0 = low_en_qp(ss)
    tpp = (tp + sig_0) * quas_z
    axgf.plot(-gss.imag / np.pi, omega, 'C0')
    llg = gf.semi_circle_hiltrans(omega + 1e-8j - tpp, quas_z) * quas_z
    axgf.plot(-llg.imag / np.pi, omega, "C3--", lw=2)
    axgf.invert_xaxis()
    axgf.set_xlabel(r'$A_{AB}(\omega)$')
    plt.xticks(rotation=40)

    # plt.plot(omega, gst.real)
    axsg.plot(ss.real, omega, 'C4', label=r'$\Re e$')
    axsg.plot(ss.imag, omega, 'C2', label=r'$\Im m$')
    axsg.plot(sig_0 + dw_sig0 * omega, omega, 'k:')
    axsg.legend(loc=2)
    axsg.invert_xaxis()

    axsg.set_ylabel(r'$\omega$')
    axsg.set_xlabel(r'$\Sigma_{AB}(\omega)$')
    axsg.set_xlim(*xlim)

    eps_k = np.linspace(-1., 1., 61)
    lat_gfs = 1 / np.add.outer(-eps_k, omega - tp - ss + 0.007j)
    Aw = -lat_gfs.imag / np.pi
    x, y = np.meshgrid(eps_k, omega)
    axsd.pcolormesh(x, y, Aw.T, cmap=plt.get_cmap(r'viridis'), vmin=0, vmax=2)
    axsd.plot(eps_k, eps_k * quas_z + tpp, "C3--")
    axsd.set_ylim([-2.3,  2.3])

    axsd.set_xlabel(r'$\epsilon$')


BETA = 512.

w = np.linspace(-4, 4, 2**12)
###############################################################################
# Along :math:`U_{c_1}`
# ---------------------

for i, (U, tp, xlim) in enumerate([
    (2.2, 0.3, (-4, 3)),
    (1.67, 0.55, (-1.52, 1)),
    (1.42, 0.8, (-1, 0.5)),
]):

    fig, ax = plt.subplots(1, 3, sharey=True, gridspec_kw=dict(
        wspace=0.05, hspace=0.1, width_ratios=[1, 1, 2.4]))
    gss, ss, w = ipt_g_s(U, tp, BETA, 'ins', w)
    plot_spectral(w, tp, gss, ss, ax, xlim)
    ax[2].set_title(r'$U/D={}$ ; $t_\perp/D={}$'.format(U, tp))

###############################################################################
# Along the Mott insulator
# ------------------------
for i, (U, tp, xlim) in enumerate([
    (2.3, 0.3, (-4, 2.7)),
    (2.3, 0.48, (-4, 2.7)),
        (2.3, 0.85, (-2.2, 2.2))]):

    fig, ax = plt.subplots(1, 3, sharey=True, gridspec_kw=dict(
        wspace=0.05, hspace=0.1, width_ratios=[1, 1, 2.4]))
    gss, ss, w = ipt_g_s(U, tp, BETA, 'ins', w)
    plot_spectral(w, tp, gss, ss, ax, xlim)
    ax[2].set_title(r'$U/D={}$ ; $t_\perp/D={}$'.format(U, tp))
