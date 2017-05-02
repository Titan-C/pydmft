# -*- coding: utf-8 -*-
r"""
Comparison of Spectral functions
================================

Here I compare the spectral function parametrizations to the Fermi
Liquid theory at zero temperature for relevant points in the dimer
phase diagram.
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


def ipt_u_tp(u_int, tp, beta, seed='ins'):

    tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=2**11))
    giw_d, giw_o = dimer.gf_met(w_n, 0., 0., 0.5, 0.)

    if 'ins' in seed:
        giw_d, giw_o = 1 / (1j * w_n + 4j / w_n), np.zeros_like(w_n) + 0j

    giw_d, giw_o, loops = dimer.ipt_dmft_loop(
        beta, u_int, tp, giw_d, giw_o, tau, w_n, 1e-12)
    g0iw_d, g0iw_o = dimer.self_consistency(
        1j * w_n, 1j * giw_d.imag, giw_o.real, 0., tp, 0.25)
    siw_d, siw_o = ipt_imag.dimer_sigma(u_int, tp, g0iw_d, g0iw_o, tau, w_n)

    return giw_d, giw_o, siw_d, siw_o, w_n


def ipt_g_s(u_int, tp, BETA, seed, w):
    giw_d, giw_o, siw_d, siw_o, w_n = ipt_u_tp(u_int, tp, BETA, seed)

    w_set = np.arange(0, 581, 4)
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


def plot_spectral(w, U, tp, gss, ss, ax, ylim):
    quas_z, sig_0, dw_sig0 = low_en_qp(ss)
    tpp = (tp + sig_0) * quas_z
    ax[0].plot(w, -gss.imag, 'C0')
    llg = gf.semi_circle_hiltrans(w + 1e-8j - tpp, quas_z) * quas_z
    ax[0].plot(w, -llg.imag, "C3--", lw=2)
    ax[0].text(0.05, 0.72, r'$Z={:.3}$'.format(quas_z) + '\n' +
               r'$\tilde{{t}}_\perp={:.2f}$'.format(tpp),
               transform=ax[0].transAxes, fontsize=14)
    for a in ax:
        a.set_title(r'$U={}$; $t_\perp={}$'.format(U, tp), fontsize=14)
    # plt.plot(w, gst.real)
    ax[1].plot(w, ss.real, 'C4', label=r'$\Re e$')
    ax[1].plot(w, ss.imag, 'C2', label=r'$\Im m$')
    ax[1].plot(w, sig_0 + dw_sig0 * w, 'k:')
    ax[0].set_ylim(0, 2)
    ax[0].set_yticks([0, 1, 2])
    ax[0].set_yticklabels([0, 1, 2])
    ax[1].legend(loc=2)
    ax[1].set_ylim(*ylim)
    ax[0].set_xlim([-2.5, 2.5])
    ax[1].set_xlim([-2.5, 2.5])


w = np.linspace(-4, 4, 2**12)
dw = w[1] - w[0]

BETA = 512.
nfp = gf.fermi_dist(w, BETA)

plt.close('all')
fig_g, axg = plt.subplots(2, 2, sharex=True, sharey=True)
fig_s, axs = plt.subplots(2, 2, sharex=True, sharey=True)

gss = gf.semi_circle_hiltrans(w + 5e-3j)
gsa = gf.semi_circle_hiltrans(w + 5e-3j)
for ax, (U, TP) in zip(zip(axg.flatten(), axs.flatten()),
                       [(3.45, 0.3), (1.2, 0.8),
                        (1.5, 0.8), (2.3, 0.3)]):
    (gss, _), (ss, _) = ipt_real.dimer_dmft(
        U, TP, nfp, w, dw, gss, gsa, conv=1e-3)
    if U == 2.3:
        gss, ss, w = ipt_g_s(U, TP, BETA, 'ins', w)
    plot_spectral(w, U, TP, gss, ss, ax, (-2.5, 2.5))

for b in [axg[0, 0], axg[1, 0]]:
    b.set_ylabel(r'$-\Im m G_{AB}(\omega)$')

for b in [axs[0, 0], axs[1, 0]]:
    b.set_ylabel(r'$\Sigma_{AB}(\omega)$')

for b in [axg[1, 0], axg[1, 1], axs[1, 0], axs[1, 1], ]:
    b.set_xlabel(r'$\omega$')

for f in [fig_g, fig_s]:
    f.subplots_adjust(wspace=0.25, hspace=0.05)

# fig_g.savefig('IPT_EF_qp_DOS.pdf')
# fig_s.savefig('IPT_EF_qp_sigma.pdf')
