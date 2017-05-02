# -*- coding: utf-8 -*-
r"""
Compare the spectra and Fermi Liquid Parametrization
====================================================

Using a calculation in Matsubara frequencies then estimating the Fermi
Liquid behavior on those frequencies calculate the low energy
theory. Then continuate by Padé. Also continuate Sigma by Padé and do
the parametrization. Finally do everything on Real frequency IPT.
"""

# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev

import dmft.common as gf
from dmft import ipt_imag
from dmft import ipt_real
import dmft.dimer as dimer


def zero_f_meas_mat(siw_d, siw_o):
    sd_zew = np.polyfit(w_n[:2], siw_d[:2].imag, 1)
    so_zew = np.polyfit(w_n[:2], siw_o[:2].real, 1)

    zet = 1 / (1 - sd_zew[0])

    return zet, so_zew[1], sd_zew[0]


def low_en_qp(ss):
    glp = np.array([0.])
    sigtck = splrep(w, ss.real, s=0)
    sig_0 = splev(glp, sigtck, der=0)[0]
    dw_sig0 = splev(glp, sigtck, der=1)[0]
    quas_z = 1 / (1 - dw_sig0)
    return quas_z, sig_0, dw_sig0


def plot_spectral(w, tp, ss, ax, ylim, low_en):
    quas_z, sig_0, dw_sig0 = low_en
    tpp = (tp + sig_0) * quas_z
    gss = gf.semi_circle_hiltrans(w + 1e-8j - tp - ss)
    llg = gf.semi_circle_hiltrans(w + 1e-8j - tpp, quas_z) * quas_z
    ax[0].plot(w, -gss.imag, "C1-", lw=2)
    ax[0].plot(w, -llg.imag, "C3--", lw=2)
    ax[0].set_title(
        r'$\beta={}$; $U={}$; $t_\perp={}$'.format(BETA, U, tp), fontsize=14)
    ax[0].text(0.05, 0.75, r'$Z={:.3}$'.format(quas_z) + '\n' +
               r'$\tilde{{t}}_\perp={:.3f}$'.format(tpp),
               transform=ax[0].transAxes, fontsize=14)
    ax[1].plot(w, ss.real, 'C4', label=r'$\Re e$')
    ax[1].plot(w, ss.imag, 'C2', label=r'$\Im m$')
    ax[1].plot(w, sig_0 + dw_sig0 * w, 'k:')
    ax[1].set_xlim(-2, 2)
    ax[1].set_ylim(*ylim)


w = np.linspace(-4, 4, 2**12)
dw = w[1] - w[0]

BETA = 756.

tau, w_n = gf.tau_wn_setup(dict(BETA=BETA, N_MATSUBARA=2**12))
nfp = gf.fermi_dist(w, BETA)

U, tp = 3.4, 0.3

# Matsubara
giw_d, giw_o = dimer.gf_met(w_n, 0., tp, 0.5, 0.)
giw_d, giw_o, _ = dimer.ipt_dmft_loop(
    BETA, U, tp, giw_d, giw_o, tau, w_n, 5e-4)
g0iw_d, g0iw_o = dimer.self_consistency(
    1j * w_n, 1j * giw_d.imag, giw_o.real, 0., tp, 0.25)
siw_d, siw_o = ipt_imag.dimer_sigma(U, tp, g0iw_d, g0iw_o, tau, w_n)

# Continuate sigma with Padé
w_set = np.array([0, 3] + list(range(5, 761, 7)))
ss, _ = dimer.pade_diag(1j * siw_d.imag, siw_o.real, w_n, w_set, w + 0.0005j)


###############################################################################
# Low Energy in Sigma matsubara
# -----------------------------

fig, ax = plt.subplots(2, 1, sharex=True)
plot_spectral(w, tp, ss, ax, (-7, 5), zero_f_meas_mat(siw_d, siw_o))
# ax[1].set_title('Low Energy Matsubara sigma + Padé Sigma for plot')
# fig.savefig('IPT_comp_mat_lowe.pdf')

###############################################################################
# Low Energy in continuated sigma
# -------------------------------

fig, ax = plt.subplots(2, 1, sharex=True)
plot_spectral(w, tp, ss, ax, (-7, 5), low_en_qp(ss))
# ax[1].set_title('Low Energy Padé Sigma')
# fig.savefig('IPT_comp_pade_lowe.pdf')

###############################################################################
# All in real frequencies
# -----------------------

gss = gf.semi_circle_hiltrans(w + 5e-3j)
gsa = gf.semi_circle_hiltrans(w + 5e-3j)
(gss, gsa), (ss, sa) = ipt_real.dimer_dmft(
    U, tp, nfp, w, dw, gss, gsa, conv=1e-3)

fig, ax = plt.subplots(2, 1, sharex=True)
plot_spectral(w, tp, ss, ax, (-7, 5), low_en_qp(ss))
# ax[1].set_title('All Real frequency IPT')
# fig.savefig('IPT_comp_real_lowe.pdf')
# plt.close('all')
