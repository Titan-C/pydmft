# -*- coding: utf-8 -*-
r"""
=========
For paper
=========

"""
# Created Mon Apr 11 14:17:29 2016
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function

import matplotlib.pyplot as plt
import numpy as np

import dmft.common as gf
import dmft.RKKY_dimer as rt
import dmft.ipt_imag as ipt


def ipt_u_tp(u_int, tp, beta, seed='ins'):
    tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=max(5 * beta, 256)))
    giw_d, giw_o = rt.gf_met(w_n, 0., 0., 0.5, 0.)
    if seed == 'ins':
        giw_d, giw_o = 1 / (1j * w_n + 4j / w_n), np.zeros_like(w_n) + 0j

    giw_d, giw_o, loops = rt.ipt_dmft_loop(
        beta, u_int, tp, giw_d, giw_o, tau, w_n)
    g0iw_d, g0iw_o = rt.self_consistency(
        1j * w_n, 1j * giw_d.imag, giw_o.real, 0., tp, 0.25)
    siw_d, siw_o = ipt.dimer_sigma(u_int, tp, g0iw_d, g0iw_o, tau, w_n)

    return siw_d, siw_o, w_n


def pade_diag(gf_d, gf_o, w_n, w_set, w):
    gf_s = 1j * gf_d.imag + gf_o.real  # Anti-bond
    pc = gf.pade_coefficients(gf_s[w_set], w_n[w_set])
    gr_s = gf.pade_rec(pc, w, w_n[w_set])

    return gr_s


def plot_pole_eq(w, gf, sig, pole, sty, ax):
    ax.plot(w, sig.imag, 'r' + sty, label=r'$\Im m \Sigma$')
    ax.plot(w, -gf.imag / np.pi, 'b' + sty, lw=2, label='DOS')
    if pole:
        ax.plot(w, (1 / gf).real, 'g' + sty, label=r'$\Re e G^{-1}$')
    ax.set_ylim([-1, .7])
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def hiltrans(zeta):
    sqr = np.sqrt(zeta**2 - 1)
    sqr = np.sign(sqr.imag) * sqr
    return 2 * (zeta - sqr)


def plot_spectral(u_int, tp, beta, seed, w, w_set, pole, sty, ax):

    siw_d, siw_o, w_n = ipt_u_tp(u_int, tp, BETA, seed)
    ss = pade_diag(siw_d, siw_o, w_n, w_set, w)
    gst = hiltrans(w - tp - (ss.real - 1j * np.abs(ss.imag)))

    plot_pole_eq(w, gst, ss, pole, sty, ax)

###############################################################################
# Metals
# ------
#
BETA = 100.
w = np.linspace(-4, 4, 800)
w_set = np.concatenate((np.arange(100), np.arange(100, 200, 20)))
f, ax = plt.subplots(3, 3, sharex=True)
plt.subplots_adjust(wspace=0, hspace=0)
plot_spectral(2.5, 0., BETA, 'met', w, w_set, True, '-', ax[0, 0])
plot_spectral(2.9, 0., BETA, 'met', w, w_set, False, '-', ax[1, 0])
plot_spectral(2.9, 0., BETA, 'ins', w, w_set, False, '-', ax[1, 0])
plot_spectral(3.9, 0., BETA, 'ins', w, w_set, True, '-', ax[2, 0])

plot_spectral(2.5, 0.3, BETA, 'met', w, w_set, True, '-', ax[0, 1])
plot_spectral(2.9, 0.3, BETA, 'met', w, w_set, False, '-', ax[1, 1])
plot_spectral(2.9, 0.3, BETA, 'ins', w, w_set, False, '-', ax[1, 1])
plot_spectral(3.9, 0.3, BETA, 'met', w, w_set, True, '-', ax[2, 1])

plot_spectral(2.5, 0.8, BETA, 'met', w, w_set, True, '-', ax[0, 2])
plot_spectral(2.9, 0.8, BETA, 'met', w, w_set, False, '-', ax[1, 2])
plot_spectral(2.9, 0.8, BETA, 'ins', w, w_set, False, '-', ax[1, 2])
plot_spectral(3.9, 0.8, BETA, 'met', w, w_set, True, '-', ax[2, 2])
