# -*- coding: utf-8 -*-
r"""
===================================
Dispersion of the spectral function
===================================

"""
# Author: Óscar Nájera

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import matplotlib.pyplot as plt
import numpy as np

import dmft.common as gf
import dmft.RKKY_dimer as rt
import dmft.ipt_imag as ipt


def loop_u_tp(u_range, tprange, beta, seed='mott gap'):
    tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=max(5 * beta, 256)))
    giw_d, giw_o = rt.gf_met(w_n, 0., 0., 0.5, 0.)
    if seed == 'mott gap':
        giw_d, giw_o = 1 / (1j * w_n + 4j / w_n), np.zeros_like(w_n) + 0j

    giw_s = []
    sigma_iw = []
    iterations = []
    for u_int, tp in zip(u_range, tprange):
        giw_d, giw_o, loops = rt.ipt_dmft_loop(
            beta, u_int, tp, giw_d, giw_o, tau, w_n)
        giw_s.append((giw_d, giw_o))
        iterations.append(loops)
        g0iw_d, g0iw_o = rt.self_consistency(
            1j * w_n, 1j * giw_d.imag, giw_o.real, 0., tp, 0.25)
        siw_d, siw_o = ipt.dimer_sigma(u_int, tp, g0iw_d, g0iw_o, tau, w_n)
        sigma_iw.append((siw_d.copy(), siw_o.copy()))

    print(np.array(iterations))

    return np.array(giw_s), np.array(sigma_iw), w_n


def pade_diag(gf_d, gf_o, w_n, w_set, w):
    gf_s = 1j * gf_d.imag + gf_o.real  # Anti-bond
    pc = gf.pade_coefficients(gf_s[w_set], w_n[w_set])
    gr_s = gf.pade_rec(pc, w, w_n[w_set])

    gf_a = 1j * gf_d.imag - gf_o.real  # bond
    pc = gf.pade_coefficients(gf_a[w_set], w_n[w_set])
    gr_a = gf.pade_rec(pc, w, w_n[w_set])

    return gr_s, gr_a


def plot_greenfunct(w, gfunc, title, ylabel, ax=None):
    if ax is None:
        f, ax = plt.subplots(1)
    ax.plot(w, gfunc.real, label=r'$\Re e$' + ylabel)
    ax.plot(w, -gfunc.imag, label=r'$-\Im m$' + ylabel)
    ax.legend(loc=0)
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel(ylabel + r'$(\omega)$')
    ax.set_title(title)
    ax.set_ylim([-3, 3])
    return ax


def plot_pole_eq(w, gf, sig, title):
    plt.figure()
    plt.plot(w, sig.imag, label=r'$\Im m \Sigma$')
    plt.plot(w, (1 / gf).real, label=r'$\Re e G^{-1}$')
    plt.plot(w, -gf.imag, label='DOS')
    plt.legend(loc=0)
    plt.title(title)
    plt.ylim([-3, 3])


def plot_dispersions(giw_s, sigma_iw, ur, tp, w_n, w, w_set):

    for U, (giw_d, giw_o), (sig_d, sig_o) in zip(ur, giw_s, sigma_iw):
        gs, ga = pade_diag(giw_d, giw_o, w_n, w_set, w)
        ss, sa = pade_diag(sig_d, sig_o, w_n, w_set, w)
        gst = gf.semi_circle_hiltrans(
            w - tp - (ss.real - 1j * np.abs(ss.imag)))

        lat_gfs = 1 / np.add.outer(-eps_k, w - tp + 5e-3j - ss)
        lat_gfa = 1 / np.add.outer(-eps_k, w + tp + 5e-3j - sa)
        #Aw = np.clip(-.5 * (lat_gfa + lat_gfs).imag / np.pi, 0, 2)

        Aw = np.clip(-lat_gfs.imag / np.pi, 0, 2)

        title = r'IPT lattice dimer $U={}$, $t_\perp={}$, $\beta={}$'.format(
            U, tp, BETA)
        ax = plot_greenfunct(w, gs, title, r'$G$')
        plot_greenfunct(w, gst, title, r'$G$', ax)
        plot_greenfunct(w, 1 / (w - tp - .25 * gst), title, r'$G0$', ax)
        ax = plot_greenfunct(w, ss, title, r'$\Sigma$')

        plot_pole_eq(w, gst, ss, title)
        gf.plot_band_dispersion(w, Aw, title, eps_k)

###############################################################################
# Metals
# ------
#
urange = [1.5, 2., 2.175, 2.5, 3.]
BETA = 100.
tp = 0.3
giw_s, sigma_iw, w_n = loop_u_tp(urange, tp * np.ones_like(urange), BETA, "M")
w = np.linspace(-4, 4, 800)
eps_k = np.linspace(-1., 1., 61)
w_set = np.concatenate((np.arange(100), np.arange(100, 200, 2)))
plot_dispersions(giw_s, sigma_iw, urange, tp, w_n, w, w_set)

###############################################################################
# Insulators
# ----------
tp = 0.3
urange = [0, .5, 1, 2.175, 2.5, 3., 3.5, 4., 5.]
giw_s, sigma_iw, w_n = loop_u_tp(
    urange, tp * np.ones_like(urange), BETA)

eps_k = np.linspace(-1., 1., 61)
plot_dispersions(giw_s, sigma_iw, urange, tp, w_n, w, w_set)

###############################################################################
# Insulators
# ----------
tp = 0.5
urange = [0, .5, 2.175, 2.5, 3., 3.5, 4., 5.]
giw_s, sigma_iw, w_n = loop_u_tp(
    urange, tp * np.ones_like(urange), BETA)

eps_k = np.linspace(-1., 1., 61)
plot_dispersions(giw_s, sigma_iw, urange, tp, w_n, w, w_set)
###############################################################################
# Insulators
# ----------
tp = 0.8
urange = [0, .5, 1.2, 1.5, 2., 2.5, 3., 3.5, 4., 5.]
giw_s, sigma_iw, w_n = loop_u_tp(
    urange, tp * np.ones_like(urange), BETA)

eps_k = np.linspace(-1., 1., 61)
plot_dispersions(giw_s, sigma_iw, urange, tp, w_n, w, w_set)

###############################################################################
# Insulators
# ----------
tp = 0.95
urange = [0, .5, 1.2, 1.5, 2., 2.5, 3., 3.5, 4., 5.]
giw_s, sigma_iw, w_n = loop_u_tp(
    urange, tp * np.ones_like(urange), BETA)

eps_k = np.linspace(-1., 1., 61)
plot_dispersions(giw_s, sigma_iw, urange, tp, w_n, w, w_set)


#urange = [3.2]
#beta = 100.
#tp = 0.2
# giw_s, sigma_iw, w_n = loop_u_tp(
# urange, tp * np.ones_like(urange), beta)
#
#w = np.linspace(-8, 8, 1600)
#w_set = np.concatenate((np.arange(80), np.arange(90, 140, 25)))
#eps_k = np.linspace(-2., 1., 61)
#ss, gs = plot_dispersions(giw_s, sigma_iw, urange, tp, w_n, w, w_set)
#
#
#tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=max(5 * beta, 256)))
#tp = 0.2
# g0iw_d, g0iw_o = rt.self_consistency(
# 1j * w_n, giw_s[0], giw_s[0], 0., tp, 0.25)
#siw_d, siw_o = ipt.dimer_sigma(urange[0], tp, g0iw_d, g0iw_o, tau, w_n)
#giw_d, giw_o = rt.dimer_dyson(g0iw_d, g0iw_o, siw_d, siw_o)
# g0iw_d, g0iw_o = rt.self_consistency(
# 1j * w_n, giw_s[0], giw_s[0], 0., tp, 0.25)
#siw_d, siw_o = ipt.dimer_sigma(urange[0], tp, g0iw_d, g0iw_o, tau, w_n)
# ss, gs = plot_dispersions(
# np.array([giw_d, giw_o]), np.array([siw_d, siw_o]), urange, tp, w_n, w,
# w_set)
