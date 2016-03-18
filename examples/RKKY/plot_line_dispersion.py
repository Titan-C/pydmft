# -*- coding: utf-8 -*-
r"""
===================================
Dispersion of the spectral function
===================================

"""
# Author: Óscar Nájera

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from math import ceil, log

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
    gf_s = 1j * gf_d.imag + gf_o
    gf_a = 1j * gf_d.imag - gf_o
    pc = gf.pade_coefficients(gf_s[w_set], w_n[w_set])
    gr_s = gf.pade_rec(pc, w + 5e-8j, w_n[w_set])

    pc = gf.pade_coefficients(gf_a[w_set], w_n[w_set])
    gr_a = gf.pade_rec(pc, w + 5e-8j, w_n[w_set])

    return gr_s, gr_a


def plot_band_dispersion(w, Aw, title):
    plt.figure()
    for i, e in enumerate(eps_k):
        plt.plot(w, e + Aw[i], 'k')
        if e == 0:
            plt.plot(w, e + Aw[i], 'g', lw=3)

    plt.ylabel(r'$\epsilon + A(\epsilon, \omega)$')
    plt.xlabel(r'$\omega$')
    plt.title(title)

    plt.figure()
    x, y = np.meshgrid(eps_k, w)
    plt.pcolormesh(
        x, y, Aw.T, cmap=plt.get_cmap(r'inferno'))
    plt.title(title)
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'$\omega$')

urange = np.linspace(2.5, 5.5, 10)
beta = 100.
tp = 0.3
giw_s, sigma_iw, w_n = loop_u_tp(
    urange, tp * np.ones_like(urange), beta)

w = np.linspace(-4, 4, 800)
eps_k = np.linspace(-1., 1., 61)


def plot_self_energy(w, sd_w, so_w, U, mu, tp, beta):
    f, ax = plt.subplots(2, sharex=True)
    ax[0].plot(w, sd_w.real, label='Real')
    ax[0].plot(w, sd_w.imag, label='Imag')
    ax[1].plot(w, so_w.real, label='Real')
    ax[1].plot(w, so_w.imag, label='Imag')
    ax[0].legend(loc=0)
    ax[1].set_xlabel(r'$\omega$')
    ax[0].set_ylabel(r'$\Sigma_{AA}(\omega)$')
    ax[1].set_ylabel(r'$\Sigma_{AB}(\omega)$')
    ax[0].set_title(
        r'Isolated dimer $U={}$, $t_\perp={}$, $\beta={}$'.format(U, tp, beta))
    plt.show()


def plot_dispersions(sigma_iw, ur, tp, w_n, w):

    for U, (sig_d, sig_o) in zip(ur, sigma_iw):
        sd, so = pade_diag(sig_d, sig_o, w_n, np.arange(120), w)
        plot_self_energy(w, sd, so, U, 0, tp, 100.)

        lat_gfs = 1 / np.add.outer(-eps_k, w - tp + 5e-2j - sd)
        lat_gfa = 1 / np.add.outer(-eps_k, w + tp + 5e-2j - so)
        Aw = np.clip(-.5 * (lat_gfa + lat_gfs).imag / np.pi, 0, 2)

        plot_band_dispersion(
            w, Aw, r'IPT lattice dimer $U={}$, $t_\perp={}$, $\beta={}$'.format(U, tp, beta))

plot_dispersions(sigma_iw, urange, tp, w_n, w)
