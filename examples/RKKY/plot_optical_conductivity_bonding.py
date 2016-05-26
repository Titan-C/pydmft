# -*- coding: utf-8 -*-
r"""
=================================================
Optical conductivity of bonding Spectral Function
=================================================

"""
# Author: Óscar Nájera

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import matplotlib.pyplot as plt
import numpy as np

import dmft.common as gf
import dmft.RKKY_dimer as rt
import dmft.ipt_imag as ipt
from slaveparticles.quantum.dos import bethe_lattice
from slaveparticles.quantum.operators import fermi_dist


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


def plot_optical_cond(giw_s, sigma_iw, ur, tp, w_n, w, w_set, beta, seed):
    nuv = w[w > 0]
    zerofreq = len(nuv)
    dw = w[1] - w[0]
    E = np.linspace(-1, 1, 300)
    dos = np.exp(-2 * E**2) / np.sqrt(np.pi / 2)
    de = E[1] - E[0]
    dosde = (dos * de).reshape(-1, 1)
    nf = fermi_dist(w, beta) - fermi_dist(np.add.outer(nuv, w), beta)
    eta = 0.8

    for U, (giw_d, giw_o), (sig_d, sig_o) in zip(ur, giw_s, sigma_iw):
        gs, ga = rt.pade_diag(giw_d, giw_o, w_n, w_set, w)
        ss, sa = rt.pade_diag(sig_d, sig_o, w_n, w_set, w)

        lat_Aa = (-1 / np.add.outer(-E, w + tp + 5e-3j - sa)).imag / np.pi
        lat_As = (-1 / np.add.outer(-E, w - tp + 5e-3j - ss)).imag / np.pi
        #lat_Aa = .5 * (lat_Aa + lat_As)
        #lat_As = lat_Aa

        a = np.array([[np.sum(lat_Aa[e] * np.roll(lat_Aa[e], -i) * nf[i])
                       for i in range(len(nuv))] for e in range(len(E))])
        a += np.array([[np.sum(lat_As[e] * np.roll(lat_As[e], -i) * nf[i])
                        for i in range(len(nuv))] for e in range(len(E))])
        b = np.array([[np.sum(lat_Aa[e] * np.roll(lat_As[e], -i) * nf[i])
                       for i in range(len(nuv))] for e in range(len(E))])
        b += np.array([[np.sum(lat_As[e] * np.roll(lat_Aa[e], -i) * nf[i])
                        for i in range(len(nuv))] for e in range(len(E))])
        b *= tp**2 * eta**2 / 2 / .25

        sigma_E_sum = (dosde * (a)).sum(axis=0) * dw / nuv
        plt.plot(nuv, sigma_E_sum, '--')
        sigma_E_sum = (dosde * (b)).sum(axis=0) * dw / nuv
        plt.plot(nuv, sigma_E_sum, ':')
        sigma_E_sum = (dosde * (a + b)).sum(axis=0) * dw / nuv
        plt.plot(nuv, sigma_E_sum)

        # To save data manually at some point
        np.savez('opt_cond{}'.format(seed), nuv=nuv, sigma_E_sum=sigma_E_sum)

        plt.xlabel(r'$\nu$')
        plt.ylabel(r'$\sigma(\nu)$')
        # plt.title(title)
        plt.ylim([0, .6])
        plt.xlim([0, 3])


###############################################################################
# Metals
# ------
#
urange = [2.5]  # [1.5, 2., 2.175, 2.5, 3.]
BETA = 100.
tp = 0.3
w = np.linspace(-6, 6, 800)
w_set = np.concatenate((np.arange(100), np.arange(100, 200, 2)))
giw_s, sigma_iw, w_n = loop_u_tp(urange, tp * np.ones_like(urange), BETA, "M")
plot_optical_cond(giw_s, sigma_iw, urange, tp, w_n, w, w_set, BETA, 'met')

giw_s, sigma_iw, w_n = loop_u_tp(urange, tp * np.ones_like(urange), BETA)
plot_optical_cond(giw_s, sigma_iw, urange, tp, w_n, w, w_set, BETA, 'ins')
