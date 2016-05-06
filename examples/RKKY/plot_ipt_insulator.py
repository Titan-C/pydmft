# -*- coding: utf-8 -*-
r"""
===================================
INSULATOR
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


def ipt_u_tp(u_int, tp, beta, seed='ins'):
    tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=1024))
    giw_d, giw_o = rt.gf_met(w_n, 0., 0., 0.5, 0.)
    if seed == 'ins':
        giw_d, giw_o = 1 / (1j * w_n + 4j / w_n), np.zeros_like(w_n) + 0j

    giw_d, giw_o, loops = rt.ipt_dmft_loop(
        beta, u_int, tp, giw_d, giw_o, tau, w_n, 1e-12)
    g0iw_d, g0iw_o = rt.self_consistency(
        1j * w_n, 1j * giw_d.imag, giw_o.real, 0., tp, 0.25)
    siw_d, siw_o = ipt.dimer_sigma(u_int, tp, g0iw_d, g0iw_o, tau, w_n)

    return giw_d, giw_o, siw_d, siw_o, g0iw_d, g0iw_o, w_n


def pade_diag(gf_d, gf_o, w_n, w_set, w):
    gf_s = 1j * gf_d.imag + gf_o.real  # Anti-bond
    pc = gf.pade_coefficients(gf_s[w_set], w_n[w_set])
    gr_s = gf.pade_rec(pc, w, w_n[w_set])

    gf_a = 1j * gf_d.imag - gf_o.real  # bond
    pc = gf.pade_coefficients(gf_a[w_set], w_n[w_set])
    gr_a = gf.pade_rec(pc, w, w_n[w_set])

    return gr_s, gr_a


###############################################################################
# Insulator
# ---------
#
u_int = 3.5
BETA = 100.
tp = 0.3
title = r'IPT lattice dimer $U={}$, $t_\perp={}$, $\beta={}$'.format(
    u_int, tp, BETA)
giw_d, giw_o, siw_d, siw_o, g0iw_d, g0iw_o, w_n = ipt_u_tp(
    u_int, tp, BETA, 'ins')
w = np.linspace(-3, 3, 800)
eps_k = np.linspace(-1., 1., 61)
w_set = np.concatenate((np.arange(100), np.arange(100, 200, 2)))

plt.figure()
g0wd = gf.pade_contination(1j * g0iw_d.imag, w_n, w, w_set)
plt.plot(w, g0wd.real, label=r'$\Re e G0_{11}$')
plt.plot(w, g0wd.imag, label=r'$\Im m G0_{11}$')
plt.legend(loc=0)
plt.ylim([-2, 4])
plt.xlabel(r'$\omega$')
plt.ylabel(r'$G0_{11}(\omega)$')
plt.title(title)
plt.grid(which='both')
plt.xticks(np.linspace(-3, 3, 13))

plt.figure()
g0wo = gf.pade_contination(g0iw_o.real, w_n, w, w_set)
plt.plot(w, g0wo.real, label=r'$\Re e G0_{12}$')
plt.plot(w, g0wo.imag, label=r'$\Im m G0_{12}$')
plt.legend(loc=0)
plt.ylim([-5, 1])
plt.xlabel(r'$\omega$')
plt.ylabel(r'$G0_{12}(\omega)$')
plt.title(title)
plt.grid(which='both')
plt.xticks(np.linspace(-3, 3, 13))

plt.figure()
gwd = gf.pade_contination(1j * giw_d.imag, w_n, w, w_set)
plt.plot(w, gwd.real, label=r'$\Re e G_{11}$')
plt.plot(w, gwd.imag, label=r'$\Im m G_{11}$')
plt.legend(loc=0)
plt.ylim([-2, 2])
plt.xlabel(r'$\omega$')
plt.ylabel(r'$G_{11}(\omega)$')
plt.title(title)
plt.grid(which='both')
plt.xticks(np.linspace(-3, 3, 13))

plt.figure()
gwo = gf.pade_contination(giw_o.real, w_n, w, w_set)
plt.plot(w, gwo.real, label=r'$\Re e G_{12}$')
plt.plot(w, gwo.imag, label=r'$\Im m G_{12}$')
plt.legend(loc=0)
plt.ylim([-2, 2])
plt.xlabel(r'$\omega$')
plt.ylabel(r'$G_{12}(\omega)$')
plt.title(title)
plt.grid(which='both')
plt.xticks(np.linspace(-3, 3, 13))

plt.figure()
swd = gf.pade_contination(1j * siw_d.imag, w_n, w, w_set)
swd = swd.real - 1j * np.abs(swd.imag)
plt.plot(w, swd.real, label=r'$\Re e \Sigma_{11}$')
plt.plot(w, swd.imag, label=r'$\Im m \Sigma_{11}$')
plt.legend(loc=0)
plt.ylim([-3, 3])
plt.xlabel(r'$\omega$')
plt.ylabel(r'$\Sigma_{11}(\omega)$')
plt.title(title)
plt.grid(which='both')
plt.xticks(np.linspace(-3, 3, 13))

plt.figure()
swo = gf.pade_contination(siw_o.real, w_n, w, w_set)
plt.plot(w, swo.real, label=r'$\Re e \Sigma_{12}$')
plt.plot(w, swo.imag, label=r'$\Im m \Sigma_{12}$')
plt.legend(loc=0)
plt.ylim([-3, 3])
plt.xlabel(r'$\omega$')
plt.ylabel(r'$\Sigma_{12}(\omega)$')
plt.title(title)
plt.grid(which='both')
plt.xticks(np.linspace(-3, 3, 13))

###############################################################################
# Reconstruction
GAB = gf.semi_circle_hiltrans(w + 1e-2j - tp - (swd + swo))
G_B = gf.semi_circle_hiltrans(w + 1e-2j + tp - (swd - swo))

backl = .5 * (GAB + G_B)
plt.figure()
plt.plot(w, gwd.imag, label=r'$\Im m G_{11} ref$')
plt.plot(w, backl.real, label=r'$\Re e G_{11}$')
plt.plot(w, backl.imag, label=r'$\Im m G_{11}$')
plt.legend(loc=0)
plt.ylim([-2, 2])
plt.xlabel(r'$\omega$')
plt.ylabel(r'$G_{11}(\omega)$')
plt.title(title)
plt.grid(which='both')
plt.xticks(np.linspace(-3, 3, 13))

backo = .5 * (GAB - G_B)
plt.figure()
plt.plot(w, gwo.imag, label=r'$\Im m G_{12} ref$')
plt.plot(w, backo.real, label=r'$\Re e G_{12}$')
plt.plot(w, backo.imag, label=r'$\Im m G_{12}$')

plt.legend(loc=0)
plt.ylim([-2, 2])
plt.xlabel(r'$\omega$')
plt.ylabel(r'$G_{11}(\omega)$')
plt.title(title)
plt.grid(which='both')
plt.xticks(np.linspace(-3, 3, 13))
