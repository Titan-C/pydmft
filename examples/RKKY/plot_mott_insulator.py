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
import dmft.ipt_imag as ipt
from dmft.ipt_real import dimer_dmft as dimer_dmft_real
import dmft.RKKY_dimer as rt
from slaveparticles.quantum import dos


def plot_gf(gw, sw, axes):
    axes[0].plot(w, -gw.imag)
    axes[1].plot(w, sw.real)
    axes[1].axhline(0, color='k')
    axes[2].plot(w, -sw.imag)

###############################################################################
# Insulator in IPT Imag local basis
# ---------------------------------
#


def dmft_solve(giw_d, giw_o, beta, u_int, tp, tau, w_n):
    giw_d, giw_o, loops = rt.ipt_dmft_loop(
        BETA, u_int, tp, giw_d, giw_o, tau, w_n, 1e-12)
    g0iw_d, g0iw_o = rt.self_consistency(
        1j * w_n, 1j * giw_d.imag, giw_o.real, 0., tp, 0.25)
    siw_d, siw_o = ipt.dimer_sigma(u_int, tp, g0iw_d, g0iw_o, tau, w_n)
    return giw_d, giw_o, siw_d, siw_o


plt.close('all')
u_int = 3.5
BETA = 100.

tau, w_n = gf.tau_wn_setup(dict(BETA=BETA, N_MATSUBARA=1024))
giw_d, giw_o = 1 / (1j * w_n + 4j / w_n), np.zeros_like(w_n) + 0j
w = np.linspace(-4, 4, 2**8)
w_set = np.concatenate((np.arange(100), np.arange(100, 200, 2)))


giw_dt0, giw_ot0, siw_dt0, siw_ot0 = dmft_solve(
    giw_d, giw_o, BETA, u_int, 0, tau, w_n)

gwdt0 = gf.pade_continuation(1j * giw_dt0.imag, w_n, w, w_set)
swdt0 = gf.pade_continuation(1j * siw_dt0.imag, w_n, w, w_set)
swdt0 = swdt0.real - 1j * np.abs(swdt0.imag)

giw_dt03, giw_ot03, siw_dt03, siw_ot03 = dmft_solve(
    giw_d, giw_o, BETA, u_int, 0.3, tau, w_n)

gwdt03 = gf.pade_continuation(1j * giw_dt03.imag, w_n, w, w_set)
swdt03 = gf.pade_continuation(1j * siw_dt03.imag, w_n, w, w_set)
swdt03 = swdt03.real - 1j * np.abs(swdt03.imag)

fig, axes = plt.subplots(3, 2, sharex=True)
fig.subplots_adjust(hspace=0, wspace=0.0)
plot_gf(gwdt0, swdt0, axes[:, 0])
plot_gf(gwdt03, swdt03, axes[:, 1])
axes[0, 0].set_ylabel(r'$A_{11}(\omega)$')
axes[1, 0].set_ylabel(r'$\Re e \Sigma_{11}(\omega)$')
axes[2, 0].set_ylabel(r'$-\Im m \Sigma_{11}(\omega)$')

for ax in axes.flatten():
    ax.set_yticks([])
for ax, lim in zip(axes, [[0, 2], [-4, 4], [0, 2]]):
    ax[0].set_ylim(lim)
    ax[1].set_ylim(lim)

axes[2, 0].set_xlabel(r'$\omega$')
axes[2, 1].set_xlabel(r'$\omega$')

###############################################################################
# Insulator in real IPT diagonal basis
# ------------------------------------
#
w = np.linspace(-6, 6, 2**13)
dw = w[1] - w[0]
nfp = dos.fermi_dist(w, BETA)
gseeds = gf.semi_circle_hiltrans(w + 1e-6j + 1.2)
gseeda = gf.semi_circle_hiltrans(w + 1e-6j - 1.2)
(gsst0, _), (sst0, _) = dimer_dmft_real(u_int, 0.3, nfp, w, dw, gseeds, gseeda)

(gsst03, _), (sst03, _) = dimer_dmft_real(
    u_int, 0.3, nfp, w, dw, gseeds, gseeda)


fig, axes = plt.subplots(3, 2, sharex=True)
fig.subplots_adjust(hspace=0, wspace=0.0)
plot_gf(gsst0, sst0, axes[:, 0])
plot_gf(gsst03, sst03, axes[:, 1])
axes[0, 0].set_ylabel(r'$A_{sym}(\omega)$')
axes[1, 0].set_ylabel(r'$\Re e \Sigma_{sym}(\omega)$')
axes[2, 0].set_ylabel(r'$-\Im m \Sigma_{sym}(\omega)$')

for ax in axes.flatten():
    ax.set_yticks([])
for ax, lim in zip(axes, [[0, 2], [-4, 4], [0, 4]]):
    ax[0].set_ylim(lim)
    ax[1].set_ylim(lim)

axes[2, 0].set_xlabel(r'$\omega$')
axes[2, 1].set_xlabel(r'$\omega$')
plt.xlim([-4, 4])
