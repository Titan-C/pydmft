# -*- coding: utf-8 -*-
r"""
=======================================================
IPT Solver Existence of Metalic and insulating solution
=======================================================

Within the iterative perturbative theory (IPT) the aim is to express the
self-energy of the impurity problem as

.. math:: \Sigma(\tau) \approx U^2 \mathcal{G}^0(\tau)^3

the contribution of the Hartree-term is not included here as it is cancelled

"""
from __future__ import division, absolute_import, print_function

from dmft.ipt_imag import dmft_loop
from dmft.common import greenF, tau_wn_setup, pade_coefficients, pade_rec, plot_band_dispersion


import numpy as np
import matplotlib.pylab as plt

U = 3.
beta = 64.
tau, w_n = tau_wn_setup(dict(BETA=beta, N_MATSUBARA=3 * beta))
omega = np.linspace(-6, 6, 1600)
g_iwn0 = greenF(w_n)

fig_giw, giw_ax = plt.subplots()
fig_gw, gw_ax = plt.subplots()
fig_sw, sw_ax = plt.subplots(1, 2, sharex=True)

g_iwn, s_iwn = dmft_loop(U, 0.5, -1.j / w_n, w_n, tau, conv=1e-8)
giw_ax.plot(w_n, g_iwn.imag, 's-', label='Insulator Solution')
pade_coefs = pade_coefficients(g_iwn, w_n)
gw_ax.plot(omega, -pade_rec(pade_coefs, omega, w_n).imag /
           np.pi, label='Insulator Solution')

pade_coefs = pade_coefficients(s_iwn, w_n)
sigma_w = pade_rec(pade_coefs, omega, w_n)
sw_ax[0].plot(omega, sigma_w.real, label='Insulator Solution')
sw_ax[1].plot(omega, sigma_w.imag, label='Insulator Solution')

eps_k = np.linspace(-1, 1, 61)
lat_gf = 1 / (np.add.outer(-eps_k, omega + 8e-2j) - sigma_w)
Aw = -lat_gf.imag / np.pi

plot_band_dispersion(
    omega, Aw, r'Insulator Band dispersion at $\beta= {}$, $U= {}$'.format(beta, U), eps_k)

g_iwn, s_iwn = dmft_loop(U, 0.5, g_iwn0, w_n, tau, conv=1e-12)
giw_ax.plot(w_n, g_iwn.imag, 's-', label='Metal Solution')
pade_coefs = pade_coefficients(g_iwn, w_n)
gw_ax.plot(omega, -pade_rec(pade_coefs, omega, w_n).imag /
           np.pi, label='Metal Solution')

pade_coefs = pade_coefficients(1j * s_iwn.imag, w_n)
sigma_w = pade_rec(pade_coefs, omega, w_n)
sw_ax[0].plot(omega, sigma_w.real, label='Metal Solution')
sw_ax[1].plot(omega, sigma_w.imag, label='Metal Solution')

giw_ax.plot(w_n, -1 / w_n, label='$1/w$ tail')
giw_ax.plot(w_n, -1 / w_n + U**2 / 4 / w_n**3, label='$w^{-1}+ w^{-3}$ tail')

cut = int(6.5 * beta / np.pi)
giw_ax.set_xlim([0, 6.5])
giw_ax.set_ylim([g_iwn.imag[:cut].min() * 1.1, 0])

giw_ax.set_ylabel(r'$G(i\omega_n)$')
giw_ax.set_xlabel(r'$i\omega_n$')
giw_ax.set_title(r'$G(i\omega_n)$ at $\beta= {}$, $U= {}$'.format(beta, U))
giw_ax.legend(loc=0)

gw_ax.set_ylabel(r'$A(\omega)$')
gw_ax.set_xlabel(r'$\omega$')
gw_ax.set_title(r'Spectral function at $\beta= {}$, $U= {}$'.format(beta, U))
gw_ax.legend(loc=0)

sw_ax[0].set_ylabel(r'$\Re e \Sigma(\omega)$')
sw_ax[0].set_ylim([-25, 25])
sw_ax[1].set_ylabel(r'$\Im m \Sigma(\omega)$')
sw_ax[0].set_xlabel(r'$\omega$')
sw_ax[1].set_xlabel(r'$\omega$')
sw_ax[0].set_title(r'Self-Energy at $\beta= {}$, $U= {}$'.format(beta, U))
sw_ax[0].legend(loc=0)
sw_ax[1].legend(loc=0)


lat_gf = 1 / (np.add.outer(-eps_k, omega + 8e-2j) - sigma_w)
Aw = -lat_gf.imag / np.pi

plot_band_dispersion(
    omega, Aw, r'Metal Band dispersion at $\beta= {}$, $U= {}$'.format(beta, U), eps_k)

omega = np.linspace(-6, 6, 1600)
from slaveparticles.quantum.dos import bethe_lattice
from slaveparticles.quantum.operators import fermi_dist


def estimate_diferentials(w):
    dw = w[1:] - w[:-1]
    mw = (w[1:] - w[:-1]) / 2
    return np.concatenate((mw, [0])) + np.concatenate(([0], mw))


def optical_cond(omega, eps_k, sigma_w, beta):
    nuv = omega[omega > 0]
    zerofreq = len(nuv)
    dw = omega[1] - omega[0]
    de = eps_k[1] - eps_k[0]

    lat_gf = 1 / (np.add.outer(-eps_k, omega + 5e-2j) - sigma_w)
    Aw = -lat_gf.imag / np.pi

    nf = fermi_dist(omega, beta) - fermi_dist(np.add.outer(nuv, omega), beta)

    a = np.array([[np.sum(Aw[e] * np.roll(Aw[e], -i) * nf[i])
                   for i in range(len(nuv))] for e in range(len(eps_k))]) / nuv

    return nuv, (bethe_lattice(eps_k, .5).reshape(-1, 1) * a).sum(axis=0) * de * dw

pade_coefs = pade_coefficients(1j * s_iwn.imag, w_n)
sigma_w = pade_rec(pade_coefs, omega, w_n)
nuv, con = optical_cond(omega, eps_k, sigma_w, beta)
plot(nuv, con)
