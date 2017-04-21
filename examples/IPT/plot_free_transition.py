# -*- coding: utf-8 -*-
r"""
Transition on free energy
=========================

"""
# Created Fri Apr 21 15:11:05 2017
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function
import numpy as np
from scipy.integrate import simps
import matplotlib.pylab as plt


from dmft.ipt_imag import dmft_loop, single_band_ipt_solver
from dmft.common import greenF, tau_wn_setup, pade_continuation, fermi_dist

U = 3.
BETA = 100.
tau, w_n = tau_wn_setup(dict(BETA=BETA, N_MATSUBARA=2**9))

ig_iwn, is_iwn = dmft_loop(
    U, 0.5, -1.j / (w_n - 1 / w_n), w_n, tau, conv=1e-12)

mg_iwn, s_iwn = dmft_loop(U, 0.5, greenF(w_n), w_n, tau, conv=1e-10)
plt.plot(w_n, ig_iwn.imag, 's:')
plt.plot(w_n, mg_iwn.imag, 's:')
solution_diff = mg_iwn - ig_iwn  # zdelta
plt.plot(w_n, solution_diff.imag, 'o:')
plt.show()


def mix(gmet, gins, l):
    return (1 - l) * gmet + l * gins


def one_loop(giw, t, u_int):
    iw_n = 1j * w_n
    g_0_iwn = 1. / (iw_n - t**2 * giw)
    g_iwn, sigma_iwn = single_band_ipt_solver(u_int, g_0_iwn, w_n, tau)
    return g_iwn


integrand = []
mix_range = np.linspace(0, 1, 101)
for l in mix_range:
    g_in = mix(mg_iwn, ig_iwn, l)
    g_grad = one_loop(g_in, 0.5, U) - g_in
    integrand.append(np.dot(g_grad, solution_diff))


plt.plot(np.real(integrand))
plt.plot(np.imag(integrand))
plt.show()

fe = [0] + [simps(np.real(integrand[:i]), mix_range[:i])
            for i in range(1, 101)]
plt.plot(mix_range, fe)
plt.show()
