# -*- coding: utf-8 -*-
r"""
Transition on free energy
=========================

"""
# Created Fri Apr 21 15:11:05 2017
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function
import numpy as np
from scipy.integrate import trapz
import matplotlib.pylab as plt


from dmft.ipt_imag import dmft_loop, single_band_ipt_solver
from dmft.common import greenF, tau_wn_setup

U = 3
BETA = 1 / 0.01
tau, w_n = tau_wn_setup(dict(BETA=BETA, N_MATSUBARA=2**11))

ig_iwn, is_iwn = dmft_loop(
    U, 0.5, -1.j / (w_n - 1 / w_n), w_n, tau, conv=1e-10)

mg_iwn, s_iwn = dmft_loop(U, 0.5, greenF(w_n), w_n, tau, conv=1e-10)
solution_diff = mg_iwn - ig_iwn  # zdelta
#plt.plot(w_n, ig_iwn.imag, '+')


def mix(gmet, gins, l):
    return (1 - l) * gins + l * gmet


def one_loop(giw, t, u_int):
    iw_n = 1j * w_n
    g_0_iwn = 1. / (iw_n - t**2 * giw)
    g_iwn, sigma_iwn = single_band_ipt_solver(u_int, g_0_iwn, w_n, tau)
    return g_iwn


integrand = []
mix_range = np.linspace(0, 1, 201)
for l in mix_range:
    g_in = mix(mg_iwn, ig_iwn, l)
    g_grad = one_loop(g_in, 0.5, U) - g_in
    integrand.append(np.dot(g_grad, solution_diff) / BETA)


fe = [0] + [trapz(np.real(integrand[:i]), mix_range[:i])
            for i in range(2, 202)]
plt.plot(mix_range, np.array(fe), label='T={:.3}'.format(1 / BETA))
plt.show()
plt.legend(loc=0)
