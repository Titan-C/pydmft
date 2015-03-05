# -*- coding: utf-8 -*-
r"""
==========
IPT Solver
==========

Within the iterative perturbative theory (IPT) the aim is to express the
self-energy of the impurity problem as

.. math:: \Sigma(\tau) \approx U^2 \mathcal{G}^0(\tau)^3

the contribution of the Hartree-term is not included here as it is cancelled

"""
from __future__ import division, absolute_import, print_function

from dmft.common import greenF, gt_fouriertrans, gw_invfouriertrans, \
    matsubara_freq
import numpy as np
import matplotlib.pylab as plt



def ipt_solver(u_int, g_0_iwn):

    g_0_tau = gw_invfouriertrans(g_0_iwn, tau, iwn, beta)
    sigma_tau = u_int**2 * g_0_tau**3
    sigma_iwn = gt_fouriertrans(sigma_tau, tau, iwn, beta)
    g_iwn = g_0_iwn / (1 - sigma_iwn * g_0_iwn)

    return g_iwn


def dmft_loop(loops, u_int, g_iwn):
    for i in range(loops):
        g_0_iwn = 1. / (iwn - t**2 * g_iwn)
        g_iwn = ipt_solver(u_int, g_0_iwn)
        plt.plot(iwn.imag, g_iwn.imag, '+-', label='it {}'.format(i))
    return g_iwn


beta = 50
t = 0.5
tau = np.linspace(0, beta, 1001)
iwn = matsubara_freq(beta, 400)
g_iwn = greenF(iwn, D=2*t)
plt.figure()
g_iwn = dmft_loop(25, 4, g_iwn)
plt.plot(iwn.imag, (1/iwn).imag)
plt.ylabel(r'$G(i\omega_n)$')
plt.xlabel(r'$i\omega_n$')
plt.title('Matusubara frequency green function')
plt.ylim([-2, 0])
plt.legend(loc=0)
