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
from dmft.common import greenF, tau_wn_setup

import numpy as np
import matplotlib.pylab as plt

U = 3.
beta = 64.
tau, w_n = tau_wn_setup(dict(BETA=beta, N_MATSUBARA=beta))
g_iwn0 = greenF(w_n)

fig_gw, gw_ax = plt.subplots()

g_iwn, _ = dmft_loop(U, 0.5, -1.j/w_n, w_n, tau)
gw_ax.plot(w_n, g_iwn.imag, 's-', label='Insulator Solution')

g_iwn, _ = dmft_loop(U, 0.5, g_iwn0, w_n, tau)
gw_ax.plot(w_n, g_iwn.imag, 's-', label='Metal Solution')

plt.plot(w_n, -1/w_n, label='$1/w$ tail')
plt.plot(w_n, -1/w_n + U**2/4/w_n**3, label='$w^{-1}+ w^{-3}$ tail')
gw_ax.set_xlim([0, max(w_n)])
cut = int(6.5*beta/np.pi)
gw_ax.set_ylim([g_iwn.imag[:cut].min()*1.1, 0])
plt.legend(loc=0)
plt.ylabel(r'$G(i\omega_n)$')
plt.xlabel(r'$i\omega_n$')
plt.title(r'$G(i\omega_n)$ at $\beta= {}$, $U= {}$'.format(beta, U))
