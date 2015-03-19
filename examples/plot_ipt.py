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

from dmft import ipt_imag

from dmft.common import greenF, matsubara_freq
import numpy as np
import matplotlib.pylab as plt

beta = 50
U = 3.2
t = 0.5
tau = np.linspace(0, beta, 1001)
iwn = matsubara_freq(beta, 400)
g_iwn0 = greenF(iwn, D=2*t)
g_iwn_log, sigma_iwn = ipt_imag.dmft_loop(25, U, t, g_iwn0, iwn, tau)
g_iwn = g_iwn_log[-1]

fig_gw, gw_ax = plt.subplots()
gw_ax.plot(iwn.imag, g_iwn.real, '+-', label='RE')
gw_ax.plot(iwn.imag, g_iwn.imag, 's-', label='IM')
plt.plot(iwn.imag, (1/iwn).imag, label='high w tail ')
gw_ax.set_xlim([0, 6.5])
cut = int(6.5*beta/np.pi)
gw_ax.set_ylim([g_iwn.imag[:cut].min()*1.1, 0])
plt.legend(loc=0)
plt.ylabel(r'$G(i\omega_n)$')
plt.xlabel(r'$i\omega_n$')
plt.title(r'$G(i\omega_n)$ at $\beta= {}$, $U= {}$'.format(beta, U) + \
          '\nConverged in {} dmft loops'.format(len(g_iwn_log)))
