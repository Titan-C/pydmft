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

from dmft.common import greenF, tau_wn_setup
import numpy as np
import matplotlib.pylab as plt

parms = {'BETA': 50, 'MU': 0, 'U': 3, 't': 0.5, 'N_TAU': 2**10, 'N_MATSUBARA': 64}
tau, w_n = tau_wn_setup(parms)
g_iwn0 = greenF(w_n, D=2*parms['t'])
g_iwn_log, sigma_iwn = ipt_imag.dmft_loop(100, parms['U'], parms['t'], g_iwn0, w_n, tau)
g_iwn = g_iwn_log[-1]

fig_gw, gw_ax = plt.subplots()
gw_ax.plot(w_n, g_iwn.real, '+-', label='RE')
gw_ax.plot(w_n, g_iwn.imag, 's-', label='IM')
plt.plot(w_n, -1/w_n, label='high w tail ')
gw_ax.set_xlim([0, 6.5])
cut = int(6.5*parms['BETA']/np.pi)
gw_ax.set_ylim([g_iwn.imag[:cut].min()*1.1, 0])
plt.legend(loc=0)
plt.ylabel(r'$G(i\omega_n)$')
plt.xlabel(r'$i\omega_n$')
plt.title(r'$G(i\omega_n)$ at $\beta= {}$, $U= {}$'.format(parms['BETA'],
          parms['U']) + '\nConverged in {} dmft loops'.format(len(g_iwn_log)))
