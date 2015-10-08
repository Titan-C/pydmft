# -*- coding: utf-8 -*-
r"""
Fourier Transforms
==================

Transforming the non-interacting Green's Function for the Bethe
lattice.  Using the analytical expression for the tails. It is
important to note that to get good enough resolution in the time
domain at leas :math:`2\beta` Matsubara frequencies are needed.
"""

# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function
import dmft.common as gf
import matplotlib.pyplot as plt
plt.matplotlib.rcParams.update({'figure.figsize': (8, 8), 'axes.labelsize': 22,
                                'axes.titlesize': 22, 'figure.autolayout': True})

tau, w_n = gf.tau_wn_setup(dict(BETA=60, N_MATSUBARA=128))
fig, ax = plt.subplots(2, 1)
c_v = ['b', 'g', 'r', 'y']

for mu, c in zip([0, 0.3, 0.6, 1.], c_v):
    giw = gf.greenF(w_n, mu=mu)
    gtau = gf.gw_invfouriertrans(giw, tau, w_n, [1., -mu, 0.25])

    ax[0].plot(w_n, giw.real, c+'o:', label=r'$\mu={}$'.format(mu))
    ax[0].plot(w_n, giw.imag, c+'s:')
    ax[1].plot(tau, gtau, c, lw=2, label=r'$\mu={}$'.format(mu))

ax[0].set_xlim([0, 6])
ax[0].set_xlabel(r'$i\omega_n$')
ax[0].set_ylabel(r'$G(i\omega_n)$')
ax[1].set_xlabel(r'$\tau$')
ax[1].set_ylabel(r'$G(\tau)$')


###############################################################################
# Including the self-energy of the atomic limit so to have a third moment

fig, ax = plt.subplots(2, 1)
sigma = 2.5**2/4/(1j*w_n)
giiw = gf.greenF(w_n, mu=0, sigma=sigma)
gitau = gf.gw_invfouriertrans(giiw, tau, w_n, [1., 0, 2.5**2/4+0.25])
ax[0].plot(w_n, giiw.imag)
ax[1].plot(tau, gitau)

ax[0].set_xlim([0, 6])
ax[0].set_xlabel(r'$i\omega_n$')
ax[0].set_ylabel(r'$G(i\omega_n)$')
ax[1].set_xlabel(r'$\tau$')
ax[1].set_ylabel(r'$G(\tau)$')
