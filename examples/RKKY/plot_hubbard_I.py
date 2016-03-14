# -*- coding: utf-8 -*-
r"""
===================
Hubbard I for dimer
===================

The atomic self-energy is extracted and plotted into the lattice
Green's function to see the behavior of the insulating state.
"""
# Created Mon Mar 14 13:56:37 2016
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function

from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from dmft.common import matsubara_freq, gw_invfouriertrans
import dmft.RKKY_dimer as rt
import slaveparticles.quantum.operators as op


def molecule_sigma(omega, U, mu, tp, beta):
    """Return molecule self-energy in the given frequency axis"""

    h_at, oper = rt.dimer_hamiltonian(U, mu, -tp)
    oper_pair = [[oper[0], oper[0]], [oper[0], oper[1]]]

    eig_e, eig_v = op.diagonalize(h_at.todense())
    gfsU = np.array([op.gf_lehmann(eig_e, eig_v, c.T, beta, omega, d)
                     for c, d in oper_pair])

    invg = rt.mat_inv(gfsU[0], gfsU[1])
    plt.plot(omega.real, -gfsU[0].imag, label='Interacting')
    plt.plot(omega.real, -rt.mat_inv(omega, -tp)[0].imag, label='Free')
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$A(\omega)$')
    plt.title(r'Isolated dimer $U={}$, $t_\perp={}$, $\beta={}$'.format(U, tp, beta))
    plt.legend(loc=0)

    return [omega - invg[0], -tp - invg[1]]


###############################################################################
# The Real axis Self-energy
# -------------------------

w = np.linspace(-2, 2, 800)
U, mu, tp, beta = 2.15, 0., 0.3, 100.
sd_w, so_w = molecule_sigma(w + 5e-2j, U, mu, tp, beta)

f, ax = plt.subplots(2, sharex=True)
ax[0].plot(w, sd_w.real, label='Real')
ax[0].plot(w, sd_w.imag, label='Imag')
ax[1].plot(w, so_w.real, label='Real')
ax[1].plot(w, so_w.imag, label='Imag')
ax[0].legend(loc=0)
ax[1].set_xlabel(r'$\omega$')
ax[0].set_ylabel(r'$\Sigma_{AA}(\omega)$')
ax[1].set_ylabel(r'$\Sigma_{AB}(\omega)$')
ax[0].set_title(
    r'Isolated dimer $U={}$, $t_\perp={}$, $\beta={}$'.format(U, tp, beta))
plt.show()


###############################################################################
# Hubbard I Band dispersion
# -------------------------

eps_k = np.linspace(-1, 1, 61)
lat_gf = rt.mat_inv(np.add.outer(-eps_k, w + 5e-2j - sd_w), -tp - so_w)
Aw = -lat_gf[0].imag / np.pi

plt.figure()
for i, e in enumerate(eps_k):
    plt.plot(w, e + Aw[i], 'k')
    if e == 0:
        plt.plot(w, e + Aw[i], 'g', lw=3)

plt.ylabel(r'$\epsilon + A(\epsilon, \omega)$')
plt.xlabel(r'$\omega$')
plt.title(r'Hubbard I dimer $U={}$, $t_\perp={}$, $\beta={}$'.format(U, tp, beta))

plt.figure()
x, y = np.meshgrid(eps_k, w)
plt.pcolormesh(
    x, y, Aw.T, cmap=plt.get_cmap(r'inferno'))
plt.title(r'Hubbard I dimer $U={}$, $t_\perp={}$, $\beta={}$'.format(U, tp, beta))
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'$\omega$')

###############################################################################
# The Real axis Self-energy
# -------------------------

w = np.linspace(-2, 2, 800)
U, mu, tp, beta = 2.15, 0., 0.3, 5.
sd_w, so_w = molecule_sigma(w + 5e-2j, U, mu, tp, beta)

f, ax = plt.subplots(2, sharex=True)
ax[0].plot(w, sd_w.real, label='Real')
ax[0].plot(w, sd_w.imag, label='Imag')
ax[1].plot(w, so_w.real, label='Real')
ax[1].plot(w, so_w.imag, label='Imag')
ax[0].legend(loc=0)
ax[1].set_xlabel(r'$\omega$')
ax[0].set_ylabel(r'$\Sigma_{AA}(\omega)$')
ax[1].set_ylabel(r'$\Sigma_{AB}(\omega)$')
ax[0].set_title(
    r'Isolated dimer $U={}$, $t_\perp={}$, $\beta={}$'.format(U, tp, beta))
plt.show()


###############################################################################
# Hubbard I Band dispersion
# -------------------------

eps_k = np.linspace(-1, 1, 61)
lat_gf = rt.mat_inv(np.add.outer(-eps_k, w + 5e-2j - sd_w), -tp - so_w)
Aw = -lat_gf[0].imag / np.pi

plt.figure()
for i, e in enumerate(eps_k):
    plt.plot(w, e + Aw[i], 'k')
    if e == 0:
        plt.plot(w, e + Aw[i], 'g', lw=3)

plt.ylabel(r'$\epsilon + A(\epsilon, \omega)$')
plt.xlabel(r'$\omega$')
plt.title(r'Hubbard I dimer $U={}$, $t_\perp={}$, $\beta={}$'.format(U, tp, beta))

plt.figure()
x, y = np.meshgrid(eps_k, w)
plt.pcolormesh(
    x, y, Aw.T, cmap=plt.get_cmap(r'inferno'))
plt.title(r'Hubbard I dimer $U={}$, $t_\perp={}$, $\beta={}$'.format(U, tp, beta))
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'$\omega$')
