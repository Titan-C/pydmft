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

    h_at, oper = rt.dimer_hamiltonian(U, mu, tp)
    oper_pair = [[oper[0], oper[0]], [oper[0], oper[1]]]

    eig_e, eig_v = op.diagonalize(h_at.todense())
    gfsU = np.array([op.gf_lehmann(eig_e, eig_v, c.T, beta, omega, d)
                     for c, d in oper_pair])

    invg = rt.mat_inv(gfsU[0], gfsU[1])
    #plt.plot(omega.real, -gfsU[0].imag, label='Interacting')
    #plt.plot(omega.real, -rt.mat_inv(omega, -tp)[0].imag, label='Free')
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

import dmft.common as gf

U, mu, tp, beta = 2.15, 0., 0.3, 5.
sd_w, so_w = molecule_sigma(w + 1e-2j, U, mu, tp, beta)
g = gf.greenF(-1j * (w + 1e-2j), sd_w)
#g = gf.greenF(-1j *( w+1e-2j), U**2/4/w)
plt.plot(w, g.real)
plt.plot(w, -20 * g.imag)


def g0s(U, w):

    b = w + U**2 / 4 / w
    s = np.sign(w)
    return ((w + U**2 / 4 / w) + s * np.sqrt((w + 1e-5j + U**2 / 4 / w)**2 - 4 * (.25 + U**2 / 4 / w))) / 2


###############################################################################
# Including a bath link t=0.5
# ---------------------------

def molecule_sigma_bath(omega, U, mu, tp, beta):
    """Return molecule self-energy in the given frequency axis"""

    h_at, oper = rt.dimer_hamiltonian(U, mu, tp)
    oper_pair = [[oper[0], oper[0]], [oper[0], oper[1]]]

    eig_e, eig_v = op.diagonalize(h_at.todense())
    gfsU = np.array([op.gf_lehmann(eig_e, eig_v, c.T, beta, omega, d)
                     for c, d in oper_pair])

    di_gfs = [gfsU[0] + gfsU[1], gfsU[0] - gfsU[1]]
    invg = [1 / g for g in di_gfs]
    plt.plot(omega.real, -di_gfs[1].imag, label='Interacting')
    plt.plot(omega.real, -di_gfs[0].imag, label='Interacting')
    plt.plot(omega.real, -rt.mat_inv(omega, -tp)[0].imag, label='Free')
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$A(\omega)$')
    plt.title(r'Isolated dimer $U={}$, $t_\perp={}$, $\beta={}$'.format(U, tp, beta))
    plt.legend(loc=0)

    return [omega - tp - invg[0], omega + tp - invg[1]]

###############################################################################
# The Real axis Self-energy
# -------------------------

w = np.linspace(-2, 2, 1800)
plt.figure()
U, mu, tp, beta = 2.15, 0., 0., 5.
sd_w, so_w = molecule_sigma_bath(w + 1e-13j, U, mu, tp, beta)
#sd_w.imag, so_w.imag = -np.abs(sd_w), np.zeros_like(so_w.imag)
g0_1s = w + 1e-3j - tp
g0_1a = w + 1e-3j + tp

for i in range(1000):
    g0_1s = w + 5e-3j - tp - .25 / (g0_1s - sd_w)
    g0_1a = w + 5e-3j + tp - .25 / (g0_1a - so_w)
plt.plot(w, (1 / (g0_1s - sd_w)).imag)
plt.plot(w, (1 / (g0_1s - sd_w)).real)
plt.plot(w, (1 / (g0_1a - so_w)).imag)
plt.plot(w, (1 / (g0_1a - so_w)).real)

f, ax = plt.subplots(2, sharex=True)
ax[0].plot(w, sd_w.real, label='Real')
ax[0].plot(w, sd_w.imag, label='Imag')
ax[1].plot(w, so_w.real, label='Real')
ax[1].plot(w, so_w.imag, label='Imag')
sd_w, so_w = .5 * (sd_w + so_w), .5 * (sd_w - so_w)
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
lat_gf = rt.mat_inv(np.add.outer(-eps_k, g0_1s - sd_w), g0_1a - so_w)

lat_gfs = 1 / np.add.outer(-eps_k, g0_1s + 5e-2j - sd_w)
lat_gfa = 1 / np.add.outer(-eps_k, g0_1a + 5e-2j - so_w)
Aw = np.clip(-.5 * (lat_gfa + lat_gfs).imag / np.pi, -.5, 10)

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
sd_w, so_w = molecule_sigma_bath(w + 5e-2j, U, mu, tp, beta)

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

import dmft.common as gf

U, mu, tp, beta = 2.15, 0., 0.3, 5.
sd_w, so_w = molecule_sigma(w + 1e-2j, U, mu, tp, beta)
U, mu, tp, beta = 3.15, 0., 0.3, 5.
sd_w, so_w = molecule_sigma_bath(w + 1e-14j, U, mu, tp, beta)
plt.figure()
g = gf.greenF(-1j * (w + 1e-2j), sd_w)
#g = gf.greenF(-1j *( w+1e-2j), U**2/4/w)
plt.plot(w, g.real)
plt.plot(w, -g.imag)
sd_w, so_w = .5 * (sd_w + so_w), .5 * (sd_w - so_w)
g = gf.greenF(-1j * (w + 1e-2j), sd_w.real - 1e-5j)
plt.figure()
plt.plot(w, g.real)
plt.plot(w, -g.imag)
plt.plot(w, sd_w.real)


def g0s(U, w):

    b = w + U**2 / 4 / w
    s = np.sign(w)
    return ((w + U**2 / 4 / w) + s * np.sqrt((w + 1e-5j + U**2 / 4 / w)**2 - 4 * (.25 + U**2 / 4 / w))) / 2
