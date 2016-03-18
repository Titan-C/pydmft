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


###############################################################################
# Approximating Hubbard I
# =======================
#
# Here I use the molecule self-energy

def molecule_sigma(omega, U, mu, tp, beta):
    """Return molecule self-energy in the given frequency axis"""

    h_at, oper = rt.dimer_hamiltonian(U, mu, tp)
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

w = np.linspace(-3, 3, 800)
U, mu, tp, beta = 2.15, 0., 0.3, 100.
sd_w, so_w = molecule_sigma(w + 5e-2j, U, mu, tp, beta)


def plot_self_energy(w, sd_w, sd_o, U, mu, tp, beta):
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

plot_self_energy(w, sd_w, so_w, U, mu, tp, beta)

###############################################################################
# Hubbard I Band dispersion
# -------------------------

eps_k = np.linspace(-1, 1, 61)
lat_gf = rt.mat_inv(np.add.outer(-eps_k, w + 5e-2j - sd_w), -tp - so_w)
Aw = -lat_gf[0].imag / np.pi


def plot_band_dispersion(w, Aw, title):
    plt.figure()
    for i, e in enumerate(eps_k):
        plt.plot(w, e + Aw[i], 'k')
        if e == 0:
            plt.plot(w, e + Aw[i], 'g', lw=3)

    plt.ylabel(r'$\epsilon + A(\epsilon, \omega)$')
    plt.xlabel(r'$\omega$')
    plt.title(title)

    plt.figure()
    x, y = np.meshgrid(eps_k, w)
    plt.pcolormesh(
        x, y, Aw.T, cmap=plt.get_cmap(r'inferno'))
    plt.title(title)
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'$\omega$')

plot_band_dispersion(
    w, Aw, r'Hubbard I dimer $U={}$, $t_\perp={}$, $\beta={}$'.format(U, tp, beta))

###############################################################################
# The Real axis Self-energy
# -------------------------

w = np.linspace(-3, 3, 800)
U, mu, tp, beta = 2.15, 0., 0.3, 5.
sd_w, so_w = molecule_sigma(w + 5e-2j, U, mu, tp, beta)
plot_self_energy(w, sd_w, so_w, U, mu, tp, beta)

###############################################################################
# Hubbard I Band dispersion
# -------------------------

eps_k = np.linspace(-1, 1, 61)
lat_gf = rt.mat_inv(np.add.outer(-eps_k, w + 5e-2j - sd_w), -tp - so_w)
Aw = -lat_gf[0].imag / np.pi
plot_band_dispersion(
    w, Aw, r'Hubbard I dimer $U={}$, $t_\perp={}$, $\beta={}$'.format(U, tp, beta))


###############################################################################
# Approximating a bath in the spirit of Hubbard III
# =================================================
#
# Here I prefer to work in the diagonal basis of the Green
# function. Then I search for the non-interacting propagator in an
# iterative fixed point way


def molecule_sigma_diag(omega, U, mu, tp, beta):
    """Return molecule self-energy in the given frequency axis"""

    h_at, oper = rt.dimer_hamiltonian(U, mu, tp)
    oper_pair = [[oper[0], oper[0]], [oper[0], oper[1]]]

    eig_e, eig_v = op.diagonalize(h_at.todense())
    gfsU = np.array([op.gf_lehmann(eig_e, eig_v, c.T, beta, omega, d)
                     for c, d in oper_pair])

    di_gfs = [gfsU[0] + gfsU[1], gfsU[0] - gfsU[1]]
    invg = [1 / g for g in di_gfs]
    plt.plot(omega.real, -di_gfs[0].imag, label='Interacting Sym')
    plt.plot(omega.real, -di_gfs[1].imag, label='Interacting ASym')
    plt.plot(omega.real, -rt.mat_inv(omega, -tp)[0].imag, label='Free')
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$A(\omega)$')
    plt.title(r'Isolated dimer $U={}$, $t_\perp={}$, $\beta={}$'.format(U, tp, beta))
    plt.legend(loc=0)

    return [omega - tp - invg[0], omega + tp - invg[1]]

###############################################################################
# The Real axis Self-energy
# -------------------------
#
# Molecule green function and self-energy are calculated with negligible
# broadening. The effect of this approximation is to create a Green
# function that is equivalente to the hilbert transform of the molecule
# green function

w = np.linspace(-3, 3, 600)
plt.figure()
U, mu, tp, beta = 2.15, 0., 0.3, 100.
ss_w, sa_w = molecule_sigma_diag(w + 1e-15j, U, mu, tp, beta)


def approximate_free_propagator(ss_w, sa_w, title):
    g0_1s = w + 1e-3j - tp
    g0_1a = w - 1e-3j + tp

    plt.figure()
    for i in range(1000):
        g0_1s = w + 5e-3j - tp - .25 / (g0_1s - ss_w)
        g0_1a = w + 5e-3j + tp - .25 / (g0_1a - sa_w)
    plt.plot(w, (1 / (g0_1s - ss_w)).real, label=r'$Re G_{SYM}$')
    plt.plot(w, (1 / (g0_1a - sa_w)).real, label=r'$Im G_{ASYM}$')
    plt.plot(w, -(1 / (g0_1s - ss_w)).imag, label=r'$Re G_{SYM}$')
    plt.plot(w, -(1 / (g0_1a - sa_w)).imag, label=r'$Im G_{ASYM}$')
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$G(\omega)$')
    plt.title(title)
    plt.legend(loc=0)

    return g0_1s, g0_1a

g0_1s, g0_1a = approximate_free_propagator(
    ss_w, sa_w, r'Bath self-consistent dimer $U={}$, $t_\perp={}$, $\beta={}$'.format(U, tp, beta))

###############################################################################
# Band dispersion
# ---------------

eps_k = np.linspace(-1., 1., 61)
lat_gfs = 1 / np.add.outer(-eps_k, g0_1s + 5e-5j - ss_w)
lat_gfa = 1 / np.add.outer(-eps_k, g0_1a + 5e-5j - sa_w)
Aw = np.clip(-.5 * (lat_gfa + lat_gfs).imag / np.pi, -.5, 2)

plot_band_dispersion(
    w, Aw,  r'Bath self-consistent dimer $U={}$, $t_\perp={}$, $\beta={}$'.format(U, tp, beta))

###############################################################################
# The Real axis Self-energy
# -------------------------

U, mu, tp, beta = 2.15, 0., 0.3, 5.
ss_w, sa_w = molecule_sigma_diag(w + 1e-15j, U, mu, tp, beta)
g0_1s, g0_1a = approximate_free_propagator(
    ss_w, sa_w, r'Bath self-consistent dimer $U={}$, $t_\perp={}$, $\beta={}$'.format(U, tp, beta))

###############################################################################
# Band dispersion
# ---------------

lat_gfs = 1 / np.add.outer(-eps_k, g0_1s + 5e-5j - ss_w)
lat_gfa = 1 / np.add.outer(-eps_k, g0_1a + 5e-5j - sa_w)
Aw = np.clip(-.5 * (lat_gfa + lat_gfs).imag / np.pi, -.5, 2)

plot_band_dispersion(
    w, Aw,  r'Bath self-consistent dimer $U={}$, $t_\perp={}$, $\beta={}$'.format(U, tp, beta))
