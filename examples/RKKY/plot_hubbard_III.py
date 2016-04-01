# -*- coding: utf-8 -*-
r"""
=====================
Hubbard III for dimer
=====================

Describing the position of the Self-Energy pole in the diagonal basis
"""
# Created Mon Mar 14 13:56:37 2016
# Author: Óscar Nájera
from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib.pyplot as plt

import dmft.common as gf

###############################################################################
# Spectral function approx
# ------------------------


w = np.linspace(-3, 3, 800)
U = 2.15
tp = 0.3
sp_2 = (U**2 + 16 * tp**2) / 4.
g0_1_b = w + tp + 1e-8j
g0_1_a = w - tp + 1e-8j

plt.plot(w, -(1 / (g0_1_a - sp_2 / g0_1_a)).imag)
plt.plot(w, -(1 / (g0_1_b - sp_2 / g0_1_b)).imag)
plt.xlabel(r'$\omega$')
plt.ylabel(r'$A(\omega)$')
plt.title(r'Isolated dimer approximation $U={}$, $t_\perp={}$'.format(U, tp))
plt.legend(loc=0)

for i in range(2000):
    g0_1_b = w + tp - .25 / (g0_1_b - sp_2 / g0_1_b)
    g0_1_a = w - tp - .25 / (g0_1_a - sp_2 / g0_1_a)

###############################################################################
# The Self-Energy
# ---------------

plt.figure()
plt.plot(w, (sp_2 / g0_1_b).real, label=r"Real")
plt.plot(w, (sp_2 / g0_1_b).imag, label=r"Imag")

plt.plot(w, (sp_2 / g0_1_a).real, label=r"Real")
plt.plot(w, (sp_2 / g0_1_a).imag, label=r"Imag")

plt.ylabel(r'$\Sigma(\omega)$')
plt.xlabel(r'$\omega$')
plt.title(r'$\Sigma(\omega)$ at $U= {}$'.format(U))
plt.legend(loc=0)
plt.ylim([-1.5, 1])

###############################################################################
# The Green Function
# ------------------

plt.figure()
plt.plot(w, (1 / (w + tp - sp_2 / g0_1_b)).real, label=r"Real")
plt.plot(w, (1 / (w + tp - sp_2 / g0_1_b)).imag, label=r"Imag")
plt.plot(w, (1 / (w - tp - sp_2 / g0_1_a)).real, label=r"Real")
plt.plot(w, (1 / (w - tp - sp_2 / g0_1_a)).imag, label=r"Imag")

plt.ylabel(r'$G(\omega)$')
plt.xlabel(r'$\omega$')
plt.title(r'$G(\omega)$ at $U= {}$'.format(U))
plt.legend(loc=0)

###############################################################################
# The Band Dispersion
# -------------------

eps_k = np.linspace(-1, 1, 61)
lat_gf = 1 / (np.add.outer(-eps_k, w + tp + 8e-2j) - sp_2 / g0_1_b) + \
    1 / (np.add.outer(-eps_k, w - tp + 8e-2j) - sp_2 / g0_1_a)
Aw = -lat_gf.imag / np.pi / 2


gf.plot_band_dispersion(w, Aw, 'Hubbard III band dispersion', eps_k)
