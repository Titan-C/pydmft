# -*- coding: utf-8 -*-
r"""
Hubbard I solver
================

Atomic limit expression of the self-energy
is described by

.. math:: \Sigma(\omega) =\frac{U n}{2} + \frac{ \frac{U^{2} n}{2} \left( 1 - \frac{n}{2}\right)}{\omega + \mu  - U \left(1 - \frac{n}{2}\right)}
"""
# Created Mon Sep 28 15:25:30 2015
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np


def hubbard_aprox(n, U, dmu, omega):

    k = np.linspace(0, np.pi, 65)

    mu = U/2 + dmu

    sigma = n*U/2 + n/2*(1-n/2)*U**2/(omega + mu - (1 - n/2)*U)
    eps_k = -2*np.cos(k)

    lat_gf = 1/(np.subtract.outer(omega + mu - sigma,  eps_k))
    A_kw = -lat_gf.imag/np.pi

    plt.figure()
    plt.pcolormesh(k, omega, A_kw , cmap='hot_r')
    plt.xticks([0, np.pi], [r'$\Gamma$', r'$X$'])
    plt.xlim([0, np.pi])
    plt.xlabel('k')
    plt.ylabel(r'$\omega$')
    plt.colorbar()

    gs = gridspec.GridSpec(2, 1, hspace=0.2, height_ratios=[1, 3])

    A_kw /= np.max(A_kw)
    A_w = np.sum(A_kw, axis=1)

    ax_Aw = plt.subplot(gs[0], )
    ax_Aw.plot(omega, A_w/max(A_w))
    A_kw += k
    ax_Akw = plt.subplot(gs[1])

    ax_Akw.plot(omega, A_kw[:, ::2], 'k')
    ax_Akw.set_xlim([min(omega), max(omega)])
    ax_Akw.set_xlabel(r'$\omega$')
    ax_Akw.set_yticks([0, np.pi])
    ax_Akw.set_yticklabels([r'$\Gamma$', r'$X$'])
    ax_Akw.set_ylabel('intensity')




omega = np.linspace(-5, 5, 600) + 0.05j
hubbard_aprox(1, 3, 0, omega)
