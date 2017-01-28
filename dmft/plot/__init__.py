# -*- coding: utf-8 -*-
r"""
Ploting utilities
"""
# Created Mon Aug  1 16:56:00 2016
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function
import numpy as np
import matplotlib.pyplot as plt


def plot_band_dispersion(omega, spec_eps_w, title, eps_k, style='both'):
    """Plot the band dispersion of spec_eps_w in 2 graphics styles

    Arpes spectral density color intensity plot
    Arpes dispersion line plot

    Parameters
    ----------
    omega : real ndarray, real frequencies
    spec_eps_w : real 2D ndarray, Dispesion of spectral function
    title : string with figure title
    eps_k : real ndarray, spacing of spectral function lines
    style : 'intensity', 'line', 'both' , plot arpes lines too
    """

    if style in ['both', 'line']:
        plt.figure()
        for i, eps in enumerate(eps_k):
            plt.plot(omega, eps + spec_eps_w[i], 'k')
            if eps == 0:
                plt.plot(omega, eps + spec_eps_w[i], 'g', lw=3)

        plt.ylabel(r'$\epsilon + A(\epsilon, \omega)$')
        plt.xlabel(r'$\omega$')
        plt.title(title)

    if style in ['both', 'intensity']:
        plt.figure()
        eps_axis, omega_axis = np.meshgrid(eps_k, omega)
        plt.pcolormesh(eps_axis, omega_axis, spec_eps_w.T,
                       cmap=plt.get_cmap(r'viridis'))
        plt.title(title)
        plt.xlabel(r'$\epsilon$')
        plt.ylabel(r'$\omega$')
