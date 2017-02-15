# -*- coding: utf-8 -*-
r"""
Tools to calculate quantities out of spectral functions
=======================================================

"""
# Created Mon Jul 18 19:08:03 2016
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function


import numpy as np
import scipy.signal as signal
from scipy.integrate import trapz


def bubble(A1, A2, nf):
    r"""Calculates the Polarization Bubble convolution given 2 Spectral functions

    It follows the formula

    .. math::

        \Pi(w') &= \int dw A_1(w) A_2(w+w') (n_f(w)-n_f(w+w')) \\
            &= \int dw A_1^+(w) A_2(w+w')-A_1(w) A_2^+(w+w')

    Parameters
    ----------
    A1 : 1D ndarrays, only information in w
        Correspond to the spectral functions
    A2 : 1D ndarrays, only information in w
        Correspond to the spectral functions
    nf : ndarray
        Fermi function
    """

    return signal.fftconvolve((A1 * nf)[::-1], A2, mode='same') - \
        signal.fftconvolve(A1[::-1], A2 * nf, mode='same')


def optical_conductivity(lat_A1, lat_A2, nf, w, dosde):
    r"""Calculates the optical conductivity from lattice spectral functions

    .. math:: \sigma(w) = \int dE \rho(E) \Pi (E,w) / w

    Parameters
    ----------
    lat_A1 : 2D ndarrays
        lattice Spectral functions A(E,w)
    lat_A2 : 2D ndarrays
        lattice Spectral functions A(E,w)
    nf : 1D ndarray
        fermi function
    w : 1D ndarray
        real frequency array
    dosde : 1D ndarray
        differentially weighted density of states dE ρ(E)

    Returns
    -------
    Re σ(w) : 1D ndarray
    Real part of optical conductivity. Posterior scaling required

    See also
    --------
    bubble

"""
    dw = w[1] - w[0]
    lat_sig = np.array([bubble(A1, A2, nf) for A1, A2 in zip(lat_A1, lat_A2)])
    resig = (dosde * lat_sig).sum(axis=0) * dw / (w - dw / 2)
    center = int(len(w) / 2)
    resig[center] = (resig[center - 1] + resig[center + 1]) / 2
    return resig


def dc_conductivity(lat_A1, lat_A2, dnf, w, dosde):
    lat_sig = np.array([trapz(A1 * A2 * dnf, w)
                        for A1, A2 in zip(lat_A1, lat_A2)])
    resig = (dosde * lat_sig).sum(axis=0)
    return resig
