# -*- coding: utf-8 -*-
"""
Green Functions with complex imaginary Time
===========================================

Interface to treat arrays as Green functions. Deals with their
Fourier Transforms from Matsubara frequencies to Imaginary time. Uses
positive and negative Frequencies in the Matsubara axis and allows for
complex Imaginary Time.
"""

from __future__ import absolute_import, division, print_function

import math

import numpy as np
import dmft.common as gf
from numpy.fft import fft, ifft
from scipy.linalg import lstsq

greenF = gf.greenF


def matsubara_freq(beta=16., pos_size=256, fer=1):
    r"""Calculates an array containing the matsubara frequencies under the
    formula

    .. math:: \omega_n = \frac{\pi(2n + f)}{\beta}

    where :math:`f=1` in the case of fermions, and zero for bosons

    Parameters
    ----------
    beta : float
            Inverse temperature of the system
    pos_size : integer
            size of the array : amount of matsubara frequencies
    fer : 0 or 1 integer
            dealing with fermionic particles

    Returns
    -------
    real ndarray
    """

    pos_size = math.ceil(pos_size)

    return gf.matsubara_freq(beta, 2 * pos_size, fer - 2 * pos_size)


def tau_wn_setup(beta, n_matsubara):
    """return two numpy arrays one corresponding to the imaginary time array
    and the other to the matsubara frequencies. The time array is twice as
    dense for best results in the Fast Fourier Transform.

    Parameters
    ----------
    beta: float
        Inverse temperature of the system
    n_matsubara: int
        amount of positive Matsubara frequencies

    Returns
    -------
    tuple (tau real ndarray, w_n real ndarray)
    """

    w_n = matsubara_freq(beta, n_matsubara)
    tau = np.arange(0, beta, beta / len(w_n))
    # numerical errors bug
    if len(w_n) < len(tau):
        tau = tau[:-1]
    tau = np.concatenate((tau - beta, tau))

    return tau, w_n


def gt_fouriertrans(g_tau, tau, w_n, tail_coef=[1., 0., 0.]):
    r"""Performs a forward fourier transform for the interacting Green function
    in which only the interval :math:`[0,\beta)` is required and output given
    into positive fermionic matsubara frequencies up to the given cutoff.
    Time array is twice as dense as frequency array

    .. math:: G(i\omega_n) = \int_0^\beta G(\tau)
       e^{i\omega_n \tau} d\tau

    Parameters
    ----------
    g_tau : real float array
            Imaginary time interacting Green function
    tau : real float array
            Imaginary time points
    w_n : real float array
            fermionic matsubara frequencies. Only use the positive ones
    tail_coef : list of floats size 3
        The first moments of the tails

    Returns
    -------
    complex ndarray
            Interacting Greens function in matsubara frequencies

    See also
    --------
    freq_tail_fourier
    gt_fouriertrans"""

    beta = -tau[0]
    freq_tail, time_tail = freq_tail_fourier(tail_coef, beta, tau, w_n)

    gtau = g_tau - time_tail
    return beta * ifft(gtau) + freq_tail


def freq_tail_fourier(tail_coef, beta, tau, w_n):
    r"""Fourier transforms analytically the slow decaying tail_coefs of
    the Greens functions [matsubara]_

    +------------------------+-----------------------------------------+
    | :math:`G(iw)`          | :math:`G(t)`                            |
    +========================+=========================================+
    | :math:`(i\omega)^{-1}` | :math:`-\frac{1}{2}`                    |
    +------------------------+-----------------------------------------+
    | :math:`(i\omega)^{-2}` | :math:`\frac{1}{2}(\tau-\beta/2)`       |
    +------------------------+-----------------------------------------+
    | :math:`(i\omega)^{-3}` | :math:`-\frac{1}{4}(\tau^2 -\beta\tau)` |
    +------------------------+-----------------------------------------+

    See also
    --------
    gw_invfouriertrans
    gt_fouriertrans

    References
    ----------
    .. [matsubara] https://en.wikipedia.org/wiki/Matsubara_frequency#Time_Domain

    """

    freq_tail =   tail_coef[0] / (1.j * w_n)\
        + tail_coef[1] / (1.j * w_n)**2\
        + tail_coef[2] / (1.j * w_n)**3

    time_tail = - tail_coef[0] / 2 \
        + tail_coef[1] / 2 * (tau - beta / 2) \
                - tail_coef[2] / 4 * (tau**2 - beta * tau)

    return freq_tail, time_tail


def gw_invfouriertrans(g_iwn, tau, w_n, tail_coef=[1., 0., 0.]):
    r"""Performs an inverse fourier transform of the green Function in which
    only the imaginary positive matsubara frequencies
    :math:`\omega_n= \pi(2n+1)/\beta` with :math:`n \in \mathbb{N}` are used.
    The high frequency tails are transformer analytically up to the third moment.

    Output is the real valued positivite imaginary time green function.
    For the positive time output :math:`\tau \in [0;\beta)`.
    Array sizes need not match between frequencies and times, but a time array
    twice as dense is recommended for best performance of the Fast Fourrier
    transform.

    .. math::
       G(\tau) &= \frac{1}{\beta} \sum_{\omega_n}
                   G(i\omega_n)e^{-i\omega_n \tau} \\
       &= \frac{1}{\beta} \sum_{\omega_n}\left( G(i\omega_n)
          -\frac{1}{i\omega_n}\right) e^{-i\omega_n \tau} +
          \frac{1}{\beta} \sum_{\omega_n}\frac{1}{i\omega_n}e^{-i\omega_n \tau} \\

    Parameters
    ----------
    g_iwn : real float array
            Imaginary time interacting Green function
    tau : real float array
            Imaginary time points
    w_n : real float array
            fermionic matsubara frequencies. Only use the positive ones
    tail_coef : list of floats size 3
        The first moments of the tails


    Returns
    -------
    complex ndarray
            Interacting Greens function in matsubara frequencies

    See also
    --------
    gt_fouriertrans
    freq_tail_fourier
    """

    beta = tau[1] + tau[-1]
    freq_tail, time_tail = freq_tail_fourier(tail_coef, beta, tau, w_n)

    giwn = g_iwn - freq_tail

    g_tau = fft(giwn)
    g_tau *= np.exp(-1j * np.pi * (1 + len(w_n)) * tau / beta) / beta
    g_tau += time_tail

    return g_tau
