# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 13:24:24 2015

@author: oscar
"""
import numpy as np


def matsubara_freq(beta=16., size=2**15, fer=1, neg=False):
    """Calculates an array containing the matsubara frequencies under the
    formula

    .. math:: i\\omega_n = i\\frac{\\pi(2n + f)}{\\beta}

    where :math:`i` is the imaginary unit and :math:`f=1` in the case of
    fermions, and zero for bosons

    Parameters
    ----------
    beta : float
            Inverse temperature of the system
    size : integer
            size of the array : amount of matsubara frequencies
    fer : 0 or 1 integer
            dealing with fermionic particles
    neg : bool
            include negative frequencies

    Returns
    -------
    out : complex ndarray

    """
    start = fer
    if neg:
        start -= size

    return 1j*np.pi*(start+2*np.arange(size)) / beta


def fft(gt, beta):
    """Fourier transform into matsubara frequencies"""
    Lrang = gt.size // 2
    # trick to treat discontinuity
    gt[Lrang] -= 0.5
    gt[0] = -gt[Lrang]
    gt[::2] *= -1
    gw = np.fft.fft(gt)*beta/2/Lrang

    return gw


def ifft(gw, beta):
    """Inverse Fast Fourier transform into time"""

    Lrang = gw.size // 2
    gt = np.fft.ifft(gw)*2*Lrang/beta
    gt[::2] *= -1
    # trick to treat discontinuity
    gt[Lrang] += 0.5
    gt[0] = -gt[Lrang]
    return gt.real


def greenF(iw, sigma=0, mu=0, D=1):
    r"""Calculate the Bethe lattice Green function, defined as part of the
    hilbert transform.

    .. math:: G(i\omega_n) = \frac{2}{i\omega_n + \mu - \Sigma +
        \sqrt{(i\omega_n + \mu - \Sigma)^2 - D^2}}

    Parameters
    ----------
    iw : complex float array
            fermionic matsubara frequencies.
    sigma : complex float or array
            local self-energy
    mu : real float
            chemical potential
    D : real
        Half-bandwidth of the bethe lattice non-interacting density of states

    Returns
    -------
    out : complex ndarray
            Interacting Greens function in matsubara frequencies, all odd
            entries are zeros
    """
    Gw = np.zeros(2*iw.size, dtype=np.complex)
    zeta = iw + mu - sigma
    sq = np.sqrt((zeta)**2 - D**2)
    sig = np.sign(sq.imag*iw.imag)
    Gw[1::2] = 2./(zeta + sig*sq)
    return Gw


def gt_fouriertrans(g_tau, tau, iwn, beta):
    r"""Performs a forward fourier transform for the interacting Green function
    in which only the interval :math:`[0,\beta]` is required and output given
    into positive fermionic matsubara frequencies up to the given cutoff. One
    benefits includes the correction of the high frequency tails.
    Array sizes need not match between frequencies and times

    .. math:: G(i\omega_n) = \int_0^\beta \left( G(\tau) + \frac{1}{2}\right)
       e^{i\omega_n \tau} d\tau + \frac{1}{i\omega_n}
    Parameters
    ----------
    g_tau : real float array
            Imaginary time interacting Green function
    tau : real float array
            Imaginary time points
    iwn : complex float array
            fermionic matsubara frequencies. Only use the positive ones
    beta : float
        Inverse temperature of the system

    Returns
    -------
    out : complex ndarray
            Interacting Greens function in matsubara frequencies
    """
    power = np.exp(iwn.reshape(-1, 1) * tau)
    g_iwn = np.sum((g_tau + 0.5) * power, axis=1)*beta/(tau.size-1) + 1/iwn
    return g_iwn


def gw_invfouriertrans(g_iwn, tau, iwn, beta):
    r"""Performs an inverse fourier transform of the green Function in which
    only the imaginary positive matsubara frequencies
    :math:`\omega_n= \pi(2n+1)/\beta` with :math:`n \in \mathbb{N}` are used.
    Output is the real valued positivite imaginary time green function.
    positive time output :math:`\tau \in [0;\beta]`.
    Array sizes need not match between frequencies and times

    .. math::
       G(\tau) &= \frac{1}{\beta} \sum_{\omega_n}
                   G(i\omega_n)e^{-i\omega_n \tau} \\
       &= \frac{1}{\beta} \sum_{\omega_n}\left( G(i\omega_n)
          -\frac{1}{i\omega_n}\right) e^{-i\omega_n \tau} +
          \frac{1}{\beta} \sum_{\omega_n}\frac{1}{i\omega_n}e^{-i\omega_n \tau} \\
       &= \frac{2}{\beta} \Re e \sum_{\omega_n>0}^{\omega_{max}} \left( G(i\omega_n)
          -\frac{1}{i\omega_n}\right) e^{-i\omega_n \tau} - \frac{1}{2}

    Parameters
    ----------
    g_iwn : real float array
            Imaginary time interacting Green function
    tau : real float array
            Imaginary time points
    iwn : complex float array
            fermionic matsubara frequencies. Only use the positive ones
    beta : float
        Inverse temperature of the system

    Returns
    -------
    out : complex ndarray
            Interacting Greens function in matsubara frequencies

    See also
    --------
    gt_fouriertrans"""

    power = np.exp(-iwn * tau.reshape(-1, 1))
    g_tau = ((g_iwn - 1/iwn)*power).real
    g_tau = np.sum(g_tau, axis=1)*2/beta - 0.5
    return g_tau
