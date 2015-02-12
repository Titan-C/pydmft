# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 13:24:24 2015

@author: oscar
"""
import numpy as np


def matsubara_freq(beta=16., fer=1, Lrang=2**15):
    """Calculates an array containing the matsubara frequencies under the
    formula

    .. math:: i\\omega_n = i\\frac{\\pi(2n + f)}{\\beta}

    where :math:`i` is the imaginary unit and :math:`f=1` in the case of
    fermions, and zero for bosons

    Parameters
    ----------
    beta : float
            Inverse temperature of the system
    fer : 0 or 1 integer
            dealing with fermionic particles
    Lrang : integer
            size of the array : amount of matsubara frequencies

    Returns
    -------
    out : complex ndarray

    """
    return 1j*np.pi*np.arange(-Lrang+fer, Lrang, 2) / beta


def fft(gt, beta=16.):
    """Fourier transform into matsubara frequencies"""
    Lrang = gt.size // 2
    # trick to treat discontinuity
    gt[Lrang] -= 0.5
    gt[0] = -gt[Lrang]
    gt[::2] *= -1
    gw = np.fft.fft(gt)*beta/2/Lrang

    return gw


def ifft(gw, beta=16.):
    """Inverse Fast Fourier transform into time"""

    Lrang = gw.size // 2
    gt = np.fft.ifft(gw)*2*Lrang/beta
    gt[::2] *= -1
    # trick to treat discontinuity
    gt[Lrang] += 0.5
    gt[0] = -gt[Lrang]
    return gt.real


def greenF(w, sigma=0, mu=0, D=1):
    """Calculate green function lattice"""
    Gw = np.zeros(2*w.size, dtype=np.complex)
    zeta = w + mu - sigma
    sq = np.sqrt((zeta)**2 - D**2)
#    sig = np.sign(sq.imag*w.imag)
    Gw[1::2] = 2./(zeta+sq)
    return Gw



def gt_fouriertrans(gt, tau, iw, beta):
    r"""Performs a forward fourier transform for the interacting Green function
    in which only the interval :math:`[0,\beta]` is required and output given
    into positive fermionic matsubara frequencies up to the given cutoff. One
    benefits includes the correction of the high frequency tails.
    Array sizes need not match between frequencies and times

    .. math:: G(i\omega_n) = \int_0^\beta \left( G(\tau) + \frac{1}{2}\right)
       e^{i\omega_n \tau} d\tau + \frac{1}{i\omega_n}
    Parameters
    ----------
    gt : real float array
            Imaginary time interacting Green function
    tau : real float array
            Imaginary time points
    iw : complex float array
            fermionic matsubara frequencies. Only use the positive ones
    beta : float
        Inverse temperature of the system

    Returns
    -------
    out : complex ndarray
            Interacting Greens function in matsubara frequencies
    """
    power = np.exp(iw.reshape(-1, 1) * tau)
    gw = np.sum((gt + 0.5) * power, axis=1)*beta/(tau.size-1) + 1/iw
    return gw


def gw_invfouriertrans(gw, tau, iw, beta):
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
    gw : real float array
            Imaginary time interacting Green function
    tau : real float array
            Imaginary time points
    iw : complex float array
            fermionic matsubara frequencies. Only use the positive ones
    beta : float
        Inverse temperature of the system

    Returns
    -------
    out : complex ndarray
            Interacting Greens function in matsubara frequencies

    See also
    --------
    forwardFT"""

    power = np.exp(-iw * tau.reshape(-1, 1))
    gt = ((gw - 1/iw)*power).real
    gt = np.sum(gt, axis=1)*2/beta - 0.5
    return gt
