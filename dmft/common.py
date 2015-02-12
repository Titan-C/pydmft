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
