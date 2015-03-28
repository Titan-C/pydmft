# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 13:24:24 2015

@author: oscar
"""
import numpy as np
from scipy.integrate import romb
from scipy.linalg.blas import dger

def matsubara_freq(beta=16., size=250, fer=1):
    """Calculates an array containing the matsubara frequencies under the
    formula

    .. math:: \\omega_n = \\frac{\\pi(2n + f)}{\\beta}

    where :math:`f=1` in the case of fermions, and zero for bosons

    Parameters
    ----------
    beta : float
            Inverse temperature of the system
    size : integer
            size of the array : amount of matsubara frequencies
    fer : 0 or 1 integer
            dealing with fermionic particles

    Returns
    -------
    out : real ndarray
    """

    return np.pi*(fer+2*np.arange(size)) / beta


def tau_wn_setup(parms):
    """return two numpy arrays one corresponding to the imaginary time array
    and the other to the matsubara frequencies.
    parms is a dictionary with the keywords BETA, N_TAU, N_MATSUBARA"""
    tau = np.linspace(0, parms['BETA'], parms['N_TAU']+1)
    w_n = matsubara_freq(parms['BETA'], parms['N_MATSUBARA'])
    return tau, w_n

def greenF(w_n, sigma=0, mu=0, D=1):
    r"""Calculate the Bethe lattice Green function, defined as part of the
    hilbert transform.

    .. math:: G(i\omega_n) = \frac{2}{i\omega_n + \mu - \Sigma +
        \sqrt{(i\omega_n + \mu - \Sigma)^2 - D^2}}

    Parameters
    ----------
    w_n : real float array
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
    zeta = 1.j*w_n + mu - sigma
    sq = np.sqrt((zeta)**2 - D**2)
    sig = np.sign(sq.imag*w_n)
    return 2./(zeta + sig*sq)


def gt_fouriertrans(g_tau, tau, w_n):
    r"""Performs a forward fourier transform for the interacting Green function
    in which only the interval :math:`[0,\beta]` is required and output given
    into positive fermionic matsubara frequencies up to the given cutoff.
    Array sizes need not match between frequencies and times

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
    beta : float
        Inverse temperature of the system

    Returns
    -------
    out : complex ndarray
            Interacting Greens function in matsubara frequencies
    """
    beta = tau[-1]
    power = np.exp(1j * dger(1, w_n, tau))

    g_shape = g_tau.shape
    g_tau = g_tau.reshape(-1, 1, g_shape[-1]) * power

    return np.squeeze(romb(g_tau, dx=beta/(tau.size-1)))


def gw_invfouriertrans(g_iwn, tau, w_n):
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
       &= \frac{2}{\beta} \sum_{\omega_n>0}^{\omega_{max}} \left[
           \Re e G_{nt}(i\omega_n) \cos(\omega_n \tau)
            + \Im m G_{nt}(i\omega_n) \sin(\omega_n \tau) \right] - \frac{1}{2}

    where :math:`G_{nt}(i\omega_n)=\left((i\omega_n) -\frac{1}{i\omega_n}\right)`

    Parameters
    ----------
    g_iwn : real float array
            Imaginary time interacting Green function
    tau : real float array
            Imaginary time points
    w_n : real float array
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

    beta = tau[-1]
    w_ntau = dger(1, tau, w_n)
    fou_cos = np.cos(w_ntau)
    fou_sin = np.sin(w_ntau)
    g_shape = g_iwn.shape

    g_iwn = (g_iwn + 1.j/w_n).reshape(-1, 1, g_shape[-1])
    g_tau = g_iwn.real * fou_cos + g_iwn.imag * fou_sin
    return np.squeeze(np.sum(g_tau, axis=-1)*2/beta - 0.5)
