# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 13:24:24 2015

@author: oscar
"""
import numpy as np


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
    and the other to the matsubara frequencies. The time array is twice as
    dense for best results in the Fast Fourier Transform.

    Parameters
    ----------
    parms: dictionary
        with keywords BETA, N_MATSUBARA

    Returns
    -------
    out: tuple (tau real ndarray, w_n real ndarray)
    """

    w_n = matsubara_freq(parms['BETA'], parms['N_MATSUBARA'])
    tau = np.arange(0, parms['BETA'], parms['BETA']/2/len(w_n))

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


def gt_fouriertrans(g_tau, tau, w_n, tail_coef=[1., 0., 0.]):
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

    beta = tau[1] + tau[-1]
    freq_tail, time_tail = freq_tail_fourier(tail_coef, beta, tau, w_n)

    gtau = g_tau - time_tail
    return beta*np.fft.ifft(gtau*np.exp(1j*np.pi*tau/beta))[...,:len(w_n)]+freq_tail


def freq_tail_fourier(tail_coef, beta, tau, w_n):
    """Fourier transforms analytically the slow decaying tail_coefs of
    the Greens functions[matsubara]

    .. [matsubara] https://en.wikipedia.org/wiki/Matsubara_frequency#Time_Domain

    in block
     """

    freq_tail =   tail_coef[0]/(1.j*w_n)\
                + tail_coef[1]/(1.j*w_n)**2\
                + tail_coef[2]/(1.j*w_n)**3

    time_tail = - tail_coef[0]/2 \
                + tail_coef[1]/2*(tau-beta/2) \
                - tail_coef[2]/4*(tau**2 - beta*tau)

    return freq_tail, time_tail


def gw_invfouriertrans(g_iwn, tau, w_n, tail_coef=[1., 0., 0.]):
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
    beta : float
        Inverse temperature of the system
    tail_coef : list of floats size 3
        The first moments of the tails

    Returns
    -------
    out : complex ndarray
            Interacting Greens function in matsubara frequencies

    See also
    --------
    gt_fouriertrans"""

    beta = tau[1] + tau[-1]
    freq_tail, time_tail = freq_tail_fourier(tail_coef, beta, tau, w_n)


    giwn = g_iwn - freq_tail

    g_tau = np.fft.fft(giwn, len(tau))
    g_tau *= np.exp(-1j*np.pi*tau/beta)

    return (g_tau*2/beta).real + time_tail


def fit_gf(w_n, giw):
    """Performs a quadratic fit of the -first- matsubara frequencies
    to estimate the value at zero energy.

    Parameters
    ----------
    w_n : real float array
            First matsubara frequencies to fit
    giw : real array
            Function to fit

    Returns
    -------
    Callable for inter - extrapolate function
    """
    n = len(w_n)
    gfit = np.squeeze(giw)[:n]
    pf = np.polyfit(w_n, gfit, 2)
    return np.poly1d(pf)
