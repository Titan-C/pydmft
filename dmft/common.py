# -*- coding: utf-8 -*-
"""
Green Functions
===============

Interface to treat arrays as the Green functions. Deals with their
Fourier Transforms from Matsubara frequencies to Imaginary time.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
from numpy.fft import fft, ifft
from scipy.linalg import lstsq


def matsubara_freq(beta=16., size=256, fer=1):
    r"""Calculates an array containing the matsubara frequencies under the
    formula

    .. math:: \omega_n = \frac{\pi(2n + f)}{\beta}

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
    real ndarray
    """

    return np.pi * (fer + 2 * np.arange(size)) / beta


def tau_wn_setup(parms):
    """return two numpy arrays one corresponding to the imaginary time array
    and the other to the matsubara frequencies. The time array is twice as
    dense for best results in the Fast Fourier Transform.

    Parameters
    ----------
    parms : dictionary
        with keywords BETA, N_MATSUBARA

    Returns
    -------
    tuple (tau real ndarray, w_n real ndarray)
    """

    w_n = matsubara_freq(parms['BETA'], parms['N_MATSUBARA'])
    tau = np.arange(0, parms['BETA'], parms['BETA'] / 2 / len(w_n))

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
    complex ndarray
            Interacting Greens function in matsubara frequencies, all odd
            entries are zeros
    """
    zeta = 1.j * w_n + mu - sigma
    sq = np.sqrt((zeta)**2 - D**2)
    sig = np.sign(sq.imag * w_n)
    return 2. / (zeta + sig * sq)


def semi_circle_hiltrans(zeta, D=1):
    """Calculate the Hilbert transform with a semicircular DOS

    See also
    --------
    greenF
    """
    sqr = np.sqrt(zeta**2 - D**2)
    sqr = np.sign(sqr.imag) * sqr
    return 2 * (zeta - sqr) / D**2


def gt_fouriertrans(g_tau, tau, w_n, tail_coef=(1., 0., 0.)):
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

    beta = tau[1] + tau[-1]
    freq_tail, time_tail = freq_tail_fourier(tail_coef, beta, tau, w_n)

    gtau = g_tau - time_tail
    return beta * ifft(gtau * np.exp(1j * np.pi * tau / beta))[..., :len(w_n)] + freq_tail


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


def gw_invfouriertrans(g_iwn, tau, w_n, tail_coef=(1., 0., 0.)):
    r"""Performs an inverse fourier transform of the green Function in which
    only the imaginary positive matsubara frequencies
    :math:`\omega_n= \pi(2n+1)/\beta` with :math:`n \in \mathbb{N}` are used.
    The high frequency tails are transformed analytically up to the third moment.

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

    g_tau = fft(giwn, len(tau)) * np.exp(-1j * np.pi * tau / beta)

    return (g_tau * 2 / beta).real + time_tail


def tail(w_n, coef, powers):
    return np.sum([c / w_n**p for c, p in zip(coef, powers)], 0)


def lin_tail_fit(w_n, inp_gf, x, span=30, negative_freq=False):
    """Perform a Least squares fit to the tail of Green function

    the fit is done in inp_gf[-x:-x + span]

    Parameters:
        w_n (real 1D ndarray) : Matsubara frequencies
        inp_gf (complex 1D ndarray) : Green function to fit
        x (int) : counting from last element from where to do the fit
        span (int) : amount of frequencies to do the fit over
        negative_freq (bool) : Array has negative Matsubara frequencies
    Returns:
        complex 1D ndarray : Tail patched Green function (copy of origal)
        real 1D ndarray : The First 3 moments
"""
    def tail_coef(w_n, data, powers):
        A = np.array([1 / w_n**p for p in powers]).T
        return lstsq(A, data)[0]

    tw_n = w_n[-x:-x + span].copy()
    datar = inp_gf[-x:-x + span].real.copy()
    datai = inp_gf[-x:-x + span].imag.copy()

    re_c = tail_coef(tw_n, datar, [2])
    re_tail = tail(w_n, re_c, [2])
    im_c = tail_coef(tw_n, datai, [1, 3])
    im_tail = tail(w_n, im_c, [1, 3])

    f_tail = re_tail + 1j * im_tail

    patgf = inp_gf.copy()
    patgf[-x:] = f_tail[-x:]
    if negative_freq:
        patgf[:x] = f_tail[:x]

    return patgf, np.array([im_c[0], re_c[0], im_c[1]])


def fit_gf(w_n, giw, p=2):
    """Performs a quadratic fit of the *first's* matsubara frequencies
    to estimate the value at zero energy.

    Parameters
    ----------
    w_n : real float array
            First's matsubara frequencies to fit
    giw : real array
            Function to fit

    Returns
    -------
    Callable for inter - extrapolate function
    """
    gfit = np.squeeze(giw)[:len(w_n)]
    pf = np.polyfit(w_n, gfit, p)
    return np.poly1d(pf)


# Pade Analytical Continuation
# Algorithm from Vidberg & Serene J. Low Temperature Phys. 29, 3-4, 179 (1977)
def pade_coefficients(g_iw, w_n):
    """Find the Pade coefficients for the desired green Function

    Parameters
    ----------
    g_iw : complex ndarray
        single axis
    w_n : real ndarray
        Matsubara frequencies
    """
    G = np.zeros((len(g_iw), len(g_iw)), dtype=np.complex)
    G[0] = g_iw
    for i in range(1, len(g_iw)):
        G[i, i:] = 1j * (G[i - 1, i - 1] / G[i - 1, i:] -
                         1.) / (w_n[i - 1] - w_n[i:])

    return np.diag(G)


def pade_rec(pc, w, w_n):
    """Pade recursion formula for continued Fractions

    Parameters
    ----------
    pc : complex ndarray
        pade coefficients
    w : real ndarray
        real frequencies
    w_n : real ndarray
        Matsubara frequencies
    """
    an_1 = 0.
    an = pc[0]
    bn = 1.
    bn_1 = 1.
    iw_n = 1j * w_n
    for i in range(len(pc) - 2):
        anp = an + (w - iw_n[i]) * pc[i + 1] * an_1
        bnp = bn + (w - iw_n[i]) * pc[i + 1] * bn_1
        an_1, an = an, anp
        bn_1, bn = bn, bnp
    return an / bn


def pade_continuation(gfunc, w_n, w, w_set=None):
    """Continate the green Function by Padé

    Parameters
    ----------
    gfunc: complex 1D ndarray
        Green function to be continued
    w_n: real 1D ndarray
        Matsubara frequencies
    w: real 1D ndarray
        real frequency array
    w_set: int or int 1D ndarray
        Amount of frequency point to sample
        index of points to sample for continuation

    See also
    --------
    pade_coefficients
    pade_rec
    """

    if w_set is None:
        w_set = np.arange(len(w_n))
    elif isinstance(w_set, int):
        w_set = np.arange(w_set)

    pc = pade_coefficients(gfunc[w_set], w_n[w_set])
    g_real = pade_rec(pc, w, w_n[w_set])

    return g_real
