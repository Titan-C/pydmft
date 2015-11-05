# -*- coding: utf-8 -*-
r"""
IPT Solver Single Band
======================

Within the iterative perturbative theory (IPT) the aim is to express the
self-energy of the impurity problem as

.. math:: \Sigma_\sigma(\tau) \approx U\mathcal{G}_\sigma^0(0)
    -U^2 \mathcal{G}_\sigma^0(\tau)
         \mathcal{G}_{\bar{\sigma}}^0(-\tau)
         \mathcal{G}_{\bar{\sigma}}^0(\tau)

The contribution of the Hartree-term is drooped because at half-filling
it cancels with the chemical potential. The next equalities are
obtained by using the anti-periodicity of the fermionic imaginary time
Green functions, and because of the half-filled case and particle hole
symmetry the last equality is fulfilled

.. math:: \mathcal{G}(-\tau) = -\mathcal{G}(-\tau+\beta)p = -\mathcal{G}(\tau)

As such for the single band paramagnetic case at half-filling the self
energy is estimated by

.. math:: \Sigma(\tau) \approx U^2 \mathcal{G}^0(\tau)^3

"""

from __future__ import division, absolute_import, print_function

from dmft.common import gt_fouriertrans, gw_invfouriertrans, greenF
import numpy as np


def single_band_ipt_solver(u_int, g_0_iwn, w_n, tau):
    r"""Given a Green function it returns a dressed one and the self-energy

    .. math:: \Sigma(\tau) \approx U^2 \mathcal{G}^0(\tau)^3

    .. math:: G = \mathcal{G}^0(i\omega_n)/(1 - \Sigma(i\omega_n)\mathcal{G}^0(i\omega_n))

    The Fourier transforms use as tail expansion of the atomic limit self-enegy

    .. math:: \Sigma(i\omega_n\rightarrow \infty) = \frac{U^2}{4i\omega_n}

    Parameters
    ----------
    u_int: float, local contact interaction
    g_0_iwn: complex 1D ndarray
        *bare* Green function, the Weiss field
    w_n: real 1D ndarray
        Matsubara frequencies
    tau: real 1D array
        Imaginary time points, not included edge point of :math:`\beta^-`
    """

    g_0_tau = gw_invfouriertrans(g_0_iwn, tau, w_n)
    sigma_tau = u_int**2 * g_0_tau**3
    sigma_iwn = gt_fouriertrans(sigma_tau, tau, w_n, [u_int**2/4., 0., 0.])
    g_iwn = g_0_iwn / (1 - sigma_iwn * g_0_iwn)

    return g_iwn, sigma_iwn


def dmft_loop(u_int, t, g_iwn, w_n, tau, mix=1, conv=1e-3):
    r"""Performs the paramagnetic(spin degenerate) self-consistent loop in a
    bethe lattice given the input

    Parameters
    ----------
    u_int : float
        Local interation strength
    t : float
        Hopping amplitude between bethe lattice nearest neightbours
    g_iwn : complex float ndarray
            Matsubara frequencies starting guess Green function
    tau : real float ndarray
            Imaginary time points. Only use the positive range
    mix : real :math:`\in [0, 1]`
            fraction of new solution for next input as bath Green function
    w_n : real float array
            fermionic matsubara frequencies. Only use the positive ones

    Returns
    -------
    out : complex ndarray
            Interacting Greens function in matsubara frequencies"""

    converged = False
    loops = 0
    iw_n = 1j*w_n
    while not converged:
        g_iwn_old = g_iwn.copy()
        g_0_iwn = 1. / (iw_n - t**2 * g_iwn_old)
        g_iwn, sigma_iwn = single_band_ipt_solver(u_int, g_0_iwn, w_n, tau)
        #Clean for Half-fill
        g_iwn.real = 0.
        converged = np.allclose(g_iwn_old, g_iwn, conv)
        loops += 1
        if loops > 500:
            converged = True
        g_iwn = mix * g_iwn + (1 - mix) * g_iwn_old
    return g_iwn, sigma_iwn
