# -*- coding: utf-8 -*-
r"""
==========
IPT Solver
==========

Within the iterative perturbative theory (IPT) the aim is to express the
self-energy of the impurity problem as

.. math:: \Sigma(\tau) \approx U^2 \mathcal{G}^0(\tau)^3

the contribution of the Hartree-term is not included here as it is cancelled

"""
from __future__ import division, absolute_import, print_function

from dmft.common import gt_fouriertrans, gw_invfouriertrans
import numpy as np


def solver(u_int, g_0_iwn, w_n, tau):

    g_0_tau = gw_invfouriertrans(g_0_iwn, tau, w_n)
    sigma_tau = u_int**2 * g_0_tau**3
    sigma_iwn = gt_fouriertrans(sigma_tau, tau, w_n, [u_int**2/4., 0., 0.])
    g_iwn = g_0_iwn / (1 - sigma_iwn * g_0_iwn)

    return g_iwn, sigma_iwn


def dmft_loop(u_int, t, g_iwn, w_n, tau, mix=1, conv=1e-3):
    """Performs the paramagnetic(spin degenerate) self-consistent loop in a
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
        g_iwn, sigma_iwn = solver(u_int, g_0_iwn, w_n, tau)
        converged = np.allclose(g_iwn_old, g_iwn, conv)
        loops += 1
        if loops > 300:
            converged = True
        g_iwn = mix * g_iwn + (1 - mix) * g_iwn_old
    return g_iwn, sigma_iwn
