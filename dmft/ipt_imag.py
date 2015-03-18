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



def solver(u_int, g_0_iwn, iwn, tau):

    g_0_tau = gw_invfouriertrans(g_0_iwn, tau, iwn)
    sigma_tau = u_int**2 * g_0_tau**3
    sigma_iwn = gt_fouriertrans(sigma_tau, tau, iwn)
    g_iwn = g_0_iwn / (1 - sigma_iwn * g_0_iwn)

    return g_iwn


def dmft_loop(loops, u_int, t, g_iwn, iwn, tau):
    """Performs the paramagnetic(spin degenerate) self-consistent loop in a
    bethe lattice given the input

    Parameters
    ----------
    loops : int
        amount of dmft loops
    u_int : float
        Local interation strength
    t : float
        Hopping amplitudue between bethe lattice nearest neightbours
    g_iwn : comples float ndarray
            Matsubara frequencies starting guess Green function
    tau : real float ndarray
            Imaginary time points. Only use the positive range
    iwn : complex float ndarray
            fermionic matsubara frequencies. Only use the positive ones

    Returns
    -------
    out : complex ndarray
            Interacting Greens function in matsubara frequencies in every
            loop step"""

    g_iwn_log = []
    for i in range(loops):
        g_0_iwn = 1. / (iwn - t**2 * g_iwn)
        g_iwn = solver(u_int, g_0_iwn, iwn, tau)
        g_iwn_log.append(g_iwn)
    return g_iwn_log
