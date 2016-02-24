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

from numba import jit
from scipy.integrate import quad, simps
from scipy.optimize import fsolve
import numpy as np
from dmft.common import gt_fouriertrans, gw_invfouriertrans
import slaveparticles.quantum.dos as dos


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

    g_0_tau = gw_invfouriertrans(g_0_iwn, tau, w_n, [1., 0., 0.25])
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
        Hopping amplitude between bethe lattice nearest neighbors
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
    tuple 2 complex ndarrays
            Interacting Green's function in matsubara frequencies
            Self energy
    """

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

###############################################################################
# Energy calculations

def ekin(g_iw, s_iw, beta, w_n, ek_mean, g_iwfree):
    """Calculates the Kinetic Energy
    """
    return 2*(1j*w_n*(g_iw-g_iwfree) - s_iw*g_iw).real.sum()/beta + ek_mean

def ekin_tau(g_iw, tau, w_n, u_int):
    gt = gw_invfouriertrans(g_iw, tau, w_n, [1., 0., u_int**2/4 + 0.25])
    return -0.5*simps(gt*gt, tau)


def epot(g_iw, s_iw, u, beta, w_n):
    r"""Calculates the Potential Energy

    Using the local Green Function and self energy the potential
    energy is calculated using :ref:`potential_energy`

    Taking the tail of this product to decay in the half-filled single
    band case as:

    .. math:: \Sigma G \rightarrow \frac{U^2}{4i\omega_n^2}

    The potential energy per spin is calculated by

    .. math:: \langle V \rangle = \frac{1}{\beta} \sum_{n} \frac{1}{2} (\Sigma(i\omega_n)G(i\omega_n) - \frac{U^2}{4i\omega_n^2} + \frac{U^2}{4i\omega_n^2})
    .. math:: = \frac{1}{\beta} \sum_{n>0} \Re e (\Sigma(i\omega_n)G(i\omega_n) - \frac{U^2}{4i\omega_n^2}) + \frac{U^2}{8\beta} \sum_{n} \frac{1}{i\omega_n^2}
    .. math:: = \frac{1}{\beta} \sum_{n>0} \Re e (\Sigma(i\omega_n)G(i\omega_n) - \frac{U^2}{4i\omega_n^2}) - \frac{U^2}{32}

    """
    # the last u/8 is because sigma to zero order has the Hartree term
    # that is avoided in IPT Sigma=U/2. Then times G->1/2 after sum
    # and times 1/2 of the formula
    return (s_iw*g_iw+u**2/4./w_n**2).real.sum()/beta - beta*u**2/32. + u/8


def n_half(mu, beta, D=1):
    """Returns the deviation from half-filling in semicircle dos"""
    return quad(dos.bethe_fermi, -D, D, args=(1., mu, D/2., beta))[0]-0.5


def e_mean(beta, D=1):
    """Returns the average kinetic Energy per particle in semicircle dos"""
    mu = fsolve(n_half, 0., beta)[0]
    return quad(dos.bethe_fermi_ene, -D, D, args=(1., mu, D/2., beta))[0]


@jit(nopython=True)
def _dimer_sigma(g0t_d, g0t_o, U):
    return g0t_d*g0t_d*g0t_d*U*U, -g0t_o*g0t_o*g0t_o*U*U

def dimer_sigma(u_int, tp, g0iw_d, g0iw_o, tau, w_n):
    r"""Given a Green function it returns a dressed one and the self-energy

    .. math:: \Sigma(\tau) \approx U^2 \mathcal{G}^0(\tau)^3

    .. math:: G = \mathcal{G}^0(i\omega_n)/(1 - \Sigma(i\omega_n)\mathcal{G}^0(i\omega_n))

    The Fourier transforms use as tail expansion of the atomic limit self-enegy

    .. math:: \Sigma(i\omega_n\rightarrow \infty) = \frac{U^2}{4i\omega_n}

    Parameters
    ----------
    u_int : float, local contact interaction
    tp : float, dimer hybridization strength
    g0iw_d : complex 1D ndarray
        *bare* local Green function, the Weiss field
    g0iw_o : complex 1D ndarray
        *bare* hybridizing Green function, the Weiss field
    tau: real 1D array
        Imaginary time points, not included edge point of :math:`\beta^-`
    w_n: real 1D ndarray
        Matsubara frequencies
    """

    g0t_d = gw_invfouriertrans(g0iw_d, tau, w_n, [1., 0., tp**2 + 0.25])
    g0t_o = gw_invfouriertrans(g0iw_o, tau, w_n, [0., tp, 0.])

    st_d, st_o = _dimer_sigma(g0t_d, g0t_o, u_int)

    sw_d = gt_fouriertrans(st_d, tau, w_n, [u_int**2/4, 0., u_int**2/4+4*tp])
    sw_o = gt_fouriertrans(st_o, tau, w_n, [0., -u_int**2/2*tp, 0.])

    return sw_d, sw_o
