# -*- coding: utf-8 -*-
"""
================================
QMC Hirsch - Fye Impurity solver
================================

To treat the Anderson impurity model and solve it using the Hirsch - Fye
Quantum Monte Carlo algorithm
"""
from __future__ import division, absolute_import, print_function
import numpy as np
from scipy.linalg import solve
from scipy.linalg.blas import dger
from scipy.interpolate import interp1d

from dmft.common import tau_wn_setup, gw_invfouriertrans, greenF


def ising_v(dtau, U, L=32, polar=0.5):
    """initialize the vector V of Ising fields

    .. math:: V = \\lambda (\\sigma_1, \\sigma_2, \\cdots, \\sigma_L)

    where the vector entries :math:`\\sigma_n=\\pm 1` are randomized subject
    to a threshold given by polar. And

    .. math:: \\cosh(\\lambda) = \\exp(\\Delta \\tau \\frac{U}{2})

    Parameters
    ----------
    dtau : float
        time spacing :math::`\\Delta\\Tau`
    U : float
        local Coulomb repulsion
    L : integer
        length of the array
    polar : float :math:`\\in (0, 1)`
        polarization threshold, probability of :math:`\\sigma_n=+ 1`

    Returns
    -------
    out : single dimension ndarray
    """
    lam = np.arccosh(np.exp(dtau*U/2))
    vis = np.ones(L)
    rand = np.random.rand(L)
    vis[rand > polar] = -1
    return vis*lam


def imp_solver(g0, v, sweeps):
    r"""Impurity solver call. Calcutaltes the interacting Green function
    as given by the contribution of the auxiliary discretized spin field.
    """

    gx = ret_weiss(g0)
    gup = gnewclean(gx, v, 1.)
    gdw = gnewclean(gx, v, -1.)

    gstup, gstdw = mcs(sweeps, 500, gup, gdw, v)

    return avg_g(gstup), avg_g(gstdw)


def ret_weiss(g0tau):
    r"""
    Takes the propagator :math:`\mathcal{G}^0(\tau)` and transforms it
    into the discretized matrix of the retarded weiss field as

    .. math:: \mathcal{G}^0_{ij} = \mathcal{G}^0(i\Delta\tau - j\Delta\tau)
    Because of the Hirsch-Fye algorithm a minus sign is included
    """
    lfak = g0tau.shape[-1]-1

    gind = lfak + np.arange(lfak).reshape(-1, 1)-np.arange(lfak).reshape(1, -1)
    return np.concatenate((g0tau[:-1], -g0tau))[gind]

def avg_g(gmat):
    lfak = gmat.shape[0]
    xga = np.zeros(2*lfak+1)
    for i in range(1, 2*lfak):
        xga[i] = np.trace(gmat, offset=lfak-i)

    xg = np.zeros(lfak+1)
    xg[:-1] = (xga[lfak:-1]-xga[:lfak]) / lfak
    xg[-1] = 1-xg[0]

    return xg

def mcs(sweeps, therm, gup, gdw, v):
    lfak = v.size
    gstup, gstdw = np.zeros((lfak, lfak)), np.zeros((lfak, lfak))
    kroneker = np.eye(lfak)

    for mcs in range(sweeps+therm):
        for j in range(lfak):
            dv = 2.*v[j]
            ratup = 1. + (1. - gup[j, j])*(np.exp(-dv)-1.)
            ratdw = 1. + (1. - gdw[j, j])*(np.exp( dv)-1.)
            rat = ratup * ratdw
            rat = rat/(1.+rat)
            if rat > np.random.rand():
                v[j] *= -1.
                gup = gnew(gup, v[j], j, 1.)
                gdw = gnew(gdw, v[j], j, -1.)

        if mcs > therm:

            gstup += gup
            gstdw += gdw

    gstup = gstup/sweeps
    gstdw = gstdw/sweeps

    return gstup, gstdw


def gnewclean(g0t, v, sign):
    """Returns the interacting function :math:`G_{ij}` for the non-interacting
    propagator :math:`\\mathcal{G}^0_{ij}`

    .. math:: G_{ij} = B^{-1}_{ij}\\mathcal{G}^0_{ij}

    where

    .. math::
        u_j &= \\exp(\\sigma v_j) - 1 \\\\
        B_{ij} &= \\delta_{ij} - u_j ( \\mathcal{G}^0_{ij} - \\delta_{ij})

    no sumation on :math:`j`
    """
    u_j = np.exp(sign*v) - 1.
    ide = np.eye(v.size)
    b = ide - u_j * (g0t-ide)

    return solve(b, g0t)

def gnew(g, v, k, sign):
    """Quick update of the interacting Green function matrix after a single
    spin flip of the auxiliary field. It calculates

    .. math:: \\alpha = \\frac{\\exp(2\\sigma v_j) - 1}
                        {1 + (1 - G_{jj})(\\exp(2\\sigma v_j) - 1)}
    .. math:: G'_{ij} = G_{ij} + \\alpha (G_{ik} - \\delta_{ik})G_{kj}

    no sumation in the indexes"""
    dv = sign*v*2
    ee = np.exp(dv)-1.
    a = ee/(1. + (1.-g[k, k])*ee)
    x = g[:, k].copy()
    x[k] -= 1
    y = g[k, :].copy()

    return dger(a, x, y, 1, 1, g, 1, 1, 1)


def interpol(gt, Lrang):
    """This function interpolates :math:`G(\\tau)` to an even numbered anti
    periodic array for it to be directly used by the fourier transform into
    matsubara frequencies"""
    t = np.linspace(0, 1, gt.size)
    f = interp1d(t, gt)
    tf = np.linspace(0, 1, Lrang+1)
    return f(tf)


def setup_PM_sim(parms):
    tau, w_n = tau_wn_setup(parms)
    gw = greenF(w_n, mu=parms['MU'])
    gt = gw_invfouriertrans(gw, tau, w_n)
    gt = interpol(gt, parms['n_tau_mc'])
    parms['dtau_mc'] = parms['BETA']/parms['n_tau_mc']
    v = ising_v(parms['dtau_mc'], parms['U'], L=parms['n_tau_mc'])

    return tau, w_n, gt, gw, v
