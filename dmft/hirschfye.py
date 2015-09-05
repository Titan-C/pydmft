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
import scipy.linalg as la
from scipy.linalg.blas import dger
from scipy.interpolate import interp1d
from mpi4py import MPI
import math
import time

from dmft.common import tau_wn_setup, gw_invfouriertrans, greenF
import dmft.hffast as hffast

comm = MPI.COMM_WORLD

def ising_v(dtau, U, L, fields=1, polar=0.5):
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
    fields: integer
        Number of auxliary ising fields
    polar : float :math:`\\in (0, 1)`
        polarization threshold, probability of :math:`\\sigma_n=+ 1`
    Returns
    -------
    out : single dimension ndarray
    """
    lam = np.arccosh(np.exp(dtau*U/2))
    vis = np.ones((fields, L))
    rand = np.random.rand(fields, L)
    vis[rand > polar] = -1
    return vis*lam


def imp_solver(G0_blocks, v, interaction, parms_user):
    r"""Impurity solver call. Calcutaltes the interacting Green function
    as given by the contribution of the auxiliary discretized spin field.
    """

    # Set up default values
    parms = {'global_flip': False,
             'save_logs': False,
             'n_tau_mc':    64,
             'N_TAU':    2**11,
             'N_MATSUBARA': 64,
             't':           0.5,
             'MU':          0.,
             'SITES':       1,
             'loops':       1,
             'sweeps':      50000,
             'therm':       5000,
             'N_meas':      4,
             'Heat_bath':   True,
             }
    parms.update(parms_user)

    GX = [retarded_weiss(gb) for gb in G0_blocks]
    kroneker = np.eye(GX[0].shape[0])  # assuming all blocks are of same shape
    Gst = [np.zeros_like(gx) for gx in GX]

    i_pairs = np.array([c.nonzero() for c in interaction.T]).reshape(-1, 2)

    vlog = []
    ar = []

    acc, anrat = 0, 0
    bas, mult = math.modf(time.time())
    hffast.set_seed(int(bas+comm.Get_rank()*341*mult))

    for mcs in xrange(parms['sweeps'] + parms['therm']):
        if mcs % parms['therm'] == 0:
            if parms['global_flip']:
                v *= -1
            int_v = np.dot(interaction, v)
            g = [gnewclean(g_sp, lv, kroneker) for g_sp, lv in zip(GX, int_v)]

        for updates in range(parms['N_meas']):
            for i, (up, dw) in enumerate(i_pairs):
                acr, nrat = hffast.updateDHS(g[up], g[dw], v[i], parms['Heat_bath'])
                acc += acr
                anrat += nrat

        if mcs > parms['therm']:
            for i in range(interaction.shape[0]):
                Gst[i] += g[i]
            if parms['save_logs']:
                vlog.append(np.copy(v))
                ar.append(acc)

    tGst = np.asarray(Gst)
    Gst = np.zeros_like(Gst)
    comm.Allreduce(tGst, Gst)
    Gst /= parms['sweeps']*comm.Get_size()

    acc /= v.size*parms['N_meas']*(parms['sweeps'] + parms['therm'])
    print('acc ', acc, 'nsign', anrat)

    if parms['save_logs']:
        return [avg_g(gst, parms) for gst in Gst],\
                np.asarray(vlog), np.asarray(ar)
    else:
        return [avg_g(gst, parms) for gst in Gst]


def retarded_weiss(g0tau):
    r"""
    Takes the propagator :math:`\mathcal{G}^0(\tau)` corresponding to the
    Weiss mean field of the electronic bath and transforms it
    into the discretized matrix of the retarded weiss field as

    .. math:: \mathcal{G}^0_{\alpha\beta_{(ij)}} =
        -\mathcal{G}^0_{\alpha\beta}(i\Delta\tau - j\Delta\tau)
    Because of the Hirsch-Fye algorithm a minus sign is included into the
    matrix expresion. :math:`\alpha,\beta` block indices :math:`i,j` indices
    within the blocks

    Parameters
    ----------
    g0tau : 3D ndarray, of retarded weiss field
        First axis numerical values, second and third axis are block indices
    """
    lfak, n1, n2 = g0tau.shape
    delta_tau = np.arange(lfak)

    gind = lfak + np.subtract.outer(delta_tau, delta_tau)
    g0t_mat = np.empty((lfak*n1,lfak*n2))
    for i in range(n1):
        for j in range(n2):
            g0t_mat[i*lfak:(i+1)*lfak, j*lfak:(j+1)*lfak] = np.concatenate((g0tau[:, i, j], -g0tau[:, i, j]))[gind]
    return g0t_mat


def avg_gblock(gmat):
    """Averages along the diagonals respecting the translational invariance of
    the Greens Function"""

    lfak = gmat.shape[0]
    xga = np.zeros(2*lfak+1)
    for i in range(1, 2*lfak):
        xga[i] = np.trace(gmat, offset=lfak-i)

    xg = np.zeros(lfak+1)
    xg[:-1] = (xga[lfak:-1]-xga[:lfak]) / lfak
    xg[-1] = -xg[0]

    return xg

def avg_g(gst, parms):
    n1, n2, lfak = parms['SITES'], parms['SITES'], parms['n_tau_mc']

    gst_m = np.empty((n1, n2, lfak+1))
    for i in range(n1):
        for j in range(n2):
            gst_m[i, j] = avg_gblock(gst[i*lfak:(i+1)*lfak, j*lfak:(j+1)*lfak])
            if i == j:
                gst_m[i,j, -1] += 1.
    return gst_m


def gnewclean(g0t, v, kroneker):
    """Returns the interacting function :math:`G_{ij}` for the non-interacting
    propagator :math:`\\mathcal{G}^0_{ij}`
    .. math:: G_{ij} = B^{-1}_{ij}\\mathcal{G}^0_{ij}
    where
    .. math::
        u_j &= \\exp(v_j) - 1 \\\\
        B_{ij} &= \\delta_{ij} - u_j ( \\mathcal{G}^0_{ij} - \\delta_{ij})
    no sumation on :math:`j`
    for memory and speed the kroneker delta needs to be and input.
    the vector :math:`v_j` contains the effective Ising fields. For
    multiorbital systems it asumes that it is already the fields addition
    """
    u_j = np.exp(v) - 1.
    b = kroneker - u_j * (g0t-kroneker)

    return la.solve(b, g0t)

def gnew(g, dv, k):
    """Quick update of the interacting Green function matrix after a single
    spin flip of the auxiliary field. It calculates

    .. math:: \\alpha = \\frac{\\exp(v'_j - v_j) - 1}
                        {1 + (1 - G_{jj})(\\exp(v'_j v_j) - 1)}
    .. math:: G'_{ij} = G_{ij} + \\alpha (G_{ik} - \\delta_{ik})G_{kj}

    no sumation in the indexes"""
    ee = np.exp(dv)-1.
    a = ee/(1. + (1.-g[k, k])*ee)
    x = g[:, k].copy()
    x[k] -= 1
    y = g[k, :].copy()
    g = dger(a, x, y, 1, 1, g, 1, 1, 1)


def interpol(gt, Lrang):
    """This function interpolates :math:`G(\\tau)` to an even numbered anti
    periodic array for it to be directly used by the fourier transform into
    matsubara frequencies"""
    t = np.linspace(0, 1, gt.size)
    f = interp1d(t, gt)
    tf = np.linspace(0, 1, Lrang+1)
    return f(tf)


def interaction_matrix(bands):
    """Output the interaction matrix between all the spin species present
    in the given amount of bands. This matrix is use to connect the interacting
    spin densities that later on are decomposed by the Hubbard-Stratanovich
    transformation so output size is :math:`N\\times(2N-1)` where :math:`N` is
    the number of orbitals"""
    particles = 2 * bands
    fields = bands * (particles - 1)
    int_matrix = np.zeros((particles, fields))
    L = 0
    for i in range(particles):
        for j in range(i+1, particles):
            int_matrix[i, L] = 1
            int_matrix[j, L] = -1
            L += 1
    return int_matrix


def setup_PM_sim(u_parms):
        # Set up default values
    parms = {'global_flip': False,
             'save_logs': False,
             'n_tau_mc':    64,
             'N_TAU':    2**11,
             'N_MATSUBARA': 64,
             't':           0.5,
             'MU':          0.,
             'BANDS':       1,
             'SITES':       1,
             'loops':       1,
             'sweeps':      50000,
             'therm':       5000,
             'N_meas':      4,
             'updater':     'discrete'
             }
    parms.update(u_parms)
    tau, w_n = tau_wn_setup(parms)
    gw = greenF(w_n, mu=parms['MU'], D=2*parms['t'])
    gt = gw_invfouriertrans(gw, tau, w_n)
    gt = interpol(gt, parms['n_tau_mc'])
    parms['dtau_mc'] = parms['BETA']/parms['n_tau_mc']
    intm = interaction_matrix(parms['BANDS'])
    v = ising_v(parms['dtau_mc'], parms['U'], parms['n_tau_mc']*parms['SITES'], intm.shape[1])

    return tau, w_n, gt, gw, v, intm
