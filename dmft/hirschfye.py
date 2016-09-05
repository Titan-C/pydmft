# -*- coding: utf-8 -*-
"""
QMC Hirsch - Fye Impurity solver
================================

To treat the Anderson impurity model and solve it using the Hirsch - Fye
Quantum Monte Carlo algorithm
"""

from __future__ import division, absolute_import, print_function
import argparse
import os
import struct
import time
from itertools import combinations, product
from math import exp

from mpi4py import MPI
from scipy.linalg.blas import dger, zgeru
import scipy.linalg as la
import numba
import numpy as np

from dmft.common import tau_wn_setup, gw_invfouriertrans, greenF
import dmft.hffast as hffast


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
    fields : integer
        Number of auxliary ising fields
    polar : float :math:`\\in (0, 1)`
        polarization threshold, probability of :math:`\\sigma_n=+ 1`

    Returns
    -------
    out : single dimension ndarray
    """
    lam = np.arccosh(np.exp(dtau * U / 2))
    vis = np.ones((fields, L))
    rand = np.random.rand(fields, L)
    vis[rand > polar] = -1
    return vis * lam


def imp_solver(g0_blocks, v, interaction, parms_user):
    r"""Impurity solver call. Calcutaltes the interacting Green function
    as given by the contribution of the auxiliary discretized spin field.
    """

    comm = MPI.COMM_WORLD
    # Set up default values
    parms = {'global_flip': False,
             'binned_meas': False,
             'double_flip_prob': 0.,
             't':           0.5,
             'SITES':       1,
             'BANDS':       1,
             'SEED':        struct.unpack("I", os.urandom(4))[0],
             'Heat_bath':   True,
             'ofile':       'hf_out.h5',
             'group':       'temp/' + time.asctime(),
             }
    parms.update(parms_user)
    if not os.path.exists(parms_user['work_dir']) and comm.rank == 0:
        os.makedirs(parms_user['work_dir'])

    # Retarded field includes the Hirsh-Fye minus sign in GF
    GX = [retarded_weiss(gb) for gb in g0_blocks]
    kroneker = np.eye(GX[0].shape[0])  # assuming all blocks are of same shape
    Gst = [np.zeros_like(gx) for gx in GX]
    Gbin = [np.zeros_like(gx) for gx in GX]

    i_pairs = np.array([c.nonzero() for c in interaction.T]).reshape(-1, 2)

    vlog = []
    ar = []

    acc, anrat = 0, 0
    flavor_pairs = list(combinations(
        product(range(2), range(parms['SITES'])), 2))
    double_occ = np.zeros(len(flavor_pairs))

    flavors = 2 * parms['BANDS'] * parms['SITES']
    flavors_ind = list(product(range(2), range(parms['SITES'])))
    occupation = np.zeros(flavors)
    double_occ = np.zeros(len(flavor_pairs))
    ntau = 2 * parms['N_MATSUBARA']
    chi = np.zeros(ntau)
    hffast.set_seed(parms['SEED'])

    update = False
    for mcs in range(parms['sweeps'] + parms['therm']):
        if mcs % parms['therm'] == 0 and parms['global_flip']:
            v *= -1
            update = True
        if mcs % 500 == 0 or update:  # dirty update clean up
            int_v = np.dot(interaction, v)
            g = [gnewclean(g_sp, lv, kroneker) for g_sp, lv in zip(GX, int_v)]
            update = False

        for _ in range(parms['meas']):
            for i, (up, dw) in enumerate(i_pairs):
                acr, nrat = hffast.updateDHS(g[up], g[dw], v[i], ntau,
                                             parms['double_flip_prob'],
                                             ['Heat_bath'])
                acc += acr
                anrat += nrat

        if mcs > parms['therm']:
            for i in range(interaction.shape[0]):
                Gbin[i] += g[i]
                Gst[i] += g[i]
            if mcs % parms['therm'] == 0 and parms['binned_meas']:
                Gbin = np.array(Gbin) / parms['therm']
                np.save(parms['work_dir'] + '/gtau_bin_mcs{}_r{}'.format(mcs,
                                                                         comm.rank),
                        np.squeeze([-1 * avg_g(gst, parms) for gst in Gbin]))
                Gbin = [np.zeros_like(gx) for gx in GX]

            orbital_occupation(g, occupation, ntau, flavors_ind)
            double_occupation(g, double_occ, ntau, flavor_pairs)
            if parms['save_logs']:
                vlog.append(v > 0)
                ar.append(acr)

    tGst = np.asarray(Gst)
    Gst = np.zeros_like(tGst)
    comm.Allreduce(tGst, Gst)
    Gst /= parms['sweeps'] * comm.Get_size()

    acc /= v.size * parms['meas'] * (parms['sweeps'] + parms['therm'])
    occupation /= 2 * parms['N_MATSUBARA'] * parms['sweeps']
    double_occ /= 2 * parms['N_MATSUBARA'] * parms['sweeps']

    print('occ', occupation)
    print('docc', double_occ, 'acc ', acc, 'nsign', anrat, 'rank', comm.rank)

    comm.Allreduce(occupation.copy(), occupation)
    comm.Allreduce(double_occ.copy(), double_occ)
    comm.Allreduce(chi.copy(), chi)

    if comm.rank == 0:
        save_output(parms, occupation / comm.Get_size(),
                    double_occ / comm.Get_size(), acc, chi, vlog, ar)

    # Recover Conventional GF sign in average
    return [-1 * avg_g(gst, parms) for gst in Gst]


@numba.jit(nopython=True)
def measure_chi(v, slices):
    """Estimates the susceptibility from the Ising auxiliary fields"""
    chi = np.zeros(slices)
    s = np.sign(v)
    for i in range(slices):
        for j in range(slices):
            k = i + j
            if k > slices:
                k -= slices  # Ising fields are bosonic and have PBC
            chi[i] += s[k] * s[j]

    return chi


def orbital_occupation(g, occupation, slices, flavors_ind):
    """Calculates the orbital occupation

    It is relevant to remember that under this Hirsh-Fye simulation
    the signs of the Green functions are reversed and that after this
    observable is calculated g(0+) one has to apply n= 1-g(0+) """

    for k, (spin_i, site_i) in enumerate(flavors_ind):
        g_i = g[spin_i][
            site_i * slices:(site_i + 1) * slices, site_i * slices:(site_i + 1) * slices]
        occupation[k] += np.trace(g_i)


def double_occupation(g, double_occ, slices, flavor_pairs):
    """Calculates the density-density correlator between spin flavors"""
    for k, (i, j) in enumerate(flavor_pairs):
        spin_i, site_i = i
        g_i = g[spin_i][
            site_i * slices:(site_i + 1) * slices, site_i * slices:(site_i + 1) * slices]
        spin_j, site_j = j
        g_j = g[spin_j][
            site_j * slices:(site_j + 1) * slices, site_j * slices:(site_j + 1) * slices]
        double_occ[k] += np.einsum('ii,ii', g_i, g_j)


def susceptibility(v):
    """Calculate the susceptibility from the Ising fields"""
    pass


def save_output(params, occupation, double_occ, acceptance, chi, vlog, ar):
    """Saves the simulation status"""
    if not os.path.exists(params['work_dir']):
        os.makedirs(params['work_dir'])
    np.save(params['work_dir'] + '/occupation', occupation)
    np.save(params['work_dir'] + '/double_occ', double_occ)
    np.save(params['work_dir'] + '/acceptance', acceptance)
    np.save(params['work_dir'] + '/chi', chi)

    if params['save_logs']:
        np.save(params['work_dir'] + '/v_ising', np.asarray(vlog))
        np.save(params['work_dir'] + '/acceptance_log', np.asarray(ar))


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
    g0t_shape = g0tau.shape
    if len(g0t_shape) > 1:
        n1, n2, slices = g0t_shape
    else:
        n1, n2, slices = 1, 1, g0t_shape[0]
        g0tau = g0tau.reshape(1, 1, -1)

    delta_tau = np.arange(slices)

    gind = slices + np.subtract.outer(delta_tau, delta_tau)
    g0t_mat = np.empty((slices * n1, slices * n2), g0tau.dtype)

    for i in range(n1):
        for j in range(n2):
            g0t_mat[i * slices:(i + 1) * slices,
                    j * slices:(j + 1) * slices] = np.concatenate(
                        (g0tau[i, j], -g0tau[i, j]))[gind]
    return g0t_mat


def avg_gblock(gmat):
    """Averages along the diagonals respecting the translational invariance of
    the Greens Function"""

    slices = gmat.shape[0]
    xga = np.zeros(2 * slices, gmat.dtype)
    for i in range(2 * slices):
        xga[i] = np.trace(gmat, offset=slices - i)

    xg = (xga[slices:] - xga[:slices]) / slices

    return xg


def avg_g(gst, parms):
    n1, n2, slices = parms['SITES'], parms['SITES'], parms['N_MATSUBARA'] * 2

    gst_m = np.empty((n1, n2, slices), gst.dtype)
    for i in range(n1):
        for j in range(n2):
            gst_m[i, j] = avg_gblock(gst[i * slices:(i + 1) * slices,
                                         j * slices:(j + 1) * slices])
    return gst_m


def gnewclean(g0t, v, kroneker):
    """Returns the interacting function :math:`G_{ij}` for the non-interacting
    propagator :math:`\\mathcal{G}^0_{ij}`

    .. math:: G_{ij} = B^{-1}_{ij}\\mathcal{G}^0_{ij}

    where

    .. math:: u_j &= \\exp(v_j) - 1 \\\\
        B_{ij} &= \\delta_{ij} - u_j ( \\mathcal{G}^0_{ij} - \\delta_{ij})

    no sumation on :math:`j`
    for memory and speed the kroneker delta needs to be and input.
    the vector :math:`v_j` contains the effective Ising fields. For
    multiorbital systems it asumes that it is already the fields addition
    """
    u_j = np.exp(v) - 1.
    b = kroneker - u_j * (g0t - kroneker)

    return la.solve(b, g0t)


def gnew(g, dv, k):
    """Quick update of the interacting Green function matrix after a single
    spin flip of the auxiliary field. It calculates

    .. math:: \\alpha = \\frac{\\exp(v'_j - v_j) - 1}
                        {1 + (1 - G_{jj})(\\exp(v'_j v_j) - 1)}
    .. math:: G'_{ij} = G_{ij} + \\alpha (G_{ik} - \\delta_{ik})G_{kj}

    no sumation in the indexes"""
    ee = exp(dv) - 1.
    a = ee / (1. + (1. - g[k, k]) * ee)
    x = g[:, k].copy()
    x[k] -= 1
    y = g[k, :].copy()
    if np.issubdtype(g.dtype, np.float):
        g = dger(a, x, y, 1, 1, g, 1, 1, 1)
    if np.issubdtype(g.dtype, np.complex128):
        g = zgeru(a, x, y, 1, 1, g, 1, 1, 1)


def g2flip(g, dv, l, k):
    r"""Update the interacting Green function at arbitrary spinflips

    Using the Woodbury matrix identity it is possible to perform an
    update of two simultaneous spin flips. I calculate

    .. math:: G'_{ij} = G_{ij}
              - U_{if}(\delta_{fg} + U_{\bar{k}g})^{-1}G_{\bar{l}j}

    where :math:`i,j,l,k\in{1..L}` the time slices and :math:`f,g\in{1,2}`
    the 2 simultaneous spin flips. :math:`\bar{l}` lists the flipped
    spin entry

    .. math:: U_{if} = (\delta_{i\bar{l}} - G_{i\bar{l}})(\exp(-2V_{\bar{l}} )-1)\delta_{\bar{l}f}

"""
    lk = [l, k]
    d2 = np.eye(len(lk))
    U = g[:, lk].copy()
    np.add.at(U, lk, -d2)
    U *= np.exp(dv) - 1.

    V = g[lk, :].copy()
    denom = la.solve(U[lk, :] - d2, V)

    g -= np.dot(U, denom)


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
        for j in range(i + 1, particles):
            int_matrix[i, L] = 1
            int_matrix[j, L] = -1
            L += 1
    return int_matrix


def setup_PM_sim(parms):
    """Setup the default state for a Paramagnetic simulation"""
    tau, w_n = tau_wn_setup(parms)
    giw = greenF(w_n, mu=parms['MU'], D=2 * parms['t'])
    gtau = gw_invfouriertrans(giw, tau, w_n)
    parms['dtau_mc'] = tau[1]
    intm = interaction_matrix(parms.get('BANDS', 1))
    v = ising_v(parms['dtau_mc'], parms['U'], len(tau) * parms['SITES'],
                intm.shape[1], parms['spin_polarization'])

    return tau, w_n, gtau, giw, v, intm


def do_input(help_string):
    """Prepares the input parser for the simulation at hand

    Parameters
    ----------
    help_string : Title of the execution script

    Returns
    -------
    parse : Argument Parser object
    """

    parser = argparse.ArgumentParser(description=help_string,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-BETA', metavar='B', type=float,
                        default=32., help='The inverse temperature')
    parser.add_argument('-n_freq', '--N_MATSUBARA', metavar='B', type=int,
                        default=32, help='Number of Matsubara frequencies. '
                        'This influences the Trotter slicing as dtau=beta/2/n_freq')
    parser.add_argument('-sweeps', metavar='MCS', type=int, default=int(1e4),
                        help='Number Monte Carlo Measurement')
    parser.add_argument('-therm', type=int, default=int(5e3),
                        help='Monte Carlo sweeps of thermalization')
    parser.add_argument('-meas', type=int, default=3,
                        help='Number of Updates before measurements')
    parser.add_argument('-Niter', metavar='N', type=int,
                        default=20, help='Number of iterations')
    parser.add_argument('-U', '--urange', nargs='+', type=float, default=[2.5],
                        help='Local interaction strength')
    parser.add_argument('-mu', '--MU', type=float, default=0.,
                        help='Chemical potential')
    parser.add_argument('-ofile', default='SB_{simt}_B{BETA}',
                        help='Output file shelve')

    parser.add_argument('-l', '--save_logs', action='store_true',
                        help='Store the changes in the auxiliary field')
    parser.add_argument('-bin', '--binned_meas', action='store_true',
                        help='Save measurements in binned intervals every therm updates')
    parser.add_argument('-spin_polarization', type=float, default=0.5,
                        help='Probability distribution of up/down'
                        'auxiliary spins for initial guess')
    parser.add_argument('-gf', '--global_flip', action='store_true',
                        help='Perform a global flip of the auxiliary'
                        'Ising spins with a period equal to the'
                        'thermalization steps ')
    parser.add_argument('-M', '--Heat_bath', action='store_false',
                        help='Use Metropolis importance sampling')
    return parser
