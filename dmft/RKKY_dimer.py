# -*- coding: utf-8 -*-
"""
Dimer Bethe lattice
===================

Non interacting dimer of a Bethe lattice
Based on the work G. Moeller et all PRB 59, 10, 6846 (1999)
"""

from __future__ import division, print_function, absolute_import
from dmft.twosite import matsubara_Z
from scipy.integrate import quad
from scipy.optimize import fsolve
import dmft.common as gf
import dmft.h5archive as h5
import dmft.ipt_imag as ipt
import matplotlib.pyplot as plt
import numpy as np
import slaveparticles.quantum.dos as dos
from slaveparticles.quantum import fermion


# Molecule
def sorted_basis():
    """Sorts the basis of states for the electrons in the molecule
    Enforces ordering in particle number an Sz projection"""

    ind = np.array([0, 1, 2, 4, 8, 5, 10, 6, 9, 12, 3, 7, 11, 13, 14, 15])
    basis = [fermion.destruct(4, sigma)[ind][:, ind] for sigma in range(4)]
    return basis


def dimer_hamiltonian(U, mu, tp):
    r"""Generate a single orbital isolated atom Hamiltonian in particle-hole
    symmetry. Include chemical potential for grand Canonical calculations

    .. math::
        \mathcal{H} - \mu N =
        -\frac{U}{2}(n_{a\uparrow} - n_{a\downarrow})^2
        -\frac{U}{2}(n_{b\uparrow} - n_{b\downarrow})^2  +
        t_\perp (a^\dagger_\uparrow b_\uparrow +
                 b^\dagger_\uparrow a_\uparrow +
                 a^\dagger_\downarrow b_\downarrow +
                 b^\dagger_\downarrow a_\downarrow)

        - \mu(n_{a\uparrow} + n_{a\downarrow})
        - \mu(n_{b\uparrow} + n_{b\downarrow})

    """
    a_up, b_up, a_dw, b_dw = sorted_basis()
    sigma_za = a_up.T * a_up - a_dw.T * a_dw
    sigma_zb = b_up.T * b_up - b_dw.T * b_dw
    H  = - U / 2 * sigma_za * sigma_za - mu * (a_up.T * a_up + a_dw.T * a_dw)
    H += - U / 2 * sigma_zb * sigma_zb - mu * (b_up.T * b_up + b_dw.T * b_dw)
    H += tp * (a_up.T * b_up + a_dw.T * b_dw + b_up.T * a_up + b_dw.T * a_dw)
    return H, [a_up, b_up, a_dw,  b_dw]


def dimer_hamiltonian_bond(U, mu, tp):
    r"""Generate a single orbital isolated atom Hamiltonian in particle-hole
    symmetry. Include chemical potential for grand Canonical calculations

    This in the basis of Bonding and Anti-bonding states

    See also
    --------
    dimer_hamiltonian
    """
    from math import sqrt
    as_up, s_up, as_dw, s_dw = sorted_basis()

    a_up = (-as_up + s_up) / sqrt(2)
    b_up = ( as_up + s_up) / sqrt(2)
    a_dw = (-as_dw + s_dw) / sqrt(2)
    b_dw = ( as_dw + s_dw) / sqrt(2)

    sigma_za = a_up.T * a_up - a_dw.T * a_dw
    sigma_zb = b_up.T * b_up - b_dw.T * b_dw
    H  = - U / 2 * sigma_za * sigma_za - mu * (a_up.T * a_up + a_dw.T * a_dw)
    H += - U / 2 * sigma_zb * sigma_zb - mu * (b_up.T * b_up + b_dw.T * b_dw)
    H += tp * (a_up.T * b_up + a_dw.T * b_dw + b_up.T * a_up + b_dw.T * a_dw)
    return H, [as_up, s_up, as_dw, s_dw]


###############################################################################
# Dimer Bethe lattice


def gf_met(omega, mu, tp, t, tn):
    """Double semi-circular density of states to represent the
    non-interacting dimer """

    g_1 = gf.greenF(omega, mu=mu - tp, D=2 * (t + tn))
    g_2 = gf.greenF(omega, mu=mu + tp, D=2 * abs(t - tn))
    g_d = .5 * (g_1 + g_2)
    g_o = .5 * (g_1 - g_2)

    return g_d, g_o


def mat_inv(a, b):
    """Inverts the relevant entries of the dimer Green's function matrix

    .. math:: [a, b]^-1 = [a, -b]/(a^2 - b^2)
    """
    det = a * a - b * b
    return a / det, -b / det


def mat_2_inv(A):
    """Inverts a 2x2 matrix"""
    det = A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1]
    return np.asarray([[A[1, 1], -A[0, 1]],  [-A[1, 0],  A[0, 0]]]) / det


def mat_mul(a, b, c, d):
    """Multiplies two Matrices of the dimer Green's Functions"""
    return a * c + b * d, a * d + b * c


def self_consistency(omega, Gd, Gc, mu, tp, t2):
    """Sets the dimer Bethe lattice self consistent condition for the diagonal
    and out of diagonal block
    """

    Dd = omega + mu - t2 * Gd
    Dc = -tp - t2 * Gc

    return mat_inv(Dd, Dc)


def dimer_dyson(g0iw_d, g0iw_o, siw_d, siw_o):
    """Returns Dressed Green Function from G0 and Sigma"""

    sgd, sgo = mat_mul(g0iw_d, g0iw_o, -siw_d, -siw_o)
    sgd += 1.
    dend, dendo = mat_inv(sgd, sgo)

    return mat_mul(dend, dendo, g0iw_d, g0iw_o)


def ipt_dmft_loop(BETA, u_int, tp, giw_d, giw_o, tau, w_n, conv=1e-12):

    converged = False
    loops = 0
    iw_n = 1j * w_n

    while not converged:
        # Half-filling, particle-hole cleaning
        giw_d.real = 0.
        giw_o.imag = 0.

        giw_d_old = giw_d.copy()
        giw_o_old = giw_o.copy()

        g0iw_d, g0iw_o = self_consistency(iw_n, giw_d, giw_o, 0., tp, 0.25)

        siw_d, siw_o = ipt.dimer_sigma(u_int, tp, g0iw_d, g0iw_o, tau, w_n)
        giw_d, giw_o = dimer_dyson(g0iw_d, g0iw_o, siw_d, siw_o)

        converged = np.allclose(giw_d_old, giw_d, conv)
        converged *= np.allclose(giw_o_old, giw_o, conv)

        loops += 1
        if loops > 3000:
            converged = True
            print('B', BETA, 'tp', tp, 'U', u_int)
            print('Failed to converge in less than 3000 iterations')

    return giw_d, giw_o, loops


def ekin(giw_d, giw_o, w_n, tp, beta, t_sqr=0.25):
    """Calculates the kinetic energy per spin from its Green Function"""
    return (tp * giw_o.real + t_sqr * (-giw_d.imag**2 + giw_o.real**2) +
            (t_sqr + tp**2) / w_n**2).real.sum() / beta * 2 - \
        beta / 4 * (t_sqr + tp**2)


def epot(giw_d, giw_o, siw_d, siw_o, w_n, tp, u_int, beta):
    """Calculates the potential energy per spin

    Using the Green Function and self-energy as in
    :ref:`potential_energy`, which in this case have a matrix
    structure in their product but one is only interested in the
    diagonal terms of such product. Also for symmetry reason. A-B and
    paramagnetism, only the first diagonal term is calculated, as one
    is interested in the per spin energy. To get the per unit cell
    multiply by 4. That is 2 sites times 2 spins, a total of 4
    flavors.

    Tail expansion is only taken relevant up to second order in the
    product from the known moments of the Green function and
    Self-Energy. In the case it ends up being the same as single band

"""
    return (-siw_d.imag * giw_d.imag + siw_o.real * giw_o.real +
            u_int**2 / 4 / w_n**2).sum() / beta - beta * u_int**2 / 32 + u_int / 8


def quasiparticle(filestr, beta):
    zet = []
    with h5.File(filestr.format(beta), 'r') as results:
        tau, w_n = gf.tau_wn_setup(
            dict(BETA=beta, N_MATSUBARA=max(5 * beta, 256)))
        for tpstr in results:
            tp = float(tpstr[2:])
            tprec = results[tpstr]
            for ustr in tprec:
                g0iw_d, g0iw_o = self_consistency(1j * w_n,
                                                  1j * tprec[ustr]['giw_d'][:],
                                                  tprec[ustr]['giw_o'][:],
                                                  0., tp, 0.25)
                u_int = float(ustr[1:])
                siw_d, _ = ipt.dimer_sigma(u_int, tp, g0iw_d, g0iw_o, tau, w_n)

                zet.append(matsubara_Z(siw_d.imag, beta))
        array_shape = (len(results.keys()), len(tprec.keys()))

    return np.asarray(zet).reshape(array_shape)


def fermi_level_dos(filestr, beta, n=3):
    with h5.File(filestr.format(beta), 'r') as results:
        w_n = gf.matsubara_freq(beta, n)
        dos_fl = np.array([gf.fit_gf(w_n, results[tpstr][uint]['giw_d'][:n])(0.)
                           for tpstr in results for uint in results[tpstr]])
        dos_fl = dos_fl.reshape((len(results.keys()), -1))
    return dos_fl

# plots


def plot_giw(beta, tp, u_int, label, filestr, axes=None):
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(13, 8), sharex=True, sharey=True)
    u_group = '/tp{}/U{}/'.format(tp, u_int)
    w_n = gf.matsubara_freq(beta, max(5 * beta, 256))
    with h5.File(filestr.format(beta), 'r') as results:
        jgiw_d, rgiw_o = results[
            u_group + 'giw_d'][:], results[u_group + 'giw_o'][:]

        axes[0].plot(w_n, jgiw_d, 's:', label=label)
        axes[1].plot(w_n, rgiw_o, 'o:', label=label)

        graf = r'$G(i\omega_n)$'
        axes[0].set_xlabel(r'$i\omega$')
        axes[0].set_xlim([0, 4])

    axes[0].set_title(
        r'Solutions at $\beta={}$ and $t_\perp={}$'.format(beta, tp))
    axes[0].set_ylabel(graf + '$_{AA}$')
    axes[0].legend(loc=0)
    axes[1].set_ylabel(graf + '$_{AB}$')
    axes[1].legend(loc=0)

    return axes


def get_giw(beta, tp, u_int, filestr):
    """Returns all Greens functions of a given cut at tp or U

    either tp or u_int have to be defined and the other must be None"""

    tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=max(5 * beta, 256)))
    with h5.File(filestr.format(beta), 'r') as results:
        if u_int:
            u_str = 'U' + str(u_int)
            gfs = [(1j * results[tpstr][u_str]['giw_d'][:],
                    results[tpstr][u_str]['giw_o'][:])
                   for tpstr in results]
        elif tp:
            tpstr = 'tp' + str(tp)
            gfs = [(1j * results[tpstr][u_str]['giw_d'][:],
                    results[tpstr][u_str]['giw_o'][:])
                   for u_str in results[tpstr]]
    return gfs


def get_g0(beta, tp, u_int, filestr):
    """Returns all the G0 functions of a given cut at tp or U

    either tp or u_int have to be defined and the other must be None"""
    tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=max(5 * beta, 256)))
    g0 = []
    with h5.File(filestr.format(beta), 'r') as results:
        if u_int:
            u_str = 'U' + str(u_int)
            for tp in results.keys():
                jgiw_d, rgiw_o = results[tp][u_str]['giw_d'][
                    :], results[tp][u_str]['giw_o'][:]
                g0.append((self_consistency(1j * w_n, 1j * jgiw_d, rgiw_o,
                                            0., float(tp[2:]), 0.25)))
        elif tp:
            tpstr = 'tp' + str(tp)
            for u_str in results[tpstr].keys():
                jgiw_d, rgiw_o = results[tpstr][u_str]['giw_d'][
                    :], results[tpstr][u_str]['giw_o'][:]
                g0.append((self_consistency(1j * w_n, 1j * jgiw_d, rgiw_o,
                                            0., tp, 0.25)))
    return g0


def get_sigmaiw(beta, tp, u_int, filestr):
    """Returns all the Sigma functions of a given cut at tp or U

    either tp or u_int have to be defined and the other must be None"""
    tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=max(5 * beta, 256)))
    sigma = []
    with h5.File(filestr.format(beta), 'r') as results:
        if u_int:
            u_str = 'U' + str(u_int)
            for tp in results.keys():
                jgiw_d, rgiw_o = results[tp][u_str]['giw_d'][
                    :], results[tp][u_str]['giw_o'][:]
                g0iw_d, g0iw_o = self_consistency(1j * w_n, 1j * jgiw_d, rgiw_o,
                                                  0., float(tp[2:]), 0.25)
                sigma.append(ipt.dimer_sigma(
                    u_int, float(tp[2:]), g0iw_d, g0iw_o, tau, w_n))
        elif tp:
            tpstr = 'tp' + str(tp)
            for u_str in results[tpstr].keys():
                jgiw_d, rgiw_o = results[tpstr][u_str]['giw_d'][
                    :], results[tpstr][u_str]['giw_o'][:]
                g0iw_d, g0iw_o = self_consistency(1j * w_n, 1j * jgiw_d, rgiw_o,
                                                  0., tp, 0.25)
                u_int = float(u_str[1:])
                sigma.append(ipt.dimer_sigma(
                    u_int, tp, g0iw_d, g0iw_o, tau, w_n))
    return sigma


def gf_ancon_gap(filestr, beta, u_int, tp, w, min_poles_fit=100):
    """Plots the analytical continuation of the local Gf along a given U or tp line

    For the insulator guess and metalic guess

    either u_int or tp can have a single value the other must be none
    """

    w_n = gf.matsubara_freq(beta, int(beta))
    regf = []

    with h5.File(filestr.format(beta), 'r') as results:
        if u_int:
            u_str = 'U' + str(u_int)
            gfs = [(results[tpstr][u_str]['giw_d'][:], results[tpstr][u_str]['giw_o'][:])
                   for tpstr in results]
        else:

            tpstr = 'tp' + str(tp)
            gfs = [(results[tpstr][u_str]['giw_d'][:], results[tpstr][u_str]['giw_o'][:])
                   for u_str in results[tpstr]]

        for giw in gfs:
            pc = gf.pade_coefficients(1j * giw[0][:len(w_n)], w_n)
            regf.append(gf.pade_rec(pc[:min(len(w_n), min_poles_fit)], w, w_n))
    return np.array(gfs), np.array(regf)
