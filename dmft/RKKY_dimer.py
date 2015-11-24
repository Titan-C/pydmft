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
import numpy as np
import slaveparticles.quantum.dos as dos

###############################################################################
# Dimer Bethe lattice


def gf_met(omega, mu, tp, t, tn):
    """Double semi-circular density of states to represent the
    non-interacting dimer """

    g_1 = gf.greenF(omega, mu=mu-tp, D=2*(t+tn))
    g_2 = gf.greenF(omega, mu=mu+tp, D=2*abs(t-tn))
    g_d = .5*(g_1 + g_2)
    g_o = .5*(g_1 - g_2)

    return g_d, g_o


def mat_inv(a, b):
    """Inverts the relevant entries of the dimer Green's function matrix

    .. math:: [a, b]^-1 = [a, -b]/(a^2 - b^2)
    """
    det = a*a - b*b
    return a/det, -b/det


def mat_mul(a, b, c, d):
    """Multiplies two Matrices of the dimer Green's Functions"""
    return a*c + b*d, a*d + b*c


def self_consistency(omega, Gd, Gc, mu, tp, t2):
    """Sets the dimer Bethe lattice self consistent condition for the diagonal
    and out of diagonal block
    """

    Dd = omega + mu - t2 * Gd
    Dc = -tp - t2 * Gc

    return mat_inv(Dd, Dc)


def dimer_dyson(g0iw_d, g0iw_o, siw_d, siw_o):

    sgd, sgo = mat_mul(g0iw_d, g0iw_o, -siw_d, -siw_o)
    sgd += 1.
    dend, dendo = mat_inv(sgd, sgo)

    return mat_mul(dend, dendo, g0iw_d, g0iw_o)


def ipt_dmft_loop(BETA, u_int, tp, giw_d, giw_o, conv=1e-3):
    tau, w_n = gf.tau_wn_setup(dict(BETA=BETA, N_MATSUBARA=max(5*BETA, 256)))

    converged = False
    loops = 0
    iw_n = 1j*w_n

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
        if loops > 2000:
            converged = True
            print('B', BETA, 'tp', tp, 'U', u_int)
            print('Failed to converge in less than 2000 iterations')


    return giw_d, giw_o, loops


def epot(filestr, beta):
    V = []
    with h5.File(filestr.format(beta), 'r') as results:
        tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=max(5*beta, 256)))
        wsqr_4 = 4*w_n*w_n
        for tpstr in results:
            tp = float(tpstr[2:])
            tprec = results[tpstr]
            for ustr in tprec:
                jgiw_d, rgiw_o = tprec[ustr]['giw_d'][:], tprec[ustr]['giw_o'][:]
                g0iw_d, g0iw_o = self_consistency(1j*w_n, 1j*jgiw_d, rgiw_o,
                                                  0., tp, 0.25)
                u_int = float(ustr[1:])
                siw_d, siw_o = ipt.dimer_sigma(u_int, tp, g0iw_d, g0iw_o, tau, w_n)

                V.append((rgiw_o*siw_o.real - jgiw_d*siw_d.imag +
                          u_int**2/wsqr_4).sum()/beta - beta*u_int**2/32)
        array_shape = (len(results.keys()), len(tprec.keys()))

    return np.asarray(V).reshape(array_shape)


def n_fill(mu, tp, beta):
    return quad(dos.bethe_fermi, -1., 1., args=(1., mu + tp, 0.5, beta))[0] + \
           quad(dos.bethe_fermi, -1., 1., args=(1., mu - tp, 0.5, beta))[0]-1.


def free_ekin(tp, beta):
    mu = fsolve(n_fill, 0., (tp, beta))[0]
    e_mean = quad(dos.bethe_fermi_ene, -1., 1., (1., mu+tp, 0.5, beta))[0] + \
             quad(dos.bethe_fermi_ene, -1., 1., (1., mu-tp, 0.5, beta))[0] - \
             tp*(quad(dos.bethe_fermi, -1., 1., (1., mu+tp, 0.5, beta))[0] -
                 quad(dos.bethe_fermi, -1., 1., (1., mu-tp, 0.5, beta))[0])
    return e_mean * 0.5


def ekin(filestr, beta):
    r"""Calculates the internal energy of the system given by Fetter-Walecka"""
    T = []
    with h5.File(filestr.format(beta), 'r') as results:
        tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=max(5*beta, 256)))
        for tpstr in results:
            tp = float(tpstr[2:])
            tprec = results[tpstr]
            giw_free_d, _ = gf_met(w_n, 0., tp, 0.5, 0.)
            e_mean = free_ekin(tp, beta)
            for ustr in tprec:
                jgiw_d, rgiw_o = tprec[ustr]['giw_d'][:], tprec[ustr]['giw_o'][:]
                g0iw_d, g0iw_o = self_consistency(1j*w_n, 1j*jgiw_d, rgiw_o,
                                                  0., tp, 0.25)
                u_int = float(ustr[1:])
                siw_d, siw_o = ipt.dimer_sigma(u_int, tp, g0iw_d, g0iw_o, tau, w_n)

                T.append(2*(w_n*(giw_free_d.imag - jgiw_d) +
                         jgiw_d*siw_d.imag - rgiw_o*siw_o.real).sum()/beta + e_mean)
        array_shape = (len(results.keys()), len(tprec.keys()))

    return np.asarray(T).reshape(array_shape)


def complexity(filestr, beta):
    """Extracts the loopcount for convergence"""
    with h5.File(filestr.format(beta), 'r') as results:
        comp = [results[tpstr][uint]['loops'].value
                for tpstr in results
                for uint in results[tpstr]]
        array_column = len(results.keys())
    return np.asarray(comp).reshape((array_column, -1))


def quasiparticle(filestr, beta):
    zet = []
    with h5.File(filestr.format(beta), 'r') as results:
        tau, w_n = gf.tau_wn_setup(dict(BETA=beta, N_MATSUBARA=max(5*beta, 256)))
        for tpstr in results:
            tp = float(tpstr[2:])
            tprec = results[tpstr]
            for ustr in tprec:
                g0iw_d, g0iw_o = self_consistency(1j*w_n,
                                                  1j*tprec[ustr]['giw_d'][:],
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
