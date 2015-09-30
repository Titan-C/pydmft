# -*- coding: utf-8 -*-
"""
===================================
Isolated molecule spectral function
===================================

For the case of contact interaction in the di-atomic molecule case
spectral function are evaluated by means of the Lehmann representation
"""
# author: Óscar Nájera

from __future__ import division, absolute_import, print_function
from dmft.common import matsubara_freq, gw_invfouriertrans
from itertools import product
from math import sqrt
from slaveparticles.quantum import fermion
import slaveparticles.quantum.operators as op
import matplotlib.pyplot as plt
import numpy as np



def hamiltonian(U, mu, tp):
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
    a_up, a_dw, b_up, b_dw = [fermion.destruct(4, sigma) for sigma in range(4)]
    sigma_za = a_up.T*a_up - a_dw.T*a_dw
    sigma_zb = b_up.T*b_up - b_dw.T*b_dw
    H =  - U/2 * sigma_za * sigma_za - mu * (a_up.T*a_up + a_dw.T*a_dw)
    H += - U/2 * sigma_zb * sigma_zb - mu * (b_up.T*b_up + b_dw.T*b_dw)
    H += -tp * (a_up.T*b_up + a_dw.T*b_dw + b_up.T*a_up + b_dw.T*a_dw)
    return H, [a_up, a_dw, b_up, b_dw]


def plot_real_gf(eig_e, eig_v, oper_pair, c_v, names):
    _, axw = plt.subplots(2, sharex=True)
    w = np.linspace(-1.5, 1.5, 500) + 1j*1e-2
    gfs = [op.gf_lehmann(eig_e, eig_v, c.T, beta, w, d) for c, d in oper_pair]
    for gw, color, name in zip(gfs, c_v, names):
        axw[0].plot(w.real, gw.real, color, label=r'${}$'.format(name))
        axw[1].plot(w.real, -1*gw.imag/np.pi, color)
        axw[0].legend()
        axw[0].set_title(r'Real Frequencies Green functions, $\beta={}$'.format(beta))
        axw[0].set_ylabel(r'$\Re e G(\omega)$')
        axw[1].set_ylabel(r'$A(\omega)$')
        axw[1].set_xlabel(r'$\omega$')


def plot_matsubara_gf(eig_e, eig_v, oper_pair, c_v, names):
    gwp, axwn = plt.subplots(2, sharex=True)
    gwp.subplots_adjust(hspace=0)
    gtp, axt = plt.subplots()
    wn = matsubara_freq(beta, 64)
    tau = np.linspace(0, beta, 2**10)
    gfs = [op.gf_lehmann(eig_e, eig_v, c.T, beta, 1j*wn, d) for c, d in oper_pair]
    for giw, color, name in zip(gfs, c_v, names):
        axwn[0].plot(wn, giw.real, color+'s-', label=r'${}$'.format(name))
        axwn[1].plot(wn, giw.imag, color+'o-')

        tail =  [0., -tp, 0.] if name[0]!=name[1] else [1., 0., 0.]
        gt = gw_invfouriertrans(giw, tau, wn, tail)
        axt.plot(tau, gt, label=r'${}$'.format(name))

        axwn[0].legend()
        axwn[0].set_title(r'Matsubara Green functions, $\beta={}$'.format(beta))
        axwn[1].set_xlabel(r'$\omega_n$')
        axwn[0].set_ylabel(r'$\Re e G(i\omega_n)$')
        axwn[1].set_ylabel(r'$\Im m G(i\omega_n)$')

        axt.set_ylim(top=0.05)
        axt.legend(loc=0)
        axt.set_title(r'Imaginary time Green functions, $\beta={}$'.format(beta))
        axt.set_xlabel(r'$\tau$')
        axt.set_ylabel(r'$G(\tau)$')


beta = 50
U = 1.
mu = 0.
tp = 0.25
c_v = ['b', 'g', 'r', 'k']
names = [r'a\uparrow', r'a\downarrow', r'b\uparrow', r'b\downarrow']

h_at, oper = hamiltonian(U, mu, tp)
eig_e, eig_v = op.diagonalize(h_at.todense())
oper_pair = list(product([oper[0], oper[2]], repeat=2))
names = list(product('AB', repeat=2))
plot_real_gf(eig_e, eig_v, oper_pair, c_v, names)
plot_matsubara_gf(eig_e, eig_v, oper_pair, c_v, names)


############################################################
# The symmetric and anti-symmetric bands
# ======================================
#

def hamiltonian_bond(U, mu, tp):
    r"""Generate a single orbital isolated atom Hamiltonian in particle-hole
    symmetry. Include chemical potential for grand Canonical calculations

    .. math::
    """
    as_up, as_dw, s_up, s_dw = [fermion.destruct(4, sigma) for sigma in range(4)]

    a_up = (-as_up + s_up)/sqrt(2)
    b_up = ( as_up + s_up)/sqrt(2)
    a_dw = (-as_dw + s_dw)/sqrt(2)
    b_dw = ( as_dw + s_dw)/sqrt(2)

    sigma_za = a_up.T*a_up - a_dw.T*a_dw
    sigma_zb = b_up.T*b_up - b_dw.T*b_dw
    H =  - U/2 * sigma_za * sigma_za - mu * (a_up.T*a_up + a_dw.T*a_dw)
    H += - U/2 * sigma_zb * sigma_zb - mu * (b_up.T*b_up + b_dw.T*b_dw)
    H += tp * (a_up.T*b_up + a_dw.T*b_dw + b_up.T*a_up + b_dw.T*a_dw)
    return H, [as_up, as_dw, s_up, s_dw]

h_at, oper = hamiltonian_bond(U, mu, tp)
eig_e, eig_v = op.diagonalize(h_at.todense())

oper_pair = product([oper[0], oper[2]], repeat=2)

#plot_real_gf(eig_e, eig_v, oper_pair, c_v, names)

# TODO: verify the asy/sym basis scale
# TODO: view in the local one the of diag terms
