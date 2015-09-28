# -*- coding: utf-8 -*-
"""
===================================
Isolated molecule spectral function
===================================

For the case of contact interaction in the di-atomic molecule case
spectral function are evaluated by means of the Lehman representation
"""
# author: Óscar Nájera

from __future__ import division, absolute_import, print_function
import numpy as np
import matplotlib.pyplot as plt
from dmft.common import matsubara_freq, gw_invfouriertrans
from slaveparticles.quantum import fermion
from slaveparticles.quantum.operators import gf_lehmann, diagonalize


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
    return H, a_up, a_dw, b_up, b_dw


def gf(w, U, mu, tp, beta):
    """Calculate by Lehmann representation the green function"""
    H, a_up, a_dw, b_up, b_dw = hamiltonian(U, mu, tp)
    e, v = diagonalize(H.todense())
    g_up = gf_lehmann(e, v, a_up.T, beta, w)
    return g_up


beta = 50
U = 1.
tp = 0.4
mu_v = np.array([0])#, 0.2, 0.45, 0.5, 0.65])
c_v = ['b', 'g', 'r', 'k', 'm']

f, axw = plt.subplots(2, sharex=True)
f.subplots_adjust(hspace=0)
w = np.linspace(-1.5, 1.5, 500) + 1j*1e-2
for mu, c in zip(mu_v, c_v):
    gw = gf(w, U, mu, tp, beta)
    axw[0].plot(w.real, gw.real, c, label=r'$\mu={}$'.format(mu))
    axw[1].plot(w.real, -1*gw.imag/np.pi, c)
axw[0].legend()
axw[0].set_title(r'Real Frequencies Green functions, $\beta={}$'.format(beta))
axw[0].set_ylabel(r'$\Re e G(\omega)$')
axw[1].set_ylabel(r'$A(\omega)$')
axw[1].set_xlabel(r'$\omega$')


gwp, axwn = plt.subplots(2, sharex=True)
gwp.subplots_adjust(hspace=0)
gtp, axt = plt.subplots()
wn = matsubara_freq(beta, 64)
tau = np.linspace(0, beta, 2**10)
for mu, c in zip(mu_v, c_v):
    giw = gf(1j*wn, U, mu, tp, beta)
    axwn[0].plot(wn, giw.real, c+'s-', label=r'$\mu={}$'.format(mu))
    axwn[1].plot(wn, giw.imag, c+'o-')

    gt = gw_invfouriertrans(giw, tau, wn)
    axt.plot(tau, gt, label=r'$\mu={}$'.format(mu))

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
