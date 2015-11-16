# -*- coding: utf-8 -*-
r"""
==============================
IPT Hysteresis Energy analysis
==============================

To study the stability of the solutions in the coexistence region

"""

from __future__ import division, absolute_import, print_function

from dmft.ipt_imag import dmft_loop
from dmft.common import greenF, tau_wn_setup, fit_gf
from dmft.twosite import matsubara_Z
from scipy.optimize import fsolve
import slaveparticles.quantum.dos as dos
from scipy.integrate import quad

import numpy as np
import matplotlib.pylab as plt



###############################################################################
# Total energy
# ------------
#
# Starting first with the kinetic energy
#
# .. math:: \langle T \rangle  = Tr \frac{1}{\beta} \sum_{k,n} \epsilon_k^0 G(k, i\omega_n)
#
# It can be transformed into a treatable form relying on local quantities
#
# .. math:: \langle T \rangle  = Tr \frac{1}{\beta} \sum_{k,n} \left( \epsilon_k^0 G(k, i\omega_n) + G(k, i\omega_n)^{-1}G(k, i\omega_n) - G^{free}(k, i\omega_n)^{-1}G^{free}(k, i\omega_n) \right)
#
# .. math::  = Tr \frac{1}{\beta} \sum_{k,n} \left( \epsilon_k^0 G(k, i\omega_n) + (i\omega_n - \epsilon_k^0 - \Sigma(i\omega_n))G(k, i\omega_n) - (i\omega_n - \epsilon_k^0)G^{free}(k, i\omega_n) \right)
#
# .. math::  = Tr \frac{1}{\beta} \sum_{k,n} \left( i\omega_n \left( G(k, i\omega_n)- G(k, i\omega_n)^{free} \right) - \Sigma(i\omega_n) G(k, i\omega_n) \epsilon_k^0 G(k, i\omega_n) + \epsilon_k^0G^{free}(k, i\omega_n) \right)
#
# The first two terms can be summed in reciprocal space to yield a
# local the quantities that come out of the DMFT self-consistency and
# the last term as it belongs to the non-interacting system is
# trivially solvable
#
# .. math::  \langle T \rangle = Tr \frac{1}{\beta} \sum_n \left( i\omega_n \left( G(i\omega_n)- G(i\omega_n)^{free} \right) - \Sigma(i\omega_n)G(i\omega_n) \right) + \int_{\infty}^\mu \epsilon\rho(\epsilon)n_F(\epsilon) d\epsilon

def hysteresis(beta, u_range):
    log_g_sig = []
    tau, w_n = tau_wn_setup(dict(BETA=beta, N_MATSUBARA=beta))
    g_iwn = greenF(w_n)
    for u_int in u_range:
        g_iwn, sigma = dmft_loop(u_int, 0.5, g_iwn, w_n, tau)
        log_g_sig.append((g_iwn, sigma))
    return log_g_sig


def energy(beta, u_range, g_s_results):
    tau, w_n = tau_wn_setup(dict(BETA=beta, N_MATSu_rangeBARA=beta))
    gfree = greenF(w_n)
    n_half = lambda e: quad(dos.bethe_fermi, -1, e, args=(1., 0., 0.5, beta))[0]-0.5
    mu = fsolve(n_half, 0.)[0]
    e_mean = quad(dos.bethe_fermi_ene, -1, mu, args=(1., 0., 0.5, beta))[0]
    T = np.asarray([2*(1j*w_n*(g-gfree) - s*g).real.sum()/beta + e_mean
                    for (g, s), u in zip(g_s_results, u_range)])
    V = np.asarray([(s*g+u**2/4./w_n**2).real.sum()/beta
                    for (g, s), u in zip(g_s_results, u_range)]) - beta*u_range**2/32.

    return T, V

U = np.linspace(2.4, 3.6, 41)
rU = U[::-1]
fige, axe = plt.subplots(nrows=3, sharex=True, sharey=True)
fige.subplots_adjust(hspace=0.)
betarange = [50, 200, 512]
for i, beta in enumerate(betarange):
    Ti, Vi = energy(beta, rU, hysteresis(beta, rU))
    Tm, Vm = energy(beta, U, hysteresis(beta, U))

    axe[0].plot(rU, Ti+Vi, '--', label=r'$\langle T \rangle$')
    axe[1].plot(rU, Tm+Vm, '--', label=r'$\langle V \rangle$')
    axe[2].plot(rU, Ti+Vi-(Tm+Vm), '--', label=r'$\langle H \rangle$')

    axe[i].set_ylabel(r'$\langle E \rangle$ @ $\beta={}$'.format(beta))
    axe[i].set_xlabel('U/D')
    axe[i].set_xlim([2, 3.5])

axe[0].set_title('Hysteresis loop of energies')
axe[2].legend(loc=0)
