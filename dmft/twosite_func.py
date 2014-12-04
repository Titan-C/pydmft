# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 00:22:51 2014

@author: oscar
"""
from __future__ import division, absolute_import, print_function
import numpy as np
from scipy.integrate import simps
from slaveparticles.quantum import dos
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, curve_fit


def lattice_gf(sim, x=np.linspace(-4, 4, 600), wide=5e-3):
    """Compute lattice green function

    .. math::
        G(\\omega) = \\int \\frac{\\rho_0(x) dx}{\\omega
        + i\\eta + \\mu - \\Sigma(w) - x }"""
    G = []
    var = sim.omega + sim.mu - sim.GF[r'$\Sigma$'] + 1j*wide
    for w in var:
        integrable = sim.rho_0/(w - x)
        G.append(simps(integrable, x))

    return np.asarray(G)


def two_pole(w, alpha_0, alpha_1, alpha_2, omega_1, omega_2):
    r"""This function evaluates a two pole real function in the shape

    .. math:: \Sigma(\omega)=\alpha_0 + \frac{\alpha_1}{\omega - \omega_1}
        +\frac{\alpha_2}{\omega - \omega_2}"""
    return alpha_0 + alpha_1/(w - omega_1) + alpha_2/(w - omega_2)


def fit_sigma(sim):
    """Fits the self-energy into its analytical two pole form"""
    w = sim.omega
    sigma = sim.GF[r'$\Sigma$']
    return curve_fit(two_pole, w, sigma)

def dmft_loop(u_int=np.arange(0, 3.2, 0.05), axis='real',
              beta=1e5, hop=0.5, hyb=0.4, filling=1):
    if axis == 'matsubara':
        return matsubara_loop(u_int, beta, hop, hyb)

    res = []
    e_c = 0
    for U in u_int:
        sim = twosite(beta, hop, axis)
        e_c, hyb = sim.selfconsitency(e_c, hyb, filling, U)
        print(U, sim.mu, e_c, hyb)

        sim.solve(e_c, U, hyb)
        hyb = sim.hyb_V()
        res.append((U, sim.imp_z(), sim))
    return np.asarray(res)

def out_plot(sim, spec, label=''):
    w = sim.omega.imag
    stl = '+-'
    if sim.freq_axis == 'real':
        w = sim.omega.real
        stl = '-'

    for gfp in spec.split():
        if 'impG' == gfp:
            key = 'Imp G'
        if 'impG0' in gfp:
            key = 'Imp G$_0$'
        if 'sigma' == gfp:
            key = r'$\Sigma$'
        if 'G' == gfp:
            key = 'Lat G'
        if 'A' == gfp:
            plt.plot(w, sim.interacting_dos(sim.mu), stl, label='A '+label)
            continue
        plt.plot(w, sim.GF[key].real, stl, label='Re {} {}'.format(key, label))
        plt.plot(w, sim.GF[key].imag, stl+'-', label='Im {} {}'.format(key, label))
