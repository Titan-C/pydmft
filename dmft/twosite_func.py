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

def interacting_dos(self, mu):
    """Evaluates the interacting density of states"""
    w = self.omega + mu - self.GF[r'$\Sigma$']
    return dos.bethe_lattice(w, self.t)

def lattice_ocupation(self, mu):
    w = np.copy(self.omega[:len(self.omega)/2+1])
    intdos = self.interacting_dos(mu)[:len(w)]
    w[-1] = 0
    intdos[-1] = (intdos[-1] + intdos[-2])/2
    dosint = 2*simps(intdos, w)
    return dosint

def find_mu(self, target_n, u_int):
    """Find the required chemical potential to give the required filling"""
    zero = lambda mu: self.lattice_ocupation(mu) - target_n
    self.mu = fsolve(zero, u_int*target_n/2, xtol=5e-4)[0]
    return self.mu

def selfconsitency(self, e_c, hyb, target_n, u_int):
    """Performs the selfconsistency loop"""
    convergence = False
    ne_ec = e_c
    if target_n == 1:
        ne_ec = u_int / 2
        self.mu = u_int / 2
    while not convergence:
        old = hyb
#            if not target_n == 1:
        old_ec = ne_ec
        self.find_mu(target_n, u_int)
        ne_ec = fsolve(self.restriction, old_ec,
                       (u_int, old), xtol=5e-3)[0]
        self.solve(ne_ec, u_int, hyb)
        hyb = self.hyb_V()
        if 2.5 < u_int < 3:
            hyb = (hyb + old)/2
        convergence = np.abs(old - hyb) < 1e-5\
            and np.abs(self.restriction(ne_ec, u_int, hyb)) < 2e-2

    return ne_ec, hyb

def restriction(self, e_c, u_int, hyb):
    """Lagrange multiplier in lattice slave spin"""
    self.solve(float(e_c), u_int, hyb)
    return np.sum(self.imp_ocupation())-self.lattice_ocupation(self.mu)


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
