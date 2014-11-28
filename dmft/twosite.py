# -*- coding: utf-8 -*-
"""
Two Site Dynamical Mean Field Theory
====================================
The two site DMFT approach given by M. Potthoff [Potthoff2001]_ on how to
treat the impurity bath as a sigle site of the DMFT. Work is around a single
impurity Anderson model.

.. [Potthoff2001] M. Potthoff PRB, 64, 165114, 2001
"""

from __future__ import division, absolute_import, print_function
import numpy as np
from scipy.integrate import simps, quad
from slaveparticles.quantum.operators import gf_lehmann, diagonalize, expected_value
from slaveparticles.quantum import dos, fermion
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, curve_fit


def m2_weight(t):
    """Calculates the :math:`M_2^{(0)}=\\int  x^2 \\rho_0(x)dx` which is the
       variance of the non-interacting density of states of a Bethe Lattice"""
    second_moment = lambda x: x*x*dos.bethe_lattice(x, t)

    return quad(second_moment, -2*t, 2*t)[0]


class twosite(object):
    """DMFT solver for an impurity and a single bath site

    Sets up environment

    Parameters
    ----------
    beta : float
           Inverse temperature of the system
    t : float
        Hopping amplitude between first neighbor lattice sites
    freq_axis : string
               'real' or 'matsubara' frequencies

    Attributes
    ----------
    GF : dictionary
         Stores the Green functions and self energy"""

    def __init__(self, beta, t, freq_axis, npoints=600):
        self.beta = beta
        self.t = t
        self.m2 = m2_weight(t)
        self.freq_axis = freq_axis

        if freq_axis == 'real':
            self.omega = np.linspace(-8, 8, npoints)
        elif freq_axis == 'matsubara':
            self.omega = 1j*np.arange(1, npoints, 2) / self.beta
        else:
            raise ValueError('Set a working frequency axis')

        self.rho_0 = dos.bethe_lattice(self.omega, self.t)
        self.eig_energies = None
        self.eig_states = None
        self.oper = [fermion.destruct(4, index) for index in range(4)]
        self.H_operators = self.hamiltonian()
        self.GF = {}
        self.mu = 0.
        self.solve(0, 0, 0)

    def hamiltonian(self):
        r"""Two site single impurity anderson model
        generate the matrix operators that will be used for this hamiltonian

        .. math::
           \mathcal{H} = -\mu d^\dagger_\sigma d_\sigma
           + (\epsilon_c - \mu) c^\dagger_\sigma c_\sigma +
           U d^\dagger_\uparrow d_\uparrow d^\dagger_\downarrow d_\downarrow
           + V(d^\dagger_\sigma c_\sigma + h.c.)"""

        d_up, d_dw, c_up, c_dw = self.oper

        H = {'impurity': d_up.T*d_up + d_dw.T*d_dw,
             'bath': c_up.T*c_up + c_dw.T*c_dw,
             'u_int': d_up.T*d_up*d_dw.T*d_dw,
             'hyb': d_up.T*c_up + d_dw.T*c_dw + c_up.T*d_up + c_dw.T*d_dw}

        return H

    def update_H(self, e_c, u_int, hyb):
        """Updates impurity hamiltonian and diagonalizes it"""
        H = - self.mu*self.H_operators['impurity'] + \
            (e_c - self.mu)*self.H_operators['bath'] + \
            u_int*self.H_operators['u_int'] + \
            hyb*self.H_operators['hyb']

        self.eig_energies, self.eig_states = diagonalize(H.todense())

    def expected(self, observable):
        """Wrapper to the expected_value function to fix the eigenbasis"""
        return expected_value(observable, self.eig_energies, self.eig_states,
                              self.beta)

    def solve(self, e_c, u_int, hyb):
        """Solves the impurity problem"""
        self.update_H(e_c, u_int, hyb)
        d_up_dag = self.oper[0].T
        self.GF['Imp G'] = gf_lehmann(self.eig_energies, self.eig_states,
                                      d_up_dag, self.beta, self.omega)
        self.GF['Imp G$_0$'] = self.imp_free_gf(e_c, hyb)
        self.GF[r'$\Sigma$'] = 1/self.GF['Imp G$_0$'] - 1/self.GF['Imp G']

    def hyb_V(self):
        """Returns the hybridization parameter :math:`V=\\sqrt{zM_2}`"""
        return np.sqrt(self.imp_z()*self.m2)

    def imp_free_gf(self, e_c, hyb):
        """Outputs the Green's Function of the free propagator of the impurity"""
        hyb2 = hyb**2
        omega = self.omega
        return (omega - e_c + self.mu) / ((omega + self.mu)*(omega - e_c + self.mu) - hyb2)

    def imp_z(self):
        """Calculates the impurity quasiparticle weight from the real part
           of the self enerry"""
        w = self.omega
        sigma = self.GF[r'$\Sigma$']
        if self.freq_axis == 'real':
            dw = w[1]-w[0]
            interval = (-dw <= w) * (w <= dw)
            sigma = sigma.real[interval]
            dsigma = np.polyfit(w[interval], sigma, 1)[0]
            zet = 1/(1 - dsigma)
        else:
            dw = 1/self.beta
            zet = 1/(1 - sigma.imag[0]/dw)
            if sigma.imag[1] > sigma.imag[0]:
                return 0.

        if zet < 1e-3:
            return 0.
        else:
            return zet

    def imp_ocupation(self):
        """gets the ocupation of the impurity"""
        return [self.expected((f.T*f).todense()) for f in self.oper[:2]]

    def interacting_dos(self, mu):
        """Evaluates the interacting density of states"""
        w = self.omega + mu - self.GF[r'$\Sigma$']
        return dos.bethe_lattice(w, self.t)

    def lattice_ocupation(self, mu):
        w = self.omega[self.omega <= 0]
        dosint = 2*simps(self.interacting_dos(mu)[:len(w)], w)
        return dosint

    def find_mu(self, target_n, u_int):
        """Find the required chemical potential to give the required filling"""
        zero = lambda mu: self.lattice_ocupation(mu) - target_n
        self.mu = fsolve(zero, u_int*target_n/2, xtol=5e-3)[0]

    def selfconsitentcy(self, e_c, hyb, target_n, u_int):
        """Performs the selfconsistency loop"""
        convergence = False
        ne_ec = e_c
        while not convergence:
            old = hyb
            old_ec = ne_ec
            self.find_mu(target_n, u_int)
            ne_ec = fsolve(self.restriction, old_ec, (u_int, old), xtol=5e-3)[0]
            self.solve(ne_ec, u_int, hyb)
            hyb = self.hyb_V()
#            print('hyb'+str(hyb))
            convergence = np.abs(old - hyb) < 1e-5\
                and np.abs(self.restriction(ne_ec, u_int, hyb)) < 1e-2

        return ne_ec, hyb

    def restriction(self, e_c, u_int, hyb):
        """Lagrange multiplier in lattice slave spin"""
#        print('e'+str(e_c))
        self.solve(float(e_c), u_int, hyb)
        return np.sum(self.imp_ocupation())-self.lattice_ocupation(self.mu)


def lattice_gf(sim, x=np.linspace(-8, 8, 600), wide=5e-3):
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


def metallic_loop(u_int=np.arange(0, 3.2, 0.05), axis='real',
                  beta=1e5, hop=1., hyb=0.4):
    res = []
    for U in u_int:
        sim = twosite(beta, hop, axis)
        sim.mu = U/2
        for i in range(80):
            old = hyb
            sim.solve(U/2, U, old)
            hyb = sim.hyb_V()
            if 2.5 < U < 3:
                hyb = (hyb + old)/2
            if np.abs(old - hyb) < 1e-5:
                break

        sim.solve(U/2, U, hyb)
        hyb = sim.hyb_V()
        res.append((U, sim.imp_z(), sim))
    return np.asarray(res)

if __name__ == "__main__":
    u = 8
    sim = metallic_loop([u])[0, 2]
    ecc = u/2
    res = []
#    filling = np.arange(1, 0.9, -0.025)
#    for n in filling:
#        old_e = ecc
#        res.append(sim.selfconsitentcy(old_e, sim.hyb_V(), n, u))
#        ecc = res[-1][0]
#
#    res = np.asarray(res)
#    plt.plot(filling, res[:, 0], label='ec')
#    plt.plot(filling, res[:, 1], label='hyb')
#    plt.plot(filling, res[:, 2], label='mu')
