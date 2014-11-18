# -*- coding: utf-8 -*-
"""
The two site DMFT approach given by M. Potthoff PRB 64, 165114 (2001)

@author: oscar
"""

from __future__ import division, absolute_import, print_function
from slaveparticles.quantum.operators import gf_lehmann, diagonalize, expected_value
import numpy as np
from scipy.integrate import simps
from slaveparticles.quantum import dos, fermion
import matplotlib.pyplot as plt


def m2_weight(t, g):
    x = np.linspace(-2*t, 2*t, g)
    return simps(x*x*dos.bethe_lattice(x, t), x)


class twosite(object):
    """DMFT solver for an impurity and a single bath site"""

    def __init__(self, beta, t, freq_axis):
        """Generates Operators and sets up environment"""
        self.beta = beta
        self.t = t
        self.m2 = m2_weight(t, 200)
        self.freq_axis = freq_axis

        if freq_axis == 'real':
            self.omega = np.linspace(-4, 4, 3500) + 8e-2j  # Real axis
        elif freq_axis == 'matsubara':
            self.omega = 1j*np.arange(1, 3500, 2) / self.beta   # Matsubara freq
        else:
            raise ValueError('Set an working frequency axis')

        self.eig_energies = None
        self.eig_states = None
        self.oper = [fermion.destruct(4, index) for index in range(4)]
        self.GF = {}

    def hamiltonian(self, mu, e_c, u_int, hyb):
        """Two site single impurity anderson model"""
        d_up, d_dw, c_up, c_dw = self.oper
        return - mu*(d_up.T*d_up + d_dw.T*d_dw) + \
            (e_c - mu)*(c_up.T*c_up + c_dw.T*c_dw) + \
            u_int*d_up.T*d_up*d_dw.T*d_dw + \
            hyb*(d_up.T*c_up + d_dw.T*c_dw + c_up.T*d_up + c_dw.T*d_dw)

    def update_H(self, mu, e_c, u_int, hyb):
        """Updates impurity hamiltonian and diagonalizes it"""
        H = self.hamiltonian(mu, e_c, u_int, hyb)
        self.eig_energies, self.eig_states = diagonalize(H.todense())

    def expected(self, observable):
        """Wrapper to the expected_value function to fix the eigenbasis"""
        return expected_value(observable, self.eig_energies, self.eig_states,
                              self.beta)

    def solve(self, mu, e_c, u_int, hyb):
        """Solves the impurity problem"""
        self.update_H(mu, e_c, u_int, hyb)
        d_up_dag = self.oper[0].T
        self.GF['Imp G'] = gf_lehmann(self.eig_energies, self.eig_states,
                                      d_up_dag, self.beta, self.omega)
        self.GF['Imp G$_0$'] = self.imp_free_gf(mu, e_c, hyb)
        self.GF['$\Sigma$'] = 1/self.GF['Imp G$_0$'] - 1/self.GF['Imp G']
        self.GF['Lat G'] = self.lattice_gf(mu)
        return np.sqrt(self.imp_z()*self.m2)

    def imp_ocupation(self):
        """gets the ocupation of the impurity"""
        d_up, d_dw, c_up, c_dw = self.oper
        n_up = self.expected((d_up.T*d_up).todense())
        n_dw = self.expected((d_dw.T*d_dw).todense())

        return n_up, n_dw

    def imp_free_gf(self, mu, e_c, hyb):
        """Outputs the Green's Function of the free propagator of the impurity"""
        hyb2 = hyb**2
        omega = self.omega
        return (omega - e_c + mu) / ((omega + mu)*(omega - e_c + mu) - hyb2)

    def imp_z(self):
        """Calculates the impurity quasiparticle weight from the real part
           of the self enerry"""
        w = self.omega.real
        sigma = self.GF['$\Sigma$']
        if self.freq_axis == 'real':
            dw = w[1]-w[0]
            return 1/(1 - np.mean(np.gradient(sigma.real, dw)[(-dw <= w) * (w <= dw)]))
        else:
            dw = 2/self.beta
            return 1/(1 -np.gradient(sigma.imag[:5], dw)[0])

    def lattice_gf(self, mu):
        """Compute lattice green function"""
        G = []
        x = np.linspace(-4, 4, 3500)
        betlat = dos.bethe_lattice(x, self.t)
        for w, s_w in zip(self.omega, self.GF['$\Sigma$']):
            integrable = betlat/(w + mu - x - s_w)
            G.append(simps(integrable, x))

        return np.asarray(G)

    def lattice_ocupation(latG, w):
        return -2*simps(latG.imag[w.real <= 0], w.real[w.real <= 0])/np.pi


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
            key = '$\Sigma$'
        if 'G' == gfp:
            key = 'Lat G'
        if 'A' == gfp:
            plt.plot(w, -1/np.pi*sim.GF['Lat G'].imag, stl, label='A '+label)
            continue
#        plt.plot(w, sim.GF[key].real, stl, label='Re {} {}'.format(key, label))
        plt.plot(w, sim.GF[key].imag, stl, label='Im {} {}'.format(key, label))


if __name__ == "__main__":
    res = []
    for U in np.linspace(-4, 4, 3500):
#
        sim = twosite(80, 0.5, 'matsubara')
        hyb = 0.3#np.sqrt(sim.imp_z()*sim.m2)
        f=plt.figure()
        for i in range(12):
            hyb = sim.solve(U/2, U/2, U, hyb)
            out_plot(sim, 'sigma', 'loop {}'.format(i))
        plt.legend()
        plt.title('U={}, hyb={}'.format(U,hyb))
        plt.ylabel('A($\omega$)')
        plt.xlabel('$\omega$')
        f.savefig('Sigma_iw_{:.2f}.png'.format(U), format='png',
              transparent=False, bbox_inches='tight', pad_inches=0.05)
        plt.close(f)
        res.append((U, sim))
