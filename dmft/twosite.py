# -*- coding: utf-8 -*-
"""
Two Site Dynamical Mean Field Theory
====================================
The two site DMFT approach given by M. Potthoff[Potthoff2001]_ on how to
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


def m2_weight(t):
    """Calculates the :math:`M_2^{(0)}=\int dx x^2 \rho_0(x)` which is the
       variance of the non-interacting density of eig_states"""
    second_moment = lambda x: x*x*dos.bethe_lattice(x, t)

    return quad(second_moment, -2*t, 2*t)[0]


class twosite(object):
    """DMFT solver for an impurity and a single bath site"""

    def __init__(self, beta, t, freq_axis, npoints=500):
        """Sets up environment

        Parameters
        ----------
        beta : float
               Inverse temperature of the system
        t : float
            Hopping amplitude between first neighbor lattice sites
        freq_axis: string
                   'real' or 'matsubara' frequencies

         Attributes
         ----------
         GF : dictionary
              Stores the Green functions and self enerry
        """
        self.beta = beta
        self.t = t
        self.m2 = m2_weight(t)
        self.freq_axis = freq_axis

        self.x = np.linspace(-4, 4, npoints)
        if freq_axis == 'real':
            self.omega = self.x
        elif freq_axis == 'matsubara':
            self.omega = 1j*np.arange(1, npoints, 2) / self.beta
        else:
            raise ValueError('Set a working frequency axis')

        self.rho_0 = dos.bethe_lattice(self.x, self.t)
        self.eig_energies = None
        self.eig_states = None
        self.oper = [fermion.destruct(4, index) for index in range(4)]
        self.H_operators = self.hamiltonian()
        self.GF = {}

    def hamiltonian(self):
        """Two site single impurity anderson model
        generate the matrix operators that will be used for this hamiltonian

        .. math:
           \mathcal{H} = -\mu d^\dagger_\sigma d_sigma
           + (\epsilon - \mu) c^\dagger_\sigma c_\sigma +
           U d^\dagger_\\uparrow d_\\uparrow d^\dagger_\downarrow d_\downarrow
           + V(d^\dagger_\sigma c_\sigma + h.c.)"""

        d_up, d_dw, c_up, c_dw = self.oper

        H = {'impurity': d_up.T*d_up + d_dw.T*d_dw,
             'bath': c_up.T*c_up + c_dw.T*c_dw,
             'u_int': d_up.T*d_up*d_dw.T*d_dw,
             'hyb': d_up.T*c_up + d_dw.T*c_dw + c_up.T*d_up + c_dw.T*d_dw}

        return H

    def update_H(self, mu, e_c, u_int, hyb):
        """Updates impurity hamiltonian and diagonalizes it"""
        H = - mu*self.H_operators['impurity'] + \
            (e_c - mu)*self.H_operators['bath'] + \
            u_int*self.H_operators['u_int'] + \
            hyb*self.H_operators['hyb']

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
        return np.sqrt(self.imp_z()*self.m2)

    def imp_free_gf(self, mu, e_c, hyb):
        """Outputs the Green's Function of the free propagator of the impurity"""
        hyb2 = hyb**2
        omega = self.omega
        return (omega - e_c + mu) / ((omega + mu)*(omega - e_c + mu) - hyb2)

    def imp_z(self):
        """Calculates the impurity quasiparticle weight from the real part
           of the self enerry"""
        w = self.omega
        sigma = self.GF['$\Sigma$']
        if self.freq_axis == 'real':
            dw = w[1]-w[0]#0.02
            interval = (-dw <= w) * (w <= dw)
            sigma = sigma.real[interval]
            dsigma = np.polyfit(w[interval], sigma, 1)[0]
            zet = 1/(1 - dsigma)
        else:
            dw = 1/self.beta
            zet = 1/(1 - sigma.imag[0]/dw)

        if zet < 1e-3:
            return 0.
        else:
            return zet

    def imp_ocupation(self):
        """gets the ocupation of the impurity"""
        d_up, d_dw, c_up, c_dw = self.oper
        n_up = self.expected((d_up.T*d_up).todense())
        n_dw = self.expected((d_dw.T*d_dw).todense())

        return n_up, n_dw

    def interacting_dos(self, mu):
        """Evaluates the interacting density of states"""
        w = self.omega + mu - self.GF['$\Sigma$']
        return dos.bethe_lattice(w, self.t)

    def lattice_ocupation(self, mu):
        w = self.omega[self.omega <= 0]
        dosint = 2*simps(self.interacting_dos(mu)[:len(w)], w)

        return dosint


def lattice_gf(sim, mu, wide=5e-3):
    """Compute lattice green function

    .. math: G(\\omega) = \\int \\frac{\\rho_0(x) dx}{\\omega + i\\eta + \\mu + x + \\Sigma(w)}"""
    G = []
    var = sim.omega + mu - sim.GF['$\Sigma$'] + 1j*wide
    for w in var:
        integrable = sim.rho_0/(w - sim.x)
        G.append(simps(integrable, sim.x))

    return np.asarray(G)

def out_plot(sim, spec, label=''):
    w = sim.omega.imag
    stl = '+-'
    if sim.freq_axis == 'real':
        w = sim.omega.real
        stl = '+-'

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
        plt.plot(w, sim.GF[key].real, stl, label='Re {} {}'.format(key, label))
        plt.plot(w, sim.GF[key].imag, stl+'-', label='Im {} {}'.format(key, label))


if __name__ == "__main__":
    res = []
    hyb = 0.4
    for U in [2.9, 3, 3.02]:
        sim = twosite(10000, 0.5, 'real')
        for i in range(80):
            old = hyb
            hyb = sim.solve(U/2, U/2, U, old)
#            if 2.5 < U < 3:
#                hyb = (hyb + old)/2
            if np.abs(old - hyb) < 1e-4:
                break

        hyb = sim.solve(U/2, U/2, U, hyb)

        hyb = sim.solve(U/2, U/2, U, hyb)
        out_plot(sim, 'sigma', 'loop {} hyb {}'.format(i, hyb))
#        gf = lattice_gf(sim, U/2, 8e-3)
#        plt.plot(sim.omega, -1/np.pi*gf.imag, '-',
#                 label='U={}, hyb={:.3f}, Z={:.3f}'.format(U, hyb, sim.imp_z()))
#
#        plt.ylim([-1, 1])

        plt.legend()
        plt.title('U={}, hyb={}, Z={}'.format(U, hyb, sim.imp_z()))
        plt.ylabel('A($\omega$)')
        plt.xlabel('$\omega$')

#        fig.savefig('Sigma_iw_{:.2f}.png'.format(U), format='png',
#                    transparent=False, bbox_inches='tight', pad_inches=0.05)
#        plt.close(fig)
        res.append((U, sim))
