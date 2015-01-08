# -*- coding: utf-8 -*-
"""
Two Site Dynamical Mean Field Theory
====================================
The two site DMFT approach given by M. Potthoff [Potthoff2001]_ on how to
treat the impurity bath as a sigle site of the DMFT. Work is around a single
impurity Anderson model.

.. [Potthoff2001] M. Potthoff PRB, 64, 165114, 2001

DMFT solver for an impurity and a single bath site

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
         Stores the Green functions and self energy
"""

from __future__ import division, absolute_import, print_function
import numpy as np
from scipy.integrate import quad
from slaveparticles.quantum.operators import gf_lehmann, diagonalize, expected_value
from slaveparticles.quantum import dos, fermion
from scipy.integrate import simps
from scipy.optimize import root


def m2_weight(t):
    """Calculates the :math:`M_2^{(0)}=\\int  x^2 \\rho_0(x)dx` which is the
       variance of the non-interacting density of states of a Bethe Lattice"""
    second_moment = lambda x: x*x*dos.bethe_lattice(x, t)

    return quad(second_moment, -2*t, 2*t)[0]


class TwoSite(object):
    """Base class for a two site DMFT solver"""

    def __init__(self, beta, t):
        self.beta = beta
        self.t = t
        self.m2 = m2_weight(t)
        self.mu = 0.
        self.e_c = 0.

        self.eig_energies = None
        self.eig_states = None
        self.oper = [fermion.destruct(4, index) for index in range(4)]
        self.H_operators = self.hamiltonian()
        self.GF = {}

    def hamiltonian(self):
        r"""Two site single impurity anderson model
        generate the matrix operators that will be used for this hamiltonian

        .. math::
           \mathcal{H} = -\mu d^\dagger_\sigma d_\sigma
           + (\epsilon_c - \mu) c^\dagger_\sigma c_\sigma +
           U d^\dagger_\uparrow d_\uparrow d^\dagger_\downarrow d_\downarrow
           + V(d^\dagger_\sigma c_\sigma + h.c.)"""

        d_up, d_dw, c_up, c_dw = self.oper

        H = {'impurity': np.array([d_up.T*d_up, d_dw.T*d_dw]),
             'bath': np.array([c_up.T*c_up, c_dw.T*c_dw]),
             'u_int': d_up.T*d_up*d_dw.T*d_dw,
             'hyb': np.array([d_up.T*c_up + c_up.T*d_up,
                              d_dw.T*c_dw + c_dw.T*d_dw])}

        return H

    def update_H(self, e_c, u_int, hyb):
        """Updates impurity hamiltonian and diagonalizes it"""
        H = u_int*self.H_operators['u_int'] + \
            np.sum(- self.mu*self.H_operators['impurity'] +
                   (e_c - self.mu)*self.H_operators['bath'] +
                   hyb*self.H_operators['hyb'])

        self.eig_energies, self.eig_states = diagonalize(H.todense())

    def expected(self, observable):
        """Wrapper to the expected_value function to fix the eigenbasis"""
        return expected_value(observable, self.eig_energies, self.eig_states,
                              self.beta)

    def imp_free_gf(self, e_c, hyb):
        """Outputs the Green's Function of the free propagator
        of the impurity"""
        hyb2 = hyb.reshape(-1, 1)**2
        omega = self.omega
        g_inv = omega + self.mu - hyb2/(omega - e_c.reshape(-1, 1) + self.mu)
        return 1 / g_inv

    def solve(self, e_c, u_int, hyb):
        """Solves the impurity problem"""
        self.update_H(e_c, u_int, hyb)
        self.GF['Imp G'] = np.asarray([gf_lehmann(self.eig_energies, self.eig_states,
                                      d.T, self.beta, self.omega) for d in self.oper[:2]])

        self.GF['Imp G$_0$'] = self.imp_free_gf(e_c, hyb)
        self.GF[r'$\Sigma$'] = 1/self.GF['Imp G$_0$'] - 1/self.GF['Imp G']

    def hyb_V(self):
        """Returns the hybridization parameter :math:`V=\\sqrt{zM_2}`"""
        return np.sqrt(self.imp_z()*self.m2)

    def ocupations(self, top=2):
        """gets the ocupation of the impurity"""
        return np.asarray([self.expected((f.T*f).todense())
                           for f in self.oper[:top]])

    def double_ocupation(self):
        """Calculates the double ocupation of the impurity"""
        d_up, d_dw = self.oper[:2]
        return self.expected((d_up.T*d_up*d_dw.T*d_dw).todense())


class TwoSite_Real(TwoSite):
    """DMFT solver in the real axis"""

    def __init__(self, beta=1e5, t=1, omega=np.linspace(-9, 9, 1200)):
        super(TwoSite_Real, self).__init__(beta, t)

        self.omega = omega

        self.rho_0 = dos.bethe_lattice(self.omega, self.t)

        self.solve(np.zeros(2), 0, np.zeros(2))

    def imp_z(self):
        """Calculates the impurity quasiparticle weight from the real part
           of the self energy"""
        w = self.omega
        dw = w[1]-w[0]
        interval = (-dw <= w) * (w <= dw)

        sigma = self.GF[r'$\Sigma$'].real[:, interval]
        dsigma = sigma[:, -1]-sigma[:, 0]
        zet = 1/(1 - dsigma/dw)

        zet[zet < 1e-3] = 0.
        return zet

    def interacting_dos(self):
        """Evaluates the interacting density of states"""
        w = self.omega + self.mu - self.GF[r'$\Sigma$']
        return dos.bethe_lattice(w, self.t)

    def lattice_ocupation(self):
        ef_cut = len(self.omega)/2+1
        w = np.copy(self.omega[:ef_cut])
        intdos = self.interacting_dos()[:, :len(w)]
        w[-1] = 0
        intdos[:, -1] = (intdos[:, -1] + intdos[:, -2])/2
        dosint = simps(intdos, w)
        return dosint

    def selfconsistency(self, e_c, hyb, mu, u_int):
        """Performs the selfconsistency loop"""
        convergence = False
        ne_ec = e_c
        self.mu = mu
        count = 0
        while not convergence:
            old = hyb
            old_ec = ne_ec
            print('U={}, V={}, e_c={}, ni={}, nl={}'.format(u_int,
                  old, ne_ec, self.ocupations().sum(),
                  self.lattice_ocupation()))
            tuned = root(self.restriction, old_ec,
                         (u_int, old), tol=1e-2)
            ne_ec = tuned.x

            print('aoe')
            if not tuned.success:
                ne_ec = old_ec
                self.solve(ne_ec, u_int, old)
                print('fail on U={}, V={}, e_c={}, ni={}, nl={}'.format(u_int,
                      old, ne_ec, self.ocupations().sum(), self.lattice_ocupation()))
                if self.hyb_V() < 1e-5:
                    break
                print('stuck'*20)
                if count > 6:
                    count += 1
                    print('exiting')
                    break
#            if np.abs(ne_ec - old_ec) < 1e-7\
#                    and np.abs(self.restriction(ne_ec, u_int, hyb)) > 1e-2:
#                ne_ec += 1e-4
#                print('jump')
#            if self.ocupations().sum() > 1. or ne_ec > mu:
#                print('balance from ni={} e_c={}'.format(self.ocupations().sum(), ne_ec))
#                ne_ec = mu
            self.solve(ne_ec, u_int, old)
            hyb = self.hyb_V()

            convergence = ((np.abs(old - hyb) < 2.5e-5).all() or (hyb < 1e-5).all())\
                and (np.abs(self.restriction(ne_ec, u_int, hyb)) < 1e-2).all()

        self.e_c = ne_ec

    def restriction(self, e_c, u_int, hyb):
        """Lagrange multiplier in lattice slave spin"""
        self.solve(e_c, u_int, hyb)
        print(e_c,hyb,self.ocupations())
        return self.ocupations()-self.lattice_ocupation()


def dmft_loop(u_int=np.arange(0, 3.2, 0.05), axis='real',
              beta=1e5, hop=0.5, hyb=0.4):
    """Perform a DMFT loop for the half-filled case of the
    Two site formulation"""
    res = []
    if axis == 'real':
        solver = TwoSite_Real

    for U in u_int:
        sim = solver(beta, hop)
        sim.mu = U/2
        convergence = False
        while not convergence:
            old = hyb
            sim.solve(U/2, U, old)
            hyb = sim.hyb_V()
            hyb = (hyb + old)/2
            convergence = np.abs(old - hyb) < 1e-5

        print(U, hyb, sim.ocupations())
        sim.solve(U/2, U, hyb)
        hyb = sim.hyb_V()
        res.append((U, sim.imp_z(), sim))
    return np.asarray(res)

if __name__ == "__main__":
    u = np.arange(0, 3.2, 0.1)
    sim = TwoSite_Real()
    u=2.5
    print(sim.imp_z())
    sim.selfconsistency(np.array([3.5,-2.25]),np.array([0.866,0.95689]),u/2.,u)
    wi=sim.omega+sim.mu-sim.GF[r'$\Sigma$']
    rho=dos.bethe_lattice(wi,1.)
    plt.plot(rho.T)

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
