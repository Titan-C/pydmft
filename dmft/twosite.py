# -*- coding: utf-8 -*-
"""
Two Site Dynamical Mean Field Theory
====================================

The two site DMFT approach given by M. Potthoff [Potthoff2001]_ on how to
treat the impurity bath as a sigle site of the DMFT. Work is around a single
impurity Anderson model.

.. [Potthoff2001] M. Potthoff PRB, 64, 165114, 2001

DMFT solver for an impurity and a single bath site

"""

from __future__ import division, absolute_import, print_function
import numpy as np
from scipy.integrate import quad
from slaveparticles.quantum.operators import gf_lehmann, diagonalize, expected_value
from slaveparticles.quantum import dos, fermion
from dmft.common import matsubara_freq, gw_invfouriertrans


def m2_weight(t):
    """Calculates the :math:`M_2^{(0)}=\\int  x^2 \\rho_0(x)dx` which is the
       variance of the non-interacting density of states of a Bethe Lattice"""
    def second_moment(x): return x * x * dos.bethe_lattice(x, t)

    return quad(second_moment, -2 * t, 2 * t)[0]


class TwoSite(object):
    """Base class for a two site DMFT solver
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

    def __init__(self, beta, t):
        self.beta = beta
        self.t = t
        self.m2 = m2_weight(t)
        self.mu = 0.
        self.e_c = 0.
        self.omega = None

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

        H = {'impurity': d_up.T * d_up + d_dw.T * d_dw,
             'bath': c_up.T * c_up + c_dw.T * c_dw,
             'u_int': d_up.T * d_up * d_dw.T * d_dw,
             'hyb': d_up.T * c_up + d_dw.T * c_dw + c_up.T * d_up + c_dw.T * d_dw}

        return H

    def update_H(self, e_c, u_int, hyb):
        """Updates impurity hamiltonian and diagonalizes it"""
        H = - self.mu * self.H_operators['impurity'] + \
            (e_c - self.mu) * self.H_operators['bath'] + \
            u_int * self.H_operators['u_int'] + \
            hyb * self.H_operators['hyb']

        self.eig_energies, self.eig_states = diagonalize(H.todense())

    def expected(self, observable):
        """Wrapper to the expected_value function to fix the eigenbasis"""
        return expected_value(observable, self.eig_energies, self.eig_states,
                              self.beta)

    def imp_free_gf(self, e_c, hyb):
        """Outputs the Green's Function of the free propagator
        of the impurity"""
        hyb2 = hyb**2
        omega = self.omega
        return (omega - e_c + self.mu) / \
               ((omega + self.mu) * (omega - e_c + self.mu) - hyb2)

    def solve(self, e_c, u_int, hyb):
        """Solves the impurity problem"""
        self.update_H(e_c, u_int, hyb)
        d_up_dag = self.oper[0].T
        self.GF['Imp G'] = gf_lehmann(self.eig_energies, self.eig_states,
                                      d_up_dag, self.beta, self.omega)
        self.GF['Imp G$_0$'] = self.imp_free_gf(e_c, hyb)
        self.GF[r'$\Sigma$'] = 1 / self.GF['Imp G$_0$'] - 1 / self.GF['Imp G']

    def hyb_V(self):
        """Returns the hybridization parameter :math:`V=\\sqrt{zM_2}`"""
        return np.sqrt(self.imp_z() * self.m2)

    def ocupations(self, top=2):
        """gets the ocupation of the impurity"""
        return np.asarray([self.expected((f.T * f).todense())
                           for f in self.oper[:top]])

    def double_ocupation(self):
        """Calculates the double ocupation of the impurity"""
        d_up, d_dw = self.oper[:2]
        return self.expected((d_up.T * d_up * d_dw.T * d_dw).todense())


class TwoSite_Real(TwoSite):
    """DMFT solver in the real axis"""

    def __init__(self, beta=1e5, t=1, omega=np.linspace(-6, 6, 1200)):
        super(TwoSite_Real, self).__init__(beta, t)

        self.omega = omega

        self.rho_0 = dos.bethe_lattice(self.omega, self.t)

        self.solve(0, 0, 0)

    def imp_z(self):
        """Calculates the impurity quasiparticle weight from the real part
           of the self energy"""
        w = self.omega
        dw = w[1] - w[0]
        interval = (-dw <= w) * (w <= dw)

        sigma = self.GF[r'$\Sigma$'].real[interval]
        dsigma = np.polyfit(w[interval], sigma, 1)[0]
        zet = 1 / (1 - dsigma)

        if zet < 1e-3:
            return 0.
        else:
            return zet


def matsubara_Z(im_sigma, beta):
    """Calculates the impurity quasiparticle weight from the imaginary
    part of the self energy in the matsubara frequencies"""

    if im_sigma[1] > im_sigma[0]:
        return 0.

    dw = np.pi / beta
    zet = 1 / (1 - im_sigma[0] / dw)
    return zet


class TwoSite_Matsubara(TwoSite):
    """DMFT solver on the matsubara frequency axis"""

    def __init__(self, beta=100, t=1, nfreq=20):
        super(TwoSite_Matsubara, self).__init__(beta, t)

        self.omega = 1j * matsubara_freq(beta, nfreq)
        self.solve(0, 0, 0)

    def imp_z(self):
        return matsubara_Z(self.GF[r'$\Sigma$'].imag, self.beta)


def refine_mat_solution(end_solver, u_int):
    """Takes the end converged dmft solver in matsubara frequencies
    and increases the range of the matsubara frequencies to get
    nicer plots"""

    beta = end_solver.beta
    sim = TwoSite_Matsubara(beta, end_solver.t, beta)
    sim.omega = 1j * matsubara_freq(beta)
    sim.mu = end_solver.mu
    sim.solve(u_int / 2, u_int, end_solver.hyb_V())

    return sim


def dmft_loop(u_int=np.arange(0, 3.2, 0.05), axis='real',
              beta=1e5, hop=0.5, hyb=0.4):
    """Perform a DMFT loop for the half-filled case of the
    Two site formulation"""
    res = []
    if axis == 'real':
        solver = TwoSite_Real
    if axis == 'matsubara':
        solver = TwoSite_Matsubara

    for U in u_int:
        sim = solver(beta, hop)
        sim.mu = U / 2
        convergence = False
        while not convergence:
            old = hyb
            sim.solve(U / 2, U, old)
            hyb = sim.hyb_V()
            hyb = (hyb + old) / 2
            convergence = np.abs(old - hyb) < 1e-5

        print(U, hyb, sim.ocupations())
        sim.solve(U / 2, U, hyb)
        hyb = sim.hyb_V()
        res.append((U, sim.imp_z(), sim))
    return np.asarray(res)


if __name__ == "__main__":
    u = [0.1, 1, 2, 2.7, 3.3]  # np.arange(0, 3.2, 0.1)
    sim = dmft_loop(u, beta=50, axis='matsubara')
    for s in sim:
        out = refine_mat_solution(s[2], s[0])
        plt.plot(out.omega.imag,
                 out.GF[r'$\Sigma$'].imag, '+-', label='U={}'.format(s[0]))
    plt.xlim([-5, 5])
    plt.figure()

    tau = np.linspace(0, 50, 1001)
    for s in sim:
        out = refine_mat_solution(s[2], s[0])
        Gw = out.GF['Imp G']
        gmt = gw_invfouriertrans(Gw, tau, out.omega.imag)
        plt.semilogy(tau, -gmt, label='U={}'.format(s[0]))
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
