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
import copy
from dmft.twosite import twosite_real
from scipy.integrate import simps
from slaveparticles.quantum import dos
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


class twosite_real_dop(twosite_real):

    def interacting_dos(self):
        """Evaluates the interacting density of states"""
        w = self.omega + self.mu - self.GF[r'$\Sigma$']
        return dos.bethe_lattice(w, self.t)

    def lattice_ocupation(self):
        w = np.copy(self.omega[:len(self.omega)/2+1])
        intdos = self.interacting_dos()[:len(w)]
        w[-1] = 0
        intdos[-1] = (intdos[-1] + intdos[-2])/2
        dosint = 2*simps(intdos, w)
        return dosint

    def selfconsistency(self, e_c, hyb, mu, u_int):
        """Performs the selfconsistency loop"""
        convergence = False
        ne_ec = e_c
        self.mu = mu
        while not convergence:
            old = hyb
            old_ec = ne_ec
            ne_ec = fsolve(self.restriction, old_ec,
                           (u_int, old), xtol=1e-2)[0]
            if np.abs(ne_ec-old_ec)< 1e-6:
                ne_ec+=1e-3
            self.solve(ne_ec, u_int, hyb)
            hyb = (self.hyb_V() + old)/2
            convergence = np.abs(old - hyb) < 1e-5\
                and np.abs(self.restriction(ne_ec, u_int, hyb)) < 1e-2

        self.e_c = ne_ec

    def restriction(self, e_c, u_int, hyb):
        """Lagrange multiplier in lattice slave spin"""
        self.solve(float(e_c), u_int, hyb)
        return np.sum(self.ocupations())-self.lattice_ocupation()


def dmft_loop_dop(u_int=4, e_c=2, hyb=0.74, mu=np.arange(2, -2, -0.05)):
    res = []
    sim = twosite_real_dop()
    sim.mu = mu[0]
    sim.e_c = e_c
    sim.solve(e_c, u_int, hyb)
    for fmu in mu:
        sim.selfconsistency(sim.e_c, sim.hyb_V(), fmu, u_int)
        res.append([np.sum(sim.ocupations()), copy.deepcopy(sim)])

    return np.asarray(res)

if __name__ == "__main__":
    mu = np.arange(2, -2, -0.05)
    try:
        res = np.load('dopU4.npy')
    except IOError:
        res = dmft_loop_dop(mu=mu)
        np.save('dopU4', res)
