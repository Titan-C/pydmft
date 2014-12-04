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
from dmft.twosite import twosite_real
from scipy.integrate import simps
from slaveparticles.quantum import dos
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


class twosite_real_dop(twosite_real):

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

    def selfconsistency(self, e_c, hyb, target_n, u_int):
        """Performs the selfconsistency loop"""
        convergence = False
        ne_ec = e_c
        if target_n == 1:
            ne_ec = u_int / 2
            self.mu = u_int / 2
        while not convergence:
            old = hyb
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

        self.e_c = ne_ec

    def restriction(self, e_c, u_int, hyb):
        """Lagrange multiplier in lattice slave spin"""
        self.solve(float(e_c), u_int, hyb)
        return np.sum(self.ocupations())-self.lattice_ocupation(self.mu)

def doping_config(res, fill):
    fig, axes = plt.subplots(3, sharex=True)
    axes[-1].set_xlabel('$<N>_{imp}$')
    for i, ax, lab in zip(range(3), axes, ['$\\epsilon_c$', 'V', '$\\mu$']):
        ax.plot(fill, res[:, i], label=lab)
        ax.set_ylabel(lab)

import copy
def dmft_loop_dop(u_int=4, e_c=2, hyb=0.74, dop=np.arange(1, 0.015, -0.2)):
    res = []
    sim=twosite_real_dop()
    sim.solve(e_c, u_int, hyb)
    filling = np.arange(1, 0.015, -0.2)
    for n in filling:
#        sim = twosite_real_dop()
        sim.selfconsistency(sim.e_c, sim.hyb_V(), n, u_int)
        res.append([sim.e_c, sim.hyb_V(), sim.mu, copy.deepcopy(sim)])

    return np.asarray(res)

if __name__ == "__main__":
    dop=np.arange(1, 0.015, -0.02)
    res = dmft_loop_dop()

    doping_config(res, dop)
