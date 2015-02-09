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
from dmft.twosite import TwoSite_Real
from scipy.integrate import simps
from slaveparticles.quantum import dos
from scipy.optimize import root


class TwoSite_Real_Dop(TwoSite_Real):

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
        count = 0
        while not convergence:
            old = hyb
            old_ec = ne_ec
            print('U={}, V={}, e_c={}, ni={}, nl={}'.format(u_int,
                  old, ne_ec, self.ocupations().sum(),
                  self.lattice_ocupation()))
            tuned = root(self.restriction, old_ec,
                         (u_int, old), tol=1e-2)
            ne_ec = float(tuned.x)

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
            if np.abs(ne_ec - old_ec) < 1e-7\
                    and np.abs(self.restriction(ne_ec, u_int, hyb)) > 1e-2:
                ne_ec += 1e-4
                print('jump')
            if self.ocupations().sum() > 1. or ne_ec > mu:
                print('balance from ni={} e_c={}'.format(self.ocupations().sum(), ne_ec))
                ne_ec = mu
            self.solve(ne_ec, u_int, old)
            hyb = self.hyb_V()

            convergence = (np.abs(old - hyb) < 2.5e-5 or hyb < 1e-5)\
                and (np.abs(self.restriction(ne_ec, u_int, hyb)) < 1e-2)

        self.e_c = ne_ec

    def restriction(self, e_c, u_int, hyb):
        """Lagrange multiplier in lattice slave spin"""
        self.solve(float(e_c), u_int, hyb)
        return np.sum(self.ocupations())-self.lattice_ocupation()


def dmft_loop_dop(u_int, mu=None):
    res = []
    sim = TwoSite_Real_Dop()
    sim.e_c = .5
    sim.solve(-15, u_int, 1.)
    if mu is None:
        mu_max = 2.0
        if u_int <= 6:
            mu_max = u_int/2.
        mu = np.linspace(-1.95, mu_max, 80)

    for fmu in mu:
        sim.selfconsistency(sim.e_c, sim.hyb_V(), fmu, u_int)
        print(fmu, u_int, '-'*30)
        res.append([np.sum(sim.ocupations()), copy.deepcopy(sim)])
        if sim.ocupations().sum() >= 1 and u_int >= 6.:
            res[-1][1].solve(sim.e_c, u_int, 1e-2)
            break
        if fmu >= u_int/2:
            break

    return np.asarray(res)

if __name__ == "__main__":
    res = dmft_loop_dop(8)
