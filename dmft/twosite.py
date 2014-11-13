# -*- coding: utf-8 -*-
"""
The two site DMFT approach given by M. Potthoff PRB 64, 165114 (2001)

@author: oscar
"""

from __future__ import division, absolute_import, print_function
from slaveparticles.quantum.operators import *
import numpy as np

d_up, d_dw, c_up, c_dw = [f_destruct(4, index) for index in range(4)]


def hamiltonian(e_d, mu, e_c, u_int, hyb):
    """Two site single inpurity anderson model"""

    return (e_d - mu)*(d_up.T*d_up + d_dw.T*d_dw) + \
        (e_c - mu)*(c_up.T*c_up + c_dw.T*c_dw) + \
        u_int*d_up.T*d_up*d_dw.T*d_dw + \
        hyb*(d_up.T*c_up + d_dw.T*c_dw + c_up.T*d_up + c_dw.T*d_dw)


def update_H(e_d, mu, e_c, u_int, hyb):

    H = hamiltonian(e_d, mu, e_c, u_int, hyb)
    eig_e, eig_states = diagonalize(H.todense())
    return eig_e, eig_states


def ocupation(eig_e, eig_states):
    """gets the ocupation of the impurity"""

    n_up = expected_value((d_up.T*d_up).todense(), eig_e, eig_states, 1e5)
    n_dw = expected_value((d_dw.T*d_dw).todense(), eig_e, eig_states, 1e5)

    return n_up, n_dw


def lehmann(eig_e, eig_states, d_dag, beta):
    """Outputs the lehmann representation of the greens function"""
    omega = np.linspace(-9, 11, 3500) +4e-3j
    zet = partition_func(beta, eig_e)
    G = 0
    for i in range(len(eig_e)):
        for j in range(len(eig_e)):
            G += np.dot(eig_states[:, j].T, d_dag.dot(eig_states[:, i])) * \
                 (np.exp(-beta*eig_e[i]) + np.exp(-beta*eig_e[j])) / \
                 (omega + eig_e[i] - eig_e[j])
    return omega, G / zet


def free_green(e_d, mu, e_c, hyb, omega):
    """Outputs the Green's Function of the free propagator of the impurity"""
    hyb2 = hyb**2
    return (omega - e_c + mu) / ((omega - e_d + mu)*(omega - e_c + mu) - hyb2)


from scipy.integrate import simps
from slaveparticles.quantum import dos
def lattice_green(mu, sigma, omega):
    """Compute lattice green function"""
    G = []
    for w, s_w in zip(omega, sigma):
        integrable = lambda x: dos.bethe_lattice(x, 2)/(w + mu - x - s_w)
        G.append(simps(integrable(omega), omega))

    return np.asarray(G)

def lattice_ocupation(latG, w):
    return simps(latG.imag[w.real<=0],w.real[w.real<=0])/np.pi

e_d, mu, e_c, u_int, hyb = 0,0,0,4,1
E,V=update_H(e_d, mu, e_c, u_int, hyb)
w,G = lehmann(E,V,f_destruct(4,0).T,1e5)
G_0=free_green(e_d, mu, e_c, hyb, w)
sigma=1/G_0 - 1/G
latG=lattice_green(mu,sigma, w)
plt.plot(w.real,G,w.real,G_0)
plt.figure()
plt.plot(w.real,latG.real,w.real,latG.imag,w.real,dos.bethe_lattice(w.real,2))
