# -*- coding: utf-8 -*-
"""
The two site DMFT approach given by M. Potthoff PRB 64, 165114 (2001)

@author: oscar
"""

from __future__ import division, absolute_import, print_function
from slaveparticles.quantum.operators import *

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
