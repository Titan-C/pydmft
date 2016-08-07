# -*- coding: utf-8 -*-
r"""
Test over the two impurity rkky interaction
"""
# Created Mon Feb 22 10:45:05 2016
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function
import numpy as np
import pytest
import dmft.RKKY_dimer as rt
import slaveparticles.quantum.operators as op


def test_sorted_basis():
    """Test sorted fermion matrix operators respect commutation relations"""
    oper = rt.sorted_basis()
    for i in range(4):
        for j in range(4):
            ant = op.anticommutator(oper[i], oper[j].T).todense()
            if i == j:
                assert np.allclose(ant, np.eye(16))
            else:
                assert np.allclose(ant, 0)


@pytest.mark.parametrize("u_int, mu, tp",
                         [(1, 0, 0.3), (2, 1, 0.5), (3, 0.2, 0.7)])
def test_hamiltonian_eigen_energies(u_int, mu, tp):
    """Test local basis and diagonal basis isolated dimer Hamiltonians
       have same energy spectrum"""
    h_loc, _ = rt.dimer_hamiltonian(u_int, mu, tp)
    h_dia, _ = rt.dimer_hamiltonian_diag(u_int, mu, tp)

    eig_e_loc, _ = op.diagonalize(h_loc.todense())
    eig_e_dia, _ = op.diagonalize(h_dia.todense())

    assert np.allclose(eig_e_loc, eig_e_dia)
