# -*- coding: utf-8 -*-
r"""
Test over the two impurity rkky interaction
"""
# Created Mon Feb 22 10:45:05 2016
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function
import numpy as np
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
