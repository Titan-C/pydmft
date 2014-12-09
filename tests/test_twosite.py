# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 12:44:23 2014

@author: oscar
"""

from __future__ import division, absolute_import, print_function
from dmft.twosite import dmft_loop
from dmft.twosite_dop import dmft_loop_dop
import numpy as np
import pytest


def test_sigma():
    pass


def test_mit_real():
    """Test the metal to insulator transition at very low temperature
    calculated in the real axis formalism"""
    z_ref = np.array([1., 0.88889, 0.75, 0.55556, 0.30556, 0.06556, 0.])
    zet = dmft_loop(u_int=[0, 1, 1.5, 2, 2.5, 2.9, 3.05], axis='real',
                        beta=1e5, hop=0.5)[:, 1]
    print(np.abs(zet-z_ref))
    assert (np.abs(zet-z_ref) < 3e-3).all()


def test_matsubara():
    z_ref = np.array([1., 0.88889, 0.75, 0.55556, 0.30556, 0.06556, 0.])
    zet = dmft_loop(u_int=[0, 1, 1.5, 2, 2.5, 2.9, 3.05], axis='matsubara',
                    beta=1e5, hop=0.5)[:, 1]
    print(np.abs(zet-z_ref))
    assert (np.abs(zet-z_ref) < 5e-4).all()


def test_doping():
    e_c_ref = np.array([2.00051, 1.16494, 0.26837, -0.81399, -2.63926])
    V_ref = np.array([0.74534, 0.79271, 0.88327, 0.95445, 0.99037])
    n_ref = np.array([1.00008, 0.85657, 0.66318, .42671, 0.17771])
    res = dmft_loop_dop(u_int=4, e_c=2, hyb=0.74, mu=np.arange(2, -2, -0.8))

    e_c = [sim.e_c for sim in res[:, 1]]
    V = [sim.hyb_V() for sim in res[:, 1]]
    n = [sim.ocupations().sum() for sim in res[:, 1]]

    for ref, test in zip([e_c_ref, V_ref, n_ref], [e_c, V, n]):
        print(np.abs(ref-test))
        assert (np.abs(ref-test) < 5e-5).all()
