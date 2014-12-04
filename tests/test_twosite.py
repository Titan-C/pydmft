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
    assert (np.abs(zet-z_ref) < 1e-5).all()


def test_doping():
    e_c_ref = np.array([2.00007, 0.88531, -0.00766, -0.94985, -2.41781])
    V_ref = np.array([0.74535, 0.82123, 0.90668, 0.95978, 0.98858])
    mu_ref = np.array([1.99956, 0.94027, 0.17653, -0.48374, -1.12271])
    res = dmft_loop_dop(u_int=4, e_c=2, hyb=0.74, dop=np.arange(1, 0.015, -0.2))

    for i, ref in enumerate([e_c_ref, V_ref, mu_ref]):
        print(np.abs(ref-res[:, i]))
        assert (np.abs(ref-res[:, i]) < 5e-5).all()

test_doping()
