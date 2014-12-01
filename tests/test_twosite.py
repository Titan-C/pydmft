# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 12:44:23 2014

@author: oscar
"""

from __future__ import division, absolute_import, print_function
from dmft.twosite import twosite, metallic_loop
import numpy as np
import pytest


def test_sigma():
    pass


def test_input():
    with pytest.raises(ValueError):
        twosite(beta=1e5, t=0.5, freq_axis='realax')


def test_mit_real():
    """Test the metal to insulator transition at very low temperature
    calculated in the real axis formalism"""
    z_ref = np.array([1., 0.88889, 0.75, 0.55556, 0.30556, 0.06556, 0.])
    zet = metallic_loop(u_int=[0, 1, 1.5, 2, 2.5, 2.9, 3.05], axis='real',
                        beta=1e5, hop=0.5)[:, 1]
    print(np.abs(zet-z_ref))
    assert (np.abs(zet-z_ref) < 3e-3).all()


def test_matsubara():
    z_ref = np.array([1., 0.85714, 0.48826, 0.03748, 0.01218, 0.00735, 0.00631])
    zet = metallic_loop(u_int=[0, 1, 1.5, 2, 2.5, 2.9, 3.05], axis='matsubara',
                        beta=10, hop=0.5)[:, 1]
    print(np.abs(zet-z_ref))
    assert (np.abs(zet-z_ref) < 1e-5).all()
