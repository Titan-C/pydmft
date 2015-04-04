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


@pytest.mark.parametrize("axis, atol", [('real', 3e-3), ('matsubara', 5e-4)])
def test_mit(axis, atol):
    """Test the metal to insulator transition at very low temperature"""
    z_ref = np.array([1., 0.88889, 0.75, 0.55556, 0.30556, 0.06556, 0.])
    zet = dmft_loop(u_int=[0, 1, 1.5, 2, 2.5, 2.9, 3.05], axis=axis,
                    beta=1e5, hop=0.5)[:, 1]
    zet = np.array(zet, dtype=np.float)
    print(np.abs(zet-z_ref))
    assert np.allclose(zet, z_ref, atol=atol)


def test_doping():
    e_c_ref = np.array([-0.98093, -0.23627, 0.38745, 0.95069, 1.48207])
    V_ref = np.array([0.96091, 0.92330, 0.87213, 0.81438, 0.76532])
    n_ref = np.array([0.39486, 0.54847, 0.69036, 0.81363, 0.91498])
    res = dmft_loop_dop(u_int=4, mu=[-.5, 0, 0.5, 1, 1.5])

    e_c = [sim.e_c for sim in res[:, 1]]
    V = [sim.hyb_V() for sim in res[:, 1]]
    n = [sim.ocupations().sum() for sim in res[:, 1]]

    for ref, test in zip([e_c_ref, V_ref, n_ref], [e_c, V, n]):
        print(np.abs(ref-test))
        assert np.allclose(ref, test, atol=5e-5)
