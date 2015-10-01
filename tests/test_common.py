# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 12:44:23 2014

@author: oscar
"""

from __future__ import division, absolute_import, print_function
from dmft.common import greenF, gw_invfouriertrans, gt_fouriertrans,\
 tau_wn_setup, matsubara_freq, fit_gf
import numpy as np
import pytest


@pytest.mark.parametrize("chempot", [0, 0.5, -0.8, 4.])
def test_fourier_trasforms(chempot, beta=50., n_tau=2**11, n_matsubara=115):
    """Test the tail improved fourier transforms"""
    parms = {'BETA': beta, 'N_TAU': n_tau, 'N_MATSUBARA': n_matsubara}
    tau, w_n = tau_wn_setup(parms)
    giw = greenF(w_n, mu=chempot)

    for gwr in [giw, np.array([giw, giw])]:
        g_tau = gw_invfouriertrans(gwr, tau, w_n, [1., -chempot, 0.25])
        g_iomega = gt_fouriertrans(g_tau, tau, w_n)
        assert np.allclose(gwr, g_iomega)


def test_fit_gf():
    """Test the interpolation of Green function in Bethe Lattice"""
    w_n = matsubara_freq(100, 3)
    giw = greenF(w_n).imag
    cont_g = fit_gf(w_n, giw)
    assert abs(cont_g(0.) + 2.) < 1e-4
