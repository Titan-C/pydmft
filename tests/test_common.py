# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 12:44:23 2014

@author: oscar
"""

from __future__ import division, absolute_import, print_function
from dmft.common import greenF, gw_invfouriertrans, gt_fouriertrans,\
 tau_wn_setup
import numpy as np
import pytest

@pytest.mark.parametrize("chempot", [ 0, 0.5, -0.8])
def test_fourier_trasforms(chempot, beta=50., n_tau=2**11, n_matsubara=64):
    """Test the tail improved fourier transforms"""
    parms = {'BETA': beta, 'N_TAU': n_tau, 'N_MATSUBARA': n_matsubara}
    tau, w_n = tau_wn_setup(parms)
    gw = greenF(w_n, mu=chempot)

    for gwr in [gw, np.array([gw, gw])]:
        g_tau = gw_invfouriertrans(gwr, tau, w_n)
        g_iomega = gt_fouriertrans(g_tau, tau, w_n)
        assert np.allclose(gwr, g_iomega)
