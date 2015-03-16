# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 12:44:23 2014

@author: oscar
"""

from __future__ import division, absolute_import, print_function
from dmft.common import greenF, gw_invfouriertrans, gt_fouriertrans,\
 matsubara_freq
import numpy as np


def test_fourier_trasforms(beta=50., n_tau=1000, n_matsubara=100):
    """Test the tail improved fourier transforms"""
    iomega_n = matsubara_freq(beta, n_matsubara)
    gwr = greenF(iomega_n)
    tau = np.linspace(0, beta, n_tau+1)

    g_tau = gw_invfouriertrans(gwr, tau, iomega_n)
    g_iomega = gt_fouriertrans(g_tau, tau, iomega_n)
    assert np.allclose(gwr, g_iomega)
