# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 12:44:23 2014

@author: oscar
"""

from __future__ import division, absolute_import, print_function
from dmft.common import greenF, gw_invfouriertrans, gt_fouriertrans,\
 matsubara_freq, fft, ifft
import numpy as np


def test_fast_fourier_transforms(beta=50.):
    """Test the fast fourier transforms"""
    iomega_n = matsubara_freq(beta, neg=True)
    gwr = greenF(iomega_n, mu=0.5)

    g_tau = ifft(gwr, beta)
    g_iomega = fft(g_tau, beta)
    assert (np.abs(gwr - g_iomega) < 5e-13).all()


def test_fourier_trasforms(beta=50., n_tau=1000, n_matsubara=100):
    """Test the tail improved fourier transforms"""
    iomega_n = matsubara_freq(beta, n_matsubara)
    gwr = greenF(iomega_n)[1::2]
    tau = np.linspace(0, beta, n_tau+1)

    g_tau = gw_invfouriertrans(gwr, tau, iomega_n, beta)
    g_iomega = gt_fouriertrans(g_tau, tau, iomega_n, beta)
    assert (np.abs(gwr - g_iomega) < 5e-13).all()
