# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 12:44:23 2014

@author: oscar
"""

from __future__ import division, absolute_import, print_function
import numpy as np
from dmft import ipt_imag
from dmft.common import greenF, tau_wn_setup
import pytest


@pytest.mark.parametrize("u_int, result",
 [(1, 1j*np.array([-0.238, -0.231, -0.225, -0.219, -0.213, -0.208, -0.203, -0.198,
       -0.193, -0.188, -0.184, -0.18 , -0.176, -0.172, -0.169, -0.165,
       -0.162, -0.159, -0.156, -0.153, -0.15 , -0.147, -0.145, -0.142,
       -0.139, -0.137, -0.135, -0.133, -0.13 , -0.128, -0.126, -0.124])),
  (2, 1j*np.array([-0.229, -0.223, -0.217, -0.212, -0.207, -0.202, -0.197, -0.192,
       -0.188, -0.184, -0.18 , -0.176, -0.172, -0.169, -0.165, -0.162,
       -0.159, -0.156, -0.153, -0.15 , -0.147, -0.145, -0.142, -0.14 ,
       -0.138, -0.135, -0.133, -0.131, -0.129, -0.127, -0.125, -0.123])),
  (3, 1j*np.array([-0.214, -0.209, -0.205, -0.2  , -0.196, -0.192, -0.187, -0.184,
       -0.18 , -0.176, -0.173, -0.169, -0.166, -0.163, -0.16 , -0.157,
       -0.154, -0.151, -0.149, -0.146, -0.144, -0.141, -0.139, -0.137,
       -0.134, -0.132, -0.13 , -0.128, -0.126, -0.124, -0.122, -0.121]))])
def test_ipt_pm_g(u_int, result, beta=50., n_tau=2**11, n_matsubara=64):
    parms = {'BETA': beta, 'N_TAU': n_tau, 'N_MATSUBARA': n_matsubara,
             't': 0.5, 'MU': 0, 'U': u_int,
             }
    tau, w_n = tau_wn_setup(parms)
    g_iwn0 = greenF(w_n, D=2*parms['t'])
    g_iwn_log, sigma_iwn = ipt_imag.dmft_loop(100, parms['U'], parms['t'], g_iwn0, w_n, tau)

    assert np.allclose(result, g_iwn_log[-1][32:], atol=1e-3 )


