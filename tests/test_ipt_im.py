# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 12:44:23 2014

@author: oscar
"""

from __future__ import division, absolute_import, print_function
import numpy as np
from dmft import ipt_imag
from dmft.common import greenF, tau_wn_setup
import dmft.RKKY_dimer as dimer
import pytest
import tempfile
import shutil
import os


ipt_ref_res = \
 [(1., 1j*np.array([-1.837, -1.572, -1.365, -1.201, -1.07 , -0.962, -0.873, -0.798,
       -0.734, -0.679, -0.631, -0.589, -0.552, -0.52 , -0.49 , -0.464,
       -0.44 , -0.418, -0.399, -0.381, -0.364, -0.349, -0.335, -0.322,
       -0.31 , -0.299, -0.288, -0.279, -0.269, -0.261, -0.253, -0.245,
       -0.238, -0.231, -0.225, -0.219, -0.213, -0.208, -0.203, -0.198,
       -0.193, -0.188, -0.184, -0.18 , -0.176, -0.172, -0.169, -0.165,
       -0.162, -0.159, -0.156, -0.153, -0.15 , -0.147, -0.145, -0.142,
       -0.139, -0.137, -0.135, -0.133, -0.13 , -0.128, -0.126, -0.124])),
  (2., 1j*np.array([-1.667, -1.25 , -1.016, -0.875, -0.781, -0.714, -0.661, -0.617,
       -0.581, -0.548, -0.519, -0.493, -0.47 , -0.448, -0.428, -0.409,
       -0.392, -0.376, -0.362, -0.348, -0.335, -0.323, -0.312, -0.301,
       -0.291, -0.282, -0.273, -0.264, -0.257, -0.249, -0.242, -0.235,
       -0.229, -0.223, -0.217, -0.212, -0.207, -0.202, -0.197, -0.192,
       -0.188, -0.184, -0.18 , -0.176, -0.172, -0.169, -0.165, -0.162,
       -0.159, -0.156, -0.153, -0.15 , -0.147, -0.145, -0.142, -0.14 ,
       -0.138, -0.135, -0.133, -0.131, -0.129, -0.127, -0.125, -0.123])),
  (3., 1j*np.array([-0.047, -0.132, -0.202, -0.256, -0.294, -0.322, -0.339, -0.35 ,
       -0.356, -0.358, -0.357, -0.354, -0.349, -0.343, -0.336, -0.329,
       -0.321, -0.313, -0.306, -0.298, -0.29 , -0.283, -0.275, -0.268,
       -0.261, -0.255, -0.248, -0.242, -0.236, -0.23 , -0.225, -0.22 ,
       -0.214, -0.209, -0.205, -0.2  , -0.196, -0.192, -0.187, -0.184,
       -0.18 , -0.176, -0.173, -0.169, -0.166, -0.163, -0.16 , -0.157,
       -0.154, -0.151, -0.149, -0.146, -0.144, -0.141, -0.139, -0.137,
       -0.134, -0.132, -0.13 , -0.128, -0.126, -0.124, -0.122, -0.121]))]


def test_GF():
    """Test the Matrix product and Matrix inversion of a 2x2 Matrix
    which for the case of the dimer has a symmetric structure in which
    only requires the first row"""

    tau, w_n = tau_wn_setup(dict(BETA=100., N_MATSUBARA=400))
    giwd, giwo = dimer.gf_met(w_n, 0., 0.25, 0.5, 0.)
    inv_giwd, inv_giwo = dimer.mat_inv(giwd, giwo)
    one, zero = dimer.mat_mul(inv_giwd, inv_giwo, giwd, giwo)
    assert np.allclose(one, 1.)
    assert np.allclose(zero, 0.)


@pytest.mark.parametrize("tp", [0., 0.2, -0.6])
def test_selfconsistency(tp):
    """Check that the Bethe lattice self-consistency is working"""
    tau, w_n = tau_wn_setup(dict(BETA=100., N_MATSUBARA=400))
    giwd, giwo = dimer.gf_met(w_n, 0., tp, 0.5, 0.)
    g0iwd, g0iwo = dimer.self_consistency(1j*w_n, giwd, giwo, 0., tp, 0.25)
    assert np.allclose(giwd, g0iwd)
    assert np.allclose(giwo, g0iwo)



@pytest.mark.parametrize("u_int, result", ipt_ref_res)
def test_ipt_pm_g(u_int, result, beta=50., n_matsubara=64):
    parms = {'BETA': beta, 'N_MATSUBARA': n_matsubara,
             't': 0.5, 'MU': 0, 'U': u_int,
             }
    tau, w_n = tau_wn_setup(parms)
    g_iwn0 = greenF(w_n, D=2*parms['t'])
    g_iwn, _ = ipt_imag.dmft_loop(parms['U'], parms['t'], g_iwn0, w_n, tau)

    assert np.allclose(result, g_iwn, atol=3e-3)


@pytest.mark.parametrize("u_int, result", ipt_ref_res)
def test_ipt_dimer_pm_g(u_int, result, beta=50.):
    tau, w_n = tau_wn_setup(dict(BETA=beta, N_MATSUBARA=256))
    giw_d, giw_o = dimer.gf_met(w_n, 0., 0, 0.5, 0.)
    giw_d = dimer.ipt_dmft_loop(beta, u_int, 0, giw_d, giw_o)[0][:64]

    assert np.allclose(result, giw_d, atol=3e-3)
