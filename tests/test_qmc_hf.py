# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 12:44:23 2014

@author: oscar
"""

from __future__ import division, absolute_import, print_function
import numpy as np
import dmft.hirschfye as hf
import dmft.common as gf
import pytest


@pytest.mark.parametrize("chempot, u_int", [ (0,2), (0.5,2.3)])
def test_hf_fast_updatecond(chempot, u_int, beta=16.,
                            n_tau=2**11, n_matsubara=64):
    parms = {'BETA': beta, 'N_TAU': n_tau, 'N_MATSUBARA': n_matsubara}
    tau, w_n = gf.tau_wn_setup(parms)
    gw = gf.greenF(w_n, mu=chempot)
    g0t = gf.gw_invfouriertrans(gw, tau, w_n)

    v = hf.ising_v(0.5, u_int, L=32)
    g0t = hf.interpol(g0t, 32)

    g0ttp = hf.ret_weiss(g0t)

    groot = hf.gnewclean(g0ttp, v, 1)
    flip = 5
    v[flip] *= -1

    g_flip = hf.gnewclean(g0ttp, v, 1)
    g_fast_flip = hf.gnew(np.copy(groot), v, flip, 1, np.eye(32))

    assert np.allclose(g_flip, g_fast_flip)
