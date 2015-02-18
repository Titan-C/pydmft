# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 12:44:23 2014

@author: oscar
"""

from __future__ import division, absolute_import, print_function
import numpy as np
import dmft.pmhf as hf
import dmft.common as gf


def test_u(beta=12., d_tau=0.5, n_tau=1000):
    iwn = gf.matsubara_freq(10, 300)
    giw = gf.greenF(iwn)[1::2]
    tau = np.linspace(0, beta, n_tau+1)
    g0t = gf.gw_invfouriertrans(giw, tau, iwn, beta)

    v = hf.ising_v(d_tau, 2, L=beta/d_tau)
    lfak = v.size
    g0t = hf.extract_g0t(g0t, lfak)

    gind = lfak + np.arange(lfak).reshape(-1, 1)-np.arange(lfak)
    g0ttp = g0t[gind]

    groot = hf.gnewclean(g0ttp, v, 1)
    flip = 5
    v[flip] *= -1

    g_flip = hf.gnewclean(g0ttp, v, 1)
    g_fast_flip = hf.gnew(groot, v, flip, 1, np.eye(lfak))

    assert np.allclose(g_flip, g_fast_flip)
