# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 12:44:23 2014

@author: oscar
"""

from __future__ import division, absolute_import, print_function
from itertools import product
import numpy as np
import dmft.hirschfye as hf
import pytest
import dmft.hffast as hffast


@pytest.mark.parametrize("chempot, u_int, updater",
                         product([0, 0.3], [2, 2.3], [hf.gnew, hffast.gnew]))
def test_hf_fast_updatecond(chempot, u_int, updater, beta=16., n_matsubara=32):
    parms = {'BETA': beta, 'N_MATSUBARA': n_matsubara,
             't': 0.5, 'BANDS': 1,
             'MU': chempot, 'U': u_int, 'dtau_mc': 0.5}
    _, _, g0t, _, v, _ = hf.setup_PM_sim(parms)
    v = np.squeeze(v)
    g0ttp = hf.retarded_weiss(g0t)
    kroneker = np.eye(v.size)

    groot = hf.gnewclean(g0ttp, v, kroneker)
    g_fast_flip = np.copy(groot)

    flip = 5
    v[flip] *= -1

    g_flip = hf.gnewclean(g0ttp, v, kroneker)
    updater(g_fast_flip, 2*v[flip], flip)

    assert np.allclose(g_flip, g_fast_flip)


@pytest.mark.parametrize("chempot, u_int, updater",
                         product([0, 0.3], [2, 2.3], [hf.g2flip, hffast.g2flip]))
def test_hf_fast_2flip(chempot, u_int, updater, beta=16., n_matsubara=64):
    parms = {'BETA': beta, 'N_MATSUBARA': n_matsubara,
             't': 0.5, 'BANDS': 1,
             'MU': chempot, 'U': u_int, 'dtau_mc': 0.5}
    _, _, g0t, _, v, _ = hf.setup_PM_sim(parms)
    v = np.abs(np.squeeze(v))
    g0ttp = hf.retarded_weiss(g0t)
    kroneker = np.eye(v.size)

    groot = hf.gnewclean(g0ttp, v, kroneker)
    g_fast_flip = np.copy(groot)
    flip = np.array([6, 10], dtype=np.intc)
    v[flip] *= -1
    updater(g_fast_flip, 2*v[flip], 6, 10)

    g_flip = hf.gnewclean(g0ttp, v, kroneker)

    assert np.allclose(g_flip, g_fast_flip)


@pytest.mark.parametrize("u_int", [1, 2, 2.5])
@pytest.mark.xfail(raises=AssertionError, reason='Atom is not well described')
def test_solver_atom(u_int):
    parms = {'BETA': 16., 'U': u_int, 'N_MATSUBARA': 16,
             'sweeps': 2000, 'therm': 1000, 'N_meas': 3, 'SEED': 4213,
             'save_logs': False, 'updater': 'discrete',
             'global_flip': True, 'SITES': 1, 'BANDS': 1}
    parms['dtau_mc'] = parms['BETA']/parms['N_MATSUBARA']/2
    v = hf.ising_v(parms['dtau_mc'], parms['U'], L=2*parms['N_MATSUBARA'])
    tau = np.linspace(0, parms['BETA'], 2*parms['N_MATSUBARA'])
    intm = hf.interaction_matrix(1)  # one orbital
    g0t = -.5 * np.ones(len(tau))
    gtu, gtd = hf.imp_solver([g0t, g0t], v, intm, parms)
    g = np.squeeze(0.5 * (gtu+gtd))
    result = np.polyfit(tau[:10], np.log(g[:10]), 1)
    assert np.allclose(result, [-u_int/2., np.log(.5)], atol=0.02)

SINGLE_BAND_GF_REF = \
 [(0, 2, np.array([-0.5  , -0.335, -0.246, -0.196, -0.164, -0.144, -0.129,
       -0.118, -0.11 , -0.104, -0.099, -0.095, -0.092, -0.09 , -0.089, -0.087,
       -0.087, -0.087, -0.089, -0.09 , -0.092, -0.095, -0.099, -0.104, -0.11 ,
       -0.118, -0.129, -0.144, -0.164, -0.196, -0.246, -0.335])),
  (0.5, 2.3, np.array([-0.451, -0.316, -0.237, -0.187, -0.154, -0.132, -0.117,
       -0.106, -0.098, -0.092, -0.088, -0.085, -0.082, -0.08 , -0.08 , -0.079,
       -0.079, -0.079, -0.08 , -0.081, -0.083, -0.085, -0.088, -0.092, -0.098,
       -0.105, -0.114, -0.127, -0.144, -0.172, -0.222, -0.322]))]
@pytest.mark.parametrize("chempot, u_int, gend", SINGLE_BAND_GF_REF)
def test_solver(chempot, u_int, gend):
    parms = {'BETA': 16., 'N_MATSUBARA': 16,
             't': 0.5, 'SITES': 1, 'BANDS': 1,
             'MU': chempot, 'U': u_int,
             'sweeps': 5000, 'therm': 1000, 'N_meas': 3, 'SEED': 4213,
             'save_logs': False, 'updater': 'discrete'}
    tau, w_n, g0t, Giw, v, intm = hf.setup_PM_sim(parms)
    G0iw = 1/(1j*w_n + parms['MU'] - .25*Giw)
    g0t = hf.gw_invfouriertrans(G0iw, tau, w_n, [1., -parms['MU'], 0.])
    gtu, gtd = hf.imp_solver([g0t, g0t], v, intm, parms)
    g = np.squeeze(-0.5 * (gtu+gtd))
    assert np.allclose(gend, g, atol=6e-3)


@pytest.mark.parametrize("chempot, u_int, gend", SINGLE_BAND_GF_REF)
def test_solver_dimer(chempot, u_int, gend):
    parms = {'BETA': 16., 'N_MATSUBARA': 16,
             't': 0.5, 'SITES': 2, 'BANDS': 1,
             'MU': chempot, 'U': u_int, 'dtau_mc': 0.5, 'n_tau_mc':    32,
             'sweeps': 5000, 'therm': 1000, 'N_meas': 3, 'SEED': 4213,
             'save_logs': False, 'updater': 'discrete'}
    tau, w_n, g0t, Giw, v, intm = hf.setup_PM_sim(parms)
    G0iw = 1/(1j*w_n + parms['MU'] - .25*Giw)
    G0t = hf.gw_invfouriertrans(G0iw, tau, w_n, [1., -parms['MU'], 0.])
    gb0t = np.array([[G0t, np.zeros_like(G0t)], [np.zeros_like(G0t), G0t]])
    gtu, gtd = hf.imp_solver([gb0t]*2, v, intm, parms)
    g = np.squeeze(-0.5 * (gtu+gtd))
    assert np.allclose(gend, g[0, 0], atol=6e-3)
    assert np.allclose(np.zeros_like(g0t), g[0, 1], atol=6e-3)
    assert np.allclose(np.zeros_like(g0t), g[1, 0], atol=6e-3)
    assert np.allclose(gend, g[1, 1], atol=6e-3)
