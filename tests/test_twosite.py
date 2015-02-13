# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 12:44:23 2014

@author: oscar
"""

from __future__ import division, absolute_import, print_function
from dmft.common import greenF, gw_invfouriertrans, gt_fouriertrans
from dmft.twosite import dmft_loop
from dmft.twosite_dop import dmft_loop_dop
import numpy as np


def test_sigma():
    pass


def test_fourier_trasforms(beta=50., Ntau=1000, Nmatsubara=100):
    """Test the tail improved fourier transforms"""
    iw = 1j*np.pi*(1+2*np.arange(Nmatsubara)) / beta
    gwr = greenF(iw)[1::2]
    tau = np.linspace(0, beta, Ntau+1)

    gt = gw_invfouriertrans(gwr, tau, iw, beta)
    gw = gt_fouriertrans(gt, tau, iw, beta)
    assert (np.abs(gwr - gw) < 5e-13).all()


def test_mit_real():
    """Test the metal to insulator transition at very low temperature
    calculated in the real axis formalism"""
    z_ref = np.array([1., 0.88889, 0.75, 0.55556, 0.30556, 0.06556, 0.])
    zet = dmft_loop(u_int=[0, 1, 1.5, 2, 2.5, 2.9, 3.05], axis='real',
                    beta=1e5, hop=0.5)[:, 1]
    print(np.abs(zet-z_ref))
    assert (np.abs(zet-z_ref) < 3e-3).all()


def test_matsubara():
    z_ref = np.array([1., 0.88889, 0.75, 0.55556, 0.30556, 0.06556, 0.])
    zet = dmft_loop(u_int=[0, 1, 1.5, 2, 2.5, 2.9, 3.05], axis='matsubara',
                    beta=1e5, hop=0.5)[:, 1]
    print(np.abs(zet-z_ref))
    assert (np.abs(zet-z_ref) < 5e-4).all()


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
        assert (np.abs(ref-test) < 5e-5).all()
