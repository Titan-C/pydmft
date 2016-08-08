# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 12:44:23 2014

@author: oscar
"""

from __future__ import division, absolute_import, print_function
import numpy as np
import pytest
import dmft.common as gf
import slaveparticles.quantum.dos as dos
from scipy.integrate import simps


@pytest.mark.parametrize("chempot", [0, 0.5, -0.8, 4.])
def test_fourier_trasforms(chempot, beta=50., n_matsubara=128):
    """Test the tail improved fourier transforms"""
    parms = {'BETA': beta, 'N_MATSUBARA': n_matsubara}
    tau, w_n = gf.tau_wn_setup(parms)
    giw = gf.greenF(w_n, mu=chempot)

    for gwr in [giw, np.array([giw, giw])]:
        g_tau = gf.gw_invfouriertrans(gwr, tau, w_n, [1., -chempot, 0.25])
        g_iomega = gf.gt_fouriertrans(g_tau, tau, w_n, [1., -chempot, 0.25])
        assert np.allclose(gwr, g_iomega)


def test_fit_gf():
    """Test the interpolation of Green function in Bethe Lattice"""
    w_n = gf.matsubara_freq(100, 3)
    giw = gf.greenF(w_n).imag
    cont_g = gf.fit_gf(w_n, giw)
    assert abs(cont_g(0.) + 2.) < 1e-4


def test_pade():
    """Test pade Analytical Continuation for the semi-circular DOS"""
    w_n = gf.matsubara_freq(200)
    giw = gf.greenF(w_n)
    omega = np.linspace(-0.7, 0.7, 100)  # avoid semicircle edges
    gw_ref = gf.greenF(-1j * omega + 1e-5)
    pade_c = gf.pade_coefficients(giw, w_n)
    gw_cont = gf.pade_rec(pade_c, omega, w_n)
    assert np.allclose(gw_ref, gw_cont, 1e-3)


@pytest.mark.parametrize("halfbandwidth", [0.5, 1., 2.])
def test_hilbert_trans_func(halfbandwidth):
    """Test Hilbert transforms of semi-circle"""
    # Match in the 2 forms
    w_n = gf.matsubara_freq(200)
    giw = gf.greenF(w_n, sigma=-1j / w_n, mu=-0.2, D=halfbandwidth)
    ss = gf.semi_circle_hiltrans(1j * w_n - .2 + 1j / w_n, D=halfbandwidth)
    assert np.allclose(ss, giw)

    # corresponds to semi-circle
    w = np.linspace(-3, 3, 2**9)
    ss = gf.semi_circle_hiltrans(w + 1e-5j, D=halfbandwidth)
    assert np.allclose(dos.bethe_lattice(
        w,  halfbandwidth / 2), - ss.imag / np.pi, atol=1e-4)


@pytest.mark.parametrize("halfbandwidth", [0.5, 1., 2.])
def test_hilbert_trans_integral(halfbandwidth):
    """Test hilbert transform of semi-circle to direct integral"""

    w = np.linspace(-3, 3, 2**9)
    w_n = gf.matsubara_freq(20)
    giw = gf.greenF(w_n, D=halfbandwidth)
    rho_w = dos.bethe_lattice(w,  halfbandwidth / 2)
    Apiw = np.array([simps(rho_w / (1j * iw - w), w) for iw in w_n])

    assert np.allclose(Apiw, giw, 2e-4)


@pytest.mark.parametrize("moments", [(1, 0, 0.5), (.5, -.2, -7)])
def test_tail_fit(moments):
    wn = gf.matsubara_freq(50)
    moment = np.array(moments)
    tail = gf.tail(wn, moment * np.array([1j, 1, 1j]),
                   np.arange(len(moment)) + 1)
    tailn = np.sum(np.random.randn(len(wn), 2) *
                   0.00005 * np.array((1, 1j)), 1)
    tail = tail + tailn

    fit_moments = gf.lin_tail_fit(wn, tail, -20, 30)[1]
    print(fit_moments - moment)
    assert np.allclose(moment, fit_moments, atol=3e-3)


def test_tail_fit_semicirc():
    wn = gf.matsubara_freq(50)
    gci = gf.greenF(wn)
    tailn = np.sum(np.random.randn(len(wn), 2) *
                   0.00006 * np.array((1, 1j)), 1) / (wn**2 + 3)
    tail = gci + tailn

    fit_moments = gf.lin_tail_fit(wn, tail, -45, 45)[1]
    moment = np.array((-1, 0, 0.25))
    print(fit_moments - moment)
    assert np.allclose(moment, fit_moments, atol=7e-3)
