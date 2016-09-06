# -*- coding: utf-8 -*-
"""
Tests the greens functions library with complex Imaginary time
@author: oscar
"""

from __future__ import division, absolute_import, print_function
import numpy as np
import pytest
import scipy.linalg as la
import dmft.common_complex as cgf


@pytest.mark.parametrize("beta", [20, 15, 31.4])
def test_matsubara_axis(beta):
    wn = cgf.matsubara_freq(beta, 2 * beta)
    assert len(wn) % 2 == 0
    assert np.allclose(wn, -wn[::-1])


@pytest.mark.parametrize("beta", np.random.rand(48) * 100)
def test_imaginary_time_array(beta):
    tau, wn = cgf.tau_wn_setup(beta, 3 * beta)
    assert len(wn) % 2 == 0
    assert len(tau) == len(wn)


@pytest.mark.parametrize("beta", np.random.rand(48) * 100)
def test_greenf(beta):
    wn = cgf.matsubara_freq(beta, 2 * beta)
    semicirc = cgf.greenF(wn)
    assert np.allclose(semicirc.real, semicirc.real[::-1])
    assert np.allclose(semicirc.imag, -semicirc.imag[::-1])


@pytest.mark.parametrize("chempot", [0, 0.5, -0.8, 4.])
def test_fourier_trasforms(chempot, beta=50., n_matsubara=50):
    """Test the tail improved fourier transforms"""
    parms = {'BETA': beta, 'N_MATSUBARA': n_matsubara}
    tau, wn = cgf.tau_wn_setup(beta, n_matsubara)
    giw = cgf.greenF(wn, mu=chempot)

    for gwr in [giw, np.array([giw, giw])]:
        g_tau = cgf.gw_invfouriertrans(gwr, tau, wn, [1., -chempot, 0.25])
        g_iomega = cgf.gt_fouriertrans(g_tau, tau, wn, [1., -chempot, 0.25])
        assert np.allclose(gwr, g_iomega)


def generate_random_gf(beta, size):
    """Generates a gf in a bethe lattice with a random Hamiltonian"""
    w_n = cgf.matsubara_freq(beta=beta, pos_size=beta)
    tau = np.arange(0, beta, beta / len(w_n))

    H_loc = np.random.randn(size, size) + 1j * np.random.randn(size, size)
    H_loc = H_loc + H_loc.T.conj()  # Make Hermitian hamiltonian
    g_0_1 = [1j * wn * np.eye(size) - H_loc for wn in w_n]
    g_0 = [la.inv(g) for g in g_0_1]

    for i in range(40):
        g_0_1 = [1j * wn * np.eye(size) - H_loc - 0.25 *
                 g for wn, g in zip(w_n, g_0)]
        g_0 = [la.inv(g) for g in g_0_1]

    g_tail = [np.eye(size).reshape(size, size, 1),
              H_loc.reshape(size, size, 1),
              0.25 * np.eye(size).reshape(size, size, 1)]
    return np.rollaxis(np.array(g_0), 0, 3), g_tail, w_n, tau


@pytest.mark.parametrize("sites", [1, 2, 3])
def test_fourier_trasforms(sites):
    g0iw, tail, wn, tau = generate_random_gf(16, sites)

    g_tau = cgf.gw_invfouriertrans(g0iw, tau, wn, tail)
    g_iomega = cgf.gt_fouriertrans(g_tau, tau, wn, tail)
    assert np.allclose(g0iw, g_iomega)
