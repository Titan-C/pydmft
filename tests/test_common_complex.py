# -*- coding: utf-8 -*-
"""
Tests the greens functions library with complex Imaginary time
@author: oscar
"""

from __future__ import division, absolute_import, print_function
import numpy as np
import pytest
import dmft.common_complex as cgf
import slaveparticles.quantum.dos as dos
from scipy.integrate import simps


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
