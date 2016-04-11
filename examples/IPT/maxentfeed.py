r"""
test maxIteration
=================

"""
from __future__ import division, absolute_import, print_function

from dmft.ipt_imag import dmft_loop
from dmft.common import greenF, tau_wn_setup, pade_coefficients, pade_rec, plot_band_dispersion


import numpy as np
import matplotlib.pylab as plt

U = 2.5
beta = 40.
tau, w_n = tau_wn_setup(dict(BETA=beta, N_MATSUBARA=3 * beta))
omega = np.linspace(-3, 3, 600)
g_iwn0 = greenF(w_n)
g_iwn, s_iwn = dmft_loop(U, 0.5, -1.j / w_n, w_n, tau, conv=1e-8)
fw = np.concatenate((-w_n[::-1], w_n))
sgiw = np.concatenate((g_iwn.conj()[::-1], g_iwn))
# plot(fw,sgiw.imag)
np.save('/home/oscar/dev/pymaxent/ipts',
        np.array((fw, sgiw, np.ones_like(fw) * 1e-5 / fw)).T)
