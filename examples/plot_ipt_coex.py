# -*- coding: utf-8 -*-
r"""
===================
IPT Histereris loop
===================

Within the iterative perturbative theory (IPT) the aim is to find the
coexistance regions.

"""
from __future__ import division, absolute_import, print_function

from dmft import ipt_imag

from dmft.common import greenF, tau_wn_setup
from dmft.twosite import matsubara_Z

import numpy as np
import matplotlib.pylab as plt

U = np.linspace(0, 2.7, 36)
U = np.concatenate((U, U[-2:11:-1]))

parms = {'MU': 0, 't': 0.5, 'N_TAU': 2**10, 'N_MATSUBARA': 2**7}
lop_g=[]
for beta in [16, 25, 50]:
    u_zet = []
    parms['BETA'] = beta
    tau, w_n = tau_wn_setup(parms)
    g_iwn0 = greenF(w_n, D=2*parms['t'])
    for u_int in U:
        mix = 0.4 if u_int > 1.5 else 1
        g_iwn_log, sigma = ipt_imag.dmft_loop(100, u_int, parms['t'], g_iwn0, w_n, tau, mix)
        g_iwn0 = g_iwn_log[-1]
        lop_g.append(g_iwn_log)
        u_zet.append(matsubara_Z(sigma.imag, beta))

    plt.plot(U, u_zet, label='$\\beta={}$'.format(beta))
plt.title('Hysteresis loop of the quasiparticle weigth')
plt.legend()
plt.ylabel('Z')
plt.xlabel('U/D')

