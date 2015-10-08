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

from dmft.common import greenF, tau_wn_setup, fit_gf
from dmft.twosite import matsubara_Z

import numpy as np
import matplotlib.pylab as plt

U = np.linspace(0, 3.2, 41)
U = np.concatenate((U, np.linspace(3.1, 2.1, 16)-0.05))

parms = {'MU': 0, 't': 0.5, 'N_MATSUBARA': 256}
lop_g=[]
figz, axz = plt.subplots()
figf, axf = plt.subplots()
for beta in [16, 25, 50, 100, 200]:
    u_zet = []
    u_fl = []
    parms['BETA'] = beta
    tau, w_n = tau_wn_setup(parms)
    g_iwn0 = greenF(w_n, D=2*parms['t'])
    for u_int in U:
        mix = 1
        g_iwn, sigma = ipt_imag.dmft_loop(u_int, parms['t'], g_iwn0, w_n, tau, mix)
        lop_g.append(g_iwn)
        u_zet.append(matsubara_Z(sigma.imag, beta))
        u_fl.append(-fit_gf(w_n[:3], g_iwn.imag)(0.))

    axz.plot(U, u_zet, '+-', label='$\\beta={}$'.format(beta))
    axf.plot(U, u_fl, 'x:', label='$\\beta={}$'.format(beta))
plt.title('Hysteresis loop of the quasiparticle weigth')
plt.legend(loc=0)
axz.set_ylabel('Z')
axz.set_xlabel('U/D')
axf.set_ylabel('Dos(0)')
axf.set_xlabel('U/D')
