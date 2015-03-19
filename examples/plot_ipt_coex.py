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

from dmft.common import greenF, matsubara_freq
from dmft.twosite import matsubara_Z

import numpy as np
import matplotlib.pylab as plt

beta = [100] # np.logspace(0, 2.5, 10)
U = np.linspace(0, 3.2, 36)
U = np.concatenate((U, U[::-1]))
t = 0.5
for b in beta:
    tau = np.linspace(0, b, 1001)
    u_zet = []
    iwn = matsubara_freq(b, int(10*b/np.pi))
    g_iwn0 = greenF(iwn, D=2*t)
    for u_int in U:
        g_iwn_log, sigma = ipt_imag.dmft_loop(25, u_int, t, g_iwn0, iwn, tau)
        g_iwn0 = g_iwn_log[-1]
        u_zet.append(matsubara_Z(sigma.imag, b))

plt.plot(U, u_zet)

