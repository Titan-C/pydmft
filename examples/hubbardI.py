# -*- coding: utf-8 -*-
r"""
Hubbard I solver
"""
# Created Mon Sep 28 15:25:30 2015
# Author: Óscar Nájera

from __future__ import division, absolute_import, print_function

import dmft.common as gf
import matplotlib.pyplot as plt
import numpy as np

n = 1.
U = 5
mu =0# U/2

omega = np.linspace(-4, 8, 200) + 1j*7e-2
k = np.linspace(0, np.pi, 51)

sigma = n*U/2 + n/2*(1-n/2)*U**2/(omega - (1 - n/2)*U)
eps_k = -np.cos(k)
lat_gf = 1/(np.subtract.outer(omega + mu - sigma,  eps_k))

plt.pcolormesh(k, omega, lat_gf.imag/np.pi, cmap='hot')
plt.colorbar()
