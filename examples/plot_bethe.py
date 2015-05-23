# -*- coding: utf-8 -*-
"""
Bethe lattice semi-eliptical DOS
================================

The Bethe lattice has a simpl form of the self-consistency. This plots how
this equation includes iteratively the poles is the Green function so one
recovers such density of states.
"""

import numpy as np
import matplotlib.pyplot as plt

w = 1e-3j+np.linspace(-4, 4, 800)
gr = np.zeros_like(w)
t = 0.5

for i in xrange(1, int(1e4+1)):
    gr = 1/(w - t**2 * gr)
    if i in [1, 4, 10000]:
        plt.plot(w.real, gr.real, label='iteration '+str(i))

plt.plot(w.real, -gr.imag/np.pi, label=r'$A(\omega)$')
plt.legend(loc=0)
plt.ylim([-3, 3])
plt.xlabel(r'$\omega$')
plt.ylabel(r'$\Re e G(\omega)$')
