# -*- coding: utf-8 -*-
"""
Dimer Bethe lattice
===================

Non interacting dimer of a Bethe lattice
"""

import numpy as np
import matplotlib.pyplot as plt
import dmft.common as gf

w = 1e-3j+np.linspace(-4, 4, 2**10)

mu, t = 0, 0.5
t2 = t**2

plt.figure()

for tab in [0, 0.25, 0.5, 0.75, 1.1]:

    G1 = gf.greenF(-1j*w, mu=mu-tab, D=2*t)
    G2 = gf.greenF(-1j*w, mu=mu+tab, D=2*t)

    Gd = .5*(G1 + G2)
    Gc = .5*(G1 - G2)

    Dd = w + mu - t2 * Gd
    Dc = tab + t2 * Gc
    Gd =  1. / (Dd - Dc**2 / Dd)
    Gc = -1. / (Dc - Dd**2 / Dc)

    plt.plot(w.real, -Gd.imag/np.pi, label=r'$t_c={}$'.format(tab))
#    plt.plot(w.real, Gd.real, label=r'$\Re e Gd$')
#    plt.plot(w.real, Gc.real, label=r'$\Re e Gc$')
#    plt.plot(w.real, Gc.imag, label=r'$\Im m Gc$')

plt.legend(loc=0)
plt.xlabel(r'$\omega$')
plt.ylabel(r'$A(\omega)$')


w_n = gf.matsubara_freq(50., 512)
iw_n = 1j*w_n
plt.figure()
for tab in [0, 0.25, 0.5, 0.75, 1.1]:

    G1 = gf.greenF(w_n, mu=mu-tab, D=2*t)
    G2 = gf.greenF(w_n, mu=mu+tab, D=2*t)

    Gd = .5*(G1 + G2)
    Gc = .5*(G1 - G2)

    Dd = iw_n + mu - t2 * Gd
    Dc = tab + t2 * Gc
    Gd =  1. / (Dd - Dc**2 / Dd)
    Gc = -1. / (Dc - Dd**2 / Dc)

    plt.plot(w_n, Gd.imag, 'o-', label=r'$t_c={}$'.format(tab))

plt.legend(loc=0)
plt.xlim([0, 6.5])
plt.xlabel(r'$i\omega_n$')
plt.ylabel(r'$G(i\omega_n)$')
