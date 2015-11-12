# -*- coding: utf-8 -*-
"""
Dimer Bethe lattice
===================

Non interacting dimer of a Bethe lattice
"""

import numpy as np
import matplotlib.pyplot as plt
import dmft.common as gf
import dmft.RKKY_dimer as rt


def self_consistency(omega, Gd, Gc, mu, tab, t2):
    """Sets the dimer Bethe lattice self consistent condition for the diagonal
    and out of diagonal block
    """

    Dd = omega + mu - t2 * Gd
    Dc = -tab + t2 * Gc

    return Dd, Dc

######################################################################
# Real frequency spectral function
# ================================

w = 1e-3j+np.linspace(-4, 4, 2**10)
mu, t = 0, 0.5
t2 = t**2

plt.figure()
for tab in [0, 0.25, 0.5, 0.75, 1.1]:

    Gd, Gc = rt.gf_met(-1j*w, mu, tab, t, 0.)
    Dd, Dc = self_consistency(w, Gd, Gc, mu, tab, t2)
    Gd, Gc = rt.mat_inv(Dd, Dc)

    plt.plot(w.real, -Gd.imag/np.pi, label=r'$t_c={}$'.format(tab))
#    plt.plot(w.real, Gd.real, label=r'$\Re e Gd$')
#    plt.plot(w.real, Gc.real, label=r'$\Re e Gc$')
#    plt.plot(w.real, Gc.imag, label=r'$\Im m Gc$')

plt.legend(loc=0)
plt.xlabel(r'$\omega$')
plt.ylabel(r'$A(\omega)$')


######################################################################
# Matsubara frequency Green's function
# ====================================

w_n = gf.matsubara_freq(50., 512)
iw_n = 1j*w_n
plt.figure()
for tab in [0, 0.25, 0.5, 0.75, 1.1]:

    Gd, Gc = rt.gf_met(w_n, mu, tab, t, 0.)
    Dd, Dc = self_consistency(iw_n, Gd, Gc, mu, tab, t2)
    Gd, Gc = rt.mat_inv(Dd, Dc)

    plt.plot(w_n, Gd.imag, 'o-', label=r'$t_c={}$'.format(tab))

plt.legend(loc=0)
plt.xlim([0, 6.5])
plt.xlabel(r'$i\omega_n$')
plt.ylabel(r'$G(i\omega_n)$')
