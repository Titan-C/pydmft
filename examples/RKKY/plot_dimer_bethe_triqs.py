# -*- coding: utf-8 -*-
"""
Dimer Bethe lattice
===================

Non interacting dimer of a Bethe lattice
"""
from __future__ import division, print_function, absolute_import
import sys
sys.path.append('/home/oscar/libs/lib/python2.7/site-packages')
from pytriqs.gf.local import GfImFreq, iOmega_n, inverse
from pytriqs.gf.local import GfReFreq, Omega
from pytriqs.plot.mpl_interface import oplot
import dmft.common as gf
import numpy as np
import matplotlib.pyplot as plt


def mix_gf_dimer(gmix, omega, mu, tab):
    gmix['A', 'A'] = omega + mu
    gmix['A', 'B'] = -tab
    gmix['B', 'A'] = -tab
    gmix['B', 'B'] = omega + mu
    return gmix


def init_gf(g_iw, omega, mu, tab, t):
    G1 = gf.greenF(omega, mu=mu-tab, D=2*t)
    G2 = gf.greenF(omega, mu=mu+tab, D=2*t)

    Gd = .5*(G1 + G2)
    Gc = .5*(G1 - G2)

    g_iw['A', 'A'].data[:, 0, 0] = Gd
    g_iw['A', 'B'].data[:, 0, 0] = Gc
    g_iw['B', 'A'] << g_iw['A', 'B']
    g_iw['B', 'B'] << g_iw['A', 'A']

if __name__ == "__main__":
    mu, t = 0.0, 0.5
    t2 = t**2
    tab = 0.3
    beta = 300.

    # Real frequency spectral function
    w = 1e-3j+np.linspace(-3, 3, 2**9)

    for tab in [0, 0.25, 0.5, 0.75, 1.1]:
        g_re = GfReFreq(indices=['A', 'B'], window=(-3, 3), n_points=len(w))
        gmix_re = mix_gf_dimer(g_re.copy(), Omega + 1e-3j, mu, tab)

        init_gf(g_re, -1j*w, mu, tab, t)
        g_re << gmix_re - t2 * g_re
        g_re.invert()

        oplot(g_re['A', 'A'], RI='S', label=r'$t_{{ab}}={}$'.format(tab), num=1)

    # Matsubara frequency Green's function
    w_n = gf.matsubara_freq(beta, 512)
    for tab in [0, 0.25, 0.5, 0.75, 1.1]:
        g_iw = GfImFreq(indices=['A', 'B'], beta=beta, n_points=len(w_n))
        gmix = mix_gf_dimer(g_iw.copy(), iOmega_n, mu, tab)

        init_gf(g_iw, w_n, mu, tab, t)
        g_iw << gmix - t2 * g_iw
        g_iw.invert()
        oplot(g_iw['A', 'A'], RI='I', label=r'$t_{{ab}}={}$'.format(tab), num=2)
    plt.xlim([0, 6.5])
    plt.ylabel(r'$A(\omega)$')
    plt.title(u'Spectral functions of dimer Bethe lattice at ' +
              u'$\\beta/D={}$ and $U/D=0$.'.format(beta) +
              u'\nAnalitical continuation Padé approximant')
    plt.legend(loc=0)
