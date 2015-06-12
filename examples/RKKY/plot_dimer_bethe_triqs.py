# -*- coding: utf-8 -*-
"""
Dimer Bethe lattice
===================

Non interacting dimer of a Bethe lattice
"""
from __future__ import division, print_function, absolute_import
from pytriqs.gf.local import GfImFreq, iOmega_n, TailGf
from pytriqs.gf.local import GfReFreq, Omega
from pytriqs.plot.mpl_interface import oplot
import numpy as np
import matplotlib.pyplot as plt
from dmft.RKKY_dimer_IPT import mix_gf_dimer, init_gf_met
import dmft.common as gf


def init_gf_met_tx(g_iw, omega, mu, tn, t):
    G1 = gf.greenF(omega, mu=mu, D=2*(t+tn))
    G2 = gf.greenF(omega, mu=mu, D=2*abs(t-tn))

    Gd = .5*(G1 + G2)
    Gc = .5*(G1 - G2)

    g_iw['A', 'A'].data[:, 0, 0] = Gd
    g_iw['A', 'B'].data[:, 0, 0] = Gc
    g_iw['B', 'A'] << g_iw['A', 'B']
    g_iw['B', 'B'] << g_iw['A', 'A']

    if isinstance(g_iw, GfImFreq):
        fixed_co = TailGf(2, 2, 4, -1)
        fixed_co[1] = np.array([[1, 0], [0, 1]])
#        fixed_co[2] = tab*np.array([[0, 1], [1, 0]])
        g_iw.fit_tail(fixed_co, 6, int(0.6*len(omega)), int(0.8*len(omega)))

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

        init_gf_met(g_re, -1j*w, mu, tab, t)
        g_re << gmix_re - t2 * g_re
        g_re.invert()

        oplot(g_re['A', 'A'], RI='S', label=r'$t_{{ab}}={}$'.format(tab), num=1)

    # Matsubara frequency Green's function
    w_n = gf.matsubara_freq(beta, 512)
    for tab in [0, 0.25, 0.5, 0.75, 1.1]:
        g_iw = GfImFreq(indices=['A', 'B'], beta=beta, n_points=len(w_n))
        gmix = mix_gf_dimer(g_iw.copy(), iOmega_n, mu, tab)

        init_gf_met(g_iw, w_n, mu, tab, t)
        g_iw << gmix - t2 * g_iw
        g_iw.invert()
        oplot(g_iw['A', 'A'], '+-', RI='I', label=r'$t_{{ab}}={}$'.format(tab), num=2)
    plt.xlim([0, 6.5])
    plt.ylabel(r'$A(\omega)$')
    plt.title(u'Spectral functions of dimer Bethe lattice at ' +
              u'$\\beta/D={}$ and $U/D=0$.'.format(beta) +
              u'\nAnalitical continuation Padé approximant')
    plt.legend(loc=0)

    for tab in [0.1, 0.25, 0.5, 0.75, 1.1]:
        g_iw = GfImFreq(indices=['A', 'B'], beta=beta, n_points=len(w_n))
        gmix = mix_gf_dimer(g_iw.copy(), iOmega_n, 0., 0.)

        a = True
        t_hop = np.matrix([[t, 0.52], [0.52, t]])
        init_gf_met_tx(g_iw, w_n, 0., 0.01, t)
        g_re.set_from_pade(g_iw,200)
        oplot(g_re['A','A'],RI='S')
#        import pdb; pdb.set_trace()
        while False:
            g_iw<< gmix - t_hop * g_iw * t_hop
            g_iw.invert()
            plt.subplot(221)
            plt.plot(w_n, g_iw['A', 'A'].data[:,0,0].real, '+-')
            plt.plot(w_n, g_iw['A', 'A'].data[:,0,0].imag, 'o-')
            plt.xlim(xmax=3)
            plt.subplot(222)
            plt.plot(w_n, g_iw['A', 'B'].data[:,0,0].real, '+-')
            plt.plot(w_n, g_iw['A', 'B'].data[:,0,0].imag, 'o-')
            plt.xlim(xmax=1.5)
            plt.subplot(223)
            plt.plot(w_n, g_iw['B', 'A'].data[:,0,0].real, '+-')
            plt.plot(w_n, g_iw['B', 'A'].data[:,0,0].imag, 'o-')
            plt.xlim(xmax=1.5)
            plt.subplot(224)
            plt.plot(w_n, g_iw['B', 'B'].data[:,0,0].real, '+-')
            plt.plot(w_n, g_iw['B', 'B'].data[:,0,0].imag, 'o-')
            plt.xlim(xmax=3)

            plt.pause(2)