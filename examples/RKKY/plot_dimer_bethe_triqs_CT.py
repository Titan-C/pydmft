# -*- coding: utf-8 -*-
"""
Dimer Bethe lattice
===================

Non interacting dimer of a Bethe lattice
"""
#from __future__ import division, print_function, absolute_import
import sys
sys.path.append('/home/oscar/libs/lib/python2.7/site-packages/')
from pytriqs.gf.local import GfImFreq, GfImTime, InverseFourier, \
    Fourier, iOmega_n, inverse
from pytriqs.gf.local import GfReFreq, Omega
from pytriqs.operators import n

from pytriqs.plot.mpl_interface import oplot
import dmft.common as gf
import numpy as np
import matplotlib.pyplot as plt
from pytriqs.archive import *
import pytriqs.utility.mpi as mpi

from pytriqs.applications.impurity_solvers.cthyb import Solver


def mix_gf_dimer(gmix, omega, mu, tab):
    gmix[0, 0] = omega + mu
    gmix[0, 1] = -tab
    gmix[1, 0] = -tab
    gmix[1, 1] = omega + mu
    return gmix


def init_gf(g_iw, omega, mu, tab, t):
    G1 = gf.greenF(omega, mu=mu-tab, D=2*t)
    G2 = gf.greenF(omega, mu=mu+tab, D=2*t)

    Gd = .5*(G1 + G2)
    Gc = .5*(G1 - G2)

    g_iw[0, 0].data[:, 0, 0] = Gd
    g_iw[0, 1].data[:, 0, 0] = Gc
    g_iw[1, 0] << g_iw[0, 1]
    g_iw[1, 1] << g_iw[0, 0]


t = 0.5
D = 2*t
t2 = t**2
tab = 0.0
beta = 25.
U = 0.
mu = U/2.

parms = {'n_cycles': 100000,
         'length_cycle': 200,
         'n_warmup_cycles': 1000}
# Matsubara interacting self-consistency

g_iw = GfImFreq(indices=[0, 1], beta=beta)
gmix = mix_gf_dimer(g_iw.copy(), iOmega_n, mu, tab)

S = Solver(beta=beta, gf_struct={'up': [0, 1], 'down': [0, 1]})

w_n = gf.matsubara_freq(beta, 1025)
init_gf(S.G_iw['up'], w_n, mu, tab, t)
S.G_iw['down'] << S.G_iw['up']

for i in xrange(2):
    # Bethe lattice bath
    for na in ['up', 'down']:
        S.G0_iw[na] << gmix - t2 * S.G_iw[na]
        S.G0_iw[na].invert()

    S.solve(h_loc=U * n('up', 0) * n('down', 0)
                 +U * n('up', 1) * n('down', 1), **parms)

    oplot(S.G_iw['up']['0','0'], RI='I', label=r'$iter {}$'.format(i))
#
real = GfReFreq(indices=[1], window=(-4.0, 4.0), n_points=400)
real.set_from_pade(S.G_iw['up']['1','1'],200,0.)
oplot(real)
        # Some intermediate saves
    if mpi.is_master_node():
      R = HDFArchive("dimer_bethe.h5")
      R["G_tau-%s"%i] = S.G_tau
      del R
plt.ylabel(r'$A(\omega)$')
plt.title(u'Spectral functions of dimer Bethe lattice at $\\beta/D=100$ and $U/D=1.5$.\n Analitical continuation Padé approximant')
