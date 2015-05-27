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
from pytriqs.operators import n

import dmft.common as gf
import numpy as np
from pytriqs.archive import *
import pytriqs.utility.mpi as mpi
from plot_dimer_bethe_triqs import mix_gf_dimer, init_gf

from pytriqs.applications.impurity_solvers.cthyb import Solver

t = 0.5
D = 2*t
t2 = t**2
tab = 0.0
beta = 75.
U = 0.5
mu = U/2.

parms = {'n_cycles': 100000,
         'length_cycle': 200,
         'n_warmup_cycles': 1000}
# Matsubara interacting self-consistency

w_n = gf.matsubara_freq(beta, 512)
g_iw = GfImFreq(indices=['A', 'B'], beta=50, n_points=512)
gmix = mix_gf_dimer(g_iw.copy(), iOmega_n, mu, tab)

S = Solver(beta=beta, gf_struct={'up': [0, 1], 'down': [0, 1]})

w_n = gf.matsubara_freq(beta, 1025)
if mpi.is_master_node():

    init_gf(S.G_iw['up'], w_n, mu,.0, t)
    S.G_iw['down'] << S.G_iw['up']

converged = False
loops = 0
#
#if mpi.is_master_node():
#    newg = 0.5 * (S.G_iw['up'] + S.G_iw['down'])
#print('uea')
#while not converged:
#    # Bethe lattice PM bath
#    if mpi.is_master_node():
#        oldg = newg.data.copy()
#        for na in ['up', 'down']:
#            S.G0_iw[na] << gmix - t2 * newg
#        S.G0_iw.invert()
#    S.G0_iw << mpi.bcast(S.G0_iw)
##
##    S.solve(h_loc=U * n('up', 0) * n('down', 0)
##                 +U * n('up', 1) * n('down', 1), **parms)
#
#    mpi.barrier()
#    if mpi.is_master_node():
#
#        newg = 0.5 * (S.G_iw['up'] + S.G_iw['down'])
#        converged = np.allclose(newg.data, oldg)
#        loops += 1
#    converged = mpi.bcast(converged)
#
#    mpi.barrier()
#
#print(loops)
#
#
##    if mpi.is_master_node():
##        Results = HDFArchive("dimert0U.5b75.h5")
##        Results["G_iw%s"%(loops)] = S.G_iw
##        del Results
##    oplot(S.G_iw['up']['0','0'], RI='I', label=r'$iter {}$'.format(i))
###
##real = GfReFreq(indices=[1], window=(-4.0, 4.0), n_points=400)
#real.set_from_pade(S.G_iw['up']['1','1'],200,0.)
#oplot(real)
#        # Some intermediate saves
#    if mpi.is_master_node():
#      R = HDFArchive("dimer_bethe.h5")
#      R["G_tau-%s"%i] = S.G_tau
#      del R
#plt.ylabel(r'$A(\omega)$')
#plt.title(u'Spectral functions of dimer Bethe lattice at $\\beta/D=100$ and $U/D=1.5$.\n Analitical continuation Padé approximant')
