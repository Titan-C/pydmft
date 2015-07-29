# -*- coding: utf-8 -*-
"""
@author: Óscar Nájera
Created on Mon Nov 10 11:18:35 2014
"""
#from __future__ import division, absolute_import, print_functionfrom pytriqs.gf.local import *
from pytriqs.gf.local import *
from pytriqs.operators import *
from pytriqs.archive import HDFArchive
import pytriqs.utility.mpi as mpi
import numpy as np

# Set up a few parameters
U = 3.2
half_bandwidth = 1.0
chemical_potential = U/2.0
beta = 100.
n_loops = 10

# Construct the CTQMC solver
from pytriqs.applications.impurity_solvers.cthyb import Solver
S = Solver(beta=beta, gf_struct={ 'up':[0], 'down':[0] },
           n_iw=1025, n_tau=10001, n_l=80)

# Set the solver parameters
params = {'n_cycles': int(1e6),
          'length_cycle': 200,
          'n_warmup_cycles': int(5e4),
          'measure_g_l': True,
          'measure_pert_order': True,
        }

# Initalize the Green's function to a semi-circular density of states

g_iw = GfImFreq(indices = [0], beta = beta, n_points=1025)
g_iw << SemiCircular(half_bandwidth)
#g_iw.data[:,0,0] = np.load('fgiws500.npy')

#R = HDFArchive("compareHF.h5")
#for name, g0block in S.G_tau:
#    g0block << R['gtau-start1']
#del R

# Initalize the Green's function to a semi-circular density of states
for name, g0block in S.G_l:
    g0block.set_from_imfreq(g_iw)


print 'got here'
# Now do the DMFT loop
for it in range(n_loops):

    # Compute S.G0_iw with the self-consistency condition while imposing paramagnetism
    g_iw.set_from_legendre( 0.5 * ( S.G_l['up'] + S.G_l['down'] ))
    g_iw.fit_tail(g_iw.tail,3,350,1025)
    for name, g0 in S.G0_iw:
        g0 << inverse( iOmega_n + chemical_potential - (half_bandwidth/2.0)**2  * g_iw )

    # Run the solver
    S.solve(h_int=U * n('up',0) * n('down',0), **params)

    # Some intermediate saves
    if mpi.is_master_node():
        with HDFArchive("legendre_insu.h") as R:
          R["G_tau-%s"%it] = S.G_tau
          R["G_iw-%s"%it] = S.G_iw
          R["G0_iw-%s"%it] = S.G0_iw
          R["G_l-%s"%it] = S.G_l


