# -*- coding: utf-8 -*-
"""
@author: Óscar Nájera
Created on Mon Nov 10 11:18:35 2014
"""
#from __future__ import division, absolute_import, print_functionfrom pytriqs.gf.local import *
from pytriqs.gf.local import *
from pytriqs.operators import *
from pytriqs.archive import *
import pytriqs.utility.mpi as mpi
import numpy as np

# Set up a few parameters
U = 3.2
half_bandwidth = 1.0
chemical_potential = U/2.0
beta = 80
n_loops = 2

# Construct the CTQMC solver
from pytriqs.applications.impurity_solvers.cthyb import Solver
S = Solver(beta=beta, gf_struct={ 'up':[0], 'down':[0] })

# Set the solver parameters
params = {}
params['n_cycles'] = 1000000                # Number of QMC cycles
params['length_cycle'] = 600                # Length of one cycle
params['n_warmup_cycles'] = 100000           # Warmup cycles

# Initalize the Green's function to a semi-circular density of states

g_iw = GfImFreq(indices = [0], beta = beta)
#g_iw << SemiCircular(half_bandwidth)
g_iw.data[:,0,0] = np.load('fgiws500.npy')

#R = HDFArchive("compareHF.h5")
#for name, g0block in S.G_tau:
#    g0block << R['gtau-start1']
#del R

# Initalize the Green's function to a semi-circular density of states
for name, g0block in S.G_tau:
    g0block << InverseFourier(g_iw)


print 'got here'
# Now do the DMFT loop
for IterationNumber in range(n_loops):

    # Compute S.G0_iw with the self-consistency condition while imposing paramagnetism
    g_iw << 0.5 * Fourier( S.G_tau['up'] + S.G_tau['down'] )
    g_iw.fit_tail(g_iw.tail,3,350,1025)
    for name, g0 in S.G0_iw:
        g0 << inverse( iOmega_n + chemical_potential - (half_bandwidth/2.0)**2  * g_iw )

    # Run the solver
    S.solve(h_int=U * n('up',0) * n('down',0), **params)

    # Some intermediate saves
    if mpi.is_master_node():
      R = HDFArchive("compareHF.h5")
      R["G_tau-%s"%IterationNumber] = S.G_tau
      del R