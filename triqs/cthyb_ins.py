# -*- coding: utf-8 -*-
"""
@author: Óscar Nájera
Created on Mon Nov 10 11:18:35 2014
"""
#from __future__ import division, absolute_import, print_functionfrom pytriqs.gf.local import *
from pytriqs.archive import HDFArchive
from pytriqs.gf.local import *
from pytriqs.operators import *
import numpy as np
import pytriqs.utility.mpi as mpi

# Set up a few parameters
U = 3.2
half_bandwidth = 1.0
chemical_potential = U/2.0
beta = 100.
n_loops = 5

# Construct the CTQMC solver
from pytriqs.applications.impurity_solvers.cthyb import Solver
S = Solver(beta=beta, gf_struct={ 'up':[0], 'down':[0] },
           n_iw=int(2*beta), n_tau=int(4*beta))

# Set the solver parameters
params = {'n_cycles': int(3e7),
          'length_cycle': 200,
          'n_warmup_cycles': int(5e4),
          'measure_pert_order': True,
        }

# Initalize the Green's function to a semi-circular density of states

g_iw = GfImFreq(indices = [0], beta = beta, n_points=1025)
#g_iw << SemiCircular(half_bandwidth)
g_iw.data[:,0,0] = np.load('Giw_out.npy')[-1] #np.load('fgiws500.npy')
fixed=TailGf(1,1,5,-1)
fixed[1]=np.array([[1]])
fixed[3]=np.array([[3.2**2/4]])
g_iw.fit_tail(fixed,5, 98, len(g_iw.mesh))


fixedg0=TailGf(1,1,4,-1)
fixedg0[1]=np.array([[1]])
fixedg0[2]=np.array([[-chemical_potential]])

print 'got here'
# Now do the DMFT loop
for it in range(n_loops):

    # Compute S.G0_iw with the self-consistency condition while imposing paramagnetism
#    g_iw.set_from_legendre( 0.5 * ( S.G_l['up'] + S.G_l['down'] ))
    g_iw << 0.5*(S.G_iw['up']+S.G_iw['down']))
    g_iw.fit_tail(fixed, 5, 98, len(g_iw.mesh))
    g_iw.data[:]=1j*g_iw.data.imag

    for name, g0 in S.G0_iw:
        g0 << inverse( iOmega_n + chemical_potential - (half_bandwidth/2.0)**2  * g_iw )
#        g0.fit_tail(fixedg0, 6, 100, 1025)
    # Run the solver
    S.solve(h_int=U * n('up',0) * n('down',0), **params)

    # Some intermediate saves
    if mpi.is_master_node():
        with HDFArchive("deep_in_ins3.h5") as R:
          R["G_tau-%s"%it] = S.G_tau
          R["G_iw-%s"%it] = S.G_iw
          R["G0_iw-%s"%it] = S.G0_iw
          R["G_l-%s"%it] = S.G_l
