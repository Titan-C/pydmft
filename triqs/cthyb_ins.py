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
half_bandwidth = 1.0
beta = 32
n_loops = 10

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

g_iw = S.G_iw['up'].copy()
g_iw << SemiCircular(half_bandwidth)
fixed = TailGf(1, 1, 5, -1)
fixed[1] = np.array([[1]])
fixedg0=TailGf(1,1,4,-1)
fixedg0[1]=np.array([[1]])

def dmft_loop_pm(U):
    chemical_potential = U/2.0

    fixed[3] = np.array([[U**2/4]])
    fixedg0[2]=np.array([[-chemical_potential]])

    print 'got here'
    # Now do the DMFT loop
    for it in range(n_loops):

        # Compute S.G0_iw with the self-consistency condition while imposing paramagnetism
        g_iw << 0.5*(S.G_iw['up']+S.G_iw['down']))
        g_iw.fit_tail(fixed, 5, int(beta), len(g_iw.mesh))
        g_iw.data[:]=1j*g_iw.data.imag

        for name, g0 in S.G0_iw:
            g0 << inverse( iOmega_n + chemical_potential - (half_bandwidth/2.0)**2  * g_iw )
        # Run the solver
        S.solve(h_int=U * n('up',0) * n('down',0), **params)

        # Some intermediate saves
        if mpi.is_master_node():
            with HDFArchive("CH_sb_b32.h5") as R:
                R["U{}/G_tau-{}".format(U, it)] = S.G_tau
                R["U{}/G_iw-{}".format(U, it)] = S.G_iw

for u in [2.2, 2.7, 3.2]:
    dmft_loop_pm(u)
