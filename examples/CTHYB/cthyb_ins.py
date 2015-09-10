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
import argparse

# Set up a few parameters
parser = argparse.ArgumentParser(description='DMFT loop for single site
Bethe lattice in CTHYB')
parser.add_argument('-beta', metavar='B', type=float,
                    default=32., help='The inverse temperature')
parser.add_argument('-Niter', metavar='N', type=int,
                    default=10, help='Number of DMFT Loops')
parser.add_argument('-U', metavar='U', nargs='+', type=float,
                    default=[2.7], help='Local interaction strength')

args = parser.parse_args()
half_bandwidth = 1.0
beta = args.beta
n_loops = args.Niter

# Construct the CTQMC solver
from pytriqs.applications.impurity_solvers.cthyb import Solver
S = Solver(beta=beta, gf_struct={ 'up':[0], 'down':[0] },
           n_iw=int(3*beta), n_tau=int(8*beta))

# Set the solver parameters
params = {'n_cycles': int(1e7),
          'length_cycle': 200,
          'n_warmup_cycles': int(5e4),
        }

# Initalize the Green's function to a semi-circular density of states

g_iw = S.G_iw['up'].copy()
g_iw << SemiCircular(half_bandwidth)
for name, g in S.G_iw:
    g << g_iw

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
        g_iw << 0.5*(S.G_iw['up']+S.G_iw['down'])
        g_iw.fit_tail(fixed, 5, int(beta), len(g_iw.mesh))
        g_iw.data[:]=1j*g_iw.data.imag

        for name, g0 in S.G0_iw:
            g0 << inverse( iOmega_n + chemical_potential - (half_bandwidth/2.0)**2  * g_iw )
        # Run the solver
        S.solve(h_int=U * n('up',0) * n('down',0), **params)

        # Some intermediate saves
        if mpi.is_master_node():
            with HDFArchive("CH_sb_b{}.h5".format(args.beta) as R:
                R["U{}/G_tau-{}".format(U, it)] = S.G_tau
                R["U{}/G_iw-{}".format(U, it)] = S.G_iw

for u in args.U:
    dmft_loop_pm(u)
