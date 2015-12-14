#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CTHYB solver for single band DMFT
=================================

Using the triqs package the single band DMFT case is solved.
"""
#from __future__ import division, absolute_import, print_function
from pytriqs.archive import HDFArchive
from pytriqs.gf.local import *
from pytriqs.operators import *
from pytriqs.applications.impurity_solvers.cthyb import Solver
import numpy as np
import pytriqs.utility.mpi as mpi
import argparse

# Set up a few parameters
parser = argparse.ArgumentParser(description='DMFT loop for single site Bethe lattice in CTHYB')
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
S = Solver(beta=beta, gf_struct={'up': [0], 'down': [0]},
           n_iw=int(2*beta))

# Set the solver parameters
params = {'n_cycles': int(1e5),
          'length_cycle': 30,
          'n_warmup_cycles': int(5e4),
          }

fixed = TailGf(1, 1, 3, 1)
fixed[1] = np.array([[1]])
# Initalize the Green's function to a semi-circular density of states

g_iw = S.G_iw['up'].copy()
g_iw << SemiCircular(half_bandwidth)
for name, g in S.G_iw:
    g << g_iw

for name, g in S.G_tau:
    g.tail[1] = np.array([[1]])


def dmft_loop_pm(U):
    chemical_potential = U/2.0

    fixed[3] = np.array([[U**2/4 + .25]])
    for name, g in S.G_tau:
        g.tail[3] = fixed[3]

    try:
        with HDFArchive("CH_sb_b{}.h5".format(args.beta), 'r') as R:
            last_loop = len(R['U{}'.format(U)].keys())
            S.G_iw << R['U{}'.format(U)]['it{:03}'.format(last_loop-1)]['giw']
    except(IOError, KeyError):
        last_loop = 0

    # Now do the DMFT loop
    for it in range(last_loop, last_loop + n_loops):
        if mpi.is_master_node():
            print('it', it)

        # Compute S.G0_iw with the self-consistency condition while imposing paramagnetism
        g_iw << 0.5*(S.G_iw['up']+S.G_iw['down'])
        g_iw.fit_tail(fixed, 5, int(beta), len(g_iw.mesh))
        g_iw.data[:] = 1j*g_iw.data.imag

        for name, g0 in S.G0_iw:
            g0 << inverse(iOmega_n + chemical_potential - (half_bandwidth/2.0)**2 * g_iw)
        # Run the solver
        S.solve(h_int=U * n('up', 0) * n('down', 0), **params)

        # Some intermediate saves
        if mpi.is_master_node():
            with HDFArchive("CH_sb_b{}.h5".format(args.beta)) as R:
                R["U{}/it{:03}/giw".format(U, it)] = S.G_iw

for u in args.U:
    dmft_loop_pm(u)
