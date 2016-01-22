#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CTHYB solver for single band DMFT
=================================

Using the triqs package the single band DMFT case is solved.
"""
from pytriqs.archive import HDFArchive
from pytriqs.gf.local import *
from pytriqs.operators import *
from pytriqs.applications.impurity_solvers.cthyb import Solver
import numpy as np
import pytriqs.utility.mpi as mpi
import argparse
import dmft.plot.triqs_sb as tsb
import os
import struct

# Set up a few parameters
parser = argparse.ArgumentParser(description='DMFT loop for single site Bethe lattice in CTHYB',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-BETA', metavar='B', type=float,
                    default=32., help='The inverse temperature')
parser.add_argument('-Niter', metavar='N', type=int,
                    default=10, help='Number of DMFT Loops')
parser.add_argument('-U', metavar='U', nargs='+', type=float,
                    default=[2.7], help='Local interaction strength')
parser.add_argument('-sweeps', metavar='MCS', type=int, default=int(1e5),
                    help='Number Monte Carlo Measurement')
parser.add_argument('-therm', type=int, default=int(5e4),
                    help='Monte Carlo sweeps of thermalization')
parser.add_argument('-meas', type=int, default=100,
                    help='Number of Updates before measurements')
parser.add_argument('-mu', '--MU', type=float, default=0.,
                    help='Chemical potential')
parser.add_argument('-afm', '--AFM', action='store_true',
                    help='Use the self-consistency for Antiferromagnetism')
parser.add_argument('-i', '--insulator', action='store_true',
                    help='Seeding Green function from atomic limit(insulator)')
parser.add_argument('-hist', '--histogram', action='store_true', default=False,
                    help='Save histogram files')

args = parser.parse_args()
t = 0.5
BETA = args.BETA

# Construct the CTQMC solver
S = Solver(beta=BETA, gf_struct={'up': [0], 'down': [0]},
           n_iw=int(2*BETA))

# Set the solver parameters
params = {'n_cycles': args.sweeps,
          'length_cycle': args.meas,
          'n_warmup_cycles': args.therm,
          'measure_pert_order': args.histogram,
         }

# Initalize the Green's function to a semi-circular density of states
g_iw = S.G_iw['up'].copy()
g_iw << SemiCircular(2*t)
if args.insulator:
    g_iw << (inverse(iOmega_n - args.U[0]) + inverse(iOmega_n + args.U[0]))/2.

for _, g in S.G_iw:
    g << g_iw


def dmft_loop_pm(U):
    chemical_potential = U/2.0 + args.MU

    simt = '_AFM_' if args.AFM else 'PM'
    mpi.barrier() # Because the global solver from master can be writing last U loop

    try:
        with HDFArchive("CH{}_sb_b{}.h5".format(simt, args.BETA), 'r') as R:
            last_loop = len(R['U{}'.format(U)].keys())
            S.G_iw << R['U{}'.format(U)]['it{:03}'.format(last_loop-1)]['giw']
    except(IOError, KeyError):
        last_loop = 0

    # Now do the DMFT loop
    for i in range(last_loop, last_loop + args.Niter):
        if mpi.is_master_node():
            print('it', i)
            print('On loop', i, 'BETA', BETA, 'U', U)

        if args.AFM:
            S.G0_iw['up'] << inverse(iOmega_n + chemical_potential - t**2 * S.G_iw['down'])
            S.G0_iw['down'] << inverse(iOmega_n + chemical_potential - t**2 * S.G_iw['up'])

        else:
            # Compute S.G0_iw with the self-consistency condition while
            # imposing paramagnetism
            g_iw << 0.5*(S.G_iw['up']+S.G_iw['down'])
            g_iw.data[:].real = 0.
            tsb.tail_clean(g_iw, U)

            for _, g0 in S.G0_iw:
                g0 << inverse(iOmega_n + chemical_potential - t**2 * g_iw)
        # Run the solver, new seed each call as c++ solver initializes
        # RNG on each call of solve
        params['random_seed'] = struct.unpack("I", os.urandom(4))[0]
        S.solve(h_int=U * n('up', 0) * n('down', 0), **params)

        # Some intermediate saves
        if mpi.is_master_node():
            with HDFArchive("CH{}_sb_b{}.h5".format(simt, args.BETA), 'a') as R:
                R["U{}/it{:03}/giw".format(U, i)] = S.G_iw


for u in args.U:
    dmft_loop_pm(u)
