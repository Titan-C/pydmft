# -*- coding: utf-8 -*-
"""
@author: Óscar Nájera
"""
#from __future__ import division, absolute_import, print_function
from pytriqs.applications.impurity_solvers.cthyb import Solver
from pytriqs.archive import *
from pytriqs.gf.local import *
from pytriqs.operators import *
from time import time
import argparse
import dmft.RKKY_dimer as rt
import dmft.common as gf
import numpy as np
import pytriqs.utility.mpi as mpi
from math import sqrt

# Set the solver parameters
params = {
    'n_cycles': int(2e6),
    'length_cycle': 80,
    'n_warmup_cycles': int(5e4),
    'move_double': True,
    'measure_pert_order': True,
    'random_seed': int(time()/2**14+time()/2**16*mpi.rank)
}

U = 1.
aup = (-c('low', 0) + c('high', 0))/sqrt(2)
adw = (-c('low', 1) + c('high', 1))/sqrt(2)

bup = (c('low', 0) + c('high', 0))/sqrt(2)
bdw = (c('low', 1) + c('high', 1))/sqrt(2)

HINT = U * (dagger(aup)*aup*dagger(adw)*adw + dagger(bup)*bup*dagger(bdw)*bdw)


def cthyb_last_run(u_int, tp, BETA, file_str):
    S = Solver(beta=BETA, gf_struct={'low': [0, 1], 'high': [0, 1]})

    for name, gblock in S.G_iw:
        gblock << SemiCircular(1)

    for name, g0block in S.G0_iw:
        g0block << inverse(iOmega_n + u_int/2. - 0.25*S.G_iw[name])

    S.solve(h_int=HINT, **params)

    if mpi.is_master_node():
        with rt.HDFArchive('diag_dimer.h5') as last_run:
            last_run[u]['it00/G_iw'] = S.G_iw
            last_run[u]['it00/G_tau'] = S.G_tau


parser = argparse.ArgumentParser(description='DMFT loop for a dimer bethe\
                                                      lattice solved by CTHYB')
parser.add_argument('beta', metavar='B', type=float,
                    default=20., help='The inverse temperature')
parser.add_argument('tp', default=0.18, help='The dimerization strength')
parser.add_argument('-U', metavar='U', nargs='+', type=float,
                    default=[1.], help='Local interaction strength')

args = parser.parse_args()

cthyb_last_run(args.U[0], args.tp, args.beta, 'save.h5')
