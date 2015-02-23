# -*- coding: utf-8 -*-
"""
=========================
DMFT loop
=========================

To treat the Anderson impurity model and solve it using the continuous time
Quantum Monte Carlo algorithm in the hybridization expansion
"""
import sys
import numpy as np
from dmft.common import matsubara_freq, greenF, gw_invfouriertrans
sys.path.append('/home/oscar/libs/lib')

import pyalps.cthyb as cthyb  # the solver module
import pyalps.mpi as mpi     # MPI library (required)
from pyalps.hdf5 import archive

# specify solver parameters
beta = 16.
U = 2.5

parms = {
    'SWEEPS'              : 100000000,
    'THERMALIZATION'      : 1000,
    'N_MEAS'              : 50,
    'MAX_TIME'            : 1,
    'N_HISTOGRAM_ORDERS'  : 50,
    'SEED'                : 0,

    'N_ORBITALS'          : 2,
    'DELTA'               : "delta.h5",
    'DELTA_IN_HDF5'       : 1,
    'BASENAME'            : 'PM_b{}_U{}'.format(beta,U),

    'U'                   : U,
    'MU'                  : U/2.,
    'N_TAU'               : 1000,
    'N_MATSUBARA'         : 200,
    'BETA'                : beta,
    'VERBOSE'             : 1,
}

def save_pm_delta(gtau):
    save_delta = archive(parms['DELTA'], 'w')
    gtau = gtau.mean(axis=0)
    save_delta['/Delta_0'] = gtau
    save_delta['/Delta_1'] = gtau
    del save_delta

def recover_g_tau(parms):
    iteration = archive(parms['BASENAME'] + '.out.h5', 'r')
    gtau = []
    for i in range(2):
        gtau.append(iteration['G_tau/{}/mean/value'.format(i)])
    del iteration
    return np.asarray(gtau)

def save_iter_step(iter_count, g):
    save = archive('steps.h5', 'w')
    for i in range(2):
        save['iter_{}/G_tau/{}/'.format(iter_count, i)] = g[i]
    del save


if mpi.rank == 0:
    iwn = matsubara_freq(parms['BETA'], parms['N_MATSUBARA'])
    tau = np.linspace(0, parms['BETA'], parms['N_TAU']+1)

    giw = greenF(iwn, mu=0.)[1::2]
    gtau = gw_invfouriertrans(giw, tau, iwn, parms['BETA'])

    save_pm_delta(np.asarray((gtau, gtau)))

mpi.world.barrier()
# solve the impurity model

## DMFT loop
gt_old=np.zeros(parms['N_TAU']+1)
term = False
for n in range(12):
    cthyb.solve(parms)
    if mpi.rank == 0:
        g_tau = recover_g_tau(parms)
        save_iter_step(n, g_tau)
        gt_new = g_tau.mean(axis=0)
        # inverting for AFM self-consistency
        save_pm_delta(g_tau)
        conv = np.abs(gt_old - gt_new).max() < 0.005
        gt_old = gt_new
        term = mpi.broadcast(value=conv, root=0)
    else:
        term = mpi.broadcast(root=0)

    mpi.world.barrier() # wait until solver input is written

    if term:
        print('end on iterartion: ', n)
        break

