# -*- coding: utf-8 -*-
"""
=========================
DMFT loop
=========================

To treat the Anderson impurity model and solve it using the continuous time
Quantum Monte Carlo algorithm in the hybridization expansion
"""
from __future__ import division, absolute_import, print_function
import sys
sys.path.append('/home/oscar/libs/lib')

import numpy as np
from dmft.common import matsubara_freq, greenF, gw_invfouriertrans, gt_fouriertrans
import pyalps.cthyb as cthyb  # the solver module
import pyalps.mpi as mpi     # MPI library (required)
from pyalps.hdf5 import archive


def save_pm_delta(parms, gtau):
    save_delta = archive(parms["DELTA"], 'w')
    gtau = parms['t']**2 * gtau.mean(axis=0)
    gtau = (gtau + gtau[::-1]) / 2
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

def save_iter_step(parms, iter_count, g):
    save = archive(parms['BASENAME']+'steps.h5', 'w')
    for i in range(2):
        save['iter_{}/G_tau/{}/'.format(iter_count, i)] = g[i]
    del save

def start_delta(parms):
    iwn = matsubara_freq(parms['BETA'], parms['N_MATSUBARA'])
    tau = np.linspace(0, parms['BETA'], parms['N_TAU']+1)

    giw = greenF(iwn, mu=0., D=2*parms['t'])
    gtau = gw_invfouriertrans(giw, tau, iwn, beta)

    save_pm_delta(parms, np.asarray((gtau, gtau)))


## DMFT loop
def dmft_loop(parms):
    gw_old = np.zeros(parms['N_MATSUBARA'])
    term = False
    iwn = matsubara_freq(parms['BETA'], parms['N_MATSUBARA'])
    tau = np.linspace(0, parms['BETA'], parms['N_TAU']+1)
    for n in range(20):
        cthyb.solve(parms)
        if mpi.rank == 0:
            print('dmft loop ', n)
            g_tau = recover_g_tau(parms)
            save_iter_step(parms, n, g_tau)
            # inverting for AFM self-consistency
            save_pm_delta(parms, g_tau)
            g_w = gt_fouriertrans(g_tau.mean(axis=0), tau, iwn, parms['BETA'])
            conv = np.abs(gw_old - g_w).max() < 0.0025
            gw_old = g_w
            term = mpi.broadcast(value=conv, root=0)
        else:
            term = mpi.broadcast(root=0)

        mpi.world.barrier() # wait until solver input is written

        if term:
            print('end on iterartion: ', n)
            break


## master looping
if __name__ == "__main__":
    BETA = [50.]#[8, 9, 13, 15, 18, 20, 25, 30, 40, 50]
    U =[5.1]# np.arange(1, 7, 0.4)
    for beta in BETA:
        for u_int in U:
            parms = {
                'SWEEPS'              : 100000000,
                'THERMALIZATION'      : 1000,
                'N_MEAS'              : 50,
                'MAX_TIME'            : 1,
                'N_HISTOGRAM_ORDERS'  : 50,
                'SEED'                : 0,

                'N_ORBITALS'          : 2,
                'DELTA'               : "delta_b{}_U{}.h5".format(beta, u_int),
                'DELTA_IN_HDF5'       : 1,
                'BASENAME'            : 'PH_b{}_U{}'.format(beta, u_int),

                't'                   : 1.,
                'U'                   : u_int,
                'MU'                  : u_int/2.,
                'N_TAU'               : 1000,
                'N_MATSUBARA'         : 250,
                'MEASURE_freq'        : 1,
                'BETA'                : beta,
                'VERBOSE'             : 1,
            }
            if mpi.rank == 0 and u_int == U[0]:
                start_delta(parms)
                print('write delta at beta ', str(beta))

            mpi.world.barrier()

            dmft_loop(parms)
