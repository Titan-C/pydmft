# -*- coding: utf-8 -*-
"""
=========================
DMFT loop
=========================

To treat the Anderson impurity model and solve it using the continuous time
Quantum Monte Carlo algorithm in the hybridization expansion
"""
from __future__ import division, absolute_import, print_function

from cthyb.storage import *
from dmft.common import gt_fouriertrans

import pyalps.cthyb as cthyb  # the solver module
import pyalps.mpi as mpi     # MPI library (required)


## DMFT loop
def dmft_loop(parms, delta_in):
    term = False
    iwn = matsubara_freq(parms['BETA'], parms['N_MATSUBARA'])
    tau = np.linspace(0, parms['BETA'], parms['N_TAU']+1)
    gw_old = gt_fouriertrans(delta_in / parms['t']**2, tau, iwn)

    for n in range(25):
        cthyb.solve(parms)
        if mpi.rank == 0:
            print('*'*80, '\n', 'End Dmft loop ', n, 'at beta', parms['BETA'])
            g_tau = recover_measurement(parms, 'G_tau')
            save_iter_step(parms, n, 'G_tau', g_tau)

            g_iwn = np.array([gt_fouriertrans(gt, tau, iwn) for gt in g_tau])

            g_w = g_iwn.mean(axis=0)*.6 + 0.4*g_w
            dev = np.abs(gw_old - g_w)[:20].max()
            print('conv criterion', dev)
            conv = dev < 0.01
            gw_old = g_w
            delta_out = save_pm_delta_tau(parms, g_tau)

            term = mpi.broadcast(value=conv, root=0)
            delta_out = mpi.broadcast(value=delta_out, root=0)
        else:
            term = mpi.broadcast(root=0)
            delta_out = mpi.broadcast(root=0)

        mpi.world.barrier()  # wait until solver input is written

        if term:
            if mpi.rank == 0:
                print('End on iterartion: ', n)
            break

    return delta_out


## master looping
if __name__ == "__main__":
    BETA = np.floor(np.logspace(1.1, 2.7, 10))
    U = np.arange(0, 6.5, 0.2)
    for beta in BETA:
        for u_int in U:
            parms = {
                'SWEEPS'              : 100000000,
                'THERMALIZATION'      : 5000,
                'N_MEAS'              : 100,
                'MAX_TIME'            : 30,
                'N_HISTOGRAM_ORDERS'  : 50,
                'SEED'                : 5,

                'N_ORBITALS'          : 2,
                'DELTA'               : "delta_b{}.h5".format(beta),
                'DELTA_IN_HDF5'       : 1,
                'BASENAME'            : 'PM_MI_b{}_U{}'.format(beta, u_int),

                't'                   : 1.,
                'U'                   : u_int,
                'MU'                  : u_int/2.,
                'N_TAU'               : 1024,
                'N_MATSUBARA'         : 256,
                'MEASURE_freq'        : 0,
                'BETA'                : beta,
                'VERBOSE'             : 1,
                'SPINFLIP'            : 1,
            }
            if mpi.rank == 0 and u_int == U[0]:
                delta_start = start_delta(parms)
                print('write delta at beta ', str(beta))
                delta_start = mpi.broadcast(value=delta_start, root=0)
            elif u_int == U[0]:
                delta_start = mpi.broadcast(root=0)

            mpi.world.barrier()

            delta_start = dmft_loop(parms, delta_start)
