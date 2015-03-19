# -*- coding: utf-8 -*-
"""
=========================
DMFT loop
=========================

To treat the Anderson impurity model and solve it using the continuous time
Quantum Monte Carlo algorithm in the hybridization expansion
"""
from __future__ import division, absolute_import, print_function

import numpy as np
from dmft.common import matsubara_freq, greenF, gw_invfouriertrans, gt_fouriertrans
from dmft import ipt_imag

import pyalps.cthyb as cthyb  # the solver module
import pyalps.mpi as mpi     # MPI library (required)
from pyalps.hdf5 import archive


def save_pm_delta_tau(parms, gtau):
    """Saves to file and returns the imaginary time hybridization function
    enforcing paramagnetism"""
    save_delta = archive(parms["DELTA"], 'w')
    delta = parms['t']**2 * gtau.mean(axis=0)
    delta[delta > -1e-5] = -1e-5

    save_delta['/Delta_0'] = delta
    save_delta['/Delta_1'] = delta
    del save_delta

    return delta


def recover_measurement(parms, measure):
    """Recovers a specific measurement from the output file"""
    iteration = archive(parms['BASENAME'] + '.out.h5', 'r')
    data = []
    for i in range(parms['N_ORBITALS']):
        data.append(iteration[measure+'/{}/mean/value'.format(i)])
    del iteration
    return np.asarray(data)


def save_iter_step(parms, iter_count, measure, data):
    """Saves the measurement results to log DMFT iterations"""
    save = archive(parms['BASENAME']+'steps.h5', 'w')
    for i, data_vector in enumerate(data):
        save['iter_{:0>2}/{}/{}/'.format(iter_count, measure, i)] = data_vector
    del save


def start_delta(parms):
    """Provides a starting guess for the hybridization function given the
    cthyb impurity solvers parameters. Guess is based on the IPT solution"""

    iwn = matsubara_freq(parms['BETA'], parms['N_MATSUBARA'])
    tau = np.linspace(0, parms['BETA'], parms['N_TAU']+1)

    giw = greenF(iwn, mu=0., D=2*parms['t'])
    giw = ipt_imag.dmft_loop(30, parms['U'], parms['t'], giw, iwn, tau)[-1]
    gtau = gw_invfouriertrans(giw, tau, iwn)

    return save_pm_delta_tau(parms, np.asarray((gtau, gtau)))


## DMFT loop
def dmft_loop(parms, delta_in):
    term = False
    iwn = matsubara_freq(parms['BETA'], parms['N_MATSUBARA'])
    tau = np.linspace(0, parms['BETA'], parms['N_TAU']+1)
    gw_old = gt_fouriertrans(delta_in / parms['t']**2, tau, iwn)

    for n in range(20):
        cthyb.solve(parms)
        if mpi.rank == 0:
            print('*'*80, '\n', 'End Dmft loop ', n, 'at beta', parms['BETA'])
            g_tau = recover_measurement(parms, 'G_tau')
            save_iter_step(parms, n, 'G_tau', g_tau)

            g_iwn = np.array([gt_fouriertrans(gt, tau, iwn) for gt in g_tau])

            g_w = g_iwn.mean(axis=0)
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
    BETA = [20.]#[8, 9, 13, 15, 18, 20, 25, 30, 40, 50]
    U = np.arange(4, 7, 0.2)
    for beta in BETA:
        for u_int in U:
            parms = {
                'SWEEPS'              : 100000000,
                'THERMALIZATION'      : 1000,
                'N_MEAS'              : 50,
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
