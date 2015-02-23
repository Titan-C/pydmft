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
import shutil
sys.path.append('/home/oscar/libs/lib')

import pyalps.cthyb as cthyb  # the solver module
import pyalps.mpi as mpi     # MPI library (required)

# specify solver parameters
parms = {
    'SWEEPS'              : 100000000,
    'THERMALIZATION'      : 1000,
    'N_MEAS'              : 50,
    'MAX_TIME'            : 1,
    'N_HISTOGRAM_ORDERS'  : 50,
    'SEED'                : 0,

    'ANTIFERROMAGNET'     : 1,
    'SYMMETRIZATION'      : 0,
    'N_ORBITALS'          : 2,
    'DELTA'               : "delta.dat",

    't'                   : 1,
    'U'                   : 2.0,
    'MU'                  : 1.0,
    'N_TAU'               : 1000,
    'N_MATSUBARA'         : 200,
    'BETA'                : 45,
    'TEXT_OUTPUT'         : 1,
    'VERBOSE'             : 1,
}


if mpi.rank == 0:
    iwn = matsubara_freq(parms['BETA'], parms['N_MATSUBARA'])
    tau = np.linspace(0, parms['BETA'], parms['N_TAU']+1)

    giw_u = greenF(iwn, mu=0.5)[1::2]
    gtau_u = gw_invfouriertrans(giw_u, tau, iwn, parms['BETA'])

    giw_d = greenF(iwn, mu=-0.5)[1::2]
    gtau_d = gw_invfouriertrans(giw_d, tau, iwn, parms['BETA'])

    np.savetxt('delta.dat', np.asarray((tau, gtau_u, gtau_d)).T)

mpi.world.barrier()
# solve the impurity model

## DMFT loop
for n in range(4):
    cthyb.solve(parms)
    if mpi.rank == 0:
        shutil.copy('Gt.dat', 'Gt_{}.dat'.format(n))
        shutil.copy('delta.dat', 'delta_{}.dat'.format(n))
        shutil.copy('orders.dat', 'orders_{}.dat'.format(n))
        shutil.copy('simulation.dat', 'simulation_{}.dat'.format(n))

        gtau = np.loadtxt('Gt_{}.dat'.format(n))
        # inverting for AFM self-consistency
        np.savetxt('delta.dat', np.asarray((tau, gtau[:, 2], gtau[:, 1])).T)

    mpi.world.barrier() # wait until solver input is written
