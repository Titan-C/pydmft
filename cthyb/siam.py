# -*- coding: utf-8 -*-
"""
=========================
SIAM - Solver single step
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

# specify solver parameters
parms = {
    'SWEEPS'              : 100000000,
    'THERMALIZATION'      : 1000,
    'N_MEAS'              : 50,
    'MAX_TIME'            : 1,
    'N_HISTOGRAM_ORDERS'  : 50,
    'SEED'                : 0,

    'N_ORBITALS'          : 2,
    'DELTA'               : "delta.dat",

    't'                   : 1,
    'U'                   : 2.0,
    'MU'                  : 1.0,
    'N_TAU'               : 1000,
    'N_MATSUBARA'         : 200,
    'MEASURE_freq'        : 1,
    'BETA'                : 45,
    'TEXT_OUTPUT'         : 1,
    'VERBOSE'             : 1,
}


if mpi.rank == 0:
    iwn = matsubara_freq(parms['BETA'], parms['N_MATSUBARA'])
    tau = np.linspace(0, parms['BETA'], parms['N_TAU']+1)

    giw_u = greenF(iwn, mu=parms['MU'], D=2*parms['t'])
    gtau_u = gw_invfouriertrans(giw_u, tau, iwn, parms['BETA'])

    giw_d = greenF(iwn, mu=-parms['MU'], D=2*parms['t'])
    gtau_d = gw_invfouriertrans(giw_d, tau, iwn, parms['BETA'])

    np.savetxt('delta.dat', np.asarray((tau, gtau_u, gtau_d)).T)

# solve the impurity model
mpi.world.barrier()
for t in [1, 60, 300]:
    parms['MAX_TIME'] = t
    parms['BASENAME'] = 'imp_time{}'.format(t)
    cthyb.solve(parms)
    mpi.world.barrier()
