 # Run this script as:
 # alpspython tutorial1.py
 #
 # This python script is MPI aware and can hence be called using mpirun:
 #
 # mpirun -np 2 alpspython tutorial1.py
 #
 # In case this does not work, try:
 #
 # mpirun -np 2 sh alpspython tutorial1.py
import sys
sys.path.append('/usr/lib')
import pyalps.cthyb as cthyb # the solver module
import pyalps.mpi as mpi     # MPI library (required)
import numpy as np
from dmft.common import  matsubara_freq, greenF, gw_invfouriertrans

# specify solver parameters
parms={
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

if mpi.rank==0:
    iwn = matsubara_freq(parms['BETA'], parms['N_MATSUBARA'])
    tau = np.linspace(0,parms['BETA'], parms['N_TAU']+1)

    giw_u = greenF(iwn, mu=-0.04)[1::2]
    gtau_u = gw_invfouriertrans(giw_u, tau, iwn, parms['BETA'])

    giw_d = greenF(iwn, mu=0.04)[1::2]
    gtau_d = gw_invfouriertrans(giw_d, tau, iwn, parms['BETA'])

    np.savetxt('delta.dat', np.asarray((tau,gtau_u,gtau_d)).T)

# solve the impurity model
cthyb.solve(parms)




