# -*- coding: utf-8 -*-
"""
@author: Óscar Nájera
Created on Mon Nov 10 11:18:35 2014
"""
#from __future__ import division, absolute_import, print_functionfrom pytriqs.gf.local import *
from pytriqs.gf.local import *
from pytriqs.operators import *
from pytriqs.archive import HDFArchive
import pytriqs.utility.mpi as mpi
import numpy as np

# Set up a few parameters
U = 3.2
half_bandwidth = 1.0
chemical_potential = U/2.0
beta = 100.
n_loops = 1

# Construct the CTQMC solver
from pytriqs.applications.impurity_solvers.cthyb import Solver
S = Solver(beta=beta, gf_struct={ 'up':[0], 'down':[0] },
           n_iw=1025, n_tau=10001, n_l=50)

# Set the solver parameters
params = {'n_cycles': int(3e7),
          'length_cycle': 200,
          'n_warmup_cycles': int(5e4),
          'measure_g_l': True,
          'measure_pert_order': True,
        }

# Initalize the Green's function to a semi-circular density of states

g_iw = GfImFreq(indices = [0], beta = beta, n_points=1025)
#g_iw << SemiCircular(half_bandwidth)
g_iw.data[:,0,0] = np.load('Giw_out.npy')[-1] #np.load('fgiws500.npy')
fixed=TailGf(1,1,3,-1)
fixed[1]=np.array([[1]])
g_iw.fit_tail(fixed,4, 300, 1025)

#R = HDFArchive("compareHF.h5")
for name, g0block in S.G_tau:
    g0block << InverseFourier(g_iw)
#del R

# Initalize the Green's function to a semi-circular density of states
#for name, g0block in S.G_l:
#    g0block.set_from_imfreq(g_iw)

import dmft.common as gf
wn=gf.matsubara_freq(100., 1025)
#g=np.squeeze(g_iw.data[:,0,0])
#den=np.abs(1j*wn +1.6-.25*g)**2
#g0iw=(1.6-.25*g.real)/den-1j*(wn-.25*g.imag)/den
fixedg0=TailGf(1,1,4,-1)
fixedg0[1]=np.array([[1]])
fixedg0[2]=np.array([[-1.6]])

print 'got here'
# Now do the DMFT loop
for it in range(n_loops):

    # Compute S.G0_iw with the self-consistency condition while imposing paramagnetism
#    g_iw.set_from_legendre( 0.5 * ( S.G_l['up'] + S.G_l['down'] ))
    g_iw << Fourier(0.5*(S.G_tau['up']+S.G_tau['down']))
    g_iw.fit_tail(fixed,3,300,1025)
    g_iw.data[:,0,0]=1j*g_iw.data[:,0,0].imag

    g=np.squeeze(g_iw.data[:,0,0])
    den=np.abs(1j*wn +1.6-.25*g)**2
    g0iw=(1.6-.25*g.real)/den-1j*(wn-.25*g.imag)/den
    for name, g0 in S.G0_iw:
#        g0 << inverse( iOmega_n + chemical_potential - (half_bandwidth/2.0)**2  * g_iw )
        g0.data[:,0,0] = g0iw
        g0.fit_tail(fixedg0, 6, 300, 1025)
    # Run the solver
    S.solve(h_int=U * n('up',0) * n('down',0), **params)

    # Some intermediate saves
    if mpi.is_master_node():
        with HDFArchive("deep_in_ins2.h5") as R:
          R["G_tau-%s"%it] = S.G_tau
          R["G_iw-%s"%it] = S.G_iw
          R["G0_iw-%s"%it] = S.G0_iw
          R["G_l-%s"%it] = S.G_l


