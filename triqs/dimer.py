# -*- coding: utf-8 -*-
"""
@author: Óscar Nájera
"""
#from __future__ import division, absolute_import, print_function
from pytriqs.gf.local import *
from pytriqs.operators import *
from pytriqs.archive import *
from pytriqs.applications.impurity_solvers.cthyb import Solver
import pytriqs.utility.mpi as mpi
import dmft.common as gf
import dmft.RKKY_dimer as rt

# Set up a few parameters
U = 2.5
half_bandwidth = 1.0
chemical_potential = U/2.0
beta =20
n_loops = 1

# Construct the CTQMC solver
S = Solver(beta=beta, gf_struct={ 'up':[0, 1], 'down':[0, 1] },
           n_iw=512, n_tau=2048, n_l=30)

# Set the solver parameters
params = {
    'n_cycles' : int(1e5),
    'length_cycle' : 300,
    'n_warmup_cycles' : int(5e4),
    'move_double': True,
}

# Initalize the Green's function to a semi-circular density of states
w_n= gf.matsubara_freq(beta, 512)
g_iw = GfImFreq(indices=['A', 'B'], beta=beta,
                             n_points=512)
gmix = rt.mix_gf_dimer(g_iw.copy(), iOmega_n, U/2., 0.2)
rt.init_gf_met(g_iw, w_n, chemical_potential, 0.2, 0., 0.5)

for name, g0block in S.G_tau:
    g0block << InverseFourier(g_iw)

for name, g0block in S.G0_iw:
    g0block << inverse(gmix - 0.25*g_iw)

for name, g0block in S.G_iw:
    g0block << g_iw

S.solve(h_int=U * n('up',0) * n('down',0) + U * n('up',1) * n('down',1), **params)

if mpi.is_master_node():
  R = rt.HDFArchive("qdimer_bethe_fwbin.h5")
  R["G_tau"] = S.G_tau
  R['G0_iw'] = S.G0_iw
  R['G_iw'] = S.G_iw
  R['G_l'] = S.G_l
  del R
# Now do the DMFT loop
#for IterationNumber in range(n_loops):
#
#    # Compute S.G0_iw with the self-consistency condition while imposing paramagnetism
#    g = 0.5 * Fourier( S.G_tau['up'] + S.G_tau['down'] )
#    for name, g0 in S.G0_iw:
#        g0 << inverse( iOmega_n + chemical_potential - (half_bandwidth/2.0)**2  * g )
#
#    # Run the solver
#    S.solve(h_int=U * n('up',0) * n('down',0) + U * n('up',1) * n('down',1), **params)
#
#    # Some intermediate saves
#    if mpi.is_master_node():
#      R = HDFArchive("single_site_bethe.h5")
#      R["G_tau-%s"%IterationNumber] = S.G_tau
#      del R
#
#from pytriqs.plot.mpl_interface import oplot
#R = HDFArchive("single_site_bethe.h5")
#for i in range(5):
#    oplot(R['G_tau-%s' % i]['down'], '-o', name='Iteration = %s'%i, x_window=(0, 2))
