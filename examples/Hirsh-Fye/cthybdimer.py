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
#U = 2.8
#half_bandwidth = 1.0
#chemical_potential = U/2.0
#beta = 20.
#n_loops = 1

# Construct the CTQMC solver
#S = Solver(beta=beta, gf_struct={ 'up':[0, 1], 'down':[0, 1] },
#           n_iw=95, n_tau=1025, n_l=50)

# Set the solver parameters
params = {
    'n_cycles' : int(6e5),
    'length_cycle' : 100,
    'n_warmup_cycles' : int(5e4),
#    'move_double': True,
    'measure_g_l': True,
    'measure_pert_order': True,
}

# Initalize the Green's function to a semi-circular density of states
#w_n= gf.matsubara_freq(beta, 95)
#g_iw = GfImFreq(indices=['A', 'B'], beta=beta,
#                             n_points=95)
#gmix = rt.mix_gf_dimer(g_iw.copy(), iOmega_n, U/2., 0.21)
#rt.init_gf_met(g_iw, w_n, 0., 0.2, 0., 0.5)
#rt.load_gf(g_iw, T['U2.8']['it08']['G_iwd'], T['U2.8']['it08']['G_iwo'])

#for name, g0block in S.G_tau:
#    g0block << InverseFourier(g_iw)

#for name, g0block in S.G0_iw:
#    g0block << inverse(gmix - 0.25*g_iw)

#for name, g0block in S.G_iw:
#    g0block << g_iw

#S.solve(h_int=U * n('up',0) * n('down',0) + U * n('up',1) * n('down',1), **params)
#
#if mpi.is_master_node():
#  with rt.HDFArchive("qdimer_bethe_fwbin.h5") as R:
#      R["G_tau"] = S.G_tau
#      R['G0_iw'] = S.G0_iw
#      R['G_iw'] = S.G_iw
#      R['G_l'] = S.G_l
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
def load_gf(g_iw, g_iwd, g_iwo):
    """Loads into the first greenfunction the equal diagonal terms and the
    offdiagonals. Input in GF_view"""

    g_iw['0', '0'] << g_iwd
    g_iw['1', '1'] << g_iwd
    g_iw['0', '1'] << g_iwo
    g_iw['1', '0'] << g_iwo

def mix_gf_dimer(gmix, omega, mu, tab):
    """Dimer formation Green function term

    .. math::

        G_{mix}(i\omega_n) =ao
    """
    gmix['0', '0'] = omega + mu
    gmix['0', '1'] = -tab
    gmix['1', '0'] = -tab
    gmix['1', '1'] = omega + mu
    return gmix


u = 'U2.6'
file_str = 'disk/metf_HF_Ul_tp0.21_B10.0.h5'
with rt.HDFArchive(file_str) as last_run:
    lastit = mpi.bcast(last_run[u].keys()[-1])
    setup = mpi.bcast(last_run[u][lastit]['setup'])
    gd = mpi.bcast(last_run[u][lastit]['G_iwd'])
    go = mpi.bcast(last_run[u][lastit]['G_iwo'])
#
S = Solver(beta=setup['BETA'], gf_struct={'up': [0, 1], 'down': [0, 1]},
           n_iw=setup['n_points'], n_tau=1025, n_l=50)
#print('gothere')
#sys.stdout.flush()
for name, gblock in S.G_iw:
#    print('gothere')
#    sys.stdout.flush()
    load_gf(gblock, gd, go)
#    print('first load')
#    sys.stdout.flush()
#
#print('gothere')
#sys.stdout.flush()
U = setup['U']
gmix = mix_gf_dimer(S.G_iw['up'].copy(), iOmega_n, U/2., setup['tp'])
#print('gothere')
#sys.stdout.flush()
#
for name, g0block in S.G0_iw:
    g0block << inverse(gmix - 0.25*S.G_iw['up'])

S.solve(h_int=U * n('up',0) * n('down',0) + U * n('up',1) * n('down',1), **params)

#if mpi.is_master_node():
#    with rt.HDFArchive(file_str.format(**setup), 'r') as last_run:
#        last_run[u]['cthyb'] = S


