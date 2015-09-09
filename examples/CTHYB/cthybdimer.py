# -*- coding: utf-8 -*-
"""
Dimer in Bethe lattice
======================

"""
from pytriqs.applications.impurity_solvers.cthyb import Solver
from pytriqs.archive import *
from pytriqs.gf.local import *
from pytriqs.operators import *
from time import time
import argparse
import dmft.RKKY_dimer as rt
import pytriqs.utility.mpi as mpi
from math import sqrt


aup = (-c('low_up', 0) + c('high_up', 0))/sqrt(2)
adw = (-c('low_dw', 0) + c('high_dw', 0))/sqrt(2)

bup = (c('low_up', 0) + c('high_up', 0))/sqrt(2)
bdw = (c('low_dw', 0) + c('high_dw', 0))/sqrt(2)



def cthyb_last_run(u_int, tp, BETA, file_str):
    S = Solver(beta=BETA, gf_struct={'low_up': [0], 'high_up': [0],
                                     'low_dw': [0], 'high_dw': [0]})

    for name, gblock in S.G_iw:
        gblock << SemiCircular(1)

    for it in range(20):

        S.G_iw['low_up'] << 0.5 * (S.G_iw['low_up'] + S.G_iw['low_dw'])
        S.G_iw['high_up'] << 0.5 * (S.G_iw['high_up'] + S.G_iw['high_dw'])
        S.G_iw['low_dw'] << S.G_iw['low_up']
        S.G_iw['high_dw'] << S.G_iw['high_up']

        for name, g0block in S.G0_iw:
            g0block << inverse(iOmega_n + u_int/2. - 0.25*S.G_iw[name])

        HINT = u_int * (dagger(aup)*aup*dagger(adw)*adw + dagger(bup)*bup*dagger(bdw)*bdw)

        S.solve(h_int=HINT, **params)

        if mpi.is_master_node():
            with rt.HDFArchive('netdiag_dimer.h5') as last_run:
                last_run['it{}/G_iw'.format(it)] = S.G_iw
                last_run['it{}/G_tau'.format(it)] = S.G_tau


def do_setup():
    """Set the solver parameters"""

    parser = argparse.ArgumentParser(description='DMFT loop for CTHYB dimer',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-sweeps', dest='n_cycles', metavar='MCS', type=int,
                        default=int(1e6), help='Number MonteCarlo Measurement')
    parser.add_argument('-therm', type=int, default=int(1e4), dest='n_warmup_cycles',
                        help='Monte Carlo sweeps of thermalization')
    parser.add_argument('-N_meas', type=int, default=200, dest='length_cycle',
                        help='Number of Updates before measurements')
    parser.add_argument('-Niter', metavar='N', type=int,
                        default=20, help='Number of iterations')
    parser.add_argument('-BETA', metavar='B', type=float,
                        default=32., help='The inverse temperature')
    parser.add_argument('-U', type=float, nargs='+',
                        default=[2.5], help='Local interaction strenght')
    parser.add_argument('tp', default=0.18, help='The dimerization strength')
    parser.add_argument('-ofile', default='SB_PM_B{BETA}.h5',
                        help='Output file shelve')

    parser.add_argument('-new_seed', type=float, nargs=3, default=False,
                        metavar=('U_src', 'U_target', 'avg_over'),
                        help='Resume DMFT loops from on disk data files')
    setup = vars(parser.parse_args())
    setup.update({'move_double': True,
                  'measure_pert_order': True,
                  'random_seed': int(time()/2**14+time()/2**16*mpi.rank)})

    return setup

if __name__ == "__main__":
    SETUP = do_setup()

    dmft_loop(SETUP)
